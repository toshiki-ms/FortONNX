"""ONNX graph -> one fused CUDA kernel per row, run by ONNX Runtime as a custom operator or called directly.

The ONNX graph stays the interface of the model: the generated graph runs on standard operators without the library
and on the kernel with it. rowgraph.py reads the graph as a function of one row, schedule.py arranges it into loops
and storage, emit.py writes the kernel, ortop.py writes the custom operator library source, its build script and the
graph that uses the operator, direct.py the entry of the library for programs that call the kernel themselves.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field, replace

import numpy as np
import onnx

from .direct import DEFAULT_PREFIX, direct_entry_source, entry_names
from .emit import KernelTooLarge, Options, emit_kernel, kernels_in_a_row, kernels_side_by_side
from .ortop import DOMAIN, build_script, operator_library_source, wrap_graph
from .rowgraph import RowGraph, UnsupportedGraph, evaluate, load_row_graph, sub_graph
from .schedule import (Limits, Plan, fold_constant_terms, make_plan, output_parts, products, rewrite_graph, sink_gathers,
                       split_before_thread_loop)
from .tensor import DenseChain

__all__ = [
    "DOMAIN", "GeneratedKernel", "KernelTooLarge", "Limits", "Options", "Plan", "RowGraph", "UnsupportedGraph",
    "check_file_names", "direct_entry_source", "entry_names", "evaluate", "generate_kernel", "load_row_graph", "make_plan",
    "operator_library_source", "wrap_graph", "write_operator_library",
]


@dataclass
class GeneratedKernel:
    """One generated kernel.

    source, header  texts of NAME.cu and NAME.h
    constants       UINT32 words that the kernel reads (FP32 values and integer tables as bit patterns)
    signature       ties the constants to the kernel; the kernel refuses constants with another signature
    input_shape, output_shape   of one row (the graph's shapes without the row axis)
    model           the graph with the kernel as its one node (and the original graph as the operator's function)
    summary         figures about the kernel
    parts           texts of further files by file name: a kernel that is two kernels in a row has the two as
                    NAME_1.cu and NAME_2.cu (with their headers), which NAME.cu includes
    """

    name: str
    source: str
    header: str
    constants: np.ndarray
    signature: str
    model: onnx.ModelProto
    summary: dict
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    parts: dict[str, str] = field(default_factory=dict)

    def write(self, directory: str) -> list[str]:
        """Write NAME.cu, NAME.h and NAME.onnx (and the files of ``parts``) into ``directory``; returns the paths
        of the three."""
        os.makedirs(directory, exist_ok=True)
        for file_name, text in self.parts.items():
            with open(os.path.join(directory, file_name), "w") as file:
                file.write(text)
        base = os.path.join(directory, self.name)
        with open(base + ".cu", "w") as file:
            file.write(self.source)
        with open(base + ".h", "w") as file:
            file.write(self.header)
        onnx.save(self.model, base + ".onnx")
        return [base + ".cu", base + ".h", base + ".onnx"]


def generate_kernel(onnx_path: str, name: str, options: Options | None = None) -> GeneratedKernel:
    """Generate the CUDA kernel ``name`` for the row-wise ONNX graph at ``onnx_path``."""
    options = options or Options()
    if not name.isidentifier() or name in ("kernel_ops", "kernel_direct"):
        raise ValueError(f"kernel name {name!r}: a C identifier other than kernel_ops and kernel_direct (files of the library)")
    start = time.perf_counter()
    with open(onnx_path, "rb") as file:
        digest = hashlib.sha256(file.read()).hexdigest()
    note = f"{os.path.basename(onnx_path)} (sha256 {digest})"
    def part_plans(whole: Plan, limits: Limits) -> list[Plan] | None:
        """The plans of the parts of the output row (schedule.output_parts), or None for one kernel. With "auto"
        the parts are taken where they bring product sums of loops onto the tensor cores: a product in code of
        one row per thread is counted as eight on the tensor cores (measured: 4 to 8
        times the time), and the parts must then cost less than two thirds of the whole graph."""
        if not options.tensor_cores or options.output_parts == "off":
            return None
        cost = lambda plan: products(plan)[0] + 8 * products(plan)[1]
        if options.output_parts == "auto" and 3 * products(whole)[0] > 2 * cost(whole):
            return None   # with the product sums on the tensor cores already, parts cannot cost a third less
        graph = load_row_graph(onnx_path)
        rewrite_graph(graph, limits, options.approximate_functions, True)
        parts = output_parts(graph)
        if len(parts) < 2:
            return None
        plans = []
        for part in parts:
            piece = sub_graph(graph, part)
            fold_constant_terms(piece)
            sink_gathers(piece, limits)   # again: a stage of copies had several index spaces, its part has one
            plans.append(make_plan(piece, limits, options.approximate_functions, True, rewritten=True))
        return plans if options.output_parts == "on" or 3 * sum(cost(plan) for plan in plans) < 2 * cost(whole) else None

    def emit(limits: Limits):
        plan = make_plan(load_row_graph(onnx_path), limits, options.approximate_functions, options.tensor_cores)
        shapes = plan.graph.input_shape, plan.graph.output_shape
        wide = any(isinstance(phase, DenseChain) and phase.in_blocks(plan.graph) for phase in plan.fragments.values())
        graphs = split_before_thread_loop(plan) if options.tensor_cores and options.split_kernels else None
        try:
            plans = part_plans(plan, limits)
            if plans is not None:   # one kernel per part of the output row
                pieces = [(f"{name}_{number + 1}", emit_kernel(part, f"{name}_{number + 1}", options,
                                                                f"{note}, part {number + 1} of {len(plans)} of the output row"))
                          for number, part in enumerate(plans)]
                return (*kernels_side_by_side(name, pieces), shapes)
            if graphs is None:
                return emit_kernel(plan, name, options, note), {}, shapes
            # two kernels in a row: everything ahead of the loop with its iterations by thread, then that loop
            settings = (options, replace(options, warpgroups=options.loop_warpgroups, tiles=options.loop_tiles))
            pair = [(f"{name}_{number + 1}",
                     emit_kernel(make_plan(graph, limits, options.approximate_functions, True, rewritten=True),
                                 f"{name}_{number + 1}", settings[number], f"{note}, kernel {number + 1} of 2"))
                    for number, graph in enumerate(graphs)]
            return (*kernels_in_a_row(name, pair, graphs[0].output_size, tuple(int(np.prod(shape)) for shape in shapes)), shapes)
        except KernelTooLarge:
            # Sums in blocks keep whole layers in the rows' slabs. Where the memory for them does not hold them,
            # the kernel is planned again: first with a narrow sum as the start of the chain it feeds (one layer
            # less per row), then without sums in blocks (these then stay sums of fmaf).
            if not (wide and limits.wide_sums):
                raise
            return emit(replace(limits, sums_once=False) if limits.sums_once else replace(limits, wide_sums=False))

    # a narrow sum as a chain of its own keeps one more layer per row: not with the rows' slabs in shared memory only
    limits = replace(options.limits, sums_once=False) if options.slab == "shared" else options.limits
    emitted, parts, (input_shape, output_shape) = emit(limits)
    model = wrap_graph(onnx.load(onnx_path), name, emitted.constants, emitted.signature)
    emitted.summary["generation_seconds"] = time.perf_counter() - start
    return GeneratedKernel(name, emitted.source, emitted.header, emitted.constants, emitted.signature, model,
                           emitted.summary, input_shape, output_shape, parts)


def check_file_names(kernels: list[GeneratedKernel]) -> None:
    """Kernels of one directory must not share a file: the names must differ from each other and from the names of
    the parts of a kernel that is two kernels in a row (NAME_1, NAME_2)."""
    files = [f"{kernel.name}.cu" for kernel in kernels] + [name for kernel in kernels for name in kernel.parts if name.endswith(".cu")]
    repeated = sorted({name for name in files if files.count(name) > 1})
    if repeated:
        raise ValueError(f"kernels of one library would share the files {repeated}: choose other kernel names")


def write_operator_library(directory: str, kernels: list[GeneratedKernel], direct_prefix: str = DEFAULT_PREFIX) -> list[str]:
    """Write the sources of the library of ``kernels`` into ``directory``: kernel_ops.cc (the custom operators of
    ONNX Runtime), kernel_direct.cc (the direct entry, functions ``direct_prefix``_open, _shape, _forward, _close)
    and build.sh."""
    check_file_names(kernels)
    paths = [os.path.join(directory, name) for name in ("kernel_ops.cc", "kernel_direct.cc", "build.sh")]
    with open(paths[0], "w") as file:
        file.write(operator_library_source(kernels))
    with open(paths[1], "w") as file:
        file.write(direct_entry_source(kernels, direct_prefix))
    with open(paths[2], "w") as file:
        architecture = "sm_90a" if any(kernel.summary["tensor_cores"] for kernel in kernels) else "sm_90"
        file.write(build_script([kernel.name for kernel in kernels], architecture, entry_names(direct_prefix)))
    return paths
