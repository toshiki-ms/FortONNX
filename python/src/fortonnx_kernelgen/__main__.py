"""Command line: python -m fortonnx_kernelgen --output-dir DIR NAME=GRAPH.onnx [NAME=GRAPH.onnx ...]"""

from __future__ import annotations

import argparse
import json

from . import Limits, Options, check_file_names, generate_kernel, write_operator_library
from .direct import DEFAULT_PREFIX


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m fortonnx_kernelgen",
        description="For every NAME=GRAPH write NAME.cu and NAME.h (one fused CUDA kernel of the row-wise ONNX graph) "
                    "and NAME.onnx (the graph with the kernel as an ONNX Runtime custom operator); and, for all of "
                    "them, the sources and the build script of one library: kernel_ops.cc (the custom operators), "
                    "kernel_direct.cc (the entry for programs that call the kernels without ONNX Runtime), build.sh.")
    parser.add_argument("kernels", nargs="+", metavar="NAME=GRAPH",
                        help="NAME: C identifier, prefix of the functions and files; GRAPH: ONNX file with one input "
                             "[rows, ...] and one output [rows, ...]")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threads-per-block", action="append", default=[], metavar="[NAME=]COUNT",
                        help="rows per block, for all kernels or for the kernel NAME (default: as many as the "
                             "shared memory holds, at most 256; 96 without per-row memory)")
    parser.add_argument("--dot-lanes", type=int, default=Options.dot_lanes,
                        help="product sums of one stage that advance together")
    parser.add_argument("--stage-bytes", type=int, default=Options.stage_bytes,
                        help="shared memory for the constants of a loop; 0 reads them from device memory")
    parser.add_argument("--body-terms", type=int, default=Limits.body_terms,
                        help="largest number of scalar operations and products in one loop body")
    parser.add_argument("--reciprocal-division", action="store_true",
                        help="divide by constants with their FP32 reciprocals (changes rounding by <= 1 ulp)")
    parser.add_argument("--approximate-functions", action="store_true",
                        help="exp, log, tanh and quotients with the hardware's approximate instructions "
                             "(rounding-size errors; see docs/kernel-generation.md)")
    parser.add_argument("--tensor-cores", action="store_true",
                        help="product sums at the head of loop bodies on the tensor cores (wgmma, sm_90a; 3xTF32)")
    parser.add_argument("--warpgroups", action="append", default=[], metavar="[NAME=]COUNT",
                        help="warpgroups (128 rows each) per block of a kernel on the tensor cores, for all kernels "
                             f"or for the kernel NAME (default {Options.warpgroups})")
    parser.add_argument("--loop-warpgroups", action="append", default=[], metavar="[NAME=]COUNT",
                        help="warpgroups per block of the second kernel, when a loop on the tensor cores becomes a "
                             f"kernel of its own (default {Options.loop_warpgroups})")
    parser.add_argument("--one-kernel", action="store_true",
                        help="keep a loop on the tensor cores in the kernel of what is ahead of it (default: two "
                             "kernels in a row, each with its own number of threads)")
    parser.add_argument("--direct-entry", default=DEFAULT_PREFIX, metavar="PREFIX",
                        help="prefix of the library's functions for programs that call the kernels without ONNX "
                             "Runtime: PREFIX_open, _shape, _forward, _close")
    parser.add_argument("--member-patterns", type=int, default=Options.member_patterns,
                        help="most bodies of one loop, one per pattern of its iterations (padded slots); 1: one body")
    parser.add_argument("--tiles", action="append", default=[], metavar="[NAME=]N",
                        help="most tiles of 64 rows per warpgroup of a kernel on the tensor cores, 1 or 2 "
                             f"(default {Options.tiles}; a kernel has one where the shared memory does not hold two)")
    parser.add_argument("--loop-tiles", action="append", default=[], metavar="[NAME=]N",
                        help=f"the same for the second kernel of two in a row (default {Options.loop_tiles})")
    parser.add_argument("--slab", choices=("auto", "shared", "device"), default=Options.slab,
                        help="where the rows keep their values between loops: the shared memory of the block, device "
                             "memory of the kernel object, or (auto) shared memory unless the block then has less "
                             "than half of its rows")
    parser.add_argument("--output-parts", choices=("auto", "on", "off"), default=Options.output_parts,
                        help="with --tensor-cores: one kernel per part of the output row whose outputs share their "
                             "product sums (a value and each of its tangents); auto: where that brings the product "
                             "sums of loops onto the tensor cores")
    parser.add_argument("--separate-stores", action="store_true",
                        help="on the tensor cores, every thread stores its own row's outputs (default: the threads of "
                             "a warp store consecutive elements of their rows together, which needs more registers)")
    arguments = parser.parse_args()
    threads, warpgroups, loop_warpgroups, tiles, loop_tiles = {}, {}, {}, {}, {}
    for items, counts in ((arguments.threads_per_block, threads), (arguments.warpgroups, warpgroups),
                          (arguments.loop_warpgroups, loop_warpgroups), (arguments.tiles, tiles),
                          (arguments.loop_tiles, loop_tiles)):
        for item in items:
            name, separator, count = item.rpartition("=")
            counts[name if separator else None] = int(count)
    kernels = []
    report = {}
    for item in arguments.kernels:
        name, separator, graph = item.partition("=")
        if not separator:
            parser.error(f"{item!r} is not NAME=GRAPH")
        options = Options(threads_per_block=threads.get(name, threads.get(None)), dot_lanes=arguments.dot_lanes,
                          stage_bytes=arguments.stage_bytes, reciprocal_division=arguments.reciprocal_division,
                          approximate_functions=arguments.approximate_functions,
                          tensor_cores=arguments.tensor_cores,
                          warpgroups=warpgroups.get(name, warpgroups.get(None, Options.warpgroups)),
                          loop_warpgroups=loop_warpgroups.get(name, loop_warpgroups.get(None, Options.loop_warpgroups)),
                          tiles=tiles.get(name, tiles.get(None, Options.tiles)), slab=arguments.slab,
                          loop_tiles=loop_tiles.get(name, loop_tiles.get(None, Options.loop_tiles)),
                          split_kernels=not arguments.one_kernel, output_parts=arguments.output_parts,
                          member_patterns=arguments.member_patterns, gathered_stores=not arguments.separate_stores,
                          limits=Limits(body_terms=arguments.body_terms))
        kernels.append(generate_kernel(graph, name, options))
    check_file_names(kernels)   # before any file is written
    for kernel in kernels:
        report[kernel.name] = {"files": kernel.write(arguments.output_dir), **kernel.summary}
    report["operator_library"] = write_operator_library(arguments.output_dir, kernels, arguments.direct_entry)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
