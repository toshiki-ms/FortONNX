"""Kernel generator: the row-wise form of small graphs against ONNX Runtime (CPU), the schedule, the generated texts
and the graph that carries the kernel as a custom operator. Nothing here needs a GPU (see test_kernelgen_cuda.py)."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from fortonnx_kernelgen import (
    DOMAIN,
    KernelTooLarge,
    Limits,
    Options,
    UnsupportedGraph,
    evaluate,
    generate_kernel,
    load_row_graph,
    make_plan,
    operator_library_source,
    write_operator_library,
)
from fortonnx_kernelgen.rowgraph import sub_graph
from fortonnx_kernelgen.schedule import fold_constant_terms, output_parts, rewrite_graph
from fortonnx_kernelgen.tensor import DenseChain

ROW_SIZE = 6


def _save(path: Path, nodes: list, initializers: dict, output_size: int, row_size: int = ROW_SIZE) -> Path:
    """Model ``rows [n, row_size] -> out [n, output_size]`` of the nodes and constants."""
    graph = helper.make_graph(
        nodes, "rows",
        [helper.make_tensor_value_info("rows", TensorProto.FLOAT, ["n", row_size])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["n", output_size])],
        initializer=[numpy_helper.from_array(np.asarray(value), name) for name, value in initializers.items()])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def _banded_model(path: Path) -> Path:
    """A small network with the constructs of the generator's target graphs: dense layers, a reduction over groups
    of planes, a Gather into padded bands, a reduction per band, a Gather back, and a constant-masked Where."""
    rng = np.random.default_rng(7)
    f32 = lambda *shape: rng.normal(size=shape).astype(np.float32) / 2
    outputs, bands, width, planes, hidden = 9, 3, 4, 3, 80
    to_slot = np.array([0, 1, 2, 4, 5, 6, 7, 8, 9], dtype=np.int64)          # output -> slot of (band, position)
    from_slot = np.array([0, 1, 2, 0, 3, 4, 5, 6, 7, 8, 0, 0], dtype=np.int64)   # slot -> output (0 where padded)
    valid = np.array([1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0], dtype=bool).reshape(1, bands, width)
    initializers = {
        "w1": f32(ROW_SIZE, hidden), "b1": f32(hidden), "w2": f32(hidden, outputs * planes), "b2": f32(outputs * planes),
        "shape_planes": np.array([0, outputs, planes], dtype=np.int64), "axis_last": np.array([2], dtype=np.int64),
        "to_slot": to_slot, "from_slot": from_slot, "shape_bands": np.array([0, bands, width], dtype=np.int64),
        "shape_flat": np.array([0, bands * width], dtype=np.int64), "valid": valid,
        "low": np.array(-1.0e30, dtype=np.float32), "scale": np.array(4.0, dtype=np.float32),
    }
    nodes = [
        helper.make_node("MatMul", ["rows", "w1"], ["d1"]),
        helper.make_node("Add", ["d1", "b1"], ["a1"]),
        helper.make_node("Tanh", ["a1"], ["h1"]),
        helper.make_node("MatMul", ["h1", "w2"], ["d2"]),
        helper.make_node("Add", ["d2", "b2"], ["a2"]),
        helper.make_node("Reshape", ["a2", "shape_planes"], ["planes"]),
        helper.make_node("ReduceMax", ["planes", "axis_last"], ["top"], keepdims=0),
        helper.make_node("Div", ["top", "scale"], ["scaled"]),
        helper.make_node("Gather", ["scaled", "from_slot"], ["slots"], axis=1),
        helper.make_node("Reshape", ["slots", "shape_bands"], ["bands"]),
        helper.make_node("Where", ["valid", "bands", "low"], ["masked"]),
        helper.make_node("ReduceMax", ["masked", "axis_last"], ["band_top"], keepdims=1),
        helper.make_node("Sub", ["masked", "band_top"], ["shifted"]),
        helper.make_node("Exp", ["shifted"], ["weights"]),
        helper.make_node("ReduceSum", ["weights", "axis_last"], ["total"], keepdims=1),
        helper.make_node("Div", ["weights", "total"], ["shares"]),
        helper.make_node("Reshape", ["shares", "shape_flat"], ["flat"]),
        helper.make_node("Gather", ["flat", "to_slot"], ["picked"], axis=1),
        helper.make_node("Sqrt", ["picked"], ["out"]),
    ]
    return _save(path, nodes, initializers, outputs)


def _rows(count: int = 37) -> np.ndarray:
    return np.random.default_rng(3).normal(size=(count, ROW_SIZE)).astype(np.float32)


def _reference(path: Path, rows: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, {"rows": rows})[0]


def test_row_graph_matches_onnx_runtime(tmp_path: Path) -> None:
    path = _banded_model(tmp_path / "banded.onnx")
    rows = _rows()
    graph = load_row_graph(str(path))
    assert graph.input_size == ROW_SIZE and graph.output_size == 9
    np.testing.assert_allclose(evaluate(graph, rows), _reference(path, rows), rtol=2e-5, atol=1e-6)


def test_schedule_keeps_values_and_fuses_the_bands(tmp_path: Path) -> None:
    path = _banded_model(tmp_path / "banded.onnx")
    rows = _rows()
    graph = load_row_graph(str(path))
    before = evaluate(graph, rows, np.float64)
    plan = make_plan(graph, Limits(register_array=8))
    # the rewrites for fusion are exact
    np.testing.assert_array_equal(evaluate(graph, rows, np.float64), before)
    loops = sorted(group.members for group in plan.groups if group.members > 1)
    assert 3 in loops, loops            # the per-band part runs as one loop over the bands
    assert 80 in loops, loops           # the hidden layer runs as a loop over its units
    # the hidden layer is the only stage that needs per-row shared memory besides the values the bands gather
    assert plan.slab_words % 4 == 0 and (plan.slab_words // 4) % 2 == 1


def test_generated_kernel_texts_and_operator_graph(tmp_path: Path) -> None:
    path = _banded_model(tmp_path / "banded.onnx")
    kernel = generate_kernel(str(path), "banded", Options(limits=Limits(register_array=8)))
    assert "__global__ void evaluate_rows" in kernel.source and "fmaf(" in kernel.source
    assert "int banded_create(const void* constants" in kernel.header
    assert kernel.constants.dtype == np.uint32 and kernel.constants.size % 4 == 0
    assert len(kernel.signature) == 64 and kernel.signature in kernel.source
    assert kernel.summary["threads_per_block"] % 32 == 0

    files = kernel.write(str(tmp_path / "out"))
    assert [Path(f).name for f in files] == ["banded.cu", "banded.h", "banded.onnx"]
    library = write_operator_library(str(tmp_path / "out"), [kernel])
    assert [Path(f).name for f in library] == ["kernel_ops.cc", "kernel_direct.cc", "build.sh"]
    assert "for name in banded; do" in Path(library[2]).read_text()

    # main graph: one node of the operator; without the library ONNX Runtime runs the function body
    model = onnx.load(files[2])
    assert [(n.domain, n.op_type) for n in model.graph.node] == [(DOMAIN, "banded")]
    assert [i.name for i in model.graph.initializer] == ["banded.constants"]
    rows = _rows()
    body = ort.InferenceSession(files[2], providers=["CPUExecutionProvider"]).run(None, {"rows": rows})[0]
    np.testing.assert_array_equal(body, _reference(path, rows))

    assert kernel.input_shape == (ROW_SIZE,) and kernel.output_shape == (9,)
    source = operator_library_source([kernel, generate_kernel(str(path), "other", Options(limits=Limits(register_array=8)))])
    assert '{"banded", &kInputShape_banded' in source and "RowOp(&kKernels[1])" in source
    with pytest.raises(ValueError, match="names must differ"):
        operator_library_source([kernel, generate_kernel(str(path), "Banded")])


def test_kernel_changes_with_settings_but_not_with_names(tmp_path: Path) -> None:
    path = _banded_model(tmp_path / "banded.onnx")
    limits = Limits(register_array=8)
    first = generate_kernel(str(path), "a", Options(limits=limits))
    again = generate_kernel(str(path), "b", Options(limits=limits))
    other = generate_kernel(str(path), "a", Options(limits=limits, stage_bytes=0))
    assert first.signature == again.signature
    np.testing.assert_array_equal(first.constants, again.constants)
    assert first.signature != other.signature


def test_row_axis_last_in_matmul(tmp_path: Path) -> None:
    """Per-band layers written as [band, out, in] @ [band, in, rows] (the row axis moved last by Transpose)."""
    rng = np.random.default_rng(11)
    bands, width = 3, 2
    weight = (rng.normal(size=(bands, width, ROW_SIZE)) / 2).astype(np.float32)
    initializers = {"weight": weight, "axis1": np.array([1], dtype=np.int64),
                    "expand": np.array([1, bands, ROW_SIZE], dtype=np.int64),
                    "flat": np.array([0, bands * width], dtype=np.int64)}
    nodes = [
        helper.make_node("Unsqueeze", ["rows", "axis1"], ["r1"]),
        helper.make_node("Expand", ["r1", "expand"], ["r2"]),
        helper.make_node("Transpose", ["r2"], ["r3"], perm=[1, 2, 0]),
        helper.make_node("MatMul", ["weight", "r3"], ["d"]),
        helper.make_node("Transpose", ["d"], ["t"], perm=[2, 0, 1]),
        helper.make_node("Reshape", ["t", "flat"], ["out"]),
    ]
    path = _save(tmp_path / "bands.onnx", nodes, initializers, bands * width)
    rows = _rows()
    np.testing.assert_allclose(evaluate(load_row_graph(str(path)), rows), _reference(path, rows), rtol=2e-5, atol=1e-6)


def _general(path: Path, nodes: list, initializers: dict, input_shape: list, output_shape: list) -> Path:
    graph = helper.make_graph(
        nodes, "rows", [helper.make_tensor_value_info("rows", TensorProto.FLOAT, ["n"] + input_shape)],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["n"] + output_shape)],
        initializer=[numpy_helper.from_array(np.asarray(value), name) for name, value in initializers.items()])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def test_values_at_the_edges_follow_onnx(tmp_path: Path) -> None:
    """Softplus of large arguments, a Slice that steps backwards past the start, divisors of both zero signs and a
    divisor whose reciprocal is not an FP32 number."""
    initializers = {
        "starts": np.array([-9], dtype=np.int64), "ends": np.array([-9], dtype=np.int64),
        "axes": np.array([1], dtype=np.int64), "steps": np.array([-1], dtype=np.int64),
        "zeros": np.array([-0.0, 0.0], dtype=np.float32), "tiny": np.array(2.0 ** -128, dtype=np.float32),
        "first_two": np.array([0, 1], dtype=np.int64), "scale": np.array(64.0, dtype=np.float32),
    }
    nodes = [
        helper.make_node("Mul", ["rows", "scale"], ["large"]),
        helper.make_node("Softplus", ["large"], ["soft"]),
        helper.make_node("Slice", ["soft", "starts", "ends", "axes", "steps"], ["head"]),      # the first column
        helper.make_node("Gather", ["rows", "first_two"], ["pair"], axis=1),
        helper.make_node("Div", ["pair", "zeros"], ["signed"]),
        helper.make_node("Div", ["pair", "tiny"], ["huge"]),
        helper.make_node("Concat", ["head", "signed", "huge"], ["out"], axis=1),
    ]
    path = _save(tmp_path / "edges.onnx", nodes, initializers, 5)
    rows = _rows()
    rows[:, 0] = np.abs(rows[:, 0]) + 2.0      # Softplus argument above 128
    expected = _reference(path, rows)
    assert np.isfinite(expected[:, 0]).all() and np.isinf(expected[:, 1:3]).all()
    with np.errstate(all="ignore"):
        np.testing.assert_allclose(evaluate(load_row_graph(str(path)), rows), expected, rtol=2e-6)
    source = generate_kernel(str(path), "edges").source
    assert "INFINITY" not in source and "/ 2.93873588e-39f" in source   # the division by 2**-128 stays a division


def test_rows_of_higher_rank_and_signature_of_the_row_width(tmp_path: Path) -> None:
    weight = np.arange(6, dtype=np.float32).reshape(3, 2) / 4
    nodes = [helper.make_node("MatMul", ["rows", "weight"], ["out"])]
    kernel = generate_kernel(str(_general(tmp_path / "rank3.onnx", nodes, {"weight": weight}, [2, 3], [2, 2])), "rank3")
    assert kernel.input_shape == (2, 3) and kernel.output_shape == (2, 2)
    assert "kInputShape_rank3 = {2, 3}" in operator_library_source([kernel])

    # the same statements for another row width are another kernel
    pick = {"first": np.array([0], dtype=np.int64)}
    column = [helper.make_node("Gather", ["rows", "first"], ["out"], axis=1)]
    narrow = generate_kernel(str(_save(tmp_path / "narrow.onnx", column, pick, 1, row_size=2)), "pick")
    wide = generate_kernel(str(_save(tmp_path / "wide.onnx", column, pick, 1, row_size=3)), "pick")
    assert narrow.signature != wide.signature


@pytest.mark.parametrize("nodes, initializers, reason", [
    ([helper.make_node("ReduceSum", ["rows", "axis0"], ["out"], keepdims=1)],
     {"axis0": np.array([0], dtype=np.int64)}, "row axis"),
    ([helper.make_node("ReduceSum", ["rows", "none"], ["out"], keepdims=1)],
     {"none": np.array([], dtype=np.int64)}, "row axis"),
    ([helper.make_node("Cast", ["rows"], ["wide"], to=TensorProto.DOUBLE),
      helper.make_node("Cast", ["wide"], ["out"], to=TensorProto.FLOAT)], {}, "FP32"),
    ([helper.make_node("Softmax", ["rows"], ["out"])], {}, "not supported"),
    ([helper.make_node("Transpose", ["rows"], ["t"], perm=[1, 0]), helper.make_node("MatMul", ["t", "rows"], ["out"])],
     {}, "row axis"),
    ([helper.make_node("Gather", ["rows", "pick"], ["out"], axis=0)], {"pick": np.array([0], dtype=np.int64)}, "rows"),
])
def test_graphs_that_are_not_row_wise_stop_the_import(tmp_path: Path, nodes, initializers, reason) -> None:
    graph = helper.make_graph(
        nodes, "bad", [helper.make_tensor_value_info("rows", TensorProto.FLOAT, ["n", ROW_SIZE])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, None)],
        initializer=[numpy_helper.from_array(value, name) for name, value in initializers.items()])
    path = tmp_path / "bad.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=8), path)
    with pytest.raises(UnsupportedGraph, match=reason):
        load_row_graph(str(path))


def test_rows_that_do_not_fit_shared_memory_stop_the_generation(tmp_path: Path) -> None:
    path = _banded_model(tmp_path / "banded.onnx")
    with pytest.raises(KernelTooLarge, match="shared memory"):
        generate_kernel(str(path), "banded", Options(shared_memory_bytes=4096, limits=Limits(register_array=8)))


def _logsumexp_model(path: Path, clamped: bool = False) -> Path:
    """log(sum(exp(.))) over groups of three as an exporter writes it (with the guards for infinite maxima). The
    terms are a gate exp(4 tanh(x / 2)) with a product with a constant behind it and the result is divided by a
    constant; or (``clamped``) the terms are tanh(x) and the divided result is limited from above."""
    initializers = {
        "shape_groups": np.array([0, 2, 3], dtype=np.int64), "axis_last": np.array([2], dtype=np.int64),
        "zero": np.array(0.0, dtype=np.float32), "half": np.array(2.0, dtype=np.float32),
        "four": np.array(4.0, dtype=np.float32), "ln10": np.array(2.302585, dtype=np.float32),
        "limit": np.array(0.65, dtype=np.float32),
    }
    terms = [helper.make_node("Tanh", ["rows"], ["terms_flat"])] if clamped else [
        helper.make_node("Div", ["rows", "half"], ["halved"]),
        helper.make_node("Tanh", ["halved"], ["bounded"]),
        helper.make_node("Mul", ["bounded", "four"], ["scaled"]),
        helper.make_node("Exp", ["scaled"], ["gate"]),
        helper.make_node("Mul", ["gate", "ln10"], ["terms_flat"]),
    ]
    result = [helper.make_node("Div", ["flat", "ln10"], ["out"])]
    if clamped:
        result = [helper.make_node("Div", ["flat", "ln10"], ["divided"]), helper.make_node("Min", ["divided", "limit"], ["out"])]
    nodes = terms + [
        helper.make_node("Reshape", ["terms_flat", "shape_groups"], ["terms"]),
        helper.make_node("ReduceMax", ["terms", "axis_last"], ["top"], keepdims=1),
        helper.make_node("IsInf", ["top"], ["top_inf"]),
        helper.make_node("IsNaN", ["top"], ["top_nan"]),
        helper.make_node("Or", ["top_inf", "top_nan"], ["top_bad"]),
        helper.make_node("Not", ["top_bad"], ["top_finite"]),
        helper.make_node("Where", ["top_finite", "top", "zero"], ["shift"]),
        helper.make_node("Sub", ["terms", "shift"], ["shifted"]),
        helper.make_node("Exp", ["shifted"], ["powers"]),
        helper.make_node("ReduceSum", ["powers", "axis_last"], ["total"], keepdims=1),
        helper.make_node("Abs", ["total"], ["magnitude"]),
        helper.make_node("Log", ["magnitude"], ["logarithm"]),
        helper.make_node("Add", ["logarithm", "shift"], ["sums"]),
        helper.make_node("Squeeze", ["sums", "axis_last"], ["flat"]),
    ] + result
    return _save(path, nodes, initializers, 2)


def test_approximate_forms_keep_the_values(tmp_path: Path) -> None:
    rows = _rows()
    # the banded network: its second dense layer is a chain on the tensor cores, its bands have padded slots
    path = _banded_model(tmp_path / "banded.onnx")
    graph = load_row_graph(str(path))
    plan = make_plan(graph, Limits(register_array=8), approximate_functions=True, tensor_cores=True)
    np.testing.assert_allclose(evaluate(graph, rows, np.float64), _reference(path, rows), rtol=2e-5, atol=1e-6)
    assert any(type(phase).__name__ == "DenseChain" for phase in plan.fragments.values())
    # the exporter's forms become single stages, and the scales around them are folded into constants
    path = _logsumexp_model(tmp_path / "logsumexp.onnx")
    graph = load_row_graph(str(path))
    make_plan(graph, Limits(), approximate_functions=True)
    operators = {stage.op for stage in graph.stages}
    assert {"logsumexp", "exp_scaled_tanh"} <= operators, operators
    fused = next(stage for stage in graph.stages if stage.op == "logsumexp")
    assert fused.log2_units == (True, True)
    np.testing.assert_allclose(evaluate(graph, rows, np.float64), _reference(path, rows), rtol=2e-5, atol=1e-6)
    kernel = generate_kernel(str(path), "sums", Options(approximate_functions=True))
    assert "approx_exp_tanh(" in kernel.source and "isinf" not in kernel.source
    # a limit that the result can reach stays (the ranges are in the units of the folded scales)
    path = _logsumexp_model(tmp_path / "clamped.onnx", clamped=True)
    graph = load_row_graph(str(path))
    make_plan(graph, Limits(), approximate_functions=True)
    np.testing.assert_allclose(evaluate(graph, rows, np.float64), _reference(path, rows), rtol=2e-5, atol=1e-6)
    # a stage that is both operands of a sum takes the factor of the product behind it once
    path = _save(tmp_path / "twice.onnx",
                 [helper.make_node("Mul", ["rows", "two"], ["doubled"]), helper.make_node("Add", ["doubled", "doubled"], ["sum"]),
                  helper.make_node("Mul", ["sum", "three"], ["out"])],
                 {"two": np.array(2.0, dtype=np.float32), "three": np.array(3.0, dtype=np.float32)}, ROW_SIZE)
    graph = load_row_graph(str(path))
    make_plan(graph, Limits(), approximate_functions=True)
    np.testing.assert_allclose(evaluate(graph, rows, np.float64), _reference(path, rows), rtol=2e-5, atol=1e-6)


def test_gather_outside_the_axis_stops_the_import(tmp_path: Path) -> None:
    path = _save(tmp_path / "gather.onnx", [helper.make_node("Gather", ["rows", "index"], ["out"], axis=1)],
                 {"index": np.array([0, ROW_SIZE], dtype=np.int64)}, 2)
    with pytest.raises(UnsupportedGraph, match="outside the axis"):
        load_row_graph(str(path))


def test_constants_added_twice_to_a_product_sum(tmp_path: Path) -> None:
    """A product sum takes an added constant as its bias only while the addition is its one reader."""
    initializers = {"w": np.eye(ROW_SIZE, dtype=np.float32), "one": np.array(1.0, dtype=np.float32),
                    "two": np.array(2.0, dtype=np.float32)}
    nodes = [helper.make_node("MatMul", ["rows", "w"], ["d"]), helper.make_node("Add", ["d", "one"], ["a"]),
             helper.make_node("Add", ["a", "two"], ["b"]), helper.make_node("Concat", ["a", "b"], ["out"], axis=1)]
    path = _save(tmp_path / "biases.onnx", nodes, initializers, 2 * ROW_SIZE)
    rows = _rows()
    graph = load_row_graph(str(path))
    make_plan(graph, Limits(), tensor_cores=True)
    np.testing.assert_allclose(evaluate(graph, rows, np.float64), _reference(path, rows), rtol=2e-5, atol=1e-6)


def test_constants_whose_sum_is_no_fp32_number_are_not_folded(tmp_path: Path) -> None:
    """Constant terms of a product sum, and constants added one after the other, stay apart where their sum alone
    overflows: with the row's values the result is finite."""
    big = np.array(2.0e38, dtype=np.float32)
    rows = np.full((4, 3), -3.0e38, dtype=np.float32)
    nodes = [helper.make_node("Where", ["first", "rows", "big"], ["x"]), helper.make_node("MatMul", ["x", "ones"], ["out"])]
    initializers = {"first": np.array([True, False, False]), "big": big, "ones": np.ones((3, 1), dtype=np.float32)}
    path = _save(tmp_path / "terms.onnx", nodes, initializers, 1, row_size=3)
    graph = load_row_graph(str(path))
    assert np.isfinite(_reference(path, rows)).all()
    np.testing.assert_allclose(evaluate(graph, rows), _reference(path, rows), rtol=1e-6)
    nodes = [helper.make_node("MatMul", ["rows", "one"], ["d"]), helper.make_node("Add", ["d", "big"], ["a"]),
             helper.make_node("Add", ["a", "big"], ["out"])]
    path = _save(tmp_path / "adds.onnx", nodes, {"one": np.ones((1, 1), dtype=np.float32), "big": big}, 1, row_size=1)
    graph = load_row_graph(str(path))
    make_plan(graph, Limits(), tensor_cores=True)
    np.testing.assert_allclose(evaluate(graph, rows[:, :1]), _reference(path, rows[:, :1]), rtol=1e-6)


def test_factors_outside_fp32_are_not_folded(tmp_path: Path) -> None:
    initializers = {"large": np.array(1.0e30, dtype=np.float32), "more": np.array(1.0e10, dtype=np.float32),
                    "tiny": np.array(1.0e-30, dtype=np.float32)}
    nodes = [helper.make_node("Mul", ["rows", "tiny"], ["small"]), helper.make_node("Mul", ["small", "large"], ["back"]),
             helper.make_node("Mul", ["back", "more"], ["out"])]
    path = _save(tmp_path / "factors.onnx", nodes, initializers, ROW_SIZE)
    rows = _rows()
    graph = load_row_graph(str(path))
    make_plan(graph, Limits(), approximate_functions=True)
    np.testing.assert_allclose(evaluate(graph, rows), _reference(path, rows), rtol=1e-5)


def test_constants_of_other_types_and_reserved_names_stop_the_generation(tmp_path: Path) -> None:
    initializers = {"a": np.array(16777217.0, dtype=np.float64), "b": np.array(16777216.0, dtype=np.float64)}
    nodes = [helper.make_node("Sub", ["a", "b"], ["difference"]),
             helper.make_node("Cast", ["difference"], ["one"], to=TensorProto.FLOAT),
             helper.make_node("Add", ["rows", "one"], ["out"])]
    path = _save(tmp_path / "double.onnx", nodes, initializers, ROW_SIZE)
    with pytest.raises(UnsupportedGraph, match="float64 constants"):
        load_row_graph(str(path))
    with pytest.raises(ValueError, match="kernel_ops"):
        generate_kernel(str(_banded_model(tmp_path / "banded.onnx")), "kernel_ops")


def test_names_of_the_model_stay_inside_comments(tmp_path: Path) -> None:
    """An ONNX name may hold line breaks: none reaches the generated source as a line of its own."""
    injected = "h\n#error INJECTED"
    path = _save(tmp_path / "names.onnx", [helper.make_node("Tanh", ["rows"], [injected]), helper.make_node("Neg", [injected], ["out"])],
                 {}, ROW_SIZE)
    kernel = generate_kernel(str(path), "names")
    assert not any(line.lstrip().startswith("#error") for line in kernel.source.splitlines())


def test_squeeze_without_axes_of_a_row_value_stops_the_import(tmp_path: Path) -> None:
    path = _save(tmp_path / "squeeze.onnx", [helper.make_node("Squeeze", ["rows", "none"], ["out"])],
                 {"none": np.array([], dtype=np.int64)}, ROW_SIZE)
    with pytest.raises(UnsupportedGraph, match="explicit axes"):
        load_row_graph(str(path))


def test_constants_keep_their_onnx_types(tmp_path: Path) -> None:
    # an FP64 exponent of Pow would lose bits as an FP32 constant
    path = _save(tmp_path / "pow.onnx", [helper.make_node("Pow", ["rows", "exponent"], ["out"])],
                 {"exponent": np.array(16777217.0, dtype=np.float64)}, ROW_SIZE)
    with pytest.raises(UnsupportedGraph, match="float64 constant"):
        load_row_graph(str(path))
    # a reduction of INT32 constants stays INT32, as in ONNX (NumPy would widen it)
    initializers = {"counts": np.array([3, 4], dtype=np.int32), "axis": np.array([0], dtype=np.int64),
                    "seven": np.array(7, dtype=np.int32)}
    nodes = [helper.make_node("ReduceSum", ["counts", "axis"], ["total"], keepdims=0),
             helper.make_node("Equal", ["total", "seven"], ["same"]),
             helper.make_node("Cast", ["same"], ["flag"], to=TensorProto.FLOAT),
             helper.make_node("Add", ["rows", "flag"], ["out"])]
    path = _save(tmp_path / "counts.onnx", nodes, initializers, ROW_SIZE)
    rows = _rows()
    np.testing.assert_allclose(evaluate(load_row_graph(str(path)), rows), _reference(path, rows), rtol=1e-6)
    # the build script takes the include directory relative to the caller's directory
    kernel = generate_kernel(str(_banded_model(tmp_path / "banded.onnx")), "banded")
    write_operator_library(str(tmp_path), [kernel])
    assert 'include=$(cd "$1" && pwd)' in (tmp_path / "build.sh").read_text()


def test_library_sources_have_both_entries(tmp_path: Path) -> None:
    kernel = generate_kernel(str(_banded_model(tmp_path / "banded.onnx")), "banded")
    paths = write_operator_library(str(tmp_path), [kernel], direct_prefix="net")
    assert [Path(path).name for path in paths] == ["kernel_ops.cc", "kernel_direct.cc", "build.sh"]
    direct = (tmp_path / "kernel_direct.cc").read_text()
    assert all(f"net_{part}(" in direct for part in ("open", "shape", "forward", "close"))
    assert "onnxruntime" not in direct and '{"banded", 6, 9, banded_create' in direct
    assert "global: RegisterCustomOps; net_open; net_shape; net_forward; net_close; local: *;" in (tmp_path / "build.sh").read_text()
    with pytest.raises(ValueError, match="identifier"):
        write_operator_library(str(tmp_path), [kernel], direct_prefix="not a name")


def _undeclared_locals(source: str) -> set[str]:
    """Names v<stage>_<element> that the kernel source uses without declaring them."""
    import re
    body = source[source.index("__global__ void evaluate_rows"):]
    declared = set(re.findall(r"(?:float|bool) (v\d+_\d+)\b", body))
    return set(re.findall(r"\bv\d+_\d+\b", body)) - declared


def test_one_body_for_iterations_with_masked_values(tmp_path: Path) -> None:
    """With one body for all iterations, a masked value that no iteration reads is not referenced."""
    valid = np.ones((1, 130), dtype=bool)
    valid[0, 0] = False
    initializers = {"valid": valid, "zero": np.array(0.0, dtype=np.float32), "pairs": np.array([0, 65, 2], dtype=np.int64),
                    "axis_last": np.array([2], dtype=np.int64), "first": np.array([0], dtype=np.int64)}
    nodes = [helper.make_node("Tanh", ["rows"], ["t"]), helper.make_node("Where", ["valid", "t", "zero"], ["masked"]),
             helper.make_node("Neg", ["masked"], ["negative"]), helper.make_node("Reshape", ["negative", "pairs"], ["paired"]),
             helper.make_node("ReduceSum", ["paired", "axis_last"], ["sums"], keepdims=0),
             helper.make_node("Gather", ["sums", "first"], ["out"], axis=1)]
    path = _save(tmp_path / "masked.onnx", nodes, initializers, 1, row_size=130)
    for options in (Options(member_patterns=1), Options()):
        assert not _undeclared_locals(generate_kernel(str(path), "masked", options).source)
    assert not _undeclared_locals(generate_kernel(str(_banded_model(tmp_path / "banded.onnx")), "banded",
                                                  Options(member_patterns=1, limits=Limits(register_array=8))).source)


def test_gate_keeps_a_scale_that_is_no_fp32_number_with_the_base_two_factor(tmp_path: Path) -> None:
    initializers = {"huge": np.array(2.0e38, dtype=np.float32), "four": np.array(4.0, dtype=np.float32)}
    nodes = [helper.make_node("Mul", ["rows", "huge"], ["scaled"]), helper.make_node("Tanh", ["scaled"], ["bounded"]),
             helper.make_node("Mul", ["bounded", "four"], ["gate"]), helper.make_node("Exp", ["gate"], ["out"])]
    path = _save(tmp_path / "gate.onnx", nodes, initializers, ROW_SIZE)
    graph = load_row_graph(str(path))
    make_plan(graph, Limits(), approximate_functions=True)
    fused = next(stage for stage in graph.stages if stage.op == "exp_scaled_tanh")
    assert (graph.const_values(fused.operands[2]) == 1.0).all()   # the product with 2e38 stays a stage of its own


def test_one_body_keeps_the_stores_and_weights_of_every_iteration(tmp_path: Path) -> None:
    """A loop with one body (a sum is accumulated in it): an element that is an output in one iteration only is
    stored in that iteration only, and every product sum takes the weights of its own iteration."""
    rng = np.random.default_rng(21)
    initializers = {"w": rng.normal(size=(80, 3)).astype(np.float32), "first": np.array([0], dtype=np.int64)}
    nodes = [helper.make_node("Tanh", ["rows"], ["t"]), helper.make_node("Neg", ["t"], ["negative"]),
             helper.make_node("Gather", ["negative", "first"], ["one"], axis=1),
             helper.make_node("MatMul", ["t", "w"], ["sums"]), helper.make_node("Concat", ["one", "sums"], ["out"], axis=1)]
    path = _save(tmp_path / "stores.onnx", nodes, initializers, 4, row_size=80)
    source = generate_kernel(str(path), "stores").source
    assert "if (active) out[0] =" not in source   # the first output is stored in one iteration of the loop only
    assert not _undeclared_locals(source)


def _iterations_by_thread(path: Path) -> Path:
    """Ten iterations whose product sums all read the same sixteen values of the row (computed ahead of the loop),
    twelve per iteration, and a reduction per iteration: four iterations share an instruction, and the loop
    becomes a kernel of its own behind the kernel of the sixteen values."""
    rng = np.random.default_rng(14)
    f32 = lambda *shape: rng.normal(size=shape).astype(np.float32) / 3
    initializers = {"w0": f32(6, 16), "w1": f32(16, 120), "b1": f32(120), "shape": np.array([0, 10, 12], dtype=np.int64),
                    "axis_last": np.array([2], dtype=np.int64)}
    nodes = [helper.make_node("MatMul", ["rows", "w0"], ["d0"]), helper.make_node("Tanh", ["d0"], ["x"]),
             helper.make_node("MatMul", ["x", "w1"], ["d1"]), helper.make_node("Add", ["d1", "b1"], ["a1"]),
             helper.make_node("Reshape", ["a1", "shape"], ["parts"]),
             helper.make_node("ReduceMax", ["parts", "axis_last"], ["out"], keepdims=0)]
    return _save(path, nodes, initializers, 10)


def _loop_tied_to_a_loop(path: Path) -> Path:
    """As ``_iterations_by_thread`` with 102 iterations, then 99 iterations that each read four of the 102 values
    in rising order (values j to j + 3; the largest minus the smallest of them): the second loop runs inside the
    first, an iteration as soon as its values exist."""
    rng = np.random.default_rng(15)
    f32 = lambda *shape: rng.normal(size=shape).astype(np.float32) / 3
    initializers = {"w0": f32(6, 16), "w1": f32(16, 1224), "b1": f32(1224), "shape": np.array([0, 102, 12], dtype=np.int64),
                    "windows": np.arange(99, dtype=np.int64)[:, None] + np.arange(4, dtype=np.int64),
                    "axis_last": np.array([2], dtype=np.int64)}
    nodes = [helper.make_node("MatMul", ["rows", "w0"], ["d0"]), helper.make_node("Tanh", ["d0"], ["x"]),
             helper.make_node("MatMul", ["x", "w1"], ["d1"]), helper.make_node("Add", ["d1", "b1"], ["a1"]),
             helper.make_node("Reshape", ["a1", "shape"], ["parts"]),
             helper.make_node("ReduceMax", ["parts", "axis_last"], ["values"], keepdims=0),
             helper.make_node("Gather", ["values", "windows"], ["four"], axis=1),
             helper.make_node("ReduceMin", ["four", "axis_last"], ["low"], keepdims=0),
             helper.make_node("ReduceMax", ["four", "axis_last"], ["high"], keepdims=0),
             helper.make_node("Sub", ["high", "low"], ["out"])]
    return _save(path, nodes, initializers, 99)


def _dense_layers(path: Path, row: int, sizes: list[int], skip: bool = False, ahead: bool = False, name: str = "",
                  nodes: list | None = None, initializers: dict | None = None) -> Path | str:
    """Product sums one after the other (``sizes`` elements each) with tanh between them, on a row of ``row``
    values. ``skip`` adds the first tanh to the second; ``ahead`` takes the tanh of the row first. With ``nodes``
    and ``initializers`` the layers are added to them and the name of the last sum is returned."""
    rng = np.random.default_rng(18 + len(sizes) + row)
    own = nodes is None
    nodes, initializers = ([], {}) if own else (nodes, initializers)
    value = "rows"
    if ahead:
        nodes.append(helper.make_node("Tanh", ["rows"], [f"{name}t"]))
        value = f"{name}t"
    for number, (terms, size) in enumerate(zip([row] + sizes[:-1], sizes)):
        initializers[f"{name}w{number}"] = (rng.normal(size=(terms, size)) / 4).astype(np.float32)
        initializers[f"{name}b{number}"] = (rng.normal(size=size) / 4).astype(np.float32)
        nodes += [helper.make_node("MatMul", [value, f"{name}w{number}"], [f"{name}d{number}"]),
                  helper.make_node("Add", [f"{name}d{number}", f"{name}b{number}"], [f"{name}a{number}"])]
        value = f"{name}a{number}"
        if number + 1 < len(sizes):
            nodes.append(helper.make_node("Tanh", [value], [f"{name}h{number}"]))
            value = f"{name}h{number}"
            if skip and number == 1:
                nodes.append(helper.make_node("Add", [value, f"{name}h0"], [f"{name}s"]))
                value = f"{name}s"
    if not own:
        return value
    nodes[-1].output[0] = "out"
    return _save(path, nodes, initializers, sizes[-1], row_size=row)


def _four_sums(path: Path) -> Path:
    return _dense_layers(path, 6, [64, 64, 64, 24])


def _sums_of_input_values(path: Path) -> Path:
    return _dense_layers(path, 40, [64, 24])


def _one_sum(path: Path) -> Path:
    return _dense_layers(path, 64, [64])


def _sums_with_a_skip(path: Path) -> Path:
    return _dense_layers(path, 6, [64, 64, 24], skip=True)


def _sums_with_a_side(path: Path) -> Path:
    """Two sums with ``tanh(first sum) + sin(rows)`` between them: the sine is computed ahead of the chain and
    read beside it."""
    rng = np.random.default_rng(19)
    initializers = {"w1": (rng.normal(size=(32, 32)) / 4).astype(np.float32), "w2": (rng.normal(size=(32, 24)) / 4).astype(np.float32)}
    nodes = [helper.make_node("Sin", ["rows"], ["side"]), helper.make_node("MatMul", ["rows", "w1"], ["d1"]),
             helper.make_node("Tanh", ["d1"], ["h1"]), helper.make_node("Add", ["h1", "side"], ["s1"]),
             helper.make_node("MatMul", ["s1", "w2"], ["out"])]
    return _save(path, nodes, initializers, 24, row_size=32)


def _sums_of_computed_values(path: Path) -> Path:
    return _dense_layers(path, 24, [64, 16], ahead=True)


def _sum_of_many_terms(path: Path) -> Path:
    return _dense_layers(path, 6, [200, 24])       # the second sum has 200 terms: two ranges of its A operand


def _sums_in_blocks(path: Path) -> Path:
    return _dense_layers(path, 150, [140, 24])     # 150 terms and 140 elements, then 140 terms


def _two_chains(path: Path) -> Path:
    nodes, initializers = [], {}
    ends = [_dense_layers(path, 6, sizes, name=name, nodes=nodes, initializers=initializers)
            for name, sizes in (("p_", [64, 64, 24]), ("q_", [32, 32, 8]))]
    nodes.append(helper.make_node("Concat", ends, ["out"], axis=1))
    return _save(path, nodes, initializers, 32)


def test_chains_of_wide_product_sums(tmp_path: Path) -> None:
    """Where the wide product sums of differently shaped graphs go: (chains, sums per chain, narrow sum at the
    start, where the first sum finds its row values, stages read beside the sums)."""
    def chains(model) -> list[tuple]:
        graph = load_row_graph(str(model(tmp_path / f"{model.__name__}.onnx")))
        rows = np.random.default_rng(3).normal(size=(5,) + tuple(graph.input_shape)).astype(np.float32)
        before = evaluate(graph, rows, np.float64)
        plan = make_plan(graph, Limits(), approximate_functions=False, tensor_cores=True)
        np.testing.assert_array_equal(evaluate(graph, rows, np.float64), before)
        found = [phase for phase in plan.fragments.values() if isinstance(phase, DenseChain)]
        return [(len(chain.wide), chain.narrow is not None, chain.place and chain.place[0], len(chain.sides)) for chain in found]

    assert chains(_four_sums) == [(3, True, None, 0)]                  # any number of sums
    assert chains(_sums_of_input_values) == [(2, False, "input", 0)]   # the first sum on values of the input row
    assert chains(_one_sum) == [(1, False, "input", 0)]                # one sum is a chain
    assert chains(_sums_with_a_skip) == [(2, False, "stage", 1)]       # an addition of row values between two sums
    assert chains(_sums_with_a_side) == [(2, False, "input", 1)]       # values computed ahead of the chain for it
    assert chains(_sums_of_computed_values) == [(2, False, None, 0)]   # row values that the kernel copies
    assert sorted(chains(_two_chains)) == [(2, True, None, 0), (2, True, None, 0)]
    # sums with more terms or elements than one instruction run in blocks: more terms at the start of a chain,
    # more elements at its end (with the tanh behind the sum), and the next chain reads the result where it is
    assert chains(_sum_of_many_terms) == [(1, True, None, 0)]
    assert chains(_sums_in_blocks) == [(1, False, "input", 0), (1, False, "stage", 0)]
    graph = load_row_graph(str(_sums_in_blocks(tmp_path / "blocks.onnx")))
    plan = make_plan(graph, Limits(), tensor_cores=True)
    first = next(phase for phase in plan.fragments.values() if isinstance(phase, DenseChain))
    assert first.in_blocks(graph) and [graph.stages[index].op for index in first.trailing] == ["tanh"]
    assert plan.slab_words <= 140 + 24 + 8   # the 140 values once, and the 24
    assert not any(isinstance(phase, DenseChain) and phase.in_blocks(graph)
                   for phase in make_plan(load_row_graph(str(tmp_path / "blocks.onnx")), Limits(wide_sums=False),
                                          tensor_cores=True).fragments.values())
    # the outputs of a chain go from its accumulators to the output rows: such a row keeps nothing in its slab
    options = Options(approximate_functions=True, tensor_cores=True)
    one = generate_kernel(str(_dense_layers(tmp_path / "wide.onnx", 128, [128])), "net", options)
    assert one.summary["threads_per_block"] == 256 and "wg_mma_128" in one.source
    assert one.summary["slab_bytes_per_row"] == 0 and "rows_out[to * kOutputs + column] = " in one.source
    # its weights pass in the largest slots with which the block keeps its rows, and a slab of many k-steps adds
    # the products with a low part first
    assert "weights in slabs of 8 k-steps" in one.source and "the products of the high parts" in one.source
    # a sum in blocks of terms and of elements holds 64 terms at a time and reads them again for every block of
    # elements: as volatile, or the compiler keeps the values of all terms on the stack
    twice = generate_kernel(str(tmp_path / "blocks.onnx"), "net", options)
    assert "const volatile float* const xa" in twice.source and re.search(r"uint32_t ah\d+\[8\]\[4\];", twice.source)
    assert "const volatile float*" not in one.source
    # A sum of few terms that feeds a sum in blocks of terms and of elements is a chain of its own: its values are
    # computed once and kept per row, in device memory where the shared memory would leave the block few rows.
    def hidden_256(path: Path) -> Path:
        return _dense_layers(path, 6, [256, 256, 56])

    assert chains(hidden_256) == [(1, False, "input", 0), (1, False, "stage", 0), (1, False, "stage", 0)]
    once = generate_kernel(str(tmp_path / "hidden_256.onnx"), "net", options)
    assert once.summary["slab"] == "device" and once.summary["rows_per_block"] == 256 and "wave" in once.source
    # with the slabs in shared memory only, that sum is the start of the chain it feeds (a layer of 256 values per
    # row less): the kernel has as many warpgroups as the shared memory holds
    fits = generate_kernel(str(tmp_path / "hidden_256.onnx"), "net", replace(options, slab="shared"))
    assert "wg_mma_128" in fits.source and fits.summary["threads_per_block"] == 128 and fits.summary["slab"] == "shared"
    # a layer of 512 by 512 keeps 2 kB per row: a warpgroup then owns one tile of 64 rows
    one_tile = generate_kernel(str(_dense_layers(tmp_path / "w512.onnx", 6, [512, 512, 56])), "net", replace(options, slab="shared"))
    assert "wg_mma_128" in one_tile.source and "constexpr int kTiles = 1;" in one_tile.source
    assert one_tile.summary["rows_per_block"] == 64 and one_tile.summary["threads_per_block"] == 128
    # where even that does not fit, the rows keep their values in device memory, and the block is full
    large = generate_kernel(str(_dense_layers(tmp_path / "w768.onnx", 6, [768, 768, 56])), "net", options)
    assert "wg_mma_128" in large.source and large.summary["slab"] == "device" and large.summary["rows_per_block"] == 256
    assert "float* __restrict__ scratch" in large.source and "kLaunchRows" in large.source
    shared = generate_kernel(str(tmp_path / "w512.onnx"), "net", replace(options, slab="shared"))
    assert shared.summary["slab"] == "shared" and shared.summary["rows_per_block"] == 64
    # with the slabs in shared memory only, that kernel is planned without the wide sums (sums of fmaf)
    assert "wg_mma_" not in generate_kernel(str(tmp_path / "w768.onnx"), "net", replace(options, slab="shared")).source


def _two_loops_tied_to_a_loop(path: Path) -> Path:
    """As ``_loop_tied_to_a_loop``, with the largest and the smallest value of every iteration: two loops over
    windows, one on each, run inside the loop on the tensor cores."""
    rng = np.random.default_rng(17)
    f32 = lambda *shape: rng.normal(size=shape).astype(np.float32) / 3
    initializers = {"w0": f32(6, 16), "w1": f32(16, 1224), "b1": f32(1224), "shape": np.array([0, 102, 12], dtype=np.int64),
                    "windows": np.arange(99, dtype=np.int64)[:, None] + np.arange(4, dtype=np.int64),
                    "axis_last": np.array([2], dtype=np.int64)}
    nodes = [helper.make_node("MatMul", ["rows", "w0"], ["d0"]), helper.make_node("Tanh", ["d0"], ["x"]),
             helper.make_node("MatMul", ["x", "w1"], ["d1"]), helper.make_node("Add", ["d1", "b1"], ["a1"]),
             helper.make_node("Reshape", ["a1", "shape"], ["parts"]),
             helper.make_node("ReduceMax", ["parts", "axis_last"], ["largest"], keepdims=0),
             helper.make_node("ReduceMin", ["parts", "axis_last"], ["smallest"], keepdims=0),
             helper.make_node("Gather", ["largest", "windows"], ["four_largest"], axis=1),
             helper.make_node("ReduceMin", ["four_largest", "axis_last"], ["low"], keepdims=0),
             helper.make_node("Gather", ["smallest", "windows"], ["four_smallest"], axis=1),
             helper.make_node("ReduceMax", ["four_smallest", "axis_last"], ["high"], keepdims=0),
             helper.make_node("Concat", ["low", "high"], ["out"], axis=1)]
    return _save(path, nodes, initializers, 198)


def test_product_sums_of_the_next_chunk_start_before_the_tied_loop(tmp_path: Path) -> None:
    options = Options(approximate_functions=True, tensor_cores=True, warpgroups=1, loop_warpgroups=1)
    tied = generate_kernel(str(_loop_tied_to_a_loop(tmp_path / "tied.onnx")), "net", options).parts["net_2.cu"]
    start, loop, wait = (tied.index(text) for text in (
        "// start the product sums of the next chunk", "// the tied loops of this chunk", "wg_wait<0>();"))
    assert start < loop < wait and "for (int q = -1;" in tied
    assert "__syncthreads" not in tied[start:wait]   # the warps of a block do not wait for each other in the loop
    alone = generate_kernel(str(_iterations_by_thread(tmp_path / "loop.onnx")), "net", options).parts["net_2.cu"]
    assert "for (int q = 0;" in alone and "if (held) " not in alone and "&& held" in alone
    two = generate_kernel(str(_two_loops_tied_to_a_loop(tmp_path / "two.onnx")), "net", options).parts["net_2.cu"]
    assert two.count("// ---- iterations of the loop over 99") == 2
    # the records of both tied loops are copied ahead of the loop, one behind the other in the stage buffer
    assert len(re.findall(r"staged\[(?:\d+ \+ )?i\] = load_constants", two)) == 2 and "staged[i] = " in two


def test_every_loop_that_reads_in_step_runs_inside_its_producer(tmp_path: Path) -> None:
    """Two loops over 600 values, each followed by a loop over windows of four of its values: both window loops
    run inside their producers (the positions of the groups change when the first one moves)."""
    rng = np.random.default_rng(16)
    initializers = {"windows": np.arange(597, dtype=np.int64)[:, None] + np.arange(4, dtype=np.int64),
                    "axis_last": np.array([2], dtype=np.int64)}
    nodes, ranges = [], []
    for name in ("a", "b"):
        initializers[f"w_{name}"] = rng.normal(size=(6, 600)).astype(np.float32) / 3
        nodes += [helper.make_node("MatMul", ["rows", f"w_{name}"], [f"d_{name}"]),
                  helper.make_node("Tanh", [f"d_{name}"], [f"t_{name}"]),
                  helper.make_node("Gather", [f"t_{name}", "windows"], [f"four_{name}"], axis=1),
                  helper.make_node("ReduceMin", [f"four_{name}", "axis_last"], [f"low_{name}"], keepdims=0),
                  helper.make_node("ReduceMax", [f"four_{name}", "axis_last"], [f"high_{name}"], keepdims=0),
                  helper.make_node("Sub", [f"high_{name}", f"low_{name}"], [f"range_{name}"])]
        ranges.append(f"range_{name}")
    nodes.append(helper.make_node("Concat", ranges, ["out"], axis=1))
    path = _save(tmp_path / "two.onnx", nodes, initializers, 1194)
    graph = load_row_graph(str(path))
    before = evaluate(graph, _rows(5), np.float64)
    plan = make_plan(graph, Limits())
    assert sorted(len(group.streamed) for group in plan.groups) == [1, 1] and len(plan.ring) == 2
    assert plan.slab_words <= 8   # the rings of four values, not the 600 values of each loop
    np.testing.assert_array_equal(evaluate(graph, _rows(5), np.float64), before)


def test_broadcasting_by_attributes_stops_and_empty_parts_pass(tmp_path: Path) -> None:
    # opset 6: Add with broadcast=1, axis=1 adds [10, 20] along the second axis, not the last
    graph = helper.make_graph(
        [helper.make_node("Add", ["rows", "pair"], ["out"], broadcast=1, axis=1)], "rows",
        [helper.make_tensor_value_info("rows", TensorProto.FLOAT, ["n", 2, 2])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["n", 2, 2])],
        initializer=[numpy_helper.from_array(np.array([10.0, 20.0], dtype=np.float32), "pair")])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 6)], ir_version=3), tmp_path / "old.onnx")
    with pytest.raises(UnsupportedGraph, match="opset 6"):
        load_row_graph(str(tmp_path / "old.onnx"))
    # a Split into parts of 0 and 6 values, an operator on the empty part, and both parts joined again
    path = _save(tmp_path / "empty.onnx",
                 [helper.make_node("Split", ["rows", "sizes"], ["none", "all"], axis=1),
                  helper.make_node("Tanh", ["none"], ["still_none"]), helper.make_node("Neg", ["all"], ["negative"]),
                  helper.make_node("Concat", ["still_none", "negative"], ["out"], axis=1)],
                 {"sizes": np.array([0, ROW_SIZE], dtype=np.int64)}, ROW_SIZE)
    np.testing.assert_array_equal(evaluate(load_row_graph(str(path)), _rows()), -_rows())
    assert "evaluate_rows" in generate_kernel(str(path), "empty").source
    # the largest and the smallest value of no values are the ends of FP32
    path = _save(tmp_path / "ends.onnx",
                 [helper.make_node("Split", ["rows", "sizes"], ["none", "all"], axis=1),
                  helper.make_node("ReduceMax", ["none", "axis"], ["largest"]), helper.make_node("ReduceMin", ["none", "axis"], ["smallest"]),
                  helper.make_node("ReduceSum", ["none", "axis"], ["total"]),
                  helper.make_node("Concat", ["all", "largest", "smallest", "total"], ["out"], axis=1)],
                 {"sizes": np.array([0, ROW_SIZE], dtype=np.int64), "axis": np.array([1], dtype=np.int64)}, ROW_SIZE + 3)
    values = evaluate(load_row_graph(str(path)), _rows())
    np.testing.assert_array_equal(values[:, ROW_SIZE:], np.tile(np.float32([-np.inf, np.inf, 0.0]), (len(values), 1)))


def test_sigmoid_of_a_large_negative_value_is_a_number(tmp_path: Path) -> None:
    """exp(-x) overflows for x = -90 in FP32; the Sigmoid of it is a (small) number, and its logarithm -90.
    (The reference is the function itself: the Sigmoid of ONNX Runtime on the CPU returns 0 for -30 and below.)"""
    nodes = [helper.make_node("Sigmoid", ["rows"], ["s"]), helper.make_node("Log", ["s"], ["out"])]
    path = _save(tmp_path / "sigmoid.onnx", nodes, {}, ROW_SIZE)
    rows = np.array([[-90.0, -30.0, -1.0, 0.0, 2.0, 40.0]], dtype=np.float32)
    values = evaluate(load_row_graph(str(path)), rows)
    np.testing.assert_allclose(values, -np.logaddexp(0.0, -rows.astype(np.float64)), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(values[:, 2:], _reference(path, rows)[:, 2:], rtol=1e-5, atol=1e-7)
    kernel = generate_kernel(str(path), "net", Options())
    assert "expf(v" in kernel.source and "expf(-v" in kernel.source   # both forms, chosen by the sign


def test_pow_of_an_integer_constant_and_evaluation_of_empty_shapes(tmp_path: Path) -> None:
    """A Pow of constants has the type of its base, as in ONNX (2 ** 0.5 of an INT32 is 1); the reference
    evaluation takes no rows, and a row output without elements is an array."""
    nodes = [helper.make_node("Pow", ["two", "half"], ["p"]), helper.make_node("Cast", ["p"], ["f"], to=TensorProto.FLOAT),
             helper.make_node("Add", ["rows", "f"], ["out"])]
    initializers = {"two": np.array([2], dtype=np.int32), "half": np.array([0.5], dtype=np.float32)}
    path = _save(tmp_path / "pow.onnx", nodes, initializers, ROW_SIZE)
    rows = _rows()
    graph = load_row_graph(str(path))
    np.testing.assert_allclose(evaluate(graph, rows), _reference(path, rows), rtol=1e-6)
    np.testing.assert_allclose(evaluate(graph, rows), rows + 1.0, rtol=1e-6)
    assert evaluate(graph, rows[:0]).shape == (0, ROW_SIZE)
    nodes = [helper.make_node("Gather", ["rows", "none"], ["out"], axis=1)]
    path = _save(tmp_path / "none.onnx", nodes, {"none": np.array([], dtype=np.int64)}, 0)
    assert evaluate(load_row_graph(str(path)), rows).shape == (len(rows), 0)


def test_reduction_of_an_empty_part_keeps_its_shape(tmp_path: Path) -> None:
    """A reduction over one axis of a part without elements along another one has no elements; a chain of one
    weight slab on one tile fills one slot only."""
    nodes = [helper.make_node("Reshape", ["rows", "shape"], ["cube"]),
             helper.make_node("Slice", ["cube", "start", "start", "one"], ["none"]),   # [n, 0, 3]
             helper.make_node("ReduceSum", ["none", "last"], ["sums"], keepdims=0),    # [n, 0]
             helper.make_node("Concat", ["sums", "rows"], ["out"], axis=1)]
    initializers = {"shape": np.array([-1, 2, 3], dtype=np.int64), "start": np.array([0], dtype=np.int64),
                    "one": np.array([1], dtype=np.int64), "last": np.array([2], dtype=np.int64)}
    path = _save(tmp_path / "empty.onnx", nodes, initializers, ROW_SIZE)
    rows = _rows()
    np.testing.assert_array_equal(evaluate(load_row_graph(str(path)), rows), _reference(path, rows))
    options = Options(approximate_functions=True, tensor_cores=True, tiles=1, slab="shared")
    one = generate_kernel(str(_dense_layers(tmp_path / "small.onnx", 16, [16])), "net", options)
    assert "wg_mma_16" in one.source and not re.search(r"copy_bulk\(.*, full\d+ \+ 1\);", one.source)


def test_weights_near_the_largest_split_into_finite_parts(tmp_path: Path) -> None:
    from fortonnx_kernelgen.tensor import tf32_split
    largest = np.finfo(np.float32).max
    values = np.float32([largest, -largest, np.nextafter(largest, np.float32(0)), 1.0, -3.5e-12, 0.0])
    high, low = tf32_split(values)
    assert np.isfinite(high).all() and np.isfinite(low).all()
    assert not (high.view(np.uint32) & np.uint32(0x1FFF)).any()   # TF32: the low 13 bits of the mantissa are zero
    np.testing.assert_allclose(high.astype(np.float64) + low.astype(np.float64), values.astype(np.float64), rtol=2.0 ** -20)


def _copies_through_layers(path: Path) -> Path:
    """Three vectors of 80 values through the same two layers, stored one after the other ([3, 80]): the first is
    the value, the others are scaled by a function of it (as the tangents of a network are)."""
    rng = np.random.default_rng(21)
    f32 = lambda *shape: (rng.normal(size=shape) / 4).astype(np.float32)
    initializers = {"w0": f32(ROW_SIZE, 240), "to_copies": np.array([0, 3, 80], dtype=np.int64), "w1": f32(80, 80),
                    "w2": f32(80, 24), "one": np.array(1.0, dtype=np.float32), "flat": np.array([0, 72], dtype=np.int64),
                    "first": np.array([0], dtype=np.int64), "second": np.array([1], dtype=np.int64),
                    "end": np.array([3], dtype=np.int64), "axis": np.array([1], dtype=np.int64)}
    nodes = [helper.make_node("MatMul", ["rows", "w0"], ["x"]), helper.make_node("Reshape", ["x", "to_copies"], ["copies"]),
             helper.make_node("MatMul", ["copies", "w1"], ["d1"]),
             helper.make_node("Slice", ["d1", "first", "second", "axis"], ["value"]),
             helper.make_node("Slice", ["d1", "second", "end", "axis"], ["others"]),
             helper.make_node("Tanh", ["value"], ["h"]), helper.make_node("Mul", ["h", "h"], ["hh"]),
             helper.make_node("Sub", ["one", "hh"], ["slope"]), helper.make_node("Mul", ["others", "slope"], ["g"]),
             helper.make_node("Concat", ["h", "g"], ["next"], axis=1), helper.make_node("MatMul", ["next", "w2"], ["d2"]),
             helper.make_node("Reshape", ["d2", "flat"], ["out"])]
    return _save(path, nodes, initializers, 72)


def test_copies_of_an_index_space_share_its_loop(tmp_path: Path) -> None:
    path = _copies_through_layers(tmp_path / "copies.onnx")
    rows = _rows()
    graph = load_row_graph(str(path))
    names = {stage.name: stage.index for stage in graph.stages}
    make_plan(graph, Limits(wide_sums=False))
    np.testing.assert_allclose(evaluate(graph, rows, np.float64), _reference(path, rows), rtol=2e-5, atol=1e-6)
    plan = make_plan(load_row_graph(str(path)), Limits())
    loops = [part for group in plan.groups for part in [group] + [stream.group for stream in group.streamed]]
    group = next(group for group in loops if names["d1"] in group.stages)
    # the sum over the three copies, the value's tanh and the scaled copies run in one loop over the 80 values
    assert group.members == 80 and {names["h"], names["g"]} <= set(group.stages)
    assert names["d1"] not in plan.slab_offset


def _maximum_of_many(path: Path) -> Path:
    """The largest tanh of 1024 values: a row that keeps 4 kB and has no constants."""
    nodes = [helper.make_node("Tanh", ["rows"], ["t"]), helper.make_node("ReduceMax", ["t", "axis"], ["out"], keepdims=1)]
    return _save(path, nodes, {"axis": np.array([1], dtype=np.int64)}, 1, row_size=1024)


def test_rows_with_all_their_words_in_device_memory(tmp_path: Path) -> None:
    """A kernel without records, tensor-core images or words of its rows in shared memory still declares the
    shared memory that the rows' pointers are written with."""
    kernel = generate_kernel(str(_maximum_of_many(tmp_path / "maximum.onnx")), "net", Options())
    assert kernel.summary["slab"] == "device" and "constexpr int kSlabWords = 0;" in kernel.source
    assert "extern __shared__" in kernel.source and "const Row s{" in kernel.source


def test_parts_of_the_output_row(tmp_path: Path) -> None:
    """Outputs that need the same product sums are one part; a part is a graph of its own, with the elements
    that its outputs need, and computes the outputs of the whole graph at its positions."""
    path = _copies_through_layers(tmp_path / "copies.onnx")
    rows = _rows()
    graph = load_row_graph(str(path))
    rewrite_graph(graph, Limits(), False, True)
    whole = evaluate(graph, rows, np.float64)
    parts = output_parts(graph)
    # the value's outputs with those of the first copy (its elements are needed there anyway), the second copy's
    assert [part.tolist() for part in parts] == [list(range(48)), list(range(48, 72))]
    for part in parts:
        piece = sub_graph(graph, part)
        fold_constant_terms(piece)
        assert piece.output_slots.tolist() == part.tolist() and piece.row_width == 72 and piece.output_size == len(part)
        np.testing.assert_array_equal(evaluate(piece, rows, np.float64), whole[:, part])
        # of the stage of three copies, the value's elements and those of the part's copy; the last layer of each
        sizes = sorted(stage.size for stage in piece.stages if stage.kind == "dot" and stage.terms == 80)
        assert sizes == [24] * (len(part) // 24) + [160]
    # with the product sums of the loops on the tensor cores either way, the graph stays one kernel
    options = Options(approximate_functions=True, tensor_cores=True)
    assert "output_parts" not in generate_kernel(str(path), "net", options).summary
    kernel = generate_kernel(str(path), "net", replace(options, output_parts="on"))
    assert kernel.summary["output_parts"] == 2 and "cudaStreamWaitEvent" in kernel.source
    first, second = (kernel.parts[f"net_{number}.cu"] for number in (1, 2))
    assert "out[47]" in first or "+ 47" in first or "kOutputs = 72" in first
    assert "constexpr int kOutputs = 72;" in first and "constexpr int kOutputs = 72;" in second


def test_slab_keeps_what_live_code_reads(tmp_path: Path) -> None:
    """A loop of 1024 values that are all outputs; the sums of 512 pairs of them, of which only the first is an
    output: the row keeps the two values of that pair, not the 1024."""
    rng = np.random.default_rng(20)
    pairs = rng.permutation(1024).reshape(512, 2).astype(np.int64)
    path = _save(tmp_path / "pairs.onnx",
                 [helper.make_node("MatMul", ["rows", "w"], ["h"]), helper.make_node("Gather", ["h", "pairs"], ["two"], axis=1),
                  helper.make_node("ReduceSum", ["two", "axis"], ["sums"], keepdims=0),
                  helper.make_node("Gather", ["sums", "first"], ["one"], axis=1),
                  helper.make_node("Concat", ["h", "one"], ["out"], axis=1)],
                 {"w": (rng.normal(size=(ROW_SIZE, 1024)) / 3).astype(np.float32), "pairs": pairs,
                  "axis": np.array([2], dtype=np.int64), "first": np.array([0], dtype=np.int64)}, 1025)
    graph = load_row_graph(str(path))
    plan = make_plan(graph, Limits())
    assert [sorted(elements.tolist()) for elements in plan.kept.values()] == [sorted(pairs[0].tolist())]
    kernel = generate_kernel(str(path), "pairs")
    assert kernel.summary["slab_bytes_per_row"] <= 48 and kernel.summary["threads_per_block"] == 256


def test_two_kernels_in_a_row_and_their_file_names(tmp_path: Path) -> None:
    path = _iterations_by_thread(tmp_path / "loop.onnx")
    options = Options(approximate_functions=True, tensor_cores=True, warpgroups=1, loop_warpgroups=1)
    kernel = generate_kernel(str(path), "net", options)
    assert sorted(kernel.parts) == ["net_1.cu", "net_1.h", "net_2.cu", "net_2.h"]
    assert kernel.summary["values_between"] == 16 and '#include "net_2.cu"' in kernel.source
    assert len(generate_kernel(str(path), "net", Options(approximate_functions=True, tensor_cores=True, warpgroups=1,
                                                         split_kernels=False)).parts) == 0
    other = generate_kernel(str(_banded_model(tmp_path / "banded.onnx")), "net_1")
    with pytest.raises(ValueError, match="net_1.cu"):
        write_operator_library(str(tmp_path), [kernel, other])
