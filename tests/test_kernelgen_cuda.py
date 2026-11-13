"""Kernel generator on a GPU: the generated kernel as an ONNX Runtime custom operator against the original graph.

Needs nvcc, a C++ compiler, the ONNX Runtime C++ headers (ONNXRUNTIME_INCLUDE: the directory of
onnxruntime_cxx_api.h) and ONNX Runtime with the CUDA provider; skipped otherwise."""

from __future__ import annotations

import os
import shutil
import re
import subprocess
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from fortonnx_kernelgen import Limits, Options, generate_kernel, write_operator_library
from onnx import helper

from test_kernelgen import (_banded_model, _copies_through_layers, _four_sums, _maximum_of_many, _iterations_by_thread, _loop_tied_to_a_loop, _one_sum, _rows,
                            _save, _sum_of_many_terms, _sums_in_blocks, _sums_of_computed_values, _sums_of_input_values,
                            _sums_with_a_side, _sums_with_a_skip, _two_chains, _two_loops_tied_to_a_loop)

INCLUDE = os.environ.get("ONNXRUNTIME_INCLUDE", "")
ARCH = os.environ.get("FORTONNX_KERNELGEN_ARCH", "sm_90")
pytestmark = pytest.mark.skipif(
    shutil.which("nvcc") is None or shutil.which("g++") is None
    or not os.path.isfile(os.path.join(INCLUDE, "onnxruntime_cxx_api.h"))
    or "CUDAExecutionProvider" not in ort.get_available_providers(),
    reason="needs nvcc, g++, ONNXRUNTIME_INCLUDE and the CUDA provider of ONNX Runtime")


def _session(path: str, library: str | None = None) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    if library:
        options.register_custom_ops_library(library)
    return ort.InferenceSession(path, options, providers=[("CUDAExecutionProvider", {"use_tf32": "0"})])


@pytest.mark.parametrize("stage_bytes", [16 * 1024, 0])
def test_custom_operator_matches_the_graph(tmp_path: Path, stage_bytes: int) -> None:
    original = _banded_model(tmp_path / "banded.onnx")
    options = Options(stage_bytes=stage_bytes, limits=Limits(register_array=8))
    kernel = generate_kernel(str(original), "banded", options)
    out = tmp_path / "out"
    kernel.write(str(out))
    write_operator_library(str(out), [kernel])
    subprocess.run(["sh", "build.sh", INCLUDE, ARCH, "libkernels.so"], cwd=out, check=True, capture_output=True, text=True)
    exported = subprocess.run(["nm", "-D", "--defined-only", "libkernels.so"], cwd=out, check=True, capture_output=True, text=True)
    assert "RegisterCustomOps" in exported.stdout and "banded_create" not in exported.stdout
    library = str(out / "libkernels.so")

    rows = _rows(1000)   # more than one block, and a last block that is not full
    expected = _session(str(original)).run(None, {"rows": rows})[0]
    session = _session(str(out / "banded.onnx"), library)
    np.testing.assert_allclose(session.run(None, {"rows": rows})[0], expected, rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(session.run(None, {"rows": rows[:1]})[0], expected[:1], rtol=2e-5, atol=1e-6)

    # the library refuses a graph whose constants were written for another kernel
    other = generate_kernel(str(original), "banded", Options(stage_bytes=stage_bytes, threads_per_block=64, limits=options.limits))
    other.write(str(tmp_path / "other"))
    with pytest.raises(Exception, match="another kernel"):
        _session(str(tmp_path / "other" / "banded.onnx"), library)


def _compute_capability() -> str:
    if shutil.which("nvidia-smi") is None:
        return ""
    query = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True)
    return query.stdout.split("\n")[0].strip()


def _products_per_iteration(path: Path) -> Path:
    """Ten iterations of two product sums, the second on the first without a stage between them."""
    rng = np.random.default_rng(11)
    initializers = {"w1": rng.normal(size=(6, 80)).astype(np.float32) / 2, "w2": rng.normal(size=(8, 8)).astype(np.float32) / 2,
                    "shape": np.array([0, 10, 8], dtype=np.int64), "axis_last": np.array([2], dtype=np.int64)}
    nodes = [helper.make_node("MatMul", ["rows", "w1"], ["first"]), helper.make_node("Reshape", ["first", "shape"], ["parts"]),
             helper.make_node("MatMul", ["parts", "w2"], ["second"]),
             helper.make_node("ReduceSum", ["second", "axis_last"], ["out"], keepdims=0)]
    return _save(path, nodes, initializers, 10)


def _logarithm_between_products(path: Path) -> Path:
    """Ten iterations of two product sums with a logarithm between them; twelve values per iteration, so that the
    tensor cores see four padded columns."""
    rng = np.random.default_rng(13)
    initializers = {"w1": np.abs(rng.normal(size=(6, 120))).astype(np.float32) + 0.1, "w2": rng.normal(size=(12, 8)).astype(np.float32) / 2,
                    "one": np.array(1.0, dtype=np.float32),
                    "shape": np.array([0, 10, 12], dtype=np.int64), "axis_last": np.array([2], dtype=np.int64)}
    nodes = [helper.make_node("Abs", ["rows"], ["size"]), helper.make_node("Add", ["size", "one"], ["positive"]),
             helper.make_node("MatMul", ["positive", "w1"], ["first"]), helper.make_node("Reshape", ["first", "shape"], ["parts"]),
             helper.make_node("Log", ["parts"], ["logs"]), helper.make_node("MatMul", ["logs", "w2"], ["second"]),
             helper.make_node("ReduceSum", ["second", "axis_last"], ["out"], keepdims=0)]
    return _save(path, nodes, initializers, 10)


def _chain_output_and_another_sum(path: Path) -> Path:
    """The last sum of a chain of wide product sums is an output; a sum over another wide stage follows it."""
    rng = np.random.default_rng(15)
    f32 = lambda *shape: rng.normal(size=shape).astype(np.float32) / 4
    initializers = {"w1": f32(6, 80), "w2": f32(80, 8), "w3": f32(6, 80), "axis_last": np.array([1], dtype=np.int64)}
    nodes = [helper.make_node("MatMul", ["rows", "w1"], ["h"]), helper.make_node("Tanh", ["h"], ["t"]),
             helper.make_node("MatMul", ["t", "w2"], ["chain"]), helper.make_node("MatMul", ["rows", "w3"], ["other"]),
             helper.make_node("Tanh", ["other"], ["bounded"]),
             helper.make_node("ReduceSum", ["bounded", "axis_last"], ["total"], keepdims=1),
             helper.make_node("Concat", ["chain", "total"], ["out"], axis=1)]
    return _save(path, nodes, initializers, 9)


def _chain_behind_a_loop(path: Path) -> Path:
    """A chain of wide product sums whose row values come from a sum that is accumulated in a loop."""
    rng = np.random.default_rng(12)
    f32 = lambda *shape: rng.normal(size=shape).astype(np.float32) / 4
    initializers = {"w1": f32(80, 6), "w2": f32(6, 80), "w3": f32(80, 8)}
    nodes = [helper.make_node("Tanh", ["rows"], ["t0"]), helper.make_node("MatMul", ["t0", "w1"], ["narrow"]),
             helper.make_node("MatMul", ["narrow", "w2"], ["wide"]), helper.make_node("Tanh", ["wide"], ["t1"]),
             helper.make_node("MatMul", ["t1", "w3"], ["last"]), helper.make_node("Relu", ["last"], ["out"])]
    return _save(path, nodes, initializers, 8, row_size=80)


@pytest.mark.skipif(_compute_capability() != "9.0", reason="the tensor-core kernels need a Hopper GPU (sm_90a)")
@pytest.mark.parametrize("model, row_size, kernels", [
    (_products_per_iteration, 6, 1), (_logarithm_between_products, 6, 1), (_chain_behind_a_loop, 80, 1),
    (_chain_output_and_another_sum, 6, 1),
    (_iterations_by_thread, 6, 2), (_iterations_by_thread, 6, 1), (_loop_tied_to_a_loop, 6, 2), (_loop_tied_to_a_loop, 6, 1),
    (_two_loops_tied_to_a_loop, 6, 2),
    (_four_sums, 6, 1), (_sums_of_input_values, 40, 1), (_one_sum, 64, 1), (_sums_with_a_skip, 6, 1),
    (_sums_of_computed_values, 24, 1), (_two_chains, 6, 1), (_sums_with_a_side, 32, 1),
    (_sum_of_many_terms, 6, 1), (_sums_in_blocks, 150, 1)])
def test_tensor_core_layouts_match_the_graph(tmp_path: Path, model, row_size: int, kernels: int) -> None:
    original = model(tmp_path / "model.onnx")
    options = Options(approximate_functions=True, tensor_cores=True, warpgroups=1, loop_warpgroups=1, split_kernels=kernels == 2)
    kernel = generate_kernel(str(original), "layout", options)
    assert "wg_mma_" in "".join([kernel.source] + list(kernel.parts.values()))
    assert len(kernel.summary.get("kernels", [1])) == kernels
    out = tmp_path / "out"
    kernel.write(str(out))
    write_operator_library(str(out), [kernel])
    subprocess.run(["sh", "build.sh", INCLUDE, "sm_90a", "libkernels.so"], cwd=out, check=True, capture_output=True, text=True)
    rows = np.random.default_rng(5).normal(size=(300, row_size)).astype(np.float32)
    expected = _session(str(original)).run(None, {"rows": rows})[0]
    session = _session(str(out / "layout.onnx"), str(out / "libkernels.so"))
    np.testing.assert_allclose(session.run(None, {"rows": rows})[0], expected, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(_compute_capability() != "9.0", reason="the tensor-core kernels need a Hopper GPU (sm_90a)")
@pytest.mark.parametrize("model, row_size", [(_four_sums, 6), (_loop_tied_to_a_loop, 6), (_products_per_iteration, 6)])
def test_one_tile_per_warpgroup_matches_the_graph(tmp_path: Path, model, row_size: int) -> None:
    """64 rows per warpgroup (the layout of kernels whose rows keep much in shared memory): a chain, a loop
    with its iterations by thread behind the kernel of its row values, and a loop with product sums per iteration."""
    original = model(tmp_path / "model.onnx")
    options = Options(approximate_functions=True, tensor_cores=True, warpgroups=1, loop_warpgroups=1, tiles=1, loop_tiles=1)
    kernel = generate_kernel(str(original), "tile", options)
    texts = list(kernel.parts.values()) or [kernel.source]
    assert all("constexpr int kTiles = 1;" in text for text in texts if "evaluate_rows" in text and "wg_mma_" in text)
    assert any("constexpr int kTiles = 1;" in text for text in texts)
    out = tmp_path / "out"
    kernel.write(str(out))
    write_operator_library(str(out), [kernel])
    subprocess.run(["sh", "build.sh", INCLUDE, "sm_90a", "libkernels.so"], cwd=out, check=True, capture_output=True, text=True)
    rows = np.random.default_rng(6).normal(size=(333, row_size)).astype(np.float32)   # blocks of 64 rows, the last one partial
    expected = _session(str(original)).run(None, {"rows": rows})[0]
    session = _session(str(out / "tile.onnx"), str(out / "libkernels.so"))
    np.testing.assert_allclose(session.run(None, {"rows": rows})[0], expected, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(_compute_capability() != "9.0", reason="the tensor-core kernels need a Hopper GPU (sm_90a)")
@pytest.mark.parametrize("model, row_size, tensor_cores, slab", [
    (_four_sums, 6, True, "device"), (_banded_model, 6, False, "device"), (_maximum_of_many, 1024, False, "device"),
    (_products_per_iteration, 6, True, "shared")])   # the values of a loop on the tensor cores stay in shared memory
def test_slabs_in_device_memory_match_the_graph(tmp_path: Path, model, row_size: int, tensor_cores: bool, slab: str) -> None:
    """The rows' values between loops in device memory (kernels whose rows keep more than the shared memory holds):
    a chain, loops of one row per thread, and a loop with product sums per iteration; more rows than one launch
    holds are several launches. (far_words 1: also the small stages, which a kernel keeps in shared memory.)"""
    original = model(tmp_path / "model.onnx")
    options = Options(approximate_functions=True, tensor_cores=tensor_cores, warpgroups=1, slab="device",
                      limits=Limits(register_array=8, far_words=1))
    kernel = generate_kernel(str(original), "device", options)
    assert kernel.summary["slab"] == slab and ("kLaunchRows" in kernel.source) == (slab == "device")
    out = tmp_path / "out"
    kernel.write(str(out))
    if slab == "device":   # launches of two blocks
        source = (out / "device.cu").read_text()
        launch = re.search(r"constexpr long long kLaunchRows = (\d+)LL \* kRowsPerBlock;", source)
        (out / "device.cu").write_text(source.replace(launch.group(0), "constexpr long long kLaunchRows = 2LL * kRowsPerBlock;"))
    write_operator_library(str(out), [kernel])
    subprocess.run(["sh", "build.sh", INCLUDE, "sm_90a", "libkernels.so"], cwd=out, check=True, capture_output=True, text=True)
    rows = np.random.default_rng(7).normal(size=(1500, row_size)).astype(np.float32)   # several launches, the last partial
    expected = _session(str(original)).run(None, {"rows": rows})[0]
    session = _session(str(out / "device.onnx"), str(out / "libkernels.so"))
    for count in (40, 1500, 700):   # the buffer of the rows' slabs grows, then is used again
        np.testing.assert_allclose(session.run(None, {"rows": rows[:count]})[0], expected[:count], rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(_compute_capability() != "9.0", reason="the tensor-core kernels need a Hopper GPU (sm_90a)")
@pytest.mark.parametrize("slab", ["shared", "device"])
def test_kernels_side_by_side_match_the_graph(tmp_path: Path, slab: str) -> None:
    """One kernel per part of the output row (a value and each of the vectors scaled by it), on streams of the
    kernel object at the same time; with the larger stages of the rows in device memory, the others in shared
    memory."""
    original = _copies_through_layers(tmp_path / "model.onnx")
    options = Options(approximate_functions=True, tensor_cores=True, output_parts="on", slab=slab,
                      limits=Limits(far_words=24))
    kernel = generate_kernel(str(original), "parts", options)
    assert kernel.summary["output_parts"] == 2 and sorted(kernel.parts) == ["parts_1.cu", "parts_1.h", "parts_2.cu", "parts_2.h"]
    assert [part["slab"] for part in kernel.summary["kernels"]] == [slab, slab]
    out = tmp_path / "out"
    kernel.write(str(out))
    write_operator_library(str(out), [kernel])
    subprocess.run(["sh", "build.sh", INCLUDE, "sm_90a", "libkernels.so"], cwd=out, check=True, capture_output=True, text=True)
    rows = np.random.default_rng(11).normal(size=(700, 6)).astype(np.float32)
    expected = _session(str(original)).run(None, {"rows": rows})[0]
    session = _session(str(out / "parts.onnx"), str(out / "libkernels.so"))
    for count in (700, 3, 200):
        np.testing.assert_allclose(session.run(None, {"rows": rows[:count]})[0], expected[:count], rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(_compute_capability() != "9.0", reason="the tensor-core kernels need a Hopper GPU (sm_90a)")
@pytest.mark.parametrize("warpgroups", [1, 2])
def test_tensor_core_kernel_matches_the_graph(tmp_path: Path, warpgroups: int) -> None:
    original = _banded_model(tmp_path / "banded.onnx")
    options = Options(approximate_functions=True, tensor_cores=True, warpgroups=warpgroups, limits=Limits(register_array=8))
    kernel = generate_kernel(str(original), "banded", options)
    assert "wg_mma_shared_" in kernel.source and "switch (" in kernel.source
    out = tmp_path / "out"
    kernel.write(str(out))
    write_operator_library(str(out), [kernel])
    subprocess.run(["sh", "build.sh", INCLUDE, "sm_90a", "libkernels.so"], cwd=out, check=True, capture_output=True, text=True)
    rows = _rows(1000)   # more than one block, and a last block that is not full
    expected = _session(str(original)).run(None, {"rows": rows})[0]
    session = _session(str(out / "banded.onnx"), str(out / "libkernels.so"))
    np.testing.assert_allclose(session.run(None, {"rows": rows})[0], expected, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(session.run(None, {"rows": rows[:1]})[0], expected[:1], rtol=1e-4, atol=1e-5)


def test_direct_entry_runs_the_kernel_of_the_graph_file(tmp_path: Path) -> None:
    """A program without ONNX Runtime gets, through the library's direct entry, the outputs of the custom operator."""
    original = _banded_model(tmp_path / "banded.onnx")
    options = Options(limits=Limits(register_array=8))
    kernel = generate_kernel(str(original), "banded", options)
    out = tmp_path / "out"
    kernel.write(str(out))
    write_operator_library(str(out), [kernel], direct_prefix="rows_kernel")
    subprocess.run(["sh", "build.sh", INCLUDE, ARCH, "libkernels.so"], cwd=out, check=True, capture_output=True, text=True)
    exported = subprocess.run(["nm", "-D", "--defined-only", "libkernels.so"], cwd=out, check=True, capture_output=True, text=True)
    assert sorted(line.split()[-1] for line in exported.stdout.splitlines()) == [
        "RegisterCustomOps", "rows_kernel_close", "rows_kernel_forward", "rows_kernel_open", "rows_kernel_shape"]
    driver = tmp_path / "direct_driver"
    subprocess.run(["nvcc", "-O2", "-std=c++17", str(Path(__file__).with_name("direct_driver.cpp")), "-o", str(driver), "-ldl"],
                   check=True, capture_output=True, text=True)
    rows = _rows(1000)
    rows.tofile(tmp_path / "rows.bin")
    library = str(out / "libkernels.so")
    run = subprocess.run([str(driver), library, "rows_kernel", str(out / "banded.onnx"), str(tmp_path / "rows.bin"),
                          str(tmp_path / "direct.bin")], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    direct = np.fromfile(tmp_path / "direct.bin", dtype=np.float32).reshape(1000, -1)
    through_onnxruntime = _session(str(out / "banded.onnx"), library).run(None, {"rows": rows})[0]
    np.testing.assert_array_equal(direct, through_onnxruntime)   # the same kernel on the same constants
    np.testing.assert_allclose(direct, _session(str(original)).run(None, {"rows": rows})[0], rtol=2e-5, atol=1e-6)

    # the library refuses a file that is not a graph of one of its kernels, with the reason
    for graph, reason in ((original, "not one node of an operator"), (tmp_path / "missing.onnx", "cannot be opened")):
        run = subprocess.run([str(driver), library, "rows_kernel", str(graph), str(tmp_path / "rows.bin"), str(tmp_path / "no.bin")],
                             capture_output=True, text=True)
        assert run.returncode == 3 and f"{graph}: " in run.stdout and reason in run.stdout, run.stdout
    other = generate_kernel(str(original), "banded", Options(threads_per_block=64, limits=options.limits))
    other.write(str(tmp_path / "other"))
    run = subprocess.run([str(driver), library, "rows_kernel", str(tmp_path / "other" / "banded.onnx"), str(tmp_path / "rows.bin"),
                          str(tmp_path / "no.bin")], capture_output=True, text=True)
    assert run.returncode == 3 and "another kernel" in run.stdout, run.stdout
