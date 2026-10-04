"""Validator-owned FP64/bool CSV decoders, mixed I/O and dtype rejection."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "fp64_validator", ROOT / "examples/scientific_model_package/validate_package.py"
)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def mixed_validator(tmp_path):
    vi = helper.make_tensor_value_info
    model = helper.make_model(helper.make_graph([
        helper.make_node("Identity", ["x"], ["y"]),
        helper.make_node("Identity", ["flag"], ["valid"]),
    ], "mixed", [vi("x", TensorProto.DOUBLE, [2, 1]), vi("flag", TensorProto.BOOL, [2, 1])],
       [vi("y", TensorProto.DOUBLE, [2, 1]), vi("valid", TensorProto.BOOL, [2, 1])]),
       opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    (tmp_path / "input.csv").write_text("x,flag\n1.0000000000009095,true\n-2,false\n")
    (tmp_path / "expected.csv").write_text("y,valid\n1.0000000000009095,1\n-2,0\n")
    v = validator.Validator(tmp_path, "description.json", "0" * 64, True)
    v.integrity_ok = True
    v.models = {"mixed": {"inputs": ["x", "flag"], "outputs": ["y", "valid"]}}
    v.onnx_models = {"mixed": model}
    v.tensors = {
        name: {"onnx_name": name, "element_type": "tensor(double)" if name in {"x", "y"} else "tensor(bool)"}
        for name in ("x", "flag", "y", "valid")
    }
    v.artifacts = {"input": {"location": "input.csv", "format": {"name": "CSV"}},
                   "expected": {"location": "expected.csv", "format": {"name": "CSV"}}}
    case = {"id": "nominal", "model": "mixed", "input_artifact": "input",
            "expected_output_artifact": "expected", "shape": [2, 1],
            "dtype": {"inputs": {"x": "float64", "flag": "bool"},
                      "outputs": {"y": "float64", "valid": "bool"}},
            "input_locator": {"tensors": {"x": {"rows": [1, 2], "column": 0},
                                             "flag": {"rows": [1, 2], "column": 1}}},
            "expected_output_locator": {"tensors": {"y": {"rows": [1, 2], "column": 0},
                                                       "valid": {"rows": [1, 2], "column": 1}}},
            "comparison": {"absolute_tolerance": 0, "relative_tolerance": 0},
            "units": {"output": "1"}}
    v.description = {"verification_cases": [case]}
    return v, case


def numeric_validator(tmp_path, nodes, rows):
    vi = helper.make_tensor_value_info
    model = helper.make_model(helper.make_graph(
        nodes, "numeric",
        [vi(name, TensorProto.DOUBLE, ["batch", 1]) for name in ("x", "z")],
        [vi("y", TensorProto.DOUBLE, ["batch", 1])],
    ), opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    (tmp_path / "verification.csv").write_text(
        "x,z,y\n" + "".join(",".join(str(value) for value in row) + "\n" for row in rows)
    )
    v = validator.Validator(tmp_path, "description.json", "0" * 64, True)
    v.integrity_ok = True
    v.models = {"numeric": {"inputs": ["x", "z"], "outputs": ["y"]}}
    v.onnx_models = {"numeric": model}
    v.tensors = {
        name: {"onnx_name": name, "element_type": "tensor(double)",
               "rank": 2, "shape": ["batch", 1]}
        for name in ("x", "z", "y")
    }
    v.artifacts = {"verification": {"location": "verification.csv", "format": {"name": "CSV"}}}
    selected_rows = list(range(1, len(rows) + 1))
    case = {"id": "nominal", "model": "numeric", "input_artifact": "verification",
            "expected_output_artifact": "verification", "shape": [len(rows), 1],
            "dtype": "float64",
            "input_locator": {"tensors": {
                name: {"rows": selected_rows, "column": column}
                for column, name in enumerate(("x", "z"))
            }},
            "expected_output_locator": {"tensors": {"y": {"rows": selected_rows, "column": 2}}},
            "comparison": {"absolute_tolerance": 0, "relative_tolerance": 0},
            "units": {"output": "1"}}
    v.description = {"verification_cases": [case]}
    return v, case


@pytest.mark.parametrize("locator_key,name", [
    ("input_locator", "z"), ("expected_output_locator", "y"),
])
def test_named_locators_require_consistent_symbolic_sizes(tmp_path, locator_key, name):
    v, case = numeric_validator(
        tmp_path, [helper.make_node("Add", ["x", "z"], ["y"])],
        [(1, 10, 11), (2, 10, 12)],
    )
    case[locator_key]["tensors"][name].update(rows=[1], shape=[1, 1])

    assert v.execute_cases() is None
    finding = v.findings["VER-003-execution"]
    assert finding["status"] == "fail"
    assert "symbolic dimension 'batch' has inconsistent sizes: 2 and 1" in finding["message"]
    assert name in finding["message"]
    assert v.verification_results == []


def test_symbolic_sizes_are_bound_per_verification_case(tmp_path):
    from copy import deepcopy

    v, case = numeric_validator(
        tmp_path, [helper.make_node("Add", ["x", "z"], ["y"])],
        [(1, 10, 11), (2, 10, 12)],
    )
    smaller_case = deepcopy(case)
    smaller_case.update(id="dynamic-batch", shape=[1, 1])
    for locator_key in ("input_locator", "expected_output_locator"):
        for locator in smaller_case[locator_key]["tensors"].values():
            locator["rows"] = [1]
    v.description["verification_cases"].append(smaller_case)

    assert v.execute_cases() is not None
    assert [case["status"] for case in v.verification_results] == ["pass", "pass"]


@pytest.mark.parametrize("nonfinite", ["inf", "nan"])
def test_nonfinite_negative_control_is_detected(tmp_path, nonfinite):
    import json

    if nonfinite == "inf":
        nodes = [helper.make_node("Div", ["x", "z"], ["y"])]
        rows = [(0, 1, 0)]  # Nominal 0/1; swapped 1/0.
    else:
        nodes = [helper.make_node("Sub", ["x", "z"], ["difference"]),
                 helper.make_node("Log", ["difference"], ["y"])]
        rows = [(2, 1, 0)]  # Nominal log(1); swapped log(-1).
    v, _ = numeric_validator(tmp_path, nodes, rows)

    with np.errstate(divide="ignore", invalid="ignore"):
        result = v.execute_cases()

    assert result is not None
    assert result["negative_control_detected"] is True
    assert result["negative_control_nonfinite_detected"] is True
    assert result["negative_control_error"] is None
    assert v.findings["VER-003-execution"]["status"] == "pass"
    assert v.verification_results[0]["status"] == "pass"
    # Non-finite control values must not leak into the JSON observations.
    json.dumps(v.findings, allow_nan=False)


@pytest.mark.parametrize("numerator", [0, 1], ids=["nan", "inf"])
def test_nonfinite_nominal_output_still_fails(tmp_path, numerator):
    v, _ = numeric_validator(
        tmp_path, [helper.make_node("Div", ["x", "z"], ["y"])],
        [(numerator, 0, 0)],
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        assert v.execute_cases() is None

    finding = v.findings["VER-003-execution"]
    assert finding["status"] == "fail"
    assert "non-finite observed verification tensor" in finding["message"]


def test_mixed_fp64_bool_execution_uses_separate_artifacts(tmp_path):
    v, _ = mixed_validator(tmp_path)
    assert v.execute_cases() is not None
    outputs = v.verification_results[0]["outputs"]
    assert [item["dtype"] for item in outputs] == ["float64", "bool"]
    assert all(item["status"] == "pass" for item in outputs)
    assert outputs[1]["comparison"] == "exact"


def test_bool_mismatch_cannot_pass_with_large_numeric_tolerance(tmp_path):
    v, case = mixed_validator(tmp_path)
    (tmp_path / "expected.csv").write_text("y,valid\n1.0000000000009095,0\n-2,0\n")
    case["comparison"] = {"absolute_tolerance": 100, "relative_tolerance": 100}
    assert v.execute_cases() is None
    assert v.verification_results[0]["outputs"][1]["status"] == "fail"


@pytest.mark.parametrize("direction,name", [("inputs", "x"), ("outputs", "y"), ("inputs", "flag"), ("outputs", "valid")])
def test_verification_dtype_mismatch_is_rejected(tmp_path, direction, name):
    v, case = mixed_validator(tmp_path)
    case["dtype"][direction][name] = "float32"
    assert v.execute_cases() is None
    assert "dtype differs" in v.findings["VER-003-execution"]["message"]


@pytest.mark.parametrize("token", ["2", "-1", "yes", "nan", "", "1.0"])
def test_bool_csv_tokens_are_strict(token):
    with pytest.raises(validator.CheckFailure, match="invalid CSV bool"):
        validator.decode_csv_tensor([[token]], {"column": 0, "rows": [1]}, np.dtype("bool"), [1, 1])


def test_csv_fp64_decoding_preserves_small_increment():
    value = validator.decode_csv_tensor([["1.0000000000009095"]],
        {"column": 0, "rows": [1]}, np.dtype("float64"), [1, 1])
    assert value.dtype == np.float64
    assert value[0, 0] == 1 + 2**-40
    assert value[0, 0] != np.float32(value[0, 0])


def test_csv_multicolumn_tensor_shape():
    value = validator.decode_csv_tensor([["1", "2"], ["3", "4"]],
        {"columns": [0, 1], "rows": [1, 2], "shape": [2, 2]}, np.dtype("float64"), [2, 1])
    np.testing.assert_array_equal(value, [[1, 2], [3, 4]])


def test_onnx_signature_type_names():
    assert validator.onnx_element_type_name(TensorProto.DOUBLE) == "tensor(double)"
    assert validator.onnx_element_type_name(TensorProto.BOOL) == "tensor(bool)"


def test_mixed_dtype_mapping_control_preserves_types(tmp_path):
    v, _ = mixed_validator(tmp_path)
    result = v.execute_cases()
    assert result["negative_control_detected"] is True


def test_complete_float64_package_signature_and_verification(tmp_path):
    """Exercise the real description adapter and ONNX dtype/shape checks."""
    import csv
    import hashlib
    import json
    import subprocess
    from onnx import numpy_helper

    package = tmp_path / "package"
    subprocess.run([sys.executable, str(ROOT / "examples/scientific_model_package/build_package.py"), str(package)], check=True)
    description_path = package / "package-description.json"
    desc = json.loads(description_path.read_text())
    model_path = package / "models/linear-response.onnx"
    model = onnx.load(model_path)
    for value in list(model.graph.input) + list(model.graph.output):
        value.type.tensor_type.elem_type = TensorProto.DOUBLE
    for tensor in model.graph.initializer:
        tensor.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(tensor).astype(np.float64), tensor.name))
    onnx.save(model, model_path)
    for contract in desc["tensors"]:
        contract["element_type"] = "tensor(double)"
    for case in desc["verification_cases"]:
        case["dtype"] = "float64"
    # Use double coefficients exactly as stored in the converted graph.
    gains = [numpy_helper.to_array(t).item() for t in model.graph.initializer]
    csv_path = package / "verification/nominal.csv"
    with csv_path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    for row in rows[1:]:
        row[2] = repr(float(row[0]) * gains[0] + float(row[1]) * gains[1])
    with csv_path.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    for artifact in desc["artifacts"]:
        path = package / artifact["location"]
        artifact["byte_size"] = path.stat().st_size
        artifact["digests"] = [{"algorithm": "sha256", "value": hashlib.sha256(path.read_bytes()).hexdigest()}]
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    for case in desc["verification_cases"]:
        case["model_digest"] = {"algorithm": "sha256", "value": digest}
    description_path.write_text(json.dumps(desc))
    expected_digest = hashlib.sha256(description_path.read_bytes()).hexdigest()
    report_path = tmp_path / "report.json"
    result = subprocess.run([sys.executable, str(ROOT / "examples/scientific_model_package/validate_package.py"), str(package),
                             "--expected-description-digest", expected_digest, "--run-verification", "--report", str(report_path)],
                            text=True, capture_output=True)
    report = json.loads(report_path.read_text())
    checks = {check["id"]: check for check in report["checks"]}
    # Other sample evidence is intentionally still the original float32 evidence;
    # this test targets actual graph contracts and validator-owned execution.
    assert checks["ONNX-003"]["status"] == "pass", result.stdout + result.stderr
    assert checks["VER-002"]["status"] == "pass"
    assert checks["VER-003"]["status"] == "pass"
    assert all(case["outputs"][0]["dtype"] == "float64" for case in report["verification_case_results"])
