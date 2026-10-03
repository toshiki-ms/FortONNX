from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = REPOSITORY_ROOT / "examples" / "scientific_model_package"
VALIDATOR = EXAMPLE_ROOT / "validate_package.py"


def build_package(package_root: Path) -> None:
    subprocess.run(
        [sys.executable, str(EXAMPLE_ROOT / "build_package.py"), str(package_root)],
        cwd=REPOSITORY_ROOT,
        check=True,
    )


def validate_package(
    package_root: Path, report_path: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            str(package_root),
            "--expected-description-digest-file",
            str(package_root / "package-description.sha256"),
            "--run-verification",
            "--report",
            str(report_path),
            "--summary",
            str(report_path.with_suffix(".txt")),
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def findings_by_id(report: dict[str, object]) -> dict[str, dict[str, object]]:
    checks = report["checks"]
    assert isinstance(checks, list)
    return {finding["id"]: finding for finding in checks}


def write_description(package_root: Path, description: dict[str, object]) -> None:
    description_path = package_root / "package-description.json"
    description_path.write_text(json.dumps(description), encoding="utf-8")
    digest = hashlib.sha256(description_path.read_bytes()).hexdigest()
    (package_root / "package-description.sha256").write_text(
        f"sha256 {digest} package-description.json\n", encoding="utf-8"
    )


def test_validator_accepts_complete_example(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    report_path = tmp_path / "validation-report.json"
    build_package(package_root)

    result = validate_package(package_root, report_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Scientific ONNX package: CONFORMING" in result.stdout
    report_text = report_path.read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert report["overall"] == {
        "conforming": True,
        "operationally_accepted": False,
        "scientifically_validated": False,
        "structurally_conforming": True,
        "verification_passed": True,
    }
    assert report["not_run"] == []
    assert len(report["checks"]) == 64
    assert len(findings_by_id(report)) == 64
    assert findings_by_id(report)["PKG-R01"]["severity"] == "warning"
    assert findings_by_id(report)["PKG-R01"]["status"] == "fail"
    assert {case["id"] for case in report["verification_case_results"]} == {
        "nominal",
        "dynamic-batch",
    }
    assert str(tmp_path) not in report_text
    assert report_path.with_suffix(".txt").is_file()


def test_validator_rejects_corrupted_indexed_artifact(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    report_path = tmp_path / "corrupt-report.json"
    build_package(package_root)
    model_path = package_root / "models" / "linear-response.onnx"
    model_path.write_bytes(model_path.read_bytes() + b"corruption")

    result = validate_package(package_root, report_path)

    assert result.returncode == 1
    assert "Scientific ONNX package: NOT CONFORMING" in result.stdout
    report = json.loads(report_path.read_text(encoding="utf-8"))
    findings = findings_by_id(report)
    assert findings["PKG-005"]["status"] == "fail"
    assert findings["ONNX-001"]["status"] == "not-run"
    assert findings["VER-003"]["status"] == "not-run"
    assert report["overall"]["conforming"] is False
    assert report["overall"]["verification_passed"] is False


@pytest.mark.parametrize("columns", [[0], [0, 1, 2]], ids=["too-few", "too-many"])
def test_validator_rejects_legacy_input_column_count(
    tmp_path: Path, columns: list[int]
) -> None:
    package_root = tmp_path / "package"
    report_path = tmp_path / "column-count-report.json"
    build_package(package_root)
    description = json.loads((package_root / "package-description.json").read_text())
    description["verification_cases"][0]["input_locator"]["columns"] = columns
    write_description(package_root, description)

    result = validate_package(package_root, report_path)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    findings = findings_by_id(report)
    for check_id in ("VER-002", "VER-003"):
        assert findings[check_id]["status"] == "fail"
        assert "exactly one column per input" in findings[check_id]["message"]
    assert report["overall"]["conforming"] is False
    assert report["overall"]["verification_passed"] is False
    assert report_path.with_suffix(".txt").is_file()


@pytest.mark.parametrize("direction", ["inputs", "outputs"])
@pytest.mark.parametrize("column_field", ["column", "columns"])
@pytest.mark.parametrize("missing_mapping", ["none", "rows", "column", "both"])
def test_validator_requires_named_tensor_locator_mapping(
    tmp_path: Path, direction: str, column_field: str, missing_mapping: str
) -> None:
    package_root = tmp_path / "package"
    report_path = tmp_path / "tensor-mapping-report.json"
    build_package(package_root)
    description = json.loads((package_root / "package-description.json").read_text())
    case = description["verification_cases"][0]
    model = next(model for model in description["models"] if model["id"] == case["model"])
    tensors = {tensor["id"]: tensor for tensor in description["tensors"]}
    locator_key = "input_locator" if direction == "inputs" else "expected_output_locator"
    legacy_locator = case[locator_key]
    columns = legacy_locator["columns"] if direction == "inputs" else [legacy_locator["column"]]
    tensor_locators = {
        tensors[tensor_id]["onnx_name"]: {
            column_field: column if column_field == "column" else [column],
            "rows": legacy_locator["rows"],
        }
        for tensor_id, column in zip(model[direction], columns, strict=True)
    }
    locator = tensor_locators[tensors[model[direction][0]]["onnx_name"]]
    if missing_mapping in {"rows", "both"}:
        del locator["rows"]
    if missing_mapping in {"column", "both"}:
        del locator[column_field]
    case[locator_key] = {"tensors": tensor_locators}
    write_description(package_root, description)

    result = validate_package(package_root, report_path)

    assert "Traceback" not in result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    findings = findings_by_id(report)
    if missing_mapping == "none":
        assert result.returncode == 0, result.stdout + result.stderr
        assert findings["VER-002"]["status"] == "pass"
        assert findings["VER-003"]["status"] == "pass"
    else:
        assert result.returncode == 1, result.stdout + result.stderr
        assert findings["VER-002"]["status"] == "fail"
        missing_field = "rows" if missing_mapping in {"rows", "both"} else "column"
        assert f"missing fields: {missing_field}" in findings["VER-002"]["message"]
        assert report["overall"]["conforming"] is False
        assert report["overall"]["verification_passed"] is False


@pytest.mark.parametrize("direction", ["inputs", "outputs"])
@pytest.mark.parametrize("locator", [None, "column 0", []], ids=["null", "string", "array"])
def test_validator_reports_non_object_tensor_locator(
    tmp_path: Path, direction: str, locator: object
) -> None:
    package_root = tmp_path / "package"
    report_path = tmp_path / "tensor-locator-report.json"
    build_package(package_root)
    description = json.loads((package_root / "package-description.json").read_text())
    case = description["verification_cases"][0]
    model = next(model for model in description["models"] if model["id"] == case["model"])
    tensors = {tensor["id"]: tensor for tensor in description["tensors"]}
    locator_key = "input_locator" if direction == "inputs" else "expected_output_locator"
    legacy_locator = case[locator_key]
    columns = legacy_locator["columns"] if direction == "inputs" else [legacy_locator["column"]]
    tensor_locators = {
        tensors[tensor_id]["onnx_name"]: {"column": column, "rows": legacy_locator["rows"]}
        for tensor_id, column in zip(model[direction], columns, strict=True)
    }
    tensor_locators[tensors[model[direction][0]]["onnx_name"]] = locator
    case[locator_key] = {"tensors": tensor_locators}
    write_description(package_root, description)

    result = validate_package(package_root, report_path)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    findings = findings_by_id(report)
    for check_id in ("VER-002", "VER-003"):
        assert findings[check_id]["status"] == "fail"
        assert "locator" in findings[check_id]["message"]
        assert "must be an object" in findings[check_id]["message"]
    assert report["overall"]["conforming"] is False
    assert report["overall"]["verification_passed"] is False
    assert report_path.with_suffix(".txt").is_file()
