from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

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
