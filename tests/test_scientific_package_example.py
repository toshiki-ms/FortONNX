from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import onnx

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = REPOSITORY_ROOT / "examples" / "scientific_model_package"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_scientific_package_builder(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    subprocess.run(
        [sys.executable, str(EXAMPLE_ROOT / "build_package.py"), str(package_root)],
        cwd=REPOSITORY_ROOT,
        check=True,
    )

    description_path = package_root / "package-description.json"
    description = json.loads(description_path.read_text(encoding="utf-8"))
    assert description["specification_id"] == "scientific-onnx-model-package"
    assert description["specification_version"] == "0.1-draft"
    assert description["package_version"] == "example-1"
    assert description["release_status"] == "research"
    assert description["primary_model"] == "linear-response-graph"
    assert description["models"][0]["opset_imports"] == [{"domain": "", "version": 18}]
    assert all(
        tensor["shape"] == ["batch-size", 1] for tensor in description["tensors"]
    )

    expected_artifacts = {
        "model-card",
        "license",
        "citation",
        "abstract-model-mapping",
        "linear-response-model",
        "nominal-cases",
        "reference-verification-report",
        "reference-python-example",
        "fortran-example",
    }
    assert {entry["id"] for entry in description["artifacts"]} == expected_artifacts
    for entry in description["artifacts"]:
        artifact_path = package_root / entry["location"]
        assert artifact_path.is_file()
        assert artifact_path.stat().st_size == entry["byte_size"]
        assert entry["digests"] == [
            {"algorithm": "sha256", "value": sha256(artifact_path)}
        ]

    detached_record = (package_root / "package-description.sha256").read_text(
        encoding="utf-8"
    )
    assert (
        detached_record
        == f"sha256 {sha256(description_path)} package-description.json\n"
    )

    model = onnx.load(package_root / "models" / "linear-response.onnx")
    onnx.checker.check_model(model)
    assert model.ir_version == 9
    assert [value.name for value in model.graph.input] == [
        "temperature_anomaly",
        "radiative_forcing",
    ]
    assert [value.name for value in model.graph.output] == ["temperature_response"]

    report_path = package_root / "conformance-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["package_description_digest"]["value"] == sha256(description_path)
    assert report["overall"] == {
        "conforming": True,
        "operationally_accepted": False,
        "scientifically_validated": False,
        "structurally_conforming": True,
        "verification_passed": True,
    }
    assert {result["id"] for result in report["verification_case_results"]} == {
        "nominal",
        "dynamic-batch",
    }
    assert any(
        finding["id"] == "PKG-R01" and finding["severity"] == "warning"
        for finding in report["checks"]
    )

    subprocess.run(
        [
            sys.executable,
            str(package_root / "examples" / "reference_run.py"),
            str(package_root),
        ],
        check=True,
    )

    original_description_digest = sha256(description_path)
    subprocess.run(
        [
            sys.executable,
            str(EXAMPLE_ROOT / "build_package.py"),
            "--replace",
            str(package_root),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    assert sha256(description_path) == original_description_digest


def test_builder_refuses_existing_destination(tmp_path: Path) -> None:
    package_root = tmp_path / "existing"
    package_root.mkdir()
    result = subprocess.run(
        [sys.executable, str(EXAMPLE_ROOT / "build_package.py"), str(package_root)],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "destination already exists" in result.stderr
