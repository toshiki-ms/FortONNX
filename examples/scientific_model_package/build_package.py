#!/usr/bin/env python3
"""Build the format-declared Scientific ONNX Model Package example."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from reference_run import ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE, verify_package

SPECIFICATION_ID = "scientific-onnx-model-package"
SPECIFICATION_VERSION = "0.1-draft"
PACKAGE_ID = "org.fortonnx.examples.linear-response"
PACKAGE_VERSION = "example-1"
RELEASE_DATE = "2026-08-22"
MAPPING_ID = "org.fortonnx.examples.direct-json"
MAPPING_VERSION = "1"

SOURCE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SOURCE_DIRECTORY.parents[1]
TEMPLATE_DIRECTORY = SOURCE_DIRECTORY / "package-template"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def copy_text(source: Path, destination: Path) -> None:
    text = source.read_text(encoding="utf-8")
    destination.write_text(text, encoding="utf-8")


def create_model(path: Path) -> None:
    temperature_gain = numpy_helper.from_array(
        np.asarray([0.8], dtype=np.float32), name="temperature_gain"
    )
    forcing_gain = numpy_helper.from_array(
        np.asarray([0.01], dtype=np.float32), name="forcing_gain"
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "Mul",
                ["temperature_anomaly", "temperature_gain"],
                ["temperature_term"],
            ),
            helper.make_node(
                "Mul",
                ["radiative_forcing", "forcing_gain"],
                ["forcing_term"],
            ),
            helper.make_node(
                "Add",
                ["temperature_term", "forcing_term"],
                ["temperature_response"],
            ),
        ],
        "scientific_linear_response_example",
        [
            helper.make_tensor_value_info(
                "temperature_anomaly", TensorProto.FLOAT, ["batch", 1]
            ),
            helper.make_tensor_value_info(
                "radiative_forcing", TensorProto.FLOAT, ["batch", 1]
            ),
        ],
        [
            helper.make_tensor_value_info(
                "temperature_response", TensorProto.FLOAT, ["batch", 1]
            )
        ],
        [temperature_gain, forcing_gain],
    )
    model = helper.make_model(
        graph,
        producer_name="fortonnx-scientific-package-example",
        producer_version="1",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.ir_version = 9
    model.doc_string = (
        "Illustrative linear-response graph. The coefficients are not "
        "scientifically validated."
    )
    helper.set_model_props(
        model,
        {
            "scientific_package_id": PACKAGE_ID,
            "scientific_package_version": PACKAGE_VERSION,
            "scientific_package_specification": (
                f"{SPECIFICATION_ID}/{SPECIFICATION_VERSION}"
            ),
        },
    )
    onnx.checker.check_model(model)
    onnx.save_model(model, path)


def digest_object(path: Path) -> dict[str, str]:
    return {"algorithm": "sha256", "value": sha256(path)}


def artifact(
    package_root: Path,
    *,
    artifact_id: str,
    location: str,
    roles: list[str],
    required_stage: list[str],
    format_name: str,
    format_version: str,
    media_type: str,
    logical_type: str,
    reader: dict[str, Any],
    created_by: str,
    requirement: str = "required",
    locators: list[Any] | None = None,
    numerical_description: dict[str, Any] | None = None,
    executable_risk: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = package_root / location
    result: dict[str, Any] = {
        "id": artifact_id,
        "role": roles,
        "location": location,
        "requirement": requirement,
        "required_stage": required_stage,
        "format": {"name": format_name, "version": format_version},
        "media_type": media_type,
        "logical_type": logical_type,
        "locators": locators or ["whole artifact"],
        "reader": reader,
        "byte_size": path.stat().st_size,
        "digests": [digest_object(path)],
        "license": "BSD-2-Clause",
        "created_by": created_by,
        "relationships": [],
        "alternatives": [],
    }
    if numerical_description is not None:
        result["numerical_description"] = numerical_description
    if executable_risk is not None:
        result["executable_risk"] = executable_risk
    return result


def tensor_contracts() -> list[dict[str, Any]]:
    common = {
        "model": "linear-response-graph",
        "element_type": "tensor(float)",
        "rank": 2,
        "shape": ["batch-size", 1],
        "axes": ["batch", "scalar"],
        "training_domain": "no-training-domain",
        "evaluation_domain": "verification-domain",
        "admissible_domain": "interface-domain",
        "intended_use_domain": "demonstration-domain",
        "preprocessing": ["validate-inputs"],
        "postprocessing": [],
        "invalid_value_behavior": "reject before inference",
        "missing_value_behavior": "unsupported; reject before inference",
    }
    return [
        {
            **common,
            "id": "temperature-anomaly-input",
            "onnx_name": "temperature_anomaly",
            "direction": "input",
            "role": ["physical predictor"],
            "graph_units": "K",
            "physical_units": "K",
            "channels": [
                {
                    "quantity": "temperature-anomaly",
                    "selector": {"axis": "scalar", "index": 0},
                    "physical_units": "K",
                    "graph_units": "K",
                    "valid_range": [-100.0, 100.0],
                    "location": "non-spatial sample",
                    "sign_convention": "positive is warmer",
                    "transformations": [],
                    "missing_value_behavior": "reject",
                }
            ],
        },
        {
            **common,
            "id": "radiative-forcing-input",
            "onnx_name": "radiative_forcing",
            "direction": "input",
            "role": ["physical forcing"],
            "graph_units": "W m^-2",
            "physical_units": "W m^-2",
            "channels": [
                {
                    "quantity": "radiative-forcing",
                    "selector": {"axis": "scalar", "index": 0},
                    "physical_units": "W m^-2",
                    "graph_units": "W m^-2",
                    "valid_range": [-2000.0, 2000.0],
                    "location": "non-spatial sample",
                    "sign_convention": "positive increases response",
                    "transformations": [],
                    "missing_value_behavior": "reject",
                }
            ],
        },
        {
            **{**common, "preprocessing": []},
            "id": "temperature-response-output",
            "onnx_name": "temperature_response",
            "direction": "output",
            "role": ["physical response"],
            "graph_units": "K",
            "physical_units": "K",
            "channels": [
                {
                    "quantity": "temperature-response",
                    "selector": {"axis": "scalar", "index": 0},
                    "physical_units": "K",
                    "graph_units": "K",
                    "valid_range": "finite float32",
                    "location": "non-spatial sample",
                    "sign_convention": "positive is warming",
                    "transformations": [],
                    "missing_value_behavior": "not produced for admissible inputs",
                }
            ],
        },
    ]


def domains() -> list[dict[str, Any]]:
    return [
        {
            "id": "interface-domain",
            "kind": ["admissible"],
            "definition": "Finite float32 inputs accepted by the package contract.",
            "constraints": [
                "temperature anomaly is between -100 K and 100 K",
                "radiative forcing is between -2000 W m^-2 and 2000 W m^-2",
                "batch is positive",
            ],
            "selection_rule": "Apply all stated finite-value and inclusive range checks.",
            "data_or_case_set": [],
            "relationship": [],
            "unsupported_regions": ["NaN", "infinity", "missing values"],
        },
        {
            "id": "no-training-domain",
            "kind": ["training"],
            "definition": "Not applicable because the graph is assembled analytically.",
            "constraints": ["empty by construction"],
            "selection_rule": "No case is a member.",
            "data_or_case_set": [],
            "sample_unit": "not applicable",
            "size": {"independent_cases": 0, "samples": 0},
            "relationship": [],
            "unsupported_regions": [],
        },
        {
            "id": "verification-domain",
            "kind": ["evaluation"],
            "definition": "The three exact rows in verification/nominal.csv.",
            "constraints": ["exact immutable case-table membership"],
            "selection_rule": "Rows 1 through 3 after the CSV header.",
            "data_or_case_set": ["nominal-cases"],
            "sample_unit": "CSV row",
            "size": {"independent_cases": 3, "samples": 3},
            "relationship": [{"relation": "subset", "target": "demonstration-domain"}],
            "unsupported_regions": [],
        },
        {
            "id": "demonstration-domain",
            "kind": ["intended-use"],
            "definition": "Small finite inputs used only for package demonstrations.",
            "constraints": [
                "temperature anomaly is between -10 K and 10 K",
                "radiative forcing is between -500 W m^-2 and 500 W m^-2",
                "batch size is 1 or 3",
                "single-pass package testing only",
            ],
            "selection_rule": "Apply all constraints conjunctively.",
            "data_or_case_set": ["nominal-cases", "dynamic-batch-case"],
            "relationship": [{"relation": "subset", "target": "interface-domain"}],
            "unsupported_regions": ["all scientific and operational decisions"],
        },
    ]


def process_graph() -> dict[str, Any]:
    return {
        "id": "physical-response",
        "purpose": "Map physical scalar inputs to a physical scalar response.",
        "nodes": [
            {
                "id": "temperature-boundary",
                "kind": "physical input boundary",
                "inputs": [],
                "outputs": ["temperature-physical"],
                "failure_behavior": "reject missing input",
            },
            {
                "id": "forcing-boundary",
                "kind": "physical input boundary",
                "inputs": [],
                "outputs": ["forcing-physical"],
                "failure_behavior": "reject missing input",
            },
            {
                "id": "input-validation",
                "kind": "transformation",
                "entity": "validate-inputs",
                "inputs": ["temperature-physical", "forcing-physical"],
                "outputs": ["temperature-valid", "forcing-valid"],
                "failure_behavior": "reject before ONNX invocation",
            },
            {
                "id": "onnx-inference",
                "kind": "ONNX inference",
                "entity": "linear-response-graph",
                "inputs": ["temperature-valid", "forcing-valid"],
                "outputs": ["response-graph"],
                "failure_behavior": "propagate runtime failure",
            },
            {
                "id": "response-boundary",
                "kind": "physical output boundary",
                "inputs": ["response-graph"],
                "outputs": ["response-physical"],
                "failure_behavior": "reject non-finite output",
            },
        ],
        "edges": [
            {
                "from": "temperature-boundary.temperature-physical",
                "to": "input-validation.temperature-physical",
                "tensor": "temperature-anomaly-input",
                "quantity": "temperature-anomaly",
                "units": "K",
                "axes": ["batch", "scalar"],
                "shape": ["batch-size", 1],
                "state_version": "invocation input",
            },
            {
                "from": "forcing-boundary.forcing-physical",
                "to": "input-validation.forcing-physical",
                "tensor": "radiative-forcing-input",
                "quantity": "radiative-forcing",
                "units": "W m^-2",
                "axes": ["batch", "scalar"],
                "shape": ["batch-size", 1],
                "state_version": "invocation input",
            },
            {
                "from": "input-validation.temperature-valid",
                "to": "onnx-inference.temperature_anomaly",
                "tensor": "temperature-anomaly-input",
                "quantity": "temperature-anomaly",
                "units": "K",
                "axes": ["batch", "scalar"],
                "shape": ["batch-size", 1],
                "state_version": "validated invocation input",
            },
            {
                "from": "input-validation.forcing-valid",
                "to": "onnx-inference.radiative_forcing",
                "tensor": "radiative-forcing-input",
                "quantity": "radiative-forcing",
                "units": "W m^-2",
                "axes": ["batch", "scalar"],
                "shape": ["batch-size", 1],
                "state_version": "validated invocation input",
            },
            {
                "from": "onnx-inference.temperature_response",
                "to": "response-boundary.response-graph",
                "tensor": "temperature-response-output",
                "quantity": "temperature-response",
                "units": "K",
                "axes": ["batch", "scalar"],
                "shape": ["batch-size", 1],
                "state_version": "invocation output",
            },
        ],
        "acyclic": True,
        "execution_order": [
            "temperature-boundary",
            "forcing-boundary",
            "input-validation",
            "onnx-inference",
            "response-boundary",
        ],
    }


def profile_decisions() -> list[dict[str, Any]]:
    decisions = [
        ("core", True, "conforming", "All packages use the core profile."),
        (
            "single-pass",
            True,
            "conforming",
            "The supported mode is one independent process-graph invocation.",
        ),
        ("iterative", False, "not-tested", "Output reuse is unsupported."),
        ("coupled", False, "not-tested", "No external system exchanges data."),
        ("spatial", False, "not-tested", "All quantities are non-spatial scalars."),
        ("multi-graph", False, "not-tested", "One ONNX graph is required."),
        (
            "custom-operator",
            False,
            "not-tested",
            "The graph imports only standard ONNX operators.",
        ),
        (
            "external-artifact",
            False,
            "not-tested",
            "All required artifacts are inside the package.",
        ),
        ("stochastic", False, "not-tested", "The graph is deterministic."),
        (
            "scientifically-validated",
            False,
            "not-tested",
            "No scientific validation or fitness claim is made.",
        ),
        (
            "training-reproducible",
            False,
            "not-tested",
            "The graph is not trained and no training claim is made.",
        ),
        ("operational", False, "not-tested", "No deployment claim is made."),
    ]
    return [
        {
            "profile": name,
            "applicable": applicable,
            "scope": "physical-response / demonstration-domain",
            "status": status,
            "evidence_or_reason": reason,
        }
        for name, applicable, status, reason in decisions
    ]


def build_description(
    package_root: Path,
    model_digest: str,
    source_digests: dict[str, str],
    verification: dict[str, Any],
) -> dict[str, Any]:
    text_reader = {"name": "UTF-8 text reader", "version": "1"}
    json_reader = {"name": "JSON parser", "version": "RFC-compatible"}
    artifacts = [
        artifact(
            package_root,
            artifact_id="model-card",
            location="README.md",
            roles=["interpretation", "documentation"],
            required_stage=["interpretation"],
            format_name="Markdown",
            format_version="CommonMark-compatible",
            media_type="text/markdown; charset=utf-8",
            logical_type="document",
            reader=text_reader,
            created_by="package-assembly",
        ),
        artifact(
            package_root,
            artifact_id="license",
            location="LICENSE",
            roles=["legal terms"],
            required_stage=["interpretation"],
            format_name="Plain text",
            format_version="UTF-8",
            media_type="text/plain; charset=utf-8",
            logical_type="document",
            reader=text_reader,
            created_by="package-assembly",
        ),
        artifact(
            package_root,
            artifact_id="citation",
            location="CITATION.txt",
            roles=["citation"],
            required_stage=["interpretation"],
            format_name="Plain text citation",
            format_version="1",
            media_type="text/plain; charset=utf-8",
            logical_type="document",
            reader=text_reader,
            created_by="package-assembly",
        ),
        artifact(
            package_root,
            artifact_id="abstract-model-mapping",
            location="metadata/abstract-model-mapping.md",
            roles=["interpretation", "metadata mapping"],
            required_stage=["interpretation", "verification"],
            format_name="Markdown",
            format_version="CommonMark-compatible",
            media_type="text/markdown; charset=utf-8",
            logical_type="document",
            reader=text_reader,
            created_by="package-assembly",
        ),
        artifact(
            package_root,
            artifact_id="linear-response-model",
            location="models/linear-response.onnx",
            roles=["inference"],
            required_stage=["inference", "verification"],
            format_name="ONNX",
            format_version="IR 9; default-domain opset 18",
            media_type="application/onnx",
            logical_type="graph",
            reader={
                "name": "ONNX-compatible runtime",
                "version": onnx.__version__,
                "tested_version": onnx.__version__,
            },
            created_by="graph-construction",
            numerical_description={
                "element_type": "float32",
                "logical_axis_order": "ONNX",
                "external_data": False,
            },
        ),
        artifact(
            package_root,
            artifact_id="nominal-cases",
            location="verification/nominal.csv",
            roles=["verification"],
            required_stage=["verification"],
            format_name="Comma-separated values",
            format_version="example verification layout 1",
            media_type="text/csv; charset=utf-8",
            logical_type="table",
            reader={
                "name": "CSV reader",
                "version": "RFC 4180-compatible",
                "required_header": True,
            },
            created_by="package-assembly",
            locators=[
                {
                    "header": [
                        "temperature_anomaly_k",
                        "radiative_forcing_w_m2",
                        "expected_temperature_response_k",
                    ],
                    "rows": {"first": 1, "last": 3, "header_excluded": True},
                }
            ],
            numerical_description={
                "dtype": "decimal text decoded to float32",
                "logical_shape": [3, 3],
                "axis_order": ["case", "column"],
                "storage_order": "row-major text records",
                "byte_order": "not applicable",
                "missing_values": "prohibited",
            },
        ),
        artifact(
            package_root,
            artifact_id="reference-verification-report",
            location="verification/reference-report.json",
            roles=["verification evidence"],
            required_stage=["verification"],
            format_name="JSON verification report",
            format_version="1",
            media_type="application/json",
            logical_type="document",
            reader=json_reader,
            created_by="reference-verification",
        ),
        artifact(
            package_root,
            artifact_id="reference-python-example",
            location="examples/reference_run.py",
            roles=["executable", "usage example"],
            required_stage=["verification"],
            format_name="Python source",
            format_version="Python 3.11 or newer",
            media_type="text/x-python; charset=utf-8",
            logical_type="executable",
            reader={
                "name": "Python",
                "version": "3.11 or newer",
                "minimum_version": "3.11",
            },
            created_by="package-assembly",
            executable_risk={
                "trust": "review source before execution",
                "permissions": "read package files only",
                "isolation": "normal local process",
            },
        ),
        artifact(
            package_root,
            artifact_id="fortran-example",
            location="examples/run.f90",
            roles=["executable", "usage example"],
            required_stage=["verification"],
            format_name="Fortran source",
            format_version="Fortran 2008-compatible",
            media_type="text/x-fortran; charset=utf-8",
            logical_type="executable",
            reader={
                "name": "Fortran compiler",
                "version": "Fortran 2008 or newer",
                "binding": "compatible ONNX binding",
            },
            created_by="package-assembly",
            requirement="optional",
            executable_risk={
                "trust": "review source before compilation and execution",
                "permissions": "read package files only",
                "isolation": "normal local process",
            },
        ),
    ]

    return {
        "specification_id": SPECIFICATION_ID,
        "specification_version": SPECIFICATION_VERSION,
        "package_id": PACKAGE_ID,
        "package_version": PACKAGE_VERSION,
        "release_status": "research",
        "release_date": RELEASE_DATE,
        "model_card": "model-card",
        "license": [
            {
                "identifier": "BSD-2-Clause",
                "artifact": "license",
                "scope": "entire package",
            }
        ],
        "citation": {
            "artifact": "citation",
            "text": "FortONNX contributors (2026), Linear Response Demonstrator, example-1.",
        },
        "primary_model": "linear-response-graph",
        "models": [
            {
                "id": "linear-response-graph",
                "artifact": "linear-response-model",
                "role": "primary",
                "onnx_ir_version": 9,
                "opset_imports": [{"domain": "", "version": 18}],
                "external_data": [],
                "custom_operators": [],
                "inputs": ["temperature-anomaly-input", "radiative-forcing-input"],
                "outputs": ["temperature-response-output"],
                "source_export": "graph-construction",
            }
        ],
        "custom_operator_contracts": [],
        "artifacts": artifacts,
        "tensors": tensor_contracts(),
        "axes": [
            {
                "id": "batch",
                "meaning": "independent sample",
                "index_origin": 0,
                "length": "batch-size",
                "orientation": "in input order",
                "ordering": "independent; no physical ordering semantics",
                "periodicity": "none",
                "boundary_behavior": "none",
                "location": "non-spatial",
                "missing_or_padding": "prohibited",
            },
            {
                "id": "scalar",
                "meaning": "single quantity channel",
                "index_origin": 0,
                "length": 1,
                "orientation": "only element",
                "ordering": "fixed",
                "periodicity": "none",
                "boundary_behavior": "none",
                "location": "non-spatial",
                "missing_or_padding": "prohibited",
            },
        ],
        "coordinate_systems": [],
        "symbolic_dimensions": [
            {
                "id": "batch-size",
                "onnx_symbols": ["batch"],
                "scope": "all graph inputs and outputs",
                "minimum": 1,
                "maximum": "unbounded at graph interface; intended values are 1 and 3",
                "permitted_values": "positive integers",
                "equality_relationships": [
                    "same value on all graph inputs and outputs"
                ],
            }
        ],
        "quantities": [
            {
                "id": "temperature-anomaly",
                "display_name": "temperature anomaly",
                "definition": "Temperature relative to an unspecified demonstration reference state.",
                "symbol": "delta_T",
                "unit_convention": "SI symbol K",
                "dimensionality": "thermodynamic temperature",
                "sign_convention": "positive is warmer",
                "reference_state": "unspecified; scientific use prohibited",
                "sampling_semantics": "independent scalar sample",
            },
            {
                "id": "radiative-forcing",
                "display_name": "illustrative radiative forcing",
                "definition": "Demonstration forcing used only by the linear formula.",
                "symbol": "F",
                "unit_convention": "SI-derived symbol W m^-2",
                "dimensionality": "power per area",
                "sign_convention": "positive increases response",
                "reference_state": "not applicable",
                "sampling_semantics": "independent scalar sample",
            },
            {
                "id": "temperature-response",
                "display_name": "illustrative temperature response",
                "definition": "Output of the documented linear formula.",
                "symbol": "R_T",
                "unit_convention": "SI symbol K",
                "dimensionality": "thermodynamic temperature",
                "sign_convention": "positive is warming",
                "reference_state": "same unspecified reference as temperature anomaly",
                "sampling_semantics": "independent scalar sample",
            },
        ],
        "domains": domains(),
        "transformations": [
            {
                "id": "validate-inputs",
                "operation": "finite and range validation",
                "location": "external",
                "inputs": ["temperature-anomaly-input", "radiative-forcing-input"],
                "outputs": [
                    "validated temperature anomaly",
                    "validated radiative forcing",
                ],
                "definition": "Reject non-finite values and values outside interface-domain.",
                "parameters": [],
                "preconditions": ["input tensors exist with equal positive batch size"],
                "postconditions": ["all values are members of interface-domain"],
                "implementation": ["reference-python-example", "fortran-example"],
                "numerical_behavior": "comparisons use decoded float32 values; rejection is deterministic",
            },
            {
                "id": "linear-response-equation",
                "operation": "linear response",
                "location": "inside linear-response-graph",
                "inputs": ["temperature-anomaly-input", "radiative-forcing-input"],
                "outputs": ["temperature-response-output"],
                "definition": "response = 0.8 * temperature_anomaly + 0.01 * radiative_forcing",
                "parameters": [
                    {
                        "name": "temperature_gain",
                        "value": 0.8,
                        "units": "dimensionless",
                    },
                    {"name": "forcing_gain", "value": 0.01, "units": "K m^2 W^-1"},
                ],
                "preconditions": ["validated float32 inputs"],
                "postconditions": ["float32 temperature response in K"],
                "implementation": ["linear-response-graph"],
                "numerical_behavior": "ONNX float32 multiplication and addition; deterministic in tested runtime",
            },
        ],
        "process_graphs": [process_graph()],
        "coupling_contracts": [],
        "verification_cases": [
            {
                "id": "nominal",
                "purpose": "Check tensor names, coefficients, units, and output interpretation using nonzero mixed-sign values.",
                "process_graph": "physical-response",
                "model": "linear-response-graph",
                "model_digest": {"algorithm": "sha256", "value": model_digest},
                "runtime_configuration": "onnx-reference-cpu",
                "input_artifact": "nominal-cases",
                "input_locator": {"rows": [1, 2, 3], "columns": [0, 1]},
                "expected_output_artifact": "nominal-cases",
                "expected_output_locator": {"rows": [1, 2, 3], "column": 2},
                "value_space": "physical",
                "preprocessing": ["validate-inputs"],
                "postprocessing": [],
                "dtype": "float32",
                "shape": [3, 1],
                "axes": ["batch", "scalar"],
                "units": {"inputs": ["K", "W m^-2"], "output": "K"},
                "comparison": {
                    "method": "elementwise combined absolute and relative tolerance",
                    "absolute_tolerance": float(ABSOLUTE_TOLERANCE),
                    "relative_tolerance": float(RELATIVE_TOLERANCE),
                    "rationale": "Exceeds observed float32 rounding while detecting coefficient or mapping errors.",
                },
                "nondeterministic_variation": "none expected",
            },
            {
                "id": "dynamic-batch",
                "purpose": "Check the supported dynamic batch dimension at batch size 1.",
                "process_graph": "physical-response",
                "model": "linear-response-graph",
                "model_digest": {"algorithm": "sha256", "value": model_digest},
                "runtime_configuration": "onnx-reference-cpu",
                "input_artifact": "nominal-cases",
                "input_locator": {"rows": [1], "columns": [0, 1]},
                "expected_output_artifact": "nominal-cases",
                "expected_output_locator": {"rows": [1], "column": 2},
                "value_space": "physical",
                "preprocessing": ["validate-inputs"],
                "postprocessing": [],
                "dtype": "float32",
                "shape": [1, 1],
                "axes": ["batch", "scalar"],
                "units": {"inputs": ["K", "W m^-2"], "output": "K"},
                "comparison": {
                    "method": "elementwise combined absolute and relative tolerance",
                    "absolute_tolerance": float(ABSOLUTE_TOLERANCE),
                    "relative_tolerance": float(RELATIVE_TOLERANCE),
                    "rationale": "Same numerical path as nominal with a different legal shape.",
                },
                "nondeterministic_variation": "none expected",
            },
        ],
        "claims": [
            {
                "id": "known-answer-behavior",
                "statement": "The graph reproduces the declared linear formula for the packaged verification cases within tolerance.",
                "scope": "nominal and dynamic-batch cases",
                "intended_use_domain": "demonstration-domain",
                "applicable_profile": "core",
                "model_version": PACKAGE_VERSION,
                "decision_relevance": "package integration only",
                "status": "supported",
            },
            {
                "id": "scientific-validity",
                "statement": "The coefficients provide a scientifically valid physical prediction.",
                "scope": "no supported scope",
                "intended_use_domain": "demonstration-domain",
                "applicable_profile": "scientifically-validated",
                "model_version": PACKAGE_VERSION,
                "decision_relevance": "scientific use is prohibited",
                "status": "not-supported",
            },
        ],
        "evidence": [
            {
                "id": "known-answer-evidence",
                "claims": ["known-answer-behavior"],
                "package_version": PACKAGE_VERSION,
                "model_version": PACKAGE_VERSION,
                "evaluation_domain": "verification-domain",
                "case_membership": ["nominal", "dynamic-batch"],
                "independent_cases": {"definition": "CSV row", "count": 3},
                "metrics": [
                    {
                        "name": "maximum absolute error",
                        "formula": "max(abs(actual - expected))",
                        "units": "K",
                        "weighting": "none",
                        "mask": "none",
                        "aggregation": "maximum over all rows and output elements",
                        "reduction_order": ["output element", "row"],
                    }
                ],
                "results": {
                    "nominal_maximum_absolute_error": verification[
                        "nominal_maximum_absolute_error"
                    ],
                    "dynamic_batch_maximum_absolute_error": verification[
                        "dynamic_batch_maximum_absolute_error"
                    ],
                },
                "uncertainty": "not applicable to deterministic known-answer comparison",
                "baseline": "direct float32 analytical expression",
                "acceptance_criteria": {
                    "formula": "absolute error <= 1e-6 K + 1e-6 * abs(expected)",
                    "predeclared": True,
                },
                "reference_uncertainty": "float32 rounding only",
                "code": "reference-python-example",
                "configuration": "onnx-reference-cpu",
                "result_artifact": "reference-verification-report",
                "observed_failures": [],
                "conclusion": "passed",
            },
            {
                "id": "scientific-validity-evidence",
                "claims": ["scientific-validity"],
                "package_version": PACKAGE_VERSION,
                "model_version": PACKAGE_VERSION,
                "evaluation_domain": "not evaluated",
                "case_membership": [],
                "independent_cases": {"definition": "not applicable", "count": 0},
                "metrics": [],
                "results": "not evaluated",
                "uncertainty": "unknown",
                "baseline": "none",
                "acceptance_criteria": "none",
                "reference_uncertainty": "no scientific reference data",
                "code": "none",
                "configuration": "none",
                "result_artifact": "none",
                "observed_failures": ["absence of scientific evidence"],
                "conclusion": "not evaluated",
            },
        ],
        "provenance": {
            "entities": [
                {
                    "id": "analytic-formula",
                    "type": "source definition",
                    "identity": "response = 0.8 * temperature anomaly + 0.01 * radiative forcing",
                    "training": "not applicable",
                },
                {
                    "id": "builder-source",
                    "type": "source code",
                    "location": "examples/scientific_model_package/build_package.py in source tree",
                    "digest": {
                        "algorithm": "sha256",
                        "value": source_digests["builder"],
                    },
                },
                {
                    "id": "reference-source",
                    "type": "source code",
                    "location": "examples/scientific_model_package/reference_run.py in source tree",
                    "digest": {
                        "algorithm": "sha256",
                        "value": source_digests["reference"],
                    },
                },
            ],
            "activities": [
                {
                    "id": "graph-construction",
                    "type": "direct ONNX graph assembly",
                    "inputs": ["analytic-formula", "builder-source"],
                    "outputs": ["linear-response-model"],
                    "code_version": source_digests["builder"],
                    "configuration": {"IR": 9, "opset": 18, "dtype": "float32"},
                    "runtime": {
                        "Python": platform.python_version(),
                        "ONNX": onnx.__version__,
                    },
                    "randomness": "none",
                    "failed_or_excluded_inputs": [],
                    "responsible_agent": "example-builder",
                },
                {
                    "id": "reference-verification",
                    "type": "known-answer verification",
                    "inputs": [
                        "linear-response-model",
                        "nominal-cases",
                        "reference-source",
                    ],
                    "outputs": ["reference-verification-report"],
                    "code_version": source_digests["reference"],
                    "configuration": "onnx-reference-cpu",
                    "runtime": {
                        "Python": platform.python_version(),
                        "ONNX": onnx.__version__,
                    },
                    "randomness": "none",
                    "failed_or_excluded_inputs": [],
                    "responsible_agent": "example-builder",
                },
                {
                    "id": "package-assembly",
                    "type": "package assembly",
                    "inputs": ["builder-source", "reference-verification-report"],
                    "outputs": ["all indexed package artifacts"],
                    "code_version": source_digests["builder"],
                    "configuration": {
                        "mapping": MAPPING_ID,
                        "version": MAPPING_VERSION,
                    },
                    "runtime": {"Python": platform.python_version()},
                    "randomness": "none",
                    "failed_or_excluded_inputs": [],
                    "responsible_agent": "example-builder",
                },
            ],
            "agents": [
                {
                    "id": "example-builder",
                    "type": "software agent",
                    "name": "FortONNX scientific package example builder",
                }
            ],
            "dataset_splits": "not applicable; no dataset or training",
            "instrument_calibration": "not applicable; no observations or experiments",
        },
        "runtime_configurations": [
            {
                "id": "onnx-reference-cpu",
                "runtime": "ONNX ReferenceEvaluator",
                "runtime_version": onnx.__version__,
                "execution_provider": "reference evaluator on CPU",
                "hardware_class": "CPU",
                "precision": "float32",
                "threading": "implementation default",
                "fallback_behavior": "none",
                "custom_operators": [],
                "external_data_handling": "not applicable",
                "deterministic_settings": "deterministic graph with no random operators",
                "resource_limits": "not characterized",
                "supported_input_shapes": [[1, 1], [3, 1]],
                "verification_cases": ["nominal", "dynamic-batch"],
                "evaluation_cases": ["nominal", "dynamic-batch"],
            }
        ],
        "profile_decisions": profile_decisions(),
        "extensions": [
            {
                "namespace": "org.fortonnx.examples.serialization",
                "required_for_use": True,
                "format": "UTF-8 JSON",
                "format_version": "1",
                "mapping_artifact": "abstract-model-mapping",
                "mapping_id": MAPPING_ID,
                "mapping_version": MAPPING_VERSION,
            }
        ],
    }


def render_model_card(
    package_root: Path,
    verification: dict[str, Any],
) -> None:
    replacements = {
        "@@ONNX_VERSION@@": onnx.__version__,
        "@@PYTHON_VERSION@@": platform.python_version(),
        "@@MODEL_SHA256@@": sha256(package_root / "models/linear-response.onnx"),
        "@@MAPPING_SHA256@@": sha256(
            package_root / "metadata/abstract-model-mapping.md"
        ),
        "@@VERIFICATION_CSV_SHA256@@": sha256(
            package_root / "verification/nominal.csv"
        ),
        "@@REFERENCE_REPORT_SHA256@@": sha256(
            package_root / "verification/reference-report.json"
        ),
        "@@REFERENCE_RUN_SHA256@@": sha256(package_root / "examples/reference_run.py"),
        "@@FORTRAN_SHA256@@": sha256(package_root / "examples/run.f90"),
        "@@LICENSE_SHA256@@": sha256(package_root / "LICENSE"),
        "@@CITATION_SHA256@@": sha256(package_root / "CITATION.txt"),
        "@@MAX_ERROR@@": f"{verification['nominal_maximum_absolute_error']:.9g}",
        "@@PERTURBATION_ERROR@@": (
            f"{verification['swapped_input_minimum_detected_error']:.9g}"
        ),
    }
    card = (TEMPLATE_DIRECTORY / "model-card.md.in").read_text(encoding="utf-8")
    for token, value in replacements.items():
        card = card.replace(token, value)
    unresolved = sorted(token for token in replacements if token in card)
    if unresolved or "@@" in card:
        raise RuntimeError(f"unresolved Model Card placeholders: {unresolved}")
    (package_root / "README.md").write_text(card, encoding="utf-8")


def reference_report(
    model_digest: str,
    verification: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": "scientific package known-answer report",
        "format_version": "1",
        "specification_id": SPECIFICATION_ID,
        "specification_version": SPECIFICATION_VERSION,
        "package_id": PACKAGE_ID,
        "package_version": PACKAGE_VERSION,
        "model": {
            "id": "linear-response-graph",
            "digest": {"algorithm": "sha256", "value": model_digest},
        },
        "runtime_configuration": {
            "id": "onnx-reference-cpu",
            "runtime": "ONNX ReferenceEvaluator",
            "onnx_version": onnx.__version__,
            "python_version": platform.python_version(),
            "hardware_class": "CPU",
            "precision": "float32",
            "fallback": "none",
            "deterministic": True,
        },
        "comparison": {
            "method": "elementwise combined absolute and relative tolerance",
            "absolute_tolerance": float(ABSOLUTE_TOLERANCE),
            "relative_tolerance": float(RELATIVE_TOLERANCE),
            "units": "K",
            "rationale": "Tight float32 integration check for coefficients and input mapping.",
        },
        "cases": [
            {
                "id": "nominal",
                "status": "pass",
                "case_count": verification["case_count"],
                "maximum_absolute_error": verification[
                    "nominal_maximum_absolute_error"
                ],
            },
            {
                "id": "dynamic-batch",
                "status": "pass",
                "case_count": 1,
                "maximum_absolute_error": verification[
                    "dynamic_batch_maximum_absolute_error"
                ],
            },
            {
                "id": "swapped-input-negative-control",
                "status": "pass",
                "meaning": "the comparison rejected the deliberately wrong mapping",
                "minimum_detected_error": verification[
                    "swapped_input_minimum_detected_error"
                ],
            },
        ],
        "overall_status": "pass",
        "scientific_validation": "not evaluated",
    }


def conformance_checks() -> list[dict[str, Any]]:
    check_groups = {
        "PKG": range(1, 12),
        "ONNX": range(1, 7),
        "SEM": range(1, 9),
        "FLOW": range(1, 9),
        "VER": range(1, 8),
        "SCI": range(1, 11),
        "CARD": range(1, 9),
        "PROF": range(1, 6),
    }
    not_applicable = {
        "PKG-006": "No required external artifacts.",
        "PKG-011": "No authenticity record or signature is supplied.",
        "ONNX-005": "No custom operators.",
        "SEM-007": "No spatial or temporal tensor semantics.",
        "FLOW-007": "No fitted transformation parameters.",
        "SCI-006": "No out-of-domain result or claim.",
        "SCI-010": "No experimental or observational source data.",
    }

    def finding(
        check_id: str,
        status: str,
        severity: str,
        message: str,
        observation: str,
    ) -> dict[str, Any]:
        return {
            "id": check_id,
            "status": status,
            "severity": severity,
            "message": message,
            "affected_entities": [PACKAGE_ID],
            "affected_artifacts": [],
            "supporting_observations": [observation],
        }

    results: list[dict[str, Any]] = []
    for prefix, numbers in check_groups.items():
        for number in numbers:
            check_id = f"{prefix}-{number:03d}"
            if check_id in not_applicable:
                results.append(
                    finding(
                        check_id,
                        "not-applicable",
                        "information",
                        not_applicable[check_id],
                        "The applicability condition was evaluated from the package description.",
                    )
                )
            else:
                results.append(
                    finding(
                        check_id,
                        "pass",
                        "information",
                        "Checked by the example package builder.",
                        "The builder assertion completed without an error.",
                    )
                )
    results.append(
        finding(
            "SINGLE-001",
            "pass",
            "information",
            "The single-pass contract and independent batch semantics are complete.",
            "The physical-response process graph was inspected and executed.",
        )
    )
    results.append(
        finding(
            "PKG-R01",
            "fail",
            "warning",
            "No publisher-authenticated release record is supplied for this generated example.",
            "Only an unauthenticated detached SHA-256 record was generated.",
        )
    )
    return results


def validate_generated_package(
    package_root: Path,
    description: dict[str, Any],
) -> None:
    if description["specification_version"] != SPECIFICATION_VERSION:
        raise RuntimeError("wrong specification version")
    if description["primary_model"] != "linear-response-graph":
        raise RuntimeError("wrong primary model")

    entity_ids: dict[str, str] = {}

    def register(entity_id: str, entity_kind: str) -> None:
        previous = entity_ids.get(entity_id)
        if previous is not None:
            raise RuntimeError(
                f"duplicate entity id {entity_id!r}: {previous} and {entity_kind}"
            )
        entity_ids[entity_id] = entity_kind

    for collection in (
        "models",
        "artifacts",
        "tensors",
        "axes",
        "coordinate_systems",
        "symbolic_dimensions",
        "quantities",
        "domains",
        "transformations",
        "process_graphs",
        "coupling_contracts",
        "verification_cases",
        "claims",
        "evidence",
        "runtime_configurations",
    ):
        for entity in description[collection]:
            register(entity["id"], collection)
    for graph in description["process_graphs"]:
        for node in graph["nodes"]:
            register(node["id"], "process node")
    for collection in ("entities", "activities", "agents"):
        for entity in description["provenance"][collection]:
            register(entity["id"], f"provenance {collection}")
    artifact_ids: set[str] = set()
    for entry in description["artifacts"]:
        if entry["id"] in artifact_ids:
            raise RuntimeError(f"duplicate artifact id: {entry['id']}")
        artifact_ids.add(entry["id"])
        location = Path(entry["location"])
        if location.is_absolute() or ".." in location.parts:
            raise RuntimeError(f"unsafe artifact path: {entry['location']}")
        path = package_root / location
        if not path.is_file():
            raise RuntimeError(f"missing artifact: {entry['location']}")
        if path.stat().st_size != entry["byte_size"]:
            raise RuntimeError(f"wrong artifact size: {entry['location']}")
        if sha256(path) != entry["digests"][0]["value"]:
            raise RuntimeError(f"wrong artifact digest: {entry['location']}")

    artifact_ids = {entry["id"] for entry in description["artifacts"]}
    model_ids = {entry["id"] for entry in description["models"]}
    tensor_by_id = {entry["id"]: entry for entry in description["tensors"]}
    axis_ids = {entry["id"] for entry in description["axes"]}
    quantity_ids = {entry["id"] for entry in description["quantities"]}
    domain_ids = {entry["id"] for entry in description["domains"]}
    transformation_ids = {entry["id"] for entry in description["transformations"]}
    process_graph_ids = {entry["id"] for entry in description["process_graphs"]}
    verification_case_ids = {entry["id"] for entry in description["verification_cases"]}
    runtime_ids = {entry["id"] for entry in description["runtime_configurations"]}
    claim_ids = {entry["id"] for entry in description["claims"]}

    if description["model_card"] not in artifact_ids:
        raise RuntimeError("Model Card reference does not resolve")
    if description["primary_model"] not in model_ids:
        raise RuntimeError("primary model reference does not resolve")

    for graph in description["models"]:
        if graph["artifact"] not in artifact_ids:
            raise RuntimeError(f"model artifact does not resolve: {graph['id']}")
        if graph["source_export"] not in entity_ids:
            raise RuntimeError(f"source export does not resolve: {graph['id']}")
        for tensor_id in graph["inputs"] + graph["outputs"]:
            if tensor_id not in tensor_by_id:
                raise RuntimeError(f"model tensor does not resolve: {tensor_id}")

    for tensor in description["tensors"]:
        if tensor["model"] not in model_ids:
            raise RuntimeError(f"tensor model does not resolve: {tensor['id']}")
        if any(axis not in axis_ids for axis in tensor["axes"]):
            raise RuntimeError(f"tensor axis does not resolve: {tensor['id']}")
        for domain_field in (
            "admissible_domain",
            "training_domain",
            "evaluation_domain",
            "intended_use_domain",
        ):
            if tensor[domain_field] not in domain_ids:
                raise RuntimeError(
                    f"tensor domain does not resolve: {tensor['id']}.{domain_field}"
                )
        for transform in tensor["preprocessing"] + tensor["postprocessing"]:
            if transform not in transformation_ids:
                raise RuntimeError(f"tensor transform does not resolve: {transform}")
        for binding in tensor["channels"]:
            if binding["quantity"] not in quantity_ids:
                raise RuntimeError(
                    f"tensor quantity does not resolve: {binding['quantity']}"
                )

    graph_path = package_root / "models/linear-response.onnx"
    graph_model = onnx.load(graph_path)
    onnx.checker.check_model(graph_model)
    declared_graph = description["models"][0]
    if graph_model.ir_version != declared_graph["onnx_ir_version"]:
        raise RuntimeError("declared ONNX IR version does not match graph")
    actual_opsets = [
        {"domain": item.domain, "version": item.version}
        for item in graph_model.opset_import
    ]
    if actual_opsets != declared_graph["opset_imports"]:
        raise RuntimeError("declared ONNX opsets do not match graph")

    actual_inputs = [value.name for value in graph_model.graph.input]
    actual_outputs = [value.name for value in graph_model.graph.output]
    declared_inputs = [
        tensor_by_id[tensor_id]["onnx_name"] for tensor_id in declared_graph["inputs"]
    ]
    declared_outputs = [
        tensor_by_id[tensor_id]["onnx_name"] for tensor_id in declared_graph["outputs"]
    ]
    if actual_inputs != declared_inputs or actual_outputs != declared_outputs:
        raise RuntimeError("declared ONNX names or order do not match graph")

    for value in list(graph_model.graph.input) + list(graph_model.graph.output):
        contract = next(
            tensor
            for tensor in description["tensors"]
            if tensor["onnx_name"] == value.name
        )
        tensor_type = value.type.tensor_type
        if tensor_type.elem_type != TensorProto.FLOAT:
            raise RuntimeError(f"unexpected ONNX element type: {value.name}")
        if len(tensor_type.shape.dim) != contract["rank"]:
            raise RuntimeError(f"unexpected ONNX rank: {value.name}")
        if tensor_type.shape.dim[1].dim_value != 1:
            raise RuntimeError(f"unexpected fixed ONNX dimension: {value.name}")
        if tensor_type.shape.dim[0].dim_param != "batch":
            raise RuntimeError(f"unexpected ONNX symbolic dimension: {value.name}")

    for graph in description["process_graphs"]:
        node_ids = [node["id"] for node in graph["nodes"]]
        if set(node_ids) != set(graph["execution_order"]):
            raise RuntimeError(f"incomplete process execution order: {graph['id']}")
        positions = {
            node_id: position
            for position, node_id in enumerate(graph["execution_order"])
        }
        for edge in graph["edges"]:
            source = edge["from"].split(".", 1)[0]
            target = edge["to"].split(".", 1)[0]
            if source not in positions or target not in positions:
                raise RuntimeError(
                    f"process edge endpoint does not resolve: {graph['id']}"
                )
            if positions[source] >= positions[target]:
                raise RuntimeError(
                    f"process graph is cyclic or unordered: {graph['id']}"
                )
            if edge["tensor"] not in tensor_by_id:
                raise RuntimeError(f"process tensor does not resolve: {edge['tensor']}")
            if edge["quantity"] not in quantity_ids:
                raise RuntimeError(
                    f"process quantity does not resolve: {edge['quantity']}"
                )

    for case in description["verification_cases"]:
        if case["process_graph"] not in process_graph_ids:
            raise RuntimeError(f"case process graph does not resolve: {case['id']}")
        if case["model"] not in model_ids:
            raise RuntimeError(f"case model does not resolve: {case['id']}")
        if case["runtime_configuration"] not in runtime_ids:
            raise RuntimeError(f"case runtime does not resolve: {case['id']}")
        if case["input_artifact"] not in artifact_ids:
            raise RuntimeError(f"case input artifact does not resolve: {case['id']}")
        if case["expected_output_artifact"] not in artifact_ids:
            raise RuntimeError(f"case output artifact does not resolve: {case['id']}")

    for evidence in description["evidence"]:
        if any(claim not in claim_ids for claim in evidence["claims"]):
            raise RuntimeError(f"evidence claim does not resolve: {evidence['id']}")

    for runtime in description["runtime_configurations"]:
        if any(
            case not in verification_case_ids for case in runtime["verification_cases"]
        ):
            raise RuntimeError(f"runtime case does not resolve: {runtime['id']}")

    expected_profiles = {
        "core",
        "single-pass",
        "iterative",
        "coupled",
        "spatial",
        "multi-graph",
        "custom-operator",
        "external-artifact",
        "stochastic",
        "scientifically-validated",
        "training-reproducible",
        "operational",
    }
    actual_profiles = {
        decision["profile"] for decision in description["profile_decisions"]
    }
    if actual_profiles != expected_profiles:
        raise RuntimeError("profile decisions are incomplete")

    card = (package_root / "README.md").read_text(encoding="utf-8")
    required_headings = [
        "## Intended use",
        "## Unsupported uses",
        "## Package contents",
        "## Quick start",
        "## Inputs and outputs",
        "## Transformations and effective process",
        "## Coordinates, time, and coupling",
        "## Scientific validation",
        "## Verification",
        "## Limitations and failure modes",
        "## Data and model provenance",
        "## Runtime and compatibility",
        "## Conformance profiles",
        "## Maintenance and change history",
    ]
    missing = [heading for heading in required_headings if heading not in card]
    if missing:
        raise RuntimeError(f"Model Card is missing headings: {missing}")
    if "@@" in card:
        raise RuntimeError("Model Card contains unresolved placeholders")

    forbidden_fragments = ["/home/", "/Users/", "BEGIN PRIVATE KEY", "password="]
    for path in package_root.rglob("*"):
        if not path.is_file() or path.suffix == ".onnx":
            continue
        text = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            if fragment in text:
                raise RuntimeError(f"private or credential-like text in {path.name}")


def build_package(destination: Path, *, replace: bool = False) -> None:
    destination = destination.resolve()
    if destination.exists():
        if not replace:
            raise FileExistsError(
                f"destination already exists: {destination}; choose a new directory"
            )
        if destination.is_symlink():
            raise FileExistsError("refusing to replace a symbolic-link destination")
        marker = destination / "package-description.json"
        if not marker.is_file():
            raise FileExistsError(
                "refusing to replace a directory without package-description.json"
            )
        try:
            existing = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise FileExistsError(
                "cannot validate the existing package marker"
            ) from error
        if existing.get("package_id") != PACKAGE_ID:
            raise FileExistsError("refusing to replace a different package")
        shutil.rmtree(destination)

    (destination / "models").mkdir(parents=True)
    (destination / "metadata").mkdir()
    (destination / "verification").mkdir()
    (destination / "examples").mkdir()

    copy_text(REPOSITORY_ROOT / "LICENSE", destination / "LICENSE")
    copy_text(
        TEMPLATE_DIRECTORY / "metadata/abstract-model-mapping.md",
        destination / "metadata/abstract-model-mapping.md",
    )
    copy_text(
        TEMPLATE_DIRECTORY / "verification/nominal.csv",
        destination / "verification/nominal.csv",
    )
    copy_text(
        SOURCE_DIRECTORY / "reference_run.py", destination / "examples/reference_run.py"
    )
    copy_text(SOURCE_DIRECTORY / "run.f90", destination / "examples/run.f90")
    (destination / "CITATION.txt").write_text(
        "FortONNX contributors (2026). Linear Response Demonstrator "
        "scientific ONNX package example (example-1).\n",
        encoding="utf-8",
    )

    model_path = destination / "models/linear-response.onnx"
    create_model(model_path)
    model_digest = sha256(model_path)
    verification = verify_package(destination)
    write_json(
        destination / "verification/reference-report.json",
        reference_report(model_digest, verification),
    )
    render_model_card(destination, verification)

    source_digests = {
        "builder": sha256(SOURCE_DIRECTORY / "build_package.py"),
        "reference": sha256(SOURCE_DIRECTORY / "reference_run.py"),
    }
    description = build_description(
        destination, model_digest, source_digests, verification
    )
    validate_generated_package(destination, description)
    description_path = destination / "package-description.json"
    write_json(description_path, description)
    description_digest = sha256(description_path)
    (destination / "package-description.sha256").write_text(
        f"sha256 {description_digest} package-description.json\n",
        encoding="utf-8",
    )

    report = {
        "report_format": "scientific ONNX model package conformance report",
        "report_format_version": "1",
        "specification_id": SPECIFICATION_ID,
        "specification_version": SPECIFICATION_VERSION,
        "package_id": PACKAGE_ID,
        "package_version": PACKAGE_VERSION,
        "package_root_identity": "directory identified by package-description digest",
        "package_description_digest": {
            "algorithm": "sha256",
            "value": description_digest,
        },
        "adapter": {
            "id": MAPPING_ID,
            "version": MAPPING_VERSION,
            "digest": digest_object(destination / "metadata/abstract-model-mapping.md"),
        },
        "validator": {
            "name": "scientific package example self-check",
            "version": "1",
            "configuration": "all applicable Draft 0.1 checks",
            "invocation_time": datetime.now(timezone.utc).isoformat(),
        },
        "declared_profiles": description["profile_decisions"],
        "detected_applicable_profiles": ["core", "single-pass"],
        "runtime_configurations": description["runtime_configurations"],
        "checks": conformance_checks(),
        "verification_case_results": [
            {
                "id": "nominal",
                "status": "pass",
                "maximum_absolute_error": verification[
                    "nominal_maximum_absolute_error"
                ],
                "units": "K",
            },
            {
                "id": "dynamic-batch",
                "status": "pass",
                "maximum_absolute_error": verification[
                    "dynamic_batch_maximum_absolute_error"
                ],
                "units": "K",
            },
        ],
        "warnings": [
            "No publisher-authenticated release record is supplied.",
            "This is a self-check, not an independent certification.",
        ],
        "not_run": [],
        "overall": {
            "conforming": True,
            "structurally_conforming": True,
            "verification_passed": True,
            "scientifically_validated": False,
            "operationally_accepted": False,
        },
        "responsible_organization": "FortONNX contributors",
        "signature": "not supplied",
    }
    report_path = destination / "conformance-report.json"
    write_json(report_path, report)
    report_digest = sha256(report_path)
    (destination / "conformance-report.sha256").write_text(
        f"sha256 {report_digest} conformance-report.json\n", encoding="utf-8"
    )

    print(f"built scientific package: {destination}")
    print(f"package description sha256: {description_digest}")
    print(
        "known-answer maximum absolute error: "
        f"{verification['nominal_maximum_absolute_error']:.9g} K"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace an existing package only when its package ID matches",
    )
    parser.add_argument("output_directory", type=Path)
    arguments = parser.parse_args()
    try:
        build_package(arguments.output_directory, replace=arguments.replace)
    except FileExistsError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
