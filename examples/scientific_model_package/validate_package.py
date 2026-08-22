#!/usr/bin/env python3
"""Validate a Scientific ONNX Model Package against the Draft 0.1 contract.

The built-in adapter accepts the direct JSON mapping used by the repository
example. Scientific payload decoders are selected from the format declarations;
this program never imports or executes code from the package.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Callable, Iterable
from urllib.parse import urlparse

import numpy as np
import onnx
from onnx import TensorProto
from onnx.reference import ReferenceEvaluator

VALIDATOR_NAME = "FortONNX Scientific Package Validator"
VALIDATOR_VERSION = "1"
SPECIFICATION_ID = "scientific-onnx-model-package"
SPECIFICATION_VERSION = "0.1-draft"
SUPPORTED_MAPPING = ("org.fortonnx.examples.direct-json", "1")
IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9._-]*$")
SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")

COLLECTIONS = (
    "models",
    "custom_operator_contracts",
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
)

ROOT_FIELDS = (
    "specification_id",
    "specification_version",
    "package_id",
    "package_version",
    "release_status",
    "release_date",
    "model_card",
    "license",
    "citation",
    "primary_model",
    *COLLECTIONS,
    "provenance",
    "profile_decisions",
    "extensions",
)

PROFILES = (
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
)

CHECK_IDS = tuple(
    [f"PKG-{number:03d}" for number in range(1, 12)]
    + [f"ONNX-{number:03d}" for number in range(1, 7)]
    + [f"SEM-{number:03d}" for number in range(1, 9)]
    + [f"FLOW-{number:03d}" for number in range(1, 9)]
    + [f"VER-{number:03d}" for number in range(1, 8)]
    + [f"SCI-{number:03d}" for number in range(1, 11)]
    + [f"CARD-{number:03d}" for number in range(1, 9)]
    + [f"PROF-{number:03d}" for number in range(1, 6)]
)

MODEL_CARD_HEADINGS = (
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
)


class CheckFailure(Exception):
    """A package requirement was not satisfied."""


class CheckNotRun(Exception):
    """An applicable requirement could not be evaluated."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def require_fields(
    value: Any,
    fields: Iterable[str],
    label: str,
    *,
    allow_empty: Iterable[str] = (),
) -> None:
    if not isinstance(value, dict):
        raise CheckFailure(f"{label} must be an object")
    missing = [field for field in fields if field not in value]
    if missing:
        raise CheckFailure(f"{label} is missing fields: {', '.join(missing)}")
    allowed = set(allow_empty)
    empty = [
        field for field in fields if field not in allowed and not nonempty(value[field])
    ]
    if empty:
        raise CheckFailure(f"{label} has empty fields: {', '.join(empty)}")


def is_external_location(location: str) -> bool:
    parsed = urlparse(location)
    return bool(parsed.scheme and parsed.scheme != "file")


def onnx_element_type_name(element_type: int) -> str:
    return f"tensor({TensorProto.DataType.Name(element_type).lower()})"


class Validator:
    def __init__(
        self,
        package_root: Path,
        description_name: str,
        expected_digest: str,
        run_verification: bool,
    ) -> None:
        self.package_root = package_root.resolve()
        self.description_name = description_name
        self.expected_digest = expected_digest.lower()
        self.run_verification = run_verification
        self.description_path: Path | None = None
        self.description: dict[str, Any] | None = None
        self.description_digest = ""
        self.findings: dict[str, dict[str, Any]] = {}
        self.verification_results: list[dict[str, Any]] = []
        self.entities: dict[str, tuple[str, dict[str, Any]]] = {}
        self.artifacts: dict[str, dict[str, Any]] = {}
        self.models: dict[str, dict[str, Any]] = {}
        self.tensors: dict[str, dict[str, Any]] = {}
        self.axes: dict[str, dict[str, Any]] = {}
        self.quantities: dict[str, dict[str, Any]] = {}
        self.domains: dict[str, dict[str, Any]] = {}
        self.transformations: dict[str, dict[str, Any]] = {}
        self.process_graphs: dict[str, dict[str, Any]] = {}
        self.runtimes: dict[str, dict[str, Any]] = {}
        self.onnx_models: dict[str, onnx.ModelProto] = {}
        self.integrity_ok = False
        self.adapter: dict[str, Any] = {
            "id": "unresolved",
            "version": "unresolved",
            "digest": "unresolved",
        }

    def finding(
        self,
        check_id: str,
        status: str,
        message: str,
        *,
        severity: str | None = None,
        entities: Iterable[str] = (),
        artifacts: Iterable[str] = (),
        observations: Iterable[Any] = (),
    ) -> None:
        if check_id in self.findings:
            raise RuntimeError(f"duplicate validator result for {check_id}")
        if severity is None:
            severity = "error" if status in {"fail", "not-run"} else "information"
        self.findings[check_id] = {
            "id": check_id,
            "status": status,
            "severity": severity,
            "message": message,
            "affected_entities": sorted(set(entities)),
            "affected_artifacts": sorted(set(artifacts)),
            "supporting_observations": list(observations),
        }

    def check(
        self,
        check_id: str,
        action: Callable[[], Any],
        pass_message: str,
        *,
        entities: Iterable[str] = (),
        artifacts: Iterable[str] = (),
    ) -> Any:
        try:
            observation = action()
        except CheckNotRun as error:
            self.finding(
                check_id,
                "not-run",
                str(error),
                entities=entities,
                artifacts=artifacts,
            )
            return None
        except (CheckFailure, OSError, ValueError, TypeError, KeyError) as error:
            self.finding(
                check_id,
                "fail",
                str(error),
                entities=entities,
                artifacts=artifacts,
            )
            return None
        observations = [] if observation is None else [observation]
        self.finding(
            check_id,
            "pass",
            pass_message,
            entities=entities,
            artifacts=artifacts,
            observations=observations,
        )
        return observation

    def not_applicable(self, check_id: str, message: str) -> None:
        self.finding(check_id, "not-applicable", message)

    def package_path(self, location: str, *, must_exist: bool = True) -> Path:
        if not isinstance(location, str) or not location:
            raise CheckFailure("artifact location must be a non-empty string")
        if "\\" in location:
            raise CheckFailure(f"backslash is not a package path separator: {location}")
        pure = PurePosixPath(location)
        if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
            raise CheckFailure(f"unsafe package-relative path: {location}")
        candidate = self.package_root.joinpath(*pure.parts)
        if must_exist and not candidate.exists():
            raise CheckFailure(f"missing packaged artifact: {location}")
        resolved = candidate.resolve(strict=must_exist)
        try:
            resolved.relative_to(self.package_root)
        except ValueError as error:
            raise CheckFailure(f"path escapes the package root: {location}") from error
        current = self.package_root
        for part in pure.parts:
            current = current / part
            if current.is_symlink():
                raise CheckFailure(f"symbolic links are not accepted: {location}")
        return candidate

    def load_description(self) -> None:
        def action() -> str:
            if not self.package_root.is_dir():
                raise CheckFailure("package root is not a directory")
            self.description_path = self.package_path(self.description_name)
            if not self.description_path.is_file():
                raise CheckFailure("package description is not a regular file")
            self.description_digest = sha256(self.description_path)
            if not SHA256.fullmatch(self.expected_digest):
                raise CheckFailure("expected description digest is not SHA-256 hex")
            if self.description_digest != self.expected_digest:
                raise CheckFailure(
                    "package-description digest mismatch: "
                    f"expected {self.expected_digest}, observed {self.description_digest}"
                )
            try:
                decoded = json.loads(self.description_path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise CheckFailure(
                    f"cannot decode direct JSON mapping: {error}"
                ) from error
            if not isinstance(decoded, dict):
                raise CheckFailure("package description root must be an object")
            extensions = decoded.get("extensions", [])
            mappings = [
                item
                for item in extensions
                if isinstance(item, dict)
                and item.get("namespace") == "org.fortonnx.examples.serialization"
            ]
            if len(mappings) != 1:
                raise CheckFailure(
                    "exactly one supported serialization mapping is required"
                )
            mapping = mappings[0]
            actual_mapping = (mapping.get("mapping_id"), mapping.get("mapping_version"))
            if actual_mapping != SUPPORTED_MAPPING:
                raise CheckNotRun(
                    "no installed adapter for mapping "
                    f"{actual_mapping[0]!r} version {actual_mapping[1]!r}"
                )
            self.description = decoded
            self.adapter = {
                "id": actual_mapping[0],
                "version": actual_mapping[1],
                "mapping_artifact": mapping.get("mapping_artifact"),
                "digest": "checked with indexed artifacts",
            }
            return (
                f"verified sha256:{self.description_digest} and decoded direct JSON v1"
            )

        self.check(
            "PKG-001",
            action,
            "The trusted package-description digest and declared adapter were verified.",
        )

    @property
    def desc(self) -> dict[str, Any]:
        if self.description is None:
            raise CheckNotRun("package description was not decoded")
        return self.description

    def index_entities(self) -> str:
        if not isinstance(self.desc, dict):
            raise CheckFailure("package must be an object")
        missing_root_fields = [field for field in ROOT_FIELDS if field not in self.desc]
        if missing_root_fields:
            raise CheckFailure(
                "package is missing fields: " + ", ".join(missing_root_fields)
            )
        if self.desc["specification_id"] != SPECIFICATION_ID:
            raise CheckFailure("unsupported specification identifier")
        if self.desc["specification_version"] != SPECIFICATION_VERSION:
            raise CheckFailure("unsupported specification version")
        if self.desc["release_status"] not in {
            "research",
            "validated",
            "operational",
            "deprecated",
        }:
            raise CheckFailure("invalid release_status")
        if not isinstance(self.desc["package_id"], str) or not self.desc["package_id"]:
            raise CheckFailure("package_id must be a persistent non-empty string")
        for collection in COLLECTIONS:
            values = self.desc[collection]
            if not isinstance(values, list):
                raise CheckFailure(f"{collection} must be a list")
            if (
                collection
                in {
                    "models",
                    "artifacts",
                    "tensors",
                    "quantities",
                    "domains",
                    "process_graphs",
                    "verification_cases",
                    "claims",
                    "evidence",
                    "runtime_configurations",
                }
                and not values
            ):
                raise CheckFailure(f"{collection} must not be empty")
            for entity in values:
                require_fields(entity, ("id",), f"{collection} entity")
                entity_id = entity["id"]
                if not isinstance(entity_id, str) or not IDENTIFIER.fullmatch(
                    entity_id
                ):
                    raise CheckFailure(f"invalid entity identifier: {entity_id!r}")
                if entity_id in self.entities:
                    previous = self.entities[entity_id][0]
                    raise CheckFailure(
                        f"duplicate entity identifier {entity_id!r} in {previous} and {collection}"
                    )
                self.entities[entity_id] = (collection, entity)
        provenance = self.desc["provenance"]
        require_fields(provenance, ("entities", "activities", "agents"), "provenance")
        for collection in ("entities", "activities", "agents"):
            if not isinstance(provenance[collection], list):
                raise CheckFailure(f"provenance.{collection} must be a list")
            for entity in provenance[collection]:
                require_fields(entity, ("id",), f"provenance.{collection} entity")
                entity_id = entity["id"]
                if not IDENTIFIER.fullmatch(entity_id) or entity_id in self.entities:
                    raise CheckFailure(
                        f"invalid or duplicate entity identifier: {entity_id}"
                    )
                self.entities[entity_id] = (f"provenance.{collection}", entity)
        self.artifacts = {item["id"]: item for item in self.desc["artifacts"]}
        self.models = {item["id"]: item for item in self.desc["models"]}
        self.tensors = {item["id"]: item for item in self.desc["tensors"]}
        self.axes = {item["id"]: item for item in self.desc["axes"]}
        self.quantities = {item["id"]: item for item in self.desc["quantities"]}
        self.domains = {item["id"]: item for item in self.desc["domains"]}
        self.transformations = {
            item["id"]: item for item in self.desc["transformations"]
        }
        self.process_graphs = {item["id"]: item for item in self.desc["process_graphs"]}
        self.runtimes = {
            item["id"]: item for item in self.desc["runtime_configurations"]
        }
        return f"indexed {len(self.entities)} globally unique entities"

    def package_checks(self) -> None:
        self.check(
            "PKG-002",
            self.index_entities,
            "Root cardinalities, identifiers, enumerations, and entity indexes are valid.",
        )

        def primary_model() -> str:
            primary = self.desc["primary_model"]
            if primary not in self.models:
                raise CheckFailure("primary_model does not resolve")
            declared = [
                model["id"]
                for model in self.models.values()
                if model.get("role") == "primary"
            ]
            if declared != [primary]:
                raise CheckFailure(
                    "exactly one model with role 'primary' must match primary_model"
                )
            return primary

        self.check("PKG-003", primary_model, "Exactly one primary model is declared.")

        def safe_paths() -> str:
            identities: dict[Path, str] = {}
            for artifact in self.artifacts.values():
                location = artifact.get("location")
                if is_external_location(location):
                    continue
                path = self.package_path(location)
                identity = path.resolve()
                if identity in identities:
                    raise CheckFailure(
                        f"conflicting paths {identities[identity]!r} and {location!r}"
                    )
                identities[identity] = location
            return f"normalized {len(identities)} packaged artifact paths"

        self.check(
            "PKG-004", safe_paths, "All packaged paths are safe and unambiguous."
        )

        def artifact_integrity() -> str:
            checked = 0
            for artifact in self.artifacts.values():
                if is_external_location(artifact["location"]):
                    continue
                path = self.package_path(artifact["location"])
                if not path.is_file():
                    raise CheckFailure(
                        f"artifact is not a regular file: {artifact['id']}"
                    )
                if artifact.get("byte_size") != path.stat().st_size:
                    raise CheckFailure(f"byte size mismatch: {artifact['id']}")
                digests = artifact.get("digests")
                if not isinstance(digests, list) or not digests:
                    raise CheckFailure(f"missing digest: {artifact['id']}")
                for digest in digests:
                    if digest.get("algorithm", "").lower() != "sha256":
                        raise CheckNotRun(
                            f"unsupported digest algorithm for {artifact['id']}: "
                            f"{digest.get('algorithm')!r}"
                        )
                    if sha256(path) != str(digest.get("value", "")).lower():
                        raise CheckFailure(f"digest mismatch: {artifact['id']}")
                checked += 1
            self.integrity_ok = True
            mapping_id = self.adapter.get("mapping_artifact")
            if mapping_id in self.artifacts:
                self.adapter["digest"] = self.artifacts[mapping_id]["digests"][0]
            return f"verified size and digest for {checked} packaged artifacts"

        self.check(
            "PKG-005",
            artifact_integrity,
            "All packaged artifact bytes match the index.",
        )

        external = [
            item
            for item in self.artifacts.values()
            if is_external_location(item.get("location", ""))
        ]
        if external:
            self.finding(
                "PKG-006",
                "not-run",
                "external artifact retrieval is disabled; required external bytes were not verified",
                artifacts=[item["id"] for item in external],
            )
        else:
            self.not_applicable(
                "PKG-006", "The package declares no external artifacts."
            )

        def artifact_metadata() -> str:
            fields = (
                "id",
                "role",
                "location",
                "requirement",
                "required_stage",
                "format",
                "logical_type",
                "reader",
                "digests",
                "license",
                "created_by",
            )
            valid_requirements = {"required", "recommended", "optional", "conditional"}
            for artifact in self.artifacts.values():
                require_fields(artifact, fields, f"artifact {artifact.get('id', '?')}")
                require_fields(
                    artifact["format"],
                    ("name", "version"),
                    f"artifact {artifact['id']} format",
                )
                require_fields(
                    artifact["reader"],
                    ("name", "version"),
                    f"artifact {artifact['id']} reader",
                )
                if artifact["requirement"] not in valid_requirements:
                    raise CheckFailure(f"invalid requirement level: {artifact['id']}")
                if artifact["created_by"] not in self.entities:
                    raise CheckFailure(
                        f"creation provenance does not resolve: {artifact['id']}"
                    )
            return f"checked {len(self.artifacts)} artifact declarations"

        self.check(
            "PKG-007",
            artifact_metadata,
            "Artifact formats, readers, roles, licenses, and provenance are declared.",
        )

        def format_safety() -> str:
            for artifact in self.artifacts.values():
                if artifact["logical_type"] == "executable":
                    require_fields(
                        artifact,
                        ("executable_risk",),
                        f"executable artifact {artifact['id']}",
                    )
                    require_fields(
                        artifact["executable_risk"],
                        ("trust", "permissions", "isolation"),
                        f"executable risk {artifact['id']}",
                    )
                if not nonempty(artifact["format"].get("version")):
                    raise CheckFailure(f"format version is missing: {artifact['id']}")
            return "all executable decoders are explicit and all formats are versioned"

        self.check(
            "PKG-008",
            format_safety,
            "Format specifications and executable risks are explicit.",
        )

        def reachability() -> str:
            model_card = self.desc["model_card"]
            if model_card not in self.artifacts:
                raise CheckFailure("Model Card artifact does not resolve")
            license_refs = [item.get("artifact") for item in self.desc["license"]]
            if not license_refs or any(
                item not in self.artifacts for item in license_refs
            ):
                raise CheckFailure("license artifact reference does not resolve")
            citation = self.desc["citation"]
            if (
                not isinstance(citation, dict)
                or citation.get("artifact") not in self.artifacts
            ):
                raise CheckFailure("citation artifact reference does not resolve")
            primary = self.models[self.desc["primary_model"]]
            if primary.get("artifact") not in self.artifacts:
                raise CheckFailure("primary ONNX artifact does not resolve")
            for case in self.desc["verification_cases"]:
                for field in ("input_artifact", "expected_output_artifact"):
                    if case.get(field) not in self.artifacts:
                        raise CheckFailure(f"{case['id']}.{field} does not resolve")
            return "Model Card, license, citation, graph, and verification artifacts resolve"

        self.check(
            "PKG-009",
            reachability,
            "Required package entry points are reachable from the description.",
        )

        def content_safety() -> str:
            forbidden = tuple(
                fragment.lower()
                for fragment in (
                    b"/home/",
                    b"/Users/",
                    b"BEGIN PRIVATE KEY",
                    b"password=",
                    b"passwd=",
                )
            )
            overlap = max(len(fragment) for fragment in forbidden) - 1
            for artifact in self.artifacts.values():
                if is_external_location(artifact["location"]):
                    continue
                path = self.package_path(artifact["location"])
                media_type = str(artifact.get("media_type", "")).lower()
                textual = media_type.startswith("text/") or any(
                    token in media_type for token in ("json", "yaml", "xml", "toml")
                )
                if textual:
                    tail = b""
                    with path.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            probe = (tail + block).lower()
                            for fragment in forbidden:
                                if fragment in probe:
                                    raise CheckFailure(
                                        "private-path or credential-like content in "
                                        f"{artifact['id']}"
                                    )
                            tail = probe[-overlap:]
                suffix = PurePosixPath(artifact["location"]).suffix.lower()
                executable_suffix = suffix in {
                    ".py",
                    ".sh",
                    ".so",
                    ".dll",
                    ".dylib",
                    ".exe",
                }
                declared = (
                    "executable" in artifact.get("role", [])
                    or artifact.get("logical_type") == "executable"
                )
                if executable_suffix and not declared:
                    raise CheckFailure(
                        f"undeclared executable artifact: {artifact['id']}"
                    )
            return "no forbidden private paths, credentials, or undeclared executables found"

        self.check(
            "PKG-010",
            content_safety,
            "Required package content passed the private-data and executable scan.",
        )

        authenticity = self.desc.get("authenticity")
        if authenticity:
            self.finding(
                "PKG-011",
                "not-run",
                "authenticity mechanism is declared but this validator has no configured trust store",
            )
        else:
            self.not_applicable(
                "PKG-011", "No authenticity record or signature is supplied."
            )
        if authenticity:
            self.finding("PKG-R01", "pass", "An authenticity mechanism is declared.")
        else:
            self.finding(
                "PKG-R01",
                "fail",
                "No publisher-authenticated release record is supplied.",
                severity="warning",
                observations=(
                    "The caller supplied a trusted digest for identity checking; publisher authenticity remains unestablished.",
                ),
            )

    def model_artifact(self, model: dict[str, Any]) -> tuple[dict[str, Any], Path]:
        artifact_id = model.get("artifact")
        if artifact_id not in self.artifacts:
            raise CheckFailure(f"model artifact does not resolve: {model.get('id')}")
        artifact = self.artifacts[artifact_id]
        if is_external_location(artifact["location"]):
            raise CheckNotRun(f"model graph is external: {model['id']}")
        return artifact, self.package_path(artifact["location"])

    def onnx_checks(self) -> None:
        def parse_graphs() -> str:
            if not self.integrity_ok:
                raise CheckNotRun(
                    "artifact integrity is unresolved; ONNX parsing was suppressed"
                )
            for model in self.models.values():
                _, path = self.model_artifact(model)
                graph = onnx.load_model(path, load_external_data=False)
                onnx.checker.check_model(graph)
                self.onnx_models[model["id"]] = graph
            return (
                f"parsed and structurally checked {len(self.onnx_models)} ONNX graph(s)"
            )

        self.check(
            "ONNX-001",
            parse_graphs,
            "All required ONNX graphs passed the structural checker.",
        )

        def graph_identity() -> str:
            if len(self.onnx_models) != len(self.models):
                raise CheckNotRun("one or more ONNX graphs were not parsed")
            for model_id, declaration in self.models.items():
                graph = self.onnx_models[model_id]
                if graph.ir_version != declaration.get("onnx_ir_version"):
                    raise CheckFailure(f"ONNX IR version mismatch: {model_id}")
                actual_opsets = [
                    {"domain": item.domain, "version": item.version}
                    for item in graph.opset_import
                ]
                if actual_opsets != declaration.get("opset_imports"):
                    raise CheckFailure(f"opset imports mismatch: {model_id}")
                external_locations = []
                for initializer in graph.graph.initializer:
                    if initializer.data_location == TensorProto.EXTERNAL:
                        values = {
                            item.key: item.value for item in initializer.external_data
                        }
                        if "location" in values:
                            external_locations.append(values["location"])
                if len(external_locations) != len(declaration.get("external_data", [])):
                    raise CheckFailure(
                        f"external-data declaration mismatch: {model_id}"
                    )
                imported_custom = {
                    node.domain
                    for node in graph.graph.node
                    if node.domain not in {"", "ai.onnx", "ai.onnx.ml"}
                }
                if bool(imported_custom) != bool(
                    declaration.get("custom_operators", [])
                ):
                    raise CheckFailure(
                        f"custom-operator declaration mismatch: {model_id}"
                    )
            return "IR versions, opsets, external data, and custom imports match"

        self.check(
            "ONNX-002",
            graph_identity,
            "Declared ONNX identities match the graph bytes.",
        )

        def signatures() -> str:
            symbolic = {item["id"]: item for item in self.desc["symbolic_dimensions"]}
            for model_id, declaration in self.models.items():
                if model_id not in self.onnx_models:
                    raise CheckNotRun(f"ONNX graph was not parsed: {model_id}")
                graph = self.onnx_models[model_id]
                actual = {
                    "input": list(graph.graph.input),
                    "output": list(graph.graph.output),
                }
                for direction, references in (
                    ("input", declaration.get("inputs", [])),
                    ("output", declaration.get("outputs", [])),
                ):
                    contracts = []
                    for reference in references:
                        if reference not in self.tensors:
                            raise CheckFailure(
                                f"tensor reference does not resolve: {reference}"
                            )
                        contracts.append(self.tensors[reference])
                    values = actual[direction]
                    if len(values) != len(contracts):
                        raise CheckFailure(f"{direction} count mismatch: {model_id}")
                    for value, contract in zip(values, contracts, strict=True):
                        if (
                            contract.get("model") != model_id
                            or contract.get("direction") != direction
                        ):
                            raise CheckFailure(
                                f"tensor ownership/direction mismatch: {contract['id']}"
                            )
                        if value.name != contract.get("onnx_name"):
                            raise CheckFailure(
                                f"ONNX value name mismatch: {contract['id']}"
                            )
                        tensor_type = value.type.tensor_type
                        if onnx_element_type_name(
                            tensor_type.elem_type
                        ) != contract.get("element_type"):
                            raise CheckFailure(
                                f"element type mismatch: {contract['id']}"
                            )
                        dimensions = list(tensor_type.shape.dim)
                        if len(dimensions) != contract.get("rank"):
                            raise CheckFailure(f"rank mismatch: {contract['id']}")
                        for declared, dimension in zip(
                            contract.get("shape", []), dimensions, strict=True
                        ):
                            if isinstance(declared, int):
                                if dimension.dim_value != declared:
                                    raise CheckFailure(
                                        f"fixed dimension mismatch: {contract['id']}"
                                    )
                            elif declared in symbolic:
                                permitted = symbolic[declared].get("onnx_symbols", [])
                                if dimension.dim_param not in permitted:
                                    raise CheckFailure(
                                        f"symbolic dimension mismatch: {contract['id']}"
                                    )
                            else:
                                raise CheckFailure(
                                    f"undeclared symbolic dimension: {declared}"
                                )
            return "all graph interface values exactly match tensor contracts"

        self.check(
            "ONNX-003",
            signatures,
            "ONNX names, directions, types, ranks, and dimensions match.",
        )

        external_refs = [
            ref
            for model in self.models.values()
            for ref in model.get("external_data", [])
        ]
        if external_refs:
            self.finding(
                "ONNX-004",
                "not-run",
                "ONNX external-data loading is not implemented by this validator",
                artifacts=external_refs,
            )
        else:
            self.not_applicable("ONNX-004", "No ONNX external data is declared.")

        custom_refs = [
            ref
            for model in self.models.values()
            for ref in model.get("custom_operators", [])
        ]
        if custom_refs:
            self.finding(
                "ONNX-005",
                "not-run",
                "custom-operator implementation loading is not enabled",
                entities=custom_refs,
            )
        else:
            self.not_applicable("ONNX-005", "No custom operators are declared.")

        def runtime_load() -> str:
            if len(self.onnx_models) != len(self.models):
                raise CheckNotRun("ONNX graph identity is unresolved")
            loaded = 0
            for runtime in self.runtimes.values():
                if "ReferenceEvaluator" not in str(runtime.get("runtime", "")):
                    raise CheckNotRun(
                        f"runtime loader is unavailable for {runtime.get('runtime')!r}"
                    )
                for model in self.onnx_models.values():
                    ReferenceEvaluator(model)
                    loaded += 1
            return f"loaded {loaded} graph/runtime combinations without fallback"

        self.check(
            "ONNX-006",
            runtime_load,
            "Every claimed runtime configuration loaded its required graph.",
        )

    def semantic_checks(self) -> None:
        def interface_resolution() -> str:
            for model_id, model in self.models.items():
                referenced = model.get("inputs", []) + model.get("outputs", [])
                if len(referenced) != len(set(referenced)):
                    raise CheckFailure(
                        f"duplicate interface tensor reference: {model_id}"
                    )
                owned = [
                    item["id"]
                    for item in self.tensors.values()
                    if item.get("model") == model_id
                ]
                if set(referenced) != set(owned):
                    raise CheckFailure(
                        f"tensor contracts do not exactly cover interface: {model_id}"
                    )
            return f"resolved {len(self.tensors)} tensor contracts exactly once"

        self.check(
            "SEM-001",
            interface_resolution,
            "Every graph input and output has exactly one tensor contract.",
        )

        def shape_axes() -> str:
            symbolic_ids = {item["id"] for item in self.desc["symbolic_dimensions"]}
            for tensor in self.tensors.values():
                rank = tensor.get("rank")
                shape = tensor.get("shape")
                axes = tensor.get("axes")
                if not isinstance(rank, int) or rank < 0:
                    raise CheckFailure(f"invalid rank: {tensor['id']}")
                if len(shape) != rank or len(axes) != rank:
                    raise CheckFailure(f"shape/axis/rank mismatch: {tensor['id']}")
                if any(axis not in self.axes for axis in axes):
                    raise CheckFailure(
                        f"axis reference does not resolve: {tensor['id']}"
                    )
                for dimension in shape:
                    if isinstance(dimension, int) and dimension <= 0:
                        raise CheckFailure(
                            f"non-positive fixed dimension: {tensor['id']}"
                        )
                    if isinstance(dimension, str) and dimension not in symbolic_ids:
                        raise CheckFailure(
                            f"symbolic dimension does not resolve: {dimension}"
                        )
            return "tensor ranks, shapes, axes, and symbolic dimensions are consistent"

        self.check(
            "SEM-002",
            shape_axes,
            "Shape, axis, and symbolic-dimension contracts are consistent.",
        )

        def axis_channel_meaning() -> str:
            axis_fields = (
                "id",
                "meaning",
                "index_origin",
                "length",
                "orientation",
                "ordering",
                "periodicity",
                "boundary_behavior",
                "location",
                "missing_or_padding",
            )
            for axis in self.axes.values():
                require_fields(axis, axis_fields, f"axis {axis.get('id', '?')}")
            for tensor in self.tensors.values():
                channels = tensor.get("channels")
                if not isinstance(channels, list) or not channels:
                    raise CheckFailure(f"tensor channels are missing: {tensor['id']}")
                for channel in channels:
                    require_fields(
                        channel,
                        (
                            "quantity",
                            "selector",
                            "physical_units",
                            "graph_units",
                            "valid_range",
                            "location",
                            "sign_convention",
                            "missing_value_behavior",
                            "transformations",
                        ),
                        f"channel in {tensor['id']}",
                        allow_empty=("transformations",),
                    )
                    if channel["quantity"] not in self.quantities:
                        raise CheckFailure(
                            f"quantity does not resolve: {channel['quantity']}"
                        )
            return "all axes and channels have ordered scientific meaning"

        self.check(
            "SEM-003",
            axis_channel_meaning,
            "Axis and channel scientific semantics are complete.",
        )

        def units() -> str:
            for tensor in self.tensors.values():
                require_fields(
                    tensor, ("graph_units", "physical_units"), f"tensor {tensor['id']}"
                )
                if (
                    tensor["graph_units"] == "mixed"
                    or tensor["physical_units"] == "mixed"
                ):
                    for channel in tensor["channels"]:
                        if not nonempty(channel.get("graph_units")) or not nonempty(
                            channel.get("physical_units")
                        ):
                            raise CheckFailure(
                                f"mixed-unit channel lacks units: {tensor['id']}"
                            )
            return "physical and graph-boundary units are explicit"

        self.check("SEM-004", units, "Physical and graph-boundary units are explicit.")

        def invalid_values() -> str:
            for tensor in self.tensors.values():
                require_fields(
                    tensor,
                    ("invalid_value_behavior", "missing_value_behavior"),
                    f"tensor {tensor['id']}",
                )
                for channel in tensor["channels"]:
                    if not nonempty(channel.get("missing_value_behavior")):
                        raise CheckFailure(
                            f"channel missing-value behavior absent: {tensor['id']}"
                        )
            return "invalid, missing, NaN, infinity, clipping, and padding behavior is declared"

        self.check(
            "SEM-005", invalid_values, "Invalid and missing value behavior is explicit."
        )

        def domain_separation() -> str:
            domain_fields = (
                "id",
                "kind",
                "definition",
                "constraints",
                "selection_rule",
            )
            for domain in self.domains.values():
                require_fields(domain, domain_fields, f"domain {domain.get('id', '?')}")
            for tensor in self.tensors.values():
                references = [
                    tensor.get("admissible_domain"),
                    tensor.get("training_domain"),
                    tensor.get("evaluation_domain"),
                    tensor.get("intended_use_domain"),
                ]
                if any(reference not in self.domains for reference in references):
                    raise CheckFailure(
                        f"domain reference does not resolve: {tensor['id']}"
                    )
            return "all four domain roles are explicitly referenced and reproducibly defined"

        self.check(
            "SEM-006",
            domain_separation,
            "Admissible, training, evaluation, and intended-use domains are explicit.",
        )

        spatial_axes = [
            axis
            for axis in self.axes.values()
            if any(
                word in str(axis.get("location", "")).lower()
                for word in ("spatial", "cell", "face", "edge", "node", "grid", "mesh")
            )
            and "non-spatial" not in str(axis.get("location", "")).lower()
        ]
        temporal_axes = [
            axis
            for axis in self.axes.values()
            if "time" in str(axis.get("meaning", "")).lower()
        ]
        if spatial_axes or temporal_axes:

            def coordinates() -> str:
                if not self.desc["coordinate_systems"]:
                    raise CheckFailure(
                        "spatial or temporal tensors require coordinate systems"
                    )
                for coordinate in self.desc["coordinate_systems"]:
                    require_fields(coordinate, ("id",), "coordinate system")
                return (
                    "spatial and temporal axes resolve to declared coordinate systems"
                )

            self.check(
                "SEM-007", coordinates, "Coordinate and time semantics are complete."
            )
        else:
            self.not_applicable(
                "SEM-007", "The package declares no spatial or temporal tensor."
            )

        def interface_consistency() -> str:
            card = self.model_card_text()
            examples = self.example_texts()
            for tensor in self.tensors.values():
                name = tensor["onnx_name"]
                if name not in card:
                    raise CheckFailure(f"Model Card omits ONNX name: {name}")
                if examples and not any(name in text for text in examples):
                    raise CheckFailure(f"execution examples omit ONNX name: {name}")
                for units_value in (tensor["graph_units"], tensor["physical_units"]):
                    if units_value not in card:
                        raise CheckFailure(
                            f"Model Card omits tensor units: {tensor['id']}"
                        )
            return "machine-readable tensor semantics agree with card and examples"

        self.check(
            "SEM-008",
            interface_consistency,
            "Interface semantics agree across metadata, Model Card, and examples.",
        )

    def model_card_text(self) -> str:
        artifact_id = self.desc["model_card"]
        if artifact_id not in self.artifacts:
            raise CheckFailure("Model Card artifact does not resolve")
        artifact = self.artifacts[artifact_id]
        return self.package_path(artifact["location"]).read_text(encoding="utf-8")

    def example_texts(self) -> list[str]:
        texts = []
        for artifact in self.artifacts.values():
            roles = artifact.get("role", [])
            if "usage example" not in roles and "executable" not in roles:
                continue
            if is_external_location(artifact["location"]):
                continue
            try:
                texts.append(
                    self.package_path(artifact["location"]).read_text(encoding="utf-8")
                )
            except UnicodeDecodeError:
                continue
        return texts

    def flow_checks(self) -> None:
        def process_dags() -> str:
            for graph in self.process_graphs.values():
                require_fields(
                    graph,
                    ("nodes", "edges", "execution_order", "purpose"),
                    f"process graph {graph['id']}",
                )
                nodes = graph["nodes"]
                node_ids = [node.get("id") for node in nodes]
                if len(node_ids) != len(set(node_ids)) or set(node_ids) != set(
                    graph["execution_order"]
                ):
                    raise CheckFailure(f"invalid execution order: {graph['id']}")
                positions = {
                    node: index for index, node in enumerate(graph["execution_order"])
                }
                for node in nodes:
                    require_fields(
                        node,
                        ("id", "kind", "inputs", "outputs", "failure_behavior"),
                        f"process node in {graph['id']}",
                        allow_empty=("inputs", "outputs"),
                    )
                for edge in graph["edges"]:
                    require_fields(
                        edge, ("from", "to"), f"process edge in {graph['id']}"
                    )
                    source = edge["from"].split(".", 1)[0]
                    target = edge["to"].split(".", 1)[0]
                    if source not in positions or target not in positions:
                        raise CheckFailure(
                            f"edge endpoint does not resolve: {graph['id']}"
                        )
                    if positions[source] >= positions[target]:
                        raise CheckFailure(
                            f"cycle or contradictory order: {graph['id']}"
                        )
            return f"validated {len(self.process_graphs)} acyclic process graph(s)"

        self.check(
            "FLOW-001",
            process_dags,
            "Each invocation process graph is acyclic and ordered.",
        )

        def complete_paths() -> str:
            def reachable(start: str, adjacency: dict[str, set[str]]) -> set[str]:
                visited: set[str] = set()
                pending = [start]
                while pending:
                    node = pending.pop()
                    for target in adjacency.get(node, set()):
                        if target not in visited:
                            visited.add(target)
                            pending.append(target)
                return visited

            for graph in self.process_graphs.values():
                nodes = {node["id"]: node for node in graph["nodes"]}
                adjacency = {node_id: set() for node_id in nodes}
                for edge in graph["edges"]:
                    source = edge["from"].split(".", 1)[0]
                    target = edge["to"].split(".", 1)[0]
                    adjacency[source].add(target)
                inputs = {
                    node_id
                    for node_id, node in nodes.items()
                    if "input boundary" in str(node.get("kind", "")).lower()
                }
                inference = {
                    node_id
                    for node_id, node in nodes.items()
                    if "onnx inference" in str(node.get("kind", "")).lower()
                }
                outputs = {
                    node_id
                    for node_id, node in nodes.items()
                    if "output boundary" in str(node.get("kind", "")).lower()
                }
                if not inputs or not inference or not outputs:
                    raise CheckFailure(
                        f"physical boundary or inference node is missing: {graph['id']}"
                    )
                if any(not (reachable(node, adjacency) & inference) for node in inputs):
                    raise CheckFailure(
                        f"an input boundary does not reach ONNX inference: {graph['id']}"
                    )
                if any(
                    not (reachable(node, adjacency) & outputs) for node in inference
                ):
                    raise CheckFailure(
                        f"an ONNX inference node does not reach an output boundary: {graph['id']}"
                    )
                reached_from_inference = set().union(
                    *(reachable(node, adjacency) for node in inference)
                )
                if not outputs <= reached_from_inference:
                    raise CheckFailure(
                        f"an output boundary is disconnected from ONNX inference: {graph['id']}"
                    )
            return "every physical input reaches inference and every inference result reaches each required output"

        self.check(
            "FLOW-002",
            complete_paths,
            "Required physical inputs and outputs are connected.",
        )

        def edge_contracts() -> str:
            required = ("tensor", "quantity", "units", "axes", "shape", "state_version")
            for graph in self.process_graphs.values():
                for edge in graph["edges"]:
                    require_fields(
                        edge, required, f"edge {edge.get('from')} -> {edge.get('to')}"
                    )
                    if edge["tensor"] not in self.tensors:
                        raise CheckFailure(
                            f"edge tensor does not resolve: {edge['tensor']}"
                        )
                    if edge["quantity"] not in self.quantities:
                        raise CheckFailure(
                            f"edge quantity does not resolve: {edge['quantity']}"
                        )
                    tensor = self.tensors[edge["tensor"]]
                    if (
                        edge["axes"] != tensor["axes"]
                        or edge["shape"] != tensor["shape"]
                    ):
                        raise CheckFailure(
                            f"edge shape or axes conflict: {edge['tensor']}"
                        )
                    allowed_units = {tensor["graph_units"], tensor["physical_units"]}
                    if edge["units"] not in allowed_units:
                        raise CheckFailure(f"edge units conflict: {edge['tensor']}")
            return (
                "edge types, shapes, axes, units, quantities, and states are compatible"
            )

        self.check(
            "FLOW-003",
            edge_contracts,
            "Every process edge carries a complete compatible contract.",
        )

        def external_operations() -> str:
            fields = (
                "id",
                "operation",
                "location",
                "inputs",
                "outputs",
                "definition",
                "parameters",
                "preconditions",
                "postconditions",
                "implementation",
                "numerical_behavior",
            )
            for transform in self.transformations.values():
                require_fields(
                    transform,
                    fields,
                    f"transformation {transform.get('id', '?')}",
                    allow_empty=(
                        "parameters",
                        "preconditions",
                        "postconditions",
                        "implementation",
                    ),
                )
            return f"checked {len(self.transformations)} transformation definitions"

        self.check(
            "FLOW-004",
            external_operations,
            "Transformations have algorithms, ordered interfaces, conditions, and failure behavior.",
        )

        def exactly_once() -> str:
            for tensor in self.tensors.values():
                sequence = tensor.get("preprocessing", []) + tensor.get(
                    "postprocessing", []
                )
                if len(sequence) != len(set(sequence)):
                    raise CheckFailure(
                        f"duplicate transform on tensor path: {tensor['id']}"
                    )
                for reference in sequence:
                    if reference not in self.transformations:
                        raise CheckFailure(
                            f"transform reference does not resolve: {reference}"
                        )
            return "no tensor path duplicates a declared transformation"

        self.check(
            "FLOW-005",
            exactly_once,
            "Required transformations occur exactly once on each tensor path.",
        )

        def parameters() -> str:
            for transform in self.transformations.values():
                for parameter in transform.get("parameters", []):
                    require_fields(
                        parameter, ("name",), f"parameter in {transform['id']}"
                    )
                    if "value" not in parameter and "artifact" not in parameter:
                        raise CheckFailure(
                            f"parameter has neither literal nor artifact: {transform['id']}"
                        )
                    if (
                        "artifact" in parameter
                        and parameter["artifact"] not in self.artifacts
                    ):
                        raise CheckFailure(
                            f"parameter artifact does not resolve: {transform['id']}"
                        )
                    if (
                        "value" in parameter
                        and isinstance(parameter["value"], float)
                        and not math.isfinite(parameter["value"])
                    ):
                        raise CheckFailure(
                            f"non-finite literal parameter: {transform['id']}"
                        )
            return "all parameters resolve to finite literals or indexed artifacts"

        self.check(
            "FLOW-006", parameters, "Transformation parameters are fully resolved."
        )

        fitted = [
            item for item in self.transformations.values() if item.get("fitted_on")
        ]
        if fitted:

            def fitted_provenance() -> str:
                for transform in fitted:
                    if (
                        transform["fitted_on"] not in self.domains
                        and transform["fitted_on"] not in self.entities
                    ):
                        raise CheckFailure(
                            f"fitted_on does not resolve: {transform['id']}"
                        )
                return "all fitted parameters resolve to declared training partitions"

            self.check(
                "FLOW-007",
                fitted_provenance,
                "Fitted-parameter training partitions are explicit.",
            )
        else:
            self.not_applicable(
                "FLOW-007", "No fitted transformation parameters are declared."
            )

        def example_flow() -> str:
            card = self.model_card_text()
            examples = self.example_texts()
            for transform in self.transformations.values():
                if transform["id"] not in card and transform["operation"] not in card:
                    raise CheckFailure(
                        f"Model Card omits transformation: {transform['id']}"
                    )
            for tensor in self.tensors.values():
                if examples and not any(
                    tensor["onnx_name"] in text for text in examples
                ):
                    raise CheckFailure(
                        f"examples omit tensor mapping: {tensor['onnx_name']}"
                    )
            return "examples expose exact tensor mappings and card exposes every transformation"

        self.check(
            "FLOW-008",
            example_flow,
            "Examples and documentation match the effective process graph.",
        )

    def verification_checks(self) -> None:
        cases = self.desc["verification_cases"]

        def case_identity() -> str:
            if not cases:
                raise CheckFailure("at least one verification case is required")
            nominal = False
            for case in cases:
                require_fields(
                    case,
                    (
                        "id",
                        "purpose",
                        "process_graph",
                        "model",
                        "model_digest",
                        "runtime_configuration",
                        "input_artifact",
                        "expected_output_artifact",
                        "input_locator",
                        "expected_output_locator",
                        "comparison",
                        "dtype",
                        "shape",
                        "axes",
                        "units",
                        "value_space",
                        "preprocessing",
                        "postprocessing",
                        "nondeterministic_variation",
                    ),
                    f"verification case {case.get('id', '?')}",
                    allow_empty=("preprocessing", "postprocessing"),
                )
                if (
                    case["model"] not in self.models
                    or case["process_graph"] not in self.process_graphs
                    or case["runtime_configuration"] not in self.runtimes
                ):
                    raise CheckFailure(
                        f"verification references do not resolve: {case['id']}"
                    )
                if (
                    "nominal" in case["id"].lower()
                    or "nonzero" in case["purpose"].lower()
                ):
                    nominal = True
                model_artifact, _ = self.model_artifact(self.models[case["model"]])
                if case["model_digest"] not in model_artifact["digests"]:
                    raise CheckFailure(
                        f"verification model digest mismatch: {case['id']}"
                    )
            if not nominal:
                raise CheckFailure("no nominal nonzero verification case is identified")
            return f"checked identity and scope of {len(cases)} verification cases"

        self.check(
            "VER-001",
            case_identity,
            "At least one nominal case identifies exact model, process, artifacts, and runtime.",
        )

        def locators() -> str:
            for case in cases:
                if case["dtype"] != "float32":
                    raise CheckNotRun(
                        f"no installed verification decoder for dtype {case['dtype']!r}"
                    )
                artifact = self.artifacts[case["input_artifact"]]
                if artifact["format"].get("name") not in {
                    "CSV",
                    "Comma-separated values",
                }:
                    raise CheckNotRun(
                        f"no installed verification decoder for {artifact['format'].get('name')!r}"
                    )
                require_fields(
                    case["input_locator"],
                    ("columns", "rows"),
                    f"input locator {case['id']}",
                )
                require_fields(
                    case["expected_output_locator"],
                    ("column", "rows"),
                    f"output locator {case['id']}",
                )
                if len(case["shape"]) != len(case["axes"]):
                    raise CheckFailure(f"case shape/axes mismatch: {case['id']}")
            return "CSV v1 locators, float32 dtype, shapes, axes, units, and value spaces are decodable"

        self.check(
            "VER-002",
            locators,
            "Known-answer values and mappings are explicitly decodable.",
        )

        if not self.run_verification:
            for check_id, reason in (
                (
                    "VER-003",
                    "executable verification was not requested; use --run-verification",
                ),
                (
                    "VER-004",
                    "outputs were not executed and compared; use --run-verification",
                ),
                (
                    "VER-005",
                    "observed errors were not recomputed; use --run-verification",
                ),
                (
                    "VER-006",
                    "additional applicable cases were not executed; use --run-verification",
                ),
                (
                    "VER-007",
                    "the mapping-sensitive negative control was not executed; use --run-verification",
                ),
            ):
                self.finding(check_id, "not-run", reason)
            return

        execution = self.execute_cases()
        if execution is None:
            for check_id in ("VER-004", "VER-005", "VER-006", "VER-007"):
                self.finding(
                    check_id, "not-run", "verification execution did not complete"
                )
            return
        self.finding(
            "VER-003",
            "pass",
            "The complete supported process was executed with validator-owned decoders and ONNX evaluator.",
            observations=(f"executed {len(self.verification_results)} declared cases",),
        )
        self.finding(
            "VER-004",
            "pass",
            "Every output was compared with its declared absolute and relative tolerances.",
        )
        self.finding(
            "VER-005",
            "pass",
            "Per-output observed errors and pass/fail results were recorded.",
        )
        dynamic = any("dynamic" in case["id"].lower() for case in cases)
        if any(self.desc["symbolic_dimensions"]):
            if dynamic:
                self.finding(
                    "VER-006", "pass", "The declared dynamic-shape case was executed."
                )
            else:
                self.finding(
                    "VER-006",
                    "fail",
                    "A dynamic dimension is declared but no dynamic-shape case exists.",
                )
        else:
            self.not_applicable(
                "VER-006", "No additional dynamic or special behavior is applicable."
            )
        if execution["negative_control_detected"]:
            self.finding(
                "VER-007",
                "pass",
                "A same-shaped input swap was rejected by the declared comparison.",
                observations=(
                    {"minimum_detected_error": execution["negative_control_error"]},
                ),
            )
        else:
            self.finding(
                "VER-007",
                "fail",
                "The mapping-sensitive input-swap control was not detected.",
            )

    def execute_cases(self) -> dict[str, Any] | None:
        def action() -> dict[str, Any]:
            if not self.integrity_ok or len(self.onnx_models) != len(self.models):
                raise CheckNotRun("artifact or graph identity is unresolved")
            for transform in self.transformations.values():
                if (
                    transform["location"] == "external"
                    and transform["operation"] != "finite and range validation"
                ):
                    raise CheckNotRun(
                        f"no trusted validator implementation for external operation {transform['operation']!r}"
                    )
            negative_error = 0.0
            negative_detected = False
            for case in self.desc["verification_cases"]:
                model_decl = self.models[case["model"]]
                model = self.onnx_models[case["model"]]
                evaluator = ReferenceEvaluator(model)
                artifact = self.artifacts[case["input_artifact"]]
                table_path = self.package_path(artifact["location"])
                with table_path.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.reader(stream))
                data_rows = rows[1:]
                input_rows = [
                    data_rows[index - 1] for index in case["input_locator"]["rows"]
                ]
                output_rows = [
                    data_rows[index - 1]
                    for index in case["expected_output_locator"]["rows"]
                ]
                arrays = []
                for column in case["input_locator"]["columns"]:
                    array = np.asarray(
                        [[float(row[column])] for row in input_rows], dtype=np.float32
                    )
                    if not np.all(np.isfinite(array)):
                        raise CheckFailure(
                            f"non-finite verification input: {case['id']}"
                        )
                    arrays.append(array)
                expected = np.asarray(
                    [
                        [float(row[case["expected_output_locator"]["column"]])]
                        for row in output_rows
                    ],
                    dtype=np.float32,
                )
                input_contracts = [self.tensors[item] for item in model_decl["inputs"]]
                feeds = {
                    contract["onnx_name"]: array
                    for contract, array in zip(input_contracts, arrays, strict=True)
                }
                output_names = [
                    self.tensors[item]["onnx_name"] for item in model_decl["outputs"]
                ]
                observed_values = evaluator.run(output_names, feeds)
                if len(observed_values) != 1:
                    raise CheckNotRun(
                        "the installed CSV decoder supports one output per case"
                    )
                observed = np.asarray(observed_values[0])
                if observed.shape != expected.shape or not np.all(
                    np.isfinite(observed)
                ):
                    raise CheckFailure(f"invalid observed output: {case['id']}")
                absolute = np.abs(observed - expected)
                tolerance = float(case["comparison"]["absolute_tolerance"]) + float(
                    case["comparison"]["relative_tolerance"]
                ) * np.abs(expected)
                passed = bool(np.all(absolute <= tolerance))
                maximum = float(np.max(absolute))
                self.verification_results.append(
                    {
                        "id": case["id"],
                        "status": "pass" if passed else "fail",
                        "outputs": [
                            {
                                "onnx_name": output_names[0],
                                "maximum_absolute_error": maximum,
                                "absolute_tolerance": case["comparison"][
                                    "absolute_tolerance"
                                ],
                                "relative_tolerance": case["comparison"][
                                    "relative_tolerance"
                                ],
                                "units": case["units"]["output"],
                                "status": "pass" if passed else "fail",
                            }
                        ],
                    }
                )
                if not passed:
                    raise CheckFailure(f"known-answer comparison failed: {case['id']}")
                if (
                    not negative_detected
                    and len(arrays) >= 2
                    and arrays[0].shape == arrays[1].shape
                ):
                    swapped = dict(feeds)
                    swapped[input_contracts[0]["onnx_name"]] = arrays[1]
                    swapped[input_contracts[1]["onnx_name"]] = arrays[0]
                    perturbed = np.asarray(evaluator.run(output_names, swapped)[0])
                    error = np.abs(perturbed - expected)
                    negative_error = float(np.min(error))
                    negative_detected = bool(np.any(error > tolerance))
            return {
                "negative_control_detected": negative_detected,
                "negative_control_error": negative_error,
            }

        return self.check(
            "VER-003-execution",
            action,
            "internal verification execution completed",
        )

    def science_checks(self) -> None:
        claims = {item["id"]: item for item in self.desc["claims"]}
        evidence = self.desc["evidence"]

        def claim_scope() -> str:
            fields = (
                "id",
                "statement",
                "scope",
                "intended_use_domain",
                "applicable_profile",
                "model_version",
                "decision_relevance",
                "status",
            )
            valid_status = {
                "proposed",
                "supported",
                "not-supported",
                "rejected",
                "withdrawn",
            }
            for claim in claims.values():
                require_fields(claim, fields, f"claim {claim.get('id', '?')}")
                if claim["intended_use_domain"] not in self.domains:
                    raise CheckFailure(f"claim domain does not resolve: {claim['id']}")
                if claim["applicable_profile"] not in PROFILES:
                    raise CheckFailure(f"claim profile is unknown: {claim['id']}")
                if claim["status"] not in valid_status:
                    raise CheckFailure(f"claim status is invalid: {claim['id']}")
            covered = {
                reference for item in evidence for reference in item.get("claims", [])
            }
            if covered != set(claims):
                raise CheckFailure(
                    "evidence does not cover every claim exactly by reference"
                )
            return f"resolved scope and evidence status for {len(claims)} claims"

        self.check(
            "SCI-001",
            claim_scope,
            "Every claim resolves to version, domain, profile, and evidence.",
        )

        evaluated = [
            item for item in evidence if item.get("conclusion") != "not evaluated"
        ]

        def numerical_evidence() -> str:
            metric_fields = (
                "name",
                "formula",
                "units",
                "weighting",
                "mask",
                "aggregation",
                "reduction_order",
            )
            for item in evaluated:
                require_fields(
                    item,
                    (
                        "package_version",
                        "model_version",
                        "evaluation_domain",
                        "case_membership",
                        "independent_cases",
                        "metrics",
                        "results",
                        "uncertainty",
                        "code",
                        "configuration",
                        "result_artifact",
                    ),
                    f"evidence {item['id']}",
                )
                for metric in item["metrics"]:
                    require_fields(metric, metric_fields, f"metric in {item['id']}")
                if item["result_artifact"] not in self.artifacts:
                    raise CheckFailure(
                        f"result artifact does not resolve: {item['id']}"
                    )
            return f"checked detailed numerical metadata for {len(evaluated)} evaluated evidence records"

        self.check(
            "SCI-002",
            numerical_evidence,
            "Numerical evidence declares cases, metrics, reductions, uncertainty, code, and results.",
        )

        def baselines() -> str:
            for item in evidence:
                require_fields(
                    item,
                    ("baseline", "reference_uncertainty"),
                    f"evidence {item['id']}",
                )
            return "baseline and reference uncertainty are explicit for every evidence record"

        self.check(
            "SCI-003",
            baselines,
            "Baseline identity and reference uncertainty are explicit.",
        )

        def criteria() -> str:
            for item in evidence:
                if "acceptance_criteria" not in item:
                    raise CheckFailure(f"acceptance criteria absent: {item['id']}")
                criteria_value = item["acceptance_criteria"]
                if (
                    isinstance(criteria_value, dict)
                    and "predeclared" not in criteria_value
                ):
                    raise CheckFailure(
                        f"criterion declaration timing absent: {item['id']}"
                    )
            return "acceptance criteria or explicit absence is recorded"

        self.check(
            "SCI-004",
            criteria,
            "Acceptance criteria and their declaration status are explicit.",
        )

        def failures() -> str:
            for item in evidence:
                if not isinstance(item.get("observed_failures"), list):
                    raise CheckFailure(
                        f"observed failures are not represented: {item['id']}"
                    )
                if "case_membership" not in item:
                    raise CheckFailure(f"case membership is absent: {item['id']}")
            return "failures, exclusions through case membership, and unevaluated evidence are represented"

        self.check(
            "SCI-005",
            failures,
            "Failed, excluded, and unevaluated results are represented explicitly.",
        )

        ood = [
            domain
            for domain in self.domains.values()
            if "out-of-domain" in domain.get("kind", [])
        ]
        if ood:

            def ood_rules() -> str:
                for domain in ood:
                    require_fields(
                        domain,
                        ("selection_rule", "constraints"),
                        f"out-of-domain {domain['id']}",
                    )
                return "all out-of-domain results use reproducible membership rules"

            self.check(
                "SCI-006",
                ood_rules,
                "Out-of-domain conditions and membership rules are explicit.",
            )
        else:
            self.not_applicable(
                "SCI-006", "No out-of-domain result or claim is declared."
            )

        def provenance() -> str:
            value = self.desc["provenance"]
            if not value["activities"] or not value["agents"]:
                raise CheckFailure(
                    "provenance requires activities and responsible agents"
                )
            activity_types = {
                str(item.get("type", "")).lower() for item in value["activities"]
            }
            if not any(
                "export" in item or "onnx graph assembly" in item
                for item in activity_types
            ):
                raise CheckFailure(
                    "ONNX export or direct graph-assembly provenance is absent"
                )
            if not any("assembly" in item for item in activity_types):
                raise CheckFailure("package assembly provenance is absent")
            for activity in value["activities"]:
                require_fields(
                    activity,
                    (
                        "id",
                        "type",
                        "inputs",
                        "outputs",
                        "code_version",
                        "configuration",
                        "responsible_agent",
                        "runtime",
                        "randomness",
                        "failed_or_excluded_inputs",
                    ),
                    f"provenance activity {activity.get('id', '?')}",
                    allow_empty=("failed_or_excluded_inputs",),
                )
            return "export, verification, and assembly provenance have immutable code identities"

        self.check(
            "SCI-007",
            provenance,
            "Source, export, evaluation, and package provenance is explicit or inapplicable.",
        )

        training_domains = [
            domain
            for domain in self.domains.values()
            if "training" in domain.get("kind", [])
            and domain.get("size", {}).get("independent_cases", 0) != 0
        ]
        if training_domains:
            self.finding(
                "SCI-008",
                "not-run",
                "training split leakage controls require a package-specific provenance adapter",
            )
        else:
            self.not_applicable(
                "SCI-008", "The model has no training data or fitted training process."
            )

        def status_wording() -> str:
            status = self.desc["release_status"]
            supported_scientific = [
                claim
                for claim in claims.values()
                if claim["applicable_profile"] == "scientifically-validated"
                and claim["status"] == "supported"
            ]
            if status == "research" and supported_scientific:
                raise CheckFailure(
                    "research status conflicts with a supported scientific-validation claim"
                )
            if status in {"validated", "operational"} and not supported_scientific:
                raise CheckFailure(
                    f"{status} status lacks a supported scientific claim"
                )
            return f"release status {status!r} does not exceed evidence status"

        self.check(
            "SCI-009",
            status_wording,
            "Release status and claim wording do not overstate evidence.",
        )

        experimental = any(
            any(
                word in str(entity).lower()
                for word in ("instrument", "sensor", "observational", "experiment")
            )
            for entity in self.desc["provenance"]["entities"]
        )
        if experimental:
            self.finding(
                "SCI-010",
                "not-run",
                "experimental calibration and measurement uncertainty require a domain adapter",
            )
        else:
            self.not_applicable(
                "SCI-010", "No experimental or observational source data is declared."
            )

    def card_checks(self) -> None:
        try:
            card = self.model_card_text()
        except (CheckFailure, OSError, UnicodeDecodeError) as error:
            for check_id in [f"CARD-{number:03d}" for number in range(1, 9)]:
                self.finding(
                    check_id, "fail", f"Model Card cannot be inspected: {error}"
                )
            return

        missing = [heading for heading in MODEL_CARD_HEADINGS if heading not in card]
        placeholders = re.findall(r"@@[^@]+@@|<(?!br\s*/?>)[^>\n]+>", card)
        if missing or placeholders:
            details = []
            if missing:
                details.append(f"missing headings: {', '.join(missing)}")
            if placeholders:
                details.append("unresolved placeholders are present")
            self.finding("CARD-001", "fail", "; ".join(details))
        else:
            self.finding(
                "CARD-001",
                "pass",
                "All required Model Card sections are present and non-placeholder.",
            )

        opening = " ".join(card[:2000].lower().split())
        output_quantity_terms: list[str] = []
        for tensor in self.tensors.values():
            if tensor.get("direction") != "output":
                continue
            for channel in tensor.get("channels", []):
                quantity = self.quantities.get(channel.get("quantity"), {})
                for value in (quantity.get("display_name"), quantity.get("id")):
                    if nonempty(value):
                        output_quantity_terms.append(
                            str(value).lower().replace("-", " ")
                        )
        opening_requirements = {
            "package identifier": str(self.desc["package_id"]).lower() in opening,
            "release status": str(self.desc["release_status"]).lower() in opening,
            "predicted quantity": bool(output_quantity_terms)
            and any(term in opening for term in output_quantity_terms),
            "intended-use boundary": "intended" in opening or "supported" in opening,
        }
        missing_opening = [
            label for label, present in opening_requirements.items() if not present
        ]
        if not missing_opening:
            self.finding(
                "CARD-002",
                "pass",
                "The opening identifies purpose, system, quantities, status, and use boundary.",
            )
        else:
            self.finding(
                "CARD-002",
                "fail",
                "The Model Card opening omits: " + ", ".join(missing_opening),
            )

        missing_artifacts = [
            item["location"]
            for item in self.artifacts.values()
            if item["requirement"] == "required" and item["location"] not in card
        ]
        if missing_artifacts:
            self.finding(
                "CARD-003",
                "fail",
                "The package map omits required artifacts.",
                artifacts=missing_artifacts,
            )
        else:
            self.finding(
                "CARD-003",
                "pass",
                "The Model Card package map covers every required artifact.",
            )

        quick = card.split("## Quick start", 1)[-1].split("\n## ", 1)[0]
        if any(term in quick.lower() for term in ("valid", "verify", "reject")) and all(
            tensor["onnx_name"] in quick for tensor in self.tensors.values()
        ):
            self.finding(
                "CARD-004",
                "pass",
                "Quick start covers the complete path and validity checking.",
            )
        else:
            self.finding(
                "CARD-004",
                "fail",
                "Quick start omits a tensor mapping or validity check.",
            )

        missing_interfaces = [
            tensor["onnx_name"]
            for tensor in self.tensors.values()
            if tensor["onnx_name"] not in card
        ]
        if missing_interfaces:
            self.finding(
                "CARD-005",
                "fail",
                "Input/output documentation is incomplete.",
                entities=missing_interfaces,
            )
        else:
            self.finding(
                "CARD-005",
                "pass",
                "Input/output documentation covers every ONNX value and channel.",
            )

        limitation_terms = ("unsupported", "limitation", "failure", "unknown")
        if all(term in card.lower() for term in limitation_terms):
            self.finding(
                "CARD-006",
                "pass",
                "Uses, limitations, failures, unknowns, and detectability are documented.",
            )
        else:
            self.finding(
                "CARD-006",
                "fail",
                "Limitations or failure-detectability documentation is incomplete.",
            )

        claims_missing = [
            claim["id"] for claim in self.desc["claims"] if claim["id"] not in card
        ]
        evidence_missing = [
            item["id"] for item in self.desc["evidence"] if item["id"] not in card
        ]
        if claims_missing or evidence_missing:
            self.finding(
                "CARD-007",
                "fail",
                "Validation tables do not link every claim and evidence record.",
                entities=claims_missing + evidence_missing,
            )
        else:
            self.finding(
                "CARD-007",
                "pass",
                "Validation tables link claims, domains, criteria, results, and evidence.",
            )

        maintenance_terms = (
            "license",
            "citation",
            "runtime",
            "contact",
            "change history",
            "provenance",
        )
        if all(term in card.lower() for term in maintenance_terms):
            self.finding(
                "CARD-008",
                "pass",
                "Provenance, runtime, legal, maintenance, and contact information is present.",
            )
        else:
            self.finding(
                "CARD-008",
                "fail",
                "Provenance, runtime, legal, maintenance, or contact information is incomplete.",
            )

    def detected_profiles(self) -> dict[str, bool]:
        spatial = any(
            "non-spatial" not in str(axis.get("location", "")).lower()
            and any(
                term in str(axis.get("location", "")).lower()
                for term in ("cell", "face", "edge", "node", "grid", "mesh", "spatial")
            )
            for axis in self.axes.values()
        )
        coupling_text = json.dumps(
            self.desc["coupling_contracts"], sort_keys=True
        ).lower()
        iterative = bool(
            self.desc["coupling_contracts"] and "iteration" in coupling_text
        )
        coupled = bool(self.desc["coupling_contracts"] and "external" in coupling_text)
        scientific = self.desc["release_status"] in {"validated", "operational"} or any(
            claim.get("applicable_profile") == "scientifically-validated"
            and claim.get("status") == "supported"
            for claim in self.desc["claims"]
        )
        return {
            "core": True,
            "single-pass": not iterative and not coupled,
            "iterative": iterative,
            "coupled": coupled,
            "spatial": spatial,
            "multi-graph": len(self.models) > 1,
            "custom-operator": any(
                model.get("custom_operators") for model in self.models.values()
            ),
            "external-artifact": any(
                is_external_location(item.get("location", ""))
                and item.get("requirement") == "required"
                for item in self.artifacts.values()
            ),
            "stochastic": any(
                str(case.get("nondeterministic_variation", "none")).lower()
                not in {"none", "none expected", "not applicable"}
                for case in self.desc["verification_cases"]
            )
            or any(
                runtime.get("deterministic") is False
                for runtime in self.desc["runtime_configurations"]
            ),
            "scientifically-validated": scientific,
            "training-reproducible": any(
                claim.get("applicable_profile") == "training-reproducible"
                and claim.get("status") == "supported"
                for claim in self.desc["claims"]
            ),
            "operational": self.desc["release_status"] == "operational",
        }

    def profile_checks(self) -> None:
        decisions = self.desc["profile_decisions"]
        by_profile = {item.get("profile"): item for item in decisions}
        if (
            set(by_profile) == set(PROFILES)
            and len(decisions) == len(PROFILES)
            and by_profile["core"].get("applicable")
        ):
            behavior = [
                profile
                for profile in ("single-pass", "iterative", "coupled")
                if by_profile[profile].get("applicable")
            ]
            if behavior:
                self.finding(
                    "PROF-001",
                    "pass",
                    "Every profile has a decision; core and a behavior profile are applicable.",
                )
            else:
                self.finding(
                    "PROF-001", "fail", "No behavior profile is declared applicable."
                )
        else:
            self.finding(
                "PROF-001",
                "fail",
                "Profile decisions are missing, duplicated, or omit core.",
            )

        detected = self.detected_profiles()
        mismatches = [
            profile
            for profile, applicable in detected.items()
            if by_profile.get(profile, {}).get("applicable") is not applicable
        ]
        if mismatches:
            self.finding(
                "PROF-002",
                "fail",
                "Declared applicability conflicts with detected package behavior.",
                entities=mismatches,
            )
        else:
            self.finding(
                "PROF-002",
                "pass",
                "No applicable profile was omitted or marked inapplicable.",
            )

        applicable = [profile for profile, value in detected.items() if value]
        unsupported = [
            profile for profile in applicable if profile not in {"core", "single-pass"}
        ]
        if unsupported:
            self.finding(
                "PROF-003",
                "not-run",
                "This validator lacks executable checks for applicable profiles.",
                entities=unsupported,
            )
        else:
            core_ready = all(
                self.findings.get(check_id, {}).get("status")
                in {"pass", "not-applicable"}
                for check_id in CHECK_IDS
                if not check_id.startswith("PROF-")
            )
            if core_ready:
                self.finding(
                    "PROF-003",
                    "pass",
                    "All additional core and single-pass requirements were executed.",
                )
            else:
                self.finding(
                    "PROF-003",
                    "fail",
                    "One or more core or single-pass requirements did not pass.",
                )

        incomplete_scope = [
            profile
            for profile in applicable
            if not nonempty(by_profile.get(profile, {}).get("scope"))
            or not nonempty(by_profile.get(profile, {}).get("evidence_or_reason"))
        ]
        if incomplete_scope:
            self.finding(
                "PROF-004",
                "fail",
                "Applicable profile scope or evidence is incomplete.",
                entities=incomplete_scope,
            )
        else:
            self.finding(
                "PROF-004",
                "pass",
                "Applicable profiles identify process/domain scope and evidence.",
            )

        bad_status = [
            profile
            for profile in applicable
            if by_profile.get(profile, {}).get("status") != "conforming"
        ]
        if bad_status:
            self.finding(
                "PROF-005",
                "fail",
                "An applicable profile is not declared conforming.",
                entities=bad_status,
            )
        else:
            self.finding(
                "PROF-005", "pass", "Every applicable profile is declared conforming."
            )

    def validate(self) -> dict[str, Any]:
        self.load_description()
        if self.description is not None:
            self.package_checks()
            self.onnx_checks()
            self.semantic_checks()
            self.flow_checks()
            self.verification_checks()
            self.science_checks()
            self.card_checks()
            self.profile_checks()
        for check_id in CHECK_IDS:
            if check_id not in self.findings:
                self.finding(
                    check_id,
                    "not-run",
                    "a prerequisite prevented this check from running",
                )
        internal_execution = self.findings.pop("VER-003-execution", None)
        if internal_execution and internal_execution["status"] not in {"pass"}:
            self.findings["VER-003"] = {
                **internal_execution,
                "id": "VER-003",
            }
        ordered = [self.findings[check_id] for check_id in CHECK_IDS]
        if "PKG-R01" in self.findings:
            ordered.append(self.findings["PKG-R01"])
        required_errors = [
            item
            for item in ordered
            if item["severity"] == "error" and item["status"] in {"fail", "not-run"}
        ]
        verification = [item for item in ordered if item["id"].startswith("VER-")]
        structural_prefixes = ("PKG-", "ONNX-", "SEM-", "FLOW-", "CARD-", "PROF-")
        structural_errors = [
            item
            for item in required_errors
            if item["id"].startswith(structural_prefixes)
        ]
        detected = self.detected_profiles() if self.description is not None else {}
        report = {
            "report_format": "scientific ONNX model package conformance report",
            "report_format_version": "1",
            "specification_id": SPECIFICATION_ID,
            "specification_version": SPECIFICATION_VERSION,
            "package_id": (
                self.description.get("package_id", "unresolved")
                if self.description
                else "unresolved"
            ),
            "package_version": (
                self.description.get("package_version", "unresolved")
                if self.description
                else "unresolved"
            ),
            "package_root_identity": "directory identified by the supplied package-description digest",
            "package_description_digest": {
                "algorithm": "sha256",
                "expected": self.expected_digest,
                "observed": self.description_digest or "unavailable",
            },
            "adapter": self.adapter,
            "validator": {
                "name": VALIDATOR_NAME,
                "version": VALIDATOR_VERSION,
                "configuration": {
                    "description_adapter": "direct JSON v1",
                    "run_verification": self.run_verification,
                    "package_code_execution": False,
                },
                "invocation_time": datetime.now(timezone.utc).isoformat(),
            },
            "declared_profiles": (
                self.description.get("profile_decisions", [])
                if self.description
                else []
            ),
            "detected_applicable_profiles": [
                profile for profile, applies in detected.items() if applies
            ],
            "runtime_configurations": (
                self.description.get("runtime_configurations", [])
                if self.description
                else []
            ),
            "checks": ordered,
            "verification_case_results": self.verification_results,
            "warnings": [
                item["message"] for item in ordered if item["severity"] == "warning"
            ],
            "not_run": [
                {"id": item["id"], "reason": item["message"]}
                for item in ordered
                if item["status"] == "not-run"
            ],
            "overall": {
                "conforming": not required_errors,
                "structurally_conforming": not structural_errors,
                "verification_passed": bool(verification)
                and all(
                    item["status"] in {"pass", "not-applicable"}
                    for item in verification
                ),
                "scientifically_validated": bool(
                    detected.get("scientifically-validated")
                )
                and not required_errors,
                "operationally_accepted": bool(detected.get("operational"))
                and not required_errors,
            },
            "responsible_organization": "validator operator",
            "signature": "not supplied",
        }
        return report


def read_expected_digest(value: str | None, record: Path | None) -> str:
    if value is not None:
        return value.strip().lower()
    if record is None:
        raise ValueError("an expected description digest is required")
    line = record.read_text(encoding="utf-8").strip()
    parts = line.split()
    if len(parts) != 3 or parts[0].lower() != "sha256":
        raise ValueError("digest record must contain: sha256 HEX filename")
    return parts[1].lower()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def summary_text(report: dict[str, Any]) -> str:
    counts: dict[str, int] = {}
    for finding in report["checks"]:
        counts[finding["status"]] = counts.get(finding["status"], 0) + 1
    errors = sum(
        finding["severity"] == "error" and finding["status"] in {"fail", "not-run"}
        for finding in report["checks"]
    )
    warnings = sum(finding["severity"] == "warning" for finding in report["checks"])
    overall = "CONFORMING" if report["overall"]["conforming"] else "NOT CONFORMING"
    lines = [
        f"Scientific ONNX package: {overall}",
        f"package: {report['package_id']} {report['package_version']}",
        "checks: "
        + ", ".join(
            (
                f"pass={counts.get('pass', 0)}",
                f"not-applicable={counts.get('not-applicable', 0)}",
                f"not-run={counts.get('not-run', 0)}",
                f"errors={errors}",
                f"warnings={warnings}",
            )
        ),
        f"structural conformance: {str(report['overall']['structurally_conforming']).lower()}",
        f"known-answer verification: {str(report['overall']['verification_passed']).lower()}",
        f"scientific validation: {str(report['overall']['scientifically_validated']).lower()}",
        f"operational acceptance: {str(report['overall']['operationally_accepted']).lower()}",
    ]
    problems = [
        item
        for item in report["checks"]
        if item["severity"] == "error" and item["status"] in {"fail", "not-run"}
    ]
    if problems:
        lines.append("required checks not satisfied:")
        lines.extend(f"  {item['id']}: {item['message']}" for item in problems)
    warnings = [item for item in report["checks"] if item["severity"] == "warning"]
    if warnings:
        lines.append("warnings:")
        lines.extend(f"  {item['id']}: {item['message']}" for item in warnings)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_root", type=Path)
    parser.add_argument(
        "--description",
        default="package-description.json",
        help="safe package-relative description path (default: package-description.json)",
    )
    digest_group = parser.add_mutually_exclusive_group(required=True)
    digest_group.add_argument(
        "--expected-description-digest",
        help="trusted expected SHA-256 for the exact package-description bytes",
    )
    digest_group.add_argument(
        "--expected-description-digest-file",
        type=Path,
        help="operator-trusted detached record containing: sha256 HEX filename",
    )
    parser.add_argument(
        "--run-verification",
        action="store_true",
        help="decode supported cases and execute ONNX with validator-owned code",
    )
    parser.add_argument(
        "--report", type=Path, help="write the machine-readable JSON report"
    )
    parser.add_argument("--summary", type=Path, help="write the human-readable summary")
    arguments = parser.parse_args()
    try:
        expected = read_expected_digest(
            arguments.expected_description_digest,
            arguments.expected_description_digest_file,
        )
        validator = Validator(
            arguments.package_root,
            arguments.description,
            expected,
            arguments.run_verification,
        )
        report = validator.validate()
    except (OSError, ValueError) as error:
        parser.error(str(error))
    summary = summary_text(report)
    sys.stdout.write(summary)
    if arguments.report:
        write_json(arguments.report, report)
    if arguments.summary:
        arguments.summary.write_text(summary, encoding="utf-8")
    return 0 if report["overall"]["conforming"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
