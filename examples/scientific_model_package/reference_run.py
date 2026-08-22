#!/usr/bin/env python3
"""Run the scientific package example with the ONNX reference evaluator."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
from typing import Any

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

ABSOLUTE_TOLERANCE = np.float32(1.0e-6)
RELATIVE_TOLERANCE = np.float32(1.0e-6)
EXPECTED_COLUMNS = [
    "temperature_anomaly_k",
    "radiative_forcing_w_m2",
    "expected_temperature_response_k",
]


def load_cases(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(f"unexpected verification columns: {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError("verification file contains no cases")

    temperature = np.asarray(
        [float(row[EXPECTED_COLUMNS[0]]) for row in rows], dtype=np.float32
    ).reshape(-1, 1)
    forcing = np.asarray(
        [float(row[EXPECTED_COLUMNS[1]]) for row in rows], dtype=np.float32
    ).reshape(-1, 1)
    expected = np.asarray(
        [float(row[EXPECTED_COLUMNS[2]]) for row in rows], dtype=np.float32
    ).reshape(-1, 1)
    return temperature, forcing, expected


def validate_inputs(temperature: np.ndarray, forcing: np.ndarray) -> None:
    if temperature.shape != forcing.shape or temperature.ndim != 2:
        raise ValueError("inputs must have equal rank-2 shapes")
    if temperature.shape[0] < 1 or temperature.shape[1] != 1:
        raise ValueError("inputs must have ONNX logical shape [batch, 1]")
    if not np.all(np.isfinite(temperature)) or not np.all(np.isfinite(forcing)):
        raise ValueError("inputs must be finite")
    if np.any((temperature < -100.0) | (temperature > 100.0)):
        raise ValueError("temperature anomaly is outside interface-domain")
    if np.any((forcing < -2000.0) | (forcing > 2000.0)):
        raise ValueError("radiative forcing is outside interface-domain")


def _run(
    evaluator: ReferenceEvaluator,
    temperature: np.ndarray,
    forcing: np.ndarray,
) -> np.ndarray:
    outputs = evaluator.run(
        None,
        {
            "temperature_anomaly": temperature,
            "radiative_forcing": forcing,
        },
    )
    return np.asarray(outputs[0], dtype=np.float32)


def _check(actual: np.ndarray, expected: np.ndarray, label: str) -> float:
    allowed = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * np.abs(expected)
    if not np.all(np.isfinite(actual)):
        raise RuntimeError(f"{label} produced a non-finite output")
    error = np.abs(actual - expected)
    if np.any(error > allowed):
        raise RuntimeError(f"{label} failed: maximum error {float(np.max(error)):.9g}")
    return float(np.max(error))


def verify_package(package_root: Path) -> dict[str, Any]:
    model_path = package_root / "models" / "linear-response.onnx"
    case_path = package_root / "verification" / "nominal.csv"
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    evaluator = ReferenceEvaluator(model)
    temperature, forcing, expected = load_cases(case_path)
    validate_inputs(temperature, forcing)

    nominal = _run(evaluator, temperature, forcing)
    nominal_error = _check(nominal, expected, "nominal batch")

    dynamic = _run(evaluator, temperature[:1], forcing[:1])
    dynamic_error = _check(dynamic, expected[:1], "dynamic batch size 1")

    perturbed = _run(evaluator, forcing, temperature)
    perturbation_error = np.abs(perturbed - expected)
    allowed = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * np.abs(expected)
    if not np.any(perturbation_error > allowed):
        raise RuntimeError("swapped-input negative control was not detected")

    return {
        "case_count": int(expected.shape[0]),
        "nominal_maximum_absolute_error": nominal_error,
        "dynamic_batch_maximum_absolute_error": dynamic_error,
        "swapped_input_minimum_detected_error": float(np.min(perturbation_error)),
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: reference_run.py PACKAGE_DIRECTORY")
    result = verify_package(Path(sys.argv[1]).resolve())
    print(
        "Scientific package reference verification passed; "
        f"max error = {result['nominal_maximum_absolute_error']:.9g}"
    )


if __name__ == "__main__":
    main()
