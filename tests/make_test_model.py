#!/usr/bin/env python3
"""Generate a tiny deterministic ONNX model used by the Fortran tests."""

from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_test_model.py OUTPUT.onnx")
    destination = Path(sys.argv[1])
    destination.parent.mkdir(parents=True, exist_ok=True)

    weights = numpy_helper.from_array(
        np.asarray([[2.0], [-3.0]], dtype=np.float32), name="weights"
    )
    bias = numpy_helper.from_array(np.asarray([0.5], dtype=np.float32), name="bias")
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["input", "weights"], ["linear"]),
            helper.make_node("Add", ["linear", "bias"], ["output"]),
        ],
        "fortonnx_linear_test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 1])],
        [weights, bias],
    )
    model = helper.make_model(
        graph,
        producer_name="fortonnx-tests",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save_model(model, destination)


if __name__ == "__main__":
    main()

