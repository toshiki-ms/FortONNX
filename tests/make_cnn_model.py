#!/usr/bin/env python3
"""Generate a deterministic Conv-plus-ReLU model for tensor binding tests."""

from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_cnn_model.py OUTPUT.onnx")
    destination = Path(sys.argv[1])
    destination.parent.mkdir(parents=True, exist_ok=True)

    kernel = numpy_helper.from_array(
        np.ones((1, 1, 3, 3), dtype=np.float32), name="kernel"
    )
    bias = numpy_helper.from_array(np.asarray([-60.0], dtype=np.float32), name="bias")
    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["image", "kernel", "bias"], ["convolved"]),
            helper.make_node("Relu", ["convolved"], ["features"]),
        ],
        "fortonnx_cnn_test",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [None, 1, 4, 4])],
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, 1, 2, 2])],
        [kernel, bias],
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
