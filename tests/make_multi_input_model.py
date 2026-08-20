#!/usr/bin/env python3
"""Generate a deterministic image model with a separate scalar control input."""

from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_multi_input_model.py OUTPUT.onnx")
    destination = Path(sys.argv[1])
    destination.parent.mkdir(parents=True, exist_ok=True)

    offset = numpy_helper.from_array(np.asarray([-1.0], dtype=np.float32), name="offset")
    graph = helper.make_graph(
        [
            helper.make_node("Mul", ["image", "gain"], ["scaled"]),
            helper.make_node("Add", ["scaled", "offset"], ["shifted"]),
            helper.make_node("Relu", ["shifted"], ["enhanced"]),
        ],
        "fortonnx_multi_input_test",
        [
            helper.make_tensor_value_info(
                "image", TensorProto.FLOAT, [None, 1, 2, 2]
            ),
            helper.make_tensor_value_info("gain", TensorProto.FLOAT, [1]),
        ],
        [
            helper.make_tensor_value_info(
                "enhanced", TensorProto.FLOAT, [None, 1, 2, 2]
            ),
            helper.make_tensor_value_info(
                "scaled", TensorProto.FLOAT, [None, 1, 2, 2]
            ),
        ],
        [offset],
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
