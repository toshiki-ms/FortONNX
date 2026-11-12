#!/usr/bin/env python3
"""Generate a tiny ONNX model with a GatherND node (refused with the cuda_graph option)."""

from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_gather_nd_model.py OUTPUT.onnx")
    destination = Path(sys.argv[1])
    destination.parent.mkdir(parents=True, exist_ok=True)

    indices = numpy_helper.from_array(np.asarray([[1], [0]], dtype=np.int64), name="indices")
    graph = helper.make_graph(
        [helper.make_node("GatherND", ["input", "indices"], ["output"])],
        "fortonnx_gather_nd_test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])],
        [indices],
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
