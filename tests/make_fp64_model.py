"""Tiny loss-of-FP64-sensitive mixed-I/O and arbitrary-rank fixtures."""
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def save(path, nodes, inputs, outputs, initializers=()):
    model = helper.make_model(
        helper.make_graph(nodes, path.stem, inputs, outputs, list(initializers)),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)


def make_models(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vi = helper.make_tensor_value_info
    save(path, [
        helper.make_node("Add", ["state", "delta"], ["shifted"]),
        helper.make_node("Where", ["enabled", "shifted", "state"], ["values"]),
        helper.make_node("Greater", ["state", "zero"], ["positive"]),
        helper.make_node("And", ["positive", "enabled"], ["valid"]),
    ], [vi("state", TensorProto.DOUBLE, ["rows", 2]),
        vi("enabled", TensorProto.BOOL, ["rows", 2])],
       [vi("values", TensorProto.DOUBLE, ["rows", 2]),
        vi("valid", TensorProto.BOOL, ["rows", 2])],
       [numpy_helper.from_array(np.asarray(2.0**-42, dtype=np.float64), "delta"),
        numpy_helper.from_array(np.asarray(0, dtype=np.float64), "zero")])
    for rank in range(1, 9):
        for dtype, elem in [("float64", TensorProto.DOUBLE), ("bool", TensorProto.BOOL)]:
            save(path.with_name(f"{path.stem}_{dtype}_{rank}.onnx"),
                 [helper.make_node("Identity", ["x"], ["y"])],
                 [vi("x", elem, [2] * rank)], [vi("y", elem, [2] * rank)])
        save(path.with_name(f"{path.stem}_compare_{rank}.onnx"),
             [helper.make_node("Greater", ["x", "zero"], ["y"])],
             [vi("x", TensorProto.DOUBLE, [2] * rank)],
             [vi("y", TensorProto.BOOL, [2] * rank)],
             [numpy_helper.from_array(np.asarray(0, dtype=np.float64), "zero")])


if __name__ == "__main__":
    make_models(sys.argv[1])
