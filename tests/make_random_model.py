#!/usr/bin/env python3
"""Generate a tiny ONNX model that the cuda_graph option refuses: a RandomNormalLike node, or (KIND dropout) a
com.microsoft BitmaskDropout node with a training_mode input."""

from pathlib import Path
import sys

import onnx
from onnx import TensorProto, helper


def main() -> None:
    if len(sys.argv) not in (2, 3) or sys.argv[2:] not in ([], ["dropout"]):
        raise SystemExit("usage: make_random_model.py OUTPUT.onnx [dropout]")
    destination = Path(sys.argv[1])
    destination.parent.mkdir(parents=True, exist_ok=True)

    if sys.argv[2:] == ["dropout"]:
        nodes = [
            helper.make_node(
                "BitmaskDropout", ["input", "ratio", "training_mode"], ["output"], domain="com.microsoft"
            )
        ]
        initializers = [
            helper.make_tensor("ratio", TensorProto.FLOAT, [], [0.5]),
            helper.make_tensor("training_mode", TensorProto.BOOL, [], [True]),
        ]
    else:
        nodes = [
            helper.make_node("RandomNormalLike", ["input"], ["noise"]),
            helper.make_node("Add", ["input", "noise"], ["output"]),
        ]
        initializers = []
    graph = helper.make_graph(
        nodes,
        "fortonnx_random_test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])],
        initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="fortonnx-tests",
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save_model(model, destination)


if __name__ == "__main__":
    main()
