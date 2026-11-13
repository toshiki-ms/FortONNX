"""Row-wise form of an ONNX graph.

The generator accepts graphs with one input ``[rows, ...]`` and one output ``[rows, ...]`` in which every operator
acts on each row independently. Such a graph is a function of one row, and this module rewrites it as a list of
*stages* over the scalar elements of that row:

- every tensor that depends on the row becomes an array of *element references* (which stage, which element);
  Slice, Concat, Reshape, Unsqueeze, Squeeze, Expand, Transpose, Split, Gather, Identity and a Where with a constant
  condition only rearrange references and leave no stage behind;
- everything that does not depend on the row is evaluated here with NumPy (FP32) and becomes constants;
- the remaining operators become stages: ``map`` (one scalar operator per element), ``dot`` (a sum of products per
  element: MatMul) and ``reduce`` (ReduceSum, ReduceMax, ...).

An operator that is not row-wise, or not supported, stops the import with ``UnsupportedGraph``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import onnx
from onnx import numpy_helper

from .cforms import comment_text

_OFFSET_BITS = 32
_OFFSET_MASK = (1 << _OFFSET_BITS) - 1
_INT64_BIG = 1 << 62


class UnsupportedGraph(ValueError):
    """The graph has an operator or a structure that the generator does not handle."""


def make_refs(stage_index: int, offsets: np.ndarray) -> np.ndarray:
    """References to the elements ``offsets`` of stage ``stage_index``."""
    return (np.int64(stage_index) << _OFFSET_BITS) | np.asarray(offsets, dtype=np.int64)


def ref_stage(refs: np.ndarray) -> np.ndarray:
    return np.asarray(refs) >> _OFFSET_BITS


def ref_offset(refs: np.ndarray) -> np.ndarray:
    return np.asarray(refs) & _OFFSET_MASK


@dataclass
class Stage:
    """One array of per-row scalars.

    kind      "input" (the row itself), "const" (values known at generation), "map", "dot" or "reduce"
    operands  map: one reference array [size] per argument; reduce: one array [size, terms];
              dot: [size, terms] (constant factors in ``weights``) or two such arrays (no ``weights``)
    weights   dot only: [size, terms] FP32 constant factor of every term
    bias      dot only: [size] FP32 constant added to the sum (constant terms folded at import), or None
    values    const only: [size] FP32 values (0 or 1 for a boolean constant)
    log2_units  logsumexp only: (the terms, the result) are in units of ln 2 (idioms.fold_scales)
    """

    index: int
    kind: str
    name: str
    size: int
    is_bool: bool = False
    op: str = ""
    operands: list[np.ndarray] = field(default_factory=list)
    weights: np.ndarray | None = None
    bias: np.ndarray | None = None
    values: np.ndarray | None = None
    log2_units: tuple[bool, bool] = (False, False)

    @property
    def terms(self) -> int:
        """Number of terms per element of a dot or reduce stage."""
        return self.operands[0].shape[1]


@dataclass
class RowGraph:
    """The stages of one row and the references that make up the output row."""

    stages: list[Stage]
    input_name: str
    output_name: str
    input_shape: tuple[int, ...]    # of one input row
    output_shape: tuple[int, ...]   # of one output row
    output_refs: np.ndarray         # [output_size] in row-major order
    # for a part of a graph (sub_graph): the positions of its outputs in the output row; None: all, in order
    output_slots: np.ndarray | None = None

    @property
    def input_size(self) -> int:
        return int(np.prod(self.input_shape))

    @property
    def output_size(self) -> int:
        return int(self.output_refs.size)

    @property
    def row_width(self) -> int:
        """Values of an output row (for a part of a graph: of the whole graph's)."""
        return self.output_size if self.output_slots is None else int(np.prod(self.output_shape))

    def is_const(self) -> np.ndarray:
        """[stage] True for constant stages."""
        return np.array([stage.kind == "const" for stage in self.stages])

    def const_values(self, refs: np.ndarray) -> np.ndarray:
        """FP32 values of references that all point to constant stages."""
        refs = np.asarray(refs)
        out = np.empty(refs.shape, dtype=np.float32)
        stage_of = ref_stage(refs)
        for index in np.unique(stage_of):
            mask = stage_of == index
            out[mask] = self.stages[index].values[ref_offset(refs[mask])]
        return out


@dataclass
class _View:
    """A tensor that depends on the row: references in the tensor's shape, with length 1 on the row axis."""

    refs: np.ndarray
    row_axis: int
    is_bool: bool = False


# Scalar operators of map stages: name -> (NumPy function, C expression, result is boolean).
# fmax/fmin return the other argument when one is NaN, like the device functions used in the kernel.
MAP_OPERATORS: dict[str, tuple] = {
    "add": (np.add, "({0} + {1})", False),
    "sub": (np.subtract, "({0} - {1})", False),
    "mul": (np.multiply, "({0} * {1})", False),
    "div": (np.divide, "({0} / {1})", False),
    "pow": (np.power, "powf({0}, {1})", False),
    "max": (np.maximum, "({0} == {0} && {1} == {1} ? fmaxf({0}, {1}) : NAN)", False),   # NaN if an operand is
    "min": (np.minimum, "({0} == {0} && {1} == {1} ? fminf({0}, {1}) : NAN)", False),
    "neg": (np.negative, "(-{0})", False),
    "abs": (np.abs, "fabsf({0})", False),
    "exp": (np.exp, "expf({0})", False),
    "log": (np.log, "logf({0})", False),
    "log1p": (np.log1p, "log1pf({0})", False),
    "logaddexp": (np.logaddexp, "({0} == {0} && {1} == {1} ? fmaxf({0}, {1}) + log1pf(expf(-fabsf({0} - {1}))) : NAN)", False),
    "isfinite": (np.isfinite, "(fabsf({0}) < INFINITY)", True),
    "tanh": (np.tanh, "tanhf({0})", False),
    "sqrt": (np.sqrt, "sqrtf({0})", False),
    "sin": (np.sin, "sinf({0})", False),
    "cos": (np.cos, "cosf({0})", False),
    "floor": (np.floor, "floorf({0})", False),
    "ceil": (np.ceil, "ceilf({0})", False),
    "reciprocal": (lambda a: 1 / a, "(1.0f / {0})", False),
    "relu": (lambda a: np.maximum(a, 0), "({0} == {0} ? fmaxf({0}, 0.0f) : NAN)", False),
    # for a negative value in the form without exp(-a): that overflows where the result is still a number
    "sigmoid": (lambda a: np.where(a >= 0, 1 / (1 + np.exp(-np.abs(a))), np.exp(-np.abs(a)) / (1 + np.exp(-np.abs(a)))),
                "({0} >= 0.0f ? 1.0f / (1.0f + expf(-{0})) : expf({0}) / (1.0f + expf({0})))", False),
    "softplus": (lambda a: np.where(a > 0, a + np.log1p(np.exp(-np.abs(a))), np.log1p(np.exp(-np.abs(a)))),
                 "({0} > 0.0f ? {0} + log1pf(expf(-fabsf({0}))) : log1pf(expf(-fabsf({0}))))", False),
    "equal": (np.equal, "({0} == {1})", True),
    "greater": (np.greater, "({0} > {1})", True),
    "less": (np.less, "({0} < {1})", True),
    "greater_equal": (np.greater_equal, "({0} >= {1})", True),
    "less_equal": (np.less_equal, "({0} <= {1})", True),
    "isinf": (np.isinf, "isinf({0})", True),
    "isnan": (np.isnan, "isnan({0})", True),
    "not": (np.logical_not, "(!{0})", True),
    "or": (np.logical_or, "({0} || {1})", True),
    "and": (np.logical_and, "({0} && {1})", True),
    "where": (np.where, "({0} ? {1} : {2})", False),
    "cast_float": (None, "({0} ? 1.0f : 0.0f)", False),
    "exp_scaled_tanh": (lambda x, c, k: np.exp(c * np.tanh(k * x)), "expf({1} * tanhf({2} * {0}))", False),
}


def logsumexp(terms: np.ndarray, axis: int, log2_units: tuple[bool, bool] = (False, False)) -> np.ndarray:
    """log(sum(exp(terms))) along ``axis`` for finite terms; terms and result in units of ln 2 where
    ``log2_units`` says so."""
    unit_in, unit_out = (np.log(2.0) if flag else 1.0 for flag in log2_units)
    natural = terms * terms.dtype.type(unit_in)
    largest = np.max(natural, axis=axis, keepdims=True)
    with np.errstate(all="ignore"):
        total = np.log(np.sum(np.exp(natural - largest), axis=axis)) + np.squeeze(largest, axis=axis)
    return (total / unit_out).astype(terms.dtype)


def whole_stage(graph: "RowGraph", refs: np.ndarray, op: str, kind: str = "map"):
    """The stage of this kind and operator whose elements, in order, are ``refs``; None otherwise."""
    index = int(ref_stage(refs).reshape(-1)[0])
    stage = graph.stages[index]
    if stage.kind != kind or stage.op != op or stage.size != refs.size:
        return None
    return stage if np.array_equal(refs.reshape(-1), make_refs(index, np.arange(stage.size))) else None

# Map operators whose arguments (all of them, or the first one of "where") are booleans.
_BOOL_ARGUMENTS = {"not": (0,), "or": (0, 1), "and": (0, 1), "where": (0,), "cast_float": (0,)}

_ELEMENTWISE_NODES = {
    "Add": "add", "Sub": "sub", "Mul": "mul", "Div": "div", "Pow": "pow", "Neg": "neg", "Abs": "abs", "Exp": "exp",
    "Log": "log", "Tanh": "tanh", "Sqrt": "sqrt", "Sin": "sin", "Cos": "cos", "Floor": "floor", "Ceil": "ceil",
    "Reciprocal": "reciprocal", "Relu": "relu", "Sigmoid": "sigmoid", "Softplus": "softplus", "Equal": "equal",
    "Greater": "greater", "Less": "less", "GreaterOrEqual": "greater_equal", "LessOrEqual": "less_equal",
    "IsInf": "isinf", "IsNaN": "isnan", "Not": "not", "Or": "or", "And": "and",
}
_VARIADIC_NODES = {"Max": "max", "Min": "min", "Sum": "add"}
_REDUCE_NODES = {"ReduceSum": "sum", "ReduceMax": "max", "ReduceMin": "min", "ReduceProd": "prod"}
# Constant operand that leaves the other operand unchanged bit for bit: (operator, operand position) -> value
# (x - (+0) is x; x - (-0) is not for x = -0, so the constants are compared with their signs).
_EXACT_IDENTITIES = {("sub", 1): 0.0, ("mul", 0): 1.0, ("mul", 1): 1.0, ("div", 1): 1.0}


def _same_bits(array: np.ndarray, value) -> bool:
    """True when every element of the constant equals ``value``, the sign of zero included."""
    array = np.asarray(array)
    if not bool((array == value).all()):
        return False
    return not np.issubdtype(array.dtype, np.floating) or bool((np.signbit(array) == np.signbit(value)).all())


def map_argument_is_bool(op: str, position: int) -> bool:
    return position in _BOOL_ARGUMENTS.get(op, ())


def apply_map(op: str, arguments: list[np.ndarray], dtype) -> np.ndarray:
    """Evaluate one map operator with NumPy in ``dtype`` (booleans stay booleans)."""
    if op == "cast_float":
        return arguments[0].astype(dtype)
    with np.errstate(all="ignore"):
        return MAP_OPERATORS[op][0](*arguments)


class _Importer:
    def __init__(self, model: onnx.ModelProto):
        self.model = model
        self.stages: list[Stage] = []
        self.values: dict[str, np.ndarray | _View] = {}
        self._const_stage_of: dict[tuple, int] = {}
        self.node = None

    # ---- errors and attributes -------------------------------------------------------------------------------

    def fail(self, reason: str):
        where = f"node {self.node.op_type} -> {list(self.node.output)}: " if self.node is not None else ""
        raise UnsupportedGraph(where + reason)

    def attribute(self, name: str, default=None):
        for attribute in self.node.attribute:
            if attribute.name == name:
                return onnx.helper.get_attribute_value(attribute)
        return default

    def constant_input(self, position: int, what: str) -> np.ndarray:
        value = self.values[self.node.input[position]]
        if isinstance(value, _View):
            self.fail(f"{what} must be a constant")
        return value

    # ---- stages ----------------------------------------------------------------------------------------------

    def add_stage(self, kind: str, size: int, **fields) -> Stage:
        name = comment_text(self.node.output[0]) if self.node is not None else kind   # names stand in comments only
        stage = Stage(index=len(self.stages), kind=kind, name=name, size=int(size), **fields)
        self.stages.append(stage)
        return stage

    def const_refs(self, array: np.ndarray) -> np.ndarray:
        """References (in the shape of ``array``) to a constant stage that holds its values."""
        array = np.asarray(array)
        is_bool = array.dtype == np.bool_
        if not is_bool and not np.issubdtype(array.dtype, np.floating):
            self.fail(f"a {array.dtype} constant is used as a row value")
        if not is_bool and array.dtype != np.float32:
            self.fail(f"a {array.dtype} constant is used with row values: constants are FP32")
        values = np.ascontiguousarray(array, dtype=np.float32).reshape(-1)
        key = (is_bool, values.tobytes())
        if key not in self._const_stage_of:
            self._const_stage_of[key] = self.add_stage("const", values.size, is_bool=is_bool, values=values).index
        return make_refs(self._const_stage_of[key], np.arange(values.size)).reshape(array.shape)

    def refs_are_const(self, refs: np.ndarray) -> np.ndarray:
        kinds = np.array([stage.kind == "const" for stage in self.stages])
        return kinds[ref_stage(refs)]

    def const_values(self, refs: np.ndarray, is_bool: bool) -> np.ndarray:
        out = np.empty(refs.shape, dtype=np.float32)
        stage_of = ref_stage(refs)
        for index in np.unique(stage_of):
            mask = stage_of == index
            out[mask] = self.stages[index].values[ref_offset(refs[mask])]
        return out.astype(np.bool_) if is_bool else out

    def new_view(self, stage: Stage, shape: tuple, row_axis: int) -> _View:
        return _View(make_refs(stage.index, np.arange(stage.size)).reshape(shape), row_axis, stage.is_bool)

    # ---- broadcasting ----------------------------------------------------------------------------------------

    def broadcast(self, operands: list) -> tuple[list, int | None, tuple]:
        """Broadcast constants and views together. Returns the arrays (values or references), the row axis (None
        when no operand depends on the row) and the common shape."""
        rank = max(np.ndim(o.refs if isinstance(o, _View) else o) for o in operands)
        row_axis = None
        arrays = []
        for operand in operands:
            array = operand.refs if isinstance(operand, _View) else np.asarray(operand)
            pad = rank - array.ndim
            array = array.reshape((1,) * pad + array.shape)
            if isinstance(operand, _View):
                axis = operand.row_axis + pad
                if row_axis is not None and axis != row_axis:
                    self.fail("operands carry the row axis at different positions")
                row_axis = axis
            arrays.append(array)
        if row_axis is not None:
            for operand, array in zip(operands, arrays):
                if not isinstance(operand, _View) and array.shape[row_axis] != 1:
                    self.fail("a constant is longer than 1 along the row axis")
        try:
            shape = np.broadcast_shapes(*(a.shape for a in arrays))
        except ValueError:
            self.fail("operand shapes do not broadcast")
        return [np.broadcast_to(a, shape) for a in arrays], row_axis, shape

    # ---- map stages ------------------------------------------------------------------------------------------

    def elementwise(self, op: str, operands: list):
        """One scalar operator on broadcast operands; folded when no element depends on the row."""
        result_is_bool = MAP_OPERATORS[op][2]
        arrays, row_axis, shape = self.broadcast(operands)
        if op == "where":
            result_is_bool = operands[1].is_bool if isinstance(operands[1], _View) else arrays[1].dtype == np.bool_
        if row_axis is not None and 0 in shape:   # no elements (an empty part of a Split): nothing to compute
            return _View(np.zeros(shape, dtype=np.int64), row_axis, bool(result_is_bool))
        if row_axis is None:
            for array in arrays:
                if np.issubdtype(array.dtype, np.floating) and array.dtype != np.float32:
                    self.fail(f"arithmetic on {array.dtype} constants: constants are evaluated in FP32")
            arguments = [self.as_argument(a, op, i) for i, a in enumerate(arrays)]
            if op == "div" and all(np.issubdtype(a.dtype, np.integer) for a in arguments):
                self.fail("integer division of constants is not supported")
            result = apply_map(op, arguments, np.float32)
            if op == "pow" and np.issubdtype(arrays[0].dtype, np.integer) and not np.issubdtype(result.dtype, np.integer):
                result = result.astype(arrays[0].dtype)   # as ONNX, the type of the base (NumPy takes the exponent's)
            return result

        refs = []
        for operand, array in zip(operands, arrays):
            refs.append(array if isinstance(operand, _View) else self.const_refs_for(array))
        const = [self.refs_are_const(r) for r in refs]
        if all(c.all() for c in const):
            arguments = [self.const_values(r, self.refs_are_bool(r)) for r in refs]
            folded = apply_map(op, arguments, np.float32)
            return _View(self.const_refs(folded), row_axis, folded.dtype == np.bool_)
        if op in ("greater", "less") and np.array_equal(refs[0], refs[1]):
            return _View(self.const_refs(np.zeros(shape, dtype=np.bool_)), row_axis, True)
        for position in (0, 1):
            identity = _EXACT_IDENTITIES.get((op, position))
            if identity is not None and const[position].all():
                if _same_bits(self.const_values(refs[position], False), np.float32(identity)):
                    return _View(refs[1 - position], row_axis, False)
        stage = self.add_stage("map", int(np.prod(shape)), op=op, is_bool=result_is_bool,
                               operands=[np.ascontiguousarray(r).reshape(-1) for r in refs])
        return self.new_view(stage, shape, row_axis)

    def as_argument(self, array: np.ndarray, op: str, position: int) -> np.ndarray:
        if map_argument_is_bool(op, position):
            return array.astype(np.bool_)
        if array.dtype == np.bool_ and op != "where":
            return array
        return array.astype(np.float32) if np.issubdtype(array.dtype, np.floating) else array

    def const_refs_for(self, array: np.ndarray) -> np.ndarray:
        """Constant references for a broadcast constant operand; a uniform operand is held once."""
        array = np.asarray(array)
        if array.size > 1 and _same_bits(array, array.reshape(-1)[0]):
            return np.broadcast_to(self.const_refs(array.reshape(-1)[:1]), array.shape)
        return self.const_refs(array)

    def refs_are_bool(self, refs: np.ndarray) -> bool:
        return refs.size > 0 and self.stages[int(ref_stage(refs).reshape(-1)[0])].is_bool

    # ---- import ----------------------------------------------------------------------------------------------

    def run(self) -> RowGraph:
        graph = self.model.graph
        # before opset 7 the elementwise operators broadcast by attributes, not by the shapes
        opsets = [opset.version for opset in self.model.opset_import if opset.domain in ("", "ai.onnx")]
        if opsets and min(opsets) < 7:
            raise UnsupportedGraph(f"opset {min(opsets)} is not supported: export the model with opset 7 or later")
        initializer_names = {t.name for t in graph.initializer}
        inputs = [i for i in graph.input if i.name not in initializer_names]
        if len(inputs) != 1 or len(graph.output) != 1:
            raise UnsupportedGraph("the graph must have one input and one output")
        for tensor in graph.initializer:
            self.values[tensor.name] = numpy_helper.to_array(tensor)
        input_type = inputs[0].type.tensor_type
        dims = [d.dim_value if d.HasField("dim_value") else None for d in input_type.shape.dim]
        if input_type.elem_type != onnx.TensorProto.FLOAT:
            raise UnsupportedGraph("the input must be FP32")
        if len(dims) < 2 or dims[0] is not None or any(d is None for d in dims[1:]):
            raise UnsupportedGraph("the input must be [rows, ...] with a dynamic first axis and fixed other axes")
        row_shape = tuple(dims[1:])
        input_stage = self.add_stage("input", int(np.prod(row_shape)))
        self.values[inputs[0].name] = self.new_view(input_stage, (1,) + row_shape, 0)

        for node in graph.node:
            self.node = node
            if node.domain not in ("", "ai.onnx"):
                self.fail(f"operator domain {node.domain!r} is not supported")
            handler = getattr(self, "op_" + node.op_type, None)
            if node.op_type in _ELEMENTWISE_NODES:
                self.check_default_attributes()
                outputs = [self.elementwise(_ELEMENTWISE_NODES[node.op_type], [self.values[n] for n in node.input])]
            elif node.op_type in _VARIADIC_NODES:
                result = self.values[node.input[0]]
                for name in node.input[1:]:
                    result = self.elementwise(_VARIADIC_NODES[node.op_type], [result, self.values[name]])
                outputs = [result]
            elif node.op_type in _REDUCE_NODES:
                outputs = [self.reduce(_REDUCE_NODES[node.op_type])]
            elif handler is not None:
                outputs = handler()
                outputs = outputs if isinstance(outputs, list) else [outputs]
            else:
                self.fail("operator is not supported")
            for name, value in zip(node.output, outputs):
                self.values[name] = value
        self.node = None

        output = self.values[graph.output[0].name]
        if not isinstance(output, _View) or output.row_axis != 0:
            raise UnsupportedGraph("the output must be [rows, ...] and depend on the input")
        if output.is_bool:
            raise UnsupportedGraph("the output must be FP32")
        return RowGraph(self.stages, inputs[0].name, graph.output[0].name, row_shape, tuple(output.refs.shape[1:]),
                        np.ascontiguousarray(output.refs).reshape(-1))

    def check_default_attributes(self):
        if self.node.op_type == "IsInf" and (self.attribute("detect_negative", 1) != 1
                                             or self.attribute("detect_positive", 1) != 1):
            self.fail("IsInf with detect_negative or detect_positive switched off is not supported")
        if self.attribute("broadcast") is not None or self.attribute("axis") is not None:
            self.fail("broadcasting by the attributes broadcast and axis (opsets before 7) is not supported")

    # ---- operators that only rearrange -----------------------------------------------------------------------

    def op_Identity(self):
        return self.values[self.node.input[0]]

    def op_Constant(self):
        value = self.attribute("value")
        if value is None:
            self.fail("only Constant with a tensor value is supported")
        return numpy_helper.to_array(value)

    def op_Cast(self):
        data = self.values[self.node.input[0]]
        to = self.attribute("to")
        if not isinstance(data, _View):
            return data.astype(onnx.helper.tensor_dtype_to_np_dtype(to))
        if to == onnx.TensorProto.FLOAT:
            return self.elementwise("cast_float", [data]) if data.is_bool else data
        if to == onnx.TensorProto.BOOL and data.is_bool:
            return data
        self.fail("a row value can only be cast to FP32 (from boolean or FP32); the kernel computes in FP32")

    def axes_argument(self, position: int, rank: int) -> list[int] | None:
        """Axes from the input at ``position`` (opset 13 and later) or the attribute ``axes``."""
        axes = None
        if len(self.node.input) > position and self.node.input[position]:
            axes = self.constant_input(position, "axes")
        elif self.attribute("axes") is not None:
            axes = self.attribute("axes")
        if axes is None:
            return None
        return [int(a) % rank for a in np.asarray(axes).reshape(-1)]

    def op_Squeeze(self):
        data = self.values[self.node.input[0]]
        array = data.refs if isinstance(data, _View) else data
        axes = self.axes_argument(1, array.ndim)
        if not isinstance(data, _View):
            return np.squeeze(data, axis=None if axes is None else tuple(axes))
        if axes is None or len(axes) == 0:
            self.fail("Squeeze of a row value needs explicit axes")
        if data.row_axis in axes or any(array.shape[a] != 1 for a in axes):
            self.fail("Squeeze removes the row axis or an axis longer than 1")
        row_axis = data.row_axis - sum(a < data.row_axis for a in axes)
        return _View(np.squeeze(array, axis=tuple(axes)), row_axis, data.is_bool)

    def op_Unsqueeze(self):
        data = self.values[self.node.input[0]]
        array = data.refs if isinstance(data, _View) else np.asarray(data)
        raw = self.constant_input(1, "axes") if len(self.node.input) > 1 else self.attribute("axes")
        rank = array.ndim + len(np.asarray(raw).reshape(-1))
        axes = sorted(int(a) % rank for a in np.asarray(raw).reshape(-1))
        expanded = np.expand_dims(array, axis=tuple(axes))
        if not isinstance(data, _View):
            return expanded
        row_axis = data.row_axis
        for axis in axes:
            if axis <= row_axis:
                row_axis += 1
        return _View(expanded, row_axis, data.is_bool)

    def op_Reshape(self):
        data = self.values[self.node.input[0]]
        shape = [int(s) for s in self.constant_input(1, "the shape of Reshape").reshape(-1)]
        if self.attribute("allowzero", 0) != 0:
            self.fail("Reshape with allowzero is not supported")
        array = data.refs if isinstance(data, _View) else np.asarray(data)
        target = [array.shape[i] if s == 0 else s for i, s in enumerate(shape)]
        if not isinstance(data, _View):
            return array.reshape(target)
        # The row axis has length 1 here; in the graph it is a 0 (copied) or the inferred -1.
        row_out = data.row_axis if (data.row_axis < len(shape) and shape[data.row_axis] == 0) else None
        known = int(np.prod([t for t in target if t != -1]))
        if -1 in target:
            position = target.index(-1)
            if row_out is None and known == array.size:
                row_out = position
                target[position] = 1
            else:
                target[position] = array.size // known if known else 0
        if row_out is None or int(np.prod(target)) != array.size:
            self.fail("Reshape does not keep the row axis")
        if int(np.prod(target[:row_out])) != int(np.prod(array.shape[:data.row_axis])):
            self.fail("Reshape mixes values of different rows")
        return _View(array.reshape(target), row_out, data.is_bool)

    def op_Expand(self):
        data = self.values[self.node.input[0]]
        shape = tuple(int(s) for s in self.constant_input(1, "the shape of Expand").reshape(-1))
        if not isinstance(data, _View):
            return np.broadcast_to(data, np.broadcast_shapes(np.shape(data), shape))
        rank = max(data.refs.ndim, len(shape))
        refs = data.refs.reshape((1,) * (rank - data.refs.ndim) + data.refs.shape)
        row_axis = data.row_axis + rank - data.refs.ndim
        full = (1,) * (rank - len(shape)) + shape
        if full[row_axis] != 1:
            self.fail("Expand changes the row axis")
        return _View(np.broadcast_to(refs, np.broadcast_shapes(refs.shape, full)), row_axis, data.is_bool)

    def op_Transpose(self):
        data = self.values[self.node.input[0]]
        array = data.refs if isinstance(data, _View) else np.asarray(data)
        perm = self.attribute("perm") or list(range(array.ndim))[::-1]
        if not isinstance(data, _View):
            return array.transpose(perm)
        return _View(array.transpose(perm), list(perm).index(data.row_axis), data.is_bool)

    def op_Concat(self):
        operands = [self.values[name] for name in self.node.input]
        views = [o for o in operands if isinstance(o, _View)]
        if not views:
            return np.concatenate(operands, axis=self.attribute("axis"))
        if len(views) != len(operands):
            self.fail("a constant is concatenated with row values")
        axis = self.attribute("axis") % views[0].refs.ndim
        if any(v.row_axis != views[0].row_axis for v in views) or axis == views[0].row_axis:
            self.fail("Concat joins along the row axis")
        if any(v.is_bool != views[0].is_bool for v in views):
            self.fail("Concat joins boolean and FP32 values")
        return _View(np.concatenate([v.refs for v in views], axis=axis), views[0].row_axis, views[0].is_bool)

    def op_Split(self):
        data = self.values[self.node.input[0]]
        array = data.refs if isinstance(data, _View) else np.asarray(data)
        axis = self.attribute("axis", 0) % array.ndim
        if len(self.node.input) > 1 and self.node.input[1]:
            sizes = [int(s) for s in self.constant_input(1, "the sizes of Split").reshape(-1)]
        elif self.attribute("split") is not None:   # opsets before 13
            sizes = [int(s) for s in self.attribute("split")]
        else:
            count = self.attribute("num_outputs") or len(self.node.output)
            sizes = [-(-array.shape[axis] // count)] * count
            sizes[-1] = array.shape[axis] - sum(sizes[:-1])
        if isinstance(data, _View) and axis == data.row_axis:
            self.fail("Split divides the row axis")
        parts = np.split(array, np.cumsum(sizes)[:-1], axis=axis)
        if not isinstance(data, _View):
            return parts
        return [_View(part, data.row_axis, data.is_bool) for part in parts]

    def op_Slice(self):
        data = self.values[self.node.input[0]]
        array = data.refs if isinstance(data, _View) else np.asarray(data)
        starts = self.constant_input(1, "the starts of Slice").reshape(-1)
        ends = self.constant_input(2, "the ends of Slice").reshape(-1)
        axes = self.axes_argument(3, array.ndim) or list(range(len(starts)))
        has_steps = len(self.node.input) > 4 and self.node.input[4]
        steps = self.constant_input(4, "the steps of Slice").reshape(-1) if has_steps else np.ones(len(starts), int)
        sliced = array
        for start, end, axis, step in zip(starts, ends, axes, steps):
            start, end, step = int(start), int(end), int(step)
            if isinstance(data, _View) and axis == data.row_axis:
                if start != 0 or end < _INT64_BIG or step != 1:
                    self.fail("Slice selects rows")
                continue
            if step == 0:
                self.fail("Slice with step 0")
            # ONNX: negative positions count from the end; then start and end are clamped to the axis
            length = array.shape[axis]
            start = start + length if start < 0 else start
            end = end + length if end < 0 else end
            if step > 0:
                start, end = min(max(start, 0), length), min(max(end, 0), length)
            else:
                start, end = min(max(start, 0), length - 1), min(max(end, -1), length - 1)
            sliced = np.take(sliced, np.arange(start, end, step), axis=axis)
        return _View(sliced, data.row_axis, data.is_bool) if isinstance(data, _View) else sliced

    def op_Gather(self):
        data = self.values[self.node.input[0]]
        indices = np.asarray(self.constant_input(1, "the indices of Gather"))
        array = data.refs if isinstance(data, _View) else np.asarray(data)
        axis = self.attribute("axis", 0) % array.ndim
        if isinstance(data, _View) and axis == data.row_axis:
            self.fail("Gather selects rows")
        if ((indices < -array.shape[axis]) | (indices >= array.shape[axis])).any():
            self.fail(f"Gather index outside the axis of length {array.shape[axis]}")
        taken = np.take(array, indices % array.shape[axis], axis=axis)
        if not isinstance(data, _View):
            return taken
        row_axis = data.row_axis + (indices.ndim - 1 if axis < data.row_axis else 0)
        return _View(taken, row_axis, data.is_bool)

    def op_Where(self):
        condition, first, second = (self.values[name] for name in self.node.input)
        if isinstance(condition, _View) and not self.refs_are_const(condition.refs).all():
            return self.elementwise("where", [condition, first, second])
        if not isinstance(first, _View) and not isinstance(second, _View) and not isinstance(condition, _View):
            return np.where(condition, first, second)
        # A constant condition selects references element by element: no stage.
        arrays, row_axis, _ = self.broadcast([condition, first, second])
        chosen = arrays[0]
        if isinstance(condition, _View):
            chosen = self.const_values(arrays[0], True)
        if row_axis is None:
            return np.where(chosen, arrays[1], arrays[2])
        is_bool = [o.is_bool if isinstance(o, _View) else np.asarray(o).dtype == np.bool_ for o in (first, second)]
        if is_bool[0] != is_bool[1]:
            self.fail("Where selects between boolean and FP32 values")
        refs = [a if isinstance(o, _View) else self.const_refs_for(a) for o, a in zip((first, second), arrays[1:])]
        return _View(np.where(chosen.astype(np.bool_), refs[0], refs[1]), row_axis, is_bool[0])

    # ---- sums ------------------------------------------------------------------------------------------------

    def op_MatMul(self):
        left, right = (self.values[name] for name in self.node.input)
        if not isinstance(left, _View) and not isinstance(right, _View):
            return np.matmul(left, right)
        operands = []
        for operand, is_left in ((left, True), (right, False)):
            array = operand.refs if isinstance(operand, _View) else np.asarray(operand, dtype=np.float32)
            row_axis = operand.row_axis if isinstance(operand, _View) else None
            if array.ndim == 1:   # a vector is a row on the left and a column on the right
                array = array[None, :] if is_left else array[:, None]
                row_axis = None if row_axis is None else row_axis + (1 if is_left else 0)
            contracted = array.ndim - 1 if is_left else array.ndim - 2
            if row_axis == contracted:
                self.fail("MatMul sums over the row axis")
            # [..., i, k] and [..., k, j] -> [..., i, j, k] each
            if is_left:
                array = array[..., :, None, :]
                row_axis = row_axis if row_axis is None or row_axis < contracted - 1 else contracted - 1
            else:
                array = np.swapaxes(array, -1, -2)[..., None, :, :]
                if row_axis is not None and row_axis >= contracted:
                    row_axis = array.ndim - 2   # the row axis was the column axis j
            operands.append(_View(array, row_axis) if isinstance(operand, _View) else array)
        arrays, row_axis, shape = self.broadcast(operands)
        vector_left = np.ndim(left.refs if isinstance(left, _View) else left) == 1
        vector_right = np.ndim(right.refs if isinstance(right, _View) else right) == 1
        out_shape, terms = shape[:-1], shape[-1]
        size = int(np.prod(out_shape))
        flat = [np.ascontiguousarray(a).reshape(size, terms) for a in arrays]
        if isinstance(operands[0], _View) and isinstance(operands[1], _View):
            stage = self.add_stage("dot", size, op="dot", operands=flat)
        else:
            refs, weights = (flat[0], flat[1]) if isinstance(operands[0], _View) else (flat[1], flat[0])
            stage = self.dot_with_constants(refs, weights.astype(np.float32))
            if stage is None:
                return self.fail("MatMul of row-independent values with a row axis is not supported")
        view = self.new_view(stage, out_shape, row_axis)
        squeeze = [axis for axis, is_vector in ((-2, vector_left), (-1, vector_right)) if is_vector]
        if squeeze:
            axes = tuple(a % view.refs.ndim for a in squeeze)
            row_axis = view.row_axis - sum(a < view.row_axis for a in axes)
            view = _View(np.squeeze(view.refs, axis=axes), row_axis, False)
        return view

    def dot_with_constants(self, refs: np.ndarray, weights: np.ndarray) -> Stage | None:
        """A dot stage with constant factors. Terms whose row value is a constant for every element are summed
        into the bias here, unless that sum alone is no FP32 number (with the other terms it may be one)."""
        const = self.refs_are_const(refs)
        bias = None
        constant_terms = const.all(axis=0)
        if constant_terms.all():
            return None
        if constant_terms.any():
            values = self.const_values(refs[:, constant_terms], False)
            with np.errstate(over="ignore", invalid="ignore"):
                summed = np.sum(values * weights[:, constant_terms], axis=1, dtype=np.float32)
            if np.isfinite(summed).all():
                bias = summed
                refs, weights = refs[:, ~constant_terms], weights[:, ~constant_terms]
        return self.add_stage("dot", refs.shape[0], op="dot", operands=[np.ascontiguousarray(refs)],
                              weights=np.ascontiguousarray(weights), bias=bias)

    def reduce(self, kind: str):
        data = self.values[self.node.input[0]]
        array = data.refs if isinstance(data, _View) else np.asarray(data)
        keepdims = self.attribute("keepdims", 1)
        axes = self.axes_argument(1, array.ndim)
        if not axes:   # no axes given, or an empty list: every axis, unless the node says "no operation"
            if self.attribute("noop_with_empty_axes", 0):
                return data
            axes = list(range(array.ndim))
        function = {"sum": np.sum, "max": np.max, "min": np.min, "prod": np.prod}[kind]

        def identity(dtype) -> dict:
            """What a reduction over no values gives (ONNX): 0, 1, and for max and min the ends of the type."""
            if kind in ("sum", "prod"):
                return {}
            ends = (-np.inf, np.inf) if np.issubdtype(dtype, np.floating) else (np.iinfo(dtype).min, np.iinfo(dtype).max)
            return {"initial": ends[kind == "min"]}

        if not isinstance(data, _View):
            empty = any(array.shape[axis] == 0 for axis in axes)
            # in the type of the constant, as ONNX reduces (NumPy would widen the sum of INT32 values)
            return function(array, axis=tuple(axes), keepdims=bool(keepdims),
                            **(identity(array.dtype) if empty else {})).astype(array.dtype)
        if data.row_axis in axes:
            self.fail("the reduction runs over the row axis")
        if data.is_bool:
            self.fail("reductions of boolean values are not supported")
        kept = [a for a in range(array.ndim) if a not in axes]
        moved = array.transpose(kept + axes)
        out_shape = moved.shape[:len(kept)]
        size = int(np.prod(out_shape))
        terms = moved.reshape(size, int(np.prod(moved.shape[len(kept):])))   # also of an empty part: no elements
        row_axis = kept.index(data.row_axis)
        if size == 0 or terms.shape[1] == 1:
            refs = terms.reshape(out_shape)
        elif terms.shape[1] == 0:   # an empty part of a Split or Slice
            values = np.zeros((size, 0), dtype=np.float32)
            refs = self.const_refs(function(values, axis=1, **identity(np.float32)).astype(np.float32).reshape(out_shape))
        elif self.refs_are_const(terms).all():
            refs = self.const_refs(function(self.const_values(terms, False), axis=1).reshape(out_shape))
        else:
            stage = self.add_stage("reduce", size, op=kind, operands=[np.ascontiguousarray(terms)])
            refs = make_refs(stage.index, np.arange(size)).reshape(out_shape)
        if keepdims:
            refs = np.expand_dims(refs, axis=tuple(axes))
            row_axis = data.row_axis
        return _View(refs, row_axis, False)


def sub_graph(graph: RowGraph, outputs: np.ndarray) -> RowGraph:
    """The part of a row graph that the outputs at the positions ``outputs`` need, as a graph of its own: the
    elements they depend on, numbered anew. Elements of one stage that read different computed stages of the
    part become stages of their own (the value and a tangent in one stage of copies are then two stages, each on
    one stage of terms). Elements that read fewer of the same stages stay with the others (a padded slot reads a
    constant where they read a value).
    Elements that no output of the whole graph needs (the padded slots of a stage of bands) stay where they are
    computed from elements of the part: its stages keep their regular shapes. Its output row is the whole
    graph's; ``output_slots`` gives the positions of its outputs in it."""
    outputs = np.asarray(outputs, dtype=np.int64)
    seed = graph.output_refs[outputs]
    computed = [stage for stage in sorted(graph.stages, key=lambda stage: stage.index) if stage.kind in ("map", "dot", "reduce")]

    def needed(refs: np.ndarray) -> list[np.ndarray]:
        """Per stage, the elements that the references and what they are computed from need."""
        live = [np.zeros(stage.size, dtype=np.bool_) for stage in graph.stages]

        def mark(refs: np.ndarray) -> None:
            stage_of = ref_stage(refs)
            for index in np.unique(stage_of):
                live[index][ref_offset(refs[stage_of == index])] = True

        mark(refs)
        for stage in reversed(computed):
            if live[stage.index].any():
                for operand in stage.operands:
                    mark(operand[live[stage.index]])
        return live

    live, unused = needed(seed), [~rows for rows in needed(graph.output_refs)]
    given = np.array([stage.kind in ("input", "const") for stage in graph.stages])
    for stage in computed:   # padded slots: unused by the whole graph, computed from elements of the part
        rows = unused[stage.index] & ~live[stage.index]
        of_part = np.zeros(stage.size, dtype=np.bool_)   # reads a computed element of the part, not constants only
        for operand in stage.operands:
            if not rows.any():
                break
            terms = operand.reshape(stage.size, -1)
            source, element = ref_stage(terms), ref_offset(terms)
            there = given[source]
            of_part |= (~np.array([graph.stages[index].kind == "const" for index in source.reshape(-1)]).reshape(source.shape)).any(axis=1)
            for index in np.unique(source[~there]):
                mask = source == index
                there[mask] = live[index][element[mask]]
            rows &= there.all(axis=1)
        rows &= of_part
        if rows.any():
            live[stage.index] |= rows
            for operand in stage.operands:   # their constants
                picked = operand[rows].reshape(-1)
                for index in np.unique(ref_stage(picked)):
                    if graph.stages[index].kind == "const":
                        live[index][ref_offset(picked[ref_stage(picked) == index])] = True
    new_stage = [np.full(stage.size, -1, dtype=np.int64) for stage in graph.stages]
    new_offset = [np.zeros(stage.size, dtype=np.int64) for stage in graph.stages]

    def moved(refs: np.ndarray) -> np.ndarray:
        stage_of, offset = ref_stage(refs), ref_offset(refs)
        result = np.empty(refs.shape, dtype=refs.dtype)
        for index in np.unique(stage_of):
            mask = stage_of == index
            result[mask] = make_refs(new_stage[index][offset[mask]], new_offset[index][offset[mask]])
        return result

    stages: list[Stage] = []
    # the input and the constants first: passes of the schedule append constants behind their readers
    for stage in sorted(graph.stages, key=lambda stage: (stage.kind != "input", stage.kind != "const", stage.index)):
        rows = np.arange(stage.size) if stage.kind == "input" else np.flatnonzero(live[stage.index])
        if len(rows) == 0:
            continue
        groups: dict[tuple, list[int]] = {}
        if stage.kind in ("input", "const"):
            groups[()] = list(rows)
        else:   # by the stages of the part that the element reads, in the order of their first elements
            sources = []
            for refs in stage.operands:
                terms = refs.reshape(stage.size, -1)[rows]
                source, element = ref_stage(terms), ref_offset(terms)
                read = np.full(terms.shape, -1, dtype=np.int64)   # -1: a constant
                for index in np.unique(source):
                    if graph.stages[index].kind != "const":
                        mask = source == index
                        read[mask] = new_stage[index][element[mask]]
                sources.append(read)
            for number, row in enumerate(rows):
                key = frozenset(int(v) for read in sources for v in read[number][read[number] >= 0])
                groups.setdefault(key, []).append(int(row))
            for key in sorted(groups, key=len):   # elements that read fewer of the stages of other elements: with those
                home = [other for other in groups if key < other]
                if len(home) == 1 or (home and not key):
                    groups[home[0]] = sorted(groups[home[0]] + groups.pop(key))
        for number, group in enumerate(groups.values()):
            kept = np.asarray(group, dtype=np.int64)
            index = len(stages)
            new_stage[stage.index][kept] = index
            new_offset[stage.index][kept] = np.arange(len(kept))
            stages.append(Stage(
                index, stage.kind, stage.name if len(groups) == 1 else f"{stage.name}_{number}", len(kept), stage.is_bool,
                stage.op, [np.ascontiguousarray(moved(refs[kept])) for refs in stage.operands],
                None if stage.weights is None else np.ascontiguousarray(stage.weights[kept]),
                None if stage.bias is None else np.ascontiguousarray(stage.bias[kept]),
                None if stage.values is None else np.ascontiguousarray(stage.values[kept]), stage.log2_units))
    slots = outputs if graph.output_slots is None else graph.output_slots[outputs]
    return RowGraph(stages, graph.input_name, graph.output_name, graph.input_shape, graph.output_shape, moved(seed), slots)


def load_row_graph(path: str) -> RowGraph:
    """Read an ONNX file and return its row-wise form."""
    return _Importer(onnx.load(path)).run()


def evaluate(graph: RowGraph, rows: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Reference evaluation of the row graph with NumPy: rows [count, input_size] -> [count, output_size].

    Sums run in NumPy's order, so FP32 results agree with other FP32 implementations only to rounding."""
    rows = np.asarray(rows, dtype=dtype).reshape(len(rows), graph.input_size)   # also of no rows
    count = rows.shape[0]
    values: dict[int, np.ndarray] = {}

    def fetch(refs: np.ndarray) -> np.ndarray:
        stage_of = ref_stage(refs)
        out = None if refs.size else np.empty((count,) + refs.shape, dtype=dtype)   # an output of no elements
        for index in np.unique(stage_of):
            mask = stage_of == index
            part = values[index][:, ref_offset(refs[mask])]
            if out is None:
                out = np.empty((count,) + refs.shape, dtype=part.dtype)
            out[:, mask] = part
        return out

    for stage in graph.stages:   # constants first: a pass of the schedule may have appended some
        if stage.kind == "const":
            constant = stage.values.astype(np.bool_ if stage.is_bool else dtype)
            values[stage.index] = np.broadcast_to(constant, (count, stage.size))
    for stage in graph.stages:
        if stage.kind == "input":
            values[stage.index] = rows
        elif stage.kind == "const":
            pass
        elif stage.kind == "map":
            values[stage.index] = apply_map(stage.op, [fetch(refs) for refs in stage.operands], dtype)
        elif stage.kind == "dot":
            first = fetch(stage.operands[0])
            second = stage.weights.astype(dtype) if stage.weights is not None else fetch(stage.operands[1])
            with np.errstate(all="ignore"):
                total = np.sum(first * second, axis=2)
            values[stage.index] = total if stage.bias is None else total + stage.bias.astype(dtype)
        else:
            terms = fetch(stage.operands[0])
            if stage.op == "logsumexp":
                values[stage.index] = logsumexp(terms, 2, stage.log2_units)
            else:
                function = {"sum": np.sum, "max": np.max, "min": np.min, "prod": np.prod}[stage.op]
                values[stage.index] = function(terms, axis=2)
    return fetch(graph.output_refs)
