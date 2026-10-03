"""Fail-closed, extensible JAXPR to ONNX exporter.

The public entry point accepts an arbitrary pure JAX function and example
arguments.  It traces the function with :func:`jax.make_jaxpr`, translates a
documented set of JAX primitives, and raises :class:`UnsupportedJaxPrimitive`
with the exact equation when a program leaves that set.

This is deliberately not presented as a complete JAX implementation.  The
translator is general over graph topology, tensor shapes, constants, PyTrees,
and the supported primitive set; model-specific exporters should only load
weights and define the JAX forward function.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

import jax
import numpy as np
import onnx
from jax.extend import core as jax_core
from onnx import TensorProto, helper, numpy_helper


class JaxOnnxExportError(RuntimeError):
    """Base class for deterministic export failures."""


class UnsupportedJaxPrimitive(JaxOnnxExportError):
    """Raised when the traced program contains an unsupported equation."""


@dataclass(frozen=True)
class ExportResult:
    path: Path
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    primitive_counts: Mapping[str, int]


@dataclass(frozen=True)
class _Value:
    name: str
    aval: Any
    dynamic_axes: Mapping[int, str]


_UNARY_OPS = {
    "abs": "Abs",
    "acos": "Acos",
    "acosh": "Acosh",
    "asin": "Asin",
    "asinh": "Asinh",
    "atan": "Atan",
    "atanh": "Atanh",
    "ceil": "Ceil",
    "cos": "Cos",
    "cosh": "Cosh",
    "erf": "Erf",
    "exp": "Exp",
    "floor": "Floor",
    "log": "Log",
    "logistic": "Sigmoid",
    "neg": "Neg",
    "round": "Round",
    "sign": "Sign",
    "sin": "Sin",
    "sinh": "Sinh",
    "sqrt": "Sqrt",
    "tan": "Tan",
    "tanh": "Tanh",
}

_BINARY_OPS = {
    "add": "Add",
    "sub": "Sub",
    "mul": "Mul",
    "div": "Div",
    "max": "Max",
    "min": "Min",
    "pow": "Pow",
    "rem": "Mod",
    "eq": "Equal",
    "gt": "Greater",
    "ge": "GreaterOrEqual",
    "lt": "Less",
    "le": "LessOrEqual",
    "and": "And",
    "or": "Or",
    "xor": "Xor",
}

_REDUCTION_OPS = {
    "reduce_sum": "ReduceSum",
    "reduce_max": "ReduceMax",
    "reduce_min": "ReduceMin",
    "reduce_prod": "ReduceProd",
}

_CALL_PRIMITIVES = {
    "call",
    "closed_call",
    "custom_jvp_call",
    "custom_vjp_call_jaxpr",
    "jit",
    "named_call",
    "pjit",
    "remat2",
    "xla_call",
}


def _shape(aval: Any) -> tuple[int, ...]:
    return tuple(int(dimension) for dimension in aval.shape)


def _dtype(aval: Any) -> np.dtype:
    try:
        return np.dtype(aval.dtype)
    except TypeError as error:
        raise JaxOnnxExportError(f"unsupported dtype {aval.dtype!r}") from error


def _onnx_dtype(dtype: Any) -> int:
    try:
        return helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
    except (TypeError, ValueError) as error:
        raise JaxOnnxExportError(f"ONNX has no supported tensor encoding for dtype {dtype!r}") from error


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return cleaned or "value"


def _normalise_names(names: Sequence[str] | None, count: int, prefix: str) -> tuple[str, ...]:
    result = tuple(names) if names is not None else tuple(f"{prefix}_{index}" for index in range(count))
    if len(result) != count:
        raise ValueError(f"expected {count} {prefix} names, received {len(result)}")
    if len(set(result)) != len(result):
        raise ValueError(f"{prefix} names must be unique")
    return tuple(_safe_name(name) for name in result)


class _Translator:
    def __init__(
        self,
        *,
        opset: int,
        dynamic_axes: Mapping[str, Mapping[int, str]],
        input_names: Sequence[str],
        custom_calls: Mapping[str, tuple[str, str]] | None = None,
    ) -> None:
        if opset < 18:
            raise ValueError("the generic exporter requires ONNX opset >= 18")
        self.opset = opset
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.env: dict[Any, _Value] = {}
        self.counter = 0
        self.primitive_counts: dict[str, int] = {}
        self.symbol_sources: dict[str, tuple[str, int, int]] = {}
        self.custom_calls = dict(custom_calls or {})
        self.custom_domains: set[str] = set()
        for name in input_names:
            for axis, symbol in dynamic_axes.get(name, {}).items():
                if symbol in self.symbol_sources:
                    continue
                self.symbol_sources[symbol] = (name, int(axis), -1)

    def fresh(self, prefix: str) -> str:
        self.counter += 1
        return f"{_safe_name(prefix)}_{self.counter:05d}"

    def add_initializer(self, value: Any, prefix: str, dtype: Any | None = None) -> str:
        array = np.asarray(value, dtype=dtype)
        _onnx_dtype(array.dtype)
        name = self.fresh(prefix)
        self.initializers.append(numpy_helper.from_array(array, name=name))
        return name

    def literal(self, atom: jax_core.Literal) -> _Value:
        array = np.asarray(atom.val, dtype=_dtype(atom.aval))
        return _Value(self.add_initializer(array, "literal"), atom.aval, {})

    def read(self, atom: Any) -> _Value:
        if isinstance(atom, jax_core.Literal):
            return self.literal(atom)
        try:
            return self.env[atom]
        except KeyError as error:
            raise JaxOnnxExportError(f"unbound JAX variable {atom}") from error

    def bind(self, variable: Any, value: _Value) -> None:
        if isinstance(variable, jax_core.DropVar):
            return
        self.env[variable] = value

    def make_value(
        self, outvar: Any, prefix: str, dynamic_axes: Mapping[int, str] | None = None
    ) -> _Value:
        return _Value(self.fresh(prefix), outvar.aval, dict(dynamic_axes or {}))

    def add_node(
        self,
        op_type: str,
        inputs: Sequence[str],
        outputs: Sequence[str],
        domain: str | None = None,
        **attributes: Any,
    ) -> None:
        self.nodes.append(helper.make_node(op_type, list(inputs), list(outputs), domain=domain, **attributes))

    def _set_input_symbol_examples(
        self, input_values: Sequence[_Value], input_names: Sequence[str]
    ) -> None:
        self.input_shapes = {name: _shape(value.aval) for value, name in zip(input_values, input_names)}
        for value, name in zip(input_values, input_names):
            for axis, symbol in value.dynamic_axes.items():
                source = self.symbol_sources.get(symbol)
                if source is None or source[2] < 0:
                    self.symbol_sources[symbol] = (name, axis, _shape(value.aval)[axis])

    def infer_symbol(self, dimension: int, axis: int) -> str | None:
        candidates = [
            (symbol, source_axis)
            for symbol, (_, source_axis, example_dimension) in self.symbol_sources.items()
            if example_dimension == dimension
        ]
        if len(candidates) == 1:
            return candidates[0][0]
        same_axis = [symbol for symbol, source_axis in candidates if source_axis == axis]
        return same_axis[0] if len(same_axis) == 1 else None

    def shape_tensor(
        self, shape: Sequence[int], dynamic_axes: Mapping[int, str], prefix: str
    ) -> str:
        if not dynamic_axes:
            return self.add_initializer(np.asarray(shape, dtype=np.int64), f"{prefix}_shape")
        pieces: list[str] = []
        for axis, dimension in enumerate(shape):
            symbol = dynamic_axes.get(axis)
            if symbol is None:
                pieces.append(
                    self.add_initializer(np.asarray([dimension], dtype=np.int64), f"{prefix}_dim")
                )
                continue
            if symbol not in self.symbol_sources:
                raise JaxOnnxExportError(f"no graph input provides dynamic dimension {symbol!r}")
            source_name, source_axis, _ = self.symbol_sources[symbol]
            source_shape = self.fresh(f"{prefix}_{symbol}_source_shape")
            self.add_node("Shape", [source_name], [source_shape])
            index = self.add_initializer(
                np.asarray([source_axis], dtype=np.int64), f"{prefix}_{symbol}_axis"
            )
            gathered = self.fresh(f"{prefix}_{symbol}_dim")
            self.add_node("Gather", [source_shape, index], [gathered], axis=0)
            pieces.append(gathered)
        if len(pieces) == 1:
            return pieces[0]
        output = self.fresh(f"{prefix}_dynamic_shape")
        self.add_node("Concat", pieces, [output], axis=0)
        return output

    def elementwise_dynamic_axes(
        self, inputs: Sequence[_Value], output_shape: Sequence[int]
    ) -> dict[int, str]:
        result: dict[int, str] = {}
        output_rank = len(output_shape)
        for value in inputs:
            input_shape = _shape(value.aval)
            shift = output_rank - len(input_shape)
            if shift < 0:
                continue
            for input_axis, symbol in value.dynamic_axes.items():
                output_axis = shift + input_axis
                if input_shape[input_axis] == 1 and output_shape[output_axis] != 1:
                    continue
                previous = result.get(output_axis)
                if previous is not None and previous != symbol:
                    raise JaxOnnxExportError(
                        f"incompatible dynamic dimensions {previous!r} and {symbol!r}"
                    )
                result[output_axis] = symbol
        for axis, dimension in enumerate(output_shape):
            if axis not in result:
                symbol = self.infer_symbol(dimension, axis)
                if symbol is not None and axis == 0:
                    result[axis] = symbol
        return result

    def translate_jaxpr(
        self,
        jaxpr: jax_core.Jaxpr,
        *,
        input_values: Sequence[_Value],
        const_values: Sequence[_Value],
    ) -> list[_Value]:
        if len(jaxpr.invars) != len(input_values):
            raise JaxOnnxExportError("nested JAXPR input count mismatch")
        if len(jaxpr.constvars) != len(const_values):
            raise JaxOnnxExportError("nested JAXPR constant count mismatch")
        for variable, value in zip(jaxpr.constvars, const_values):
            self.bind(variable, value)
        for variable, value in zip(jaxpr.invars, input_values):
            self.bind(variable, value)
        for equation in jaxpr.eqns:
            primitive = equation.primitive.name
            self.primitive_counts[primitive] = self.primitive_counts.get(primitive, 0) + 1
            try:
                outputs = self.translate_equation(equation)
            except UnsupportedJaxPrimitive:
                raise
            except Exception as error:
                raise JaxOnnxExportError(
                    f"failed to translate JAX primitive {primitive!r}: {equation}"
                ) from error
            if len(outputs) != len(equation.outvars):
                raise JaxOnnxExportError(f"output count mismatch while translating {equation}")
            for variable, value in zip(equation.outvars, outputs):
                self.bind(variable, value)
        return [self.read(variable) for variable in jaxpr.outvars]

    def unsupported(self, equation: Any, reason: str = "") -> UnsupportedJaxPrimitive:
        suffix = f": {reason}" if reason else ""
        return UnsupportedJaxPrimitive(
            f"unsupported JAX primitive {equation.primitive.name!r}{suffix}\nEquation: {equation}"
        )


    def value_info(self, value: _Value) -> onnx.ValueInfoProto:
        dimensions = [
            value.dynamic_axes.get(axis, dimension)
            for axis, dimension in enumerate(_shape(value.aval))
        ]
        return helper.make_tensor_value_info(
            value.name, _onnx_dtype(_dtype(value.aval)), dimensions
        )

    def translate_subgraph(
        self,
        jaxpr: jax_core.Jaxpr,
        *,
        input_values: Sequence[_Value],
        const_values: Sequence[_Value],
        graph_inputs: Sequence[_Value],
        name: str,
    ) -> tuple[onnx.GraphProto, list[_Value]]:
        saved_nodes = self.nodes
        saved_env = self.env
        self.nodes = []
        self.env = dict(saved_env)
        try:
            translated = self.translate_jaxpr(
                jaxpr, input_values=input_values, const_values=const_values
            )
            outputs: list[_Value] = []
            for value in translated:
                output = _Value(
                    self.fresh(f"{name}_output"), value.aval, value.dynamic_axes
                )
                self.add_node("Identity", [value.name], [output.name])
                outputs.append(output)
            nodes = self.nodes
        finally:
            self.nodes = saved_nodes
            self.env = saved_env
        graph = helper.make_graph(
            nodes,
            _safe_name(name),
            [self.value_info(value) for value in graph_inputs],
            [self.value_info(value) for value in outputs],
        )
        return graph, outputs

    def nested_jaxpr_parts(
        self, candidate: Any, equation: Any
    ) -> tuple[jax_core.Jaxpr, list[_Value]]:
        if isinstance(candidate, jax_core.ClosedJaxpr):
            constants = [
                _Value(self.add_initializer(value, "nested_constant"), variable.aval, {})
                for variable, value in zip(candidate.jaxpr.constvars, candidate.consts)
            ]
            return candidate.jaxpr, constants
        if isinstance(candidate, jax_core.Jaxpr):
            if candidate.constvars:
                raise self.unsupported(
                    equation, "an open nested JAXPR with constvars is unsupported"
                )
            return candidate, []
        raise self.unsupported(equation, "no nested JAXPR was found")

    def translate_inline_jaxpr(
        self,
        jaxpr: jax_core.Jaxpr,
        *,
        input_values: Sequence[_Value],
        const_values: Sequence[_Value],
    ) -> list[_Value]:
        saved_env = self.env
        self.env = dict(saved_env)
        try:
            return self.translate_jaxpr(
                jaxpr, input_values=input_values, const_values=const_values
            )
        finally:
            self.env = saved_env

    def index_mode(self, equation: Any) -> str:
        mode = equation.params.get("mode")
        return getattr(mode, "name", str(mode).split(".")[-1])

    def cast_indices_to_i64(self, value: _Value, prefix: str) -> str:
        if _dtype(value.aval) == np.dtype(np.int64):
            return value.name
        if not np.issubdtype(_dtype(value.aval), np.integer):
            raise JaxOnnxExportError(
                f"{prefix} indices must be integer, received {_dtype(value.aval)}"
            )
        output = self.fresh(f"{prefix}_i64")
        self.add_node(
            "Cast", [value.name], [output], to=_onnx_dtype(np.dtype(np.int64))
        )
        return output

    def normalize_index_tensor(
        self,
        indices_name: str,
        maxima: Sequence[int],
        mode: str,
        prefix: str,
    ) -> tuple[str, str | None]:
        if mode == "PROMISE_IN_BOUNDS":
            return indices_name, None
        if mode not in {"CLIP", "FILL_OR_DROP"}:
            raise JaxOnnxExportError(f"{prefix} mode {mode!r} is unsupported")
        zero = self.add_initializer(
            np.zeros(len(maxima), dtype=np.int64), f"{prefix}_zero"
        )
        upper = self.add_initializer(
            np.asarray(maxima, dtype=np.int64), f"{prefix}_upper"
        )
        nonnegative = self.fresh(f"{prefix}_nonnegative")
        clipped = self.fresh(f"{prefix}_clipped")
        self.add_node("Max", [indices_name, zero], [nonnegative])
        self.add_node("Min", [nonnegative, upper], [clipped])
        if mode == "CLIP":
            return clipped, None
        at_least_zero = self.fresh(f"{prefix}_at_least_zero")
        at_most_upper = self.fresh(f"{prefix}_at_most_upper")
        valid_components = self.fresh(f"{prefix}_valid_components")
        self.add_node("GreaterOrEqual", [indices_name, zero], [at_least_zero])
        self.add_node("LessOrEqual", [indices_name, upper], [at_most_upper])
        self.add_node("And", [at_least_zero, at_most_upper], [valid_components])
        valid_integer = self.fresh(f"{prefix}_valid_integer")
        self.add_node(
            "Cast",
            [valid_components],
            [valid_integer],
            to=_onnx_dtype(np.dtype(np.int64)),
        )
        axes = self.add_initializer(
            np.asarray([-1], dtype=np.int64), f"{prefix}_component_axis"
        )
        valid_reduced = self.fresh(f"{prefix}_valid_reduced")
        self.add_node(
            "ReduceMin", [valid_integer, axes], [valid_reduced], keepdims=0
        )
        valid = self.fresh(f"{prefix}_valid")
        self.add_node(
            "Cast", [valid_reduced], [valid], to=_onnx_dtype(np.dtype(np.bool_))
        )
        return clipped, valid

    def broadcast_trailing_mask(
        self, mask: str, trailing_rank: int, prefix: str
    ) -> str:
        if trailing_rank == 0:
            return mask
        axes = self.add_initializer(
            np.arange(-trailing_rank, 0, dtype=np.int64),
            f"{prefix}_mask_axes",
        )
        output = self.fresh(f"{prefix}_mask")
        self.add_node("Unsqueeze", [mask, axes], [output])
        return output

    def translate_equation(self, equation: Any) -> list[_Value]:
        primitive = equation.primitive.name
        inputs = [self.read(variable) for variable in equation.invars]
        if primitive in _CALL_PRIMITIVES:
            return self.translate_call(equation, inputs)
        if primitive == "cond":
            return self.translate_cond(equation, inputs)
        if primitive == "scan":
            return self.translate_scan(equation, inputs)
        if primitive == "while":
            return self.translate_while(equation, inputs)
        if primitive == "top_k":
            return self.translate_top_k(equation, inputs)
        if primitive == "ffi_call":
            return self.translate_ffi_call(equation, inputs)
        if len(equation.outvars) != 1:
            raise self.unsupported(equation, "multiple outputs are not implemented")
        outvar = equation.outvars[0]

        if primitive in _UNARY_OPS:
            dynamic = dict(inputs[0].dynamic_axes)
            output = self.make_value(outvar, primitive, dynamic)
            self.add_node(_UNARY_OPS[primitive], [inputs[0].name], [output.name])
            return [output]
        if primitive in _BINARY_OPS:
            dynamic = self.elementwise_dynamic_axes(inputs, _shape(outvar.aval))
            output = self.make_value(outvar, primitive, dynamic)
            attributes = {"fmod": 1} if primitive == "rem" else {}
            self.add_node(
                _BINARY_OPS[primitive], [value.name for value in inputs], [output.name], **attributes
            )
            return [output]
        if primitive == "ne":
            dynamic = self.elementwise_dynamic_axes(inputs, _shape(outvar.aval))
            equal = self.fresh("equal")
            output = self.make_value(outvar, "not_equal", dynamic)
            self.add_node("Equal", [value.name for value in inputs], [equal])
            self.add_node("Not", [equal], [output.name])
            return [output]
        if primitive == "not":
            output = self.make_value(outvar, "not", inputs[0].dynamic_axes)
            self.add_node("Not", [inputs[0].name], [output.name])
            return [output]
        if primitive in {"copy", "device_put", "stop_gradient"}:
            output = self.make_value(outvar, primitive, inputs[0].dynamic_axes)
            self.add_node("Identity", [inputs[0].name], [output.name])
            return [output]
        if primitive == "is_finite":
            dynamic = dict(inputs[0].dynamic_axes)
            is_inf = self.fresh("is_inf")
            is_nan = self.fresh("is_nan")
            invalid = self.fresh("invalid")
            output = self.make_value(outvar, "is_finite", dynamic)
            self.add_node("IsInf", [inputs[0].name], [is_inf])
            self.add_node("IsNaN", [inputs[0].name], [is_nan])
            self.add_node("Or", [is_inf, is_nan], [invalid])
            self.add_node("Not", [invalid], [output.name])
            return [output]
        if primitive in {"log1p", "expm1", "rsqrt"}:
            return [self.translate_composite_unary(primitive, inputs[0], outvar)]
        if primitive == "integer_pow":
            exponent = equation.params["y"]
            exponent_name = self.add_initializer(
                np.asarray(exponent, dtype=_dtype(inputs[0].aval)), "integer_power"
            )
            output = self.make_value(outvar, "integer_pow", inputs[0].dynamic_axes)
            self.add_node("Pow", [inputs[0].name, exponent_name], [output.name])
            return [output]
        if primitive == "convert_element_type":
            output = self.make_value(outvar, "cast", inputs[0].dynamic_axes)
            self.add_node("Cast", [inputs[0].name], [output.name], to=_onnx_dtype(_dtype(outvar.aval)))
            return [output]
        if primitive == "select_n":
            if len(inputs) != 3 or _dtype(inputs[0].aval) != np.dtype(np.bool_):
                raise self.unsupported(equation, "only boolean two-way select_n is supported")
            dynamic = self.elementwise_dynamic_axes(inputs, _shape(outvar.aval))
            output = self.make_value(outvar, "where", dynamic)
            self.add_node("Where", [inputs[0].name, inputs[2].name, inputs[1].name], [output.name])
            return [output]
        if primitive == "clamp":
            if len(inputs) != 3:
                raise self.unsupported(equation, "clamp requires min, value, max")
            dynamic = self.elementwise_dynamic_axes(inputs, _shape(outvar.aval))
            output = self.make_value(outvar, "clip", dynamic)
            self.add_node("Clip", [inputs[1].name, inputs[0].name, inputs[2].name], [output.name])
            return [output]
        if primitive in _REDUCTION_OPS:
            return [self.translate_reduction(primitive, equation, inputs[0], outvar)]
        if primitive in {"reduce_window_max", "reduce_window_sum"}:
            return [self.translate_pooling(primitive, equation, inputs[0], outvar)]
        if primitive == "reduce_window":
            return [self.translate_general_reduce_window(equation, inputs, outvar)]
        if primitive == "cumsum":
            return [self.translate_cumsum(equation, inputs[0], outvar)]
        if primitive in {"argmax", "argmin"}:
            axes = tuple(equation.params.get("axes", (equation.params.get("axis"),)))
            if len(axes) != 1:
                raise self.unsupported(equation, "ONNX ArgMax/ArgMin requires one reduction axis")
            axis = int(axes[0])
            dynamic = self.reduced_dynamic_axes(inputs[0], (axis,))
            output = self.make_value(outvar, primitive, dynamic)
            self.add_node(
                "ArgMax" if primitive == "argmax" else "ArgMin",
                [inputs[0].name],
                [output.name],
                axis=axis,
                keepdims=0,
                select_last_index=0,
            )
            return [output]
        if primitive == "broadcast_in_dim":
            return [self.translate_broadcast(equation, inputs[0], outvar)]
        if primitive == "reshape":
            return [self.translate_reshape(equation, inputs[0], outvar)]
        if primitive == "squeeze":
            axes = tuple(int(axis) for axis in equation.params["dimensions"])
            dynamic = self.reduced_dynamic_axes(inputs[0], axes)
            output = self.make_value(outvar, "squeeze", dynamic)
            axes_name = self.add_initializer(np.asarray(axes, dtype=np.int64), "squeeze_axes")
            self.add_node("Squeeze", [inputs[0].name, axes_name], [output.name])
            return [output]
        if primitive == "transpose":
            permutation = tuple(int(axis) for axis in equation.params["permutation"])
            dynamic = {
                permutation.index(axis): symbol for axis, symbol in inputs[0].dynamic_axes.items()
            }
            output = self.make_value(outvar, "transpose", dynamic)
            self.add_node("Transpose", [inputs[0].name], [output.name], perm=permutation)
            return [output]
        if primitive == "slice":
            return [self.translate_slice(equation, inputs[0], outvar)]
        if primitive == "dynamic_slice":
            return [self.translate_dynamic_slice(equation, inputs, outvar)]
        if primitive == "dynamic_update_slice":
            return [self.translate_dynamic_update_slice(equation, inputs, outvar)]
        if primitive == "rev":
            return [self.translate_reverse(equation, inputs[0], outvar)]
        if primitive == "concatenate":
            return [self.translate_concatenate(equation, inputs, outvar)]
        if primitive == "stack":
            return [self.translate_stack(equation, inputs, outvar)]
        if primitive == "pad":
            return [self.translate_pad(equation, inputs, outvar)]
        if primitive == "gather":
            return [self.translate_gather(equation, inputs, outvar)]
        if primitive in {"scatter", "scatter-add"}:
            return [self.translate_scatter(primitive, equation, inputs, outvar)]
        if primitive == "dot_general":
            return [self.translate_dot_general(equation, inputs, outvar)]
        if primitive == "conv_general_dilated":
            return [self.translate_convolution(equation, inputs, outvar)]
        raise self.unsupported(equation)


    def translate_cond(self, equation: Any, inputs: Sequence[_Value]) -> list[_Value]:
        branches = equation.params.get("branches", ())
        if len(branches) != 2 or not inputs:
            raise self.unsupported(equation, "only a two-branch cond is supported")
        selector = inputs[0]
        if _shape(selector.aval):
            raise self.unsupported(equation, "the cond selector must be scalar")
        if _dtype(selector.aval) == np.dtype(np.bool_):
            condition_name = selector.name
        elif np.issubdtype(_dtype(selector.aval), np.integer):
            true_index = self.add_initializer(
                np.asarray(1, dtype=_dtype(selector.aval)), "cond_true_index"
            )
            condition_name = self.fresh("cond_condition")
            self.add_node("Equal", [selector.name, true_index], [condition_name])
        else:
            raise self.unsupported(equation, "the cond selector must be boolean or integer")

        branch_graphs: list[onnx.GraphProto] = []
        branch_outputs: list[list[_Value]] = []
        for index, candidate in enumerate(branches):
            branch_jaxpr, constants = self.nested_jaxpr_parts(candidate, equation)
            if len(branch_jaxpr.invars) != len(inputs) - 1:
                raise self.unsupported(equation, "cond operand count mismatch")
            graph, outputs = self.translate_subgraph(
                branch_jaxpr,
                input_values=inputs[1:],
                const_values=constants,
                graph_inputs=(),
                name=f"cond_branch_{index}",
            )
            branch_graphs.append(graph)
            branch_outputs.append(outputs)
        if any(len(outputs) != len(equation.outvars) for outputs in branch_outputs):
            raise self.unsupported(equation, "cond branch output count mismatch")

        outputs: list[_Value] = []
        for output_index, outvar in enumerate(equation.outvars):
            dynamic: dict[int, str] = {}
            for branch in branch_outputs:
                for axis, symbol in branch[output_index].dynamic_axes.items():
                    previous = dynamic.get(axis)
                    if previous is not None and previous != symbol:
                        raise self.unsupported(
                            equation, "cond branches have incompatible dynamic dimensions"
                        )
                    dynamic[axis] = symbol
            outputs.append(self.make_value(outvar, "cond", dynamic))
        self.add_node(
            "If",
            [condition_name],
            [value.name for value in outputs],
            else_branch=branch_graphs[0],
            then_branch=branch_graphs[1],
        )
        return outputs

    def translate_scan(self, equation: Any, inputs: Sequence[_Value]) -> list[_Value]:
        candidate = equation.params.get("jaxpr")
        body_jaxpr, body_constants = self.nested_jaxpr_parts(candidate, equation)
        if "num_consts" in equation.params:
            number_of_constants = int(equation.params["num_consts"])
            number_of_carries = int(equation.params["num_carry"])
        else:
            flattened_inputs = equation.params.get("ft_in")
            sections = getattr(flattened_inputs, "elts", flattened_inputs)
            if not isinstance(sections, tuple) or len(sections) != 3:
                raise self.unsupported(equation, "scan carry metadata is unavailable")
            number_of_constants = len(sections[0])
            number_of_carries = len(sections[1])
        number_of_scan_inputs = len(inputs) - number_of_constants - number_of_carries
        if number_of_scan_inputs < 0:
            raise self.unsupported(equation, "scan input metadata is inconsistent")
        if len(body_jaxpr.invars) != len(inputs):
            raise self.unsupported(equation, "scan body input count mismatch")
        number_of_scan_outputs = len(equation.outvars) - number_of_carries
        if number_of_scan_outputs < 0 or len(body_jaxpr.outvars) != len(equation.outvars):
            raise self.unsupported(equation, "scan body output count mismatch")

        captured = list(inputs[:number_of_constants])
        carries = list(inputs[number_of_constants : number_of_constants + number_of_carries])
        scan_inputs = list(inputs[number_of_constants + number_of_carries :])
        body_inputs: list[_Value] = []
        body_offset = number_of_constants
        for index, outer in enumerate(carries):
            variable = body_jaxpr.invars[body_offset + index]
            body_inputs.append(
                _Value(self.fresh("scan_state_input"), variable.aval, outer.dynamic_axes)
            )
        body_offset += number_of_carries
        for index, outer in enumerate(scan_inputs):
            variable = body_jaxpr.invars[body_offset + index]
            element_dynamic = {
                axis - 1: symbol
                for axis, symbol in outer.dynamic_axes.items()
                if axis > 0
            }
            body_inputs.append(
                _Value(self.fresh("scan_element_input"), variable.aval, element_dynamic)
            )

        onnx_scan_inputs = list(scan_inputs)
        graph_inputs = list(body_inputs)
        if not onnx_scan_inputs:
            length = int(equation.params.get("length", -1))
            if length < 0:
                raise self.unsupported(
                    equation, "scan without an input sequence requires fixed length"
                )
            range_start = self.add_initializer(
                np.asarray(0, dtype=np.int64), "scan_range_start"
            )
            range_limit = self.add_initializer(
                np.asarray(length, dtype=np.int64), "scan_range_limit"
            )
            range_step = self.add_initializer(
                np.asarray(1, dtype=np.int64), "scan_range_step"
            )
            dummy_sequence_name = self.fresh("scan_dummy_sequence")
            self.add_node(
                "Range",
                [range_start, range_limit, range_step],
                [dummy_sequence_name],
            )
            onnx_scan_inputs.append(
                _Value(
                    dummy_sequence_name,
                    jax.ShapeDtypeStruct((length,), np.int64),
                    {},
                )
            )
            graph_inputs.append(
                _Value(
                    self.fresh("scan_dummy_element"),
                    jax.ShapeDtypeStruct((), np.int64),
                    {},
                )
            )

        body_graph, body_outputs = self.translate_subgraph(
            body_jaxpr,
            input_values=[*captured, *body_inputs],
            const_values=body_constants,
            graph_inputs=graph_inputs,
            name="scan_body",
        )

        for carry_index in range(number_of_carries):
            if _shape(body_outputs[carry_index].aval) != _shape(carries[carry_index].aval):
                raise self.unsupported(equation, "scan carry shape changes across iterations")
        sequence_symbol: str | None = None
        for value in scan_inputs:
            candidate_symbol = value.dynamic_axes.get(0)
            if candidate_symbol is None:
                continue
            if sequence_symbol is not None and sequence_symbol != candidate_symbol:
                raise self.unsupported(equation, "scan inputs have incompatible sequence lengths")
            sequence_symbol = candidate_symbol

        outputs: list[_Value] = []
        for index, outvar in enumerate(equation.outvars):
            if index < number_of_carries:
                dynamic = dict(body_outputs[index].dynamic_axes)
            else:
                dynamic = {
                    axis + 1: symbol
                    for axis, symbol in body_outputs[index].dynamic_axes.items()
                }
                if sequence_symbol is not None:
                    dynamic[0] = sequence_symbol
            outputs.append(self.make_value(outvar, "scan", dynamic))
        reverse = 1 if bool(equation.params.get("reverse", False)) else 0
        onnx_scan_count = len(onnx_scan_inputs)
        attributes: dict[str, Any] = {
            "body": body_graph,
            "num_scan_inputs": onnx_scan_count,
            "scan_input_axes": [0] * onnx_scan_count,
            "scan_input_directions": [reverse] * onnx_scan_count,
        }
        if number_of_scan_outputs:
            attributes["scan_output_axes"] = [0] * number_of_scan_outputs
            attributes["scan_output_directions"] = [reverse] * number_of_scan_outputs
        self.add_node(
            "Scan",
            [value.name for value in [*carries, *onnx_scan_inputs]],
            [value.name for value in outputs],
            **attributes,
        )
        return outputs

    def translate_while(self, equation: Any, inputs: Sequence[_Value]) -> list[_Value]:
        condition_jaxpr, condition_internal_constants = self.nested_jaxpr_parts(
            equation.params.get("cond_jaxpr"), equation
        )
        body_jaxpr, body_internal_constants = self.nested_jaxpr_parts(
            equation.params.get("body_jaxpr"), equation
        )
        number_of_condition_constants = int(equation.params.get("cond_nconsts", 0))
        number_of_body_constants = int(equation.params.get("body_nconsts", 0))
        number_of_carries = (
            len(inputs) - number_of_condition_constants - number_of_body_constants
        )
        if number_of_carries < 1 or len(equation.outvars) != number_of_carries:
            raise self.unsupported(equation, "while carry metadata is inconsistent")
        condition_constants = list(inputs[:number_of_condition_constants])
        body_constants = list(
            inputs[
                number_of_condition_constants :
                number_of_condition_constants + number_of_body_constants
            ]
        )
        carries = list(inputs[number_of_condition_constants + number_of_body_constants :])
        if len(condition_jaxpr.invars) != number_of_condition_constants + number_of_carries:
            raise self.unsupported(equation, "while condition input count mismatch")
        if len(body_jaxpr.invars) != number_of_body_constants + number_of_carries:
            raise self.unsupported(equation, "while body input count mismatch")
        if len(body_jaxpr.outvars) != number_of_carries:
            raise self.unsupported(equation, "while body output count mismatch")

        initial_condition_outputs = self.translate_inline_jaxpr(
            condition_jaxpr,
            input_values=[*condition_constants, *carries],
            const_values=condition_internal_constants,
        )
        if (
            len(initial_condition_outputs) != 1
            or _shape(initial_condition_outputs[0].aval)
            or _dtype(initial_condition_outputs[0].aval) != np.dtype(np.bool_)
        ):
            raise self.unsupported(equation, "while condition must return one scalar boolean")

        saved_nodes = self.nodes
        saved_env = self.env
        self.nodes = []
        self.env = dict(saved_env)
        try:
            iteration = _Value(
                self.fresh("loop_iteration"),
                jax.ShapeDtypeStruct((), np.int64),
                {},
            )
            condition_input = _Value(
                self.fresh("loop_condition_input"),
                jax.ShapeDtypeStruct((), np.bool_),
                {},
            )
            carry_inputs = [
                _Value(
                    self.fresh("loop_carry_input"),
                    body_jaxpr.invars[number_of_body_constants + index].aval,
                    carry.dynamic_axes,
                )
                for index, carry in enumerate(carries)
            ]
            body_outputs = self.translate_jaxpr(
                body_jaxpr,
                input_values=[*body_constants, *carry_inputs],
                const_values=body_internal_constants,
            )
            condition_outputs = self.translate_jaxpr(
                condition_jaxpr,
                input_values=[*condition_constants, *body_outputs],
                const_values=condition_internal_constants,
            )
            if (
                len(condition_outputs) != 1
                or _shape(condition_outputs[0].aval)
                or _dtype(condition_outputs[0].aval) != np.dtype(np.bool_)
            ):
                raise self.unsupported(
                    equation, "while condition must return one scalar boolean"
                )
            local_outputs: list[_Value] = []
            for prefix, value in [
                ("loop_condition_output", condition_outputs[0]),
                *[("loop_carry_output", value) for value in body_outputs],
            ]:
                output = _Value(
                    self.fresh(prefix), value.aval, value.dynamic_axes
                )
                self.add_node("Identity", [value.name], [output.name])
                local_outputs.append(output)
            body_nodes = self.nodes
        finally:
            self.nodes = saved_nodes
            self.env = saved_env

        for index, value in enumerate(body_outputs):
            if _shape(value.aval) != _shape(carries[index].aval):
                raise self.unsupported(
                    equation, "while carry shape changes across iterations"
                )
            if _dtype(value.aval) != _dtype(carries[index].aval):
                raise self.unsupported(
                    equation, "while carry dtype changes across iterations"
                )
        loop_body = helper.make_graph(
            body_nodes,
            "while_body",
            [
                self.value_info(iteration),
                self.value_info(condition_input),
                *[self.value_info(value) for value in carry_inputs],
            ],
            [self.value_info(value) for value in local_outputs],
        )
        outputs = [
            self.make_value(outvar, "while", carry.dynamic_axes)
            for outvar, carry in zip(equation.outvars, carries)
        ]
        self.add_node(
            "Loop",
            ["", initial_condition_outputs[0].name, *[value.name for value in carries]],
            [value.name for value in outputs],
            body=loop_body,
        )
        return outputs

    def translate_top_k(self, equation: Any, inputs: Sequence[_Value]) -> list[_Value]:
        if len(inputs) != 1 or len(equation.outvars) != 2:
            raise self.unsupported(equation, "top_k expects one input and two outputs")
        axis = int(equation.params.get("axis", -1)) % len(_shape(inputs[0].aval))
        count = int(equation.params["k"])
        count_name = self.add_initializer(np.asarray([count], dtype=np.int64), "top_k_count")
        dynamic = {
            input_axis: symbol
            for input_axis, symbol in inputs[0].dynamic_axes.items()
            if input_axis != axis
        }
        values = self.make_value(equation.outvars[0], "top_k_values", dynamic)
        raw_indices = self.fresh("top_k_indices_i64")
        indices = self.make_value(equation.outvars[1], "top_k_indices", dynamic)
        self.add_node(
            "TopK",
            [inputs[0].name, count_name],
            [values.name, raw_indices],
            axis=axis,
            largest=1,
            sorted=1,
        )
        self.add_node(
            "Cast", [raw_indices], [indices.name], to=_onnx_dtype(_dtype(indices.aval))
        )
        return [values, indices]

    def translate_general_reduce_window(
        self, equation: Any, inputs: Sequence[_Value], outvar: Any
    ) -> _Value:
        if len(inputs) != 2 or len(equation.invars) != 2:
            raise self.unsupported(
                equation, "pooling reduce_window expects an operand and scalar identity"
            )
        reducer, constants = self.nested_jaxpr_parts(
            equation.params.get("jaxpr"), equation
        )
        if constants or len(reducer.invars) != 2 or len(reducer.outvars) != 1:
            raise self.unsupported(equation, "reduce_window reducer signature is unsupported")
        if len(reducer.eqns) != 1:
            raise self.unsupported(equation, "reduce_window reducer must be one operation")
        reducer_equation = reducer.eqns[0]
        reducer_primitive = reducer_equation.primitive.name
        if (
            len(reducer_equation.invars) != 2
            or len(reducer_equation.outvars) != 1
            or reducer_equation.outvars[0] != reducer.outvars[0]
            or set(reducer_equation.invars) != set(reducer.invars)
        ):
            raise self.unsupported(equation, "reduce_window reducer data flow is unsupported")
        identity_atom = equation.invars[1]
        if not isinstance(identity_atom, jax_core.Literal):
            raise self.unsupported(equation, "reduce_window identity must be constant")
        identity = np.asarray(identity_atom.val)
        if reducer_primitive == "add" and np.all(identity == 0):
            primitive = "reduce_window_sum"
        elif reducer_primitive == "max" and (
            np.all(np.isneginf(identity))
            or (
                np.issubdtype(identity.dtype, np.integer)
                and np.all(identity == np.iinfo(identity.dtype).min)
            )
        ):
            primitive = "reduce_window_max"
        else:
            raise self.unsupported(
                equation, "only additive or maximum pooling reducers are supported"
            )
        return self.translate_pooling(primitive, equation, inputs[0], outvar)

    def translate_pooling(
        self, primitive: str, equation: Any, value: _Value, outvar: Any
    ) -> _Value:
        params = equation.params
        windows = tuple(int(item) for item in params["window_dimensions"])
        strides = tuple(int(item) for item in params["window_strides"])
        padding = tuple(tuple(int(item) for item in pair) for pair in params["padding"])
        base_dilation = tuple(int(item) for item in params["base_dilation"])
        window_dilation = tuple(int(item) for item in params["window_dilation"])
        rank = len(windows)
        if rank not in (3, 4, 5):
            raise self.unsupported(equation, "pooling supports one to three spatial dimensions")
        if any(item != 1 for item in base_dilation + window_dilation):
            raise self.unsupported(equation, "dilated pooling is unsupported")
        if any(low < 0 or high < 0 for low, high in padding):
            raise self.unsupported(equation, "negative pooling padding is unsupported")

        def passthrough(axis: int) -> bool:
            return windows[axis] == 1 and strides[axis] == 1 and padding[axis] == (0, 0)

        if not passthrough(0):
            raise self.unsupported(equation, "pooling axis zero must be the batch axis")
        candidates = [axis for axis in (1, rank - 1) if passthrough(axis)]
        candidates = list(dict.fromkeys(candidates))
        if len(candidates) != 1:
            raise self.unsupported(
                equation, "pooling layout must be unambiguously NCHW or NHWC"
            )
        channel_axis = candidates[0]
        spatial_axes = tuple(axis for axis in range(rank) if axis not in (0, channel_axis))
        permutation = (0, channel_axis, *spatial_axes)
        current = value.name
        if permutation != tuple(range(rank)):
            transposed = self.fresh("pool_input_nchw")
            self.add_node("Transpose", [current], [transposed], perm=permutation)
            current = transposed
        pool_dtype = _dtype(value.aval)
        max_pool_dtypes = {
            np.dtype(np.float16),
            np.dtype(np.float32),
            np.dtype(np.float64),
            np.dtype(np.int8),
            np.dtype(np.uint8),
        }
        average_pool_dtypes = {
            np.dtype(np.float16),
            np.dtype(np.float32),
            np.dtype(np.float64),
        }
        if primitive == "reduce_window_max" and pool_dtype not in max_pool_dtypes:
            raise self.unsupported(equation, f"ONNX MaxPool does not support {pool_dtype}")
        if primitive == "reduce_window_sum" and pool_dtype not in average_pool_dtypes:
            raise self.unsupported(
                equation, f"sum pooling does not support {pool_dtype}"
            )
        pool_output = self.fresh("pool_canonical")
        attributes = {
            "kernel_shape": tuple(windows[axis] for axis in spatial_axes),
            "strides": tuple(strides[axis] for axis in spatial_axes),
            "pads": tuple(padding[axis][0] for axis in spatial_axes)
            + tuple(padding[axis][1] for axis in spatial_axes),
            "ceil_mode": 0,
        }
        if primitive == "reduce_window_max":
            self.add_node("MaxPool", [current], [pool_output], **attributes)
        else:
            average = self.fresh("pool_average")
            self.add_node(
                "AveragePool", [current], [average], count_include_pad=1, **attributes
            )
            scale = self.add_initializer(
                np.asarray(np.prod(attributes["kernel_shape"]), dtype=_dtype(value.aval)),
                "pool_sum_scale",
            )
            self.add_node("Mul", [average, scale], [pool_output])
        current = pool_output
        inverse = tuple(permutation.index(axis) for axis in range(rank))
        output_dynamic = {
            axis: symbol
            for axis, symbol in value.dynamic_axes.items()
            if passthrough(axis)
        }
        output = self.make_value(outvar, primitive, output_dynamic)
        if inverse == tuple(range(rank)):
            self.add_node("Identity", [current], [output.name])
        else:
            self.add_node("Transpose", [current], [output.name], perm=inverse)
        return output

    def translate_cumsum(self, equation: Any, value: _Value, outvar: Any) -> _Value:
        axis = int(equation.params["axis"])
        axis_name = self.add_initializer(np.asarray(axis, dtype=np.int64), "cumsum_axis")
        output = self.make_value(outvar, "cumsum", value.dynamic_axes)
        self.add_node(
            "CumSum",
            [value.name, axis_name],
            [output.name],
            exclusive=0,
            reverse=1 if bool(equation.params.get("reverse", False)) else 0,
        )
        return output

    def translate_dynamic_slice(
        self, equation: Any, inputs: Sequence[_Value], outvar: Any
    ) -> _Value:
        operand = inputs[0]
        rank = len(_shape(operand.aval))
        slice_sizes = tuple(int(item) for item in equation.params["slice_sizes"])
        if len(inputs) != rank + 1 or len(slice_sizes) != rank:
            raise self.unsupported(equation, "dynamic_slice rank mismatch")
        if operand.dynamic_axes:
            raise self.unsupported(
                equation, "dynamic_slice currently requires fixed operand dimensions"
            )
        vector_axis = self.add_initializer(np.asarray([0], dtype=np.int64), "slice_vector_axis")
        starts: list[str] = []
        ends: list[str] = []
        for axis, (start, dimension, size) in enumerate(
            zip(inputs[1:], _shape(operand.aval), slice_sizes)
        ):
            start_i64 = self.fresh(f"dynamic_slice_start_{axis}_i64")
            self.add_node(
                "Cast", [start.name], [start_i64], to=_onnx_dtype(np.dtype(np.int64))
            )
            zero = self.add_initializer(np.asarray(0, dtype=np.int64), "dynamic_slice_zero")
            maximum = self.add_initializer(
                np.asarray(max(dimension - size, 0), dtype=np.int64),
                "dynamic_slice_maximum",
            )
            nonnegative = self.fresh(f"dynamic_slice_start_{axis}_nonnegative")
            clamped = self.fresh(f"dynamic_slice_start_{axis}_clamped")
            self.add_node("Max", [start_i64, zero], [nonnegative])
            self.add_node("Min", [nonnegative, maximum], [clamped])
            size_name = self.add_initializer(
                np.asarray(size, dtype=np.int64), "dynamic_slice_size"
            )
            end = self.fresh(f"dynamic_slice_end_{axis}")
            self.add_node("Add", [clamped, size_name], [end])
            start_vector = self.fresh(f"dynamic_slice_start_{axis}_vector")
            end_vector = self.fresh(f"dynamic_slice_end_{axis}_vector")
            self.add_node("Unsqueeze", [clamped, vector_axis], [start_vector])
            self.add_node("Unsqueeze", [end, vector_axis], [end_vector])
            starts.append(start_vector)
            ends.append(end_vector)
        starts_name = self.fresh("dynamic_slice_starts")
        ends_name = self.fresh("dynamic_slice_ends")
        self.add_node("Concat", starts, [starts_name], axis=0)
        self.add_node("Concat", ends, [ends_name], axis=0)
        axes_name = self.add_initializer(np.arange(rank, dtype=np.int64), "dynamic_slice_axes")
        steps_name = self.add_initializer(np.ones(rank, dtype=np.int64), "dynamic_slice_steps")
        output = self.make_value(outvar, "dynamic_slice")
        self.add_node(
            "Slice",
            [operand.name, starts_name, ends_name, axes_name, steps_name],
            [output.name],
        )
        return output

    def translate_dynamic_update_slice(
        self, equation: Any, inputs: Sequence[_Value], outvar: Any
    ) -> _Value:
        if len(inputs) < 3:
            raise self.unsupported(
                equation, "dynamic_update_slice expects operand, update, and starts"
            )
        operand, update = inputs[:2]
        operand_shape = _shape(operand.aval)
        update_shape = _shape(update.aval)
        rank = len(operand_shape)
        if len(update_shape) != rank or len(inputs) != rank + 2:
            raise self.unsupported(equation, "dynamic_update_slice rank mismatch")
        if operand.dynamic_axes or update.dynamic_axes:
            raise self.unsupported(
                equation, "dynamic_update_slice currently requires fixed tensor shapes"
            )
        if _dtype(operand.aval) != _dtype(update.aval):
            raise self.unsupported(equation, "dynamic_update_slice dtype mismatch")
        if any(update_size > operand_size for operand_size, update_size in zip(
            operand_shape, update_shape
        )):
            raise self.unsupported(
                equation, "dynamic_update_slice update exceeds operand shape"
            )

        starts: list[str] = []
        for axis, (start, operand_size, update_size) in enumerate(
            zip(inputs[2:], operand_shape, update_shape)
        ):
            if _shape(start.aval):
                raise self.unsupported(
                    equation, "dynamic_update_slice starts must be scalar"
                )
            start_i64 = self.cast_indices_to_i64(
                start, f"dynamic_update_start_{axis}"
            )
            zero = self.add_initializer(
                np.asarray(0, dtype=np.int64), "dynamic_update_zero"
            )
            maximum = self.add_initializer(
                np.asarray(operand_size - update_size, dtype=np.int64),
                "dynamic_update_maximum",
            )
            nonnegative = self.fresh(f"dynamic_update_start_{axis}_nonnegative")
            clamped = self.fresh(f"dynamic_update_start_{axis}_clamped")
            self.add_node("Max", [start_i64, zero], [nonnegative])
            self.add_node("Min", [nonnegative, maximum], [clamped])
            starts.append(clamped)

        update_shape_name = self.add_initializer(
            np.asarray(update_shape, dtype=np.int64), "dynamic_update_shape"
        )
        singleton_axis = self.add_initializer(
            np.asarray([-1], dtype=np.int64), "dynamic_update_coordinate_axis"
        )
        coordinate_components: list[str] = []
        for axis, (start, size) in enumerate(zip(starts, update_shape)):
            offsets = self.add_initializer(
                np.arange(size, dtype=np.int64),
                f"dynamic_update_offsets_{axis}",
            )
            positions = self.fresh(f"dynamic_update_positions_{axis}")
            self.add_node("Add", [offsets, start], [positions])
            reshape_dimensions = [1] * rank
            reshape_dimensions[axis] = size
            reshape_name = self.add_initializer(
                np.asarray(reshape_dimensions, dtype=np.int64),
                f"dynamic_update_reshape_{axis}",
            )
            reshaped = self.fresh(f"dynamic_update_grid_axis_{axis}")
            expanded = self.fresh(f"dynamic_update_grid_{axis}")
            component = self.fresh(f"dynamic_update_component_{axis}")
            self.add_node("Reshape", [positions, reshape_name], [reshaped])
            self.add_node("Expand", [reshaped, update_shape_name], [expanded])
            self.add_node("Unsqueeze", [expanded, singleton_axis], [component])
            coordinate_components.append(component)
        coordinates = self.fresh("dynamic_update_coordinates")
        self.add_node("Concat", coordinate_components, [coordinates], axis=rank)
        output = self.make_value(outvar, "dynamic_update_slice")
        self.add_node(
            "ScatterND",
            [operand.name, coordinates, update.name],
            [output.name],
            reduction="none",
        )
        return output

    def translate_composite_unary(self, primitive: str, value: _Value, outvar: Any) -> _Value:
        output = self.make_value(outvar, primitive, value.dynamic_axes)
        if primitive == "rsqrt":
            square_root = self.fresh("sqrt")
            self.add_node("Sqrt", [value.name], [square_root])
            self.add_node("Reciprocal", [square_root], [output.name])
            return output
        # Rounding-accurate forms (Goldberg / Kahan) with ONNX primitives only:
        #   log1p(x) = x                         if fl(1+x) == 1
        #            = log(u) * x / (u - 1)       with u = fl(1+x)   otherwise
        #   expm1(x) = x                         if fl(exp(x)) == 1
        #            = (u - 1) * x / log(u)       with u = fl(exp(x)) otherwise
        # A plain log(1+x) / exp(x)-1 loses all digits for |x| below the dtype epsilon.
        dtype = _dtype(value.aval)
        one = self.add_initializer(np.asarray(1, dtype=dtype), f"{primitive}_one")
        u, d, logu, num, den, safe, ratio, is_one = (self.fresh(f"{primitive}_{tag}") for tag in
                                                      ("u", "d", "log", "num", "den", "safe", "ratio", "is_one"))
        if primitive == "log1p":
            self.add_node("Add", [value.name, one], [u])
            self.add_node("Sub", [u, one], [d])
            self.add_node("Log", [u], [logu])
            self.add_node("Mul", [logu, value.name], [num])
            den_source = d
        else:
            self.add_node("Exp", [value.name], [u])
            self.add_node("Sub", [u, one], [d])
            self.add_node("Log", [u], [logu])
            self.add_node("Mul", [d, value.name], [num])
            den_source = logu
        self.add_node("Equal", [u, one], [is_one])
        self.add_node("Where", [is_one, one, den_source], [safe])
        self.add_node("Div", [num, safe], [ratio])
        self.add_node("Where", [is_one, value.name, ratio], [output.name])
        return output

    def translate_call(self, equation: Any, inputs: Sequence[_Value]) -> list[_Value]:
        closed = None
        for key in ("jaxpr", "call_jaxpr", "fun_jaxpr", "closed_jaxpr"):
            candidate = equation.params.get(key)
            if isinstance(candidate, (jax_core.ClosedJaxpr, jax_core.Jaxpr)):
                closed = candidate
                break
        if closed is None:
            raise self.unsupported(equation, "no nested JAXPR was found")
        if isinstance(closed, jax_core.ClosedJaxpr):
            const_values = [
                _Value(self.add_initializer(value, "nested_constant"), variable.aval, {})
                for variable, value in zip(closed.jaxpr.constvars, closed.consts)
            ]
            inner_inputs = list(inputs)
            inner_jaxpr = closed.jaxpr
        else:
            inner_jaxpr = closed
            number_of_constants = int(equation.params.get("num_consts", len(inner_jaxpr.constvars)))
            const_values = list(inputs[:number_of_constants])
            inner_inputs = list(inputs[number_of_constants:])
        inner_outputs = self.translate_jaxpr(
            inner_jaxpr, input_values=inner_inputs, const_values=const_values
        )
        if len(inner_outputs) != len(equation.outvars):
            raise self.unsupported(equation, "nested JAXPR output count mismatch")
        return [
            _Value(value.name, outvar.aval, value.dynamic_axes)
            for outvar, value in zip(equation.outvars, inner_outputs)
        ]

    def reduced_dynamic_axes(self, value: _Value, axes: Sequence[int]) -> dict[int, str]:
        rank = len(_shape(value.aval))
        normalized = {axis % rank for axis in axes}
        result: dict[int, str] = {}
        for input_axis, symbol in value.dynamic_axes.items():
            if input_axis in normalized:
                continue
            output_axis = input_axis - sum(axis < input_axis for axis in normalized)
            result[output_axis] = symbol
        return result

    def translate_reduction(
        self, primitive: str, equation: Any, value: _Value, outvar: Any
    ) -> _Value:
        axes = tuple(int(axis) for axis in equation.params["axes"])
        dynamic = self.reduced_dynamic_axes(value, axes)
        output = self.make_value(outvar, primitive, dynamic)
        axes_name = self.add_initializer(np.asarray(axes, dtype=np.int64), f"{primitive}_axes")
        self.add_node(_REDUCTION_OPS[primitive], [value.name, axes_name], [output.name], keepdims=0)
        return output

    def translate_broadcast(self, equation: Any, value: _Value, outvar: Any) -> _Value:
        output_shape = _shape(outvar.aval)
        dimensions = tuple(int(axis) for axis in equation.params.get("broadcast_dimensions", ()))
        if len(dimensions) != len(_shape(value.aval)):
            raise self.unsupported(equation, "invalid broadcast_dimensions")
        dynamic = {dimensions[axis]: symbol for axis, symbol in value.dynamic_axes.items()}
        for axis, dimension in enumerate(output_shape):
            if axis not in dynamic:
                symbol = self.infer_symbol(dimension, axis)
                if symbol is not None and axis == 0:
                    dynamic[axis] = symbol
        current = value.name
        missing_axes = tuple(axis for axis in range(len(output_shape)) if axis not in dimensions)
        if missing_axes:
            axes_name = self.add_initializer(
                np.asarray(missing_axes, dtype=np.int64), "broadcast_unsqueeze_axes"
            )
            unsqueezed = self.fresh("broadcast_unsqueezed")
            self.add_node("Unsqueeze", [current, axes_name], [unsqueezed])
            current = unsqueezed
        output = self.make_value(outvar, "broadcast", dynamic)
        operand_dynamic = {dimensions[axis] for axis in value.dynamic_axes}
        new_dynamic = {axis: symbol for axis, symbol in dynamic.items() if axis not in operand_dynamic}
        zero_source = self.batch_zeros(new_dynamic, len(output_shape), _dtype(outvar.aval)) if new_dynamic else None
        if new_dynamic and zero_source is None:
            shape_name = self.shape_tensor(output_shape, dynamic, "broadcast")
            self.add_node("Expand", [current, shape_name], [output.name])
            return output
        # Static Expand shape: a dynamic axis is 1 (kept from the operand by multidirectional broadcasting, or
        # supplied by an exact-zero batch column), so no Shape/Gather/Concat subgraph is needed.
        static_shape = [1 if axis in dynamic else int(size) for axis, size in enumerate(output_shape)]
        shape_name = self.add_initializer(np.asarray(static_shape, dtype=np.int64), "broadcast_shape")
        if zero_source is None:
            self.add_node("Expand", [current, shape_name], [output.name])
        else:
            expanded = self.fresh("broadcast_static")
            self.add_node("Expand", [current, shape_name], [expanded])
            empty_axes = [axis for axis, size in enumerate(output_shape) if axis not in dynamic and int(size) == 0]
            if empty_axes:   # broadcasting a length-1 axis against length 0 must give length 0
                names = [self.add_initializer(np.asarray(v, dtype=np.int64), f"broadcast_empty_{t}") for t, v in
                         (("starts", [0]*len(empty_axes)), ("ends", [0]*len(empty_axes)), ("axes", empty_axes))]
                emptied = self.fresh("batch_zeros_empty")
                self.add_node("Slice", [zero_source, *names], [emptied])
                zero_source = emptied
            self.add_node("Or" if _dtype(outvar.aval) == np.bool_ else "Add", [expanded, zero_source], [output.name])
        return output

    def batch_zeros(self, new_dynamic: Mapping[int, str], rank: int, dtype: Any) -> str | None:
        """Exact zeros of shape (n, 1, ..., 1) taken from the graph input that defines the batch symbol.

        Greater(x, x) is false for every value including NaN and infinities, so Cast gives exact zeros without
        Shape/ConstantOfShape (which ONNX Runtime keeps on the CPU). Only the leading axis is supported.
        """
        if set(new_dynamic) != {0}:
            return None
        symbol = new_dynamic[0]
        if symbol not in self.symbol_sources:
            return None
        source_name, source_axis, _ = self.symbol_sources[symbol]
        if source_axis != 0:
            return None
        source_rank = len(getattr(self, "input_shapes", {}).get(source_name, ()))
        if source_rank < 1:
            return None
        current = source_name
        if source_rank > 1:
            names = [self.add_initializer(np.asarray(v, dtype=np.int64), f"batch_zeros_{t}") for t, v in
                     (("starts", [0]*(source_rank-1)), ("ends", [1]*(source_rank-1)), ("axes", list(range(1, source_rank))))]
            sliced = self.fresh("batch_zeros_slice")
            self.add_node("Slice", [current, *names], [sliced])
            current = sliced
        shape = self.add_initializer(np.asarray([0] + [1]*(rank-1), dtype=np.int64), "batch_zeros_shape")
        reshaped = self.fresh("batch_zeros_reshape")
        self.add_node("Reshape", [current, shape], [reshaped], allowzero=0)
        flag = self.fresh("batch_zeros_false")
        self.add_node("Greater", [reshaped, reshaped], [flag])
        zeros = self.fresh("batch_zeros")
        self.add_node("Cast", [flag], [zeros], to=_onnx_dtype(dtype))
        return zeros

    def translate_reshape(self, equation: Any, value: _Value, outvar: Any) -> _Value:
        if equation.params.get("dimensions") is not None:
            raise self.unsupported(equation, "reshape with dimensions permutation is unsupported")
        output_shape = _shape(outvar.aval)
        dynamic: dict[int, str] = {}
        input_shape = _shape(value.aval)
        for input_axis, symbol in value.dynamic_axes.items():
            if (input_axis == 0 and output_shape and output_shape[0] == input_shape[0]
                    and int(np.prod(input_shape[1:])) == int(np.prod(output_shape[1:]))):
                # Row-major reshape of the trailing axes only: the leading (batch) axis is carried unchanged,
                # even when a trailing output axis happens to have the same example size.
                dynamic[0] = symbol
                continue
            matches = [axis for axis, size in enumerate(output_shape) if size == input_shape[input_axis]]
            if len(matches) == 1:
                dynamic[matches[0]] = symbol
            else:
                raise self.unsupported(
                    equation, f"cannot preserve dynamic dimension {symbol!r} through reshape"
                )
        output = self.make_value(outvar, "reshape", dynamic)
        if set(dynamic) == {0} and set(value.dynamic_axes) == {0} and all(int(d) > 0 for d in output_shape[1:]):
            # ONNX Reshape copies a 0 entry from the input (allowzero=0): no Shape subgraph for the batch axis.
            shape_name = self.add_initializer(np.asarray([0] + [int(d) for d in output_shape[1:]], dtype=np.int64), "reshape_shape")
            self.add_node("Reshape", [value.name, shape_name], [output.name], allowzero=0)
            return output
        shape_name = self.shape_tensor(output_shape, dynamic, "reshape")
        self.add_node("Reshape", [value.name, shape_name], [output.name], allowzero=0)
        return output

    def translate_slice(self, equation: Any, value: _Value, outvar: Any) -> _Value:
        starts = list(int(index) for index in equation.params["start_indices"])
        ends = list(int(index) for index in equation.params["limit_indices"])
        steps_param = equation.params.get("strides")
        steps = [1] * len(starts) if steps_param is None else [int(step) for step in steps_param]
        axes = list(range(len(starts)))
        input_shape = _shape(value.aval)
        dynamic: dict[int, str] = {}
        for axis, symbol in value.dynamic_axes.items():
            if starts[axis] == 0 and ends[axis] == input_shape[axis] and steps[axis] == 1:
                ends[axis] = np.iinfo(np.int64).max
                dynamic[axis] = symbol
        output = self.make_value(outvar, "slice", dynamic)
        names = [
            self.add_initializer(np.asarray(array, dtype=np.int64), f"slice_{label}")
            for label, array in (("starts", starts), ("ends", ends), ("axes", axes), ("steps", steps))
        ]
        self.add_node("Slice", [value.name, *names], [output.name])
        return output

    def translate_reverse(self, equation: Any, value: _Value, outvar: Any) -> _Value:
        axes = tuple(int(axis) for axis in equation.params["dimensions"])
        starts = [-1] * len(axes)
        ends = [np.iinfo(np.int64).min] * len(axes)
        steps = [-1] * len(axes)
        names = [
            self.add_initializer(np.asarray(array, dtype=np.int64), f"reverse_{label}")
            for label, array in (("starts", starts), ("ends", ends), ("axes", axes), ("steps", steps))
        ]
        output = self.make_value(outvar, "reverse", value.dynamic_axes)
        self.add_node("Slice", [value.name, *names], [output.name])
        return output

    def translate_concatenate(
        self, equation: Any, inputs: Sequence[_Value], outvar: Any
    ) -> _Value:
        axis = int(equation.params["dimension"])
        dynamic: dict[int, str] = {}
        rank = len(_shape(outvar.aval))
        for candidate_axis in range(rank):
            if candidate_axis == axis:
                continue
            symbols = {
                value.dynamic_axes[candidate_axis]
                for value in inputs
                if candidate_axis in value.dynamic_axes
            }
            if len(symbols) == 1:
                dynamic[candidate_axis] = symbols.pop()
            elif len(symbols) > 1:
                raise self.unsupported(equation, "incompatible dynamic concatenate inputs")
        output = self.make_value(outvar, "concatenate", dynamic)
        self.add_node("Concat", [value.name for value in inputs], [output.name], axis=axis)
        return output

    def translate_stack(self, equation: Any, inputs: Sequence[_Value], outvar: Any) -> _Value:
        """jnp.stack: Unsqueeze every operand at the new axis, then Concat along it."""
        axis = int(equation.params["axis"])
        rank = len(_shape(outvar.aval))
        if axis < 0:
            axis += rank
        dynamic: dict[int, str] = {}
        for value in inputs:
            for input_axis, symbol in value.dynamic_axes.items():
                output_axis = input_axis + (1 if input_axis >= axis else 0)
                if dynamic.setdefault(output_axis, symbol) != symbol:
                    raise self.unsupported(equation, "incompatible dynamic stack inputs")
        axes_name = self.add_initializer(np.asarray([axis], dtype=np.int64), "stack_axis")
        pieces = []
        for value in inputs:
            unsqueezed = self.fresh("stack_operand")
            self.add_node("Unsqueeze", [value.name, axes_name], [unsqueezed])
            pieces.append(unsqueezed)
        output = self.make_value(outvar, "stack", dynamic)
        self.add_node("Concat", pieces, [output.name], axis=axis)
        return output

    def translate_ffi_call(self, equation: Any, inputs: Sequence[_Value]) -> list[_Value]:
        """An XLA FFI call becomes a custom-domain ONNX node, only when the caller maps its target explicitly.

        The runtime must provide the operator (e.g. an ONNX Runtime custom-op library for the selected
        execution provider). Scalar integer/float FFI attributes become ONNX attributes of the same name.
        A dynamic leading axis of the first operand is carried to every output with the same leading size.
        """
        target = str(equation.params.get("target_name"))
        if target not in self.custom_calls:
            raise self.unsupported(equation, f"FFI target {target!r} has no custom_calls mapping")
        if equation.params.get("has_side_effect") or equation.params.get("input_output_aliases"):
            raise self.unsupported(equation, "side-effecting or aliasing FFI calls are not exportable")
        domain, op_type = self.custom_calls[target]
        attributes: dict[str, Any] = {}
        for name, value in equation.params.get("attributes", ()):
            array = np.asarray(value)
            if array.shape != ():
                raise self.unsupported(equation, f"non-scalar FFI attribute {name!r}")
            if np.issubdtype(array.dtype, np.integer) or array.dtype == np.bool_:
                attributes[str(name)] = int(array)
            elif np.issubdtype(array.dtype, np.floating):
                attributes[str(name)] = float(array)
            else:
                raise self.unsupported(equation, f"unsupported FFI attribute type for {name!r}")
        lead = inputs[0].dynamic_axes.get(0) if inputs else None
        lead_size = _shape(inputs[0].aval)[0] if inputs and _shape(inputs[0].aval) else None
        outputs = []
        for outvar in equation.outvars:
            shape = _shape(outvar.aval)
            dynamic = {0: lead} if lead is not None and shape and shape[0] == lead_size else {}
            outputs.append(self.make_value(outvar, f"{op_type}", dynamic))
        self.add_node(op_type, [value.name for value in inputs], [value.name for value in outputs], domain=domain, **attributes)
        self.custom_domains.add(domain)
        return outputs

    def translate_pad(self, equation: Any, inputs: Sequence[_Value], outvar: Any) -> _Value:
        if len(inputs) != 2:
            raise self.unsupported(equation, "pad expects an operand and a scalar fill value")
        config = tuple(tuple(int(item) for item in entry) for entry in equation.params["padding_config"])
        if any(interior != 0 for _, _, interior in config):
            raise self.unsupported(equation, "interior padding is not representable by ONNX Pad")
        pads = [low for low, _, _ in config] + [high for _, high, _ in config]
        pads_name = self.add_initializer(np.asarray(pads, dtype=np.int64), "pads")
        output = self.make_value(outvar, "pad", inputs[0].dynamic_axes)
        self.add_node("Pad", [inputs[0].name, pads_name, inputs[1].name], [output.name], mode="constant")
        return output

    def translate_gather(
        self, equation: Any, inputs: Sequence[_Value], outvar: Any
    ) -> _Value:
        if len(inputs) != 2:
            raise self.unsupported(equation, "gather expects operand and start_indices")
        dimensions = equation.params["dimension_numbers"]
        operand_batching = tuple(int(axis) for axis in dimensions.operand_batching_dims)
        index_batching = tuple(
            int(axis) for axis in dimensions.start_indices_batching_dims
        )
        if operand_batching or index_batching:
            return self.translate_gather_elements(equation, inputs, outvar)
        return self.translate_gather_nd(equation, inputs, outvar)

    def translate_gather_elements(
        self, equation: Any, inputs: Sequence[_Value], outvar: Any
    ) -> _Value:
        operand, start_indices = inputs
        dimensions = equation.params["dimension_numbers"]
        start_index_map = tuple(int(axis) for axis in dimensions.start_index_map)
        collapsed = tuple(int(axis) for axis in dimensions.collapsed_slice_dims)
        offset_dims = tuple(int(axis) for axis in dimensions.offset_dims)
        operand_batching = tuple(int(axis) for axis in dimensions.operand_batching_dims)
        index_batching = tuple(
            int(axis) for axis in dimensions.start_indices_batching_dims
        )
        operand_shape = _shape(operand.aval)
        indices_shape = _shape(start_indices.aval)
        if len(start_index_map) != 1 or collapsed != start_index_map:
            raise self.unsupported(
                equation, "batched gather requires one collapsed indexed axis"
            )
        axis = start_index_map[0]
        expected_batching = tuple(
            candidate for candidate in range(len(operand_shape)) if candidate != axis
        )
        if (
            operand_batching != expected_batching
            or index_batching != expected_batching
            or offset_dims
        ):
            raise self.unsupported(
                equation, "batched gather layout differs from take_along_axis"
            )
        if not indices_shape or indices_shape[-1] != 1:
            raise self.unsupported(
                equation, "take_along_axis indices need a trailing coordinate axis"
            )
        output_shape = indices_shape[:-1]
        if len(output_shape) != len(operand_shape) or output_shape != _shape(outvar.aval):
            raise self.unsupported(equation, "take_along_axis output shape mismatch")
        slice_sizes = tuple(int(size) for size in equation.params["slice_sizes"])
        if slice_sizes != (1,) * len(operand_shape):
            raise self.unsupported(
                equation, "take_along_axis gather slices must be scalar"
            )

        indices_i64 = self.cast_indices_to_i64(start_indices, "gather_elements")
        mode = self.index_mode(equation)
        normalized, valid = self.normalize_index_tensor(
            indices_i64,
            [operand_shape[axis] - 1],
            mode,
            "gather_elements",
        )
        squeeze_axis = self.add_initializer(
            np.asarray([len(indices_shape) - 1], dtype=np.int64),
            "gather_elements_coordinate_axis",
        )
        squeezed = self.fresh("gather_elements_indices")
        self.add_node("Squeeze", [normalized, squeeze_axis], [squeezed])
        dynamic: dict[int, str] = {
            input_axis: symbol
            for input_axis, symbol in start_indices.dynamic_axes.items()
            if input_axis < len(output_shape)
        }
        for input_axis, symbol in operand.dynamic_axes.items():
            if input_axis == axis:
                continue
            previous = dynamic.get(input_axis)
            if previous is not None and previous != symbol:
                raise self.unsupported(
                    equation, "take_along_axis dynamic dimensions are incompatible"
                )
            dynamic[input_axis] = symbol
        output = self.make_value(outvar, "gather_elements", dynamic)
        gathered = output.name if valid is None else self.fresh("gather_elements_values")
        self.add_node(
            "GatherElements", [operand.name, squeezed], [gathered], axis=axis
        )
        if valid is not None:
            fill_value = equation.params.get("fill_value")
            if fill_value is None:
                raise self.unsupported(
                    equation, "fill/drop gather requires an explicit fill value"
                )
            fill = self.add_initializer(
                np.asarray(fill_value, dtype=_dtype(operand.aval)),
                "gather_elements_fill",
            )
            self.add_node("Where", [valid, gathered, fill], [output.name])
        return output

    def translate_gather_nd(
        self, equation: Any, inputs: Sequence[_Value], outvar: Any
    ) -> _Value:
        operand, start_indices = inputs
        dimensions = equation.params["dimension_numbers"]
        start_index_map = tuple(int(axis) for axis in dimensions.start_index_map)
        collapsed = tuple(int(axis) for axis in dimensions.collapsed_slice_dims)
        offset_dims = tuple(int(axis) for axis in dimensions.offset_dims)
        operand_shape = _shape(operand.aval)
        indices_shape = _shape(start_indices.aval)
        index_depth = len(start_index_map)
        if index_depth < 1 or set(collapsed) != set(start_index_map):
            raise self.unsupported(
                equation, "GatherND requires every indexed axis to be collapsed"
            )
        if len(set(start_index_map)) != index_depth:
            raise self.unsupported(equation, "gather index axes must be unique")
        if not indices_shape or indices_shape[-1] != index_depth:
            raise self.unsupported(
                equation, "gather coordinates must occupy the trailing axis"
            )
        remaining_axes = tuple(
            axis for axis in range(len(operand_shape)) if axis not in start_index_map
        )
        slice_sizes = tuple(int(size) for size in equation.params["slice_sizes"])
        if len(slice_sizes) != len(operand_shape) or any(
            size != (1 if axis in start_index_map else operand_shape[axis])
            for axis, size in enumerate(slice_sizes)
        ):
            raise self.unsupported(
                equation, "GatherND slices must span all non-indexed axes"
            )
        index_batch_shape = indices_shape[:-1]
        output_rank = len(index_batch_shape) + len(remaining_axes)
        if (
            len(offset_dims) != len(remaining_axes)
            or len(set(offset_dims)) != len(offset_dims)
            or any(axis < 0 or axis >= output_rank for axis in offset_dims)
        ):
            raise self.unsupported(equation, "gather offset dimensions are invalid")
        index_output_dims = tuple(
            axis for axis in range(output_rank) if axis not in offset_dims
        )
        if len(index_output_dims) != len(index_batch_shape):
            raise self.unsupported(equation, "gather index output dimensions mismatch")
        expected_shape = [0] * output_rank
        for axis, size in zip(index_output_dims, index_batch_shape):
            expected_shape[axis] = size
        for axis, operand_axis in zip(offset_dims, remaining_axes):
            expected_shape[axis] = operand_shape[operand_axis]
        if tuple(expected_shape) != _shape(outvar.aval):
            raise self.unsupported(equation, "gather output layout mismatch")

        indices_i64 = self.cast_indices_to_i64(start_indices, "gather_nd")
        mode = self.index_mode(equation)
        normalized, valid = self.normalize_index_tensor(
            indices_i64,
            [operand_shape[axis] - 1 for axis in start_index_map],
            mode,
            "gather_nd",
        )
        operand_permutation = (*start_index_map, *remaining_axes)
        current_operand = operand.name
        if operand_permutation != tuple(range(len(operand_shape))):
            current_operand = self.fresh("gather_nd_operand")
            self.add_node(
                "Transpose",
                [operand.name],
                [current_operand],
                perm=operand_permutation,
            )
        gathered = self.fresh("gather_nd_values")
        self.add_node("GatherND", [current_operand, normalized], [gathered])
        current = gathered
        if valid is not None:
            fill_value = equation.params.get("fill_value")
            if fill_value is None:
                raise self.unsupported(
                    equation, "fill/drop gather requires an explicit fill value"
                )
            condition = self.broadcast_trailing_mask(
                valid, len(remaining_axes), "gather_nd"
            )
            fill = self.add_initializer(
                np.asarray(fill_value, dtype=_dtype(operand.aval)),
                "gather_nd_fill",
            )
            filled = self.fresh("gather_nd_filled")
            self.add_node("Where", [condition, current, fill], [filled])
            current = filled

        canonical_to_output = [0] * output_rank
        for canonical_axis, output_axis in enumerate(index_output_dims):
            canonical_to_output[output_axis] = canonical_axis
        for remaining_index, output_axis in enumerate(offset_dims):
            canonical_to_output[output_axis] = len(index_batch_shape) + remaining_index
        dynamic: dict[int, str] = {}
        for input_axis, symbol in start_indices.dynamic_axes.items():
            if input_axis < len(index_batch_shape):
                dynamic[index_output_dims[input_axis]] = symbol
        for operand_axis, output_axis in zip(remaining_axes, offset_dims):
            if operand_axis in operand.dynamic_axes:
                dynamic[output_axis] = operand.dynamic_axes[operand_axis]
        output = self.make_value(outvar, "gather_nd", dynamic)
        permutation = tuple(canonical_to_output)
        if permutation == tuple(range(output_rank)):
            self.add_node("Identity", [current], [output.name])
        else:
            self.add_node("Transpose", [current], [output.name], perm=permutation)
        return output

    def translate_scatter(
        self,
        primitive: str,
        equation: Any,
        inputs: Sequence[_Value],
        outvar: Any,
    ) -> _Value:
        if len(inputs) != 3:
            raise self.unsupported(
                equation, "scatter expects operand, indices, and updates"
            )
        operand, start_indices, updates = inputs
        if operand.dynamic_axes:
            raise self.unsupported(
                equation, "scatter currently requires a fixed operand shape"
            )
        if _dtype(operand.aval) != _dtype(updates.aval):
            raise self.unsupported(equation, "scatter operand/update dtype mismatch")
        dimensions = equation.params["dimension_numbers"]
        update_window_dims = tuple(int(axis) for axis in dimensions.update_window_dims)
        inserted_window_dims = tuple(
            int(axis) for axis in dimensions.inserted_window_dims
        )
        scatter_to_operand = tuple(
            int(axis) for axis in dimensions.scatter_dims_to_operand_dims
        )
        operand_batching = tuple(
            int(axis) for axis in dimensions.operand_batching_dims
        )
        index_batching = tuple(
            int(axis) for axis in dimensions.scatter_indices_batching_dims
        )
        if operand_batching or index_batching:
            raise self.unsupported(equation, "batched scatter dimensions are unsupported")
        index_depth = len(scatter_to_operand)
        indices_shape = _shape(start_indices.aval)
        operand_shape = _shape(operand.aval)
        updates_shape = _shape(updates.aval)
        if (
            index_depth < 1
            or len(set(scatter_to_operand)) != index_depth
            or not indices_shape
            or indices_shape[-1] != index_depth
        ):
            raise self.unsupported(equation, "scatter coordinate layout is invalid")
        if set(inserted_window_dims) != set(scatter_to_operand):
            raise self.unsupported(
                equation, "scatter indexed axes must be inserted window dimensions"
            )
        remaining_axes = tuple(
            axis
            for axis in range(len(operand_shape))
            if axis not in inserted_window_dims
        )
        if len(update_window_dims) != len(remaining_axes):
            raise self.unsupported(equation, "scatter update window rank mismatch")
        scatter_batch_axes = tuple(
            axis for axis in range(len(updates_shape)) if axis not in update_window_dims
        )
        if tuple(updates_shape[axis] for axis in scatter_batch_axes) != indices_shape[:-1]:
            raise self.unsupported(equation, "scatter index/update batch shape mismatch")
        if any(
            updates_shape[update_axis] != operand_shape[operand_axis]
            for update_axis, operand_axis in zip(update_window_dims, remaining_axes)
        ):
            raise self.unsupported(
                equation, "scatter updates must span every non-indexed operand axis"
            )

        if primitive == "scatter":
            if equation.params.get("update_jaxpr") is not None:
                raise self.unsupported(equation, "custom scatter update is unsupported")
            if not bool(equation.params.get("unique_indices", False)):
                raise self.unsupported(
                    equation, "scatter set requires unique_indices=True"
                )
            reduction = "none"
        else:
            update_jaxpr, update_constants = self.nested_jaxpr_parts(
                equation.params.get("update_jaxpr"), equation
            )
            if (
                update_constants
                or len(update_jaxpr.eqns) != 1
                or update_jaxpr.eqns[0].primitive.name != "add"
            ):
                raise self.unsupported(
                    equation, "scatter-add requires the standard addition reducer"
                )
            reduction = "add"

        indices_i64 = self.cast_indices_to_i64(start_indices, "scatter")
        mode = self.index_mode(equation)
        normalized, valid = self.normalize_index_tensor(
            indices_i64,
            [operand_shape[axis] - 1 for axis in scatter_to_operand],
            mode,
            "scatter",
        )
        update_permutation = (*scatter_batch_axes, *update_window_dims)
        current_updates = updates.name
        if update_permutation != tuple(range(len(updates_shape))):
            current_updates = self.fresh("scatter_updates")
            self.add_node(
                "Transpose",
                [updates.name],
                [current_updates],
                perm=update_permutation,
            )
        current_indices = normalized
        if valid is not None and reduction == "add":
            condition = self.broadcast_trailing_mask(
                valid, len(remaining_axes), "scatter"
            )
            zero = self.add_initializer(
                np.asarray(0, dtype=_dtype(updates.aval)), "scatter_zero"
            )
            masked_updates = self.fresh("scatter_masked_updates")
            self.add_node(
                "Where",
                [condition, current_updates, zero],
                [masked_updates],
            )
            current_updates = masked_updates

        operand_permutation = (*scatter_to_operand, *remaining_axes)
        current_operand = operand.name
        if operand_permutation != tuple(range(len(operand_shape))):
            current_operand = self.fresh("scatter_operand")
            self.add_node(
                "Transpose",
                [operand.name],
                [current_operand],
                perm=operand_permutation,
            )

        if valid is not None and reduction == "none":
            valid_shape = self.fresh("scatter_set_valid_shape")
            invalid_count = self.fresh("scatter_set_invalid_slots")
            self.add_node("Shape", [valid], [valid_shape])
            self.add_node(
                "ReduceProd", [valid_shape], [invalid_count], keepdims=0
            )
            range_start = self.add_initializer(
                np.asarray(0, dtype=np.int64), "scatter_set_range_start"
            )
            range_step = self.add_initializer(
                np.asarray(1, dtype=np.int64), "scatter_set_range_step"
            )
            flat_slots = self.fresh("scatter_set_flat_slots")
            slots = self.fresh("scatter_set_slots")
            self.add_node(
                "Range",
                [range_start, invalid_count, range_step],
                [flat_slots],
            )
            self.add_node("Reshape", [flat_slots, valid_shape], [slots])
            first_axis_size = self.add_initializer(
                np.asarray(operand_shape[operand_permutation[0]], dtype=np.int64),
                "scatter_set_first_axis_size",
            )
            first_sentinel = self.fresh("scatter_set_first_sentinel")
            self.add_node("Add", [slots, first_axis_size], [first_sentinel])
            coordinate_axis = self.add_initializer(
                np.asarray([-1], dtype=np.int64),
                "scatter_set_coordinate_axis",
            )
            sentinel_first_column = self.fresh(
                "scatter_set_sentinel_first_column"
            )
            self.add_node(
                "Unsqueeze",
                [first_sentinel, coordinate_axis],
                [sentinel_first_column],
            )
            sentinel_indices = sentinel_first_column
            if index_depth > 1:
                coordinate_tail = self.add_initializer(
                    np.asarray([index_depth - 1], dtype=np.int64),
                    "scatter_set_coordinate_tail",
                )
                sentinel_tail_shape = self.fresh(
                    "scatter_set_sentinel_tail_shape"
                )
                sentinel_tail = self.fresh("scatter_set_sentinel_tail")
                self.add_node(
                    "Concat",
                    [valid_shape, coordinate_tail],
                    [sentinel_tail_shape],
                    axis=0,
                )
                self.add_node(
                    "ConstantOfShape",
                    [sentinel_tail_shape],
                    [sentinel_tail],
                    value=numpy_helper.from_array(
                        np.asarray([0], dtype=np.int64)
                    ),
                )
                sentinel_indices = self.fresh("scatter_set_sentinel_indices")
                self.add_node(
                    "Concat",
                    [sentinel_first_column, sentinel_tail],
                    [sentinel_indices],
                    axis=-1,
                )
            valid_coordinates = self.fresh("scatter_set_valid_coordinates")
            redirected_indices = self.fresh("scatter_set_redirected_indices")
            self.add_node(
                "Unsqueeze",
                [valid, coordinate_axis],
                [valid_coordinates],
            )
            self.add_node(
                "Where",
                [valid_coordinates, current_indices, sentinel_indices],
                [redirected_indices],
            )

            slot_axis = self.add_initializer(
                np.asarray([0], dtype=np.int64), "scatter_set_slot_axis"
            )
            slot_count = self.fresh("scatter_set_slot_count")
            self.add_node(
                "Unsqueeze", [invalid_count, slot_axis], [slot_count]
            )
            canonical_shape = tuple(
                operand_shape[axis] for axis in operand_permutation
            )
            if len(canonical_shape) == 1:
                filler_shape = slot_count
            else:
                filler_tail = self.add_initializer(
                    np.asarray(canonical_shape[1:], dtype=np.int64),
                    "scatter_set_filler_tail",
                )
                filler_shape = self.fresh("scatter_set_filler_shape")
                self.add_node(
                    "Concat",
                    [slot_count, filler_tail],
                    [filler_shape],
                    axis=0,
                )
            filler = self.fresh("scatter_set_filler")
            self.add_node(
                "ConstantOfShape",
                [filler_shape],
                [filler],
                value=numpy_helper.from_array(
                    np.asarray([0], dtype=_dtype(updates.aval))
                ),
            )
            padded_operand = self.fresh("scatter_set_padded_operand")
            self.add_node(
                "Concat", [current_operand, filler], [padded_operand], axis=0
            )
            padded_result = self.fresh("scatter_set_padded_result")
            self.add_node(
                "ScatterND",
                [padded_operand, redirected_indices, current_updates],
                [padded_result],
                reduction="none",
            )
            slice_starts = self.add_initializer(
                np.asarray([0], dtype=np.int64), "scatter_set_slice_start"
            )
            slice_ends = self.add_initializer(
                np.asarray([canonical_shape[0]], dtype=np.int64),
                "scatter_set_slice_end",
            )
            slice_axes = self.add_initializer(
                np.asarray([0], dtype=np.int64), "scatter_set_slice_axis"
            )
            slice_steps = self.add_initializer(
                np.asarray([1], dtype=np.int64), "scatter_set_slice_step"
            )
            scattered = self.fresh("scatter_canonical")
            self.add_node(
                "Slice",
                [
                    padded_result,
                    slice_starts,
                    slice_ends,
                    slice_axes,
                    slice_steps,
                ],
                [scattered],
            )
        else:
            scattered = self.fresh("scatter_canonical")
            self.add_node(
                "ScatterND",
                [current_operand, current_indices, current_updates],
                [scattered],
                reduction=reduction,
            )
        inverse = tuple(
            operand_permutation.index(axis) for axis in range(len(operand_shape))
        )
        output = self.make_value(outvar, primitive)
        if inverse == tuple(range(len(operand_shape))):
            self.add_node("Identity", [scattered], [output.name])
        else:
            self.add_node("Transpose", [scattered], [output.name], perm=inverse)
        return output

    def dot_dynamic_axes(
        self, lhs: _Value, rhs: _Value, dimension_numbers: Any
    ) -> dict[int, str]:
        (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
        result: dict[int, str] = {}
        output_axis = 0
        for left_axis, right_axis in zip(lhs_batch, rhs_batch):
            symbol = lhs.dynamic_axes.get(left_axis) or rhs.dynamic_axes.get(right_axis)
            if symbol is not None:
                result[output_axis] = symbol
            output_axis += 1
        for axis in range(len(_shape(lhs.aval))):
            if axis not in lhs_contract and axis not in lhs_batch:
                if axis in lhs.dynamic_axes:
                    result[output_axis] = lhs.dynamic_axes[axis]
                output_axis += 1
        for axis in range(len(_shape(rhs.aval))):
            if axis not in rhs_contract and axis not in rhs_batch:
                if axis in rhs.dynamic_axes:
                    result[output_axis] = rhs.dynamic_axes[axis]
                output_axis += 1
        return result

    def translate_dot_general(
        self, equation: Any, inputs: Sequence[_Value], outvar: Any
    ) -> _Value:
        if len(inputs) != 2:
            raise self.unsupported(equation, "dot_general expects two operands")
        lhs, rhs = inputs
        dimension_numbers = equation.params["dimension_numbers"]
        (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
        preferred = equation.params.get("preferred_element_type")
        if preferred is not None and np.dtype(preferred) != _dtype(outvar.aval):
            raise self.unsupported(equation, "preferred accumulation dtype differs from output dtype")
        dynamic = self.dot_dynamic_axes(lhs, rhs, dimension_numbers)
        output = self.make_value(outvar, "dot", dynamic)

        lhs_rank = len(_shape(lhs.aval))
        rhs_rank = len(_shape(rhs.aval))
        standard = (
            tuple(lhs_contract) == (lhs_rank - 1,)
            and tuple(rhs_contract) == ((rhs_rank - 2) if rhs_rank >= 2 else 0,)
            and tuple(lhs_batch) == tuple(range(max(lhs_rank - 2, 0)))
            and tuple(rhs_batch) == tuple(range(max(rhs_rank - 2, 0)))
        )
        if standard:
            self.add_node("MatMul", [lhs.name, rhs.name], [output.name])
            return output

        labels = iter("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        if lhs_rank + rhs_rank > 52:
            raise self.unsupported(equation, "dot_general rank exceeds ONNX Einsum labels")
        lhs_labels: list[str | None] = [None] * lhs_rank
        rhs_labels: list[str | None] = [None] * rhs_rank
        output_labels: list[str] = []
        for left_axis, right_axis in zip(lhs_batch, rhs_batch):
            label = next(labels)
            lhs_labels[left_axis] = label
            rhs_labels[right_axis] = label
            output_labels.append(label)
        for left_axis, right_axis in zip(lhs_contract, rhs_contract):
            label = next(labels)
            lhs_labels[left_axis] = label
            rhs_labels[right_axis] = label
        for axis in range(lhs_rank):
            if lhs_labels[axis] is None:
                lhs_labels[axis] = next(labels)
                output_labels.append(lhs_labels[axis])
        for axis in range(rhs_rank):
            if rhs_labels[axis] is None:
                rhs_labels[axis] = next(labels)
                output_labels.append(rhs_labels[axis])
        einsum_equation = (
            "".join(lhs_labels) + "," + "".join(rhs_labels) + "->" + "".join(output_labels)
        )
        self.add_node("Einsum", [lhs.name, rhs.name], [output.name], equation=einsum_equation)
        return output

    def transpose_value(self, value: _Value, permutation: Sequence[int], prefix: str) -> _Value:
        if tuple(permutation) == tuple(range(len(permutation))):
            return value
        output_name = self.fresh(prefix)
        dynamic = {
            tuple(permutation).index(axis): symbol for axis, symbol in value.dynamic_axes.items()
        }
        self.add_node("Transpose", [value.name], [output_name], perm=tuple(permutation))
        return _Value(output_name, value.aval, dynamic)

    def translate_convolution(
        self, equation: Any, inputs: Sequence[_Value], outvar: Any
    ) -> _Value:
        if len(inputs) != 2:
            raise self.unsupported(equation, "convolution expects input and kernel")
        params = equation.params
        if any(int(value) != 1 for value in params["lhs_dilation"]):
            raise self.unsupported(equation, "transposed/lhs-dilated convolution is unsupported")
        if int(params.get("batch_group_count", 1)) != 1:
            raise self.unsupported(equation, "batch-grouped convolution is unsupported")
        dimensions = params["dimension_numbers"]
        lhs_perm = tuple(int(axis) for axis in dimensions.lhs_spec)
        rhs_perm = tuple(int(axis) for axis in dimensions.rhs_spec)
        out_spec = tuple(int(axis) for axis in dimensions.out_spec)
        lhs = self.transpose_value(inputs[0], lhs_perm, "conv_lhs_nchw")
        rhs = self.transpose_value(inputs[1], rhs_perm, "conv_kernel_oihw")
        canonical_output = self.fresh("conv_canonical")
        padding = tuple(tuple(int(item) for item in pair) for pair in params["padding"])
        pads = [low for low, _ in padding] + [high for _, high in padding]
        self.add_node(
            "Conv",
            [lhs.name, rhs.name],
            [canonical_output],
            strides=tuple(int(item) for item in params["window_strides"]),
            dilations=tuple(int(item) for item in params["rhs_dilation"]),
            pads=pads,
            group=int(params["feature_group_count"]),
        )
        inverse_output = tuple(out_spec.index(axis) for axis in range(len(out_spec)))
        dynamic_canonical = {
            lhs_perm.index(axis): symbol for axis, symbol in inputs[0].dynamic_axes.items()
        }
        if inverse_output == tuple(range(len(inverse_output))):
            return _Value(canonical_output, outvar.aval, dynamic_canonical)
        output = self.make_value(
            outvar,
            "conv_output",
            {
                inverse_output.index(axis): symbol
                for axis, symbol in dynamic_canonical.items()
            },
        )
        self.add_node("Transpose", [canonical_output], [output.name], perm=inverse_output)
        return output


def _const_values(translator: _Translator, closed_jaxpr: jax_core.ClosedJaxpr) -> list[_Value]:
    values: list[_Value] = []
    for variable, value in zip(closed_jaxpr.jaxpr.constvars, closed_jaxpr.consts):
        array = np.asarray(value)
        name = translator.add_initializer(array, "constant")
        values.append(_Value(name, variable.aval, {}))
    return values


def export_jax_to_onnx(
    function: Callable[..., Any],
    example_args: Sequence[Any],
    path: Path | str,
    *,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    dynamic_batch: bool = True,
    dynamic_axes: Mapping[str, Mapping[int, str]] | None = None,
    output_dynamic_axes: Mapping[str, Mapping[int, str]] | None = None,
    opset: int = 18,
    producer_name: str = "fortonnx_export.jaxpr",
    metadata: Mapping[str, str] | None = None,
    check_model: bool = True,
    ir_version: int | None = None,
    custom_calls: Mapping[str, tuple[str, str]] | None = None,
) -> ExportResult:
    """Trace a pure JAX function and export its supported inference graph.

    ``example_args`` may contain arbitrary input PyTrees.  The ONNX interface
    is the flattened sequence of their array leaves.  Parameters normally stay
    in the Python closure and become ONNX initializers.  The default dynamic
    batch policy marks axis zero of every compatible input/output leaf.
    """

    arguments = tuple(example_args)
    flat_inputs = jax.tree.leaves(arguments)
    names = _normalise_names(input_names, len(flat_inputs), "input")
    closed = jax.make_jaxpr(function)(*arguments)
    if len(closed.jaxpr.invars) != len(flat_inputs):
        raise JaxOnnxExportError("JAX input flattening did not match traced JAXPR inputs")
    output_name_tuple = _normalise_names(output_names, len(closed.jaxpr.outvars), "output")

    configured_dynamic: dict[str, dict[int, str]] = {
        name: {int(axis): str(symbol) for axis, symbol in axes.items()}
        for name, axes in (dynamic_axes or {}).items()
    }
    batch_size: int | None = None
    if dynamic_batch:
        for leaf in flat_inputs:
            if len(leaf.shape) > 0:
                batch_size = int(leaf.shape[0])
                break
        if batch_size is not None:
            for name, leaf in zip(names, flat_inputs):
                if len(leaf.shape) > 0 and int(leaf.shape[0]) == batch_size:
                    configured_dynamic.setdefault(name, {}).setdefault(0, "batch")

    translator = _Translator(
        opset=opset, dynamic_axes=configured_dynamic, input_names=names, custom_calls=custom_calls
    )
    graph_inputs: list[onnx.ValueInfoProto] = []
    input_values: list[_Value] = []
    for name, leaf, variable in zip(names, flat_inputs, closed.jaxpr.invars):
        axes = configured_dynamic.get(name, {})
        dimensions: list[int | str] = [
            axes.get(axis, int(dimension)) for axis, dimension in enumerate(leaf.shape)
        ]
        graph_inputs.append(
            helper.make_tensor_value_info(name, _onnx_dtype(leaf.dtype), dimensions)
        )
        input_values.append(_Value(name, variable.aval, axes))
    translator._set_input_symbol_examples(input_values, names)
    outputs = translator.translate_jaxpr(
        closed.jaxpr,
        input_values=input_values,
        const_values=_const_values(translator, closed),
    )

    configured_output_dynamic = {
        name: {int(axis): str(symbol) for axis, symbol in axes.items()}
        for name, axes in (output_dynamic_axes or {}).items()
    }
    graph_outputs: list[onnx.ValueInfoProto] = []
    for name, value in zip(output_name_tuple, outputs):
        axes = dict(value.dynamic_axes)
        axes.update(configured_output_dynamic.get(name, {}))
        value_shape = _shape(value.aval)
        if dynamic_batch and batch_size is not None and value_shape:
            if value_shape[0] == batch_size:
                axes.setdefault(0, "batch")
        if value.name != name:
            translator.add_node("Identity", [value.name], [name])
        dimensions = [axes.get(axis, dimension) for axis, dimension in enumerate(value_shape)]
        graph_outputs.append(
            helper.make_tensor_value_info(name, _onnx_dtype(_dtype(value.aval)), dimensions)
        )

    graph = helper.make_graph(
        translator.nodes,
        "jaxpr_exported_graph",
        graph_inputs,
        graph_outputs,
        initializer=translator.initializers,
    )
    opset_imports = [helper.make_opsetid("", opset)] + [
        helper.make_opsetid(domain, 1) for domain in sorted(translator.custom_domains)
    ]
    model = helper.make_model(
        graph,
        producer_name=producer_name,
        opset_imports=opset_imports,
        # Declare the IR version the opset requires, not the newest one the installed onnx package knows,
        # so a newer onnx package does not make the model unloadable by runtimes that support the opset.
        ir_version=(helper.find_min_ir_version_for(opset_imports[:1]) if ir_version is None else int(ir_version)),
    )
    properties = {
        "exporter": "fail_closed_jaxpr_to_onnx",
        "jax_version": jax.__version__,
        "onnx_opset": str(opset),
        "supported_primitive_counts": ",".join(
            f"{name}:{count}" for name, count in sorted(translator.primitive_counts.items())
        ),
    }
    properties.update(metadata or {})
    for key, value in properties.items():
        model.metadata_props.add(key=str(key), value=str(value))
    if check_model:
        onnx.checker.check_model(model, full_check=True)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    return ExportResult(
        path=output_path,
        input_names=names,
        output_names=output_name_tuple,
        primitive_counts=dict(sorted(translator.primitive_counts.items())),
    )


def supported_primitives() -> tuple[str, ...]:
    """Return the explicitly implemented primitive names."""

    explicit = {
        "argmax",
        "argmin",
        "broadcast_in_dim",
        "clamp",
        "concatenate",
        "cond",
        "conv_general_dilated",
        "convert_element_type",
        "copy",
        "cumsum",
        "device_put",
        "dot_general",
        "dynamic_slice",
        "dynamic_update_slice",
        "expm1",
        "gather",
        "integer_pow",
        "is_finite",
        "log1p",
        "ne",
        "not",
        "pad",
        "reshape",
        "reduce_window",
        "reduce_window_max",
        "reduce_window_sum",
        "rev",
        "rsqrt",
        "select_n",
        "stack",
        "ffi_call",
        "scan",
        "scatter",
        "scatter-add",
        "slice",
        "squeeze",
        "stop_gradient",
        "transpose",
        "top_k",
        "while",
    }
    return tuple(
        sorted(
            set(_UNARY_OPS)
            | set(_BINARY_OPS)
            | set(_REDUCTION_OPS)
            | _CALL_PRIMITIVES
            | explicit
        )
    )

