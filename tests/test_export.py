from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from fortonnx_export import (
    UnsupportedJaxPrimitive,
    export_jax_to_onnx,
    supported_primitives,
)


def _session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def test_generic_export_handles_dynamic_mlp_and_nested_softplus(tmp_path: Path) -> None:
    key1, key2 = jax.random.split(jax.random.PRNGKey(4))
    weight1 = jax.random.normal(key1, (3, 7), dtype=jnp.float32) / 3.0
    bias1 = jnp.linspace(-0.2, 0.2, 7, dtype=jnp.float32)
    weight2 = jax.random.normal(key2, (7, 2), dtype=jnp.float32) / 4.0
    bias2 = jnp.array([0.1, -0.15], dtype=jnp.float32)

    def model(inputs: jax.Array) -> jax.Array:
        hidden = jnp.tanh(inputs @ weight1 + bias1)
        return jax.nn.softplus(hidden @ weight2 + bias2)

    path = tmp_path / "mlp.onnx"
    result = export_jax_to_onnx(
        model,
        (jax.ShapeDtypeStruct((3, 3), jnp.float32),),
        path,
        input_names=("features",),
        output_names=("prediction",),
    )
    assert "dot_general" in result.primitive_counts
    assert "custom_jvp_call" in result.primitive_counts
    session = _session(path)
    for batch in (1, 3, 11):
        inputs = np.linspace(-1.0, 1.0, batch * 3, dtype=np.float32).reshape(batch, 3)
        actual = session.run(["prediction"], {"features": inputs})[0]
        expected = np.asarray(model(inputs))
        np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=2.0e-6)


def test_generic_export_handles_pytree_multioutput_and_common_array_ops(
    tmp_path: Path,
) -> None:
    def model(inputs: jax.Array, controls: dict[str, jax.Array]):
        scaled = inputs * controls["scale"]
        probabilities = jax.nn.softmax(scaled, axis=1)
        selected = jnp.where(scaled > 0, jnp.clip(scaled, -0.75, 0.75), 0.0)
        reshaped = jnp.reshape(selected, (selected.shape[0], 2, 3))
        return {
            "mass": jnp.sum(probabilities, axis=1),
            "transposed": jnp.transpose(reshaped, (0, 2, 1)),
        }

    example = (
        jax.ShapeDtypeStruct((4, 6), jnp.float32),
        {"scale": jax.ShapeDtypeStruct((6,), jnp.float32)},
    )
    path = tmp_path / "array_ops.onnx"
    export_jax_to_onnx(
        model,
        example,
        path,
        input_names=("inputs", "scale"),
        output_names=("mass", "transposed"),
    )
    session = _session(path)
    rng = np.random.default_rng(8)
    inputs = rng.normal(size=(9, 6)).astype(np.float32)
    scale = np.linspace(0.5, 1.5, 6, dtype=np.float32)
    actual = session.run(None, {"inputs": inputs, "scale": scale})
    expected_tree = model(inputs, {"scale": scale})
    expected = jax.tree.leaves(expected_tree)
    for actual_leaf, expected_leaf in zip(actual, expected):
        np.testing.assert_allclose(actual_leaf, np.asarray(expected_leaf), rtol=2.0e-6, atol=2.0e-6)


@pytest.mark.parametrize(
    ("dimension_numbers", "input_shape", "kernel_shape"),
    [
        (("NCHW", "OIHW", "NCHW"), (2, 3, 8, 7), (4, 3, 3, 3)),
        (("NHWC", "HWIO", "NHWC"), (2, 8, 7, 3), (3, 3, 3, 4)),
    ],
)
def test_generic_export_handles_common_convolution_layouts(
    tmp_path: Path,
    dimension_numbers: tuple[str, str, str],
    input_shape: tuple[int, ...],
    kernel_shape: tuple[int, ...],
) -> None:
    kernel = np.linspace(-0.2, 0.3, np.prod(kernel_shape), dtype=np.float32).reshape(
        kernel_shape
    )

    def model(inputs: jax.Array) -> jax.Array:
        return jax.lax.conv_general_dilated(
            inputs,
            jnp.asarray(kernel),
            window_strides=(2, 1),
            padding="SAME",
            dimension_numbers=dimension_numbers,
        )

    path = tmp_path / f"conv_{dimension_numbers[0].lower()}.onnx"
    export_jax_to_onnx(
        model,
        (jax.ShapeDtypeStruct(input_shape, jnp.float32),),
        path,
        input_names=("image",),
        output_names=("features",),
    )
    session = _session(path)
    actual_shape = (5, *input_shape[1:])
    inputs = np.linspace(-1.0, 1.0, np.prod(actual_shape), dtype=np.float32).reshape(
        actual_shape
    )
    actual = session.run(["features"], {"image": inputs})[0]
    expected = np.asarray(model(inputs))
    np.testing.assert_allclose(actual, expected, rtol=2.0e-5, atol=2.0e-5)


def test_generic_export_fails_closed_with_equation_diagnostic(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedJaxPrimitive, match="unsupported JAX primitive 'sort'"):
        export_jax_to_onnx(
            lambda inputs: jnp.sort(inputs, axis=1),
            (jax.ShapeDtypeStruct((3, 5), jnp.float32),),
            tmp_path / "unsupported.onnx",
        )


def test_generic_export_handles_embedding_gather(tmp_path: Path) -> None:
    embedding = jnp.arange(60, dtype=jnp.float32).reshape(10, 6) / 30.0

    def model(indices: jax.Array) -> jax.Array:
        return embedding[indices]

    path = tmp_path / "embedding.onnx"
    export_jax_to_onnx(
        model,
        (jax.ShapeDtypeStruct((3, 2), jnp.int32),),
        path,
        input_names=("indices",),
        output_names=("vectors",),
    )
    session = _session(path)
    indices = np.asarray([[0, 3], [4, 9], [2, 6], [1, 8], [5, 7]], dtype=np.int32)
    actual = session.run(["vectors"], {"indices": indices})[0]
    np.testing.assert_array_equal(actual, np.asarray(model(indices)))


def test_supported_primitive_list_and_metadata(tmp_path: Path) -> None:
    assert {
        "cond",
        "conv_general_dilated",
        "cumsum",
        "dot_general",
        "dynamic_slice",
        "dynamic_update_slice",
        "jit",
        "reduce_window",
        "scan",
        "scatter",
        "scatter-add",
        "slice",
        "top_k",
        "while",
    } <= set(supported_primitives())
    path = tmp_path / "metadata.onnx"
    export_jax_to_onnx(
        lambda inputs: jnp.sin(inputs) + jnp.float32(1),
        (jax.ShapeDtypeStruct((2, 3), jnp.float32),),
        path,
        metadata={"model_family": "test"},
    )
    model = onnx.load(path)
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    assert metadata["exporter"] == "fail_closed_jaxpr_to_onnx"
    assert metadata["model_family"] == "test"
    assert "sin:1" in metadata["supported_primitive_counts"]


def test_generic_export_is_byte_deterministic(tmp_path: Path) -> None:
    weight = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)

    def model(inputs: jax.Array) -> jax.Array:
        return jnp.tanh(inputs @ weight)

    first = tmp_path / "first.onnx"
    second = tmp_path / "second.onnx"
    options = {
        "input_names": ("state",),
        "output_names": ("prediction",),
        "metadata": {"revision": "fixed"},
    }
    example = (jax.ShapeDtypeStruct((3, 3), jnp.float32),)
    export_jax_to_onnx(model, example, first, **options)
    export_jax_to_onnx(model, example, second, **options)
    assert first.read_bytes() == second.read_bytes()



@pytest.mark.parametrize(
    ("reduction", "input_shape", "window", "strides"),
    [
        ("max", (2, 8, 7, 3), (1, 2, 3, 1), (1, 2, 2, 1)),
        ("sum", (2, 3, 8, 7), (1, 1, 2, 3), (1, 1, 2, 2)),
    ],
)
def test_generic_export_handles_pooling_layouts_and_dynamic_batch(
    tmp_path: Path,
    reduction: str,
    input_shape: tuple[int, ...],
    window: tuple[int, ...],
    strides: tuple[int, ...],
) -> None:
    def model(inputs: jax.Array) -> jax.Array:
        if reduction == "max":
            return jax.lax.reduce_window(
                inputs, -jnp.inf, jax.lax.max, window, strides, "SAME"
            )
        return jax.lax.reduce_window(
            inputs, jnp.float32(0), jax.lax.add, window, strides, "SAME"
        )

    path = tmp_path / f"{reduction}_pool.onnx"
    export_jax_to_onnx(
        model,
        (jax.ShapeDtypeStruct(input_shape, jnp.float32),),
        path,
        input_names=("features",),
        output_names=("pooled",),
    )
    session = _session(path)
    rng = np.random.default_rng(19)
    for batch in (1, 2, 5):
        inputs = rng.normal(size=(batch, *input_shape[1:])).astype(np.float32)
        actual = session.run(["pooled"], {"features": inputs})[0]
        expected = np.asarray(model(inputs))
        np.testing.assert_allclose(actual, expected, rtol=3.0e-5, atol=3.0e-5)


def test_generic_export_handles_dynamic_slice_clamping(tmp_path: Path) -> None:
    def model(
        inputs: jax.Array, row: jax.Array, column: jax.Array
    ) -> jax.Array:
        return jax.lax.dynamic_slice(inputs, (row, column), (3, 4))

    path = tmp_path / "dynamic_slice.onnx"
    export_jax_to_onnx(
        model,
        (
            jax.ShapeDtypeStruct((8, 9), jnp.float32),
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.int32),
        ),
        path,
        input_names=("values", "row", "column"),
        output_names=("window",),
        dynamic_batch=False,
    )
    session = _session(path)
    inputs = np.arange(72, dtype=np.float32).reshape(8, 9)
    for row, column in ((2, 3), (7, 8), (-2, -3)):
        row_value = np.asarray(row, dtype=np.int32)
        column_value = np.asarray(column, dtype=np.int32)
        actual = session.run(
            ["window"],
            {"values": inputs, "row": row_value, "column": column_value},
        )[0]
        expected = np.asarray(model(inputs, row_value, column_value))
        np.testing.assert_array_equal(actual, expected)


def test_generic_export_handles_cumsum_and_top_k(tmp_path: Path) -> None:
    def cumulative(inputs: jax.Array) -> tuple[jax.Array, jax.Array]:
        return (
            jax.lax.cumsum(inputs, axis=1),
            jax.lax.cumsum(inputs, axis=1, reverse=True),
        )

    cumulative_path = tmp_path / "cumsum.onnx"
    export_jax_to_onnx(
        cumulative,
        (jax.ShapeDtypeStruct((3, 5), jnp.float32),),
        cumulative_path,
        input_names=("values",),
        output_names=("forward", "reverse"),
    )
    rng = np.random.default_rng(20)
    cumulative_inputs = rng.normal(size=(7, 5)).astype(np.float32)
    actual_cumulative = _session(cumulative_path).run(
        None, {"values": cumulative_inputs}
    )
    expected_cumulative = cumulative(cumulative_inputs)
    for actual, expected in zip(actual_cumulative, expected_cumulative):
        np.testing.assert_allclose(actual, np.asarray(expected), rtol=2.0e-6, atol=2.0e-6)

    def top_k(inputs: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jax.lax.top_k(inputs, 3)

    top_k_path = tmp_path / "top_k.onnx"
    export_jax_to_onnx(
        top_k,
        (jax.ShapeDtypeStruct((2, 6), jnp.float32),),
        top_k_path,
        input_names=("scores",),
        output_names=("values", "indices"),
    )
    scores = np.asarray(
        [[2, 2, 1, 5, 5, 0], [3, 1, 4, 1, 5, 9], [8, 7, 6, 5, 4, 3]],
        dtype=np.float32,
    )
    actual_top_k = _session(top_k_path).run(None, {"scores": scores})
    expected_top_k = top_k(scores)
    np.testing.assert_array_equal(actual_top_k[0], np.asarray(expected_top_k[0]))
    np.testing.assert_array_equal(actual_top_k[1], np.asarray(expected_top_k[1]))


def test_generic_export_handles_dynamic_cond_multioutput(tmp_path: Path) -> None:
    def model(
        predicate: jax.Array, values: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return jax.lax.cond(
            predicate,
            lambda operand: (operand * 2, operand + 1),
            lambda operand: (operand - 2, -operand),
            values,
        )

    path = tmp_path / "cond.onnx"
    export_jax_to_onnx(
        model,
        (
            jax.ShapeDtypeStruct((), jnp.bool_),
            jax.ShapeDtypeStruct((3, 4), jnp.float32),
        ),
        path,
        input_names=("predicate", "values"),
        output_names=("first", "second"),
    )
    session = _session(path)
    rng = np.random.default_rng(21)
    for predicate in (False, True):
        for batch in (1, 3, 5):
            values = rng.normal(size=(batch, 4)).astype(np.float32)
            predicate_value = np.asarray(predicate)
            actual = session.run(
                None, {"predicate": predicate_value, "values": values}
            )
            expected = model(predicate_value, values)
            for actual_leaf, expected_leaf in zip(actual, expected):
                np.testing.assert_allclose(
                    actual_leaf, np.asarray(expected_leaf), rtol=2.0e-6, atol=2.0e-6
                )


@pytest.mark.parametrize("reverse", [False, True])
def test_generic_export_handles_dynamic_length_scan(
    tmp_path: Path, reverse: bool
) -> None:
    weight = jnp.arange(16, dtype=jnp.float32).reshape(4, 4) / 40

    def model(
        sequence: jax.Array,
        initial_state: jax.Array,
        initial_accumulator: jax.Array,
    ) -> tuple[
        tuple[jax.Array, jax.Array],
        tuple[jax.Array, jax.Array],
    ]:
        def step(
            carry: tuple[jax.Array, jax.Array], item: jax.Array
        ) -> tuple[
            tuple[jax.Array, jax.Array],
            tuple[jax.Array, jax.Array],
        ]:
            state, accumulator = carry
            state = jnp.tanh(state @ weight + item)
            accumulator = accumulator + state
            return (state, accumulator), (state, accumulator)

        return jax.lax.scan(
            step,
            (initial_state, initial_accumulator),
            sequence,
            reverse=reverse,
        )

    path = tmp_path / f"scan_{reverse}.onnx"
    export_jax_to_onnx(
        model,
        (
            jax.ShapeDtypeStruct((6, 4), jnp.float32),
            jax.ShapeDtypeStruct((4,), jnp.float32),
            jax.ShapeDtypeStruct((4,), jnp.float32),
        ),
        path,
        input_names=("sequence", "initial_state", "initial_accumulator"),
        output_names=("final_state", "final_accumulator", "states", "accumulators"),
        dynamic_batch=False,
        dynamic_axes={"sequence": {0: "steps"}},
    )
    session = _session(path)
    rng = np.random.default_rng(22)
    for steps in (2, 6, 9):
        sequence = rng.normal(size=(steps, 4)).astype(np.float32)
        initial_state = rng.normal(size=(4,)).astype(np.float32)
        initial_accumulator = rng.normal(size=(4,)).astype(np.float32)
        actual = session.run(
            None,
            {
                "sequence": sequence,
                "initial_state": initial_state,
                "initial_accumulator": initial_accumulator,
            },
        )
        expected = jax.tree.leaves(
            model(sequence, initial_state, initial_accumulator)
        )
        for actual_leaf, expected_leaf in zip(actual, expected):
            np.testing.assert_allclose(
                actual_leaf, np.asarray(expected_leaf), rtol=3.0e-5, atol=3.0e-5
            )


def test_generic_export_rejects_onnx_unsupported_pool_dtype(
    tmp_path: Path,
) -> None:
    def model(inputs: jax.Array) -> jax.Array:
        return jax.lax.reduce_window(
            inputs,
            jnp.iinfo(jnp.int32).min,
            jax.lax.max,
            (1, 1, 2, 2),
            (1, 1, 2, 2),
            "VALID",
        )

    with pytest.raises(UnsupportedJaxPrimitive, match="MaxPool does not support int32"):
        export_jax_to_onnx(
            model,
            (jax.ShapeDtypeStruct((2, 3, 8, 8), jnp.int32),),
            tmp_path / "unsupported_int32_pool.onnx",
        )


def test_generic_export_handles_dynamic_update_slice_and_kv_cache(
    tmp_path: Path,
) -> None:
    def update_window(
        operand: jax.Array,
        update: jax.Array,
        row: jax.Array,
        column: jax.Array,
    ) -> jax.Array:
        return jax.lax.dynamic_update_slice(
            operand, update, (row, column)
        )

    window_path = tmp_path / "dynamic_update.onnx"
    export_jax_to_onnx(
        update_window,
        (
            jax.ShapeDtypeStruct((8, 9), jnp.float32),
            jax.ShapeDtypeStruct((3, 4), jnp.float32),
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.int32),
        ),
        window_path,
        input_names=("operand", "update", "row", "column"),
        output_names=("result",),
        dynamic_batch=False,
    )
    window_session = _session(window_path)
    operand = np.arange(72, dtype=np.float32).reshape(8, 9)
    update = np.full((3, 4), -5, dtype=np.float32)
    for row, column in ((2, 3), (7, 8), (-2, -3)):
        inputs = {
            "operand": operand,
            "update": update,
            "row": np.asarray(row, dtype=np.int32),
            "column": np.asarray(column, dtype=np.int32),
        }
        actual = window_session.run(["result"], inputs)[0]
        expected = update_window(*inputs.values())
        np.testing.assert_array_equal(actual, np.asarray(expected))

    def update_cache(
        cache: jax.Array, value: jax.Array, position: jax.Array
    ) -> jax.Array:
        return jax.lax.dynamic_update_slice(
            cache, value, (0, 0, position, 0)
        )

    cache_path = tmp_path / "kv_update.onnx"
    export_jax_to_onnx(
        update_cache,
        (
            jax.ShapeDtypeStruct((2, 3, 8, 4), jnp.float32),
            jax.ShapeDtypeStruct((2, 3, 1, 4), jnp.float32),
            jax.ShapeDtypeStruct((), jnp.int32),
        ),
        cache_path,
        input_names=("cache", "value", "position"),
        output_names=("updated",),
        dynamic_batch=False,
    )
    cache_session = _session(cache_path)
    rng = np.random.default_rng(23)
    for position in (0, 4, 9, -2):
        cache = rng.normal(size=(2, 3, 8, 4)).astype(np.float32)
        value = rng.normal(size=(2, 3, 1, 4)).astype(np.float32)
        position_value = np.asarray(position, dtype=np.int32)
        actual = cache_session.run(
            ["updated"],
            {"cache": cache, "value": value, "position": position_value},
        )[0]
        expected = update_cache(cache, value, position_value)
        np.testing.assert_array_equal(actual, np.asarray(expected))


def test_generic_export_handles_gather_elements_and_coordinates(
    tmp_path: Path,
) -> None:
    def take_axis(inputs: jax.Array, indices: jax.Array) -> jax.Array:
        return jnp.take_along_axis(inputs, indices, axis=1)

    elements_path = tmp_path / "gather_elements.onnx"
    export_jax_to_onnx(
        take_axis,
        (
            jax.ShapeDtypeStruct((3, 8), jnp.float32),
            jax.ShapeDtypeStruct((3, 3), jnp.int32),
        ),
        elements_path,
        input_names=("values", "indices"),
        output_names=("selected",),
        dynamic_batch=False,
    )
    values = np.arange(24, dtype=np.float32).reshape(3, 8)
    indices = np.asarray(
        [[1, 7, 8], [-1, -9, 3], [2, 5, 99]], dtype=np.int32
    )
    actual = _session(elements_path).run(
        ["selected"], {"values": values, "indices": indices}
    )[0]
    expected = take_axis(values, indices)
    np.testing.assert_allclose(
        actual, np.asarray(expected), rtol=0, atol=0, equal_nan=True
    )

    def coordinate_gather(
        inputs: jax.Array, rows: jax.Array, columns: jax.Array
    ) -> jax.Array:
        return inputs[rows, columns]

    coordinates_path = tmp_path / "gather_coordinates.onnx"
    export_jax_to_onnx(
        coordinate_gather,
        (
            jax.ShapeDtypeStruct((5, 6), jnp.float32),
            jax.ShapeDtypeStruct((3,), jnp.int32),
            jax.ShapeDtypeStruct((3,), jnp.int32),
        ),
        coordinates_path,
        input_names=("values", "rows", "columns"),
        output_names=("selected",),
        dynamic_batch=False,
    )
    coordinate_values = np.arange(30, dtype=np.float32).reshape(5, 6)
    rows = np.asarray([0, 3, 4], dtype=np.int32)
    columns = np.asarray([2, 1, 5], dtype=np.int32)
    actual = _session(coordinates_path).run(
        ["selected"],
        {"values": coordinate_values, "rows": rows, "columns": columns},
    )[0]
    expected = coordinate_gather(coordinate_values, rows, columns)
    np.testing.assert_array_equal(actual, np.asarray(expected))


@pytest.mark.parametrize("mode", ["clip", "fill"])
def test_generic_export_handles_gather_nd_boundary_modes(
    tmp_path: Path, mode: str
) -> None:
    def model(inputs: jax.Array, rows: jax.Array) -> jax.Array:
        if mode == "fill":
            return jnp.asarray(inputs).at[rows].get(
                mode="fill", fill_value=np.float32(-123)
            )
        return jnp.asarray(inputs).at[rows].get(mode="clip")

    path = tmp_path / f"gather_nd_{mode}.onnx"
    export_jax_to_onnx(
        model,
        (
            jax.ShapeDtypeStruct((5, 6), jnp.float32),
            jax.ShapeDtypeStruct((3,), jnp.int32),
        ),
        path,
        input_names=("values", "rows"),
        output_names=("selected",),
        dynamic_batch=False,
    )
    values = np.arange(30, dtype=np.float32).reshape(5, 6)
    rows = np.asarray([-9, 2, 99], dtype=np.int32)
    actual = _session(path).run(
        ["selected"], {"values": values, "rows": rows}
    )[0]
    expected = model(values, rows)
    np.testing.assert_array_equal(actual, np.asarray(expected))


@pytest.mark.parametrize(
    ("mode", "indices"),
    [
        ("promise_in_bounds", np.asarray([1, 4, 1], dtype=np.int32)),
        ("clip", np.asarray([-9, 4, 99], dtype=np.int32)),
        ("drop", np.asarray([-9, 4, 99, 4], dtype=np.int32)),
    ],
)
def test_generic_export_handles_scatter_add_modes_and_duplicates(
    tmp_path: Path, mode: str, indices: np.ndarray
) -> None:
    def model(
        operand: jax.Array, positions: jax.Array, updates: jax.Array
    ) -> jax.Array:
        return jnp.asarray(operand).at[positions].add(updates, mode=mode)

    updates = (
        np.arange(indices.size * 3, dtype=np.float32).reshape(indices.size, 3)
        / 7
    )
    path = tmp_path / f"scatter_add_{mode}.onnx"
    export_jax_to_onnx(
        model,
        (
            jax.ShapeDtypeStruct((8, 3), jnp.float32),
            jax.ShapeDtypeStruct(indices.shape, jnp.int32),
            jax.ShapeDtypeStruct(updates.shape, jnp.float32),
        ),
        path,
        input_names=("operand", "indices", "updates"),
        output_names=("result",),
        dynamic_batch=False,
    )
    operand = np.zeros((8, 3), dtype=np.float32)
    actual = _session(path).run(
        ["result"],
        {"operand": operand, "indices": indices, "updates": updates},
    )[0]
    expected = model(operand, indices, updates)
    np.testing.assert_allclose(
        actual, np.asarray(expected), rtol=2.0e-6, atol=2.0e-6
    )


@pytest.mark.parametrize(
    ("mode", "indices"),
    [
        ("promise_in_bounds", np.asarray([1, 4, 6], dtype=np.int32)),
        ("clip", np.asarray([-9, 4, 99], dtype=np.int32)),
        ("drop", np.asarray([-9, 4, 99], dtype=np.int32)),
    ],
)
def test_generic_export_handles_unique_scatter_set_modes(
    tmp_path: Path, mode: str, indices: np.ndarray
) -> None:
    def model(
        operand: jax.Array, positions: jax.Array, updates: jax.Array
    ) -> jax.Array:
        return jnp.asarray(operand).at[positions].set(
            updates, mode=mode, unique_indices=True
        )

    updates = np.arange(9, dtype=np.float32).reshape(3, 3)
    path = tmp_path / f"scatter_set_{mode}.onnx"
    export_jax_to_onnx(
        model,
        (
            jax.ShapeDtypeStruct((8, 3), jnp.float32),
            jax.ShapeDtypeStruct((3,), jnp.int32),
            jax.ShapeDtypeStruct((3, 3), jnp.float32),
        ),
        path,
        input_names=("operand", "indices", "updates"),
        output_names=("result",),
        dynamic_batch=False,
    )
    operand = np.zeros((8, 3), dtype=np.float32)
    actual = _session(path).run(
        ["result"],
        {"operand": operand, "indices": indices, "updates": updates},
    )[0]
    expected = model(operand, indices, updates)
    np.testing.assert_array_equal(actual, np.asarray(expected))


@pytest.mark.parametrize("dtype", [np.int32, np.bool_])
def test_generic_export_handles_typed_scatter_set_drop(
    tmp_path: Path, dtype: type[np.generic]
) -> None:
    def model(
        operand: jax.Array, positions: jax.Array, updates: jax.Array
    ) -> jax.Array:
        return jnp.asarray(operand).at[positions].set(
            updates, mode="drop", unique_indices=True
        )

    indices = np.asarray([-9, 4, 99], dtype=np.int32)
    operand = np.arange(24).reshape(8, 3).astype(dtype)
    updates = (100 + np.arange(9)).reshape(3, 3).astype(dtype)
    path = tmp_path / f"scatter_set_drop_{np.dtype(dtype).name}.onnx"
    export_jax_to_onnx(
        model,
        (
            jax.ShapeDtypeStruct((8, 3), dtype),
            jax.ShapeDtypeStruct((3,), jnp.int32),
            jax.ShapeDtypeStruct((3, 3), dtype),
        ),
        path,
        input_names=("operand", "indices", "updates"),
        output_names=("result",),
        dynamic_batch=False,
    )
    actual = _session(path).run(
        ["result"],
        {"operand": operand, "indices": indices, "updates": updates},
    )[0]
    expected = model(operand, indices, updates)
    np.testing.assert_array_equal(actual, np.asarray(expected))


def test_generic_export_handles_coordinate_scatter_add(tmp_path: Path) -> None:
    def model(
        operand: jax.Array,
        rows: jax.Array,
        columns: jax.Array,
        updates: jax.Array,
    ) -> jax.Array:
        return jnp.asarray(operand).at[rows, columns].add(
            updates, mode="drop"
        )

    path = tmp_path / "coordinate_scatter_add.onnx"
    export_jax_to_onnx(
        model,
        (
            jax.ShapeDtypeStruct((5, 6), jnp.float32),
            jax.ShapeDtypeStruct((5,), jnp.int32),
            jax.ShapeDtypeStruct((5,), jnp.int32),
            jax.ShapeDtypeStruct((5,), jnp.float32),
        ),
        path,
        input_names=("operand", "rows", "columns", "updates"),
        output_names=("result",),
        dynamic_batch=False,
    )
    operand = np.zeros((5, 6), dtype=np.float32)
    rows = np.asarray([0, -9, 3, 4, 99], dtype=np.int32)
    columns = np.asarray([2, 1, 99, 5, 0], dtype=np.int32)
    updates = np.asarray([1, 2, 3, 4, 5], dtype=np.float32)
    actual = _session(path).run(
        ["result"],
        {
            "operand": operand,
            "rows": rows,
            "columns": columns,
            "updates": updates,
        },
    )[0]
    expected = model(operand, rows, columns, updates)
    np.testing.assert_array_equal(actual, np.asarray(expected))


def test_generic_export_handles_coordinate_scatter_set_drop(
    tmp_path: Path,
) -> None:
    def model(
        operand: jax.Array,
        rows: jax.Array,
        columns: jax.Array,
        updates: jax.Array,
    ) -> jax.Array:
        return jnp.asarray(operand).at[rows, columns].set(
            updates, mode="drop", unique_indices=True
        )

    path = tmp_path / "coordinate_scatter_set_drop.onnx"
    export_jax_to_onnx(
        model,
        (
            jax.ShapeDtypeStruct((5, 6), jnp.int32),
            jax.ShapeDtypeStruct((5,), jnp.int32),
            jax.ShapeDtypeStruct((5,), jnp.int32),
            jax.ShapeDtypeStruct((5,), jnp.int32),
        ),
        path,
        input_names=("operand", "rows", "columns", "updates"),
        output_names=("result",),
        dynamic_batch=False,
    )
    operand = np.arange(30, dtype=np.int32).reshape(5, 6)
    rows = np.asarray([0, -9, 3, 4, 99], dtype=np.int32)
    columns = np.asarray([2, 1, 99, 5, 0], dtype=np.int32)
    updates = np.asarray([101, 102, 103, 104, 105], dtype=np.int32)
    actual = _session(path).run(
        ["result"],
        {
            "operand": operand,
            "rows": rows,
            "columns": columns,
            "updates": updates,
        },
    )[0]
    expected = model(operand, rows, columns, updates)
    np.testing.assert_array_equal(actual, np.asarray(expected))


def test_generic_export_rejects_nonunique_scatter_set(tmp_path: Path) -> None:
    def model(
        operand: jax.Array, indices: jax.Array, updates: jax.Array
    ) -> jax.Array:
        return jnp.asarray(operand).at[indices].set(updates)

    with pytest.raises(
        UnsupportedJaxPrimitive, match="scatter set requires unique_indices=True"
    ):
        export_jax_to_onnx(
            model,
            (
                jax.ShapeDtypeStruct((8, 3), jnp.float32),
                jax.ShapeDtypeStruct((2,), jnp.int32),
                jax.ShapeDtypeStruct((2, 3), jnp.float32),
            ),
            tmp_path / "nonunique_scatter_set.onnx",
            dynamic_batch=False,
        )


def test_generic_export_handles_while_and_dynamic_fori_loop(
    tmp_path: Path,
) -> None:
    def while_model(
        limit: jax.Array, initial: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        def condition(state: tuple[jax.Array, jax.Array]) -> jax.Array:
            index, _ = state
            return index < limit

        def body(
            state: tuple[jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array]:
            index, value = state
            return index + 1, jnp.tanh(value + index)

        return jax.lax.while_loop(
            condition, body, (jnp.int32(0), initial)
        )

    while_path = tmp_path / "while.onnx"
    export_jax_to_onnx(
        while_model,
        (
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((3,), jnp.float32),
        ),
        while_path,
        input_names=("limit", "initial"),
        output_names=("iterations", "result"),
    )
    while_session = _session(while_path)
    rng = np.random.default_rng(24)
    for limit in (0, 1, 7):
        for size in (1, 3, 6):
            initial = rng.normal(size=(size,)).astype(np.float32)
            limit_value = np.asarray(limit, dtype=np.int32)
            actual = while_session.run(
                None, {"limit": limit_value, "initial": initial}
            )
            expected = while_model(limit_value, initial)
            np.testing.assert_array_equal(actual[0], np.asarray(expected[0]))
            np.testing.assert_allclose(
                actual[1], np.asarray(expected[1]), rtol=3.0e-6, atol=3.0e-6
            )

    def dynamic_fori(
        lower: jax.Array, upper: jax.Array, values: jax.Array
    ) -> jax.Array:
        return jax.lax.fori_loop(
            lower, upper, lambda index, value: value + index, values
        )

    fori_path = tmp_path / "dynamic_fori.onnx"
    export_jax_to_onnx(
        dynamic_fori,
        (
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((3,), jnp.float32),
        ),
        fori_path,
        input_names=("lower", "upper", "values"),
        output_names=("result",),
    )
    fori_session = _session(fori_path)
    for lower, upper in ((0, 0), (1, 5), (4, 2)):
        values = rng.normal(size=(5,)).astype(np.float32)
        lower_value = np.asarray(lower, dtype=np.int32)
        upper_value = np.asarray(upper, dtype=np.int32)
        actual = fori_session.run(
            ["result"],
            {"lower": lower_value, "upper": upper_value, "values": values},
        )[0]
        expected = dynamic_fori(lower_value, upper_value, values)
        np.testing.assert_array_equal(actual, np.asarray(expected))


def test_generic_export_handles_static_fori_and_scan_without_xs(
    tmp_path: Path,
) -> None:
    def static_fori(values: jax.Array) -> jax.Array:
        return jax.lax.fori_loop(
            0, 4, lambda index, value: value + index, values
        )

    fori_path = tmp_path / "static_fori.onnx"
    export_jax_to_onnx(
        static_fori,
        (jax.ShapeDtypeStruct((3,), jnp.float32),),
        fori_path,
        input_names=("values",),
        output_names=("result",),
    )
    rng = np.random.default_rng(25)
    for size in (1, 3, 7):
        values = rng.normal(size=(size,)).astype(np.float32)
        actual = _session(fori_path).run(["result"], {"values": values})[0]
        expected = static_fori(values)
        np.testing.assert_array_equal(actual, np.asarray(expected))

    def scan_without_xs(
        initial: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        def step(
            carry: jax.Array, unused: None
        ) -> tuple[jax.Array, jax.Array]:
            del unused
            carry = jnp.tanh(carry + 0.1)
            return carry, carry

        return jax.lax.scan(step, initial, xs=None, length=5)

    scan_path = tmp_path / "scan_without_xs.onnx"
    export_jax_to_onnx(
        scan_without_xs,
        (jax.ShapeDtypeStruct((3,), jnp.float32),),
        scan_path,
        input_names=("initial",),
        output_names=("final", "states"),
    )
    scan_session = _session(scan_path)
    for size in (1, 3, 7):
        initial = rng.normal(size=(size,)).astype(np.float32)
        actual = scan_session.run(None, {"initial": initial})
        expected = scan_without_xs(initial)
        for actual_leaf, expected_leaf in zip(actual, expected):
            np.testing.assert_allclose(
                actual_leaf, np.asarray(expected_leaf), rtol=3.0e-6, atol=3.0e-6
            )


def test_generic_export_handles_stack_batch_reshape_and_accurate_log1p(tmp_path: Path) -> None:
    weight = jax.random.normal(jax.random.PRNGKey(7), (6, 24), dtype=jnp.float32) / 3.0

    def model(inputs: jax.Array) -> jax.Array:
        stacked = jnp.stack(
            [inputs[:, 0], 2.0 * inputs[:, 1], jnp.log1p(inputs[:, 2]), jnp.expm1(inputs[:, 3])], axis=1
        )
        grouped = (inputs @ weight).reshape(inputs.shape[0], 2, 3, 4)
        return jnp.concatenate([stacked, grouped.sum(axis=(1, 2))], axis=1)

    path = tmp_path / "stack_reshape.onnx"
    # The example batch (4) equals a trailing reshape size: the leading axis must still be the dynamic batch.
    export_jax_to_onnx(
        model, (jax.ShapeDtypeStruct((4, 6), jnp.float32),), path,
        input_names=("x",), output_names=("y",),
    )
    session = _session(path)
    for batch in (4, 5, 9):
        x = np.asarray(jax.random.normal(jax.random.PRNGKey(batch), (batch, 6)), np.float32) * 0.5
        x[:, 2] = np.abs(x[:, 2])
        x[: min(batch, 3), 2] = np.asarray([1e-12, 3e-9, 2e-7], np.float32)[: min(batch, 3)]
        x[: min(batch, 3), 3] = np.asarray([-1e-12, 4e-9, -3e-7], np.float32)[: min(batch, 3)]
        (actual,) = session.run(None, {"x": x})
        expected = np.asarray(model(jnp.asarray(x)))
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-6)
        tiny = slice(0, min(batch, 3))
        np.testing.assert_allclose(actual[tiny, 2], np.log1p(x[tiny, 2].astype(np.float64)), rtol=2e-7)
        np.testing.assert_allclose(actual[tiny, 3], np.expm1(x[tiny, 3].astype(np.float64)), rtol=2e-7)


def test_generic_export_declares_minimum_ir_version_for_opset(tmp_path: Path) -> None:
    path = tmp_path / "ir.onnx"
    export_jax_to_onnx(lambda x: jnp.tanh(x), (jax.ShapeDtypeStruct((2, 3), jnp.float32),), path)
    model = onnx.load(path)
    expected = onnx.helper.find_min_ir_version_for([onnx.helper.make_opsetid("", 18)])
    assert model.ir_version == expected
    _session(path)


def test_generic_export_dynamic_batch_constants_need_no_shape_ops(tmp_path: Path) -> None:
    weight = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 7.0

    def model(inputs: jax.Array) -> jax.Array:
        n = inputs.shape[0]
        ones = jnp.ones((n, 1), jnp.float32)
        zeros = jnp.zeros((n, 2), jnp.float32)
        empty = jnp.zeros((n, 0), jnp.float32)
        features = jnp.concatenate([ones, inputs[:, :2], empty, zeros], axis=1)
        repeated = jnp.repeat(inputs, 2, axis=1).reshape(n, 2, 3).sum(axis=1)
        return jnp.concatenate([features, repeated @ weight], axis=1)

    path = tmp_path / "no_shape_ops.onnx"
    export_jax_to_onnx(model, (jax.ShapeDtypeStruct((5, 3), jnp.float32),), path)
    ops = {node.op_type for node in onnx.load(path).graph.node}
    assert not ops & {"Shape", "ConstantOfShape"}, ops
    session = _session(path)
    for batch in (1, 5, 8):
        x = np.array(jax.random.normal(jax.random.PRNGKey(batch), (batch, 3)), np.float32)
        x[0, 0] = np.nan if batch == 8 else x[0, 0]
        (actual,) = session.run(None, {session.get_inputs()[0].name: x})
        np.testing.assert_allclose(actual, np.asarray(model(jnp.asarray(x))), rtol=1e-6, atol=1e-6, equal_nan=True)


def test_export_preserves_fp64_and_bool_mixed_io(tmp_path: Path) -> None:
    with jax.enable_x64():
        def model(state, enabled):
            return (jnp.where(enabled, state + np.float64(2**-42), state),
                    jnp.logical_and(enabled, state > 0))

        path = tmp_path / "fp64_bool.onnx"
        export_jax_to_onnx(model, (
            jax.ShapeDtypeStruct((3, 2), jnp.float64),
            jax.ShapeDtypeStruct((3, 2), jnp.bool_),
        ), path, input_names=("state", "enabled"), output_names=("values", "valid"))
        graph = onnx.load(path)
        assert [value.type.tensor_type.elem_type for value in graph.graph.input] == [onnx.TensorProto.DOUBLE, onnx.TensorProto.BOOL]
        assert [value.type.tensor_type.elem_type for value in graph.graph.output] == [onnx.TensorProto.DOUBLE, onnx.TensorProto.BOOL]
        assert not any(tensor.data_type == onnx.TensorProto.FLOAT for tensor in graph.graph.initializer)
        session = _session(path)
        for batch in (1, 3, 5):
            state = np.full((batch, 2), 1 + 2**-40, dtype=np.float64)
            enabled = np.ones((batch, 2), dtype=np.bool_)
            actual = session.run(None, {"state": state, "enabled": enabled})
            assert actual[0].dtype == np.float64
            assert actual[1].dtype == np.bool_
            np.testing.assert_array_equal(actual[0], state + 2**-42)
            np.testing.assert_array_equal(actual[1], enabled)
