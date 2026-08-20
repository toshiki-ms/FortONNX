# Tensor shapes and layout

FortONNX accepts multiple named float32 input and output tensors per model.
Tensor rank must be positive. Model dimensions may be static or dynamic; every
dimension supplied when buffers are bound must be positive.
A static model dimension must equal the bound dimension. ONNX Runtime performs
the remaining graph-dependent shape validation during inference.

## Fortran and ONNX axis order

ONNX tensors use row-major storage and Fortran arrays use column-major storage.
FortONNX obtains a zero-copy view by reversing the dimension order rather than
transposing data:

| ONNX logical shape | Contiguous Fortran declaration |
|---|---|
| `[N,F]` | `x(F,N)` |
| NCHW `[N,C,H,W]` | `x(W,H,C,N)` |
| NHWC `[N,H,W,C]` | `x(C,W,H,N)` |
| NCDHW `[N,C,D,H,W]` | `x(W,H,D,C,N)` |

Thus, the first Fortran subscript is the fastest-changing ONNX axis. Direct
host-array and CUDA Fortran array overloads perform this shape reversal only in
metadata; they do not copy or transpose tensor elements.

## Shape inspection

`runtime%shapes` returns the model input and output shapes in ONNX order.
Dynamic dimensions are normally reported as `-1`:

```fortran
integer(c_int64_t), allocatable :: model_input(:), model_output(:)

call runtime%shapes('cnn', model_input, model_output, status)
! Example values: [-1, 3, 64, 64] and [-1, 10]
```

The older `runtime%features` method is defined only for rank-2 input and output
tensors whose second dimensions are static and positive, and only when the
model has exactly one input and one output. `runtime%shapes` has the same
single-I/O compatibility scope. Multi-I/O models use:

```fortran
call runtime%io_counts('model', input_count, output_count, status)
call runtime%tensor_shape('model', fortonnx_input, input_index, shape, status)
call runtime%tensor_shape('model', fortonnx_output, output_index, shape, status)
```

Tensor indices in this Fortran API are one-based.

## Multiple inputs and outputs

`fortonnx_host_tensor` and `fortonnx_device_tensor` create small borrowed views
containing a buffer address, its concrete ONNX-order shape, memory location,
and an optional ONNX tensor name. For example, an image and a separate control
value can be bound in either order:

```fortran
type(fortonnx_tensor) :: inputs(2)

inputs(1) = fortonnx_host_tensor(control, 'scale')
inputs(2) = fortonnx_host_tensor(image, 'image')
call runtime%bind('model', inputs, fortonnx_host_tensor(output, 'output'), status)
```

When a name is present, FortONNX resolves it against the ONNX model and rejects
unknown or duplicate names. An unnamed view is matched by its array position.
The number of supplied views must exactly equal the model signature. If there
are multiple outputs, pass arrays on both sides:

```fortran
call runtime%bind('model', input_views, output_views, status)
```

## Pointer binding

When buffers come from CUDA C, OpenACC, another library, or a custom allocator,
pass their addresses and concrete shapes in ONNX order:

```fortran
call runtime%bind('cnn', input_pointer, output_pointer, &
    [batch, channels, height, width], [batch, classes], status)
```

This overload has no rank ceiling in the FortONNX bridge. The two pointers must
refer to host memory for a CPU runtime or device memory for a CUDA/TensorRT
runtime.

Multiple foreign-pointer inputs use the same tensor-view array API:

```fortran
inputs(1) = fortonnx_pointer_tensor(image_ptr, image_shape, fortonnx_cuda, 'image')
inputs(2) = fortonnx_pointer_tensor(scale_ptr, scale_shape, fortonnx_cuda, 'scale')
call runtime%bind('model', inputs, output_view, status)
```

## CUDA Fortran arrays

An NVFortran GPU build accepts direct same-rank, single-input/output `device`
array pairs from rank 1 through rank 7. For models whose ranks differ or which
have multiple inputs, create temporary borrowed views:

```fortran
call runtime%bind('classifier', fortonnx_device_tensor(image), &
    fortonnx_device_tensor(logits), status)
```

The view records `c_devloc(array)` and its reversed shape. FortONNX copies that
small descriptor during binding and retains only the device buffer address.
