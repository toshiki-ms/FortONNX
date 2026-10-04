# Tensor shapes and layout

FortONNX accepts multiple named float32, float64 and bool input and output tensors per model.
Tensor rank must be positive. Model dimensions may be static or dynamic; every
dimension supplied when buffers are bound must be positive.
A static model dimension must equal the bound dimension. ONNX Runtime performs
the remaining graph-dependent shape validation during inference.

## Element types

| ONNX type | Fortran array type | C buffer element | Bytes | Type tag |
|---|---|---|---|---|
| FLOAT (`float32`) | `real(c_float)` | `float` | 4 | `fortonnx_float32` / `FORTONNX_FLOAT32` |
| DOUBLE (`float64`) | `real(c_double)` | `double` | 8 | `fortonnx_float64` / `FORTONNX_FLOAT64` |
| BOOL (`bool`) | `logical(c_bool)` | `uint8_t` (0 or 1) | 1 | `fortonnx_bool` / `FORTONNX_BOOL` |

All three types use the same `runtime%bind`, `fortonnx_host_tensor` and
`fortonnx_device_tensor` generics and the same borrowed-buffer lifetime rules.
Host assumed-rank overloads have no API rank ceiling (subject to compiler
array-rank limits); direct CUDA Fortran arrays and device views support ranks 1 through 7. A direct input/output pair
can have different element types; device pairs must still have the same rank.
Use independent views for heterogeneous ranks or multiple inputs/outputs.

Array overloads infer the caller's element type. Binding compares it against
the model tensor and reports the direction, ONNX name, expected type and
caller type on mismatch (for example `input tensor 'state' element type
mismatch: model float64, caller float32`). No data conversion or transpose is
performed. Default Fortran `logical` is **not** an ONNX BOOL buffer: use
`logical(c_bool)` and `.true._c_bool` / `.false._c_bool`.

With `nvfortran`, compile applications that use BOOL buffers with
`-Munixlogical` so true values are stored as 1, not the default all-one bits.
The default FortONNX `nvfortran` flags include this option; keep it when
overriding `FFLAGS` or compiling an application separately. See the
[NVIDIA compiler reference](https://docs.nvidia.com/hpc-sdk/compilers/hpc-compilers-ref-guide/).

```fortran
real(c_double), target :: state(2, rows), values(2, rows)
logical(c_bool), target :: enabled(2, rows), valid(2, rows)
type(fortonnx_tensor) :: inputs(2), outputs(2)

inputs = [fortonnx_host_tensor(state, 'state'), &
          fortonnx_host_tensor(enabled, 'enabled')]
outputs = [fortonnx_host_tensor(values, 'values'), &
           fortonnx_host_tensor(valid, 'valid')]
call runtime%bind('scientific', inputs, outputs, status)
```

Inspect model types with `runtime%tensor_type(name, fortonnx_input, index,
element_type, status)` (or `fortonnx_output`); indices are one-based, as with
`tensor_shape`. C callers use `fortonnx_session_get_tensor_type_at`, with
zero-based indices.

CPU and CUDA buffer paths preserve all FP64 bits. TensorRT requests containing
FP64 in tensor types, constants, nested graphs or function bodies instead use
CUDA for the entire model, preventing TensorRT DOUBLE-weight downcasts.
Uninspectable graphs conservatively use CUDA too. Unsupported CUDA operators
produce a load/run error rather than silently using host buffers or FP32.

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

Pointer views take an optional `element_type` keyword, defaulting to float32
for backwards compatibility:

```fortran
inputs(1) = fortonnx_pointer_tensor(state_ptr, state_shape, fortonnx_cuda, &
    'state', element_type=fortonnx_float64)
outputs(1) = fortonnx_pointer_tensor(valid_ptr, valid_shape, fortonnx_cuda, &
    'valid', element_type=fortonnx_bool)
```

The single-I/O pointer `bind` overloads likewise accept optional trailing
`input_type` and `output_type` keywords (both default to `fortonnx_float32`),
with either explicit shapes or the legacy batch-size argument. The caller is
responsible for the actual allocation size and correct type tag.

The C `_typed` entry points (`fortonnx_session_bind_input_typed`,
`fortonnx_session_bind_output_typed`, `fortonnx_session_bind_tensor_typed`)
accept explicit element-type tags. The original C functions remain float32
bindings. The shared CPU/CUDA bridge computes overflow-checked byte counts
using the model type and creates ONNX tensors of exactly that type.

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
