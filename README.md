# FortONNX

Modern Fortran inference with ONNX Runtime on CPU, CUDA, and TensorRT.

FortONNX provides a small Fortran API for loading and running several ONNX
surrogate models. The portable runtime uses standard Fortran 2018 and
`iso_c_binding`. CUDA Fortran support is selected internally at build time;
applications use the same `fortonnx` module and `runtime%bind` interface on
CPU and GPU. A fail-closed JAXPR-to-ONNX exporter is distributed as a separate
Python package in the same repository.

FortONNX is an independent project and is not affiliated with Microsoft or the
ONNX project.

## Current scope

Version 0.1 supports dense and convolutional scientific surrogates through a
small tensor interface:

- multiple named tensor inputs and outputs per model;
- `float32` tensors of any positive rank through pointer-plus-shape binding;
- static and dynamic ONNX dimensions, with concrete positive dimensions at bind time;
- direct contiguous host arrays, plus CUDA Fortran device arrays of ranks 1 through 7;
- multiple named models in one `fortonnx_runtime`;
- ONNX Runtime graph optimizations;
- shared CPU intra-op/inter-op thread pools;
- CPU, CUDA, and TensorRT execution providers;
- GPU-resident input and output through device pointers;
- asynchronous GPU execution when a caller-owned CUDA stream is supplied.

Unsupported tensor dtypes and non-tensor model arguments are rejected when the
model is loaded. Additional dtype support is planned after this small API is
stable.

## Requirements

- ONNX Runtime shared library and `onnxruntime_c_api.h` from the same release;
- a C compiler;
- one of `gfortran`, `flang`, `ifx`, or `nvfortran`;
- Python with `onnx` only for generating the test fixture;
- NVFortran and an ONNX Runtime GPU build for CUDA-resident Fortran arrays.

The supported LLVM compiler command is `flang`.

## Build

GNU Make is the only build system. Configure with an ONNX Runtime release root:

```sh
./configure --fc gfortran --backend cpu \
  --onnxruntime-root /opt/onnxruntime
make -j
make test
```

The include and library locations can also be supplied separately:

```sh
./configure --fc flang --backend cpu \
  --onnxruntime-include /path/to/include \
  --onnxruntime-libdir /path/to/lib
make -j
```

For a Python ONNX Runtime wheel, the shared library is detected from the active
Python environment, but the C header must normally be supplied separately.
`configure` records the exact library and its ELF soname in the ignored
`config.mk`; the repository never stores generated symlinks or runtime
binaries.

Compiler/backend combinations use separate directories below `build/native`,
so CPU and GPU builds can coexist without reusing incompatible object or module
files.

Install the library and compiler-specific module files with:

```sh
make install PREFIX=$HOME/.local
```

Fortran `.mod` files are compiler-specific. Applications must use the same
Fortran compiler family used to install FortONNX.

Compile an application against an installation with the installed helper:

```sh
gfortran $(fortonnx-config --fflags) application.f90 \
  $(fortonnx-config --libs) -o application
```

`fortonnx-config` is relocatable and contains no configure-time paths. For an
ONNX Runtime installation without standard linker names or search paths,
supply its location when invoking the helper:

```sh
export ONNXRUNTIME_LIBRARY=/path/to/libonnxruntime.so.VERSION
export ONNXRUNTIME_LIBDIR=/path/to/onnxruntime/lib
export FORTONNX_RUNTIME_LIBRARY_PATH=/path/to/provider/dependencies
gfortran $(fortonnx-config --fflags) application.f90 \
  $(fortonnx-config --libs) -o application
```

Installation does not copy or link ONNX Runtime, CUDA, cuDNN, or TensorRT
libraries into the prefix. They remain external dependencies and must be
available through the system loader or application environment.

## CPU example

```fortran
use, intrinsic :: iso_c_binding
use fortonnx

type(fortonnx_runtime) :: runtime
type(fortonnx_options) :: options
type(fortonnx_status) :: status
real(c_float), target :: x(8, 32), y(4, 32)

options%backend = fortonnx_cpu
options%intra_op_threads = 8
options%inter_op_threads = 1

call runtime%initialize(options, status)
if (.not. status%ok()) error stop status%message
call runtime%load('example', 'example.onnx', status)
if (.not. status%ok()) error stop status%message
call runtime%bind('example', x, y, status)
if (.not. status%ok()) error stop status%message
call runtime%run('example', status)
if (.not. status%ok()) error stop status%message
call runtime%close()
```

Models containing ONNX Runtime custom operators can pass the provider-specific
custom-op shared library when the model is loaded:

```fortran
call runtime%load("custom_model", "model_with_custom_ops.onnx", status, &
    custom_op_library="libmodel_ort_ops.so")
```

The same `load` call is available with the CPU, CUDA, and TensorRT backends.
The library must implement the operator for the selected execution provider
and must be ABI-compatible with the linked ONNX Runtime. CPU and CUDA builds
may therefore use different custom-op libraries with the same `.onnx` file.
The argument is optional, so existing applications and ordinary ONNX models
are unchanged.

ONNX uses `[batch, features]`, while the corresponding contiguous Fortran array
is declared `(features, batch)`. No transpose or copy is performed.

The same rule applies at every rank: the Fortran dimensions are the ONNX
dimensions in reverse order. For an NCHW convolutional model, an ONNX input
`[N,C,H,W]` is therefore declared as `image(W,H,C,N)` in Fortran. Shape
metadata, including `-1` for a dynamic dimension, is available through
`runtime%shapes`:

```fortran
real(c_float), target :: image(64, 64, 3, 32)
real(c_float), target :: feature_map(62, 62, 16, 32)
integer(c_int64_t), allocatable :: input_shape(:), output_shape(:)

call runtime%load('cnn', 'cnn.onnx', status)
call runtime%shapes('cnn', input_shape, output_shape, status)
call runtime%bind('cnn', image, feature_map, status)
call runtime%run('cnn', status)
```

The pointer overload takes explicit shapes in ONNX order and supports any
positive rank:

```fortran
call runtime%bind('cnn', input_pointer, output_pointer, &
    [32_c_int64_t, 3_c_int64_t, 64_c_int64_t, 64_c_int64_t], &
    [32_c_int64_t, 16_c_int64_t, 62_c_int64_t, 62_c_int64_t], status)
```

`runtime%features` remains as a compatibility convenience for rank-2 models
with static feature dimensions. See `docs/tensors.md` for layout and dynamic
shape rules.

## Multiple inputs

For heterogeneous input ranks, create borrowed tensor views. Names are matched
to ONNX input/output names, so the views do not need to be in model order:

```fortran
type(fortonnx_tensor) :: inputs(2)
real(c_float), target :: image(width, height, channels, batch)
real(c_float), target :: scale(1)
real(c_float), target :: result(out_width, out_height, channels, batch)

inputs(1) = fortonnx_host_tensor(scale, 'scale')
inputs(2) = fortonnx_host_tensor(image, 'image')
call runtime%bind('super_resolution', inputs, &
    fortonnx_host_tensor(result, 'output'), status)
```

Names are optional; unnamed views are bound positionally. For multiple outputs,
pass tensor-view arrays for both the `inputs` and `outputs` arguments. Use
`runtime%io_counts` and `runtime%tensor_shape` to inspect a multi-I/O signature.

See `examples/multi_model_cpu.f90` for a runtime containing several named
models. CPU parallelism is controlled by ONNX Runtime; the Fortran source does
not require OpenMP.

## Scientific model package example

`examples/scientific_model_package/` builds an ordinary linear ONNX graph as a
complete scientific package. The generated directory includes a Model Card,
format-declared machine-readable description, artifact digests, known-answer
cases, a reference verification report, and Python and Fortran execution
examples. The ONNX graph requires no special runtime behavior.

Generate only the package files with:

```sh
make scientific-package-files PYTHON=python3
```

Validate the generated package, rerun its known-answer cases, and write JSON
and text reports with:

```sh
make scientific-package-validate PYTHON=python3
```

The generated digest record demonstrates local consistency. A release decision
must supply the expected package-description digest through a separately
trusted channel.

With a configured CPU build, generate the package and run its Fortran
known-answer check with:

```sh
make scientific-package-example PYTHON=python3 BACKEND=cpu
```

See the [example instructions](examples/scientific_model_package/README.md) and
[scientific package specification](docs/scientific-model-package/README.md).

## CUDA and TensorRT

Build the same runtime with a GPU execution provider:

```sh
./configure --fc nvfortran --backend cuda \
  --onnxruntime-root /path/to/onnxruntime-gpu
make -j
```

or:

```sh
./configure --fc nvfortran --backend tensorrt \
  --onnxruntime-root /path/to/onnxruntime-gpu
make -j
```

The public `runtime%bind` interface accepts host arrays, device pointers, and,
in an NVFortran GPU build, CUDA Fortran `device` arrays. All forms are exported
from `use fortonnx`. The same module also supplies `fortonnx_cuda_stream` in an
NVFortran GPU build. A non-null user stream makes `run` asynchronous; the
caller must synchronize that stream before consuming output or destroying
buffers.

Direct CUDA Fortran array binding accepts same-rank single-input/output pairs
from rank 1 through 7. For differing ranks or multiple inputs, wrap each array
independently; this is still a borrowed zero-copy view:

```fortran
call runtime%bind('classifier', fortonnx_device_tensor(image_device), &
    fortonnx_device_tensor(logits_device), status)
```

For multiple GPU-resident inputs, construct a `type(fortonnx_tensor)` array
with `fortonnx_device_tensor(array, name)` and pass it to the same `bind`
interface used on CPU.

TensorRT is registered first and CUDA remains its fallback for unsupported
nodes. `use_tf32`, TensorRT FP16, and engine/timing cache settings live in
`fortonnx_options`, so model-management code does not change when the backend
changes.

## JAX export

Install the optional exporter:

```sh
python -m pip install -e '.[test]'
```

```python
from fortonnx_export import export_jax_to_onnx

result = export_jax_to_onnx(
    model,
    (example_input,),
    "model.onnx",
    input_names=("features",),
    output_names=("prediction",),
)
```

The exporter translates a documented set of JAX primitives and raises
`UnsupportedJaxPrimitive` with the offending JAXPR equation when it cannot
preserve semantics. It is general over supported graph topology, parameters,
PyTrees, and dynamic batch sizes, but it is not a complete JAX implementation.
See `docs/jax-export.md`.

## Repository layout

```text
src/                  portable Fortran API and C bridge
python/               separately installable JAX exporter
examples/             applications and scientific package example
tests/                deterministic CPU and exporter tests
docs/                 runtime details and scientific model-package specification
make/compiler/        compiler-specific flags
```

Build products, ONNX models, TensorRT engines, timing caches, and provider
libraries are intentionally excluded from version control.

An ONNX graph intended for scientific distribution should be accompanied by
units, coordinates, transformations, provenance, validation evidence, and a
known-answer case. See the [scientific ONNX model package
specification](docs/scientific-model-package/README.md) for the package
requirements, conformance rules, and Model Card template.

Before creating a source release, follow `RELEASE.md`; in particular run
`make distclean` and `make release-check`. Locally compiled binaries must not be
published as release assets.

## License

BSD 2-Clause License (`BSD-2-Clause`). See `LICENSE`.
