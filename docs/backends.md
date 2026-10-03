# Execution backends

The selected backend is stored in `fortonnx_options%backend` and applies to all
models loaded into one runtime.

## CPU

`fortonnx_cpu` creates one ONNX Runtime environment with global thread pools.
All sessions in the runtime share the configured intra-op and inter-op pools.
An intra-op value of zero lets ONNX Runtime choose its default. Inter-op must be
at least one; values greater than one enable parallel graph execution.

CPU execution works with `gfortran`, `flang`, `ifx`, and `nvfortran`. The
portable source has no OpenMP or vendor module dependency.

## CUDA

`fortonnx_cuda` registers the CUDA execution provider. `use_tf32` is forwarded
when the session is created. If `user_compute_stream` is null, ONNX Runtime
synchronizes execution before `run` returns. If it is non-null, FortONNX asks
ONNX Runtime not to synchronize providers; the caller owns and synchronizes
the stream.

## TensorRT

`fortonnx_tensorrt` registers TensorRT and then CUDA. TensorRT executes the
supported partitions and CUDA handles remaining nodes. FP64 graphs bypass
TensorRT entirely and use CUDA, because TensorRT's ONNX parser can cast DOUBLE
weights to FLOAT (see the [NVIDIA parser documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/c-api/_nv_onnx_parser_8h_source.html)).
FortONNX inspects tensor types, constants, nested graphs and function bodies
before registering TensorRT; failed inspection conservatively selects CUDA.
This affects provider selection, not the declared tensor types or data.
The device-pointer path disables CPU execution-provider fallback; unavailable
FP64/bool CUDA kernels return an error, never a precision-reducing conversion.
The cache directory is required when engine caching is enabled. Engine and timing cache files are
runtime artifacts and must not be committed.

The compiled ONNX Runtime library determines whether CUDA and TensorRT are
available. A CPU-only FortONNX build contains an explicit GPU stub and returns
a status error if GPU model loading is attempted.

### Tested CUDA 13 Conda environment

The TensorRT backend was validated with Python 3.13, ONNX Runtime GPU 1.29.0,
ONNX 1.22.0, and TensorRT 10.16.1.11 for CUDA 13:

```sh
conda activate gpu
python -m pip install onnxruntime-gpu==1.29.0 onnx==1.22.0 \
  tensorrt-cu13==10.16.1.11
```

The ONNX Runtime wheel requires TensorRT major version 10
(`libnvinfer.so.10`) in this configuration. The TensorRT, CUDA, and cuDNN
library directories must be present in the runtime loader path supplied to
`configure`. The wheel does not provide `onnxruntime_c_api.h`; use headers
from the same ONNX Runtime release.

TensorRT and its Python wheels are external NVIDIA dependencies and are not
covered by the FortONNX BSD-2-Clause license.

