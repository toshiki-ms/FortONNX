# Contributing

Keep the portable Fortran path standard-conforming. Compiler-specific code in
the common module must be guarded by a narrowly scoped build-time feature and
must not change the common model-management API.
New public operations must report failures through `fortonnx_status`; the
library must not terminate its caller.

Before submitting a change, run:

```sh
make test
make test-python
```

GPU changes should additionally be tested with both CUDA and TensorRT builds.
Do not commit model weights, generated ONNX files, TensorRT caches, shared
provider libraries, or compiler module files.
