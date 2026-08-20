# Changelog

## 0.1.0 - unreleased

- Add portable Fortran runtime with named multi-model management.
- Add shared CPU thread-pool configuration.
- Add CUDA and TensorRT execution-provider configuration.
- Add GPU-resident pointer binding and asynchronous user-stream execution.
- Keep CPU and CUDA Fortran conveniences in the common `fortonnx` module.
- Add arbitrary-rank pointer binding and rank-1-through-7 array conveniences.
- Add zero-copy NCHW convolutional model tests on CPU, CUDA, and TensorRT.
- Add named multi-input/multi-output tensor views with positional fallback.
- Test heterogeneous rank-4/rank-1 inputs on CPU, CUDA, and TensorRT.
- Add fail-closed JAXPR-to-ONNX exporter.
- Extend JAX export with pooling, dynamic slice, cumulative sum, top-k,
  two-way conditionals, and scans.
- Add fixed-shape dynamic update, GatherElements/GatherND indexing, exact
  gather/scatter boundary modes, scatter-add/set, while/fori loops, and
  carry-only scans to JAX export.
- Use GNU Make as the sole build system.
- Keep installed helpers and shared-library RPATH free of build-host paths.
- Add `distclean`, release-hygiene checks, and a source-release checklist.
