# Changelog

## 0.1.0 - unreleased

- Add `fortonnx_kernelgen`: for an ONNX graph that acts on every row
  independently, one fused CUDA kernel per row, the ONNX Runtime custom
  operator that runs it, and the graph that uses the operator (with the
  original graph as the operator's function). Unsupported and non-row-wise
  operators stop the generation. Optionally the kernel uses the tensor cores
  of Hopper GPUs for its product sums and the base-2 instructions for its
  functions, in the forms of hand-written kernels. The library also has a
  direct entry (four C functions) that runs the kernel of a graph file
  without ONNX Runtime.

- Add the `cuda_graph` option of the CUDA and TensorRT backends: CUDA Graph
  capture and replay, one graph per binding with the CUDA provider and one
  binding per model with a TensorRT engine; models with random-number
  operators, and with GatherND on the CUDA provider, are refused.
- JAX export: Gather instead of GatherND for one indexed axis, MatMul instead
  of Einsum for one contracted axis pair, one Split for slices that
  partition an axis, no Expand before an elementwise operator that
  broadcasts (unless the Expand adds a dynamic axis), and no nodes whose
  results reach no graph output.

- Add zero-copy float64 and one-byte bool tensors on CPU and CUDA, with
  type-checked C entry points and Fortran array/view generics.
- Preserve existing float32 APIs and add model tensor element-type inspection.
- Route FP64 TensorRT requests through CUDA without precision reduction.
- Decode scientific-package CSV cases using per-tensor float32/float64/bool
  dtypes, including mixed multiple outputs and exact boolean comparisons.
- Add FP64/bool CPU/CUDA and JAX/validator regression tests; add
  `test-gpu-build` for compiling GPU tests without running them.

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
