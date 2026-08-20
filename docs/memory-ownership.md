# Memory and lifetime rules

Host and device buffers are borrowed, not owned, by FortONNX.

1. A buffer must remain allocated at the same address after `runtime%bind`.
2. Rebinding replaces the previous input/output pair for that model.
   Shape arrays and temporary `fortonnx_host_tensor`/`fortonnx_device_tensor`
   views are copied during binding; only the underlying data buffers must
   remain alive.
3. A synchronous CPU or GPU `run` may be followed immediately by buffer use.
4. With a caller-owned CUDA stream, `run` only enqueues work. Synchronize the
   stream before reading output, rebinding, deallocating buffers, unloading the
   model, or closing the runtime.
5. Close the runtime before destroying a CPU context indirectly owned by it.

One `fortonnx_runtime` should not be mutated concurrently. Independent runtime
objects may be used by independent application threads. Reusing one session
concurrently would also race on its reusable ONNX I/O binding.
