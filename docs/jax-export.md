# JAXPR to ONNX export

The fortonnx_export package traces a pure function with jax.make_jaxpr, lowers
supported primitives to ONNX opset 18, checks the resulting model, and writes
provenance metadata. Parameters captured by the function closure become ONNX
initializers.

The exporter is fail-closed: an unknown primitive is an error rather than an
approximation. The supported_primitives() function returns the implemented
primitive names. The current implementation includes:

- common elementwise operations, reductions, cumulative sum, top-k, reshaping,
  static slicing, fixed-shape dynamic slicing, concatenation, and padding;
- dot_general, including common matrix multiplication and general two-operand
  contractions lowered through ONNX Einsum;
- one-, two-, and three-dimensional maximum and additive pooling in
  unambiguous NCHW or NHWC layouts;
- common NCHW/NHWC convolution;
- indexed reads through ONNX GatherElements for take-along-axis layouts and
  GatherND for coordinate-index layouts;
- fixed-shape dynamic updates, scatter-add, and unique-index scatter-set, with
  JAX promise-in-bounds, clip, and fill/drop index modes;
- nested JIT/custom-JVP calls and PyTree multi-input/multi-output interfaces;
- scalar two-branch lax.cond, lowered to ONNX If;
- lax.scan with or without an input sequence, including captured constants,
  multiple carries and outputs, reverse traversal, dynamic sequence length
  when a sequence is present, and carry-only loops;
- lax.while_loop and both static and dynamic lax.fori_loop.

Pooling requires axis zero to be the batch axis, an unambiguous channel-first
or channel-last axis, and unit base/window dilation. Additive pooling is
implemented by AveragePool followed by the exact window volume and is
therefore limited to ONNX floating-point pooling types. Max pooling accepts
the floating-point, int8, and uint8 types supported by ONNX MaxPool.

Dynamic slicing and dynamic update reproduce JAX start-index clamping but
currently require fixed operand shapes. The GatherElements path accepts the
take-along-axis dimension-number pattern. The GatherND path accepts coordinate
indices without operand/index batching dimensions and requires every
non-indexed operand axis to be selected in full. Out-of-range fill/drop gathers
are implemented as a validity mask and ONNX Where, so invalid reads return the
JAX fill value rather than a clipped value.

Scatter currently requires a fixed operand shape, no operand/index batching
dimensions, and updates that span every non-indexed operand axis. Scatter-add
accepts the standard addition reducer and preserves duplicate-index
accumulation. Scatter-set is accepted only when JAX declares
unique_indices=True; otherwise export fails because ONNX replacement order is
not a portable definition for duplicate indices. Fill/drop scatter-add masks
invalid updates to zero. Fill/drop scatter-set redirects each invalid update to
a distinct temporary padded slot and slices those slots away, preserving exact
replacement semantics without a dynamically sized Compress operation.

A while-loop condition must return one scalar boolean, and every loop carry
must preserve its shape and dtype. Static fori_loop traces through carry-only
Scan, while a dynamic bound traces through Loop/while. Scan with xs=None has a
trace-time fixed iteration count and is represented with an internal dummy
sequence that is not exposed as a model input.

The remaining fail-closed cases include multi-way switch, non-unique
scatter-set, custom scatter reducers, scatter batching/partial update windows,
dynamically shaped operands for slice/update/scatter, sort, FFT, random
operations, cumulative product, custom calls, and gather dimension-number
patterns outside the two layouts above.

Export should always be followed by numerical comparison against the JAX
function over representative and boundary inputs. ONNX validity also does not
imply that every execution provider will accelerate every node. With CPU
fallback disabled, CUDA executes the tested dynamic update (including a KV
cache layout), GatherElements/GatherND, all three scatter index modes, while,
static and dynamic fori_loop, and xs=None Scan graphs. A TensorRT-first, CUDA
fallback chain also executes these tests entirely on the GPU. TensorRT 10.16
does not itself implement ScatterND addition, so those nodes are assigned to
CUDA. It also rejects an If branch that captures a dynamic outer operand during
partitioning instead of falling back; select CUDA alone for that model.

JAX internals evolve, so the Python package declares the JAX version range it
is tested with. Unsupported changes should produce an export error, not a
silently altered model.
