# Kernel generation

`fortonnx_kernelgen` turns an ONNX graph that acts on every row independently into one CUDA kernel that evaluates
a whole row, and into an ONNX Runtime custom operator that runs this kernel. A program that runs its graphs through
FortONNX keeps doing so: it passes the custom operator library when it loads the model and needs no code for the
graph.

ONNX Runtime and TensorRT run such a graph operator by operator, and every operator reads and writes whole arrays in
device memory. For a graph of a few hundred small operators on about a million rows this is slow. The generated
kernel keeps the intermediate values of a row in registers and shared memory and touches device memory only for the
input row, the constants and the output row.

## What the generator accepts

- one input `[rows, ...]` (FP32, first axis dynamic, the others fixed) and one output `[rows, ...]` (FP32);
- every operator acts on each row independently. Everything is fixed at generation except the number of rows.

Supported operators (opset 18): Add, Sub, Mul, Div, Pow, Neg, Abs, Exp, Log, Tanh, Sqrt, Sin, Cos, Floor, Ceil,
Reciprocal, Relu, Sigmoid, Softplus, Max, Min, Sum, Equal, Greater, Less, GreaterOrEqual, LessOrEqual, IsInf, IsNaN,
Not, Or, And, Where, Cast (boolean to FP32), MatMul, ReduceSum, ReduceMax, ReduceMin, ReduceProd, and the operators
that only rearrange values: Identity, Constant, Slice, Split, Concat, Reshape, Squeeze, Unsqueeze, Expand, Transpose,
Gather (constant indices).

The generator stops with `UnsupportedGraph` (naming the node) for any other operator and for an operator that mixes
rows: a reduction, MatMul, Gather, Slice, Split, Concat or Reshape along the row axis. It stops with `KernelTooLarge`
when the values that one row must keep do not fit the shared memory of a block. Nothing switches to another way of
running the graph.

## Use

```sh
python -m fortonnx_kernelgen --output-dir generated net_a=a.onnx net_b=b.onnx
```

writes, for every `NAME=GRAPH`,

| file | content |
| --- | --- |
| `NAME.cu`, `NAME.h` | the kernel, and its functions as the library calls them (with `NAME_1.cu`, `NAME_2.cu` and their headers when the kernel is two kernels in a row) |
| `NAME.onnx` | the graph with the kernel as its one node (see below) |

and, for all the kernels of the call, the sources of one library: `kernel_ops.cc` (the custom operators of ONNX
Runtime), `kernel_direct.cc` (the direct entry, see below) and `build.sh`. Build the library (`ORT_INCLUDE` is the
directory of `onnxruntime_cxx_api.h`):

```sh
sh generated/build.sh $ORT_INCLUDE sm_90 libkernels.so      # kernels on the tensor cores: sm_90a
```

The script compiles the kernels with `nvcc --fmad=false`, which keeps the products and sums of the graph's
elementwise operators separate operations, as ONNX Runtime computes them (the product sums of MatMul use `fmaf`
explicitly). The library links the CUDA runtime statically and exports `RegisterCustomOps` and the four functions
of the direct entry only. Either way the ONNX file stays the interface of the model.

In Fortran:

```fortran
call runtime%load('a', 'generated/net_a.onnx', status, custom_op_library='generated/libkernels.so')
```

### The graph and the operator

`NAME.onnx` has one node, `fortonnx.kernel::NAME`:

| | |
| --- | --- |
| input 0 | the rows `[n, ...]` in the shape of the original input, FP32, on the device |
| input 1 | the kernel's constants, UINT32 `[words]`: a constant of the graph, read on the host when the session is created |
| attribute `signature` | the signature of the kernel the constants were written for |
| output | `[n, ...]` in the shape of the original output, FP32, on the device, computed on ONNX Runtime's stream |

The model also defines `fortonnx.kernel::NAME` as an ONNX function whose body is the original graph. A session that
registers the library runs the kernel; a session that does not runs the original operators. A library whose kernel has
another signature (generated from another graph, or with other settings) refuses the graph when the session is
created.

The constants hold the weights in the order the kernel reads them. Retrained weights of the same graph need a new
`NAME.onnx`; the library stays valid as long as the signature is the same. The signature changes when a weight that
the kernel holds as a literal changes (a constant that is equal for all iterations of a loop).

### The direct entry

A program with its own device arrays and CUDA stream (a Fortran model on OpenACC) can call a kernel without ONNX
Runtime. The library exports four C functions (`--direct-entry PREFIX` names them; the default is `direct_kernel`):

```c
int  direct_kernel_open(const char* graph, void** kernel, char* message, int message_length);
void direct_kernel_shape(void* kernel, int* row_columns, int* output_columns);
int  direct_kernel_forward(void* kernel, const float* rows, int n, float* out, void* stream);
void direct_kernel_close(void* kernel);
```

| function | |
| --- | --- |
| `_open` | reads `NAME.onnx` (the same file as above) and copies the kernel's constants to the current device; returns 0, or 1 with a null kernel and `"<graph>: <reason>"` in `message` (not a graph of one of the library's kernels, another signature, the file cannot be read) |
| `_shape` | values of one row and of its output (the shapes of the graph, flattened) |
| `_forward` | queues the outputs of `n` rows on the CUDA stream; `rows` and `out` are FP32 device arrays, row-major; nothing is allocated per call; returns 0 or the code of the failed launch |
| `_close` | frees the constants |

The library reads the node, its signature and its constants from the graph file itself, so both entries run the
same kernel on the same constants and give the same bits. The functions name no network, return a status and give
pointers back through arguments: a Fortran program calls them as named `bind(C)` functions.

## How the kernel is built

1. **Row-wise form** (`rowgraph.py`). Every tensor that depends on the row becomes an array of references to scalar
   elements. Slice, Concat, Reshape, Unsqueeze, Expand, Transpose, Split, Gather and a Where with a constant
   condition only rearrange references: they leave no code and no copy. Everything that does not depend on the row is
   evaluated with NumPy (FP32). What remains are *stages*: one scalar operator per element, a sum of products per
   element (MatMul), or a reduction per element.
2. **Schedule** (`schedule.py`). Stages that read each other only at matching positions share a loop: in iteration
   `m` each of them computes its elements of that iteration as local scalars. A group of small stages is straight-line
   code. An elementwise stage behind a Gather is moved onto the index space it selects from, so that it lines up with
   its producer (elements nobody selected are computed and never read). A stage that holds several copies of one
   index space, one copy after the other (a value and its tangents carried through the same layers as `[copies, M]`),
   is reordered to `[M, copies]`: the copies of an element then stand next to each other and share the loop over
   `M` with the stages of `M` elements, instead of passing whole stages through the rows' slabs. Copies of one sum
   share its factors: each is loaded once. A sum over the iterations of a looped stage
   (a product sum with constant factors or a ReduceSum, over all elements of every iteration or over the same
   ones) is accumulated inside that loop. Values needed outside their loop get a range of the row's slab in shared
   memory; ranges are reused when their last reader has run. Of a looped stage of which other code reads only some
   elements, only those are kept there (a graph that returns a large stage and uses a few of its elements again
   keeps a few words per row, not the stage).
3. **Kernel** (`emit.py`). One thread evaluates one row; the threads of a block run the same statements, so
   constants are the same for all of them. Constants that differ between the iterations of a loop are laid out per
   iteration in the order of use (a *record*); the block copies the records of the next iterations into shared memory
   and reads them there. Reading them from device memory instead stalls every row of the block at each new address
   (`--stage-bytes 0` keeps that form for comparison).

By default no tensor-core operation is used: every sum of products is a chain of FP32 `fmaf` in the order of the
graph's terms, and every function is the C library's. Such a kernel agrees with ONNX Runtime to rounding. (One difference seen: the Sigmoid of a value of -30 or below is 0 in ONNX Runtime on the CPU; the kernel
returns the small value of the function, so that its logarithm is a number.)

## Kernels at the level of hand-written ones

`--approximate-functions --tensor-cores` writes the kernel the way a hand-tuned kernel for Hopper GPUs is written.
The forms are those a hand-written kernel uses. The results
then differ from the graph's by rounding-size errors: compare the kernel with ONNX Runtime and with an FP64
evaluation of the graph before use. The kernel assumes finite inputs: where the graph guards against infinite or
NaN values, or where an operator overflows or leaves its domain, the results differ.

**Functions and forms** (`--approximate-functions`; `cforms.py`, `idioms.py`, `ranges.py`):

| in the graph | in the kernel |
| --- | --- |
| `exp`, `log`, a quotient | the base-2 instructions `ex2`, `lg2` and the reciprocal `rcp` |
| `tanh` | branch-free: a polynomial below 0.625, `1 - 2 / (1 + 2^(2x log2 e))` above (relative error 1e-7) |
| the exporter's `logaddexp(x, c)`, softplus | `max + 2 atanh(e / (2 + e))`, one exponential |
| the exporter's `logsumexp` (about ten operators with guards) | `m + lg2(sum(ex2(t - m)))` |
| `exp(c tanh(k x))` with `c` at most 16 in magnitude | `ex2(c' - 2c' rcp(1 + ex2(k' x)))` |
| products with constants in a row; the factors `log2 e`, `ln 2` of the base-2 instructions | folded into one constant, into the constants around a `logsumexp`, or into the weights and biases of the product sums |
| a floor that a bounded value never reaches, the terms of a masked element (`max(-inf, x)`, `exp(-inf)`) | left out: the ranges of the values show that they have no effect (`Floor` and `Ceil` count one integer wider on both sides: FP32 rounding of their argument may cross an integer) |
| a loop whose iterations have different numbers of valid elements (padded slots) | one body per pattern, chosen by a switch (`--member-patterns`) |

**Product sums on the tensor cores** (`--tensor-cores`, `wgmma` of sm_90a; `tensor.py`). Every FP32 value is split
into a TF32 high part and the exact remainder, and a product is the sum of three (low x high, high x low,
high x high) accumulated in FP32. (The high part is rounded: for a value of the row within 2^-12 of the largest
FP32 value it is infinite and the results are not finite. Weights in that range are split without this.) A warpgroup of 128 threads owns 128 rows as two tiles of 64. Three layouts:

| the graph has | the kernel does |
| --- | --- |
| a loop whose body starts with a chain of product sums (layer after layer, per iteration) | one instruction width per layer, the accumulators of a layer are the A operand of the next; the two tiles alternate; the rest of the body runs one row per thread |
| a loop whose product sums all read values of the row | four iterations per instruction: every thread holds all elements of one iteration for two rows and runs the rest of the body on them; the A operand stays in registers (high parts) and in shared memory (low parts) for the whole loop |
| product sums outside a loop with 16 to 128 terms and at most 128 elements, each on all its terms (dense layers; any number of them one after the other, with elementwise stages between) | a chain: the first sum takes its terms from the input rows or the rows' slabs as A operand (a first sum of at most 8 row values is computed directly as A operand instead), the accumulators of a sum are the A operand of the next, the weights of all sums pass through shared memory in slabs of a few k-steps, each slab is accumulated from zero and added in FP32, the last sum goes to the rows' slabs |

**Two kernels in a row.** A loop with its iterations by thread reads the row only through the row values of its
product sums. Everything ahead of it then becomes a first kernel that writes these values (in whole k-steps) into a
device buffer of the kernel object, and the loop a second kernel that reads them as its rows. Each kernel has its
own number of threads: the first one may use many registers per thread (wide product sums), the second one runs
more warps (`--loop-warpgroups`). The buffer is allocated when a call has more rows than any before it, never per
call. `NAME.cu` then includes `NAME_1.cu` and `NAME_2.cu`; the graph, the operator and the direct entry are those of
one kernel. `--one-kernel` keeps everything in one kernel.

**Stores.** A thread that stores its own row's outputs one by one makes the 32 threads of a warp write to 32 rows
per instruction. On the tensor cores the warp stores them together instead, consecutive lanes consecutive elements
of a row: the outputs of an iteration that are consecutive in the output row, and the outputs of straight-line code
(`--separate-stores` turns this off). This needs more registers per thread.

**Chains in general.** An elementwise stage between two sums of a chain may read other values of the row that
exist before the chain (a skip connection `tanh(y2) + x1`): the threads read them from the rows' slabs. Where the
value between two sums has another reader, the chain ends and the next sum starts a chain of its own from the
rows' slabs; a graph may have any number of chains. A product sum that is the body of a loop (several elements per
iteration) joins a chain only with 64 terms or more: below that the loop's own layouts keep its terms in registers.
**Sums in blocks.** The registers of a thread hold the A operand of 128 terms and the accumulators of 128
elements. A sum with more terms (up to 1024) is the first of its chain: the threads build its A operand for 64
terms at a time, from the row values where they are, and again for every block of elements (with 128 terms per
A operand such a kernel has no register left; row values that are read again are read as `volatile`, or the
compiler keeps them all on the stack instead). A sum of few terms whose values are the terms of such a sum (the
first layer of a wide network, `tanh(W1 x + b1)` with a few inputs) is a chain of its own on the tensor cores: its
values are computed once per row and kept in the rows' slabs, where the wide sum reads them. (With `--slab shared`
that layer is not kept: the sum is then the start of the wide sum's chain and is computed again for every block
of elements, four values at a time.) A sum with more elements is the last of its chain: its elements are computed
128 at a time. A chain also computes the elementwise stages behind
its last sum, up to the value that other code reads, and the next chain reads that value where it is: of a layer
`tanh(W x + b)` only the result is kept per row. The threads that hold a row in a tile store the result from their
accumulators: the elements that other code reads in the rows' slabs, the elements that are outputs of the kernel
in the output rows (they do not pass through the slabs). Such chains keep whole layers in the rows' slabs (a layer of 256
elements: 1 kB per row, one warpgroup per block). Where the shared memory does not hold them, the kernel is planned
again without sums in blocks: these then stay sums of `fmaf`.

A kernel on the tensor cores has 128 rows per warpgroup (two tiles of 64, one after the other) and as many
warpgroups, at most `--warpgroups`, as the shared memory holds with what a row keeps there. Where one warpgroup
does not fit, the kernel is written with one tile per warpgroup (64 rows; `--tiles 1` asks for that), and where that
does not fit either, without the stores that the warp makes together (they take words of the rows' slabs).

**Slabs in device memory.** A row that keeps much between its loops leaves a block few rows in shared memory, or
none (`KernelTooLarge`). The stages that keep much (64 words or more per row, kept from one loop to a later one)
can instead be in device memory of the kernel object (`--slab device`); the small values that the loops read and
write all the time stay in shared memory (a network that keeps 76 words per row: 2.72 ms with them in
shared memory, 6.19 ms with them in device memory). Any
graph then fits, with full blocks (256 rows, or two tiles for each warpgroup), and the launches run in batches of
rows so that the slabs stay below 256 MiB. A batch of more blocks than the device has processors is a whole number
of times that many: a block of such a kernel may take a processor for itself, and the blocks left over at the end
of a launch would run on a few processors while the others wait (measured, hidden layers of 896 values: 18.0 ms
with launches of 146 blocks, 10.9 ms with launches of 132). A kernel object keeps one such buffer per CUDA stream it is used on; a
buffer grows when a call has more rows and is never allocated per call. Every thread then reads and writes its own
lines of device memory, which is slower per row than shared memory, and faster in sum only where the shared
memory holds few rows. Measured (GH200, one GPU): code of one row per thread is 1.6 to 4.5 times slower with the
slabs in device memory when the shared memory holds 96 to 128 of 256 rows, and 3 times faster when it holds 27;
chains on the tensor cores are 1.2 to 2 times faster with full blocks than with 64 of 256 rows. The default
(`--slab auto`) follows these numbers: shared memory, unless the kernel does not fit there or would have less than
a quarter of its rows per block (less than half for a kernel whose loops are all on the tensor cores).
`--slab shared` never uses device memory: a chain in blocks that does not fit is then planned as sums of `fmaf`.

**Parts of the output row.** A graph that carries a value and its tangents through one network (a tangent-linear
graph) keeps them as stages of copies, and its loops run over all copies at once: no layout of the tensor cores
takes such a loop, and one thread computes everything of its row. With `--tensor-cores` the outputs of a graph are
therefore sorted into parts by the products of the product sums that they need: an output belongs to the first
part that computes at least 90 % of its products anyway, and starts a part otherwise. For a tangent-linear graph
these are the tangents, with the value's outputs in the first part. Each part is written as a graph of its own
(the elements that its outputs need; the value and the tangent of a stage of copies as two stages; padded slots
stay where they are computed from elements of the part; terms that are constants in the part go into the bias),
and it is the network of the value and of one tangent, with the regular loops of the value's network. The kernel
is then one kernel per part, side by side: all read the same rows and store their outputs into the same output
rows, the first on the caller's stream and the others at the same time on streams of the kernel object, which
start when the caller's stream reaches the call and which the caller's stream waits for (events). `--output-parts
auto` (the default) takes the parts where they bring the product sums of loops onto the tensor cores: a product
in code of one row per thread counts as eight on the tensor cores, and the parts together must cost less than
two thirds of the whole graph (`on` always takes them, `off` never). Measured on the tangent-linear graph of r
(6 inputs, 1244 outputs; GH200): six kernels of 3.6 to 4.7 thousand lines instead of one of 139 thousand, build
20 s instead of 215 s; 16384 rows in 2.5 ms (one kernel 5.6 ms, ONNX Runtime CUDA 7.3 ms), 1024 rows in 0.72 ms
(3.0 ms, 1.3 ms). The value's network is computed in every part.

**Weight slots of a chain.** The weights of a chain pass through two slots of shared memory, slab after slab; the
products of a slab are summed on the tensor cores (those with a low part first, then those of the high parts)
and then added to the sum in FP32, and the warps wait once per slab. A slot holds 4096, 8192 or 16384 floats: the
kernel is written with the largest with which the block keeps its rows. Measured (GH200, 262144 rows, 512 x 512
from the input rows): 4.02 ms with 4096 floats, 3.04 ms with 8192, 2.41 ms with 16384; the largest difference to
the FP32 sums of ONNX Runtime on the CPU stays below 1e-6 of the largest value.

The records of a loop reach shared memory through a ring of two slots: asynchronous copies that the block waits for
once per iteration, or, where warps need not run in step, bulk copies that complete on a barrier in shared memory.

**Warps that do not wait for each other.** In a loop with its iterations by thread and in a chain of wide product
sums, nothing inside the loop over the two tiles makes the whole block wait: a warp waits for the slot that holds
its weights (the last warp through with a slot fills it again, and the slots keep their order from one tile to the
next), the warps of a warpgroup wait for each other where they have written the low parts of their A operand, and
the records of a loop that runs inside the loop (a loop tied to it: one that reads its results in rising order) are
copied before it. With tied loops, an iteration of the loop over the chunks starts the product sums of the next
chunk, runs the tied loops of this one, and then waits for the product sums: they run on the tensor cores
meanwhile, and their accumulators are written and read inside one iteration (registers of a product sum in flight
that stay in use around the loop make the assembler insert a wait where the product sums start; it reports this
as `wgmma.wait_group is injected`). In the last chunk a thread without an iteration computes the last one again
and stores nothing: the stores have the condition, not the computation.

## Settings

| option | meaning | default |
| --- | --- | --- |
| `--threads-per-block [NAME=]N` | rows per block | as many as the shared memory holds (at most 256, which any kernel can launch); 96 for a kernel without per-row shared memory |
| `--dot-lanes N` | product sums of one stage that advance together | 8 |
| `--stage-bytes N` | shared memory for the constants of a loop; 0 reads them from device memory | 16384 |
| `--reciprocal-division` | divide by constants with their FP32 reciprocals; changes the rounding of each quotient by at most one unit in the last place | off |
| `--approximate-functions` | functions and forms of the table above | off |
| `--tensor-cores` | product sums on the tensor cores (needs sm_90a) | off |
| `--warpgroups [NAME=]N` | most warpgroups (128 rows) per block of a kernel on the tensor cores (it has as many as the shared memory holds); the registers of a thread limit it (256 threads: 255 registers, 384 threads: 170) | 2 |
| `--loop-warpgroups [NAME=]N` | warpgroups per block of the second kernel of two in a row | 3 |
| `--tiles [NAME=]N`, `--loop-tiles [NAME=]N` | most tiles of 64 rows per warpgroup (of the second kernel of two in a row): 2, or 1 for 64 rows per warpgroup | 2 |
| `--slab auto\|shared\|device` | where the rows keep their values between loops: shared memory of the block, device memory of the kernel object, or chosen by the rows per block (above) | auto |
| `--output-parts auto\|on\|off` | with `--tensor-cores`: one kernel per part of the output row whose outputs share their product sums (above) | auto |
| `--one-kernel` | no second kernel for a loop with its iterations by thread | off |
| `--member-patterns N` | most bodies of one loop, one per pattern of its iterations; 1 writes one body | 16 |
| `--separate-stores` | on the tensor cores, every thread stores its own row's outputs; by default the threads of a warp store consecutive output elements of their rows together (far fewer memory sectors per store, more registers per thread) | off |

## Tests

`make test-python` runs `tests/test_kernelgen.py` (no GPU). `tests/test_kernelgen_cuda.py` builds a kernel and
compares the custom operator with the graph on a GPU; it needs `nvcc`, `g++`, `ONNXRUNTIME_INCLUDE` and ONNX
Runtime with the CUDA provider.
