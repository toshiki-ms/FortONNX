"""Product sums on the tensor cores of Hopper GPUs (wgmma, sm_90a).

The form is that of the hand-written kernels this generator has to match: inside a loop of the
schedule, the product sums at the head of the body and the elementwise stages between them are computed for 64 rows
at a time in *fragments*; the rest of the body runs one row per thread as everywhere else.

Layout. A warpgroup (4 warps, 128 threads) owns 128 rows as two tiles of 64. In each tile the thread
``(w, lane = 4 g + t)`` holds the rows ``16 w + g`` and ``16 w + g + 8``:

  accumulators of an N-wide product sum   d[4 nt + i] = row g + 8 (i >> 1), column 8 nt + 2 t + (i & 1)
  A operand of k-step ks                  a[0], a[1] = rows g, g + 8 at k = 8 ks + t; a[2], a[3] = at k = 8 ks + t + 4
  B operand (weights), shared memory      per k-step [k half 2][n block N/8][8 n][4 k] (K-major core matrices)

An elementwise stage acts on the accumulators where they are. The next product sum takes them as its A operand
directly: its k index ``8 nt + t`` is column ``8 nt + 2 t`` and ``8 nt + t + 4`` is column ``8 nt + 2 t + 1``, so its
weights are stored in that order and no value moves between threads. A product sum over values of the rows (not
fragments) reads them from the rows' slabs, where the kernel has put them next to each other.

Iterations by thread. When every product sum of the phase reads values of the row, four iterations of the loop
share one instruction: column ``c`` of n-block ``nt`` is element
``2 nt + c % 2`` of iteration ``4 q + c / 2`` of chunk ``q``. The thread ``t`` of a quad then holds every element of
its iteration ``4 q + t`` for its two rows, and the rest of the body runs on these registers, two rows per thread,
with no value moving between threads. The A operand is the same for all chunks: its high parts stay in registers,
its low parts are an image in shared memory that the instruction reads directly. The warpgroup goes through all
chunks for its first tile, then for its second: registers and image hold the A operand of one tile.

Wide product sums outside a loop: a chain ``y1 = W1 x0 + b1``, ``x1 = f(y1)``,
``y2 = W2 x1 + b2``, ... of any length, each sum with up to 128 terms and 128 elements (more: in blocks, below). ``x0`` is either values of
the row (the threads read them from the rows' slabs or from the input rows) or ``f(narrow sum of row values)``,
which the threads compute directly as A operand (each thread the terms of its register slots, for its two rows).
Every wide sum is one instruction width (m64nNk8), its accumulators become the next A operand, and the last sum
goes to the rows' slabs. An elementwise stage between two sums may also read values of the row that exist before
the chain (a skip connection): the thread reads them from the rows' slabs, element by element. Where a value
between two sums has another reader, the chain ends there and the next sum starts a chain of its own from the
rows' slabs: a graph may have several chains. The
weights are too large for the shared memory at once: the slabs (a few k-steps) of all sums of a tile pass, in
order, through one ring of two slots. The products of a slab are accumulated from zero on the tensor cores and
added to the sum in FP32: a sum of a hundred terms keeps its accuracy this way (the accumulation of the tensor
cores does not round to nearest).

Blocks. The registers of a thread hold the A operand of 128 terms and the accumulators of 128 elements. A sum
with more terms is the first of its chain: the threads build its A operand for 128 terms at a time (from the row
values, which stay where they are) and add the products of these terms to the accumulators. A sum with more
elements is the last of its chain: its elements are computed 128 at a time and go to the rows' slabs, after the
elementwise stages that follow the sum (a chain computes those up to the value that other code reads). A layer of
256 by 256 is both: a chain of one sum in four blocks.

Precision. The tensor cores read TF32 operands (FP32 with 13 mantissa bits dropped). Every FP32 value is split into
a TF32 high part (rounded to nearest) and the exact remainder, and a product is the sum of three: low x high,
high x low, high x high, accumulated in FP32 from the bias (3xTF32). Rounding the high part of a value within
2^-12 of the largest FP32 value gives infinity. A weight in that range gets the largest TF32 value as its high
part (tf32_split). A value of the row in that range does not: the kernel splits the values as the hand-written
kernels do, without a test (a test cost 20 % of the time of a kernel that splits in its loops), and its results
are not finite there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cforms import APPROXIMATE_FUNCTIONS, float_literal
from .rowgraph import MAP_OPERATORS, RowGraph, Stage, make_refs, ref_offset, ref_stage

MAX_COLUMNS = 48   # widest product sum of one instruction here (24 accumulators per thread)
CHUNK = 4          # iterations per instruction when the threads of a quad each hold one iteration
THREAD_COLUMNS = 96   # widest product sum of such an instruction (48 accumulators per thread)
CHAIN_TERMS = 16      # a product sum outside a loop with at least this many terms goes to the tensor cores
LOOP_TERMS = 64       # and one that is the body of a loop (several elements per iteration) with at least this many:
                      # with fewer, the loop's own layouts keep its terms in registers
CHAIN_COLUMNS = 128   # widest product sum of an instruction of a chain (64 accumulators and 64 A registers per thread)
# Terms per A operand of a sum with more terms than CHAIN_COLUMNS. Its A operand is built again for every block of
# elements, inside the loops over the blocks: with 128 terms (64 A registers beside the accumulators and the
# products of a slab) the kernel has no register left and keeps values on the stack (measured: 256 x 256 from the
# input rows 1.10 ms with 128 terms per A operand, 0.72 ms with 64).
CHAIN_BLOCK_TERMS = 64
CHAIN_WIDEST = 1024   # most terms and elements of a product sum of a chain: beyond CHAIN_COLUMNS it runs in blocks
# Weights of a chain that are in shared memory at a time (floats per slot of the ring), from small to large. The
# products of one slab are summed on the tensor cores and then added in FP32, so larger slabs are fewer waits
# (measured, 512 x 512 from the input rows: 4.02 ms with 4096, 3.04 with 8192, 2.41 with 16384; largest
# difference to the sums in FP32 5.5e-7, 6.1e-7 and 7.7e-7 of the largest value). The emitter takes the largest
# with which the block keeps its rows (emit.emit_kernel).
SLOT_FLOATS = (4096, 8192, 16384)
MIN_INNER = 8      # narrowest product sum worth a tile: fewer elements per iteration stay one row per thread


@dataclass
class Dense:
    """A product-sum stage computed by wgmma.

    columns  N: elements per iteration, padded to a multiple of 8
    steps    k-steps of 8 terms
    source   stage whose fragment is the A operand, or None: the vector ``vector`` of row values
    weights  [members, 8 steps, N] FP32 in the order of the A operand (zero where padded)
    bias     [members, N] FP32, or None
    """

    index: int
    columns: int
    steps: int
    source: int | None
    vector: int | None
    weights: np.ndarray
    bias: np.ndarray | None


@dataclass
class FragmentPhase:
    """The head of a loop body that runs in fragments.

    stages    in order: product sums and the elementwise stages on their accumulators
    inner     stage -> elements per iteration (the fragment has pad8 of them)
    vectors   row values that product sums read: each [terms] references, the same in every iteration
    exported  fragment stages that the rest of the body reads: written to the rows' slabs
    tail      the stages of the group after the phase (one row per thread)
    by_thread four iterations per instruction, every thread holds the elements of one iteration (module text):
              the tail then runs on the accumulators, and nothing is exported
    direct    by_thread only: per vector and k-step, (stage, first element) when the eight values stand next to
              each other in the rows' slabs, else None: these are copied to ``vector_offset`` (storage assignment)
    vector_offset, scratch_offset   slab words, set by the storage assignment
    """

    stages: list[int] = field(default_factory=list)
    dense: dict[int, Dense] = field(default_factory=dict)
    inner: dict[int, int] = field(default_factory=dict)
    vectors: list[np.ndarray] = field(default_factory=list)
    exported: list[int] = field(default_factory=list)
    tail: list[int] = field(default_factory=list)
    by_thread: bool = False
    direct: list[list[tuple[int, int] | None]] = field(default_factory=list)
    vector_offset: list[int] = field(default_factory=list)
    scratch_offset: dict[int, int] = field(default_factory=dict)

    def columns(self, index: int) -> int:
        return -(-self.inner[index] // 8) * 8


def _dense(graph: RowGraph, stage: Stage, members: int, held: dict[int, int], phase: FragmentPhase) -> Dense | None:
    inner = stage.size // members
    columns = -(-inner // 8) * 8
    if stage.weights is None or columns > MAX_COLUMNS or inner < MIN_INNER:
        return None
    terms = stage.terms
    refs = stage.operands[0].reshape(members, inner, terms)
    weights = stage.weights.reshape(members, inner, terms)
    if graph.is_const()[ref_stage(refs)].any():
        return None
    sources = np.unique(ref_stage(refs))
    if not any(int(s) in held for s in sources):
        # values of the row, the same vector for every element and iteration
        if not (refs == refs[0, 0]).all():
            return None
        key = refs[0, 0].tobytes()
        known = [v.tobytes() for v in phase.vectors]
        if key not in known:
            phase.vectors.append(np.ascontiguousarray(refs[0, 0]))
            known.append(key)
        steps = -(-terms // 8)
        ordered = np.zeros((members, 8 * steps, columns), dtype=np.float32)
        ordered[:, :terms, :inner] = weights.transpose(0, 2, 1)
        source, vector = None, known.index(key)
    else:
        if len(sources) != 1:
            return None
        source = int(sources[0])
        width = held[source]
        offset = ref_offset(refs)
        column = offset % width
        same_iteration = (offset // width == np.arange(members)[:, None, None]).all()
        if not same_iteration or not (column == column[0, 0]).all() or len(np.unique(column[0, 0])) != terms:
            return None
        steps = -(-width // 8) * 8 // 8
        position = 8 * (column[0, 0] // 8) + (column[0, 0] % 8) // 2 + 4 * (column[0, 0] % 2)   # k index of a column
        ordered = np.zeros((members, 8 * steps, columns), dtype=np.float32)
        ordered[:, position, :inner] = weights.transpose(0, 2, 1)
        vector = None
    bias = None
    if stage.bias is not None:
        bias = np.zeros((members, columns), dtype=np.float32)
        bias[:, :inner] = stage.bias.reshape(members, inner)
    return Dense(stage.index, columns, steps, source, vector, ordered, bias)


def _acts_on_fragments(graph: RowGraph, stage: Stage, members: int, held: dict[int, int]) -> bool:
    inner = stage.size // members
    is_const = graph.is_const()
    variable = False
    for refs in stage.operands:
        const = is_const[ref_stage(refs)]
        if const.all():
            continue
        sources = np.unique(ref_stage(refs))
        if const.any() or len(sources) != 1 or held.get(int(sources[0])) != inner:
            return False
        if not np.array_equal(ref_offset(refs), np.arange(stage.size)):
            return False
        variable = True
    return variable and not stage.is_bool


def find_fragment_phase(graph: RowGraph, group, all_stages_of_other_groups: list[int]) -> FragmentPhase | None:
    """The fragment phase of a loop group, or None when its head has no product sum that fits the tensor cores."""
    members = group.members
    if members == 1:
        return None
    phase = FragmentPhase()
    held: dict[int, int] = {}
    for index in group.stages:
        stage = graph.stages[index]
        if stage.kind == "dot" and len(stage.operands) == 1:
            dense = _dense(graph, stage, members, held, phase)
            if dense is None:
                break
            phase.dense[index] = dense
        elif not (stage.kind == "map" and _acts_on_fragments(graph, stage, members, held)):
            break
        held[index] = stage.size // members
        phase.stages.append(index)
    # Elementwise stages after the last product sum stay one row per thread: there they run on the elements that are
    # read, not on the padded columns, and the stored values are the accumulators either way.
    while phase.stages and phase.stages[-1] not in phase.dense:
        del held[phase.stages.pop()]
    if not phase.dense:
        return None
    phase.inner = held
    phase.tail = [index for index in group.stages if index not in held]

    def reads(readers: list[int]) -> set[int]:
        found = set()
        for index in readers:
            for refs in graph.stages[index].operands:
                found.update(int(s) for s in np.unique(ref_stage(refs)) if int(s) in held)
        return found

    outside = reads(all_stages_of_other_groups) | {int(s) for s in np.unique(ref_stage(graph.output_refs)) if int(s) in held}
    if outside:
        return None   # fragments are stored for the rest of the same loop body only
    phase.exported = sorted(reads(phase.tail + group.accumulated))
    return phase


def find_thread_phase(graph: RowGraph, group, all_stages_of_other_groups: list[int]) -> FragmentPhase | None:
    """The fragment phase of a loop group with its iterations by thread (module text), or None. It needs: every
    product sum with constant factors reads the same row values in all iterations (they may stand anywhere in the
    body: all are issued first); the rest of the body reads nothing of the row but what the group computes (its
    statements run for two rows, neither of them the thread's own); no sum is accumulated over the loop; nothing
    outside the loop body reads the product sums."""
    members = group.members
    if members == 1 or group.accumulated:
        return None
    phase = FragmentPhase(by_thread=True)
    for index in group.stages:
        stage = graph.stages[index]
        if stage.kind != "dot" or stage.weights is None:
            continue
        dense = _dense(graph, stage, members, {}, phase)
        if dense is None or 8 * -(-(stage.size // members) // 2) > THREAD_COLUMNS:
            return None
        phase.dense[index] = dense
        phase.inner[index] = stage.size // members
        phase.stages.append(index)
    if not phase.dense:
        return None
    phase.tail = [index for index in group.stages if index not in phase.dense]
    is_const = graph.is_const()
    local = set(group.stages)
    for index in phase.tail:
        for refs in graph.stages[index].operands:
            if not all(is_const[source] or int(source) in local for source in np.unique(ref_stage(refs))):
                return None
    readers = [graph.stages[index].operands for index in all_stages_of_other_groups] + [[graph.output_refs]]
    if any(int(source) in phase.dense for operands in readers for refs in operands for source in np.unique(ref_stage(refs))):
        return None
    return phase


def tf32_split(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(high, low) TF32 parts of FP32 values: high rounded to nearest, low the rounded remainder. A value that
    would round up to infinity has the largest TF32 value as its high part."""
    def nearest(v: np.ndarray) -> np.ndarray:
        bits = np.ascontiguousarray(v, dtype=np.float32).view(np.uint32)
        rounded = (bits + np.uint32(0x1000)) & np.uint32(0xFFFFE000)
        infinite = (rounded & np.uint32(0x7F800000)) == np.uint32(0x7F800000)
        return np.where(infinite, (rounded & np.uint32(0x80000000)) | np.uint32(0x7F7FE000), rounded).view(np.float32)

    high = nearest(values)
    return high, nearest((values - high).astype(np.float32))


def core_matrices(ordered: np.ndarray) -> np.ndarray:
    """[members, 8 steps, N] weights -> [members, steps * 8 N] in the order wgmma reads B from shared memory."""
    members, terms, columns = ordered.shape
    blocks = ordered.reshape(members, terms // 8, 2, 4, columns // 8, 8)      # [m, ks, k half, k, n block, n]
    return np.ascontiguousarray(blocks.transpose(0, 1, 2, 4, 5, 3)).reshape(members, -1)


HELPERS = """
// Tensor-core product sums (see fortonnx_kernelgen/tensor.py for the layout).
// FP32 -> TF32 high part (rounded to nearest) and the exact remainder, as bit patterns for the A operand.
__device__ __forceinline__ void split(float v, uint32_t& high, uint32_t& low) {
  high = (__float_as_uint(v) + 0x1000u) & 0xFFFFE000u;
  low = __float_as_uint(v - __uint_as_float(high));
}
// Registers of a product sum in flight must not be touched: these mark them as used here.
template <int K> __device__ __forceinline__ void keep(float (&d)[K]) {
#pragma unroll
  for (int i = 0; i < K; ++i) asm volatile("" : "+f"(d[i]) :: "memory");
}
template <int K> __device__ __forceinline__ void keep(uint32_t (&a)[K][4]) {
#pragma unroll
  for (int i = 0; i < 4 * K; ++i) asm volatile("" : "+r"(a[i >> 2][i & 3]) :: "memory");
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\\n" :: "n"(N) : "memory");
}
// Asynchronous copies from device memory to shared memory, 16 bytes each: the block issues the copies of a record,
// goes on computing, and waits for them before the record is read.
__device__ __forceinline__ void copy_async(float4* target, const float4* source) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\\n"
               :: "r"((unsigned)__cvta_generic_to_shared(target)), "l"(source) : "memory");
}
__device__ __forceinline__ void commit_async() { asm volatile("cp.async.commit_group;\\n" ::: "memory"); }
__device__ __forceinline__ void wait_async() { asm volatile("cp.async.wait_group 0;\\n" ::: "memory"); }
// Bulk copies from device memory to shared memory that complete on a barrier in shared memory (mbarrier): one
// thread issues the copy, every warp waits on the barrier before it reads, and no thread waits for another.
__device__ __forceinline__ void barrier_init(uint64_t* barrier) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\\n" :: "r"((unsigned)__cvta_generic_to_shared(barrier)) : "memory");
}
__device__ __forceinline__ void barrier_init_done() { asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory"); }
__device__ __forceinline__ void barrier_wait(uint64_t* barrier, int parity) {
  asm volatile("{\\n.reg .pred P;\\nWAIT_%=:\\nmbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\\n@!P bra WAIT_%=;\\n}\\n"
               :: "r"((unsigned)__cvta_generic_to_shared(barrier)), "r"(parity) : "memory");
}
__device__ __forceinline__ void copy_bulk(void* target, const void* source, int bytes, uint64_t* barrier) {
  const unsigned b = (unsigned)__cvta_generic_to_shared(barrier);
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" :: "r"(b), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\\n"
               :: "r"((unsigned)__cvta_generic_to_shared(target)), "l"(source), "r"(bytes), "r"(b) : "memory");
}
// Barrier of one warpgroup (128 threads): its image of an A operand is complete.
__device__ __forceinline__ void warpgroup_sync(int warpgroup) {
  asm volatile("bar.sync %0, 128;\\n" :: "r"(1 + warpgroup) : "memory");
}
// Shared-memory stores become visible to the tensor cores.
__device__ __forceinline__ void fence_async() { asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory"); }
// Descriptor of a B operand in shared memory: core matrices of 8 n x 4 k floats, `leading_bytes` between the two
// k halves of a k-step, 128 bytes between n blocks.
__device__ __forceinline__ uint64_t wg_descriptor(const void* address, int leading_bytes) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(address));
  return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)(leading_bytes >> 4) << 16) | ((uint64_t)(128 >> 4) << 32);
}
"""


def mma_helper(columns: int) -> str:
    """Device functions of one wgmma instruction m64n{columns}k8: d += A x B (descriptor), with A from registers
    and with A from shared memory (descriptor)."""
    count = columns // 2
    outputs = ",".join(f"%{i}" for i in range(count))
    inputs = ",".join(f"%{count + i}" for i in range(4))
    accumulators = "      : " + ", ".join(f'"+f"(d[{i}])' for i in range(count))
    instruction = f"wgmma.mma_async.sync.aligned.m64n{columns}k8.f32.tf32.tf32 {{{outputs}}}"
    lines = [f"__device__ __forceinline__ void wg_mma_{columns}(float (&d)[{count}], const uint32_t (&a)[4], uint64_t b) {{",
             f'  asm volatile("{instruction}, {{{inputs}}}, %{count + 4}, 1, 1, 1;\\n"',
             accumulators,
             '      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b));',
             "}",
             f"__device__ __forceinline__ void wg_mma_shared_{columns}(float (&d)[{count}], uint64_t a, uint64_t b) {{",
             f'  asm volatile("{instruction}, %{count}, %{count + 1}, 1, 1, 1;\\n"',
             accumulators,
             '      : "l"(a), "l"(b));',
             "}"]
    return "\n".join(lines)


def vector_stores(body, phase: FragmentPhase) -> list[str]:
    """Before the loop, one row per thread: the row values that the product sums read go next to each other into
    the row's slab, where the threads that hold the row in a tile find them."""
    lines = []
    for vector, offset in zip(phase.vectors, phase.vector_offset):
        lines.append("// values of the row that the tensor-core product sums read")
        for k, ref in enumerate(vector):
            lines.append(f"s[{offset + k}] = {body.outside_value(ref)};")
    return lines + ["__syncwarp();"]


class FragmentWriter:
    """Statements of a fragment phase. ``body`` is the emitter's body of the group: it owns the record of the
    iteration (weights and constants) and the options."""

    def __init__(self, body, phase: FragmentPhase):
        self.body = body
        self.phase = phase
        self.graph: RowGraph = body.graph
        self.members = body.members
        self.options = body.emitter.options
        self.is_const = body.emitter.is_const
        self.blocks: dict[int, tuple[int, int]] = {}     # dense stage -> record words of its high and low weights
        self.bias: dict[int, int] = {}                   # dense stage -> record words of its bias
        # waves: product sums issued together, then the elementwise stages on their accumulators
        self.waves: list[tuple[list[Dense], list[Stage]]] = []
        for index in phase.stages:
            if index in phase.dense:
                issued = [dense.index for dense in self.waves[-1][0]] if self.waves else []
                if not self.waves or self.waves[-1][1] or phase.dense[index].source in issued:
                    self.waves.append(([], []))
                self.waves[-1][0].append(phase.dense[index])
            else:
                self.waves[-1][1].append(self.graph.stages[index])

    # ---- names -----------------------------------------------------------------------------------------------

    def value(self, index: int, tile: int, q: int) -> str:
        return f"d{index}_{tile}[{q}]" if index in self.phase.dense else f"f{index}_{tile}_{q}"

    def column_constant(self, values: np.ndarray):
        """Text of a constant with one value per iteration and column ([members, n]), as a function of (nt, e)."""
        columns = -(-values.shape[1] // 8) * 8
        padded = np.repeat(values[:, :1], columns, axis=1).astype(np.float32)
        padded[:, :values.shape[1]] = values
        if (padded.view(np.uint32) == padded.view(np.uint32)[0, 0]).all():
            text = float_literal(padded[0, 0])
            return lambda nt, e: text
        offset = self.body.float_block(padded, 4)
        return lambda nt, e: f"{self.body.rec}[{offset + 8 * nt + e} + 2 * t]"

    # ---- pieces of one tile ----------------------------------------------------------------------------------

    def before_loop(self) -> list[str]:
        return vector_stores(self.body, self.phase)

    def load_inputs(self, dense: Dense, tile: int) -> list[str]:
        name = f"{dense.index}_{tile}"
        lines = [f"uint32_t ah{name}[{dense.steps}][4], al{name}[{dense.steps}][4];"]
        if dense.source is not None:
            return lines
        terms = len(self.phase.vectors[dense.vector])
        rows, stride, offset = self.body.emitter.slab_rows(self.phase.vector_offset[dense.vector])
        lines.append(f"const float* const x{name}a = {rows} + (lr0 + {64 * tile} + g) * {stride} + {offset};")
        lines.append(f"const float* const x{name}b = x{name}a + 8 * {stride};")
        for ks in range(dense.steps):
            for half in (0, 1):
                first = 8 * ks + 4 * half
                real = min(max(terms - first, 0), 4)
                for row, slot in (("a", 2 * half), ("b", 2 * half + 1)):
                    target = f"ah{name}[{ks}][{slot}], al{name}[{ks}][{slot}]"
                    if real == 0:
                        lines.append(f"ah{name}[{ks}][{slot}] = al{name}[{ks}][{slot}] = 0u;")
                    elif real == 4:
                        lines.append(f"split(x{name}{row}[{first} + t], {target});")
                    else:
                        lines.append(f"split(t < {real} ? x{name}{row}[{first} + t] : 0.0f, {target});")
        return lines

    def issue(self, wave, tile: int) -> list[str]:
        lines = []
        rec = self.body.rec
        for dense in wave[0]:
            name = f"{dense.index}_{tile}"
            count = dense.columns // 2
            lines.append(f"float d{name}[{count}];")
            if dense.bias is None:
                lines.append(f"for (int i = 0; i < {count}; ++i) d{name}[i] = 0.0f;")
            else:
                if dense.index not in self.bias:
                    self.bias[dense.index] = self.body.float_block(dense.bias, 4)
                for nt in range(dense.columns // 8):
                    word = f"{rec}[{self.bias[dense.index] + 8 * nt} + 2 * t"
                    lines.append(f"d{name}[{4 * nt}] = d{name}[{4 * nt + 2}] = {word}];")
                    lines.append(f"d{name}[{4 * nt + 1}] = d{name}[{4 * nt + 3}] = {word} + 1];")
            if dense.index not in self.blocks:
                high, low = tf32_split(core_matrices(dense.weights))
                self.blocks[dense.index] = (self.body.float_block(high, 32), self.body.float_block(low, 32))
        for dense in wave[0]:
            name = f"{dense.index}_{tile}"
            lines += [f"keep(d{name});", f"keep(ah{name});", f"keep(al{name});"]
        lines.append("wg_fence();")
        for dense in wave[0]:
            name = f"{dense.index}_{tile}"
            high, low = self.blocks[dense.index]
            lines.append("{")
            lines.append(f"  const uint64_t bh = wg_descriptor({rec} + {high}, {16 * dense.columns});")
            lines.append(f"  const uint64_t bl = wg_descriptor({rec} + {low}, {16 * dense.columns});")
            lines.append("#pragma unroll")
            lines.append(f"  for (int ks = 0; ks < {dense.steps}; ++ks) {{   // small terms first")
            step = 2 * dense.columns
            lines.append(f"    wg_mma_{dense.columns}(d{name}, al{name}[ks], bh + ks * {step});")
            lines.append(f"    wg_mma_{dense.columns}(d{name}, ah{name}[ks], bl + ks * {step});")
            lines.append(f"    wg_mma_{dense.columns}(d{name}, ah{name}[ks], bh + ks * {step});")
            lines += ["  }", "}"]
        lines.append("wg_commit();")
        return lines

    def map_text(self, stage: Stage, arguments: list[str], reciprocal: bool) -> str:
        template = MAP_OPERATORS[stage.op][1]
        if reciprocal:
            template = MAP_OPERATORS["mul"][1]
        elif self.options.approximate_functions:
            template = APPROXIMATE_FUNCTIONS.get(stage.op, template)
        return template.format(*arguments)

    def finish(self, wave, following, tile: int) -> list[str]:
        """After the wave's product sums of the tile are complete: elementwise stages, the A operands of the next
        wave, and the stores of the values that the rest of the body reads."""
        lines = []
        for dense in wave[0]:
            name = f"{dense.index}_{tile}"
            lines += [f"keep(d{name});", f"keep(ah{name});", f"keep(al{name});"]
        for stage in wave[1]:
            inner = self.phase.inner[stage.index]
            columns = self.phase.columns(stage.index)
            operands = []
            reciprocal = False
            for position, refs in enumerate(stage.operands):
                block = refs.reshape(self.members, inner)
                if self.is_const[ref_stage(block)].all():
                    values = self.graph.const_values(block)
                    if stage.op == "div" and position == 1:
                        inverse = self.body.reciprocal_values(values)
                        if inverse is not None:
                            values, reciprocal = inverse, True
                    operands.append(self.column_constant(values))
                else:
                    source = int(ref_stage(block)[0, 0])
                    operands.append(source)
            lines.append(f"// {stage.name}: {stage.op}")
            for q in range(columns // 2):
                nt, i = divmod(q, 4)
                arguments = [o(nt, i & 1) if callable(o) else self.value(o, tile, q) for o in operands]
                lines.append(f"const float f{stage.index}_{tile}_{q} = {self.map_text(stage, arguments, reciprocal)};")
        for dense in (following[0] if following else []):
            if dense.source is None:
                continue
            name = f"{dense.index}_{tile}"
            lines.append(f"uint32_t ah{name}[{dense.steps}][4], al{name}[{dense.steps}][4];")
            elements = self.phase.inner[dense.source]
            for nt in range(dense.steps):
                for slot, i in enumerate((0, 2, 1, 3)):
                    value = self.value(dense.source, tile, 4 * nt + i)
                    if 8 * nt + 8 > elements:   # padded columns hold whatever the stages made of zero: not a term
                        value = f"({8 * nt + (i & 1)} + 2 * t < {elements} ? {value} : 0.0f)"
                    lines.append(f"split({value}, ah{name}[{nt}][{slot}], al{name}[{nt}][{slot}]);")
        for index in self.phase.exported:
            if index not in [d.index for d in wave[0]] + [s.index for s in wave[1]]:
                continue
            inner = self.phase.inner[index]
            rows, stride, offset = self.body.emitter.slab_rows(self.phase.scratch_offset[index])
            for half in (0, 1):
                row = f"o{index}_{tile}_{half}"
                lines.append(f"float* const {row} = {rows} + (lr0 + {64 * tile + 8 * half} + g) * {stride} + {offset};")
                for nt in range(self.phase.columns(index) // 8):
                    for e in (0, 1):
                        first = 8 * nt + e
                        if first >= inner:
                            continue
                        guard = "" if first + 6 < inner else f"if ({first} + 2 * t < {inner}) "
                        lines.append(f"{guard}{row}[{first} + 2 * t] = {self.value(index, tile, 4 * nt + 2 * half + e)};")
        return lines

    # ---- the phase -------------------------------------------------------------------------------------------

    def lines(self) -> list[str]:
        """Statements of the phase inside the loop: the two tiles of the warpgroup alternate, so that the product
        sums of one are in flight while the elementwise stages of the other run. (A warpgroup with one tile
        waits for its product sums before the stages on them.)"""
        waves = self.waves
        tiles = range(self.body.emitter.tiles_planned)
        others = len(tiles) - 1   # product sums of the other tile that are started later and may still run
        lines = ["// ---- tensor cores: product sums and the stages on their accumulators, 64 rows per tile"]
        for tile in tiles:
            for dense in waves[0][0]:
                lines += self.load_inputs(dense, tile)
            lines += self.issue(waves[0], tile)
        for number in range(1, len(waves)):
            for tile in tiles:
                lines.append(f"wg_wait<{others}>();")
                lines += self.finish(waves[number - 1], waves[number], tile)
                for dense in waves[number][0]:
                    if dense.source is None:
                        lines += self.load_inputs(dense, tile)
                lines += self.issue(waves[number], tile)
        for tile in tiles:
            lines.append(f"wg_wait<{others - tile}>();")
            lines += self.finish(waves[-1], None, tile)
        return lines + ["__syncwarp();", "// ---- one row per thread"]

    def stored_values(self) -> list[str]:
        """Loads of the values that the rest of the body reads (those of the iterations the body is written for)."""
        lines = []
        for index in self.phase.exported:
            offset = self.phase.scratch_offset[index]
            lines += [f"const float v{index}_{slot} = s[{offset + slot}];"
                      for slot in self.body.live_slots(self.graph.stages[index])]
        return lines

    def mma_widths(self) -> set[int]:
        return {dense.columns for dense in self.phase.dense.values()}


class ThreadFragments:
    """Statements of a fragment phase with the iterations by thread (module text). ``body`` is the emitter's body
    of the group; its record is that of one iteration (biases and the constants of the tail). The weights of a
    chunk of four iterations are the blocks of ``chunk_blocks``."""

    def __init__(self, body, phase: FragmentPhase):
        self.body = body
        self.phase = phase
        self.graph: RowGraph = body.graph
        self.members = body.members
        self.chunks = -(-self.members // CHUNK)
        self.name = f"{body.position}"
        self.dense = [phase.dense[index] for index in phase.stages]
        self.chunk_blocks: list[np.ndarray] = []         # [chunks, words] each, on 32-word boundaries
        self.chunk_words = 0
        self.blocks: dict[int, tuple[int, int]] = {}     # dense stage -> chunk words of its high and low weights
        self.bias: dict[int, int] = {}                   # dense stage -> words of its biases in the iteration's record
        # the image of the low parts of every vector, per tile: [k-step][k half][warp, row half][8 rows][4 k]
        self.image_offset = []
        self.image_words = 0
        for vector in phase.vectors:
            self.image_offset.append(self.image_words)
            self.image_words += -(-len(vector) // 8) * 2 * 8 * 32
        for dense in self.dense:
            high, low = tf32_split(core_matrices(self.chunk_weights(dense)))
            self.blocks[dense.index] = (self.block(high), self.block(low))

    def columns(self, dense: Dense) -> int:
        """N of the instruction: two columns per element pair and iteration."""
        return 8 * -(-self.phase.inner[dense.index] // 2)

    def chunk_weights(self, dense: Dense) -> np.ndarray:
        """[chunks, 8 steps, N]: column 8 nt + c is element 2 nt + c % 2 of iteration 4 q + c / 2."""
        pairs = self.columns(dense) // 8
        terms = dense.weights.shape[1]
        padded = np.zeros((CHUNK * self.chunks, terms, 2 * pairs), dtype=np.float32)
        used = min(2 * pairs, dense.weights.shape[2])
        padded[:self.members, :, :used] = dense.weights[:, :, :used]
        by_chunk = padded.reshape(self.chunks, CHUNK, terms, pairs, 2)
        return np.ascontiguousarray(by_chunk.transpose(0, 2, 3, 1, 4)).reshape(self.chunks, terms, 8 * pairs)

    def block(self, values: np.ndarray) -> int:
        offset = self.chunk_words
        words = -(-values.shape[1] // 32) * 32
        block = np.zeros((self.chunks, words), dtype=np.float32)
        block[:, :values.shape[1]] = values
        self.chunk_blocks.append(block)
        self.chunk_words += words
        return offset

    def steps(self, number: int) -> list[tuple[str, int]]:
        """Where the first value of every k-step of a vector is: ("input", element of the input row) or ("slab",
        word of the row's slab: the values themselves, or their copy)."""
        places, copies = [], 0
        for step in self.phase.direct[number]:
            if step is None:
                places.append(("slab", self.phase.vector_offset[number] + 8 * copies))
                copies += 1
            elif self.graph.stages[step[0]].kind == "input":
                places.append(("input", step[1]))
            else:
                places.append(("slab", self.body.emitter.plan.slab_offset[step[0]] + step[1]))
        return places

    def before_loop(self) -> list[str]:
        """Before the loop, one row per thread: the row values that the product sums read and that do not stand
        next to each other in the row's slab or in the input row are copied to the slab."""
        lines = ["// values of the row that the tensor-core product sums read"]
        for number, vector in enumerate(self.phase.vectors):
            for ks, (step, (_, word)) in enumerate(zip(self.phase.direct[number], self.steps(number))):
                if step is None:
                    lines += [f"s[{word + j}] = {self.body.outside_value(ref)};" for j, ref in enumerate(vector[8 * ks:8 * ks + 8])]
        return lines + ["__syncwarp();"]

    def tile_setup(self) -> list[str]:
        """At the start of a tile: every thread takes the A operands of its two rows from the slabs or the input
        rows, high parts into registers, low parts into the warpgroup's image."""
        places = [self.steps(number) for number in range(len(self.phase.vectors))]
        kinds = {kind for steps in places for kind, _ in steps}
        lines = []
        slab_rows = self.body.emitter.slab_rows
        for rows, stride in sorted({slab_rows(word)[:2] for steps in places for kind, word in steps if kind == "slab"}, reverse=True):
            name = "rows" if rows == "slab" else "far_rows"
            lines.append(f"const float* const {name} = {rows} + (lr0 + 64 * tile + g) * {stride};   // rows g and g + 8 of the tile")
        if "input" in kinds:   # a row past the last one repeats the last row
            lines += ["const long long first_row = (long long)blockIdx.x * kRowsPerBlock + lr0 + 64 * tile + g;",
                      "const float* const inputs[2] = {rows_in + min(first_row, row_count - 1) * kInputs,",
                      "                                rows_in + min(first_row + 8, row_count - 1) * kInputs};"]
        for number, vector in enumerate(self.phase.vectors):
            steps = -(-len(vector) // 8)
            lines.append(f"uint32_t ah{self.name}_{number}[{steps}][4];")
            for ks, (kind, word) in enumerate(places[number]):
                real = min(len(vector) - 8 * ks, 8)
                value = "x[k]" if real == 8 else f"k < {real} ? x[k] : 0.0f"
                if kind == "input":
                    source = f"inputs[i & 1] + {word}"
                else:
                    rows, stride, word = slab_rows(word)
                    source = f"{'rows' if rows == 'slab' else 'far_rows'} + {word} + (i & 1) * 8 * {stride}"
                lines += [
                    "#pragma unroll",
                    "for (int i = 0; i < 4; ++i) {",
                    f"  const float* const x = {source};",
                    "  const int k = 4 * (i >> 1) + t;",
                    "  uint32_t low;",
                    f"  split({value}, ah{self.name}_{number}[{ks}][i], low);",
                    f"  images[wg * kTileImageWords + {self.image_offset[number]} + (({2 * ks} + (i >> 1)) * 8 + 2 * w + (i & 1)) * 32 + 4 * g + t]"
                    " = __uint_as_float(low);",
                    "}",
                ]
        # the image is the warpgroup's own: its warps wait for each other, not for the block
        return lines + ["fence_async();", "warpgroup_sync(wg);"]

    def declarations(self) -> list[str]:
        """The accumulators of the product sums (they live across the chunks of a tile)."""
        return [f"float d{dense.index}[{self.columns(dense) // 2}];" for dense in self.dense]

    def issue_lines(self, chunk: str) -> list[str]:
        """Start the product sums of the tile for the chunk whose blocks begin at ``chunk``; the accumulators
        start from the biases of the thread's iteration (in the record ``body.rec`` of that iteration)."""
        rec = self.body.rec
        lines = []
        for dense in self.dense:
            count = self.columns(dense) // 2
            if dense.bias is None:
                lines.append(f"for (int i = 0; i < {count}; ++i) d{dense.index}[i] = 0.0f;")
                continue
            if dense.index not in self.bias:
                self.bias[dense.index] = self.body.float_block(dense.bias[:, :min(count // 2, dense.bias.shape[1])], 4)
            for nt in range(count // 4):
                for e in (0, 1):
                    value = f"{rec}[{self.bias[dense.index] + 2 * nt + e}]" if 2 * nt + e < dense.bias.shape[1] else "0.0f"
                    lines.append(f"d{dense.index}[{4 * nt + e}] = d{dense.index}[{4 * nt + 2 + e}] = {value};")
        lines += [f"keep(d{dense.index});" for dense in self.dense]
        lines.append("wg_fence();")
        for dense in self.dense:
            columns = self.columns(dense)
            high, low = self.blocks[dense.index]
            a = f"ah{self.name}_{dense.vector}"
            lines += [
                "{",
                f"  const uint64_t bh = wg_descriptor({chunk} + {high}, {16 * columns});",
                f"  const uint64_t bl = wg_descriptor({chunk} + {low}, {16 * columns});",
                f"  const uint64_t al = wg_descriptor(images + wg * kTileImageWords + {self.image_offset[dense.vector]}, 1024);",
                "#pragma unroll",
                f"  for (int ks = 0; ks < {dense.steps}; ++ks) {{   // small terms first",
                f"    wg_mma_shared_{columns}(d{dense.index}, al + ks * 128, bh + ks * {2 * columns});",
                f"    wg_mma_{columns}(d{dense.index}, {a}[ks], bl + ks * {2 * columns});",
                f"    wg_mma_{columns}(d{dense.index}, {a}[ks], bh + ks * {2 * columns});",
                "  }",
                "}",
            ]
        return lines + ["wg_commit();"]

    def complete_lines(self) -> list[str]:
        """Wait for the product sums that were started last."""
        return ["wg_wait<0>();"] + [f"keep(d{dense.index});" for dense in self.dense]

    def row_values(self) -> list[str]:
        """Inside the loop over the thread's two rows ``h``: the elements of its iteration as scalars."""
        lines = []
        for dense in self.dense:
            for slot in self.body.live_slots(self.graph.stages[dense.index]):
                lines.append(f"const float v{dense.index}_{slot} = d{dense.index}[{4 * (slot // 2) + slot % 2} + 2 * h];")
        return lines

    def mma_widths(self) -> set[int]:
        return {self.columns(dense) for dense in self.dense}


@dataclass
class DenseChain:
    """A chain of wide product sums outside a loop (module text).

    narrow   the product sum of a few row values at its start, or None when the first wide sum reads row values
    maps     per wide product sum, the elementwise stages on its input: maps[0] act on the narrow sum (none
             without one), maps[l] on the wide sum l - 1
    wide     the wide product sums in order
    vector   references of the row values that the narrow sum, or without one the first wide sum, reads
    place    where the threads find the vector: ("input", element of the input row), ("stage", stage, element)
             for values that are in the rows' slabs anyway, or None for a copy at vector_offset[0]
    sides    stages of row values that the elementwise stages between the sums read besides the sum (they are
             kept in the rows' slabs while the chain runs)
    trailing elementwise stages on the last sum that the chain computes too: the last of them, or the last sum,
             is the ``result``
    read     elements of the result that other code reads: the threads that hold a row in a tile store them in
             the rows' slabs, and the elements that are outputs of the kernel in the output rows
    """

    narrow: int | None
    maps: list[list[int]]
    wide: list[int]
    vector: np.ndarray
    by_thread: bool = False
    vectors: list[np.ndarray] = field(default_factory=list)
    vector_offset: list[int] = field(default_factory=list)
    exported: list[int] = field(default_factory=list)
    place: tuple | None = None
    sides: list[int] = field(default_factory=list)
    trailing: list[int] = field(default_factory=list)
    read: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    @property
    def stages(self) -> list[int]:
        start = [] if self.narrow is None else [self.narrow]
        return sorted(start + self.wide + self.trailing + [index for maps in self.maps for index in maps])

    @property
    def result(self) -> int:
        return self.trailing[-1] if self.trailing else self.wide[-1]

    def in_blocks(self, graph: RowGraph) -> bool:
        """Whether a sum of the chain has more terms or elements than one instruction."""
        return any(max(graph.stages[index].terms, graph.stages[index].size) > CHAIN_COLUMNS for index in self.wide)


def find_dense_chains(graph: RowGraph, readers: dict[int, set[int]], in_loop_body, blocks: bool = True,
                      once: bool = True) -> list[DenseChain]:
    """The chains of wide product sums of the graph (module text), in the order of their first stages.
    ``readers`` gives for every stage the stages that read it; ``in_loop_body(stage)`` says whether a product sum
    is the body of a loop over several elements each; without ``blocks`` no sum has more terms or elements than
    one instruction; with ``once`` a sum of few terms that feeds a sum in blocks of terms and of elements is a
    chain of its own (feeds_blocks below)."""
    widest = CHAIN_WIDEST if blocks else CHAIN_COLUMNS
    is_const = graph.is_const()

    def same_terms(stage: Stage) -> bool:
        return stage.kind == "dot" and stage.weights is not None and bool((stage.operands[0] == stage.operands[0][0]).all())

    def wide(stage: Stage) -> bool:
        return (same_terms(stage) and (LOOP_TERMS if in_loop_body(stage) else CHAIN_TERMS) <= stage.terms <= widest
                and stage.size <= widest)

    def whole(refs: np.ndarray, size: int) -> int | None:
        """The stage whose ``size`` elements, in order, the references are."""
        source = int(ref_stage(refs)[0])
        return source if graph.stages[source].size == size and np.array_equal(refs, make_refs(source, np.arange(size))) else None

    def whole_input(stage: Stage) -> int | None:
        """The stage whose elements, in order, are the terms of every element of this wide product sum."""
        return whole(stage.operands[0][0], stage.terms)

    def variable_operand(stage: Stage) -> int | None:
        """The stage that an elementwise stage maps element by element (its other operands are one constant)."""
        if stage.kind != "map" or stage.is_bool:
            return None
        found = None
        for refs in stage.operands:
            if is_const[ref_stage(refs)].all():
                if len(np.unique(graph.const_values(refs).view(np.uint32))) != 1:
                    return None
                continue
            source = int(ref_stage(refs)[0])
            whole = graph.stages[source].size == stage.size and np.array_equal(refs, make_refs(source, np.arange(stage.size)))
            if found is not None or not whole:
                return None
            found = source
        return found

    def row_values_beside(stage: Stage, source: int, start: int) -> list[int] | None:
        """For an elementwise stage on ``source``: the stages of row values (computed ahead of stage ``start``)
        that it reads besides, element by element; None when it is no such stage."""
        if stage.kind != "map" or stage.is_bool:
            return None
        beside, found = [], False
        for refs in stage.operands:
            if is_const[ref_stage(refs)].all():
                if len(np.unique(graph.const_values(refs).view(np.uint32))) != 1:
                    return None
                continue
            other = whole(refs, stage.size)
            if other is None:
                return None
            computed = graph.stages[other].kind in ("map", "dot", "reduce") and not graph.stages[other].is_bool
            if other != source and (other >= start or not computed):
                return None   # only computed values of the row are in the rows' slabs
            found = found or other == source
            beside += [other] if other != source else []
        return beside if found else None

    def in_blocks_both_ways(stage: Stage) -> bool:
        """A wide sum whose A operand is built again for every block of its elements."""
        return wide(stage) and stage.terms > CHAIN_COLUMNS and stage.size > CHAIN_COLUMNS

    def feeds_blocks(stage: Stage) -> bool:
        """A sum of few terms whose values, through elementwise stages, are the terms of a sum in blocks of terms
        and of elements. As the start of that sum's chain it would be computed again for every block of elements:
        it is a chain of its own, computed once, and the wide sum reads its values in the rows' slabs."""
        if not once or not same_terms(stage) or stage.terms >= CHAIN_TERMS or stage.size > widest:
            return False
        last = stage.index
        while len(readers.get(last, ())) == 1:
            (reader,) = readers[last]
            if reader < 0:
                return False
            if variable_operand(graph.stages[reader]) == last:
                last = reader
                continue
            return in_blocks_both_ways(graph.stages[reader]) and whole_input(graph.stages[reader]) == last
        return False

    chains: list[DenseChain] = []
    used: set[int] = set()
    for first in graph.stages:
        if first.index in used or first.index not in readers or not (wide(first) or feeds_blocks(first)):
            continue
        if is_const[ref_stage(first.operands[0][0])].any():
            continue
        chain = None
        source = whole_input(first)
        if source is not None:
            # back from the input of the first wide sum to a narrow sum of row values
            maps, reader = [], first.index
            while variable_operand(graph.stages[source]) is not None and readers.get(source) == {reader}:
                maps.insert(0, source)
                reader, source = source, variable_operand(graph.stages[source])
            narrow = graph.stages[source]
            if (not (once and in_blocks_both_ways(first)) and same_terms(narrow) and narrow.terms <= 8
                    and readers.get(source) == {reader}
                    and not is_const[ref_stage(narrow.operands[0])].any() and not used & set(maps + [source])):
                chain = DenseChain(narrow.index, [maps], [first.index], narrow.operands[0][0])
        if chain is None:   # the first wide sum reads values of the row
            chain = DenseChain(None, [[]], [first.index], first.operands[0][0])
        # forward: elementwise stages on the wide sum, then the next wide sum
        start = min(chain.stages)
        while True:
            maps, sides, last = [], [], chain.wide[-1]
            while len(readers.get(last, ())) == 1:
                (reader,) = readers[last]
                beside = row_values_beside(graph.stages[reader], last, start) if reader >= 0 else None
                if beside is None:
                    break
                maps.append(reader)
                sides += beside
                last = reader
            # the accumulators of a sum of one instruction are the A operand of the next sum
            linked = -(-graph.stages[chain.wide[-1]].size // 8) * 8 <= CHAIN_COLUMNS
            following = [index for index in readers.get(last, ()) if linked and index >= 0 and index not in used
                         and wide(graph.stages[index]) and whole_input(graph.stages[index]) == last]
            chain.sides += [index for index in sides if index not in chain.sides]
            if len(readers.get(last, ())) != 1 or not following or not maps:
                chain.trailing = maps   # the chain ends: it stores the value that other code reads
                break
            chain.maps.append(maps)
            chain.wide.append(following[0])
        chain.vectors = [chain.vector]
        used |= set(chain.stages)
        chains.append(chain)
    return chains


def _runs(values: np.ndarray) -> list[tuple[int, int, int]]:
    """(first index, index behind the last, first value) of the runs of an integer vector whose values grow by one
    from index to index; negative values belong to no run."""
    runs: list[tuple[int, int, int]] = []
    for index, value in enumerate(values):
        if value < 0:
            continue
        if runs and runs[-1][1] == index and runs[-1][2] + index - runs[-1][0] == value:
            runs[-1] = (runs[-1][0], index + 1, runs[-1][2])
        else:
            runs.append((index, index + 1, int(value)))
    return runs


class ChainWriter:
    """Statements of a chain of wide product sums. ``body`` is the emitter's body of the chain's group."""

    def __init__(self, body, chain: DenseChain):
        self.body = body
        self.chain = chain
        self.graph: RowGraph = body.graph
        self.emitter = body.emitter
        self.options = body.emitter.options
        self.name = f"{body.position}"
        self.table: list[np.ndarray] = []
        self.table_words = 0

    def tabulate(self, values: np.ndarray) -> int:
        """Word offset of constants in the chain's table (shared memory, read by element index)."""
        offset = self.table_words
        self.table.append(np.asarray(values, dtype=np.float32))
        self.table_words += len(values)
        return offset

    def map_lines(self, stages: list[int], value: str, element: str = "") -> list[str]:
        """Statements that apply the elementwise stages to the float variable ``value``. ``element`` is the slab
        word, without the stage's offset, of the row and element that ``value`` belongs to: the stages may read
        other values of the row there (DenseChain.sides)."""
        lines = []
        is_const = self.graph.is_const()
        for index in stages:
            stage = self.graph.stages[index]
            template = MAP_OPERATORS[stage.op][1]
            if self.options.approximate_functions:
                template = APPROXIMATE_FUNCTIONS.get(stage.op, template)
            arguments = []
            for refs in stage.operands:
                source = int(ref_stage(refs)[0])
                if is_const[ref_stage(refs)].all():
                    arguments.append(float_literal(self.graph.const_values(refs)[0]))
                elif source in self.chain.sides:   # "element" indexes the rows in shared memory, "far_element" the others
                    rows, _, offset = self.emitter.slab_rows(self.emitter.plan.slab_offset[source])
                    arguments.append(f"{rows}[{element if rows == 'slab' else 'far_' + element} + {offset}]")
                else:
                    arguments.append(value)
            lines.append(f"{value} = {template.format(*arguments)};   // {stage.name}")
        return lines

    def weights(self, link: int, elements: tuple[int, int], terms: tuple[int, int], room: int) -> tuple[np.ndarray, int, int, int]:
        """(slabs [count, floats], k-steps per slab, columns N, k-steps) of the block ``elements`` x ``terms`` of a
        wide product sum: per slab the high parts of its k-steps, then the low parts. A slab has at most ``room``
        floats (and one k-step at least)."""
        stage = self.graph.stages[self.chain.wide[link]]
        (n0, n1), (k0, k1) = elements, terms
        columns, steps = -(-(n1 - n0) // 8) * 8, -(-(k1 - k0) // 8)
        ordered = np.zeros((1, 8 * steps, columns), dtype=np.float32)
        term = np.arange(k1 - k0)
        if link > 0:   # the A operand is the accumulators of the sum before: k index of a column
            term = 8 * (term // 8) + (term % 8) // 2 + 4 * (term % 2)
        ordered[0, term, :n1 - n0] = stage.weights[n0:n1, k0:k1].T
        high, low = tf32_split(core_matrices(ordered))
        together = max(g for g in range(1, steps + 1) if steps % g == 0 and (g * 16 * columns <= room or g == 1))
        count = steps // together
        return np.concatenate([high.reshape(count, -1), low.reshape(count, -1)], axis=1), together, columns, steps

    def lines(self) -> list[str]:
        chain, graph = self.chain, self.graph
        wide = [graph.stages[index] for index in chain.wide]
        narrow = None if chain.narrow is None else graph.stages[chain.narrow]
        size0 = len(chain.vector) if narrow is None else narrow.size   # terms of the first wide sum
        if narrow is not None:
            bias0 = self.tabulate(narrow.bias if narrow.bias is not None else np.zeros(size0))
            factor0 = [self.tabulate(narrow.weights[:, j]) for j in range(narrow.terms)]
        bias = []
        for stage in wide:
            padded = np.zeros(-(-stage.size // 8) * 8, dtype=np.float32)
            if stage.bias is not None:
                padded[:stage.size] = stage.bias
            bias.append(self.tabulate(padded))
        table = np.zeros((1, -(-self.table_words // 32) * 32), dtype=np.float32)
        table[0, :self.table_words] = np.concatenate(self.table)
        table_offset = self.emitter.records.append(table.view(np.uint32))

        def ranges(size: int, width: int = CHAIN_COLUMNS) -> list[tuple[int, int]]:
            return [(first, min(first + width, size)) for first in range(0, size, width)]

        def term_ranges(terms: int) -> list[tuple[int, int]]:
            """The terms of a sum in ranges of one A operand."""
            return ranges(terms, CHAIN_COLUMNS if terms <= CHAIN_COLUMNS else CHAIN_BLOCK_TERMS)

        # Blocks of every sum: its elements and its terms in ranges of one instruction. Their weight slabs, in the
        # order of the computation (sum, elements, terms), pass through one ring of two slots (each as large as
        # the largest slab); the sequence runs over both tiles. Stage buffer: the table, the barriers and counters
        # of the two slots, the slots. A slot has the emitter's floats, or what the shared memory of one warpgroup leaves
        # beside the rows' slabs, the image and the table (a chain in blocks keeps a whole layer in the slabs).
        image_words = max(self.emitter.tile_image_words,
                          max(-(-(last - first) // 8) for stage in wide for first, last in term_ranges(stage.terms)) * 2 * 8 * 32)
        in_shared = 128 * (self.emitter.plan.slab_words - self.emitter.plan.far_words)
        left = self.options.shared_memory_bytes // 4 - in_shared - image_words - 64 - table.shape[1] - 32
        room = min(self.emitter.slot_floats, left // 2)
        pieces = []   # (sum, elements, terms, slabs, k-steps per slab, columns, k-steps, constants, position in a tile)
        for link, stage in enumerate(wide):
            for elements in ranges(stage.size):
                for terms in term_ranges(stage.terms):
                    blocks, together, columns, steps = self.weights(link, elements, terms, room)
                    place = self.emitter.records.append(blocks.view(np.uint32)) // 4
                    pieces.append((link, elements, terms, blocks, together, columns, steps, place,
                                   sum(piece[3].shape[0] for piece in pieces)))
        per_tile = sum(piece[3].shape[0] for piece in pieces)
        slot_quads = max(piece[3].shape[1] for piece in pieces) // 4
        slots = f"(staged + {table.shape[1] // 4 + 8})"
        self.emitter.stage_words = max(self.emitter.stage_words, table.shape[1] + 32 + 8 * slot_quads)

        def slab_at(position: int) -> tuple[int, int]:
            """(first float4 in the constants, bytes) of the slab at a position of the tile's sequence."""
            *_, blocks, _, _, _, place, start = max((piece for piece in pieces if piece[8] <= position), key=lambda piece: piece[8])
            return place + (position - start) * (blocks.shape[1] // 4), 4 * blocks.shape[1]

        def slab_expressions(position: str) -> tuple[str, str]:
            """The same for a position that the kernel computes."""
            source, size = "", ""
            for number, (*_, blocks, _, _, _, place, start) in enumerate(pieces):
                choice = f"{position} < {start + blocks.shape[0]} ? " if number + 1 < len(pieces) else ""
                source += f"{choice}{place} + ({position} - {start}) * {blocks.shape[1] // 4}" + (" : " if choice else "")
                size += f"{choice}{4 * blocks.shape[1]}" + (" : " if choice else "")
            return source, size

        full, done = f"full{self.name}", f"done{self.name}"
        steps_most = max(piece[6] for piece in pieces)
        self.emitter.tile_image_words = max(self.emitter.tile_image_words, steps_most * 2 * 8 * 32)
        self.emitter.mma_widths |= {piece[5] for piece in pieces}
        names = ", ".join(stage.name for stage in wide)
        ah, image = f"ah{self.name}", "images + wg * kTileImageWords"

        def first_operand(terms: tuple[int, int]) -> list[str]:
            """The A operand of the first sum for a range of its terms: every thread the values of its register
            slots, for its two rows, from the row values where the threads that hold a row in a tile find them."""
            k0, steps = terms[0], -(-(terms[1] - terms[0]) // 8)
            lines = ["  {"]
            # Row values that are read again for every block of elements are read as volatile: the compiler
            # otherwise keeps the values of all terms from one block of elements to the next (nothing else writes
            # the input rows), on the stack, with no register left for the product sums.
            again = narrow is None and len(term_ranges(size0)) > 1 and len(ranges(wide[0].size)) > 1
            pointer = "const volatile float*" if again else "const float*"
            if chain.place is not None and chain.place[0] == "input":
                lines += ["    const long long first_row = (long long)blockIdx.x * kRowsPerBlock + lr0 + 64 * tile + g;",
                          f"    {pointer} const xa = rows_in + min(first_row, row_count - 1) * kInputs + {chain.place[1]};",
                          f"    {pointer} const xb = rows_in + min(first_row + 8, row_count - 1) * kInputs + {chain.place[1]};"]
            else:
                rows, stride, word = self.emitter.slab_rows(
                    chain.vector_offset[0] if chain.place is None else self.emitter.plan.slab_offset[chain.place[1]] + chain.place[2])
                lines += [f"    {pointer} const xa = {rows} + (lr0 + 64 * tile + g) * {stride} + {word};",
                          f"    {pointer} const xb = xa + 8 * {stride};"]
            rows = "xb : xa"
            if narrow is not None:   # the few row values of the narrow sum: read here once, not for every term
                for row in "ab":
                    values = ", ".join(f"x{row}[{j}]" for j in range(narrow.terms))
                    lines.append(f"    const float row_{row}[{narrow.terms}] = {{{values}}};")
                rows = "row_b : row_a"
            padded = terms[1] == size0 and size0 % 8 != 0   # the last k-step of the sum has fewer than eight terms
            if narrow is not None and len(term_ranges(size0)) > 1:
                # A narrow sum that is computed again for every block: the four values of a k-step advance
                # together, one operation at a time (a thread that computes one value after the other waits for
                # every load of a factor before it goes on).
                k = f"{f'{k0} + ' if k0 else ''}8 * ks + 4 * (i >> 1) + t"
                steps_of_a_value = [f"value[i] = table{self.name}[{bias0} + {k}];"]
                steps_of_a_value += [f"value[i] = fmaf(((i & 1) ? {rows})[{j}], table{self.name}[{offset} + {k}], value[i]);"
                                     for j, offset in enumerate(factor0)]
                steps_of_a_value += self.map_lines(chain.maps[0], "value[i]")
                steps_of_a_value += [f"if ({k} >= {size0}) value[i] = 0.0f;"] if padded else []
                lines += [
                    f"    float* const image = {image};",
                    "#pragma unroll",
                    f"    for (int ks = 0; ks < {steps}; ++ks) {{   // the A operand of the first sum, computed where it is used",
                    "      float value[4];",
                ]
                for step in steps_of_a_value:
                    lines += ["#pragma unroll", f"      for (int i = 0; i < 4; ++i) {step}"]
                return lines + [
                    "#pragma unroll",
                    "      for (int i = 0; i < 4; ++i) {",
                    "        uint32_t low;",
                    f"        split(value[i], {ah}[ks][i], low);",
                    "        image[((2 * ks + (i >> 1)) * 8 + 2 * w + (i & 1)) * 32 + 4 * g + t] = __uint_as_float(low);",
                    "      }",
                    "    }",
                    "  }",
                    "  fence_async();",
                    "  warpgroup_sync(wg);   // the image is the warpgroup's own: its warps wait for each other, not for the block",
                ]
            lines += [
                f"    float* const image = {image};",
                "#pragma unroll",
                f"    for (int ks = 0; ks < {steps}; ++ks) {{",
                "#pragma unroll",
                "      for (int i = 0; i < 4; ++i) {   // the A operand of the first sum, computed where it is used",
                f"        const int k = {f'{k0} + ' if k0 else ''}8 * ks + 4 * (i >> 1) + t;",
                f"        {pointer} const x = (i & 1) ? {rows};",
            ]
            if narrow is None:   # the values of the row themselves
                lines.append(f"        float value = k < {size0} ? x[k] : 0.0f;" if padded else "        float value = x[k];")
            else:
                lines.append(f"        float value = table{self.name}[{bias0} + k];")
                lines += [f"        value = fmaf(x[{j}], table{self.name}[{offset} + k], value);" for j, offset in enumerate(factor0)]
                lines += ["        " + line for line in self.map_lines(chain.maps[0], "value")]
                if padded:
                    lines.append(f"        if (k >= {size0}) value = 0.0f;")
            return lines + [
                "        uint32_t low;",
                f"        split(value, {ah}[ks][i], low);",
                "        image[((2 * ks + (i >> 1)) * 8 + 2 * w + (i & 1)) * 32 + 4 * g + t] = __uint_as_float(low);",
                "      }",
                "    }",
                "  }",
                "  fence_async();",
                "  warpgroup_sync(wg);   // the image is the warpgroup's own: its warps wait for each other, not for the block",
            ]

        lines = ["", f"// ---- product sums on the tensor cores, 64 rows per tile: {names}"]
        if chain.place is None:
            lines += vector_stores(self.body, chain)[:-1]
        lines += [
            "__syncthreads();",
            f"for (int i = threadIdx.x; i < {table.shape[1] // 4}; i += kThreadsPerBlock)",
            f"  staged[i] = load_constants(constants + {table_offset // 4} + i);",
            "__syncthreads();",
            f"const float* const table{self.name} = (const float*)staged;",
            f"uint64_t* const {full} = (uint64_t*)(staged + {table.shape[1] // 4});   // barriers of the two weight slots",
            f"int* const {done} = (int*)({full} + 2);   // warps that are through with a slot",
            "if (threadIdx.x == 0) {",
            f"  barrier_init({full});",
            f"  barrier_init({full} + 1);",
            f"  {done}[0] = {done}[1] = 0;",
            "  barrier_init_done();",
            "}",
            "__syncthreads();",
            "if (threadIdx.x == 0) {   // the first two weight slabs; the slots then run over all sums and both tiles",
        ]
        for slot in range(min(2, self.emitter.tiles_planned * per_tile)):   # no copy that no warp waits for
            source, size = slab_at(slot % per_tile)
            lines.append(f"  copy_bulk({slots} + {slot * slot_quads}, constants + {source}, {size}, {full} + {slot});")
        lines += [
            "}",
            "for (int tile = 0; tile < kTiles; ++tile) {   // the tiles of the warpgroup, one after the other",
            f"  uint32_t {ah}[{steps_most}][4];",
        ]
        lines += first_operand(term_ranges(size0)[0])
        next_source, next_bytes = slab_expressions("next")

        # Where the result goes, from the accumulators: the elements that other code reads to the rows' slabs, the
        # elements that are outputs of the kernel to the output rows. Runs of elements (first, behind the last,
        # slab word or output of the first).
        size = graph.stages[chain.result].size
        slab_runs, result_rows = [], "slab[in_block * kSlabWords + {}]"
        if len(chain.read):
            rows, stride, first_word = self.emitter.slab_rows(self.emitter.plan.slab_offset[chain.result])
            result_rows = f"{rows}[in_block * {stride} + {{}}]"
            words = np.full(size, -1, dtype=np.int64)
            words[chain.read] = first_word + chain.read
            slab_runs = _runs(words)
        far_sides = any(self.emitter.slab_rows(self.emitter.plan.slab_offset[side])[0] == "far" for side in chain.sides)
        positions = self.emitter.output_positions.get(chain.result)
        output_runs = [] if positions is None else [run for use in positions.T for run in _runs(use)]

        def within(first: int, last: int, columns: int) -> str:
            """Condition for the columns ``first`` to ``last - 1`` of a block of ``columns``: by n-tile where the
            bounds allow it (then known where the loop over the n-tiles is unrolled)."""
            name, scale = ("nt", 8) if first % 8 == 0 and last % 8 == 0 else ("column", 1)
            return " && ".join(([f"{name} >= {first // scale}"] if first else [])
                               + ([f"{name} < {last // scale}"] if last < columns else []))
        for link, stage in enumerate(wide):
            of_sum = [piece for piece in pieces if piece[0] == link]
            in_blocks = len(ranges(stage.size)) > 1
            for number, elements in enumerate(ranges(stage.size)):
                n0, columns = elements[0], of_sum[0][5] if not in_blocks else -(-(elements[1] - elements[0]) // 8) * 8
                d = f"d{stage.index}_{number}" if in_blocks else f"d{stage.index}"
                lines.append(f"  float {d}[{columns // 2}];   // {stage.name}" + (f", elements {n0} to {elements[1] - 1}" if in_blocks else ""))
                for nt in range(columns // 8):
                    for e in (0, 1):
                        lines.append(f"  {d}[{4 * nt + e}] = {d}[{4 * nt + 2 + e}] = table{self.name}[{bias[link] + n0 + 8 * nt + e} + 2 * t];")
                for _, _, terms, blocks, together, _, _, _, start in [piece for piece in of_sum if piece[1] == elements]:
                    if link == 0 and len(term_ranges(size0)) > 1 and (number, terms[0]) != (0, 0):
                        lines += first_operand(terms)   # the registers and the image hold one range of the terms
                    lines += [
                        "#pragma unroll",
                        f"  for (int slab_number = 0; slab_number < {blocks.shape[0]}; ++slab_number) {{   // weights in slabs of {together} k-steps",
                        f"    const int sequence = tile * {per_tile} + {start} + slab_number, slot = sequence & 1;",
                        f"    barrier_wait({full} + slot, (sequence >> 1) & 1);",
                        f"    const float* const weights = (const float*)({slots} + slot * {slot_quads});",
                        f"    const uint64_t bh = wg_descriptor(weights, {16 * columns});",
                        f"    const uint64_t bl = wg_descriptor(weights + {together * 8 * columns}, {16 * columns});",
                        f"    const uint64_t al = wg_descriptor({image}, 1024);",
                        f"    float part[{columns // 2}];   // the products of this slab",
                        f"    for (int i = 0; i < {columns // 2}; ++i) part[i] = 0.0f;",
                        "    keep(part);",
                        "    wg_fence();",
                    ]
                    if together <= 2:
                        lines += [
                            "#pragma unroll",
                            f"    for (int e = 0; e < {together}; ++e) {{   // small terms first",
                            f"      const int ks = {together} * slab_number + e;",
                            f"      wg_mma_shared_{columns}(part, al + ks * 128, bh + e * {2 * columns});",
                            f"      wg_mma_{columns}(part, {ah}[ks], bl + e * {2 * columns});",
                            f"      wg_mma_{columns}(part, {ah}[ks], bh + e * {2 * columns});",
                            "    }",
                        ]
                    else:   # a slab of many k-steps: the small products of all of them, then the large ones
                        lines += [
                            "#pragma unroll",
                            f"    for (int e = 0; e < {together}; ++e) {{   // the products with a low part",
                            f"      const int ks = {together} * slab_number + e;",
                            f"      wg_mma_shared_{columns}(part, al + ks * 128, bh + e * {2 * columns});",
                            f"      wg_mma_{columns}(part, {ah}[ks], bl + e * {2 * columns});",
                            "    }",
                            "#pragma unroll",
                            f"    for (int e = 0; e < {together}; ++e)   // the products of the high parts",
                            f"      wg_mma_{columns}(part, {ah}[{together} * slab_number + e], bh + e * {2 * columns});",
                        ]
                    lines += [
                        "    wg_commit();",
                        "    wg_wait<0>();",
                        "    keep(part);",
                        "#pragma unroll",
                        f"    for (int i = 0; i < {columns // 2}; ++i) {d}[i] += part[i];",
                        "    __syncwarp();",
                        f"    if (lane == 0 && atomicAdd({done} + slot, 1) == kThreadsPerBlock / 32 - 1) {{",
                        f"      {done}[slot] = 0;   // the last warp through with the slot fills it again",
                        f"      if (sequence + 2 < kTiles * {per_tile}) {{",
                        f"        const int next = (sequence + 2) % {per_tile};   // position in the sequence of a tile",
                        f"        copy_bulk({slots} + slot * {slot_quads}, constants + ({next_source}), ({next_bytes}), {full} + slot);",
                        "      }",
                        "    }",
                        "  }",
                    ]
                if link + 1 < len(wide):   # a sum of one instruction: its accumulators are the next A operand
                    lines += [
                        "#pragma unroll",
                        f"  for (int nt = 0; nt < {columns // 8}; ++nt) {{   // the A operand of the next sum, where the values are",
                        "#pragma unroll",
                        "    for (int i = 0; i < 4; ++i) {",
                        f"      float value = {d}[4 * nt + i];",
                    ]
                    if chain.sides:   # the row and (for a column of the padding, the last) element of this value
                        lines.append(f"      const int element = (lr0 + 64 * tile + g + 8 * (i >> 1)) * kSlabWords"
                                     f" + min(8 * nt + 2 * t + (i & 1), {stage.size - 1});")
                        if far_sides:
                            lines.append(f"      const int far_element = (lr0 + 64 * tile + g + 8 * (i >> 1)) * kFarWords"
                                         f" + min(8 * nt + 2 * t + (i & 1), {stage.size - 1});")
                    lines += ["      " + line for line in self.map_lines(chain.maps[link + 1], "value", "element")]
                    if stage.size % 8:
                        lines.append(f"      if (8 * nt + 2 * t + (i & 1) >= {stage.size}) value = 0.0f;")
                    lines += [
                        "      uint32_t low;",
                        f"      split(value, {ah}[nt][2 * (i & 1) + (i >> 1)], low);",
                        f"      ({image})[((2 * nt + (i & 1)) * 8 + 2 * w + (i >> 1)) * 32 + 4 * g + t] = __uint_as_float(low);",
                        "    }",
                        "  }",
                        "  fence_async();",
                        "  warpgroup_sync(wg);",
                    ]
                else:
                    lines += [
                        "#pragma unroll",
                        f"  for (int nt = 0; nt < {columns // 8}; ++nt) {{   // to the rows' slabs and to the output rows",
                        "#pragma unroll",
                        "    for (int i = 0; i < 4; ++i) {",
                        "      const int column = 8 * nt + 2 * t + (i & 1);",
                        "      const int in_block = lr0 + 64 * tile + g + 8 * (i >> 1);   // the row of this value",
                    ]
                    stored = f"{d}[4 * nt + i]"
                    if chain.trailing:   # the stages behind the sum, up to the value that other code reads
                        lines.append(f"      float value = {d}[4 * nt + i];")
                        if chain.sides:
                            lines.append(f"      const int element = in_block * kSlabWords + min({n0} + column, {stage.size - 1});")
                            if far_sides:
                                lines.append(f"      const int far_element = in_block * kFarWords + min({n0} + column, {stage.size - 1});")
                        lines += ["      " + line for line in self.map_lines(chain.trailing, "value", "element")]
                        stored = "value"
                    if output_runs:
                        lines.append("      const long long to = (long long)blockIdx.x * kRowsPerBlock + in_block;")
                    for runs, target, valid in ((slab_runs, result_rows, ""),
                                                (output_runs, "rows_out[to * kOutputs + {}]", "to < row_count")):
                        for first, last, word in runs:   # the part of the run in this block of elements
                            low, high = max(first, n0) - n0, min(last, elements[1]) - n0
                            if low < high:
                                condition = " && ".join(filter(None, [within(low, high, columns), valid]))
                                shift = word - first + n0   # word or output of column 0
                                place = target.format(f"{shift} + column" if shift > 0 else f"column - {-shift}" if shift else "column")
                                lines.append(f"      {f'if ({condition}) ' if condition else ''}{place} = {stored};")
                    lines += ["    }", "  }"]
        return lines + ["}", "__syncthreads();"]
