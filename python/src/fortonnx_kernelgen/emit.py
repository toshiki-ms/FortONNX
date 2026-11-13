"""CUDA source of the kernel for one row graph.

One thread evaluates one row. The threads of a block run the same statements on different rows, so every constant
(weights, tables) is the same for all of them. Layout of the generated kernel:

- the row's input elements are loaded into scalars;
- each group of the plan (schedule.py) follows in order: straight-line code, or ``for (int m = 0; m < members; ++m)``
  whose body computes the group's elements of iteration ``m`` as local scalars;
- constants that differ between iterations come from the group's *record* of iteration ``m`` (``rec``: FP32 columns,
  ``irec``: integer columns behind them), which holds them in the order of use; constants equal in all iterations
  are literals. The block copies the records of a loop into shared memory a few iterations at a time and reads them
  there: a load from device memory would stall all rows of the block at every new address;
- values needed outside their loop are written to the row's slab in shared memory (``s``); output elements are
  stored to the output row where they are computed.

Device memory is touched only for the input row, the records and the output row.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace

import numpy as np

from .cforms import (APPROXIMATE_FUNCTIONS, APPROXIMATE_HELPERS, affine as _affine, comment_text, float_literal,
                     index_text as _index)
from .rowgraph import MAP_OPERATORS, RowGraph, Stage, ref_offset, ref_stage
from .schedule import Group, Limits, Plan, assign_storage
from .tensor import (CHUNK, HELPERS as TENSOR_HELPERS, SLOT_FLOATS, ChainWriter, DenseChain, FragmentWriter, ThreadFragments,
                     mma_helper)

# Operand texts that need no local of their own: names and literals.
_SIMPLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$|^\(?-?[0-9][0-9.e+-]*f\)?$|^\(-INFINITY\)$")
_LANE = "xyzw"
# Rows per block that can always be launched: a thread may use up to 255 registers of the 65536 of one SM.
_SAFE_THREADS = 256


def _ranges_text(numbers: np.ndarray) -> str:
    """'0-3, 7, 9-10' for sorted integers."""
    runs = np.split(numbers, np.flatnonzero(np.diff(numbers) != 1) + 1)
    return ", ".join(str(run[0]) if len(run) == 1 else f"{run[0]}-{run[-1]}" for run in runs)


class KernelTooLarge(ValueError):
    """The per-row values do not fit the shared memory of one block."""


@dataclass
class Options:
    """Settings of the generated kernel.

    threads_per_block    rows per block; None chooses as many as the shared memory holds (a multiple of 32 up to
                         256 when at least 32 fit), and 96 for a kernel without per-row shared memory. A larger
                         count must also fit the registers of one SM (65536 / registers per thread), which only
                         the compiler reports
    shared_memory_bytes  shared memory available to one block (Hopper: 227 KiB)
    stage_bytes          shared memory for the records of a loop: as many iterations as fit are copied at a time
                         (at least one); 0 reads the records from device memory instead
    register_hoist       most slab values one loop may copy into registers before it starts
    slab_dot_terms       a product sum over at least this many consecutive slab values is written as a loop
    dot_lanes            product sums of one stage that advance together (independent FMA chains)
    member_patterns      a loop whose iterations differ in which elements are read or which operands are constants
                         (padded slots) gets one body per pattern, chosen by a switch, up to this many; more
                         patterns share one body that computes every element some iteration reads
    gathered_stores      in a tensor-core loop, the threads of a warp store consecutive output elements of
                         their rows together (_Emitter.gathered_stores) instead of each its own row's: far fewer
                         memory sectors per store. The compiler then needs more registers per thread (the reason
                         for two warpgroups by default: 256 threads may use all 255 registers)
    reciprocal_division  write a division by a constant as a product with its FP32 reciprocal. This changes the
                         rounding of the quotient (by at most one unit in the last place); a division by a power
                         of two is always written as a product, which is exact.
    approximate_functions  compute exp, log, tanh and quotients with the hardware's approximate instructions
                         (APPROXIMATE_FUNCTIONS below gives the forms and their errors) and divisions by constants
                         as products with reciprocals. The results differ from the graph's by rounding-size errors;
                         the exporter's accurate log1p is recognised and stays accurate.
    tensor_cores         product sums at the head of loop bodies on the tensor cores (wgmma, sm_90a; tensor.py):
                         3xTF32 products with FP32 accumulation, 128 rows per warpgroup
    warpgroups           most warpgroups per block of a kernel that uses the tensor cores (it has as many as the
                         shared memory holds)
    output_parts         with tensor_cores: the outputs in parts that share their product sums (schedule.output_parts),
                         one kernel per part, each for the whole output row ("on"); never ("off"); or ("auto") where
                         that brings product sums of loops onto the tensor cores (generate_kernel)
    split_kernels        a loop on the tensor cores with its iterations by thread becomes a kernel of its own,
                         behind the kernel of everything ahead of it (schedule.split_before_thread_loop)
    slab                 where the rows keep their values between loops: "shared" memory of the block, "device"
                         memory of the kernel object for the stages that keep much (Limits.far_words; any graph then
                         fits, with full blocks of rows; the small values stay in shared memory), or "auto"
                         (emit_kernel): shared memory unless the kernel does not fit there or would have less
                         than a quarter of its rows per block (half, with all its loops on the tensor cores)
    tiles                most tiles of 64 rows that a warpgroup owns: two, one after the other (a kernel has one
                         where the shared memory does not hold the rows of two); loop_tiles: of that second kernel
    loop_warpgroups      warpgroups per block of that second kernel (it holds no wide product sums: with few
                         enough registers per thread, three warpgroups run)
    """

    threads_per_block: int | None = None
    shared_memory_bytes: int = 227 * 1024
    stage_bytes: int = 16 * 1024
    register_hoist: int = 64
    slab_dot_terms: int = 16
    dot_lanes: int = 8
    reciprocal_division: bool = False
    member_patterns: int = 16
    gathered_stores: bool = True
    approximate_functions: bool = False
    tensor_cores: bool = False
    warpgroups: int = 2
    tiles: int = 2
    loop_tiles: int = 2
    slab: str = "auto"
    split_kernels: bool = True
    loop_warpgroups: int = 3
    output_parts: str = "auto"
    limits: Limits = field(default_factory=Limits)


# The slab of one row of a kernel that keeps much per row: word by word as one array.
ROW_OF_TWO_PARTS = """
// The slab of one row: its first kFarWords words in device memory, the others in shared memory.
struct Row {
  float* near_words;
  float* far_words;
  __device__ __forceinline__ float& operator[](int word) const {
    return word < kFarWords ? far_words[word] : near_words[word - kFarWords];
  }
};
"""


class _Records:
    """Constants of a kernel as one array of 32-bit words: per group, a matrix [members, stride] whose rows hold
    the FP32 columns (padded to a multiple of 4) followed by the integer columns (padded likewise)."""

    def __init__(self):
        self.parts: list[np.ndarray] = []
        self.words = 0

    @staticmethod
    def matrix(float_columns: list[np.ndarray], int_columns: list[np.ndarray], members: int, align: int = 4):
        """([members, stride] words, words of the FP32 part) of one group's records; the stride is a multiple of
        ``align`` words."""
        float_words = -(-len(float_columns) // 4) * 4
        stride = -(-(float_words + -(-len(int_columns) // 4) * 4) // align) * align
        matrix = np.zeros((members, stride), dtype=np.uint32)
        if float_columns:
            matrix[:, :len(float_columns)] = np.stack(float_columns, axis=1).astype(np.float32).view(np.uint32)
        if int_columns:
            integers = np.stack(int_columns, axis=1).astype(np.int32)
            matrix[:, float_words:float_words + len(int_columns)] = integers.view(np.uint32)
        return matrix, float_words

    def append(self, matrix: np.ndarray) -> int:
        """Append records [count, stride]; returns their word offset."""
        offset = self.words
        if matrix.size:
            self.parts.append(np.ascontiguousarray(matrix, dtype=np.uint32).reshape(-1))
            self.words += matrix.size
        return offset

    def add(self, float_columns: list[np.ndarray], int_columns: list[np.ndarray], members: int, align: int = 4):
        """Append one group's records; returns (word offset, stride, words of the FP32 part)."""
        matrix, float_words = self.matrix(float_columns, int_columns, members, align)
        return self.append(matrix), matrix.shape[1], float_words


class _Body:
    """Statements of one group and the columns of its records."""

    def __init__(self, emitter: "_Emitter", position, group: Group, tensor: bool = False, var: str = "m"):
        self.emitter = emitter
        self.graph = emitter.graph
        self.position = position
        self.members = group.members
        self.var = var                       # name of the loop variable
        self.local = set(group.stages)
        self.lines: list[str] = []
        self.before: list[str] = []          # statements ahead of the loop (values copied from the slab, sums)
        self.after: list[str] = []           # statements behind the loop (stores of accumulated sums)
        self.float_columns: list[np.ndarray] = []
        self.int_columns: list[np.ndarray] = []
        self.cache: dict[bytes, str] = {}
        self.hoisted: dict[int, set[int]] = {}   # slab stage -> elements this group copied into registers
        self.rec = f"rec{position}"
        self.irec = f"irec{position}"
        # a loop reads its records from shared memory (see group_lines), straight-line code from device memory
        # (the tensor cores read their weights from shared memory in any case)
        self.select: np.ndarray | None = None   # iterations the statements are written for (None: all)
        self.gathered: dict[int, tuple[int, list[int]]] = {}   # stage -> slab word and elements of gathered stores
        self.by_thread = False               # the statements run for the two rows of a tile that the thread holds
        self.store_guard = ""                # condition of the slab stores (rows of one tile only), or none
        self.orders = _Orders()              # what is known about the order of the values of minima and maxima
        self.anchors: dict[str, list] = {}   # per use ("slab", "out"): integers of the record already in locals
        self.staged = group.members > 1 and (emitter.options.stage_bytes > 0 or tensor)
        self.align = 4   # words to which the record's blocks and length are aligned
        self.load = "load_shared" if self.staged else "load_constants"

    def chosen(self, array: np.ndarray, read: np.ndarray | None = None) -> np.ndarray:
        """[members, ...] with the entries of the iterations that do not count replaced by those of the first that
        does: what holds for the iterations that count then holds for all, and the columns of the records have a
        valid entry everywhere. An iteration counts when the statements are written for it (``select``) and, with
        ``read``, when it reads the element whose operands these are."""
        keep = self.select
        if read is not None:
            keep = read if keep is None else keep & read
        if keep is None or keep.all() or not keep.any():
            return array
        out = np.array(array)
        out[~keep] = out[int(np.flatnonzero(keep)[0])]
        return out

    # ---- record columns --------------------------------------------------------------------------------------

    def float_column(self, values: np.ndarray) -> int:
        self.float_columns.append(np.asarray(values, dtype=np.float32))
        return len(self.float_columns) - 1

    def float4_columns(self, values: np.ndarray) -> int:
        """Four consecutive columns on a 16-byte boundary ([members, <= 4] values); returns the float4 index."""
        while len(self.float_columns) % 4:
            self.float_columns.append(np.zeros(self.members, dtype=np.float32))
        start = len(self.float_columns)
        for lane in range(4):
            column = values[:, lane] if lane < values.shape[1] else np.zeros(self.members, dtype=np.float32)
            self.float_columns.append(np.asarray(column, dtype=np.float32))
        return start // 4

    def index(self, base: int, step: int) -> str:
        """Text of ``base + step * (loop variable)``."""
        return _index(base, step).replace("m", self.var)

    def float_block(self, values: np.ndarray, align: int) -> int:
        """Consecutive columns [members, size] starting on a multiple of ``align`` words; returns the first."""
        while len(self.float_columns) % align:
            self.float_columns.append(np.zeros(self.members, dtype=np.float32))
        start = len(self.float_columns)
        self.float_columns += [np.asarray(values[:, i], dtype=np.float32) for i in range(values.shape[1])]
        self.align = max(self.align, align)
        return start

    def outside_value(self, ref) -> str:
        """Text of a row value computed before this group, for statements ahead of the loop."""
        source, element = int(ref_stage(ref)), int(ref_offset(ref))
        plan = self.emitter.plan
        return f"s[{plan.word(source, element)}]" if source in plan.slab_offset else f"v{source}_{element}"

    def int_column(self, values: np.ndarray) -> int:
        self.int_columns.append(np.asarray(values, dtype=np.int64))
        return len(self.int_columns) - 1

    # ---- operands --------------------------------------------------------------------------------------------

    def constant(self, values: np.ndarray, is_bool: bool) -> str:
        """Expression of a constant operand with one value per iteration."""
        if (values.view(np.uint32) == values.view(np.uint32)[0]).all():
            if is_bool:
                return "true" if values[0] else "false"
            return float_literal(values[0])
        column = self.float_column(values)
        return f"({self.rec}[{column}] != 0.0f)" if is_bool else f"{self.rec}[{column}]"

    def slab_value(self, index_text: str, is_bool: bool) -> str:
        return f"(s[{index_text}] != 0.0f)" if is_bool else f"s[{index_text}]"

    def variable(self, stage_of: np.ndarray, offset: np.ndarray, selected: np.ndarray, is_bool: bool) -> str:
        """Expression of an operand that is a computed value in the iterations ``selected``."""
        emitter = self.emitter
        sources = np.unique(stage_of)
        if len(sources) == 1 and int(sources[0]) in self.local:
            source = self.graph.stages[int(sources[0])]
            inner = offset % (source.size // self.members)
            if not (inner == inner[0]).all():
                raise AssertionError(f"stage {source.name}: position inside the loop depends on the iteration")
            return f"v{source.index}_{int(inner[0])}"
        same = len(sources) == 1 and selected.all() and (offset == offset[0]).all()
        if same:
            source, element = int(sources[0]), int(offset[0])
            if source not in emitter.plan.slab_offset:
                assert not self.by_thread, f"stage {source} is not in the slab of the row"
                return f"v{source}_{element}"
            word = int(emitter.plan.word(source, element))
            if source in emitter.plan.ring:   # written by the loop that this one runs inside: read where it is
                word = emitter.plan.slab_offset[source] + (element & (emitter.plan.ring[source] - 1))
                return self.slab_value(str(word), is_bool)
            copied = self.hoisted.setdefault(source, set())
            hoist = self.members > 1 and not self.by_thread
            if hoist and (element in copied or len(copied) < emitter.options.register_hoist):
                name = f"h{self.position}_{source}_{element}"
                if element not in copied:
                    copied.add(element)
                    kind = "bool" if is_bool else "float"
                    self.before.append(f"const {kind} {name} = {self.slab_value(str(word), is_bool)};")
                return name
            return self.slab_value(str(word), is_bool)
        missing = [int(s) for s in sources if int(s) not in emitter.plan.slab_offset]
        if missing:
            raise AssertionError(f"stages {missing} are read at iteration-dependent positions but have no slab range")
        base = np.array([emitter.plan.slab_offset[int(s)] for s in stage_of], dtype=np.int64)
        ring = np.array([emitter.plan.ring.get(int(s), 0) for s in stage_of], dtype=np.int64)
        words = np.zeros(self.members, dtype=np.int64)
        words[selected] = base + np.where(ring > 0, offset & (ring - 1), offset)
        some = [int(s) for s in sources if int(s) in emitter.plan.kept]   # stages with only some elements in the slab
        for source in some:
            of_source = stage_of == source
            words[np.flatnonzero(selected)[of_source]] = emitter.plan.word(source, offset[of_source])
        affine = _affine(words) if selected.all() else None
        if affine is not None:
            return self.slab_value(self.index(*affine), is_bool)
        if selected.all() and len(sources) == 1 and not some:
            name, constant = self.anchored(offset, f"slab {int(sources[0])}")
            if ring[0]:
                return self.slab_value(f"{int(base[0])} + (({name} + {constant}) & {int(ring[0]) - 1})", is_bool)
            return self.slab_value(f"{int(base[0]) + constant} + {name}", is_bool)
        return self.slab_value(f"{self.irec}[{self.int_column(words)}]", is_bool)

    def anchored(self, values: np.ndarray, use: str) -> tuple[str, int]:
        """(name of an integer local, constant) with ``values`` = local + constant in every iteration. Positions
        that differ between the iterations come from the record; the positions of the elements of one iteration
        mostly differ by constants (a band of consecutive values): one integer is loaded, the others follow."""
        for name, anchor in self.anchors.setdefault(use, []):
            difference = values - anchor
            if (difference == difference[0]).all():
                return name, int(difference[0])
        name = f"i{self.emitter.next_name()}"
        self.lines.append(f"const int {name} = {self.irec}[{self.int_column(values)}];")
        self.anchors[use].append((name, np.array(values)))
        return name, 0

    def access(self, refs: np.ndarray) -> str:
        """Name or literal of one operand; ``refs`` holds its reference in every iteration of the group."""
        refs = self.chosen(refs)
        key = refs.tobytes()
        if key in self.cache:
            return self.cache[key]
        stage_of, offset = ref_stage(refs), ref_offset(refs)
        const = self.emitter.is_const[stage_of]
        is_bool = self.graph.stages[int(stage_of[0])].is_bool
        constant_text = None
        if const.any():
            values = np.zeros(self.members, dtype=np.float32)
            values[const] = self.graph.const_values(refs[const])
            if not const.all():
                values[~const] = values[const][0]
            constant_text = self.constant(values, is_bool)
        if const.all():
            text = constant_text
        else:
            text = self.variable(stage_of[~const], offset[~const], ~const, is_bool)
            if const.any():
                mask = self.int_column((~const).astype(np.int64))
                text = f"({self.irec}[{mask}] ? {text} : {constant_text})"
        if not _SIMPLE.match(text):
            name = f"a{self.emitter.next_name()}"
            self.lines.append(f"const {'bool' if is_bool else 'float'} {name} = {text};")
            text = name
        self.cache[key] = text
        return text

    # ---- stages ----------------------------------------------------------------------------------------------

    def live_slots(self, stage: Stage) -> list[int]:
        live = self.emitter.plan.live[stage.index].reshape(self.members, -1)
        if self.select is not None:
            live = live[self.select]
        return [int(j) for j in np.flatnonzero(live.any(axis=0))]

    def per_member(self, array: np.ndarray, slot: int, stage: Stage | None = None) -> np.ndarray:
        """[members, ...] part of a per-element array for the element ``slot`` of every iteration. With ``stage``
        (the array holds operands of that stage's element): an iteration in which nobody reads the element takes the
        entry of one that does, so that no operand of an unread iteration (possibly a value that is not computed
        at all) is referenced."""
        read = None
        if stage is not None and self.members > 1:
            read = self.emitter.plan.live[stage.index].reshape(self.members, -1)[:, slot]
        return self.chosen(array.reshape((self.members, -1) + array.shape[1:])[:, slot], read)

    def emit_stages(self, stages: list[int], accumulated: list[int]) -> None:
        """Write the statements of stages of the group in order (independent statements of one stage side by side),
        then the sums that the loop accumulates."""
        for index in stages:
            stage = self.graph.stages[index]
            operator = stage.op if stage.kind != "dot" else "sum of products"
            self.lines.append(f"// {stage.name}: {operator}, {stage.size // self.members} per iteration"
                              if self.members > 1 else f"// {stage.name}: {operator}")
            for emit in self.units(stage):
                emit()
        for index in accumulated:
            stage = self.graph.stages[index]
            self.lines.append(f"// {stage.name}: sum of products over the iterations, {stage.size} sums")
            self.accumulate(stage)

    def units(self, stage: Stage) -> list:
        """Functions that write the statements of the stage's elements of one iteration, in order."""
        slots = self.live_slots(stage)
        if stage.kind == "map":
            return [lambda slot=slot: self.map_element(stage, slot) for slot in slots]
        if stage.kind == "reduce":
            return [lambda slot=slot: self.reduce_element(stage, slot) for slot in slots]
        options = self.emitter.options
        units = []
        pending = []
        for slot in slots:
            run = None
            if stage.weights is not None and stage.terms >= options.slab_dot_terms:
                run = self.slab_run(self.per_member(stage.operands[0], slot, stage))
            if run is None:
                pending.append(slot)
            else:
                units.append(lambda slot=slot, run=run: self.dot_over_slab(stage, slot, run))
        for start in range(0, len(pending), options.dot_lanes):
            lanes = pending[start:start + options.dot_lanes]
            units.append(lambda lanes=lanes: self.dot_lanes(stage, lanes))
        return units

    def map_element(self, stage: Stage, slot: int) -> None:
        template = MAP_OPERATORS[stage.op][1]
        operands = [np.ascontiguousarray(self.per_member(refs, slot, stage)) for refs in stage.operands]
        arguments = [self.access(refs) for refs in operands[:1]]
        reciprocal = self.reciprocal(operands[1]) if stage.op == "div" else None
        if reciprocal is not None:
            template = MAP_OPERATORS["mul"][1]
            arguments.append(self.constant(reciprocal, False))
            arguments += [self.access(refs) for refs in operands[2:]]
        else:
            if self.emitter.options.approximate_functions:
                template = APPROXIMATE_FUNCTIONS.get(stage.op, template)
            arguments += [self.access(refs) for refs in operands[1:]]
        kind = "bool" if stage.is_bool else "float"
        value = template.format(*arguments)
        # A minimum or maximum whose result is always one of its operands is that operand: with -inf or +inf (the
        # pad of an iteration with fewer elements), or because the comparisons before it have put the two in order
        # (the later rounds of a sorting network on fewer elements). An approximate kernel takes computed values
        # as finite numbers.
        if stage.op in ("max", "min") and self.emitter.options.approximate_functions:
            value = self.orders.result(stage.op, f"v{stage.index}_{slot}", arguments) or value
        self.lines.append(f"const {kind} v{stage.index}_{slot} = {value};")
        self.finish(stage, slot)

    def reduce_element(self, stage: Stage, slot: int) -> None:
        step = {"sum": "{0} + {1}", "prod": "{0} * {1}", "max": MAP_OPERATORS["max"][1], "min": MAP_OPERATORS["min"][1]}
        if self.emitter.options.approximate_functions:
            step.update({op: APPROXIMATE_FUNCTIONS[op] for op in ("max", "min")})
        terms = self.per_member(stage.operands[0], slot, stage)
        name = f"v{stage.index}_{slot}"
        arguments = [self.access(np.ascontiguousarray(terms[:, k])) for k in range(stage.terms)]
        if stage.op == "logsumexp":
            self.logsumexp(stage, name, arguments)
            self.finish(stage, slot)
            return
        # terms without effect: -inf in a maximum, +inf in a minimum; in an approximate kernel also the zeros of a
        # sum (the sign of a zero sum may then differ)
        neutral = {"max": "(-INFINITY)", "min": "INFINITY"}
        if self.emitter.options.approximate_functions:
            neutral["sum"] = "0.0f"
        arguments = [a for a in arguments if a != neutral.get(stage.op)] or arguments[:1]
        self.lines.append(f"float {name} = {arguments[0]};")
        for argument in arguments[1:]:
            self.lines.append(f"{name} = {step[stage.op].format(name, argument)};")
        self.finish(stage, slot)

    def logsumexp(self, stage: Stage, name: str, arguments: list[str]) -> None:
        """m + log(sum(exp(t - m))), m the largest term. With the base-2 instructions of an approximate kernel the
        factors log2 e and ln 2 are written only where the neighbouring constants have not taken them."""
        largest = f"m{self.emitter.next_name()}"
        self.lines.append(f"float {largest} = {arguments[0]};")
        self.lines += [f"{largest} = fmaxf({largest}, {argument});" for argument in arguments[1:]]
        if not self.emitter.options.approximate_functions:
            total = " + ".join(f"expf({argument} - {largest})" for argument in arguments)
            self.lines.append(f"const float {name} = {largest} + logf({total});")
            return
        terms_in_log2, result_in_log2 = stage.log2_units
        scale = "" if terms_in_log2 else " * 1.44269504f"
        total = " + ".join(f"approx_ex2(({argument} - {largest}){scale})" for argument in arguments)
        if terms_in_log2 == result_in_log2:
            unit = "" if result_in_log2 else " * 0.693147181f"
            self.lines.append(f"const float {name} = {largest} + approx_lg2({total}){unit};")
        elif result_in_log2:
            self.lines.append(f"const float {name} = {largest} * 1.44269504f + approx_lg2({total});")
        else:
            self.lines.append(f"const float {name} = ({largest} + approx_lg2({total})) * 0.693147181f;")

    def accumulate(self, stage: Stage) -> None:
        """A sum over the iterations of a stage of this loop (schedule.accumulate_in_loops): the sums start before
        the loop, every iteration adds its terms, the stores follow the loop."""
        source = self.graph.stages[int(ref_stage(stage.operands[0])[0, 0])]
        inner = source.size // self.members
        elements = [int(e) for e in np.flatnonzero(self.emitter.plan.live[stage.index])]
        weights = np.zeros((stage.size, source.size), dtype=np.float32)
        is_term = np.zeros((stage.size, source.size), dtype=bool)
        at = (np.arange(stage.size)[:, None], ref_offset(stage.operands[0]))
        weights[at] = 1.0 if stage.weights is None else stage.weights   # a sum without factors: ReduceSum
        is_term[at] = True
        for element in elements:
            bias = 0.0 if stage.bias is None else stage.bias[element]
            self.before.append(f"float v{stage.index}_{element} = {float_literal(bias)};")
            self.finish_scalar(stage, element)
        for slot in range(inner):
            adding = [element for element in elements if is_term[element, slot]]   # the sums this element is a term of
            plain = [element for element in adding if (weights[element, slot::inner] == 1.0).all()]
            for element in plain:
                self.lines.append(f"v{stage.index}_{element} = v{stage.index}_{element} + v{source.index}_{slot};")
            scaled = [element for element in adding if element not in plain]
            for start in range(0, len(scaled), 4):
                quad = scaled[start:start + 4]
                factors = np.stack([weights[e, slot::inner] for e in quad], axis=1)   # [members, <= 4]
                name = self.load_float4(factors)
                for lane, element in enumerate(quad):
                    target = f"v{stage.index}_{element}"
                    self.lines.append(f"{target} = fmaf(v{source.index}_{slot}, {name}.{_LANE[lane]}, {target});")

    def finish_scalar(self, stage: Stage, element: int) -> None:
        """Stores after the loop of an element that the loop accumulated."""
        emitter = self.emitter
        name = f"v{stage.index}_{element}"
        if stage.index in emitter.plan.slab_offset:
            self.after.append(f"s[{emitter.plan.slab_offset[stage.index] + element}] = {name};")
        positions = emitter.output_positions.get(stage.index)
        if positions is not None:
            self.after += [f"if (active) out[{o}] = {name};" for o in positions[element] if o >= 0]

    def reciprocal(self, refs: np.ndarray) -> np.ndarray | None:
        """FP32 reciprocals of a constant divisor (one per iteration) when the division may be written as a
        product, else None."""
        if not self.emitter.is_const[ref_stage(refs)].all():
            return None
        return self.reciprocal_values(self.graph.const_values(refs))

    def reciprocal_values(self, values: np.ndarray) -> np.ndarray | None:
        """FP32 reciprocals of constant divisors when the division may be written as a product: always for powers
        of two (exact), for other constants with ``reciprocal_division`` or ``approximate_functions``."""
        with np.errstate(all="ignore"):
            reciprocals = (np.float32(1.0) / values).astype(np.float32)
        if not (np.isfinite(reciprocals).all() and (reciprocals != 0).all()):
            return None   # the FP32 reciprocal overflows or vanishes: keep the division
        mantissa, _ = np.frexp(values)
        options = self.emitter.options
        if not (np.abs(mantissa) == 0.5).all() and not (options.reciprocal_division or options.approximate_functions):
            return None
        return reciprocals

    def bias_text(self, stage: Stage, slot: int) -> str:
        if stage.bias is None:
            return "0.0f"
        return self.constant(np.ascontiguousarray(self.per_member(stage.bias, slot, stage)), False)

    def slab_run(self, terms: np.ndarray) -> tuple[int, int] | None:
        """(first word, step per iteration) when the terms [members, count] are consecutive values of one slab
        range that starts on a 16-byte boundary."""
        stage_of, offset = ref_stage(terms), ref_offset(terms)
        source = int(stage_of[0, 0])
        if (stage_of != source).any() or source not in self.emitter.plan.slab_offset or source in self.local:
            return None
        if source in self.emitter.plan.kept:
            return None
        if source in self.emitter.plan.ring:
            return None
        if not np.array_equal(offset, offset[:, :1] + np.arange(terms.shape[1])):
            return None
        first = _affine(self.emitter.plan.slab_offset[source] + offset[:, 0])
        if first is None or first[0] % 4 or first[1] % 4:
            return None
        return first

    def dot_lanes(self, stage: Stage, lanes: list[int]) -> None:
        """Product sums of a few elements, advancing together term by term."""
        first = stage.operands[0]
        second = stage.operands[1] if stage.weights is None else None
        quads = [lanes[i:i + 4] for i in range(0, len(lanes), 4)]
        for quad in quads:
            self.dot_start(stage, quad)
        for k in range(stage.terms):
            factor = {}
            if stage.weights is not None:
                columns = {slot: np.ascontiguousarray(self.per_member(stage.weights, slot, stage)[:, k]) for slot in lanes}
                if len({column.tobytes() for column in columns.values()}) < len(lanes):
                    # elements that share factors (copies of one sum on different values): each factor is one
                    # column of the record and is loaded once
                    factor = {slot: self.shared_factor(column) for slot, column in columns.items()}
                else:
                    for quad in quads:
                        name = self.load_float4(np.stack([columns[slot] for slot in quad], axis=1))
                        factor.update({slot: f"{name}.{_LANE[lane]}" for lane, slot in enumerate(quad)})
            for slot in lanes:
                value = self.access(np.ascontiguousarray(self.per_member(first, slot, stage)[:, k]))
                other = factor.get(slot)
                if other is None:
                    other = self.access(np.ascontiguousarray(self.per_member(second, slot, stage)[:, k]))
                target = f"v{stage.index}_{slot}"
                self.lines.append(f"{target} = fmaf({value}, {other}, {target});")
        for slot in lanes:
            self.finish(stage, slot)

    def load_float4(self, values: np.ndarray) -> str:
        """Statement that loads [members, <= 4] constants with one 16-byte read; returns the float4's name."""
        name = f"w{self.emitter.next_name()}"
        self.lines.append(f"const float4 {name} = {self.load}({self.rec}q + {self.float4_columns(values)});")
        return name

    def shared_factor(self, column: np.ndarray) -> str:
        """Name or literal of a constant factor with one value per iteration that several statements use."""
        key = b"factor" + column.tobytes()
        if key not in self.cache:
            text = self.constant(column, False)
            if not _SIMPLE.match(text):
                name = f"a{self.emitter.next_name()}"
                self.lines.append(f"const float {name} = {text};")
                text = name
            self.cache[key] = text
        return self.cache[key]

    def dot_start(self, stage: Stage, quad: list[int]) -> None:
        """Declare the sums of up to four elements, starting from their biases."""
        names = [f"v{stage.index}_{slot}" for slot in quad]
        if stage.bias is None:
            self.lines += [f"float {name} = 0.0f;" for name in names]
            return
        bias = np.stack([self.per_member(stage.bias, slot, stage) for slot in quad], axis=1)
        bias = np.ascontiguousarray(bias, dtype=np.float32)
        if (bias.view(np.uint32) == bias.view(np.uint32)[0]).all():
            self.lines += [f"float {name} = {float_literal(value)};" for name, value in zip(names, bias[0])]
            return
        loaded = self.load_float4(bias)
        self.lines += [f"float {name} = {loaded}.{_LANE[lane]};" for lane, name in enumerate(names)]

    def dot_over_slab(self, stage: Stage, slot: int, run: tuple[int, int]) -> None:
        """Product sum over consecutive slab values: 16-byte loads of values and weights, four partial sums."""
        count = stage.terms
        quads = count // 4
        weights = self.per_member(stage.weights, slot, stage)
        first_quad = None
        for q in range(quads):
            index = self.float4_columns(weights[:, 4 * q:4 * q + 4])
            first_quad = index if first_quad is None else first_quad
        tail = [self.float_column(weights[:, k]) for k in range(4 * quads, count)]
        name = f"v{stage.index}_{slot}"
        # the values of one stage: in one part of a row whose slab has two (the address of their first word)
        values = f"&s[{self.index(*run)}]" if self.emitter.plan.far_words else f"s + {self.index(*run)}"
        self.lines += [
            f"float {name};",
            "{",
            f"  const float4* const values = (const float4*)({values});",
            f"  const float4* const weights = {self.rec}q + {first_quad};",
            f"  float t0 = {self.bias_text(stage, slot)}, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;",
            f"  for (int q = 0; q < {quads}; ++q) {{",
            "    const float4 x = load_shared(values + q);",
            f"    const float4 w = {self.load}(weights + q);",
            "    t0 = fmaf(x.x, w.x, t0);",
            "    t1 = fmaf(x.y, w.y, t1);",
            "    t2 = fmaf(x.z, w.z, t2);",
            "    t3 = fmaf(x.w, w.w, t3);",
            "  }",
        ]
        for lane, column in enumerate(tail):
            word = self.index(run[0] + 4 * quads + lane, run[1])
            self.lines.append(f"  t{lane} = fmaf(s[{word}], {self.rec}[{column}], t{lane});")
        self.lines += [f"  {name} = (t0 + t1) + (t2 + t3);", "}"]
        self.finish(stage, slot)

    def finish(self, stage: Stage, slot: int) -> None:
        """Stores of a computed element: to the slab when other groups read it, and to the output row."""
        emitter = self.emitter
        name = f"v{stage.index}_{slot}"
        inner = stage.size // self.members
        value = name + " ? 1.0f : 0.0f" if stage.is_bool else name
        guard = f"if ({self.store_guard}) " if self.store_guard else ""
        if stage.index in emitter.plan.kept:   # only the elements that other code reads go to the slab
            elements = np.arange(self.members) * inner + slot
            is_kept = np.isin(elements, emitter.plan.kept[stage.index])
            words = self.chosen(np.where(is_kept, emitter.plan.word(stage.index, elements), -1))
            if (words >= 0).all() and _affine(words) is not None:
                self.lines.append(f"{guard}s[{self.index(*_affine(words))}] = {value};")
            elif (words >= 0).any():
                local = f"i{emitter.next_name()}"
                self.lines += [f"const int {local} = {self.irec}[{self.int_column(words)}];",
                               f"if ({local} >= 0{' && ' + self.store_guard if self.store_guard else ''}) s[{local}] = {value};"]
        elif stage.index in emitter.plan.slab_offset:
            word = self.index(emitter.plan.slab_offset[stage.index] + slot, inner if self.members > 1 else 0)
            if stage.index in emitter.plan.ring:   # a ring: the loop that reads it runs inside this one
                word = f"{emitter.plan.slab_offset[stage.index]} + ({self.var} & {emitter.plan.ring[stage.index] - 1})"
            self.lines.append(f"{guard}s[{word}] = {value};")
        positions = emitter.output_positions.get(stage.index)
        if positions is None:
            return
        if stage.index in self.gathered:   # stored by the warp together (_Emitter.gathered_stores)
            word, slots = self.gathered[stage.index]
            self.lines.append(f"s[{word + slots.index(slot)}] = {name};")
            return
        for column in self.chosen(positions.reshape(self.members, inner, -1)[:, slot]).T:
            if (column < 0).all():
                continue
            affine = _affine(column) if (column >= 0).all() else None
            if affine is not None:
                if self.members == 1 and not self.by_thread:
                    self.lines += emitter.row_output(affine[0], name)
                else:
                    self.lines.append(f"if (active) out[{self.index(*affine)}] = {name};")
            elif (column >= 0).all():
                local, constant = self.anchored(column, "out")
                self.lines.append(f"if (active) out[{local} + {constant}] = {name};")
            else:
                index = f"{self.irec}[{self.int_column(column)}]"
                self.lines.append(f"if (active && {index} >= 0) out[{index}] = {name};")


def _stores_together(lines: list[str]) -> list[str]:
    """The statements of a loop body with its output stores ``if (active) out[..] = ..;`` moved to its end, under
    one condition: one branch for the rows past the last one instead of one per store."""
    stores = [line for line in lines if line.startswith("if (active) out[")]
    if len(stores) < 2:
        return lines
    rest = [line for line in lines if not line.startswith("if (active) out[")]
    return rest + ["if (active) {"] + ["  " + line[len("if (active) "):] for line in stores] + ["}"]


class _Orders:
    """Which of two operands a minimum or maximum always is, for the minima and maxima of one loop body.

    A value built from other values by minima and maxima only is known by its results on all inputs of zeros and
    ones (the 0-1 principle of sorting networks): ``a <= b`` holds for all inputs when it holds for these. Every
    operand that is not such a minimum or maximum is one input; at most ``INPUTS`` of them are followed."""

    INPUTS = 12

    def __init__(self):
        self.tables: dict[str, np.ndarray] = {"INFINITY": np.ones(1 << self.INPUTS, dtype=np.bool_),
                                             "(-INFINITY)": np.zeros(1 << self.INPUTS, dtype=np.bool_)}
        self.inputs = 0

    def table(self, operand: str) -> np.ndarray | None:
        if operand not in self.tables:
            if self.inputs == self.INPUTS or not re.match(r"^[A-Za-z_]", operand):
                return None   # too many inputs, or a literal number: its order with the others is not followed
            self.tables[operand] = (np.arange(1 << self.INPUTS) >> self.inputs & 1).astype(np.bool_)
            self.inputs += 1
        return self.tables[operand]

    def result(self, op: str, name: str, operands: list[str]) -> str | None:
        """The operand that the minimum or maximum ``name`` of the two operands always is, or None."""
        first, second = (self.table(operand) for operand in operands)
        if first is None or second is None:
            return None
        self.tables[name] = first & second if op == "min" else first | second
        if not (first & ~second).any():   # first <= second for all inputs
            return operands[0] if op == "min" else operands[1]
        if not (second & ~first).any():
            return operands[1] if op == "min" else operands[0]
        return None


@dataclass
class _Staging:
    """Statements that make the records of a loop readable: ``start`` and ``ready`` before the loop (around the
    other statements there), ``top`` and ``end`` in its body; ``words`` of the stage buffer are used."""

    top: list[str] = field(default_factory=list)
    words: int = 0
    start: list[str] = field(default_factory=list)
    ready: list[str] = field(default_factory=list)
    end: list[str] = field(default_factory=list)


class _Emitter:
    def __init__(self, plan: Plan, name: str, options: Options, source_note: str, slot_floats: int = SLOT_FLOATS[0]):
        self.plan = plan
        self.graph: RowGraph = plan.graph
        self.name = name
        self.options = options
        self.source_note = source_note
        self.slot_floats = slot_floats   # weights of a chain in shared memory at a time (tensor.ChainWriter)
        self.is_const = self.graph.is_const()
        self.records = _Records()
        self.stage_words = 0
        self.mma_widths: set[int] = set()
        self.tile_image_words = 0   # shared memory per tile for the low parts of A operands (tensor.ThreadFragments)
        self.tiles_planned = 2 if options.tiles >= 2 else 1   # tiles of 64 rows per warpgroup on the tensor cores
        # Outputs of straight-line code by position (value text), when the warp stores them together at the end of
        # the kernel (row_output_lines); None: every thread stores its own row's outputs where they are computed.
        self.row_outputs: dict[int, str] | None = {} if plan.fragments and options.gathered_stores else None
        self.counter = 0
        self.output_positions = self.collect_output_positions()

    def next_name(self) -> int:
        self.counter += 1
        return self.counter

    def slab_rows(self, word: int) -> tuple[str, str, int]:
        """Where a word of the rows' slabs is: (the first row of the block, words per row, the word in it). The
        slabs are in the shared memory of the block ("slab"); of a kernel whose rows keep much (emit_kernel), the
        first ``far_words`` of every row are in device memory ("far"), the others in shared memory."""
        far = self.plan.far_words
        return ("far", "kFarWords", word) if word < far else ("slab", "kSlabWords", word - far)

    def row_output(self, position: int, value: str) -> list[str]:
        """Statements that store an output of straight-line code (a local of the kernel, a slab value or a literal):
        the store itself, or none when the warp stores the outputs together at the end of the kernel."""
        if self.row_outputs is None:
            return [f"if (active) out[{position}] = {value};"]
        self.row_outputs[position] = value
        return []

    def row_output_lines(self) -> list[str]:
        """The outputs of straight-line code, stored by the warp together: a thread that stores its own row's
        elements one by one makes the 32 threads of a warp write to 32 rows per instruction; here consecutive
        lanes write consecutive elements of a row. Runs of outputs that stand next to each other in the row's
        slab are read there; the other values are put into slab words behind the planned ones first."""
        if not self.row_outputs:
            return []
        runs: list[list] = []   # [first output, slab word of its value or None, values]
        for position, value in sorted(self.row_outputs.items()):
            in_slab = re.fullmatch(r"s\[(\d+)\]", value)
            word = int(in_slab.group(1)) if in_slab else None
            last = runs[-1] if runs else None
            joins = last is not None and last[0] + len(last[2]) == position and (
                (word is None and last[1] is None) or (word is not None and last[1] is not None and last[1] + len(last[2]) == word
                                                       and word != self.plan.far_words))
            if joins:
                last[2].append(value)
            else:
                runs.append([position, word, [value]])
        lines = ["", "// ---- the outputs of the row, stored by the warp together: consecutive lanes consecutive elements"]
        extra = 0
        for run in runs:
            if run[1] is None:   # values of registers and literals: into the row's slab first
                run[1] = self.plan.slab_words + extra
                lines += [f"s[{run[1] + number}] = {value};" for number, value in enumerate(run[2])]
                extra += len(run[2])
        lines.append("__syncwarp();")
        for first, word, values in runs:
            count = len(values)
            rows, stride, word = self.slab_rows(word)
            lines += [
                "{",
                f"  float* const target = warp_out + {first};",
                f"  const float* const source = {rows} + lr0 * {stride} + {word};",
                "#pragma unroll",
                f"  for (int j = 0; j < {count}; ++j) {{",
                f"    const int element = j * 32 + lane, r = element / {count}, column = element - r * {count};",
                "    const int from = ((r >> 4) << 6) + (r & 15);   // row of the warp: its first or second tile",
                f"    if (from < warp_rows) target[from * kOutputs + column] = source[from * {stride} + column];",
                "  }",
                "}",
            ]
        # the slab of a row grows by the words of the copied values (in shared memory: 4 x an odd number of words)
        words = -(-(self.plan.slab_words + extra) // 4) * 4
        near = words - self.plan.far_words
        self.plan.slab_words = words + (4 if near and (near // 4) % 2 == 0 else 0)
        return lines

    def collect_output_positions(self) -> dict[int, np.ndarray]:
        """Per stage, [size, most uses] output positions of every element (-1: none)."""
        stage_of, offset = ref_stage(self.graph.output_refs), ref_offset(self.graph.output_refs)
        result = {}
        for index in np.unique(stage_of):
            index = int(index)
            outputs = np.flatnonzero(stage_of == index)
            elements = offset[outputs]
            if self.graph.output_slots is not None:   # a part of a graph: where its outputs stand in the row
                outputs = self.graph.output_slots[outputs]
            uses = np.bincount(elements, minlength=self.graph.stages[index].size)
            table = np.full((self.graph.stages[index].size, int(uses.max())), -1, dtype=np.int64)
            filled = np.zeros(self.graph.stages[index].size, dtype=np.int64)
            for output, element in zip(outputs, elements):
                table[element, filled[element]] = output
                filled[element] += 1
            result[index] = table
        return result

    def threads_per_block(self, stage_words: int) -> int:
        def left(warpgroups: int) -> int:
            images = 4 * (warpgroups * self.tile_image_words + 32) if self.tile_image_words else 0
            return self.options.shared_memory_bytes - 4 * stage_words - images

        budget = left(self.options.warpgroups)
        slab_bytes = 4 * (self.plan.slab_words - self.plan.far_words)   # of the shared memory
        self.tiles = 0   # tiles of 64 rows per warpgroup of a kernel on the tensor cores
        if self.plan.fragments:
            # A warpgroup of 128 threads owns two tiles of 64 rows, one after the other, or one (emit_kernel);
            # a block has as many warpgroups, at most those of the options, as the shared memory holds.
            self.tiles = self.tiles_planned
            warpgroups = next((n for n in range(self.options.warpgroups, 1, -1)
                               if 64 * self.tiles * n * slab_bytes <= left(n)), 1)
            threads, budget = 128 * warpgroups, left(warpgroups)
            rows = 64 * self.tiles * warpgroups
            if rows * slab_bytes > budget:
                raise KernelTooLarge(
                    f"{self.name}: one row keeps {slab_bytes} bytes in shared memory and the records "
                    f"{4 * stage_words}; {rows} rows per block do not fit {self.options.shared_memory_bytes} bytes")
            return threads
        elif self.options.threads_per_block is not None:
            threads = self.options.threads_per_block
        elif slab_bytes == 0:
            threads = _SAFE_THREADS if self.plan.far_words else 96
        else:
            threads = min(_SAFE_THREADS, budget // slab_bytes // 32 * 32)
            if threads == 0 and budget >= slab_bytes:
                threads = budget // slab_bytes   # fewer than one warp of rows fits
        if threads < 1 or threads * slab_bytes > budget:
            raise KernelTooLarge(
                f"{self.name}: one row keeps {slab_bytes} bytes in shared memory and the records "
                f"{4 * stage_words}; {threads} rows per block do not fit {self.options.shared_memory_bytes} bytes")
        return threads

    def prologue(self) -> list[str]:
        lines = []
        graph = self.graph
        for element in np.flatnonzero(self.plan.live[0]):
            lines.append(f"const float v0_{element} = in[{element}];")
            if 0 in self.plan.slab_offset:
                lines.append(f"s[{self.plan.slab_offset[0] + element}] = v0_{element};")
        for index, table in self.output_positions.items():
            stage = graph.stages[index]
            if stage.kind == "const":
                for element, outputs in enumerate(table):
                    for o in outputs:
                        lines += self.row_output(int(o), float_literal(stage.values[element])) if o >= 0 else []
            elif stage.kind == "input":
                for element, outputs in enumerate(table):
                    for o in outputs:
                        lines += self.row_output(int(o), f"v0_{element}") if o >= 0 else []
        return lines

    def member_patterns(self, group: Group, stages: list[int]) -> list[np.ndarray]:
        """Selections of the iterations of a loop that compute the same elements from the same kinds of operands
        (computed value or constant) and store the same elements. One selection (all iterations) when the loop has
        one such pattern or more than ``Options.member_patterns``, or when other loops are tied to its body."""
        members = group.members
        everything = [np.ones(members, dtype=np.bool_)]
        if members == 1 or group.accumulated or group.streamed or self.options.member_patterns <= 1:
            return everything
        parts = []
        for index in stages:
            stage = self.graph.stages[index]
            live = self.plan.live[index].reshape(members, -1)
            parts.append(live)
            for refs in stage.operands:
                constant = self.is_const[ref_stage(refs)].reshape(members, live.shape[1], -1)
                parts.append((constant & live[:, :, None]).reshape(members, -1))
            positions = self.output_positions.get(index)
            if positions is not None:
                parts.append((positions >= 0).reshape(members, -1))
        keys = [row.tobytes() for row in np.concatenate(parts, axis=1)] if parts else []
        distinct = sorted(set(keys), key=keys.index)
        if not 1 < len(distinct) <= self.options.member_patterns:
            return everything
        return [np.array([key == pattern for key in keys]) for pattern in distinct]

    def write_body(self, body: _Body, group: Group, stages: list[int], accumulated: list[int], writer=None,
                   phase=None) -> None:
        """Statements of the stages of a loop body: one body, or one per pattern of the iterations (which
        elements are read, which operands are constants), chosen by a switch."""
        patterns = self.member_patterns(group, stages)
        if len(patterns) == 1:
            start = len(body.lines)
            body.lines += writer.stored_values() if writer else []
            stores = self.gathered_stores(body, phase, stages)
            body.emit_stages(stages, accumulated)
            body.lines[start:] = _stores_together(body.lines[start:]) + stores
            return
        number = np.zeros(group.members, dtype=np.int64)
        for case, select in enumerate(patterns):
            number[select] = case
        body.lines.append(f"switch ({body.irec}[{body.int_column(number)}]) {{")
        common = dict(body.cache)
        for case, select in enumerate(patterns):
            body.select, body.cache, body.orders, body.anchors = select, dict(common), _Orders(), {}
            start = len(body.lines)
            body.lines += writer.stored_values() if writer else []
            stores = self.gathered_stores(body, phase, stages)
            body.emit_stages(stages, [])
            body.lines[start:] = _stores_together(body.lines[start:]) + stores
            body.lines[start:] = ([f"case {case}: {{   // iterations {_ranges_text(np.flatnonzero(select))}"]
                                  + ["  " + line for line in body.lines[start:]] + ["} break;"])
        body.select, body.cache, body.anchors = None, common, {}
        body.lines.append("}")

    def gathered_stores(self, body: _Body, phase, stages: list[int]) -> list[str]:
        """Output stores of a tensor-core loop body that the warp makes together. A thread that stores its own
        row's elements one by one makes the 32 threads of a warp write to 32 rows at once; when the elements of an
        iteration are consecutive in the output row, the warp instead puts them into the rows' slabs and every
        store instruction writes consecutive elements of a few rows (32 / count of them). Sets ``body.gathered``
        for the stages stored this way and returns the statements that follow the body."""
        body.gathered = {}
        lines: list[str] = []
        exported = phase.exported if phase and self.options.gathered_stores else []
        free = {index: phase.scratch_offset[index] for index in exported}
        for index in stages:
            positions = self.output_positions.get(index)
            stage = self.graph.stages[index]
            slots = body.live_slots(stage)
            room = [e for e in free if phase.inner[e] >= len(slots)]
            if positions is None or positions.shape[1] != 1 or len(slots) < 2 or not room:
                continue
            table = body.chosen(positions.reshape(body.members, -1))[:, slots]
            if (table[:, 0] < 0).any() or not np.array_equal(table, table[:, :1] + np.arange(len(slots))):
                continue
            word, count = free.pop(room[0]), len(slots)
            body.gathered[index] = (word, slots)
            first = f"{body.irec}[{body.int_column(table[:, 0])}]"
            rows, stride, word = self.slab_rows(word)
            lines += [
                f"// {stage.name}: the warp stores the elements of its rows, consecutive lanes consecutive elements",
                "__syncwarp();",
                "{",
                f"  float* const target = warp_out + {first};",
                f"  const float* const source = {rows} + lr0 * {stride} + {word};",
                "#pragma unroll",
                f"  for (int j = 0; j < {count}; ++j) {{",
                f"    const int element = j * 32 + lane, r = element / {count}, column = element - r * {count};",
                "    const int from = ((r >> 4) << 6) + (r & 15);   // row of the warp: its first or second tile",
                f"    if (from < warp_rows) target[from * kOutputs + column] = source[from * {stride} + column];",
                "  }",
                "}",
            ]
        return lines

    def record_lines(self, body: _Body, group: Group, base: int, tensor: bool, ahead: bool = False) -> "_Staging":
        """How the loop body reaches the record of its iteration; ``base`` is the first free word of the stage
        buffer. ``ahead``: the records may be copied before the loop when the stage buffer holds them all (for a
        loop that runs inside another one, whose warps must not wait for each other)."""
        offset, stride, float_words = self.records.add(body.float_columns, body.int_columns, group.members, body.align)
        rec, irec, var = body.rec, body.irec, body.var
        assert offset % 4 == 0 and stride % 4 == 0 and base % 4 == 0
        quads = stride // 4
        tail = []
        if body.int_columns or any(re.search(rf"\b{rec}\b", line) for line in body.lines):
            tail.append(f"const float* const {rec} = (const float*){rec}q;")
        if body.int_columns:
            tail.append(f"const int* const {irec} = (const int*)({rec} + {float_words});")
        if not stride:
            return _Staging()
        if group.members == 1:
            if 0 < 4 * stride <= self.options.stage_bytes:
                # straight-line code reads its constants from shared memory too: the block copies them first
                body.lines[:] = [line.replace("load_constants(", "load_shared(") for line in body.lines]
                target = f"staged[{base // 4} + i]" if base else "staged[i]"
                top = ["__syncthreads();",
                       f"for (int i = threadIdx.x; i < {quads}; i += kThreadsPerBlock)",
                       f"  {target} = load_constants(constants + {offset // 4} + i);",
                       "__syncthreads();",
                       f"const float4* const {rec}q = staged + {base // 4};"]
                return _Staging(top=top + tail, words=stride)
            return _Staging(top=[f"const float4* const {rec}q = constants + {offset // 4};"] + tail)
        if not body.staged:
            return _Staging(top=[f"const float4* const {rec}q = constants + {offset // 4} + {var} * {quads};"] + tail)
        together = max(1, min(group.members, self.options.stage_bytes // (4 * stride)))
        place = f"staged + {base // 4}" if base else "staged"
        source = f"constants + {offset // 4}"
        if ahead and together == group.members:
            start = [f"for (int i = threadIdx.x; i < {group.members * quads}; i += kThreadsPerBlock)",
                     f"  staged[{f'{base // 4} + ' if base else ''}i] = load_constants({source} + i);",
                     "__syncthreads();"]
            return _Staging(top=[f"const float4* const {rec}q = {place} + {var} * {quads};"] + tail,
                            words=group.members * stride, start=start)
        if tensor and together == 1:
            pointer = f"const float4* const {rec}q = {place} + ({var} & 1) * {quads};"
            return self.ring(place, source, quads, var, group.members, [pointer] + tail)
        # the block copies the records of the next `together` iterations into shared memory, 16 bytes per thread
        if together == 1:
            opening, count = "{", str(quads)
        else:
            opening = f"if ({var} % {together} == 0) {{"
            count = f"min({together}, {group.members} - {var}) * {quads}"
            place += f" + ({var} % {together}) * {quads}"
        target = f"staged[{base // 4} + i]" if base else "staged[i]"
        lines = [
            opening,
            "  __syncthreads();",
            f"  const float4* const source = {source} + {var} * {quads};",
            f"  for (int i = threadIdx.x; i < {count}; i += kThreadsPerBlock) {target} = load_constants(source + i);",
        ] + (["  fence_async();"] if tensor else []) + [
            "  __syncthreads();",
            "}",
            f"const float4* const {rec}q = {place};",
        ] + tail
        return _Staging(top=lines, words=together * stride)

    @staticmethod
    def ring(place: str, source: str, quads: int, var: str, count: int, pointers: list[str]) -> "_Staging":
        """Two slots for records of ``quads`` float4: the record of the next iteration is copied while this one
        computes."""
        def copy(member: str, slot: str) -> list[str]:
            return [f"for (int i = threadIdx.x; i < {quads}; i += kThreadsPerBlock)",
                    f"  copy_async({place} + {slot} + i, {source} + {member} + i);",
                    "commit_async();"]

        arrived = ["wait_async();", "fence_async();", "__syncthreads();"]
        top = ([f"if ({var} + 1 < {count}) {{"]
               + ["  " + line for line in copy(f"({var} + 1) * {quads}", f"(({var} + 1) & 1) * {quads}")] + ["}"])
        # the code before the loop may still read the stage buffer in other warps: together before the first copy
        return _Staging(top=top + pointers, words=8 * quads, start=["__syncthreads();"] + copy("0", "0"), ready=arrived,
                        end=arrived)

    def thread_group_lines(self, position: int, group: Group, phase) -> list[str]:
        """A loop on the tensor cores with its iterations by thread (tensor.ThreadFragments): chunks of four
        iterations; per chunk and tile the product sums, then every thread runs the rest of the body for its
        iteration and its two rows; the loops tied to this one follow, one row per thread.

        The warps of a block do not wait for each other inside the loop: the chunk records come through two slots
        with a barrier each, the image of a tile is its warpgroup's own, and the records of the tied loops are
        copied before the loop. With tied loops, the product sums of the next chunk are started before them and
        run on the tensor cores meanwhile."""
        body = _Body(self, position, group, tensor=True, var="mm")
        body.by_thread = True
        fragments = ThreadFragments(body, phase)
        self.mma_widths |= fragments.mma_widths()
        self.tile_image_words = max(self.tile_image_words, fragments.image_words)
        chunk, last = f"chunk{position}", f"irec{position}_last"
        chunks = fragments.chunks
        issue = fragments.issue_lines(chunk)   # also puts the biases into the record of the iteration
        # In the last chunk a thread may hold no iteration of the loop: it computes the last one again (the record
        # repeats it) and stores nothing. No branch around the computation: only the stores have the condition.
        partial = group.members % CHUNK != 0
        body.store_guard = "held" if partial else ""
        body.emit_stages(phase.tail, [])
        assert not body.before and not body.after
        inside = []
        for number, stream in enumerate(group.streamed):
            inner = _Body(self, f"{position}_{number}", stream.group, var="n")
            inner.store_guard = "current"   # the rows of the other tile have no values yet, or are done
            self.write_body(inner, stream.group, stream.group.stages, [])
            inside.append((f"{position}_{number}", f"{last}[{body.int_column(stream.done)}]", stream.group, inner))
        # the record of a chunk: its weight blocks, then the records of its four iterations (the last iteration
        # repeated where the loop ends inside the chunk)
        matrix, float_words = _Records.matrix(body.float_columns, body.int_columns, group.members)
        stride = matrix.shape[1]
        padded = np.concatenate([matrix, np.repeat(matrix[-1:], CHUNK * chunks - group.members, axis=0)])
        blocks = [block.view(np.uint32) for block in fragments.chunk_blocks]
        row = np.concatenate(blocks + [padded.reshape(chunks, -1)], axis=1)
        row = np.pad(row, ((0, 0), (0, -row.shape[1] % 32)))
        offset, quads = self.records.append(row), row.shape[1] // 4
        assert offset % 4 == 0
        # stage buffer: the records of the loops inside, then two slots for the chunk records when the shared
        # memory holds them, else one
        words = 0
        for _, _, streamed, inner in inside:
            inner.staging = self.record_lines(inner, streamed, words, False, ahead=True)
            words += inner.staging.words
        words = -(-words // 32) * 32
        place = f"(staged + {words // 4})"
        source = f"constants + {offset // 4}"
        rows = 128 * self.options.warpgroups * (self.plan.slab_words - self.plan.far_words)   # of the shared memory
        fixed = rows + 32 + self.options.warpgroups * self.tile_image_words
        two_slots = 4 * (fixed + words + 32 + 8 * quads) <= self.options.shared_memory_bytes
        full, done, slots = f"full{position}", f"done{position}", f"(staged + {words // 4 + 8})"
        records = fragments.chunk_words

        def pointers(number: str, wait: bool) -> list[str]:
            """The record of chunk ``number`` of the tile and, in it, that of the thread's iteration."""
            if two_slots:
                lines = [f"const int sequence = tile * {chunks} + {number}, slot = sequence & 1;"]
                lines += [f"barrier_wait({full} + slot, (sequence >> 1) & 1);"] if wait else []
                lines.append(f"const float* const {chunk} = (const float*)({slots} + slot * {quads});")
            else:
                lines = [f"const float* const {chunk} = (const float*){place};"]
            return lines + [
                f"const int mm = {CHUNK} * ({number}) + t;   // the iteration that this thread holds",
            ] + ([f"const bool held = mm < {group.members};"] if partial else []) + [
                f"const float4* const {body.rec}q = (const float4*)({chunk} + {records}) + t * {stride // 4};",
                f"const float* const {body.rec} = (const float*){body.rec}q;",
                f"const int* const {body.irec} = (const int*)({body.rec} + {float_words});",
                f"const int* const {last} = (const int*)({chunk} + {records + (CHUNK - 1) * stride + float_words});",
            ]

        # the statements for the thread's iteration and each of its two rows of the tile
        tail = [
            "#pragma unroll",
            "for (int h = 0; h < 2; ++h) {   // the thread's two rows of the tile",
            "  const int tile_row = lr0 + 64 * tile + g + 8 * h;",
        ] + ([("  const Row s{slab + tile_row * kSlabWords, far + tile_row * kFarWords};" if self.plan.far_words else
               "  float* const s = slab + tile_row * kSlabWords;")] if self.plan.slab_words else []) + [
            "  const long long block_row = (long long)blockIdx.x * kRowsPerBlock + tile_row;",
            f"  const bool active = block_row < row_count{' && held' if partial else ''};",
            "  float* const out = rows_out + (active ? block_row : row_count - 1) * kOutputs;",
        ] + ["  " + line for line in fragments.row_values() + body.lines] + ["}", "__syncwarp();"]

        names = ", ".join(self.graph.stages[i].name for i in group.stages[:3])
        head = ["", f"// ---- loop over {group.members} in chunks of {CHUNK}, one iteration per thread: {names}, ..."]
        head += fragments.before_loop()
        # the code before this loop may still read the stage buffer in other warps: together before it is written
        head.append("__syncthreads();")
        setup = fragments.tile_setup() + fragments.declarations()
        tied = []   # the loops tied to this one: one row per thread again, for the rows of this tile
        for name, complete, streamed, inner in inside:
            head += inner.before + inner.staging.start
            setup += [f"int next{name} = 0;", f"int ready{name} = 0;   // iterations whose input is complete"]
            tail.append(f"ready{name} = {complete};")
            first = self.graph.stages[streamed.stages[0]].name
            tied += [f"  // ---- iterations of the loop over {streamed.members} ({first}, ...) whose input is complete",
                     f"  while (next{name} < ready{name}) {{", f"    const int n = next{name};"]
            tied += ["    " + line for line in inner.staging.top + inner.lines] + [f"    ++next{name};", "  }"]
        if tied:   # the other half of the warp stores nothing
            tied = ["{", "  const bool current = (lane >> 4) == tile, active = row_active && current;"] + tied + ["}"]
        if two_slots:
            # Two slots filled by bulk copies that complete on barriers in shared memory: a warp waits for the
            # chunk it needs and for no other warp; the last warp to finish with a slot issues the copy of the
            # chunk after the next into it. The sequence runs over the two tiles.
            head += [f"uint64_t* const {full} = (uint64_t*){place};   // barriers of the two slots",
                     f"int* const {done} = (int*)({full} + 2);   // warps that are through with a slot",
                     "if (threadIdx.x == 0) {",
                     f"  barrier_init({full});",
                     f"  barrier_init({full} + 1);",
                     f"  {done}[0] = {done}[1] = 0;",
                     "  barrier_init_done();",
                     "}",
                     "__syncthreads();",
                     "if (threadIdx.x == 0) {",
                     f"  copy_bulk({slots}, {source}, {16 * quads}, {full});",
                     # the second slot only where a chunk is read from it: no copy that no warp waits for
                     *([f"  copy_bulk({slots} + {quads}, {source} + {(1 % chunks) * quads}, {16 * quads}, {full} + 1);"]
                       if self.tiles_planned * chunks > 1 else []),
                     "}"]
            release = ["if (lane == 0 && atomicAdd({done} + slot, 1) == kThreadsPerBlock / 32 - 1) {{".format(done=done),
                       f"  {done}[slot] = 0;",
                       f"  if (sequence + 2 < kTiles * {chunks})",
                       f"    copy_bulk({slots} + slot * {quads}, {source} + ((sequence + 2) % {chunks}) * {quads}, {16 * quads}, {full} + slot);",
                       "}"]
            self.stage_words = max(self.stage_words, words + 32 + 8 * quads)
        if two_slots and tied:
            # An iteration of the loop over the chunks starts the product sums of chunk q + 1, runs the tied loops
            # for chunk q, then waits for the product sums and runs the thread's part of chunk q + 1 (which reads
            # the chunk's record: its slot is given back there). The accumulators are written and read inside one
            # iteration: registers of a product sum in flight that stayed in use around the loop would make the
            # compiler wait for the product sums where they start.
            lines = ([f"if (q + 1 < {chunks}) {{   // start the product sums of the next chunk"]
                     + ["  " + line for line in pointers("q + 1", True) + issue] + ["}"]
                     + ["if (q >= 0) {   // the tied loops of this chunk"] + ["  " + line for line in tied] + ["}"]
                     + [f"if (q + 1 < {chunks}) {{   // the thread's part of the next chunk"]
                     + ["  " + line for line in pointers("q + 1", False) + fragments.complete_lines() + tail + release]
                     + ["}"])
            first_chunk = -1
        elif two_slots:
            lines = pointers("q", True) + issue + fragments.complete_lines() + tail + release
            first_chunk = 0
        else:
            # one slot: the block copies the record of the chunk, then every warp computes it
            lines = ["__syncthreads();",
                     f"for (int i = threadIdx.x; i < {quads}; i += kThreadsPerBlock)",
                     f"  {place}[i] = load_constants({source} + q * {quads} + i);",
                     "fence_async();",
                     "__syncthreads();"]
            lines += pointers("q", False) + issue + fragments.complete_lines() + tail + tied
            first_chunk = 0
            self.stage_words = max(self.stage_words, words + 4 * quads)
        loop = (["for (int tile = 0; tile < kTiles; ++tile) {   // the tiles of the warpgroup, one after the other"]
                + ["  " + line for line in setup]
                + [f"  for (int q = {first_chunk}; q < {chunks}; ++q) {{"]
                + ["    " + line for line in lines] + ["  }", "}"])
        # the warps ran apart: together again before the next group uses the shared memory
        return head + loop + ["__syncthreads();"]

    def group_lines(self, position: int, group: Group) -> list[str]:
        phase = self.plan.fragments.get(position)
        if isinstance(phase, DenseChain):
            return ChainWriter(_Body(self, position, group), phase).lines()
        if phase is not None and phase.by_thread:
            return self.thread_group_lines(position, group, phase)
        body = _Body(self, position, group, tensor=phase is not None)
        ahead: list[str] = []
        writer = None
        if phase is not None:
            writer = FragmentWriter(body, phase)
            ahead = writer.before_loop()
            body.lines += writer.lines()
            self.mma_widths |= writer.mma_widths()
        self.write_body(body, group, group.stages if phase is None else phase.tail, group.accumulated, writer, phase)
        if phase is not None:
            body.lines.append("__syncwarp();")
        # loops that run inside this one, each iteration as soon as its input is complete
        inside = []
        for number, stream in enumerate(group.streamed):
            inner = _Body(self, f"{position}_{number}", stream.group, var="n")
            self.write_body(inner, stream.group, stream.group.stages, [])
            inside.append((f"next{position}_{number}", f"{body.irec}[{body.int_column(stream.done)}]", stream.group, inner))
        staging = self.record_lines(body, group, 0, phase is not None)
        if group.members == 1:
            self.stage_words = max(self.stage_words, staging.words)
            return [""] + staging.top + body.lines
        names = ", ".join(self.graph.stages[i].name for i in group.stages[:3])
        more = f" and {len(group.stages) - 3} more" if len(group.stages) > 3 else ""
        head = ["", f"// ---- loop over {group.members}: {names}{more}"] + staging.start + body.before + ahead
        lines = staging.top + body.lines
        words = staging.words
        for counter, complete, streamed, inner in inside:
            inner_staging = self.record_lines(inner, streamed, words, False)
            words += inner_staging.words
            head += inner.before + [f"int {counter} = 0;"]
            first = self.graph.stages[streamed.stages[0]].name
            lines += [f"// ---- iterations of the loop over {streamed.members} ({first}, ...) whose input is complete",
                      f"while ({counter} < {complete}) {{", f"  const int n = {counter};"]
            lines += ["  " + line for line in inner_staging.top + inner.lines] + [f"  ++{counter};", "}"]
        self.stage_words = max(self.stage_words, words)
        head += staging.ready + [f"for (int m = 0; m < {group.members}; ++m) {{"]
        return head + ["  " + line for line in lines + staging.end] + ["}"] + body.after

    def source(self) -> "EmittedKernel":
        graph, name = self.graph, self.name
        self.stage_words = 0
        kernel_lines = self.prologue()
        for position, group in enumerate(self.plan.groups):
            kernel_lines += self.group_lines(position, group)
        kernel_lines += self.row_output_lines()
        threads = self.threads_per_block(self.stage_words)
        slab_words, far_words = self.plan.slab_words, self.plan.far_words
        near_words = slab_words - far_words   # per row in shared memory
        constants = np.concatenate(self.records.parts) if self.records.parts else np.zeros(4, dtype=np.uint32)
        shared = []
        if self.plan.fragments:
            shared += [
                "// Tensor-core layout: warpgroup wg, warp w of it, lane = 4 g + t; the warp owns the rows",
                "// lr0 .. lr0 + 15 of the first tile of 64 rows and, with two tiles, lr0 + 64 .. lr0 + 79 of the second.",
                "// In the code of one row per thread the first half of the warp has the rows of the first tile and",
                "// the second half those of the second (with one tile, the same rows once more: it stores nothing).",
                "const int wg = threadIdx.x >> 7, w = (threadIdx.x >> 5) & 3, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;",
                "const int lr0 = wg * (64 * kTiles) + w * 16;",
                "const int local_row = lr0 + (kTiles > 1 ? (lane >> 4) << 6 : 0) + (lane & 15);",
                "// For stores that the warp makes together: its first output row, and how far the rows exist.",
                "const long long warp_row = (long long)blockIdx.x * kRowsPerBlock + lr0;",
                "float* const warp_out = rows_out + (warp_row < row_count ? warp_row : 0) * kOutputs;",
                "const int warp_rows = warp_row < row_count ? (int)min(row_count - warp_row, kTiles > 1 ? 128LL : 16LL) : 0;",
            ]
        else:
            shared.append("const int local_row = threadIdx.x;")
        shared += [
            "// Every thread runs to the end (the block synchronises on the records); a thread past the last row repeats",
            "// the last row and stores nothing.",
            "const long long block_row = (long long)blockIdx.x * kRowsPerBlock + local_row;",
        ]
        if self.plan.fragments:   # with one tile, the second half of a warp computes the rows of the first again
            shared += ["const bool row_active = block_row < row_count && (kTiles > 1 || lane < 16);",
                       "const bool active = row_active;",
                       "const long long row = block_row < row_count ? block_row : row_count - 1;"]
        else:
            shared += ["const bool row_active = block_row < row_count;",
                       "const bool active = row_active;",
                       "const long long row = active ? block_row : row_count - 1;"]
        shared += [
            "const float* const in = rows_in + row * kInputs;",
            "float* const out = rows_out + row * kOutputs;",
        ]
        if self.stage_words or slab_words or self.tile_image_words:   # also where all words of the rows are in device memory
            shared.append("extern __shared__ __align__(128) float4 shared_memory[];")
        if self.stage_words:
            shared.append("float4* const staged = shared_memory;   // records of the running loop, shared by the block")
        if far_words:   # the first words of a row in device memory, the others in shared memory
            shared += ["float* const far = scratch + (long long)blockIdx.x * (kRowsPerBlock * kFarWords);   // rows of the block",
                       "float* const slab = (float*)shared_memory + kStageWords;   // their words in shared memory",
                       "const Row s{slab + local_row * kSlabWords, far + local_row * kFarWords};   // this row"]
        elif slab_words:
            shared.append("float* const slab = (float*)shared_memory + kStageWords;   // rows of the block")
            shared.append("float* const s = slab + local_row * kSlabWords;   // this row")
        if self.tile_image_words:
            shared.append("float* const images = (float*)shared_memory + kImageBase;   // low parts of A operands, per tile")
        tensor_helpers = ""
        if self.plan.fragments:
            tensor_helpers = TENSOR_HELPERS + "".join("\n" + mma_helper(n) + "\n" for n in sorted(self.mma_widths))
        body = "\n".join(("  " + line) if line else "" for line in shared + kernel_lines)
        # The signature ties a constants array to the kernel that reads it: the kernel text fixes the layout.
        layout = f"{graph.input_shape} {graph.output_shape} {threads} {slab_words} {self.stage_words} {len(constants)}"
        signature = hashlib.sha256(f"{layout}\n{body}".encode()).hexdigest()
        host = f"""int {name}_create(const void* constants, size_t words, const char* signature, void** kernel) {{
  using namespace {name}_kernel;
  *kernel = nullptr;
  if (words != kConstantWords || signature == nullptr || strcmp(signature, kSignature) != 0)
    return int(cudaErrorInvalidValue);
  void* device = nullptr;
  cudaError_t status = cudaMalloc(&device, words * sizeof(unsigned int));
  if (status == cudaSuccess) status = cudaMemcpy(device, constants, words * sizeof(unsigned int), cudaMemcpyHostToDevice);
  if (status == cudaSuccess && kSharedBytes > 0)
    status = cudaFuncSetAttribute(evaluate_rows, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kSharedBytes));
  if (status != cudaSuccess) {{
    cudaFree(device);
    return int(status);
  }}
  *kernel = device;
  return 0;
}}

int {name}_run(void* kernel, const float* rows, float* outputs, long long row_count, void* stream) {{
  using namespace {name}_kernel;
  if (kernel == nullptr) return int(cudaErrorInvalidValue);
  if (row_count <= 0) return 0;
  const unsigned int blocks = (unsigned int)((row_count + kRowsPerBlock - 1) / kRowsPerBlock);
  evaluate_rows<<<blocks, kThreadsPerBlock, kSharedBytes, cudaStream_t(stream)>>>(
      rows, outputs, row_count, static_cast<const float4*>(kernel));
  return int(cudaGetLastError());
}}

void {name}_destroy(void* kernel) {{ cudaFree(kernel); }}
"""
        device_object, device_includes = "", ""
        if far_words:
            # Words of the rows' slabs in device memory: a buffer of the kernel object per CUDA stream for the rows
            # of one launch; a call with more rows is several launches, one after the other on the stream.
            rows_per_block = 64 * self.tiles * (threads // 128) if self.tiles else threads
            device_includes = "\n#include <map>\n#include <mutex>\n#include <utility>\n#include <vector>\n"
            device_object = f"""
// The rows keep values between loops in device memory: kFarWords per row of a launch. A launch has at most
// kLaunchRows rows (a buffer of 256 MiB at most); a call with more rows is several launches.
constexpr long long kLaunchRows = {max(1, (1 << 28) // (4 * far_words * rows_per_block))}LL * kRowsPerBlock;

struct Kernel {{
  void* constants = nullptr;
  std::mutex lock;
  // Rows of the blocks that the processors of the device run at a time, one block each (0: not known). A launch
  // of more rows is a whole number of such waves: a block of this kernel may take a processor for itself, and the
  // blocks left over at the end of a launch would run on a few processors while the others wait for the next.
  long long wave = 0;
  // Per CUDA stream, the buffers of the rows' slabs (device memory, rows each holds); the last one is in use. A
  // buffer is allocated when a call has more rows than the last one holds (twice as many then, up to kLaunchRows),
  // and the earlier ones stay until the kernel is destroyed: a queued launch or a captured CUDA graph may still
  // use them.
  std::map<void*, std::vector<std::pair<float*, long long>>> slabs;
}};
"""
            host = f"""int {name}_create(const void* constants, size_t words, const char* signature, void** kernel) {{
  using namespace {name}_kernel;
  *kernel = nullptr;
  if (words != kConstantWords || signature == nullptr || strcmp(signature, kSignature) != 0)
    return int(cudaErrorInvalidValue);
  Kernel* made = new Kernel;
  cudaError_t status = cudaMalloc(&made->constants, words * sizeof(unsigned int));
  if (status == cudaSuccess)
    status = cudaMemcpy(made->constants, constants, words * sizeof(unsigned int), cudaMemcpyHostToDevice);
  if (status == cudaSuccess && kSharedBytes > 0)
    status = cudaFuncSetAttribute(evaluate_rows, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kSharedBytes));
  int device = 0, processors = 0;
  if (status == cudaSuccess) status = cudaGetDevice(&device);
  if (status == cudaSuccess) status = cudaDeviceGetAttribute(&processors, cudaDevAttrMultiProcessorCount, device);
  if (status != cudaSuccess) {{
    {name}_destroy(made);
    return int(status);
  }}
  made->wave = processors * kRowsPerBlock;
  *kernel = made;
  return 0;
}}

int {name}_run(void* kernel, const float* rows, float* outputs, long long row_count, void* stream) {{
  using namespace {name}_kernel;
  if (kernel == nullptr) return int(cudaErrorInvalidValue);
  if (row_count <= 0) return 0;
  Kernel* k = static_cast<Kernel*>(kernel);
  std::lock_guard<std::mutex> hold(k->lock);   // the launches of a call stay together on the stream
  const long long blocks = (row_count + kRowsPerBlock - 1) / kRowsPerBlock;
  const long long most = k->wave > 0 && kLaunchRows > k->wave ? kLaunchRows - kLaunchRows % k->wave : kLaunchRows;
  const long long needed = blocks * kRowsPerBlock < most ? blocks * kRowsPerBlock : most;
  auto& buffers = k->slabs[stream];
  if (buffers.empty() || buffers.back().second < needed) {{
    long long capacity = buffers.empty() ? needed : 2 * buffers.back().second;
    capacity = capacity < needed ? needed : (capacity > most ? most : capacity);
    float* data = nullptr;
    const cudaError_t status = cudaMalloc(&data, size_t(capacity) * kFarWords * sizeof(float));
    if (status != cudaSuccess) return int(status);
    buffers.emplace_back(data, capacity);
  }}
  long long launch = buffers.back().second;   // a multiple of kRowsPerBlock
  if (k->wave > 0 && launch > k->wave) launch -= launch % k->wave;   // whole waves
  for (long long first = 0; first < row_count; first += launch) {{
    const long long count = row_count - first < launch ? row_count - first : launch;
    evaluate_rows<<<(unsigned int)((count + kRowsPerBlock - 1) / kRowsPerBlock), kThreadsPerBlock, kSharedBytes,
                    cudaStream_t(stream)>>>(rows + first * kInputs, outputs + first * kOutputs, count,
                                            static_cast<const float4*>(k->constants), buffers.back().first);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return int(status);
  }}
  return 0;
}}

void {name}_destroy(void* kernel) {{
  using namespace {name}_kernel;
  Kernel* k = static_cast<Kernel*>(kernel);
  if (k == nullptr) return;
  cudaFree(k->constants);
  for (auto& stream : k->slabs)
    for (auto& buffer : stream.second) cudaFree(buffer.first);
  delete k;
}}
"""
        source = f"""// Generated by fortonnx_kernelgen. Do not edit: regenerate from the graph.
// Graph: {comment_text(self.source_note)}
// Input "{comment_text(graph.input_name)}" [rows, {graph.input_size}] FP32 -> output "{comment_text(graph.output_name)}" [rows, {graph.row_width}] FP32.
// One thread evaluates one row; {64 * self.tiles * (threads // 128) if self.tiles else threads} rows and {threads} threads per block. Shared memory: {4 * near_words} bytes per row and
// {4 * self.stage_words} bytes for the constants of the running loop.
// Build with nvcc --fmad=false: products and sums of the graph's elementwise operators stay separate operations
// (the product sums of MatMul use fmaf explicitly).
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
{device_includes}
#include "{name}.h"

namespace {name}_kernel {{

constexpr int kInputs = {graph.input_size};
constexpr int kOutputs = {graph.row_width};
constexpr size_t kConstantWords = {len(constants)};
constexpr int kStageWords = {self.stage_words};
constexpr int kSlabWords = {near_words};
{f"constexpr int kFarWords = {far_words};   // words per row in device memory, ahead of those in shared memory" + chr(10) if far_words else ""}constexpr int kThreadsPerBlock = {threads};
constexpr int kTileImageWords = {self.tile_image_words};
constexpr int kTiles = {self.tiles};   // tiles of 64 rows per warpgroup (0: one row per thread)
constexpr int kRowsPerBlock = {64 * self.tiles * (threads // 128) if self.tiles else threads};
constexpr int kImageBase = (kStageWords + kRowsPerBlock * kSlabWords + 31) / 32 * 32;
constexpr size_t kSharedBytes = size_t(kImageBase + {threads // 128 if self.plan.fragments else 0} * kTileImageWords) * sizeof(float);
const char kSignature[] = "{signature}";

// One 16-byte load: of constants in device memory (read-only data cache) and of shared memory.
__device__ __forceinline__ float4 load_constants(const float4* address) {{ return __ldg(address); }}

__device__ __forceinline__ float4 load_shared(const float4* address) {{ return *address; }}
{ROW_OF_TWO_PARTS if far_words else ""}{APPROXIMATE_HELPERS if self.options.approximate_functions else ""}{tensor_helpers}
__global__ void evaluate_rows(const float* __restrict__ rows_in, float* __restrict__ rows_out, long long row_count,
                              const float4* __restrict__ constants{', float* __restrict__ scratch' if far_words else ''}) {{
{body}
}}
{device_object}
}}  // namespace {name}_kernel

extern "C" {{

{host}
}}  // extern "C"
"""
        header = f"""// Generated by fortonnx_kernelgen: the kernel "{name}" as the custom operator library calls it (kernel_ops.cc).
// These functions are internal to the library; programs reach the kernel through ONNX Runtime.
// Rows and outputs are device pointers to row-major FP32 arrays [rows, {graph.input_size}] and
// [rows, {graph.row_width}]. The functions that return int return 0 or a cudaError_t code.
#ifndef FORTONNX_KERNEL_{name}_H
#define FORTONNX_KERNEL_{name}_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {{
#endif

// Copies the constants (host memory: the constant input of the graph's node) to the current device. A kernel with
// another signature than the one stored with the constants refuses them (cudaErrorInvalidValue).
int {name}_create(const void* constants, size_t words, const char* signature, void** kernel);
// Queues the evaluation of row_count rows on the CUDA stream (a cudaStream_t).
int {name}_run(void* kernel, const float* rows, float* outputs, long long row_count, void* stream);
void {name}_destroy(void* kernel);

#ifdef __cplusplus
}}
#endif

#endif
"""
        summary = {
            "stages": sum(1 for s in graph.stages if s.kind in ("map", "dot", "reduce")),
            "groups": len(self.plan.groups),
            "loops": [g.members for g in self.plan.groups if g.members > 1],
            "slab_bytes_per_row": 4 * slab_words,
            "stage_bytes": 4 * self.stage_words,
            "threads_per_block": threads,
            "rows_per_block": 64 * self.tiles * (threads // 128) if self.tiles else threads,
            "slab": "device" if far_words else "shared",
            "constant_bytes": int(constants.nbytes),
            "kernel_lines": len(kernel_lines),
            "tensor_cores": bool(self.plan.fragments),
            "signature": signature,
        }
        return EmittedKernel(source, header, constants, signature, summary)


@dataclass
class EmittedKernel:
    """Texts and constants of one kernel: source (NAME.cu), header (NAME.h, for the custom operator library), the
    constants as 32-bit words in the order the kernel reads them, and the signature that the kernel checks when it
    receives them."""

    source: str
    header: str
    constants: np.ndarray
    signature: str
    summary: dict


def emit_kernel(plan: Plan, name: str, options: Options, source_note: str) -> EmittedKernel:
    """Kernel source, header and constants of the planned row graph."""
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        raise ValueError(f"kernel name {name!r} is not a C identifier")
    if options.dot_lanes < 1:
        raise ValueError("dot_lanes must be at least 1")
    # Where the rows keep their values between loops. In the shared memory of the block the accesses are fast, but
    # a row that keeps much leaves the block few rows; in device memory the block is full, but every thread then
    # reads and writes its own lines of device memory. Measured: code of one row per thread is 1.6 to 4.5 times
    # slower on it with 96 to 128 of 256 rows in shared memory and 3 times faster with 27; chains on the tensor
    # cores are 1.2 to 2 times faster with a full block than with 64 of 256 rows. With "auto":
    #   - the slabs are in shared memory, on the tensor cores with one tile per warpgroup where two do not fit,
    #     and then without the stores that the warp makes together (they take words of the slabs);
    #   - they are in device memory where the kernel does not fit otherwise, or would have less than a quarter of
    #     its rows per block (less than half for a kernel whose loops are all on the tensor cores).
    full = 128 * options.warpgroups if plan.fragments else (options.threads_per_block or _SAFE_THREADS)
    tensor_loops_only = bool(plan.fragments) and all(group.members == 1 or position in plan.fragments
                                                     for position, group in enumerate(plan.groups))
    attempts = [replace(options, slab="device")] if options.slab == "device" else [replace(options, slab="shared")]
    if options.slab != "device" and plan.fragments:
        attempts += [replace(attempts[0], tiles=1)] if options.tiles >= 2 else []
        attempts += [replace(attempts[0], tiles=1, gathered_stores=False)] if options.gathered_stores else []
    chains = any(isinstance(phase, DenseChain) for phase in plan.fragments.values())

    def with_slots(settings: Options, far_from: int | None) -> EmittedKernel:
        """The kernel with these settings and this split of the rows' slabs (schedule.assign_storage); its chains
        with the largest weight slots (tensor.SLOT_FLOATS) with which the block keeps its rows."""
        chosen = None
        for floats in SLOT_FLOATS if chains else SLOT_FLOATS[:1]:
            assign_storage(plan, far_from)
            try:
                kernel = _Emitter(plan, name, settings, source_note, floats).source()
            except KernelTooLarge:
                if chosen is None:
                    raise
                break
            if chosen is not None and kernel.summary["rows_per_block"] < chosen.summary["rows_per_block"]:
                break
            chosen = kernel
        return chosen

    def emit(settings: Options) -> EmittedKernel:
        """The kernel with these settings. With the rows' slabs in device memory, only the stages of
        Limits.far_words words or more are there: the values that the loops read and write all the time stay in
        shared memory. Where the block does not keep its rows that way, the kernel is also written with one tile
        per warpgroup (more warpgroups for the same rows: measured 1.2 to 1.4 times faster than two tiles of one
        warpgroup) and with all stages that are kept between loops in device memory; the one with the most
        threads per block, then the most rows, is taken."""
        if settings.slab != "device":
            return with_slots(settings, None)
        found = []
        for tiles in ((settings.tiles, 1) if plan.fragments and settings.tiles >= 2 else (settings.tiles,)):
            try:
                kernel = with_slots(replace(settings, tiles=tiles), settings.limits.far_words)
            except KernelTooLarge:
                continue
            if kernel.summary["rows_per_block"] >= full:
                return kernel
            found.append(kernel)
        try:
            found.append(with_slots(settings, 1))
        except KernelTooLarge:
            if not found:
                raise
        return max(found, key=lambda kernel: (kernel.summary["threads_per_block"], kernel.summary["rows_per_block"]))

    emitted = None
    for attempt in attempts:
        try:
            emitted = emit(attempt)
            break
        except KernelTooLarge:
            if options.slab != "auto" and attempt is attempts[-1]:
                raise
    few_rows = emitted is not None and (2 if tensor_loops_only else 4) * emitted.summary["rows_per_block"] < full
    if options.slab == "auto" and (emitted is None or few_rows):
        emitted = emit(replace(options, slab="device"))
    return emitted


def kernels_in_a_row(name: str, parts: list[tuple[str, EmittedKernel]], width: int, rows: tuple[int, int]) -> tuple[EmittedKernel, dict[str, str]]:
    """The kernel ``name`` as two kernels that run one after the other: the first writes ``width`` values per row
    into a device buffer of the kernel object, the second reads them as its rows. Returns the kernel (source,
    header, constants and signature of the pair, as for one kernel) and the texts of the parts' files by file name;
    ``rows`` are the values per row of the input and of the output."""
    (first_name, first), (second_name, second) = parts
    constants = np.concatenate([first.constants, second.constants])
    signature = hashlib.sha256(f"{first.signature} {second.signature} {width}".encode()).hexdigest()
    header = first.header.replace(first_name, name).replace(f"[rows, {width}]", f"[rows, {rows[1]}]")
    source = f"""// Generated by fortonnx_kernelgen. Do not edit: regenerate from the graph.
// The kernel "{name}" as two kernels in a row: {first_name} writes {width} values per row into a device buffer of
// the kernel object, {second_name} reads them as its rows. Each has its own number of threads per block.
#include <cuda_runtime.h>
#include <string.h>

#include <map>
#include <mutex>
#include <utility>
#include <vector>

#include "{name}.h"
#include "{first_name}.cu"
#include "{second_name}.cu"

namespace {name}_kernels {{

constexpr size_t kFirstWords = {len(first.constants)};
constexpr size_t kSecondWords = {len(second.constants)};
constexpr int kWidth = {width};   // values per row between the kernels
const char kSignature[] = "{signature}";

struct Kernels {{
  void* first = nullptr;
  void* second = nullptr;
  std::mutex lock;
  // Per CUDA stream, the buffers of the rows between the kernels (device memory, rows each holds); the last one is
  // in use. A buffer is allocated when a call has more rows than the last one holds (at least twice as many
  // then), never per call, and the earlier ones stay until the kernel is destroyed: a queued launch or a captured
  // CUDA graph may still read them.
  std::map<void*, std::vector<std::pair<float*, long long>>> between;
}};

}}  // namespace {name}_kernels

extern "C" {{

int {name}_create(const void* constants, size_t words, const char* signature, void** kernel) {{
  using namespace {name}_kernels;
  *kernel = nullptr;
  if (words != kFirstWords + kSecondWords || signature == nullptr || strcmp(signature, kSignature) != 0)
    return int(cudaErrorInvalidValue);
  Kernels* made = new Kernels;
  const unsigned int* w = static_cast<const unsigned int*>(constants);
  int status = {first_name}_create(w, kFirstWords, "{first.signature}", &made->first);
  if (status == 0) status = {second_name}_create(w + kFirstWords, kSecondWords, "{second.signature}", &made->second);
  if (status != 0) {{
    {name}_destroy(made);
    return status;
  }}
  *kernel = made;
  return 0;
}}

int {name}_run(void* kernel, const float* rows, float* outputs, long long row_count, void* stream) {{
  using namespace {name}_kernels;
  if (kernel == nullptr) return int(cudaErrorInvalidValue);
  if (row_count <= 0) return 0;
  Kernels* k = static_cast<Kernels*>(kernel);
  std::lock_guard<std::mutex> hold(k->lock);   // the two launches of a call stay together on the stream
  auto& buffers = k->between[stream];
  if (buffers.empty() || buffers.back().second < row_count) {{
    // at least twice the last one: the buffers of calls with growing row counts stay small in sum
    const long long capacity = buffers.empty() || 2 * buffers.back().second < row_count ? row_count : 2 * buffers.back().second;
    float* data = nullptr;
    const cudaError_t status = cudaMalloc(&data, size_t(capacity) * kWidth * sizeof(float));
    if (status != cudaSuccess) return int(status);
    buffers.emplace_back(data, capacity);
  }}
  float* const between = buffers.back().first;
  const int status = {first_name}_run(k->first, rows, between, row_count, stream);
  return status != 0 ? status : {second_name}_run(k->second, between, outputs, row_count, stream);
}}

void {name}_destroy(void* kernel) {{
  using namespace {name}_kernels;
  Kernels* k = static_cast<Kernels*>(kernel);
  if (k == nullptr) return;
  {first_name}_destroy(k->first);
  {second_name}_destroy(k->second);
  for (auto& stream : k->between)
    for (auto& buffer : stream.second) cudaFree(buffer.first);
  delete k;
}}

}}  // extern "C"
"""
    summary = {
        "kernels": [first.summary, second.summary],
        "values_between": width,
        "threads_per_block": [first.summary["threads_per_block"], second.summary["threads_per_block"]],
        "constant_bytes": int(constants.nbytes),
        "tensor_cores": True,
        "signature": signature,
    }
    files = {f"{first_name}.cu": first.source, f"{first_name}.h": first.header,
             f"{second_name}.cu": second.source, f"{second_name}.h": second.header}
    return EmittedKernel(source, header, constants, signature, summary), files


def kernels_side_by_side(name: str, parts: list[tuple[str, EmittedKernel]]) -> tuple[EmittedKernel, dict[str, str]]:
    """The kernel ``name`` as kernels that each compute a part of the output row (schedule.output_parts): they read
    the same rows and store into the same output rows. Returns the kernel (source, header, constants and signature
    of all, as for one kernel) and the texts of the parts' files."""
    constants = np.concatenate([kernel.constants for _, kernel in parts])
    signature = hashlib.sha256(" ".join(["parts"] + [kernel.signature for _, kernel in parts]).encode()).hexdigest()
    header = parts[0][1].header.replace(parts[0][0], name)
    includes = "\n".join(f'#include "{part}.cu"' for part, _ in parts)
    creates, first = [], 0
    for number, (part, kernel) in enumerate(parts):
        creates.append(f'  if (status == 0) status = {part}_create(w + {first}, {len(kernel.constants)}, "{kernel.signature}", '
                       f"&made->part[{number}]);")
        first += len(kernel.constants)
    others = "\n".join(
        f"  if (status == 0) status = int(cudaStreamWaitEvent(k->side[{number - 1}], k->ready, 0));\n"
        f"  if (status == 0) status = {part}_run(k->part[{number}], rows, outputs, row_count, k->side[{number - 1}]);\n"
        f"  if (status == 0) status = int(cudaEventRecord(k->done[{number - 1}], k->side[{number - 1}]));"
        for number, (part, _) in enumerate(parts) if number > 0)
    destroys = "\n".join(f"  {part}_destroy(k->part[{number}]);" for number, (part, _) in enumerate(parts))
    source = f"""// Generated by fortonnx_kernelgen. Do not edit: regenerate from the graph.
// The kernel "{name}" as {len(parts)} kernels side by side: each computes a part of the output row (the outputs that
// share their product sums) from the same rows. The first runs on the caller's stream, the others on streams of
// the kernel object at the same time: they start when the caller's stream has reached the call, and the caller's
// stream goes on when they are through.
#include <cuda_runtime.h>
#include <string.h>

#include <mutex>

#include "{name}.h"
{includes}

namespace {name}_kernels {{

constexpr size_t kWords = {len(constants)};
constexpr int kParts = {len(parts)};
const char kSignature[] = "{signature}";

struct Kernels {{
  void* part[kParts] = {{}};
  std::mutex lock;                         // the events of a call belong to it until its launches are queued
  cudaStream_t side[kParts - 1] = {{}};     // streams of the parts behind the first
  cudaEvent_t ready = nullptr;             // the caller's stream has reached the call
  cudaEvent_t done[kParts - 1] = {{}};      // a part behind the first is through
}};

}}  // namespace {name}_kernels

extern "C" {{

int {name}_create(const void* constants, size_t words, const char* signature, void** kernel) {{
  using namespace {name}_kernels;
  *kernel = nullptr;
  if (words != kWords || signature == nullptr || strcmp(signature, kSignature) != 0)
    return int(cudaErrorInvalidValue);
  Kernels* made = new Kernels;
  const unsigned int* w = static_cast<const unsigned int*>(constants);
  int status = int(cudaEventCreateWithFlags(&made->ready, cudaEventDisableTiming));
  for (int i = 0; i < kParts - 1 && status == 0; ++i) {{
    status = int(cudaStreamCreateWithFlags(&made->side[i], cudaStreamNonBlocking));
    if (status == 0) status = int(cudaEventCreateWithFlags(&made->done[i], cudaEventDisableTiming));
  }}
{chr(10).join(creates)}
  if (status != 0) {{
    {name}_destroy(made);
    return status;
  }}
  *kernel = made;
  return 0;
}}

int {name}_run(void* kernel, const float* rows, float* outputs, long long row_count, void* stream) {{
  using namespace {name}_kernels;
  if (kernel == nullptr) return int(cudaErrorInvalidValue);
  if (row_count <= 0) return 0;
  Kernels* k = static_cast<Kernels*>(kernel);
  std::lock_guard<std::mutex> hold(k->lock);
  int status = int(cudaEventRecord(k->ready, cudaStream_t(stream)));
{others}
  if (status == 0) status = {parts[0][0]}_run(k->part[0], rows, outputs, row_count, stream);
  for (int i = 0; i < kParts - 1 && status == 0; ++i) status = int(cudaStreamWaitEvent(cudaStream_t(stream), k->done[i], 0));
  return status;
}}

void {name}_destroy(void* kernel) {{
  using namespace {name}_kernels;
  Kernels* k = static_cast<Kernels*>(kernel);
  if (k == nullptr) return;
  for (int i = 0; i < kParts - 1; ++i) {{   // what is queued on the streams of the parts first
    if (k->side[i] != nullptr) cudaStreamSynchronize(k->side[i]);
  }}
{destroys}
  for (int i = 0; i < kParts - 1; ++i) {{
    if (k->done[i] != nullptr) cudaEventDestroy(k->done[i]);
    if (k->side[i] != nullptr) cudaStreamDestroy(k->side[i]);
  }}
  if (k->ready != nullptr) cudaEventDestroy(k->ready);
  delete k;
}}

}}  // extern "C"
"""
    summary = {
        "kernels": [kernel.summary for _, kernel in parts],
        "output_parts": len(parts),
        "threads_per_block": [kernel.summary["threads_per_block"] for _, kernel in parts],
        "constant_bytes": int(constants.nbytes),
        "tensor_cores": True,
        "signature": signature,
    }
    files = {f"{part}{suffix}": text for part, kernel in parts for suffix, text in ((".cu", kernel.source), (".h", kernel.header))}
    return EmittedKernel(source, header, constants, signature, summary), files
