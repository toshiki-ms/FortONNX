"""Loop structure and storage of the kernel for one row.

The row graph (rowgraph.py) is a list of stages whose operands are concrete element references. This module decides

- which stages share a loop. A *group* runs ``members`` iterations; in iteration ``m`` every stage of the group
  computes its elements ``m * inner .. (m + 1) * inner - 1`` (``inner = size / members``) as local scalars. Stages
  may share a group when each of them reads the others only inside the same iteration and at positions that do not
  depend on ``m``. A group of small stages is written as straight-line code (``members = 1``).
- where values live. Values read only inside their own group are locals of the loop body; values of straight-line
  groups are scalars of the kernel; every other stage gets a range of the per-row slab in shared memory.

Two rewrites of the row graph come first, both exact:

- ``sink_gathers``: an elementwise stage that reads a permuted part of one index space (after a Gather) is moved onto
  that index space, so that it lines up with its producer; elements nobody selected are computed and never read.
- ``merge_repeated_elements``: an elementwise stage applied to a broadcast value keeps one element per distinct
  operand tuple.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import gcd

import numpy as np

from .idioms import _readers, fold_scales, fuse_exp_tanh, fuse_logsumexp
from .ranges import simplify_by_ranges
from .rowgraph import RowGraph, Stage, make_refs, ref_offset, ref_stage, whole_stage
from .tensor import CHUNK, DenseChain, FragmentPhase, find_dense_chains, find_fragment_phase, find_thread_phase

_OFFSET_BITS = 32


@dataclass
class Limits:
    """Generation settings.

    register_array   largest stage (elements) that may be written as scalars of the kernel instead of a loop
    body_terms       largest number of scalar operations and products in one loop body
    sink_growth      a stage is moved onto its producer's index space when that is at most this much larger
    wide_sums        on the tensor cores, product sums outside loops with more than 128 terms or elements run in
                     blocks (tensor.py); without, they stay sums of fmaf
    sums_once        on the tensor cores, a sum of few terms whose values are the terms of a sum in blocks of terms
                     and of elements is a chain of its own: computed once per row and kept in the rows' slabs.
                     Without, it is the start of that sum's chain: computed again for every block of elements,
                     with less per row in the slabs
    far_words        with the rows' slabs in device memory (emit.emit_kernel): the stages of this many words or
                     more are there, the smaller ones stay in shared memory. (Measured on a network
                     whose rows keep 76 words: 2.72 ms with them in shared memory, 6.19 ms in device memory;
                     the loops of a kernel read and write their small values all the time.)
    """

    register_array: int = 64
    body_terms: int = 32768
    sink_growth: float = 2.0
    wide_sums: bool = True
    sums_once: bool = True
    far_words: int = 64


@dataclass
class Group:
    """One loop (``members`` iterations) or one piece of straight-line code (``members`` 1).

    stages       computed in the body, in order
    accumulated  product sums over all elements of a stage of this loop: every iteration adds its terms to scalars
                 that live across the loop, so the summed stage needs no storage of its own
    chain        the chain of wide product sums that this group is (tensor.DenseChain), else None
    """

    members: int
    stages: list[int]
    accumulated: list[int] = field(default_factory=list)
    streamed: list["Stream"] = field(default_factory=list)
    chain: object = None


@dataclass
class Stream:
    """A loop that runs inside another loop (the producer): iteration ``b`` of ``group`` runs as soon as the
    producer has computed the last element of ``source`` that it reads. ``done[m]`` is the number of iterations of
    ``group`` that are complete after iteration ``m`` of the producer; ``source`` is then kept as a ring of ``ring``
    values per row instead of a whole array."""

    group: Group
    source: int
    done: np.ndarray
    ring: int


@dataclass
class Plan:
    """The schedule of one kernel.

    live         per stage, [size] True for elements that the output depends on
    groups       in emission order
    slab_offset  word offset in the per-row slab of every stage kept in shared memory
    kept         for a stage of which only some elements are kept there: these elements, in order (the slab has
                 one word for each of them)
    slab_words   words of the slab of one row (4 x an odd number, or 0)
    far_words    the first words of them, where the rows' slabs are split (assign_storage): the stages that keep
                 much per row; the kernel has these in device memory and the other words in shared memory
    """

    graph: RowGraph
    live: list[np.ndarray]
    groups: list[Group]
    slab_offset: dict[int, int] = field(default_factory=dict)
    kept: dict[int, np.ndarray] = field(default_factory=dict)
    slab_words: int = 0
    far_words: int = 0
    fragments: dict[int, FragmentPhase] = field(default_factory=dict)   # group position -> its tensor-core part
    ring: dict[int, int] = field(default_factory=dict)                  # stage kept as a ring -> its length

    def word(self, stage: int, element):
        """Slab word of an element (or of an array of elements) of a stage that is kept in the rows' slabs."""
        if stage in self.kept:
            return self.slab_offset[stage] + np.searchsorted(self.kept[stage], element)
        return self.slab_offset[stage] + element

    def group_position(self) -> dict[int, int]:
        """Stage -> position of the loop (or straight-line code) in which it is computed."""
        return {index: position for position, group in enumerate(self.groups)
                for part in [group] + [stream.group for stream in group.streamed]
                for index in part.stages + part.accumulated}


def _remap(graph: RowGraph, stage_index: int, new_offset: np.ndarray) -> None:
    """Element ``e`` of the stage became element ``new_offset[e]``: rewrite every reference to it."""

    def rewrite(refs: np.ndarray) -> np.ndarray:
        mask = ref_stage(refs) == stage_index
        if not mask.any():
            return refs
        moved = refs.copy()
        moved[mask] = make_refs(stage_index, new_offset[ref_offset(refs[mask])])
        return moved

    for stage in graph.stages[stage_index + 1:]:
        stage.operands = [rewrite(refs) for refs in stage.operands]
    graph.output_refs = rewrite(graph.output_refs)


def fold_biases(graph: RowGraph) -> None:
    """A constant added to a product sum becomes the bias of the sum (which then starts from it), when nothing else
    reads the sum. The tensor-core product sums start from their bias."""
    is_const = graph.is_const()
    readers: dict[int, int] = {}
    for stage in graph.stages:
        for source in {int(v) for refs in stage.operands for v in np.unique(ref_stage(refs))}:
            readers[source] = readers.get(source, 0) + 1
    for source in np.unique(ref_stage(graph.output_refs)):
        readers[int(source)] = readers.get(int(source), 0) + 1
    for add in graph.stages:
        if add.kind != "map" or add.op != "add":
            continue
        for position in (0, 1):
            sum_refs, constant = add.operands[position], add.operands[1 - position]
            index = int(ref_stage(sum_refs)[0])
            dot = graph.stages[index]
            if not is_const[ref_stage(constant)].all() or dot.kind != "dot" or dot.weights is None:
                continue
            if dot.size != add.size or readers.get(index) != 1:
                continue
            if not np.array_equal(sum_refs, make_refs(index, np.arange(dot.size))):
                continue
            values = graph.const_values(constant)
            with np.errstate(over="ignore", invalid="ignore"):
                bias = values if dot.bias is None else (dot.bias + values).astype(np.float32)
            if not np.isfinite(bias).all():   # two constants whose sum alone is no FP32 number stay two additions
                continue
            dot.bias = bias
            for later in graph.stages[add.index + 1:]:
                later.operands = [np.where(ref_stage(refs) == add.index, make_refs(index, ref_offset(refs)), refs)
                                  for refs in later.operands]
            graph.output_refs = np.where(ref_stage(graph.output_refs) == add.index,
                                         make_refs(index, ref_offset(graph.output_refs)), graph.output_refs)
            readers[index] = readers.get(add.index, 0)   # the readers of the sum are now those of the addition
            break


def fuse_log1p(graph: RowGraph) -> None:
    """Replace the FortONNX exporter's accurate log1p by one log1p stage.

    The exporter writes log1p(x) as ``u = x + 1; x if u == 1 else log(u) * x / (u - 1)``, which is accurate only when
    log is accurate to its last bits near 1. With approximate functions (emit.Options.approximate_functions) the form
    would lose that, so it is recognised and computed by log1pf. The stages of the form that nothing else reads are
    dropped with the dead elements."""
    stages = graph.stages
    is_const = graph.is_const()

    def whole(refs: np.ndarray, op: str):
        """The map stage ``op`` whose elements, in order, are ``refs``; None otherwise."""
        index = int(ref_stage(refs).reshape(-1)[0])
        stage = stages[index]
        if stage.kind != "map" or stage.op != op or stage.size != refs.size:
            return None
        if not np.array_equal(refs, make_refs(index, np.arange(stage.size))):
            return None
        return stage

    def ones(refs: np.ndarray) -> bool:
        return bool(is_const[ref_stage(refs)].all()) and bool((graph.const_values(refs) == 1.0).all())

    for out in stages:
        if out.kind != "map" or out.op != "where":
            continue
        condition, x, ratio_refs = out.operands
        equal, ratio = whole(condition, "equal"), whole(ratio_refs, "div")
        if equal is None or ratio is None or not ones(equal.operands[1]):
            continue
        u = equal.operands[0]
        add, product, safe = whole(u, "add"), whole(ratio.operands[0], "mul"), whole(ratio.operands[1], "where")
        if add is None or product is None or safe is None:
            continue
        if not (np.array_equal(add.operands[0], x) and ones(add.operands[1])):
            continue
        log = whole(product.operands[0], "log")
        if log is None or not np.array_equal(log.operands[0], u) or not np.array_equal(product.operands[1], x):
            continue
        difference = whole(safe.operands[2], "sub")
        if difference is None or not np.array_equal(safe.operands[0], condition) or not ones(safe.operands[1]):
            continue
        if not np.array_equal(difference.operands[0], u) or not ones(difference.operands[1]):
            continue
        out.op = "log1p"
        out.operands = [x]


def fuse_logaddexp(graph: RowGraph) -> None:
    """Replace the exported form of logaddexp(x, c) with a constant c (softplus is c = 0),

        where(isnan(x - c), x + c, max(x, c) + log1p(exp(-|x - c|))),

    by one logaddexp stage (after fuse_log1p). The kernel then computes it with one exponential and a short series
    instead of two library functions; the stages of the form that nothing else reads are dropped."""
    is_const = graph.is_const()
    for out in graph.stages:
        if out.kind != "map" or out.op != "where":
            continue
        nan, plain_sum, total = (whole_stage(graph, out.operands[0], "not"), whole_stage(graph, out.operands[1], "add"),
                                 whole_stage(graph, out.operands[2], "add"))
        if nan is None or plain_sum is None or total is None:
            continue
        largest, log1p = whole_stage(graph, total.operands[0], "max"), whole_stage(graph, total.operands[1], "log1p")
        equal = whole_stage(graph, nan.operands[0], "equal")
        if largest is None or log1p is None or equal is None:
            continue
        x, c = largest.operands
        if not is_const[ref_stage(c)].all() or not np.array_equal(plain_sum.operands[0], x):
            continue
        exp = whole_stage(graph, log1p.operands[0], "exp")
        negative = whole_stage(graph, exp.operands[0], "neg") if exp is not None else None
        absolute = whole_stage(graph, negative.operands[0], "abs") if negative is not None else None
        if absolute is None or not np.array_equal(equal.operands[1], equal.operands[0]):
            continue
        if not np.array_equal(absolute.operands[0], equal.operands[0]) or not np.array_equal(plain_sum.operands[1], c):
            continue
        # the difference x - c: a stage, or x itself where the import has dropped the subtraction of zero
        difference = whole_stage(graph, equal.operands[0], "sub")
        if difference is not None:
            if not (np.array_equal(difference.operands[0], x) and np.array_equal(difference.operands[1], c)):
                continue
        elif not (np.array_equal(equal.operands[0], x) and not graph.const_values(c).any()):
            continue
        out.op = "logaddexp"
        out.operands = [x, c]


def fuse_finite_tests(graph: RowGraph) -> None:
    """not(isinf(x) or isnan(x)) -> isfinite(x): one comparison instead of four operations."""
    for out in graph.stages:
        if out.kind != "map" or out.op != "not":
            continue
        either = whole_stage(graph, out.operands[0], "or")
        if either is None:
            continue
        inf, nan = whole_stage(graph, either.operands[0], "isinf"), whole_stage(graph, either.operands[1], "isnan")
        if inf is None or nan is None or not np.array_equal(inf.operands[0], nan.operands[0]):
            continue
        out.op = "isfinite"
        out.operands = [inf.operands[0]]


def sink_gathers(graph: RowGraph, limits: Limits) -> None:
    """Move elementwise stages that read a selection of one index space onto that index space."""
    is_const = graph.is_const()
    for stage in graph.stages:
        if stage.kind != "map" or stage.size <= limits.register_array:
            continue
        offsets = None
        source_size = 0
        variable = []
        for refs in stage.operands:
            const = is_const[ref_stage(refs)]
            sources = np.unique(ref_stage(refs))
            variable.append(not const.all())
            if const.all():
                continue
            source = graph.stages[int(sources[0])]
            same_space = offsets is None or (source.size == source_size and np.array_equal(ref_offset(refs), offsets))
            if const.any() or len(sources) != 1 or not same_space:
                offsets = None
                break
            offsets, source_size = ref_offset(refs), source.size
        if offsets is None or len(np.unique(offsets)) != stage.size:
            continue
        if source_size == stage.size and np.array_equal(offsets, np.arange(stage.size)):
            continue
        if source_size > limits.sink_growth * stage.size:
            continue
        # constants of an element nobody selected: those of the nearest selected element before it (or the first)
        selected = np.full(source_size, -1, dtype=np.int64)
        selected[offsets] = np.arange(stage.size)
        position = np.where(selected >= 0, np.arange(source_size), -1)
        nearest = np.maximum.accumulate(position)
        nearest[nearest < 0] = position[position >= 0][0]
        donor = selected[nearest]
        operands = []
        for refs, is_variable in zip(stage.operands, variable):
            if is_variable:
                operands.append(make_refs(int(ref_stage(refs)[0]), np.arange(source_size)))
            else:
                operands.append(refs[donor])
        stage.operands = operands
        stage.size = source_size
        _remap(graph, stage.index, offsets)


def interleave_copies(graph: RowGraph, limits: Limits) -> None:
    """Reorder stages that hold several copies of one index space, one copy after the other, so that the values
    of one index stand next to each other.

    A graph that carries a batch of vectors through the same network (a value and its tangents) keeps them as
    ``[copies, M]``: element ``c M + i`` of such a stage belongs with element ``i`` of the stages of ``M``
    elements, and no loop over ``M`` (or over the bands inside ``M``) holds both. Such a stage is found where an
    elementwise stage reads it at its own index modulo ``M``; it becomes ``[M, copies]`` (element ``i copies +
    c``), and the references to it follow. The stages then share loops instead of passing whole stages through
    the rows' slabs."""
    is_const = graph.is_const()
    spaces: dict[int, int] = {}   # stage -> M, where the stage is size / M copies of a space of M elements
    for stage in graph.stages:
        if stage.kind != "map" or stage.size <= limits.register_array:
            continue
        for refs in stage.operands:
            sources = np.unique(ref_stage(refs))
            if len(sources) != 1 or is_const[sources[0]]:
                continue
            source = graph.stages[int(sources[0])]
            if source.kind not in ("map", "dot", "reduce"):
                continue
            common = gcd(stage.size, source.size)
            position = np.arange(stage.size)
            fits = [m for m in _divisors_descending(common)
                    if m > limits.register_array and np.array_equal(ref_offset(refs) % m, position % m)]
            if not fits:
                continue
            for index, size in ((stage.index, stage.size), (source.index, source.size)):
                if fits[0] < size:
                    spaces[index] = min(spaces.get(index, fits[0]), fits[0])
    for index, space in sorted(spaces.items()):
        stage = graph.stages[index]
        copies = stage.size // space
        element = np.arange(stage.size)
        moved = (element % space) * copies + element // space   # element c M + i -> i copies + c
        order = np.argsort(moved)                               # the old element at every new place
        stage.operands = [np.ascontiguousarray(refs[order]) for refs in stage.operands]
        if stage.weights is not None:
            stage.weights = np.ascontiguousarray(stage.weights[order])
        if stage.bias is not None:
            stage.bias = np.ascontiguousarray(stage.bias[order])
        _remap(graph, index, moved)


def merge_repeated_elements(graph: RowGraph) -> None:
    """Keep one element per distinct operand tuple in elementwise stages whose tuples all repeat equally often."""
    for stage in graph.stages:
        if stage.kind != "map" or stage.size < 2:
            continue
        keys = np.stack(stage.operands, axis=1)
        _, first, inverse, counts = np.unique(keys, axis=0, return_index=True, return_inverse=True, return_counts=True)
        if len(first) == stage.size or counts.min() != counts.max():
            continue
        order = np.argsort(first)   # distinct tuples in the order of their first element
        rank = np.empty(len(first), dtype=np.int64)
        rank[order] = np.arange(len(first))
        kept = first[order]
        stage.operands = [refs[kept] for refs in stage.operands]
        stage.size = len(kept)
        _remap(graph, stage.index, rank[inverse.reshape(-1)])


def live_elements(graph: RowGraph) -> list[np.ndarray]:
    """Per stage, the elements that the output depends on."""
    live = [np.zeros(stage.size, dtype=np.bool_) for stage in graph.stages]

    def mark(refs: np.ndarray) -> None:
        stage_of = ref_stage(refs)
        for index in np.unique(stage_of):
            live[index][ref_offset(refs[stage_of == index])] = True

    mark(graph.output_refs)
    for stage in reversed(graph.stages):
        rows = live[stage.index]
        if stage.kind in ("map", "dot", "reduce") and rows.any():
            for refs in stage.operands:
                mark(refs[rows])
    return live


def _is_computed(stage: Stage) -> bool:
    return stage.kind in ("map", "dot", "reduce")


def _body_terms(graph: RowGraph, stages: list[int], members: int) -> int:
    total = 0
    for index in stages:
        stage = graph.stages[index]
        per_element = 1 if stage.kind == "map" else stage.terms * len(stage.operands)
        total += stage.size // members * per_element
    return total


def _aligned(graph: RowGraph, sizes: np.ndarray, is_const: np.ndarray, stages: list[int], members: int) -> bool:
    """True when the stages can share a loop of ``members`` iterations: every reference between them stays in the
    iteration, at a position that is the same in all iterations (or is replaced by a constant in some)."""
    in_group = np.zeros(len(graph.stages), dtype=np.bool_)
    in_group[stages] = True
    for index in stages:
        stage = graph.stages[index]
        inner = stage.size // members
        for refs in stage.operands:
            stage_of = ref_stage(refs)
            local = in_group[stage_of]
            if not local.any():
                continue
            block = np.maximum(sizes[stage_of] // members, 1)
            offset = ref_offset(refs)
            member = (np.arange(stage.size) // inner).reshape((-1,) + (1,) * (refs.ndim - 1))
            if np.any((offset // block != member) & local):
                return False
            position = np.where(local, (stage_of << _OFFSET_BITS) | (offset % block), -1).reshape(members, -1)
            reference = position.max(axis=0)
            if np.any((position != reference) & (position != -1)):
                return False
            replaced = (position == -1) & (reference != -1)
            if replaced.any() and not is_const[stage_of.reshape(members, -1)[replaced]].all():
                return False
    return True


def _divisors_descending(value: int) -> list[int]:
    return [d for d in range(value, 1, -1) if value % d == 0]


def build_groups(graph: RowGraph, live: list[np.ndarray], limits: Limits) -> list[Group]:
    """Fuse stages into loops, greedily from producers to consumers, and order the loops."""
    computed = [s.index for s in graph.stages if _is_computed(s) and live[s.index].any()]
    sizes = np.array([stage.size for stage in graph.stages], dtype=np.int64)
    is_const = graph.is_const()
    is_computed = np.zeros(len(graph.stages), dtype=np.bool_)
    is_computed[computed] = True
    producers = {}
    consumers = {index: set() for index in computed}
    for index in computed:
        sources = set()
        for refs in graph.stages[index].operands:
            sources.update(int(s) for s in np.unique(ref_stage(refs)) if is_computed[s])
        producers[index] = sorted(sources)
        for source in sources:
            consumers[source].add(index)

    group_of = {index: index for index in computed}       # stage -> key of its group
    stages_of = {index: [index] for index in computed}    # key -> stages
    members = {index: int(sizes[index]) for index in computed}

    def reaches(start: int, goal: int) -> bool:
        """True when group ``goal`` depends on group ``start`` through a third group."""
        seen = set()
        frontier = {group_of[c] for s in stages_of[start] for c in consumers[s]} - {start, goal}
        while frontier:
            key = frontier.pop()
            seen.add(key)
            for stage in stages_of[key]:
                for consumer in consumers[stage]:
                    target = group_of[consumer]
                    if target == goal:
                        return True
                    if target not in seen and target != start:
                        frontier.add(target)
        return False

    def shared_members(first: int, second: int) -> int:
        union = stages_of[first] + stages_of[second]
        for candidate in _divisors_descending(gcd(members[first], members[second])):
            if _body_terms(graph, union, candidate) > limits.body_terms:
                break
            if _aligned(graph, sizes, is_const, union, candidate):
                return candidate
        return 0

    for index in computed:
        while True:
            own = group_of[index]
            best = (0, -1)
            for key in {group_of[p] for p in producers[index]} - {own}:
                candidate = shared_members(key, own)
                if candidate > best[0] and not reaches(key, own):
                    best = (candidate, key)
            if best[0] == 0:
                break
            count, key = best
            for stage in stages_of[own]:
                group_of[stage] = key
            stages_of[key] = sorted(stages_of[key] + stages_of.pop(own))
            members[key] = count
            del members[own]

    for key, stages in stages_of.items():
        if all(sizes[s] <= limits.register_array for s in stages):
            members[key] = 1

    # Order: every group after the groups it reads (stage order breaks ties).
    waiting = {key: {group_of[p] for s in stages for p in producers[s]} - {key} for key, stages in stages_of.items()}
    ordered = []
    while waiting:
        key = min(k for k, needs in waiting.items() if not needs)
        del waiting[key]
        for needs in waiting.values():
            needs.discard(key)
        ordered.append(Group(members[key], stages_of[key]))
    return ordered


def accumulate_in_loops(plan: Plan, limits: Limits) -> None:
    """Move sums that run over the iterations of one looped stage into that loop.

    The sum ``c[e] = bias[e] + sum_p w[e, p] * x[p]`` over elements ``p`` of a stage computed in a loop is
    accumulated while the loop produces ``x``, and ``x`` is not stored for it. It qualifies with constant factors
    (a sum without factors, ReduceSum, has the factors one), few enough elements to live in registers, and terms
    that are the same elements of every iteration: all of them, or some (the sum of one element over the
    iterations)."""
    graph = plan.graph
    position = plan.group_position()
    is_const = graph.is_const()
    for group in list(plan.groups):
        if group.members != 1 or group.chain is not None:
            continue
        for index in list(group.stages):
            stage = graph.stages[index]
            product_sum = stage.kind == "dot" and stage.weights is not None
            if not (product_sum or stage.kind == "reduce" and stage.op == "sum") or stage.size > limits.register_array:
                continue
            refs = stage.operands[0]
            sources = np.unique(ref_stage(refs))
            if len(sources) != 1 or is_const[sources[0]]:
                continue
            source = graph.stages[int(sources[0])]
            loop = plan.groups[position[source.index]] if source.index in position else None
            if loop is None or loop.members == 1 or source.index not in loop.stages:
                continue
            # per element of the sum: the elements of an iteration that are its terms, the same in every iteration
            inner = source.size // loop.members
            terms = np.zeros((stage.size, source.size), dtype=bool)
            terms[np.arange(stage.size)[:, None], ref_offset(refs)] = True
            by_iteration = terms.reshape(stage.size, loop.members, inner)
            if terms.sum() != refs.size or not (by_iteration == by_iteration[:, :1]).all():
                continue
            group.stages.remove(index)
            loop.accumulated.append(index)
    plan.groups = [group for group in plan.groups if group.stages or group.accumulated]


def stream_consumers(plan: Plan, delay: int = 0) -> None:
    """Run a loop inside the loop that produces its input when it reads that input in step with it.

    A loop C whose only looped input is one stage T (one element per iteration of the producer loop P), read in
    rising order, does not need all of T: iteration b of C runs right after P has computed the last element b reads,
    and T becomes a ring. This keeps the per-row shared memory small (the hand-written head sorts each band as soon
    as its last value exists). ``delay`` is the number of further iterations that P may have run by then (a loop on
    the tensor cores computes several iterations at a time): the ring holds them too."""
    graph = plan.graph
    is_const = graph.is_const()

    def reader_positions(position: dict[int, int]) -> dict[int, set[int]]:
        """Stage -> positions of the groups that read it."""
        readers: dict[int, set[int]] = {}
        for index, at in position.items():
            for refs in graph.stages[index].operands:
                for source in np.unique(ref_stage(refs)):
                    readers.setdefault(int(source), set()).add(at)
        return readers

    position = plan.group_position()
    readers = reader_positions(position)
    output_sources = {int(source) for source in np.unique(ref_stage(graph.output_refs))}
    for consumer in list(plan.groups):
        at = plan.groups.index(consumer)
        if consumer.members == 1 or consumer.accumulated or consumer.streamed:
            continue
        first = np.full(consumer.members, np.iinfo(np.int64).max)
        last = np.full(consumer.members, -1)
        sources = set()
        early = True
        for index in consumer.stages:
            for refs in graph.stages[index].operands:
                stage_of = ref_stage(refs).reshape(consumer.members, -1)
                offset = ref_offset(refs).reshape(consumer.members, -1)
                for source in np.unique(stage_of):
                    source = int(source)
                    if is_const[source] or source in consumer.stages:
                        continue
                    producer = plan.groups[position[source]] if source in position else None
                    if producer is None or producer.members == 1:
                        early = early and (producer is None or position[source] < at)
                        continue
                    sources.add(source)
                    mask = stage_of == source
                    first = np.minimum(first, np.where(mask, offset, first[:, None]).min(axis=1))
                    last = np.maximum(last, np.where(mask, offset, -1).max(axis=1))
        if len(sources) != 1 or not early:
            continue
        source = sources.pop()
        producer = plan.groups[position[source]]
        if graph.stages[source].size != producer.members or readers[source] != {at} or source in output_sources:
            continue
        if source not in producer.stages:
            continue   # a sum that P accumulates exists only after P
        if producer is consumer or (last < 0).any() or (np.diff(last) < 0).any() or position[source] > at:
            continue
        # everything else C reads exists before P starts
        if not all(position.get(int(v), -1) < position[source] or int(v) == source or int(v) in consumer.stages
                   for index in consumer.stages for refs in graph.stages[index].operands
                   for v in np.unique(ref_stage(refs)) if not is_const[int(v)]):
            continue
        ring = 4
        while ring < int((last - first).max()) + 1 + delay:
            ring *= 2
        done = np.searchsorted(last, np.arange(producer.members), side="right")
        producer.streamed.append(Stream(consumer, source, done, ring))
        plan.ring[source] = ring
        plan.groups.remove(consumer)
        position = plan.group_position()   # the groups behind the consumer moved up
        readers = reader_positions(position)


def place_dense_chains(plan: Plan, limits: Limits) -> None:
    """Give every chain of wide product sums of the graph (tensor.find_dense_chains) a group of its own, behind the
    groups that compute the row values it reads and ahead of those that read its last sum. A group that does
    both is split around the chain; a chain that cannot be placed this way stays where its stages are."""
    graph = plan.graph
    members = {index: group.members for group in plan.groups for index in group.stages}
    chains = find_dense_chains(graph, _readers(graph), lambda stage: members.get(stage.index, 1) not in (1, stage.size),
                               limits.wide_sums, limits.sums_once)
    for chain in chains:
        inside, first = set(chain.stages), min(chain.stages)
        # what the chain reads of the row: the values of its first sum and those beside its sums
        before = {int(source) for source in np.unique(ref_stage(chain.vector))} | set(chain.sides)
        groups = []
        for group in plan.groups:
            rest = [index for index in group.stages if index not in inside]
            if group.chain is not None or len(rest) == len(group.stages):
                groups.append(group)   # another chain, or a group without stages of this one
            elif before & set(rest) and any(index > first for index in rest):
                # the group both computes values that the chain reads and comes after its start
                groups += [Group(group.members, part) for part in ([index for index in rest if index < first],
                                                                    [index for index in rest if index > first]) if part]
            elif rest:
                groups.append(Group(group.members, rest))
        position = 1 + max((number for number, group in enumerate(groups) if before & set(group.stages)), default=-1)
        readers = _readers(graph)
        reads_chain = [number for number, group in enumerate(groups)
                       if any(readers.get(index, set()) & set(group.stages) for index in inside)]
        if reads_chain and min(reads_chain) < position:
            continue
        groups.insert(position, Group(1, chain.stages, chain=chain))
        plan.groups = groups


def split_before_thread_loop(plan: Plan) -> tuple[RowGraph, RowGraph] | None:
    """The graph of a plan as two graphs for two kernels that run one after the other, or None.

    A loop on the tensor cores with its iterations by thread reads the row only through the row values of its
    product sums. Everything ahead of it becomes the first kernel, whose output row is these values in whole k-steps
    (zeros where padded); the loop and what follows it become the second kernel, which reads them as its input row.
    Each kernel then has its own number of threads: the first one is free to use many registers (wide product
    sums), the second one to run more warps. The stages keep their numbers in both graphs; a stage of the other
    part is not computed."""
    graph = plan.graph
    at = next((position for position, phase in sorted(plan.fragments.items()) if phase.by_thread), None)
    if at is None or graph.stages[0].kind != "input":
        return None
    phase = plan.fragments[at]
    behind = {index for group in plan.groups[at:] for part in [group] + [stream.group for stream in group.streamed]
              for index in part.stages + part.accumulated}
    is_const = graph.is_const()

    def ahead(refs: np.ndarray) -> bool:
        return any(not is_const[source] and int(source) not in behind for source in np.unique(ref_stage(refs)))

    if not any(ref_stage(vector).any() for vector in phase.vectors):
        return None   # the loop reads the input row itself: nothing to compute ahead of it
    # behind the cut, only the product sums of the loop read what is ahead of it
    if ahead(graph.output_refs) or any(ahead(refs) for index in behind - set(phase.dense) for refs in graph.stages[index].operands):
        return None
    zero = Stage(len(graph.stages), "const", "zero", 1, values=np.zeros(1, dtype=np.float32))
    graph.stages.append(zero)
    between, first = [], []
    for vector in phase.vectors:
        first.append(len(between))
        between += [int(ref) for ref in vector] + [int(make_refs(zero.index, 0))] * (-len(vector) % 8)
    one = RowGraph(list(graph.stages), graph.input_name, "between", graph.input_shape, (len(between),),
                   np.array(between, dtype=np.int64))
    stages = list(graph.stages)
    stages[0] = Stage(0, "input", "between", len(between))
    for index, dense in phase.dense.items():
        stage = graph.stages[index]
        terms = make_refs(0, first[dense.vector] + np.arange(stage.terms))
        stages[index] = replace(stage, operands=[np.ascontiguousarray(np.broadcast_to(terms, stage.operands[0].shape))])
    two = RowGraph(stages, "between", graph.output_name, (len(between),), graph.output_shape, graph.output_refs.copy())
    return one, two


def assign_storage(plan: Plan, far_from: int | None = None) -> None:
    """Choose the stages that live in the per-row slab and give them word ranges that do not overlap in time.
    With ``far_from`` the stages of that many words or more that are kept from one group to a later one, and
    the row values of that size that a chain copies, are the first words of the slab, ``far_words`` of them: a
    kernel whose rows keep much has these in device memory (emit.emit_kernel) and the others, the values that
    its loops read and write all the time, in shared memory."""
    plan.slab_words = plan.far_words = 0
    graph = plan.graph
    position = plan.group_position()
    parts = [(group_index, part) for group_index, group in enumerate(plan.groups)
             for part in [group] + [stream.group for stream in group.streamed]]
    members = {index: part.members for _, part in parts for index in part.stages}
    in_slab = set()
    last_use: dict[int, int] = {}
    read_outside: dict[int, list] = {}   # stage -> (reading group, elements) for the groups that do not compute it
    for group_index, group in parts:
        for index in group.stages:
            stage = graph.stages[index]
            for refs in stage.operands:
                stage_of = ref_stage(refs).reshape(group.members, -1)
                offset = ref_offset(refs).reshape(group.members, -1)
                for source in np.unique(stage_of):
                    source = int(source)
                    producer = graph.stages[source]
                    if producer.kind == "const" or source in group.stages:
                        continue
                    last_use[source] = max(last_use.get(source, -1), group_index)
                    alive = np.repeat(plan.live[index], refs.size // stage.size).reshape(stage_of.shape)   # of the reader
                    read_outside.setdefault(source, []).append((group_index, np.unique(offset[(stage_of == source) & alive])))
                    columns = (stage_of == source).any(axis=0)
                    same = (stage_of[:, columns] == source).all() and (offset[:, columns] == offset[0, columns]).all()
                    if not same or members.get(source, 1) > 1:
                        in_slab.add(source)

    # the threads of a chain read the row values beside its sums from the slabs of the rows they hold
    in_slab |= {index for _, part in parts if part.chain is not None for index in part.chain.sides}

    # Requests (words, first group, last group, where to record the offset). A fragment phase needs, while its
    # loop runs, the row values that its product sums read and the fragment stages that the rest of the body reads.
    requests = []
    chain_results = {phase.result for phase in plan.fragments.values() if isinstance(phase, DenseChain)}
    # A stage computed in a loop of which other code reads only some elements keeps only those. The loop and its
    # readers must be code of one row per thread: the layouts of the tensor cores address whole stages.
    special = set(plan.ring) | chain_results | {index for _, part in parts if part.chain is not None for index in part.chain.sides}
    for source in in_slab - special:
        readers = read_outside.get(source, [])
        phase = plan.fragments.get(position.get(source))   # in a loop on the tensor cores: the stages behind its sums
        if phase is not None and (isinstance(phase, DenseChain) or phase.by_thread or source not in phase.tail):
            continue
        if members.get(source, 1) == 1 or not readers:
            continue
        if any(group_index in plan.fragments for group_index, _ in readers):
            continue
        elements = np.unique(np.concatenate([elements for _, elements in readers]))
        if len(elements) < graph.stages[source].size:
            plan.kept[source] = elements
    for source in in_slab - chain_results:
        size = len(plan.kept[source]) if source in plan.kept else plan.ring.get(source, graph.stages[source].size)
        requests.append((size, position.get(source, -1), last_use[source], plan.slab_offset, source))
    for group_index, phase in plan.fragments.items():
        if isinstance(phase, DenseChain):
            # The threads that hold a row in a tile store its result: the elements that other code reads in the
            # rows' slabs (a later chain reads there too), the outputs of the kernel in the output rows.
            readers = read_outside.get(phase.result, [])
            if readers:
                phase.read = np.unique(np.concatenate([elements for _, elements in readers]))
                requests.append((graph.stages[phase.result].size, group_index, last_use[phase.result], plan.slab_offset, phase.result))
    for group_index, phase in plan.fragments.items():
        phase.vector_offset = [0] * len(phase.vectors)
        if not phase.by_thread:
            for number, vector in enumerate(phase.vectors):
                if isinstance(phase, DenseChain) and phase.narrow is None:
                    # the first sum reads the values where they are, when they stand next to each other in the
                    # input row or in a stage that is in the rows' slabs anyway
                    source, first = int(ref_stage(vector)[0]), int(ref_offset(vector)[0])
                    if np.array_equal(vector, make_refs(source, first + np.arange(len(vector)))):
                        if graph.stages[source].kind == "input":
                            phase.place = ("input", first)
                            continue
                        if (source in in_slab or source in chain_results) and source not in plan.ring:
                            phase.place = ("stage", source, first)   # also the result of the chain before
                            continue
                requests.append((-(-len(vector) // 8) * 8, group_index, group_index, phase.vector_offset, number))
            for index in phase.exported:
                requests.append((phase.inner[index], group_index, group_index, phase.scratch_offset, index))
            continue
        # Iterations by thread: the row values are read once, before the loop (what the loop stores may take
        # their place), and a k-step of eight values that stand next to each other in the slab is read there.
        # A k-step of values that stand next to each other in the input row is read from the input rows.
        resident = (in_slab | chain_results) - set(plan.ring)
        phase.direct = []
        for number, vector in enumerate(phase.vectors):
            steps = []
            for start in range(0, len(vector), 8):
                refs = vector[start:start + 8]
                source, first = int(ref_stage(refs)[0]), int(ref_offset(refs)[0])
                together = np.array_equal(refs, make_refs(source, first + np.arange(len(refs))))
                if graph.stages[source].kind == "input":
                    together = together and first + 8 <= graph.stages[source].size
                else:
                    together = together and len(refs) == 8 and source in resident
                steps.append((source, first) if together else None)
            phase.direct.append(steps)
            copied = 8 * sum(step is None for step in steps)
            if copied:
                requests.append((copied, group_index - 0.5, group_index - 0.5, phase.vector_offset, number))

    def first_fit(chosen: list, base: int) -> int:
        """First fit over the groups in order, from word ``base``; a range is free again after the last group
        that reads it. Returns the words used."""
        active: list[tuple[int, int, int]] = []   # (offset, words, last use)
        extent = 0
        for size, start, end, record, key in sorted(chosen, key=lambda r: (r[1], r[2])):
            active = [entry for entry in active if entry[2] >= start]
            words = -(-size // 4) * 4
            offset = 0
            for used_offset, used_words, _ in sorted(active):
                if offset + words <= used_offset:
                    break
                offset = max(offset, used_offset + used_words)
            active.append((offset, words, end))
            record[key] = base + offset
            extent = max(extent, offset + words)
        return extent

    between = [plan.slab_offset] + [phase.vector_offset for phase in plan.fragments.values() if isinstance(phase, DenseChain)]
    far = [request for request in requests
           if far_from is not None and request[0] >= far_from and any(request[3] is record for record in between)]
    plan.far_words = first_fit(far, 0)
    extent = first_fit([request for request in requests if not any(request is other for other in far)], plan.far_words)
    # 16-byte loads need a multiple of 4 words; 4 x odd spreads the rows of a warp over all shared-memory banks.
    if extent or plan.far_words:
        plan.slab_words = plan.far_words + extent + (4 if extent and (extent // 4) % 2 == 0 else 0)


def rewrite_graph(graph: RowGraph, limits: Limits, approximate_functions: bool = False, tensor_cores: bool = False) -> None:
    """The rewrites of the row graph for fusion (make_plan), in place."""
    if approximate_functions:
        fuse_log1p(graph)
        fuse_logaddexp(graph)
        fuse_finite_tests(graph)
        fuse_logsumexp(graph)
        fuse_exp_tanh(graph)
        fold_scales(graph)
        simplify_by_ranges(graph)
    if tensor_cores:
        fold_biases(graph)
    sink_gathers(graph, limits)
    merge_repeated_elements(graph)
    interleave_copies(graph, limits)


def fold_constant_terms(graph: RowGraph) -> None:
    """Terms of the product sums with constant factors whose row value is a constant for every element: into the
    bias, as at the import. A part of a graph (rowgraph.sub_graph) has such terms where the stage of the whole
    graph had elements with other values in their place (the tangent of a feature that does not depend on the
    direction is zero). A sum whose terms are all constants becomes a constant; a sum of constants that is no
    FP32 number alone keeps its terms."""
    for stage in graph.stages:
        if stage.kind != "dot" or stage.weights is None:
            continue
        refs = stage.operands[0]
        constant = graph.is_const()[ref_stage(refs)].all(axis=0)
        if not constant.any():
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            bias = np.sum(graph.const_values(refs[:, constant]) * stage.weights[:, constant], axis=1, dtype=np.float32)
            if stage.bias is not None:
                bias = (bias + stage.bias).astype(np.float32)
        if not np.isfinite(bias).all():
            continue
        if constant.all():
            stage.kind, stage.op, stage.operands, stage.weights, stage.bias, stage.values = "const", "", [], None, None, bias
        else:
            stage.operands = [np.ascontiguousarray(refs[:, ~constant])]
            stage.weights, stage.bias = np.ascontiguousarray(stage.weights[:, ~constant]), bias


# An output joins a part of the output row when the part computes this share of the products that the output needs.
PART_SHARE = 0.9


def output_parts(graph: RowGraph) -> list[np.ndarray]:
    """The outputs of the graph in parts (positions in the output row), by the products of the product sums that
    they need: an output belongs to the first part that computes at least PART_SHARE of its products anyway, and
    starts a part of its own otherwise (the outputs with the most products first).

    A graph that carries a value and its tangents through one network has one part per tangent (with the value's
    outputs in the first): each part alone (rowgraph.sub_graph) is the network of the value and of one tangent,
    with the regular loops that the layouts of the tensor cores take, where the whole graph has loops over
    stages of copies."""
    count = graph.output_refs.size
    words = -(-count // 64)
    need = [np.zeros((stage.size, words), dtype=np.uint64) for stage in graph.stages]   # bit o: output o needs it
    position = np.arange(count)
    bits = np.zeros((count, words), dtype=np.uint64)
    bits[position, position // 64] = np.uint64(1) << (position % 64).astype(np.uint64)
    stage_of, offset = ref_stage(graph.output_refs), ref_offset(graph.output_refs)
    for index in np.unique(stage_of):
        np.bitwise_or.at(need[index], offset[stage_of == index], bits[stage_of == index])
    for stage in sorted(graph.stages, key=lambda stage: -stage.index):
        own = need[stage.index]
        if stage.kind not in ("map", "dot", "reduce") or not own.any():
            continue
        for refs in stage.operands:
            terms = refs.reshape(stage.size, -1)
            carried = np.repeat(own, terms.shape[1], axis=0)
            source, element = ref_stage(terms).reshape(-1), ref_offset(terms).reshape(-1)
            for index in np.unique(source):
                mask = source == index
                np.bitwise_or.at(need[index], element[mask], carried[mask])
    sums = [stage for stage in graph.stages if stage.kind == "dot"]
    if not sums:
        return [position]
    needed = np.concatenate([np.unpackbits(need[stage.index].view(np.uint8), axis=1, bitorder="little")[:, :count]
                             for stage in sums], axis=0).astype(np.bool_)   # [elements of the product sums, outputs]
    products = np.concatenate([np.full(stage.size, stage.terms, dtype=np.int64) for stage in sums])
    same: dict[bytes, list[int]] = {}   # outputs with the same needs
    packed = np.packbits(needed, axis=0)
    for output in range(count):
        same.setdefault(packed[:, output].tobytes(), []).append(output)
    groups = sorted(same.values(), key=lambda group: -int(products[needed[:, group[0]]].sum()))   # most products first
    parts: list[tuple[np.ndarray, list[int]]] = []   # (elements of the product sums that the part computes, outputs)
    for group in groups:
        elements = needed[:, group[0]]
        work = int(products[elements].sum())
        home = next((part for part in parts if int(products[elements & part[0]].sum()) >= PART_SHARE * work), None)
        if home is None:
            parts.append((elements.copy(), list(group)))
        else:
            home[0][:] |= elements
            home[1].extend(group)
    return [np.array(sorted(outputs), dtype=np.int64) for _, outputs in parts]


def products(plan: Plan) -> tuple[int, int]:
    """Products per row of the plan's product sums: (on the tensor cores, in code of one row per thread)."""
    on = off = 0
    for position, group in enumerate(plan.groups):
        phase = plan.fragments.get(position)
        for part in [group] + [stream.group for stream in group.streamed]:
            for index in part.stages + part.accumulated:
                stage = plan.graph.stages[index]
                if stage.kind != "dot":
                    continue
                count = int(plan.live[index].sum()) * stage.terms
                cores = phase is not None and part is group and (isinstance(phase, DenseChain) or index in phase.dense)
                on, off = on + count * cores, off + count * (not cores)
    return on, off


def make_plan(graph: RowGraph, limits: Limits, approximate_functions: bool = False, tensor_cores: bool = False,
              rewritten: bool = False) -> Plan:
    """Rewrite the row graph for fusion and schedule it.

    approximate_functions  the kernel will compute log, exp, tanh and quotients approximately: forms that need the
                           exact functions are replaced first
    tensor_cores           product sums at the head of loop bodies run on the tensor cores (tensor.py)
    rewritten              the graph is a part of a graph that was rewritten already (split_before_thread_loop):
                           it is scheduled as it is
    """
    if not rewritten:
        rewrite_graph(graph, limits, approximate_functions, tensor_cores)
    live = live_elements(graph)
    plan = Plan(graph, live, build_groups(graph, live, limits))
    if tensor_cores:
        place_dense_chains(plan, limits)
    accumulate_in_loops(plan, limits)
    stream_consumers(plan, delay=CHUNK - 1 if tensor_cores else 0)
    if tensor_cores:
        for position, group in enumerate(plan.groups):
            others = [i for other in plan.groups if other is not group for i in other.stages + other.accumulated]
            if group.chain is not None:
                plan.fragments[position] = group.chain
                continue
            phase = find_thread_phase(graph, group, others) or find_fragment_phase(graph, group, others)
            if phase is not None:
                plan.fragments[position] = phase
    assign_storage(plan)
    return plan
