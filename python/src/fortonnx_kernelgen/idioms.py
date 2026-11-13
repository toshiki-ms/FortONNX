"""Forms of the exported graph that an approximate kernel computes the way a hand-written kernel would.

An exporter writes ``logsumexp`` as about ten operators with guards for infinite and NaN terms, and a gate
``exp(c tanh(k x))`` as four. For finite inputs each is one short expression:

  logsumexp        ``m + log(sum(exp(t - m)))``, m the largest term: one stage (a reduction)
  exp_scaled_tanh  ``exp(c tanh(k x)) = 2^(c' - 2 c' / (1 + 2^(k' x)))``: one stage. tanh in this plain form has an
                   absolute error of 1e-7, which the exponential turns into a relative error ``c`` 1e-7; the
                   form is used for ``|c| <= 16``
  scales           a product with a constant that only feeds another one is folded into it; the factors log2 e
                   and ln 2 that the base-2 instructions need in logsumexp go into the constants of the stages
                   before and behind it (``Stage.log2_units``); a product with a constant of a sum of product
                   sums goes into their weights and biases (``c (a + g b) = (c a) + g (c b)``)

The folded constants are rounded once instead of the values being rounded at every product, so the results differ
from the graph's by rounding; a row with infinite or NaN terms gives another result than the guarded form. The
passes are part of the approximate kernels only (``emit.Options.approximate_functions``).
"""

from __future__ import annotations

import numpy as np

from .rowgraph import RowGraph, Stage, make_refs, ref_stage, ref_offset, whole_stage

LOG2_E = 1.4426950408889634
LN_2 = 0.6931471805599453
GATE_SCALE = 16.0   # largest |c| of exp(c tanh(.)) written with the plain tanh form


def _readers(graph: RowGraph) -> dict[int, set[int]]:
    """Stage -> the stages that read it (-1: the output row), among the stages that the output depends on (the
    stages of a form that has been replaced no longer count)."""
    readers: dict[int, set[int]] = {int(source): {-1} for source in np.unique(ref_stage(graph.output_refs))}
    for stage in sorted(graph.stages, key=lambda stage: -stage.index):   # computed stages are in order of use
        if stage.index in readers:
            for refs in stage.operands:
                for source in np.unique(ref_stage(refs)):
                    readers.setdefault(int(source), set()).add(stage.index)
    return readers


def _constant_refs(graph: RowGraph, values: np.ndarray) -> np.ndarray:
    """References to a new constant stage with these values."""
    stage = Stage(len(graph.stages), "const", f"folded_{len(graph.stages):05d}", values.size,
                  values=np.asarray(values, dtype=np.float32).reshape(-1))
    graph.stages.append(stage)
    return make_refs(stage.index, np.arange(values.size)).reshape(values.shape)


def _scaling(graph: RowGraph, stage: Stage | None):
    """(source references, FP64 factors) of a stage that multiplies every element by a constant, else None."""
    if stage is None or stage.kind != "map":
        return None
    is_const = graph.is_const()
    if stage.op == "neg":
        return stage.operands[0], np.full(stage.size, -1.0)
    if stage.op in ("mul", "div"):
        first, second = stage.operands
        if is_const[ref_stage(second)].all():
            factors = graph.const_values(second).astype(np.float64)
            if stage.op == "div":
                with np.errstate(all="ignore"):
                    factors = 1.0 / factors
            return (first, factors) if np.isfinite(factors).all() else None
        if stage.op == "mul" and is_const[ref_stage(first)].all():
            return second, graph.const_values(first).astype(np.float64)
    return None


def _fits_fp32(factors: np.ndarray) -> bool:
    """Whether the factors are FP32 numbers: none overflows, none but zero rounds to zero."""
    with np.errstate(all="ignore"):
        rounded = np.asarray(factors, dtype=np.float64).astype(np.float32)
    return bool(np.isfinite(rounded).all() and ((rounded != 0) | (np.asarray(factors) == 0)).all())


def _set_scaling(graph: RowGraph, stage: Stage, source: np.ndarray, factors: np.ndarray) -> None:
    stage.op = "mul"
    stage.operands = [source, _constant_refs(graph, factors)]


def _source_stage(graph: RowGraph, refs: np.ndarray) -> Stage | None:
    """The stage whose elements, in order, are ``refs``."""
    stage = graph.stages[int(ref_stage(refs).reshape(-1)[0])]
    return stage if whole_stage(graph, refs, stage.op, stage.kind) is stage else None


def fuse_logsumexp(graph: RowGraph) -> None:
    """Replace ``log(|sum(exp(t - m))|) + m``, m the largest term (possibly through ``max(-inf, .)`` and
    ``where(isfinite(.), ., 0)``), by one logsumexp stage (after schedule.fuse_finite_tests)."""
    is_const = graph.is_const()

    def minus_infinity(refs: np.ndarray) -> bool:
        return bool(is_const[ref_stage(refs)].all() and (graph.const_values(refs) == -np.inf).all())

    for out in graph.stages:
        if out.kind != "map" or out.op != "add":
            continue
        for side in (0, 1):
            log, shift = whole_stage(graph, out.operands[side], "log"), out.operands[1 - side]
            if log is None:
                continue
            total_refs = log.operands[0]
            absolute = whole_stage(graph, total_refs, "abs")
            if absolute is not None:
                total_refs = absolute.operands[0]
            total = whole_stage(graph, total_refs, "sum", "reduce")
            if total is None:
                continue
            exp = graph.stages[int(ref_stage(total.operands[0]).reshape(-1)[0])]
            if exp.kind != "map" or exp.op != "exp" or (ref_stage(total.operands[0]) != exp.index).any():
                continue
            difference = whole_stage(graph, exp.operands[0], "sub")
            if difference is None:
                continue
            elements = ref_offset(total.operands[0])   # [size, terms] elements of the differences
            terms = difference.operands[0][elements]
            if not np.array_equal(difference.operands[1][elements], np.broadcast_to(shift[:, None], elements.shape)):
                continue
            peak = shift
            guard = whole_stage(graph, peak, "where")
            if guard is not None:
                test = whole_stage(graph, guard.operands[0], "isfinite")
                if test is None or not np.array_equal(test.operands[0], guard.operands[1]):
                    continue
                peak = guard.operands[1]
            floor = whole_stage(graph, peak, "max")
            if floor is not None:
                if minus_infinity(floor.operands[0]):
                    peak = floor.operands[1]
                elif minus_infinity(floor.operands[1]):
                    peak = floor.operands[0]
                else:
                    continue
            largest = whole_stage(graph, peak, "max", "reduce")
            if largest is None or not np.array_equal(largest.operands[0], terms):
                continue
            out.kind, out.op, out.operands = "reduce", "logsumexp", [terms]
            break


def fuse_exp_tanh(graph: RowGraph) -> None:
    """exp(c tanh(k x)) with constants c (at most GATE_SCALE) and k, each stage read by the next only, becomes one
    exp_scaled_tanh stage with the operands (x, c, k)."""
    readers = _readers(graph)
    for exp in graph.stages:
        if exp.kind != "map" or exp.op != "exp":
            continue
        outer = _source_stage(graph, exp.operands[0])
        scaled = _scaling(graph, outer)
        if scaled is None or readers.get(outer.index) != {exp.index}:
            continue
        tanh = whole_stage(graph, scaled[0], "tanh")
        if tanh is None or readers.get(tanh.index) != {outer.index} or np.abs(scaled[1]).max() > GATE_SCALE:
            continue
        argument, inner_factors = tanh.operands[0], np.ones(tanh.size)
        inner = _source_stage(graph, argument)
        absorbed = _scaling(graph, inner)
        # the factor of the argument goes into the form with 2 log2 e: only while that stays an FP32 number
        if absorbed is not None and readers.get(inner.index) == {tanh.index} and _fits_fp32(absorbed[1] * 2.0 * LOG2_E):
            argument, inner_factors = absorbed
        exp.op = "exp_scaled_tanh"
        exp.operands = [argument, _constant_refs(graph, scaled[1]), _constant_refs(graph, inner_factors)]


def _push_scale(graph: RowGraph, readers: dict[int, set[int]], parent: Stage, position: int, factors: np.ndarray,
                apply: bool) -> bool:
    """Whether the operand ``position`` of ``parent`` can take the factors of a product with constants: constants,
    or a stage that only ``parent`` reads and that is a product sum with constant factors (weights and bias), a
    product with constants, a sum or difference (both operands take them) or a product (one operand takes them).
    ``apply`` does it."""
    refs = parent.operands[position]
    if graph.is_const()[ref_stage(refs)].all():
        scaled_constants = graph.const_values(refs).astype(np.float64) * factors
        if apply:
            parent.operands[position] = _constant_refs(graph, scaled_constants)
        return _fits_fp32(scaled_constants)
    stage = _source_stage(graph, refs)
    if stage is None or readers.get(stage.index) != {parent.index}:
        return False
    if stage.kind == "dot" and stage.weights is not None:
        weights = stage.weights.astype(np.float64) * factors[:, None]
        bias = None if stage.bias is None else stage.bias.astype(np.float64) * factors
        if not _fits_fp32(weights) or (bias is not None and not _fits_fp32(bias)):
            return False
        if apply:
            stage.weights = weights.astype(np.float32)
            stage.bias = None if bias is None else bias.astype(np.float32)
        return True
    scaled = _scaling(graph, stage)
    if scaled is not None:
        if apply:
            _set_scaling(graph, stage, scaled[0], scaled[1] * factors)
        return _fits_fp32(scaled[1] * factors)
    if stage.kind != "map" or stage.op not in ("add", "sub", "mul"):
        return False
    first, second = (set(np.unique(ref_stage(refs)).tolist()) for refs in stage.operands)
    if first & second:
        return False   # a stage read by both operands would take the factors twice (or change the other operand)
    takes = [_push_scale(graph, readers, stage, operand, factors, False) for operand in (0, 1)]
    chosen = [0, 1] if stage.op != "mul" else [operand for operand in (0, 1) if takes[operand]][:1]
    if not chosen or not all(takes[operand] for operand in chosen):
        return False
    return all(_push_scale(graph, readers, stage, operand, factors, apply) for operand in chosen)


def _bypass(graph: RowGraph, stage: Stage, targets: np.ndarray) -> None:
    """Every reference to an element of the stage becomes the reference ``targets`` of that element."""
    def rewrite(refs: np.ndarray) -> np.ndarray:
        mask = ref_stage(refs) == stage.index
        if not mask.any():
            return refs
        out = refs.copy()
        out[mask] = targets[ref_offset(refs[mask])]
        return out

    for other in graph.stages:
        other.operands = [rewrite(refs) for refs in other.operands]
    graph.output_refs = rewrite(graph.output_refs)


def fold_scales(graph: RowGraph) -> None:
    """Fold chains of products with constants, move the base-2 factors of logsumexp into the constants of the
    stages before and behind it, and move products with constants into the product sums they scale (module
    text)."""
    readers = _readers(graph)
    for stage in list(graph.stages):
        scaled = _scaling(graph, stage)
        if scaled is None:
            continue
        before = _source_stage(graph, scaled[0])
        inner = _scaling(graph, before)
        if inner is not None and readers.get(before.index) == {stage.index} and _fits_fp32(inner[1] * scaled[1]):
            _set_scaling(graph, stage, inner[0], inner[1] * scaled[1])
    readers = _readers(graph)
    for stage in list(graph.stages):
        if stage.kind != "reduce" or stage.op != "logsumexp":
            continue
        terms_in_log2 = result_in_log2 = False
        before = graph.stages[int(ref_stage(stage.operands[0]).reshape(-1)[0])]
        scaled = _scaling(graph, before)
        if (scaled is not None and (ref_stage(stage.operands[0]) == before.index).all()
                and readers.get(before.index) == {stage.index} and _fits_fp32(scaled[1] * LOG2_E)):
            _set_scaling(graph, before, scaled[0], scaled[1] * LOG2_E)
            terms_in_log2 = True
        behind = [graph.stages[index] for index in readers.get(stage.index, set()) if index >= 0]
        behind_scaled = [_scaling(graph, reader) for reader in behind]
        if (-1 not in readers.get(stage.index, {-1}) and behind
                and all(s is not None and (ref_stage(s[0]) == stage.index).all() and _fits_fp32(s[1] * LN_2)
                        for s in behind_scaled)):
            for reader, (source, factors) in zip(behind, behind_scaled):
                _set_scaling(graph, reader, source, factors * LN_2)
            result_in_log2 = True
        stage.log2_units = (terms_in_log2, result_in_log2)
    readers = _readers(graph)
    for stage in list(graph.stages):
        scaled = _scaling(graph, stage)
        if scaled is None or stage.index not in readers:
            continue
        position = next(i for i, refs in enumerate(stage.operands) if refs is scaled[0])
        if _push_scale(graph, readers, stage, position, scaled[1], False):
            _push_scale(graph, readers, stage, position, scaled[1], True)
            _bypass(graph, stage, scaled[0])
            readers = _readers(graph)
