"""Value ranges of the elements of a row, and the simplifications that follow from them.

A hand-written kernel leaves out what the author knows to be without effect: the terms of a masked element, a floor
that a bounded value never reaches. The generator finds the same from the ranges of the values:

  known values   an element whose range is one value (``-inf - x`` for a finite x, ``exp(-inf)``, ``c * 0``) is a
                 constant: its readers read the constant, and the element is not computed
  max, min       ``max(a, b)`` is ``a`` when ``a >= b`` over the whole range in every element of the stage
  logaddexp      ``log(exp(a) + exp(b))`` is ``a`` when ``b - a <= -30 ln 2`` over the whole range in every element
                 (the dropped term is below 1e-9 of the sum)

The ranges are those of exact arithmetic for finite inputs, and a computed value counts as finite when it is
compared with an infinite constant (``max(-inf, x)`` is ``x``): a kernel simplified this way differs from the graph
for rows with infinite or NaN values and where an operator overflows or leaves its domain, and by the dropped
terms. The FP32 value of an element may lie a few units in the last place outside its range; that moves a
continuous result by as little, but ``floor`` and ``ceil`` jump: their ranges are one integer (and a relative
2^-20) wider on both sides unless the argument is a known value. It is used for the approximate kernels only
(``emit.Options.approximate_functions``).
"""

from __future__ import annotations

import numpy as np

from .rowgraph import RowGraph, Stage, make_refs, ref_offset, ref_stage

NEGLIGIBLE = -30.0 * np.log(2.0)   # difference of two logarithms below which the smaller term is dropped

_INCREASING = {
    "tanh": np.tanh, "exp": np.exp, "log": np.log, "log1p": np.log1p, "sqrt": np.sqrt,
    "relu": lambda a: np.maximum(a, 0.0), "sigmoid": lambda a: 1.0 / (1.0 + np.exp(-a)),
    "softplus": lambda a: np.logaddexp(a, 0.0),
}
_STEPS = {"floor": np.floor, "ceil": np.ceil}   # increasing, with jumps: see the module text
_COMPARISONS = {"less": "below", "less_equal": "below", "greater": "above", "greater_equal": "above"}


class _Ranges:
    """[lo, hi] of every element (FP64), filled stage by stage; unknown is [-inf, inf]."""

    def __init__(self, graph: RowGraph):
        self.graph = graph
        self.lo: dict[int, np.ndarray] = {}
        self.hi: dict[int, np.ndarray] = {}

    def of(self, refs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        stage_of, offset = ref_stage(refs), ref_offset(refs)
        lo, hi = np.empty(refs.shape), np.empty(refs.shape)
        for index in np.unique(stage_of):
            mask = stage_of == index
            lo[mask], hi[mask] = self.lo[int(index)][offset[mask]], self.hi[int(index)][offset[mask]]
        return lo, hi

    def branch(self, where: Stage, position: int) -> tuple[np.ndarray, np.ndarray]:
        """Range of the operand of a where stage in the elements that select it. Where the operand is the value
        that the condition compares with a constant, the condition bounds it (``where(x < c, c, x)``: x >= c)."""
        refs = where.operands[position]
        lo, hi = self.of(refs)
        condition = where.operands[0]
        for index in np.unique(ref_stage(condition)):
            test = self.graph.stages[index]
            if test.kind != "map" or test.op not in _COMPARISONS:
                continue
            mask = ref_stage(condition) == index
            elements = ref_offset(condition[mask])
            left, right = test.operands[0][elements], test.operands[1][elements]
            for value, bound, side in ((left, right, _COMPARISONS[test.op]),
                                       (right, left, "above" if _COMPARISONS[test.op] == "below" else "below")):
                bound_lo, bound_hi = self.of(bound)
                compared = (value == refs[mask]) & (bound_lo == bound_hi)
                if position == 2:   # the condition is false in the elements that take this operand
                    side = "above" if side == "below" else "below"
                part_lo, part_hi = lo[mask], hi[mask]
                if side == "below":
                    part_hi = np.where(compared, np.minimum(part_hi, bound_hi), part_hi)
                else:
                    part_lo = np.where(compared, np.maximum(part_lo, bound_lo), part_lo)
                lo[mask], hi[mask] = part_lo, part_hi
        return lo, hi

    def stage(self, stage: Stage) -> tuple[np.ndarray, np.ndarray]:
        unknown = np.full(stage.size, -np.inf), np.full(stage.size, np.inf)
        if stage.kind == "const":
            values = stage.values.astype(np.float64)
            return values, values.copy()
        if stage.kind == "input":
            return unknown
        if stage.is_bool or stage.op == "cast_float":
            return np.zeros(stage.size), np.ones(stage.size)
        with np.errstate(all="ignore"):
            if stage.kind == "dot":
                lo, hi = self.of(stage.operands[0])
                if stage.weights is not None:
                    weights = stage.weights.astype(np.float64)
                    corners = [weights * lo, weights * hi]
                else:
                    other_lo, other_hi = self.of(stage.operands[1])
                    corners = [lo * other_lo, lo * other_hi, hi * other_lo, hi * other_hi]
                bias = 0.0 if stage.bias is None else stage.bias.astype(np.float64)
                lo, hi = np.min(corners, axis=0).sum(axis=1) + bias, np.max(corners, axis=0).sum(axis=1) + bias
            elif stage.kind == "reduce":
                lo, hi = self.of(stage.operands[0])
                if stage.op == "prod":
                    return unknown
                if stage.op == "logsumexp":
                    # natural logarithms inside; terms and result may be in units of ln 2
                    unit_in, unit_out = (np.log(2.0) if flag else 1.0 for flag in stage.log2_units)
                    lo = np.max(lo, axis=1) * unit_in / unit_out
                    hi = (np.max(hi, axis=1) * unit_in + np.log(hi.shape[1])) / unit_out
                else:
                    function = {"sum": np.sum, "max": np.max, "min": np.min}[stage.op]
                    lo, hi = function(lo, axis=1), function(hi, axis=1)
            elif stage.op == "where":
                (lo_then, hi_then), (lo_else, hi_else) = self.branch(stage, 1), self.branch(stage, 2)
                lo, hi = np.minimum(lo_then, lo_else), np.maximum(hi_then, hi_else)
            elif stage.op in _INCREASING:
                lo, hi = (_INCREASING[stage.op](bound) for bound in self.of(stage.operands[0]))
            elif stage.op in _STEPS:
                a_lo, a_hi = self.of(stage.operands[0])
                margin = np.where(a_lo == a_hi, 0.0, 1.0 + 2.0 ** -20 * np.maximum(np.abs(a_lo), np.abs(a_hi)))
                lo, hi = _STEPS[stage.op](a_lo) - margin, _STEPS[stage.op](a_hi) + margin
            elif stage.op == "exp_scaled_tanh":
                scale = np.abs(self.of(stage.operands[1])[0])
                lo, hi = np.exp(-scale), np.exp(scale)
            elif stage.op in ("sin", "cos"):
                lo, hi = -np.ones(stage.size), np.ones(stage.size)
            elif stage.op == "neg":
                hi, lo = (-bound for bound in self.of(stage.operands[0]))
            elif stage.op == "abs":
                a_lo, a_hi = self.of(stage.operands[0])
                lo = np.where(a_lo >= 0, a_lo, np.where(a_hi <= 0, -a_hi, 0.0))
                hi = np.maximum(np.abs(a_lo), np.abs(a_hi))
            elif stage.op in ("add", "sub", "mul", "div", "max", "min", "logaddexp"):
                (a_lo, a_hi), (b_lo, b_hi) = self.of(stage.operands[0]), self.of(stage.operands[1])
                if stage.op == "add":
                    lo, hi = a_lo + b_lo, a_hi + b_hi
                elif stage.op == "sub":
                    lo, hi = a_lo - b_hi, a_hi - b_lo
                elif stage.op == "max":
                    lo, hi = np.maximum(a_lo, b_lo), np.maximum(a_hi, b_hi)
                elif stage.op == "min":
                    lo, hi = np.minimum(a_lo, b_lo), np.minimum(a_hi, b_hi)
                elif stage.op == "logaddexp":
                    lo, hi = np.logaddexp(a_lo, b_lo), np.logaddexp(a_hi, b_hi)
                else:
                    if stage.op == "div":   # by a constant other than zero only
                        constant = (b_lo == b_hi) & (b_lo != 0) & np.isfinite(b_lo)
                        b_lo, b_hi = np.where(constant, 1.0 / b_lo, np.nan), np.where(constant, 1.0 / b_hi, np.nan)
                    corners = [a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi]
                    lo, hi = np.min(corners, axis=0), np.max(corners, axis=0)
            else:
                return unknown
        undefined = np.isnan(lo) | np.isnan(hi)
        return np.where(undefined, -np.inf, lo), np.where(undefined, np.inf, hi)


def simplify_by_ranges(graph: RowGraph) -> None:
    """Rewrite the references to elements whose value follows from the ranges (see the module text). The stages
    keep their elements; those that nothing reads any more are not computed (schedule.live_elements)."""
    ranges = _Ranges(graph)
    replaced: dict[int, np.ndarray] = {}    # stage -> per element, the reference to read instead (-1: itself)
    known: dict[int, np.ndarray] = {}       # stage -> elements with one known value
    known_stage = len(graph.stages)         # a constant stage appended below holds the known values
    known_values: list[np.float32] = []
    known_position: dict[int, int] = {}

    def resolve(refs: np.ndarray) -> np.ndarray:
        stage_of = ref_stage(refs)
        out = refs
        for index in np.unique(stage_of):
            if int(index) in replaced:
                mask = stage_of == index
                target = replaced[int(index)][ref_offset(refs[mask])]
                if out is refs:
                    out = refs.copy()
                out[mask] = np.where(target >= 0, target, refs[mask])
        return out

    is_const = np.append(graph.is_const(), True)   # the last entry: the stage of known values
    largest = float(np.finfo(np.float32).max)

    def finite(refs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Range with the bounds of computed values inside the finite numbers."""
        lo, hi = ranges.of(refs)
        computed = ~is_const[ref_stage(refs)]
        return np.where(computed, np.maximum(lo, -largest), lo), np.where(computed, np.minimum(hi, largest), hi)

    for stage in graph.stages:   # constants first: earlier passes may have appended some
        if stage.kind == "const":
            ranges.lo[stage.index], ranges.hi[stage.index] = ranges.stage(stage)
    for stage in list(graph.stages):
        if stage.kind == "const":
            continue
        stage.operands = [resolve(refs) for refs in stage.operands]
        lo, hi = ranges.stage(stage)
        ranges.lo[stage.index], ranges.hi[stage.index] = lo, hi
        if stage.kind not in ("map", "dot", "reduce") or stage.is_bool:
            continue
        target = np.full(stage.size, -1, dtype=np.int64)
        if stage.kind == "map" and stage.op in ("max", "min", "logaddexp"):
            first, second = stage.operands
            (a_lo, a_hi), (b_lo, b_hi) = finite(first), finite(second)
            margin = NEGLIGIBLE if stage.op == "logaddexp" else 0.0
            with np.errstate(invalid="ignore"):
                if stage.op == "min":
                    takes_first, takes_second = a_hi - b_lo <= margin, b_hi - a_lo <= margin
                else:
                    takes_first, takes_second = b_hi - a_lo <= margin, a_hi - b_lo <= margin
            # for the whole stage only: a stage whose elements read different stages no longer lines up with
            # its neighbours in a loop (emit._Body.map_element drops infinite pads per pattern of iterations)
            if takes_first.all():
                target = first.copy()
            elif takes_second.all():
                target = second.copy()
        for value in np.unique(lo[(lo == hi) & (target < 0)]):
            bits = int(np.float32(value).view(np.uint32))
            if bits not in known_position:
                known_position[bits] = len(known_values)
                known_values.append(np.float32(value))
                ranges.lo[known_stage] = ranges.hi[known_stage] = np.array(known_values, dtype=np.float64)
            target = np.where((lo == hi) & (lo == value), make_refs(known_stage, known_position[bits]), target)
        if (target >= 0).any():
            replaced[stage.index] = target
    if not replaced:
        return
    graph.output_refs = resolve(graph.output_refs)
    graph.stages.append(Stage(known_stage, "const", "known_values", max(len(known_values), 1),
                              values=np.array(known_values or [0.0], dtype=np.float32)))
