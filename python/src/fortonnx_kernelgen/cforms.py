"""C texts shared by the emitters: literals, index expressions and the approximate forms of functions."""

from __future__ import annotations

import numpy as np

# Map operators with an approximate form (Options.approximate_functions): C expression, and the helper functions
# that the kernel source then defines. The forms are those of hand-written kernels:
# ex2.approx has a relative error of 2^-22.5, lg2.approx an absolute error of 2^-22.6 of the logarithm, rcp.approx
# of 1 unit in the last place.
#   exp(x)          = ex2(x log2 e)                        relative error about |x| 2^-24 + 2^-22.5
#   log(x)          = lg2(x) ln 2                          absolute error about 1.2e-7 (no relative bound near 1)
#   tanh(x)         polynomial below 0.625, 1 - 2 / (1 + ex2(2 |x| log2 e)) above: relative error about 1e-7
#   a / b           = a rcp(b)                             about 2 units in the last place; where 1 / b overflows
#                                                          or vanishes the product is not the quotient
#                                                          (0 / 1e-40 gives NaN): finite, normal divisors only
#   max, min, relu  fmax, fmin                             a NaN operand is ignored (the graph's result is NaN)
#   logaddexp(a, b) = max + 2 atanh(e / (2 + e)), e = ex2(-|a - b| log2 e): relative error about 1e-7
APPROXIMATE_FUNCTIONS = {
    "exp": "approx_ex2({0} * 1.44269504f)",
    "log": "(approx_lg2({0}) * 0.693147181f)",
    "tanh": "approx_tanh({0})",
    "div": "({0} * approx_rcp({1}))",
    "logaddexp": "approx_logaddexp({0}, {1})",
    "max": "fmaxf({0}, {1})",
    "min": "fminf({0}, {1})",
    "relu": "fmaxf({0}, 0.0f)",
    "exp_scaled_tanh": "approx_exp_tanh({0}, {1} * 1.44269504f, {2} * 2.88539008f)",
}
APPROXIMATE_HELPERS = """
// Approximate instructions of the hardware and the functions built on them (errors: fortonnx_kernelgen/cforms.py).
__device__ __forceinline__ float approx_ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float approx_lg2(float x) {
  float y;
  asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float approx_rcp(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// Branch-free tanh: odd polynomial below 0.625, 1 - 2 / (1 + 2^(2 |x| log2 e)) above.
__device__ __forceinline__ float approx_tanh(float a) {
  const float x = fabsf(a), u = x * x;
  float p = -0.005696116480976343f;
  p = fmaf(p, u, 0.020628780126571655f);
  p = fmaf(p, u, -0.053735263645648956f);
  p = fmaf(p, u, 0.13331355154514313f);
  p = fmaf(p, u, -0.33333274722099304f);
  const float small = fmaf(x * u, p, x);
  const float big = 1.0f - 2.0f * approx_rcp(1.0f + approx_ex2(2.885390081777927f * x));
  return copysignf(x < 0.625f ? small : big, a);
}
// exp(c tanh(k x)) = 2^(cl - 2 cl / (1 + 2^(kl x))), cl = c log2 e, kl = 2 k log2 e (idioms.py).
__device__ __forceinline__ float approx_exp_tanh(float x, float cl, float kl) {
  return approx_ex2(fmaf(-2.0f * cl, approx_rcp(1.0f + approx_ex2(x * kl)), cl));
}
// log(exp(a) + exp(b)) = max(a, b) + log1p(e), e = exp(-|a - b|) in (0, 1]; log1p(e) = 2 atanh(e / (2 + e)).
__device__ __forceinline__ float approx_logaddexp(float a, float b) {
  const float e = approx_ex2(-1.4426950408889634f * fabsf(a - b));
  const float s = e * approx_rcp(2.0f + e), s2 = s * s;
  float p = 1.0f / 17.0f;
  p = fmaf(p, s2, 1.0f / 15.0f);
  p = fmaf(p, s2, 1.0f / 13.0f);
  p = fmaf(p, s2, 1.0f / 11.0f);
  p = fmaf(p, s2, 1.0f / 9.0f);
  p = fmaf(p, s2, 1.0f / 7.0f);
  p = fmaf(p, s2, 1.0f / 5.0f);
  p = fmaf(p, s2, 1.0f / 3.0f);
  return fmaxf(a, b) + 2.0f * fmaf(s * s2, p, s);
}
"""


def comment_text(name: str) -> str:
    """A name of the model as it may stand in a comment of the generated source: printable ASCII, no backslash
    (an ONNX name may hold line breaks, which would end the comment)."""
    return "".join(c if " " <= c <= "~" and c != "\\" else "?" for c in name)


def float_literal(value: float) -> str:
    value = np.float32(value)
    if np.isnan(value):
        return "NAN"
    if np.isinf(value):
        return "INFINITY" if value > 0 else "(-INFINITY)"
    text = f"{float(value):.9g}"
    if "." not in text and "e" not in text:
        text += ".0"
    return f"({text}f)" if text.startswith("-") else text + "f"


def affine(values: np.ndarray) -> tuple[int, int] | None:
    """(a, b) with values[m] == a + b * m, or None."""
    if len(values) == 1:
        return int(values[0]), 0
    step = int(values[1] - values[0])
    if np.array_equal(values, values[0] + step * np.arange(len(values))):
        return int(values[0]), step
    return None


def index_text(base: int, step: int) -> str:
    if step == 0:
        return str(base)
    return f"{base} + {step} * m" if base else f"{step} * m"
