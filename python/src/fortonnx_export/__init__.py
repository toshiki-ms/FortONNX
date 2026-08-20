"""Fail-closed export of supported JAX programs to ONNX."""

from .jaxpr import (
    ExportResult,
    JaxOnnxExportError,
    UnsupportedJaxPrimitive,
    export_jax_to_onnx,
    supported_primitives,
)

__all__ = [
    "ExportResult",
    "JaxOnnxExportError",
    "UnsupportedJaxPrimitive",
    "export_jax_to_onnx",
    "supported_primitives",
]

