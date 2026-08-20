# Compatibility policy

The portable Fortran layer targets Fortran 2018 plus `iso_c_binding` and is
kept free of compiler extensions. The intended CPU compiler set is:

| Compiler | Command | CPU | CUDA Fortran arrays |
|---|---|---:|---:|
| GNU Fortran | `gfortran` | yes | no |
| LLVM Flang | `flang` | yes | no |
| Intel Fortran | `ifx` | yes | no |
| NVIDIA Fortran | `nvfortran` | yes | yes |

GPU-enabled libraries can also be built with a CPU-oriented compiler when the
application obtains device pointers and streams through another interoperable
layer. The low-level `type(c_ptr)` plus ONNX-shape form of `runtime%bind` is
compiler-neutral and supports any positive tensor rank.
In an NVFortran GPU build, the same `fortonnx` module adds overloads that
directly accept CUDA Fortran `device` arrays of ranks 1 through 7. Input and
output arrays of different ranks use `fortonnx_device_tensor(array)` views.
The vendor-specific declarations are removed by preprocessing for all other
builds.

ONNX Runtime C headers and the runtime library must come from the same release.
FortONNX does not vendor provider libraries. The v0.1 implementation is tested
with ONNX Runtime 1.29 and exports ONNX opset 18 models.

Installed FortONNX shared libraries use `$ORIGIN` as their only project-added
RPATH. Build-time ONNX Runtime and provider locations are not stored in
`fortonnx-config`; applications supply nonstandard dependency paths through
their environment when linking or running.
