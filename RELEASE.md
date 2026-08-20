# Release checklist

FortONNX source releases must be created from a clean Git commit, never by
archiving the working directory directly.

1. Run CPU, exporter, CUDA, and TensorRT tests in the supported environments.
2. Run `make distclean` to remove `config.mk`, builds, models, caches, compiler
   modules, and Python metadata.
3. Run `make release-check`. Set `FORTONNX_PRIVATE_PATTERNS` to any additional
   literal text that must not occur in the release.
4. Review `git status --short` and `git diff --check`.
5. Confirm the version in `pyproject.toml`, `CHANGELOG.md`, and
   `tools/fortonnx-config`.
6. Create the source archive from Git, for example:

   ```sh
   git archive --format=tar.gz --prefix=fortonnx-0.1.0/ \
     -o fortonnx-0.1.0.tar.gz v0.1.0
   ```

Binary releases need a separate reproducible build environment. Do not publish
locally compiled libraries: compiler objects, ELF RPATH entries, and dependency
links may reveal build-host paths or depend on machine-specific installations.

ONNX Runtime, CUDA, cuDNN, and TensorRT libraries are external dependencies and
must not be copied into a FortONNX source release.

