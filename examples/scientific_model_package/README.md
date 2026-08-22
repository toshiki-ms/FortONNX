# Scientific model package example

This example packages an ordinary ONNX graph with the scientific contract
defined by the [Scientific ONNX Model Package
Specification](../../docs/scientific-model-package/README.md). The graph itself
uses ordinary ONNX operators and requires no package-specific runtime behavior.

The model computes the illustrative linear response

```text
response [K] = 0.8 * temperature_anomaly [K]
             + 0.01 [K m^2 W^-1] * radiative_forcing [W m^-2]
```

The coefficients are deliberately simple and are not scientifically validated.
The example demonstrates packaging and verification, not a physical prediction
model.

## Generate the package

The generator requires Python, NumPy, and ONNX:

```sh
make scientific-package-files PYTHON=python3
```

The generated package is written to
`build/examples/scientific_model_package/`. Generated ONNX and report files
remain under `build/` and are not source-controlled.

The generator creates:

```text
scientific_model_package/
├── README.md
├── LICENSE
├── CITATION.txt
├── package-description.json
├── package-description.sha256
├── conformance-report.json
├── conformance-report.sha256
├── models/linear-response.onnx
├── metadata/abstract-model-mapping.md
├── verification/nominal.csv
├── verification/reference-report.json
├── examples/reference_run.py
└── examples/run.f90
```

JSON and CSV are choices made by this example. The specification does not
require either format.

## Validate the package

Run the independent validator with:

```sh
make scientific-package-validate PYTHON=python3
```

This writes a machine-readable report and a short summary beside the generated
package under `build/examples/`. The validator checks the supplied description
digest, the direct-JSON mapping, required fields and references, safe paths,
artifact sizes and digests, ONNX structure and signature, scientific interface
and process contracts, Model Card coverage, profiles, claims, and provenance.
With `--run-verification`, it also decodes the declared CSV cases, executes the
ONNX graph using validator-owned code, compares every output, and performs the
same-shaped input-swap negative control. It never imports or executes packaged
source code.

The Make target uses the generated detached digest record for a local
consistency demonstration. For a distribution decision, supply a digest from a
separately trusted release record or channel:

```sh
python3 examples/scientific_model_package/validate_package.py PACKAGE_DIRECTORY \
  --expected-description-digest TRUSTED_SHA256 \
  --run-verification \
  --report validation-report.json \
  --summary validation-summary.txt
```

The bundled description adapter supports the example's declared direct JSON
mapping, and the bundled verification decoder supports its declared CSV layout.
These are implementations selected for this example, not formats required by
the package specification. An applicable check that cannot be performed is
reported as `not-run` and prevents an overall conformance result.

## Run the Fortran known-answer check

With a configured CPU build:

```sh
make scientific-package-example PYTHON=python3 BACKEND=cpu
```

The target generates the package, compiles `examples/run.f90`, loads the ONNX
graph, reads the CSV known-answer cases, binds tensors by their exact ONNX
names, runs inference, and checks the documented tolerance.

The package can also be generated directly:

```sh
python3 examples/scientific_model_package/build_package.py OUTPUT_DIRECTORY
```

The destination must not already exist. This prevents an existing package or
unrelated files from being overwritten.
