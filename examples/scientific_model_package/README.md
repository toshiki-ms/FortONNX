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

## FP64 and boolean verification tensors

The linear example remains float32. The validator also supports CSV-decoded
float64 and bool inputs/outputs, including multiple mixed-type outputs. Every
model tensor records its ONNX `element_type` (`tensor(float)`, `tensor(double)`
or `tensor(bool)`). Numeric CSV values are decoded directly into the declared
NumPy dtype; boolean tokens must be `true`, `false`, `1`, or `0` (case-insensitive).
Boolean outputs always use exact equality, regardless of numeric tolerances.

Existing homogeneous cases may retain `"dtype": "float32"` (or `"float64"` /
`"bool"`). Mixed cases record dtype by ONNX name, for example:

```json
"dtype": {
  "inputs": {"state": "float64", "enabled": "bool"},
  "outputs": {"values": "float64", "valid": "bool"}
}
```

For heterogeneous shapes, multi-column inputs or multiple outputs, both
`input_locator` and `expected_output_locator` accept named tensor locators:

```json
"input_locator": {"tensors": {
  "state": {"rows": [1, 2, 3], "columns": [0, 1], "shape": [3, 2]},
  "enabled": {"rows": [1, 2, 3], "columns": [2, 3], "shape": [3, 2]}
}},
"expected_output_locator": {"tensors": {
  "values": {"rows": [1, 2, 3], "columns": [4, 5], "shape": [3, 2]},
  "valid": {"rows": [1, 2, 3], "columns": [6, 7], "shape": [3, 2]}
}}
```

Rows are one-based after the header; columns are zero-based. `column` is the
single-column shorthand; `shape` defaults to the case's `shape`. Values are
flattened in row-major ONNX order. Expected values come from
`expected_output_artifact`, independently of the input CSV. Missing or
mismatched dtype records are rejected. Mapping-sensitive negative controls
swap compatible inputs, or permute values within individual mixed-type inputs;
they never swap or cast buffers of different dtypes.
