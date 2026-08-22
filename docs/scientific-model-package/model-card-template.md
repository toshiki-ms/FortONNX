# <Model name>

Specification: `scientific-onnx-model-package` `0.1-draft`<br>
Version: `<package version>`<br>
Release status: `<research | validated | operational | deprecated>`<br>
Release date: `<date>`<br>
Package identifier: `<persistent identifier>`

<State the scientific system, the predicted physical quantities, the spatial
or temporal scale, and the boundary of the intended use. State whether this is
a surrogate, emulator, parameterization, forecast model, or another learned
model.>

Responsible organization: `<organization>`<br>
Contact: `<maintained contact>`<br>
License: `<license and scope>`<br>
Preferred citation: `<complete citation>`

## Intended use

<Describe the supported scientific tasks and the expertise required to use and
interpret the model.>

| Condition | Supported range or category | Evidence |
|---|---|---|
| Spatial domain | `<value>` | `<claim or evidence reference>` |
| Spatial resolution | `<value>` | `<claim or evidence reference>` |
| Time step or sampling interval | `<value>` | `<claim or evidence reference>` |
| Forecast or rollout horizon | `<value>` | `<claim or evidence reference>` |
| State and forcing range | `<value>` | `<claim or evidence reference>` |
| Boundary conditions | `<value>` | `<claim or evidence reference>` |
| Operating mode | `<single-pass | iterative | coupled>` | `<process graph>` |
| Runtime scope | `<tested configurations>` | `<verification/evidence reference>` |

Required initialization, forcing, boundary, and coupling assumptions:

- `<assumption>`
- `<assumption>`

## Unsupported uses

- `<unsupported regime, resolution, parameter range, extrapolation, horizon,
  intervention, or decision>`
- `<unsupported safety-critical or operational use>`

Required human oversight: `<oversight or not required with reason>`

## Package contents

Machine-readable package description: `<path or immutable URI>`<br>
Description format and version: `<declared format>`<br>
Abstract-model adapter or mapping: `<identifier, version, and digest>`<br>
Trusted distribution record or signature: `<immutable location>`

| Artifact ID | Path or immutable URI | Role | Required stage | Format and version | Reader | Digest |
|---|---|---|---|---|---|---|
| `<id>` | `models/<model>.onnx` | Primary inference graph | Inference | `<ONNX version>` | `<runtime>` | `<algorithm:digest>` |
| `<id>` | `<location>` | `<role>` | `<stage>` | `<format>` | `<reader/version>` | `<algorithm:digest>` |

The authoritative representation of equivalent artifacts is `<artifact ID>`.
Equivalence was checked by `<procedure and tolerance>`.

## Quick start

Tested configuration: `<runtime, version, provider, hardware class, precision>`

1. Obtain the trusted expected digest for the package description, verify it,
   and then verify every indexed required artifact.
2. Read `<supporting artifacts>` using `<declared readers>`.
3. Construct physical inputs as specified in `<domain and tensor contracts>`.
4. Apply `<ordered preprocessing transformations>`.
5. Bind `<exact ONNX names>` and run `models/<model>.onnx`.
6. Apply `<ordered postprocessing and corrections>`.
7. Interpret the result using `<units, axes, coordinates, and time convention>`.
8. Reject or flag the result when `<validity and domain checks>` fail.
9. Run verification case `<case ID>` and require `<comparison criteria>`.

Complete executable example: `<path or immutable URI>`

## Inputs and outputs

### ONNX interface

| ONNX name | Direction | Role | Element type | Logical shape | Axes | Transformations |
|---|---|---|---|---|---|---|
| `<exact name>` | `<input/output>` | `<state/forcing/...>` | `<ONNX type>` | `<ordered dimensions>` | `<ordered axis IDs>` | `<ordered transform IDs or none>` |

### Scientific quantities

| Tensor and channel selector | Quantity | Physical units | Graph units | Location and convention | Valid-value behavior |
|---|---|---|---|---|---|
| `<tensor[channel]>` | `<defined quantity>` | `<units>` | `<units or dimensionless>` | `<grid/mesh/vertical/time/sign convention>` | `<reject/clip/mask/...>` |

### Domain coverage

| Tensor or channel | Admissible domain | Training support | Evaluation support | Intended-use support |
|---|---|---|---|---|
| `<tensor/channel>` | `<domain ID>` | `<domain ID>` | `<domain ID or not evaluated>` | `<domain ID>` |

### Dimensions and axes

| Axis ID | Meaning | Length | Coordinates | Orientation and order | Boundary, staggering, and mask |
|---|---|---:|---|---|---|
| `<axis>` | `<meaning>` | `<fixed or symbolic>` | `<coordinate reference>` | `<convention>` | `<behavior>` |

Tensor shapes and axes are stated in ONNX logical order. For `<language or
binding>`, the corresponding memory-layout mapping is `<mapping>`.

Missing values, padding, masks, NaNs, and infinities are handled as follows:

- `<tensor or channel>: <exact behavior and interactions>`

## Transformations and effective process

| Transform ID | Location | Input | Output | Definition | Parameters | Numerical behavior |
|---|---|---|---|---|---|---|
| `<id>` | `<inside graph/external>` | `<port>` | `<port>` | `<formula or algorithm>` | `<literal or artifact locator>` | `<precision, rounding, failure>` |

Parameters fitted from data were derived from `<training partition>` using
`<method>`. `<No evaluation/deployment data were used | exact transductive use
and its consequence>`.

Effective process graph: `<process graph ID and authoritative locator>`

```text
<physical inputs>
  -> <decode/unit conversion/feature construction>
  -> <ONNX graph and exact ports>
  -> <reconstruction/correction>
  -> <physical outputs and validity checks>
```

Each operation in this path is applied exactly once. Required behavior not
visible in the diagram is `<none or exact references>`.

## Coordinates, time, and coupling

Coordinate reference system: `<definition>`<br>
Grid or mesh: `<topology, indexing, entity locations, seams, poles, periodicity>`<br>
Vertical coordinate: `<definition and coefficient artifacts>`<br>
Regridding: `<source, destination, algorithm, order, boundary and conservation>`<br>
Time reference and calendar: `<definition>`<br>
Sampling, averaging, or accumulation: `<definition>`

### Iteration or coupling contract

Operating mode: `<not applicable with reason | iterative/coupled>`

| Order | Action | Input state/time | Output state/time | Failure behavior |
|---:|---|---|---|---|
| 1 | `<initialize/read forcing/...>` | `<state>` | `<state>` | `<behavior>` |
| 2 | `<invoke process graph>` | `<state>` | `<prediction>` | `<behavior>` |
| 3 | `<update/correct/exchange>` | `<prediction>` | `<next state>` | `<behavior>` |

Step duration and timestamp semantics: `<definition>`<br>
Restart behavior: `<definition>`<br>
Longest evaluated rollout: `<duration/steps and domain>`

## Scientific validation

| Claim ID and statement | Evaluation domain | Metric and units | Result and uncertainty | Baseline | Acceptance criterion | Conclusion | Evidence |
|---|---|---|---|---|---|---|---|
| `<id: falsifiable claim>` | `<domain>` | `<fully defined metric>` | `<result>` | `<identity/result>` | `<criterion; predeclared or retrospective>` | `<passed/failed/inconclusive/not evaluated>` | `<artifact locator>` |

Metric weighting, masks, aggregation, and reduction order: `<definition>`<br>
Independent case unit and count: `<definition and count>`<br>
Uncertainty-estimation method, interval level, and resampling unit:<br>
`<definition>`
Reference uncertainty and known bias: `<description>`

### Evaluation coverage

| Evaluation | Status | Domain and runtime | Result location |
|---|---|---|---|
| Source model versus exported ONNX | `<passed/failed/not evaluated>` | `<cases/configuration>` | `<evidence>` |
| Independent-sample or single-step accuracy | `<status>` | `<domain>` | `<evidence>` |
| Physical constraints and conservation | `<status>` | `<domain>` | `<evidence>` |
| Boundary and stress tests | `<status>` | `<domain>` | `<evidence>` |
| Out-of-domain tests | `<status>` | `<defined shift>` | `<evidence>` |
| Long-horizon or coupled stability | `<status/not applicable>` | `<horizon/mode>` | `<evidence>` |
| Precision and runtime sensitivity | `<status>` | `<configurations>` | `<evidence>` |
| Uncertainty calibration | `<status/not applicable>` | `<domain>` | `<evidence>` |

Validation authority and decision date: `<organization/person and date, or no
validation decision>`

## Verification

| Case ID | Purpose | Process graph | Runtime | Comparison | Result |
|---|---|---|---|---|---|
| `<case>` | `<mapping/axis/transform/...>` | `<id>` | `<configuration>` | `<per-output tolerance>` | `<expected status or report>` |

Input and expected-output artifact locators: `<exact references>`<br>
Values are stored in: `<physical space | graph space>`<br>
Tolerance rationale: `<numerical or scientific basis>`<br>
Mapping-error perturbation detected by the case: `<axis swap/channel swap/etc.>`

Verification confirms correct package integration; it does not establish
scientific accuracy.

## Limitations and failure modes

- `<observed failure, affected domain, severity, and evidence>`
- `<drift, non-conservation, unresolved scale, bias, or sensitivity>`
- `<important unevaluated condition or unknown>`
- `<failure that cannot be detected from outputs>`

Uncertainty estimates: `<meaning and limitations | not provided>`<br>
Out-of-domain detection: `<method and action | not provided>`

## Data and model provenance

### Source simulations, experiments, observations, or analyses

| Source ID | Version and digest | Scientific configuration | Coverage/fidelity | Known bias or uncertainty | License/access |
|---|---|---|---|---|---|
| `<id>` | `<immutable identity>` | `<equations/method/boundary/forcing/...>` | `<coverage>` | `<limitations>` | `<terms>` |

For experimental or observational sources, instrument or sensor identity,
calibration, acquisition protocol, detection limits, measurement uncertainty,
corrections, and quality flags: `<details or not applicable>`

### Dataset construction and splits

Sampling, quality control, filtering, missing-data handling, and target
construction: `<procedure>`

| Partition | Immutable membership or procedure | Independent unit | Count | Leakage control |
|---|---|---|---:|---|
| Training | `<reference>` | `<unit>` | `<count>` | `<control>` |
| Validation | `<reference>` | `<unit>` | `<count>` | `<control>` |
| Test | `<reference>` | `<unit>` | `<count>` | `<control>` |
| Out-of-domain | `<reference>` | `<unit>` | `<count>` | `<defined shift>` |

### Training and export

Training code/configuration: `<immutable references>`<br>
Architecture, loss, optimizer, hyperparameter and trial selection, precision,
seed, stopping and checkpoint rule:<br>
`<details>`<br>
Source checkpoint: `<identifier and digest>`<br>
Exporter, version, opset, and options: `<details>`<br>
Graph transformations: `<details>`<br>
Source-to-ONNX agreement: `<cases, metrics, tolerance, maximum error, evidence>`

## Runtime and compatibility

ONNX IR version: `<version>`<br>
Opset imports: `<domain:version>`<br>
External ONNX data: `<artifact IDs or none>`<br>
Custom operators: `<contracts or none>`

| Runtime configuration | Runtime/provider | Hardware class | Precision | Shapes | Fallback | Determinism | Verification status |
|---|---|---|---|---|---|---|---|
| `<id>` | `<name/version>` | `<class>` | `<precision>` | `<supported>` | `<behavior>` | `<settings>` | `<case/report>` |

Resource limits: `<memory, accelerator, threads, shape-dependent limits>`<br>
Performance measurement: `<shape, batch, hardware, software, precision,
warm-up, repetitions, included stages, result>`

## Conformance profiles

| Profile | Applicability decision | Process graph or mode | Status | Evidence or reason |
|---|---|---|---|---|
| `core` | Applicable | `<scope>` | `<status>` | `<report>` |
| `<profile>` | `<applicable/not applicable and reason>` | `<scope>` | `<status>` | `<report/reason>` |

Latest conformance report: `<path or immutable URI and digest>`

## Maintenance and change history

Support contact: `<contact>`<br>
Review interval: `<interval>`<br>
Support lifetime and deprecation policy: `<policy>`

| Package version | Date | Change class | Changes | Compatibility |
|---|---|---|---|---|
| `<version>` | `<date>` | `<behavior/interface/evidence/documentation>` | `<description>` | `<compatible/incompatible>` |
