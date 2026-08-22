# Scientific ONNX model package specification

Status: Normative, Draft 0.1

## 1. Scope

A scientific ONNX model package consists of one or more executable ONNX graphs,
a Model Card, a machine-readable package description, and the supporting
artifacts required to interpret and reproduce its scientific behavior. Exactly
one graph must be identified as primary.

The Model Card must state:

- the represented scientific system and quantities;
- every artifact required for inference and interpretation;
- input and output meanings, units, coordinates, shapes, and transformations;
- evaluated operating conditions, results, acceptance criteria, and validation
  status;
- unsupported conditions and known failure modes;
- training and evaluation data provenance;
- verification procedures and expected results.

Conformance does not require a particular programming language, inference
library, training framework, hardware vendor, validator, or application
framework.

This document is used together with the [abstract information
model](information-model.md), [profiles](profiles.md), and [conformance
specification](conformance.md). A package must satisfy all four normative
documents. The [Model Card template](model-card-template.md) and [worked
example](reference-example.md) are informative.

## 2. Payload-format independence

No serialization format is prescribed for scientific payload data. The
executable graph is ONNX. Serialization of normalization arrays, grids, meshes,
forcing, initial conditions, verification cases, and detailed evaluation
results is package-defined. Package conformance is independent of those format
choices.

For every payload artifact, the package must state the applicable items below:

- what the artifact contains;
- where it is located;
- its format and format version;
- how a value or variable is located inside it;
- the dtype, shape, axis order, storage order, and byte order when relevant;
- the software or specification needed to read it;
- its integrity digest.

A filename extension is not a sufficient format declaration. ONNX portability
does not automatically make surrounding data portable. A declared language or
environment compatibility requires documented readers for every required
artifact or an equivalent readable artifact.

The words **required**, **conditionally required**, **recommended**, and
**optional** have the following meanings:

- **required**: every conforming package includes the item;
- **conditionally required**: the item is required when the described feature
  is used;
- **recommended**: omission is allowed, but the Model Card should explain why.
- **optional**: omission does not affect conformance and needs no explanation.

## 3. Package layout

Angle-bracketed names are package-defined and do not imply a data format.

```text
<model-package>/
├── README.md                         # required: Model Card
├── LICENSE                           # required: model and package terms
├── models/
│   ├── <primary-model>.onnx          # required: primary ONNX graph
│   └── <additional-model-files>      # conditional: components/external data
├── <package-description>             # required: machine-readable package index
├── citation.<ext>                    # recommended: citation information
├── metadata/
│   └── <structured-metadata>         # recommended: machine-readable metadata
├── parameters/
│   └── <transformation-artifacts>    # conditional: normalization, constants
├── coordinates/
│   └── <coordinate-artifacts>        # conditional: grids, meshes, masks
├── verification/
│   ├── <case-description>
│   └── <known-answer-artifacts>      # required: at least one verification case
├── evaluation/
│   └── <detailed-result-artifacts>   # recommended: data behind reported results
├── examples/
│   └── <usage-examples>              # recommended: minimal complete use
└── docs/
    └── <additional-documentation>    # optional: derivations and reports
```

The directory names describe roles, not serialization formats. An alternative
physical layout is allowed when `README.md` gives the exact path and role of
every required artifact. The package description may have any path and
serialization. The Model Card must state its path or URI, format and version,
and adapter or published mapping to the abstract information model. Its expected
digest is supplied as defined by the specification-set index.

Training datasets and source-simulator output do not need to be copied into the
package. When they are external, the Model Card must identify their immutable
version, location or persistent identifier, access conditions, and digest or
version manifest.

## 4. Model Card

`README.md` is required. It must contain the operational instructions,
scientific interpretation, validation evidence, and limitations needed to use
the packaged model correctly.

Instructions required to prepare inputs, run inference, and interpret outputs,
together with scientific claims and limitations, must appear directly in the
Model Card. Detailed provenance, configurations, and result tables may reside in
declared artifacts when the Model Card summarizes them, gives their exact
locations, and identifies the authoritative source. Domain-specific terms must
be defined or linked to an appropriate reference.

The following sections are required. Additional sections are allowed.

### 4.1 Model overview

State, in plain language:

- model name and release version;
- scientific domain and modeled system;
- whether it is a surrogate, emulator, parameterization, forecast model, or
  another kind of learned model;
- what the model predicts and over what spatial or temporal scale;
- release status, such as research, validated for a stated scope, or deprecated;
- responsible organization, contact, license, and preferred citation.

The opening paragraph must state scientific applicability before implementation
details. A `validated` status must name the validated scope, acceptance
criteria, validation authority, and decision date.

### 4.2 Intended use

Describe the tasks and workflows for which the model is intended. State:

- required domain expertise;
- whether use is research, experimental, or operational;
- supported spatial domains, resolutions, time steps, forecast horizons, and
  parameter ranges;
- assumptions about initialization, boundary conditions, forcing, and coupling;
- whether the model is used once, autoregressively, or inside another simulator.

### 4.3 Uses that are not supported

List conditions under which the model should not be used. Include unsupported
resolutions, regimes, parameter ranges, extrapolations, interventions, horizons,
or safety-critical decisions. Do not describe an untested use as supported.

State downstream decision risks, required human oversight, legal or data-use
restrictions, and other consequences that limit responsible use.

### 4.4 Package map

Provide a table with one row for every artifact required for inference or
interpretation:

| Item | Path or URI | Role | Required | Format and version | How to read it |
|---|---|---|---:|---|---|
| Primary graph | package-defined | ONNX inference graph | yes | ONNX version | declared ONNX runtime |
| Other artifact | package-defined | package-defined | yes/no | package-declared | package-declared |

The table must include ONNX external-data files, custom-operator libraries,
normalization parameters, grids or meshes, channel definitions, forcing,
initial conditions, correction procedures, and verification data when they are
used.

For each non-ONNX artifact, identify its format and format version explicitly.
If special software is required, give the library name, compatible versions,
and installation or access instructions. If alternative serializations are
provided, state which one is authoritative and how equivalence was checked.

### 4.5 Quick start

Show the shortest complete path from package files to a scientifically
interpretable result. The example should make clear:

1. which ONNX graph is loaded;
2. which supporting artifacts are read;
3. how physical inputs are transformed into graph inputs;
4. how exact ONNX tensor names are bound;
5. how inference is executed;
6. how graph outputs are transformed back to physical quantities;
7. how the result is checked for validity.

At least one complete example is required. Each additional claimed execution
environment must either have an example or document all differences in runtime,
artifact access, data layout, and synchronization. Examples must not hide a
required transformation or correction behind an unexplained helper function.

### 4.6 Inputs and outputs

Provide a graph-interface table with one row for every ONNX input and output:

| ONNX name | Direction | Role | Graph dtype | ONNX shape | Axes | Transform |
|---|---|---|---|---|---|---|

Provide scientific-semantics rows for each tensor or stacked channel:

| Tensor or channel | Scientific quantity | Physical units | Graph units | Coordinates and conventions |
|---|---|---|---|---|

Keep interface acceptance separate from empirical data coverage:

| Tensor or channel | Admissible domain | Training support | Evaluated support |
|---|---|---|---|

For every tensor, explain:

- the exact, case-sensitive ONNX tensor name;
- whether it is a state, forcing, boundary value, coordinate, parameter, mask,
  diagnostic, increment, tendency, or prediction;
- physical meaning, symbol, units, and sign convention;
- dtype and shape in ONNX logical axis order;
- the meaning of every fixed and dynamic dimension;
- coordinate system, grid or mesh location, staggering, and orientation;
- admissible values and the behavior for rejected or out-of-range values;
- ranges represented in training data and in scientific evaluation;
- missing-value, mask, padding, NaN, and infinity behavior;
- preprocessing and postprocessing, in exact execution order.

Tensor metadata is always expressed in ONNX logical order. A language binding
may use a different memory layout. Examples for a supported language must show
the mapping explicitly; the scientific metadata itself must not change with the
consumer language.

Physical and graph-boundary units must use an identified convention.
Dimensionless values must be marked explicitly. Angle conventions, logarithmic
quantities, accumulated versus instantaneous values, averages versus point
samples, and flux directions must not be left implicit.

The admissible input domain, training-data domain, scientific-evaluation domain,
and intended-use domain are distinct. The Model Card must identify each one and
must not use evidence from one domain as evidence for another.

For a tensor that stacks several physical variables, give an ordered channel
table. Each channel needs its own name, units, valid range, vertical location,
and transformation. Writing only "state vector" or "normalized features" is
not sufficient.

### 4.7 Required transformations and supporting data

Describe every operation outside the ONNX graph that is required to obtain the
published behavior. This includes, when applicable:

- unit conversion;
- normalization or nondimensionalization;
- feature construction;
- clipping and missing-value handling;
- regridding, interpolation, padding, or masking;
- output reconstruction;
- filtering, fixers, conservation corrections, or bias correction.

For each operation, give the mathematical definition, order of application,
and the artifact containing any parameters. The artifact description must give
its declared format, internal locator, dtype, shape, axes, broadcast rules,
derivation dataset and partition, parameter-estimation method, and digest.

Parameters fitted from data must use the training partition only. Any
transductive use of validation, test, or deployment data must be identified in
the Model Card and treated as a limitation of the reported evaluation.

If an operation is encoded inside the ONNX graph, its location must be stated so
that the operation is not applied again outside the graph.

The complete physical-input-to-physical-output behavior must be represented as
an effective process graph conforming to the abstract information model. Its
nodes must include every required artifact decode, transformation, coordinate
operation, ONNX invocation, correction, and state update in unambiguous order.
For one inference invocation the process graph must be acyclic. Iteration and
external coupling must be described by a coupling contract rather than hidden
inside example code.

### 4.8 Coordinates, time, and coupling

When the model uses spatial or temporal structure, describe:

- coordinate reference system and axis orientation;
- grid, mesh, topology, indexing base, periodicity, seams, poles, staggering,
  and whether values reside at points, cells, faces, or edges;
- vertical-coordinate definition and required coefficients;
- interpolation or regridding algorithm, order, boundary handling, and stated
  conservation properties;
- timestamp convention, time reference, calendar, time step, averaging or
  accumulation interval, and leap-time handling;
- forcing time, output time, and state-update order;
- initial and boundary conditions;
- restart behavior and any difference between the first and later steps.

For an iterative or coupled model, provide a short step-by-step coupling
contract identifying which outputs replace state, which are increments or
tendencies, and when corrections are applied.

### 4.9 Scientific validation

Summarize evidence for each claim that matters to the intended use. Use a table
such as:

| Claim | Evaluation domain | Metric and units | Result | Baseline | Acceptance criterion | Evidence |
|---|---|---|---:|---:|---:|---|

Each result must identify the exact model version, evaluation dataset or cases,
number of independent cases, metric definition, units, weighting, mask,
aggregation and reduction order, result-generating code version, uncertainty or
variability, uncertainty-estimation method, interval level when applicable,
independent or resampling unit, and detailed evidence. A criterion not defined
before evaluation must be labeled as post hoc; absence of an acceptance
criterion must be stated. Detailed result artifacts may use any declared format
and may be stored under `evaluation/` or referenced externally.

Depending on the scientific use, evaluation should include:

- source-framework versus ONNX conversion agreement;
- single-step or independent-sample accuracy;
- comparison with numerical, statistical, persistence, or previous-model
  baselines;
- conservation, balance, invariance, symmetry, or constraint violations;
- spectra, extremes, integral quantities, or other domain-relevant diagnostics;
- interpolation, boundary-of-domain stress tests, and explicitly defined
  out-of-domain tests;
- long-horizon, autoregressive, free-running, or coupled stability;
- sensitivity to precision, runtime implementation, execution backend, and
  initial conditions;
- predictive uncertainty calibration and coverage when uncertainty is supplied;
- uncertainty and known bias in the simulation, observation, or analysis used as
  the reference.

Passing an ONNX checker is not scientific validation. A held-out sample from the
same trajectory is not, by itself, evidence of out-of-domain generalization.
An out-of-domain result must identify which geometry, state, forcing, parameter,
resolution, time period, or other condition lies outside training support and
how that condition was selected or quantified.

### 4.10 Limitations and failure modes

Describe observed failures and important unknowns. Include accumulated error,
drift, non-conservation, unstable regimes, unresolved scales, bias, sensitivity
to inputs, and domains that were not tested. Negative results must not be
omitted because they make the model look less accurate.

If the model does not provide uncertainty estimates, say so. If failure cannot
be detected from outputs, state that clearly.

### 4.11 Data and model provenance

Identify the source simulations, experiments, observations, or analyses used to
train and evaluate the model. Include:

- source-system name, version, code revision, configuration, governing-equation
  family, closures, numerical method, resolution or fidelity level, convergence
  criteria, parameterizations, boundary conditions, forcing, excluded or failed
  source runs, and known deficiencies or uncertainty;
- for experimental or observational sources, instrument or sensor identity,
  calibration version, acquisition protocol, detection limits, resolution,
  sampling design, measurement uncertainty, corrections, and quality flags;
- dataset identity, version, license, access restrictions, spatial and temporal
  coverage, parameter ranges, number of independent cases, and sample counts;
- sampling, filtering, quality control, missing-data handling, target construction,
  and transformations applied before training;
- exact train, validation, test, and out-of-domain split membership or a fully
  reproducible procedure for deriving it;
- the independent unit used for splitting and how leakage was prevented;
- training code and framework versions, architecture, loss definitions and
  weights, hyperparameter search and trial-selection procedure, precision,
  random seeds, optimizer, stopping rule, and checkpoint-selection rule;
- source-checkpoint digest, ONNX exporter and version, opset, export options,
  graph transformations, and final model digest;
- numerical comparison between the source model and exported ONNX model,
  including cases, metric definitions, tolerances, and maximum observed error.

Large or restricted datasets may remain external, but the package must identify
the exact data used. A mutable branch name or an unversioned web page is not an
adequate scientific identifier.

### 4.12 Runtime and compatibility

State the ONNX IR version, opset imports, supported dtypes and shapes, and tested
runtime implementations. For every tested configuration, report software
version, execution backend, hardware class, precision, and whether fallback was
used.

List custom operators and external model data explicitly. A custom operator
must include its ABI requirements, supported backends, license, and digest.
Hardware-specific engines or caches may be provided as optional derived
artifacts, but they are not substitutes for the ONNX graph.

State whether results are deterministic and which settings affect numerical
reproducibility. Performance claims must state the input shape, batch size,
hardware, software, precision, warm-up, repetitions, and whether data transfer
and transformations are included.

Minimum memory, accelerator requirements, thread settings, and shape-dependent
resource limits must be stated when they constrain use. The documented tensor
names, dtypes, ranks, and static dimensions must agree with the distributed ONNX
graph.

### 4.13 Maintenance and change history

Give a stable package identifier, release date, package version, support contact,
deprecation policy, and change log. The identifier remains unchanged across
releases of the same model. Changes to graph behavior, weights, tensor
interfaces, transformation parameters, scientific meaning, or validation
evidence require a new package version. Interface-incompatible changes must be
identified explicitly.

## 5. Artifact descriptions

Artifact descriptions must appear in the machine-readable package description
and be summarized where required in the Model Card. Additional structured
metadata files may be used. When values are duplicated, the Model Card must
identify the authoritative location.

### 5.1 Package index

A machine-readable package index is required. It must identify the package,
package version, Model Card, primary ONNX graph, every required artifact other
than itself, the digest of every indexed required artifact, and dependency
relationships among artifacts. It must map without loss to the abstract
information model. Its serialization is package-defined. Its own expected digest
is supplied by a trusted distribution record, signature, or validator invocation
outside its digest closure.

### 5.2 Artifact entries

Every required artifact description must contain the following information when
applicable:

| Field | Meaning |
|---|---|
| Identifier | Stable name used within the package |
| Role | Why the artifact is needed |
| Path or URI | Exact package-relative path or persistent external location |
| Required stage | Interpretation, preprocessing, inference, postprocessing, verification, or evaluation |
| Format | Declared format name and version |
| Media type | Registered or package-defined media type when available |
| Locator | Variable, record, dataset, member, offset, or other selection rule |
| Logical type | Scalar, tensor, table, grid, mesh, graph, document, or executable |
| Numerical description | Dtype, shape, axes, units, storage order, and byte order |
| Encoding | Compression, chunking, packing, missing values, and decoding rules |
| Reader | Format specification and compatible reader implementations |
| Integrity | Digest algorithm and digest of the exact artifact |
| Provenance | How and from which data or process it was created |
| Alternatives | Equivalent representations and their equivalence tolerance |
| Requirement | Required, conditional, recommended, or optional |
| Size | Exact byte size when the artifact is packaged |
| Access and license | Access conditions and applicable license |
| Relationships | Parent artifact, derived-from relation, and required companions |

A custom format is allowed only when its specification is packaged or linked by
an immutable identifier. Every claimed environment must have a documented
reader for each required custom-format artifact.

## 6. Verification case

At least one known-answer case is required to detect an incorrect file set,
tensor-name mapping, axis order, transformation, parameter selection, runtime
configuration, or output interpretation.

A verification case must specify:

- case identifier and purpose;
- exact ONNX model version and digest;
- exact mapping from ONNX tensor names to input and expected-output artifacts;
- declared artifact format and locator for every value;
- whether stored values are in physical space or graph space;
- preprocessing and postprocessing to apply;
- dtype, ONNX logical shape, axes, and units;
- runtime and numerical precision used to generate the expected result;
- comparison method, absolute, relative, and per-variable tolerances, and the
  numerical or scientific rationale for those tolerances;
- expected nondeterministic variation, if any.

At least one case must exercise nominal nonzero values. Boundary conditions,
masks, dynamic shapes, multiple operating modes, and multiple graphs require
additional cases when they affect package use.

No verification-data format is prescribed. The Quick start must include a
complete verification procedure for each claimed environment. Verification
demonstrates correct package use; it does not replace scientific validation.

## 7. Multiple graphs and alternative artifacts

A package may contain several ONNX graphs, such as ensemble members, regional
components, or coupled submodels. The Model Card must identify the primary
graph and explain the role, invocation order, data flow, shared state, and
failure behavior of every required graph.

A package may also contain alternative serializations of supporting data. Each
alternative must have a separate artifact description. The Model Card must state:

- which representation is authoritative;
- which environments each representation supports;
- how representations were converted;
- how numerical equivalence was tested;
- the tolerance within which they are considered equivalent.

## 8. Integrity and safe distribution

Every required artifact must have a cryptographic digest. Digests may be stored
in any declared manifest or metadata representation; no checksum-file format is
prescribed.
Digest algorithms must be identified and suitable for collision-resistant
artifact identity at the release date. A deprecated or non-collision-resistant
digest may be included for compatibility but cannot be the only integrity
identifier.

A published release should provide an authenticated release record or signature
that binds the package identifier, package version, and package-description
digest. If no authenticity mechanism is supplied, the Model Card must state
that digests detect mismatch only relative to a separately trusted value.

The package must not contain credentials, private host paths, undeclared
personal data, or dependencies that cannot legally be distributed.
Data artifacts must not require execution of embedded code merely to decode
scientific values. Executable examples and custom operators must be identified
as executable and included in the package's trust and license discussion.

An archive must not escape its package root when extracted. External ONNX data
must use safe relative paths. Archives and directory trees must not contain
links or path references that resolve outside the package root. Remote artifacts
required for inference must use immutable versions and digests.

## 9. Applicable profiles

Every package must record an applicability decision for every defined profile.
The core profile and at least one behavior profile are always applicable. Every
additional profile whose applicability rule is true must be declared applicable
and satisfied. Applicability follows package behavior and supported-use
statements; it is not selected to avoid requirements. The rules are defined in
[Conformance profiles](profiles.md).

## 10. Conformance summary

A conforming package satisfies all applicable requirements below:

- The Model Card presents the scientific purpose before implementation details.
- Every artifact required for inference and interpretation has an exact path or
  persistent URI.
- Every supporting artifact declares its format, version, locator, and reader
  without assuming a programming language.
- Every ONNX input and output has an exact name, scientific meaning, physical
  and graph units, dtype, shape, axes, transformations, and distinct admissible,
  training, evaluated, and intended-use domains.
- All transformations, grids, parameters, forcing, initial conditions, and
  corrections are reproducible from the declared artifacts and instructions.
- The effective process graph covers the entire physical-input-to-physical-output
  path with no hidden, duplicated, or unordered operation.
- The Quick start covers the complete path from physical input to physical
  output.
- At least one known-answer verification case is executable as documented.
- Source data, independent data partitions, training, checkpoint selection,
  ONNX export, and conversion agreement are traceable to immutable versions.
- Scientific claims identify metrics, domains, baselines, acceptance criteria,
  uncertainty, and detailed evidence.
- Iterative or coupled behavior is evaluated in the documented operating mode
  and for the claimed horizon.
- Unsupported uses, failed tests, unknowns, and failure modes are stated.
- Runtime requirements, external data, custom operators, license, citation,
  maintenance contact, and version history are present.
- Every applicable profile is declared and passes its additional requirements.

A package containing only an ONNX graph and an accuracy claim is not conforming.
The complete required check catalogue and report rules are defined in the
[Conformance specification](conformance.md).
