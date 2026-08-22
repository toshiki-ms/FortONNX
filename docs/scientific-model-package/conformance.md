# Conformance specification

Status: Normative, Draft 0.1

## 1. Conformance target

Conformance is assessed for one exact package version, one set of package
bytes and external artifacts, and the runtime configurations named in the
report. A report for one version or runtime must not be reused for another.

The validation input consists of:

- the package root;
- the machine-readable package-description artifact;
- an expected digest for that exact artifact from a trusted distribution record,
  signature, or validator invocation;
- its declared format and version;
- an adapter that maps it to the abstract information model;
- all required packaged and accessible external artifacts; and
- the runtime configurations in which executable checks are performed.

No adapter language, metadata serialization, report serialization, or
scientific payload format is prescribed. The adapter is trusted validator
configuration, not package data that may be executed automatically. A packaged
adapter implementation must be treated as an executable artifact and requires
an explicit trust decision before use.

## 2. Results and severity

Each check produces one of:

- `pass`: the requirement was evaluated and satisfied;
- `fail`: the requirement was evaluated and not satisfied;
- `not-applicable`: its applicability rule was evaluated and false; or
- `not-run`: the check was applicable but could not be completed.

Each finding has severity `error`, `warning`, or `information`. A required check
that returns `fail` or `not-run` is an error. A package is conforming only when
every applicable required check passes and every declared profile is
conforming. Recommended checks may produce warnings but cannot silently be
discarded.

The validator must continue safe, independent checks after an error when doing
so cannot execute untrusted data or produce misleading results. It must not run
inference when artifact integrity, graph identity, or required decoding is
unresolved.

## 3. Authority and contradiction

The following sources are authoritative within their stated scope:

| Scope | Authoritative source |
|---|---|
| ONNX nodes, signature, opsets, and external-data references | Distributed ONNX graph bytes |
| Package entity identifiers and relationships | Machine-readable package description |
| Detailed numerical evaluation results | Declared evidence artifacts |
| Human instructions, intended use, claims, and limitations | Model Card |

Authoritative sources must agree. A contradiction is an error. Authority tells
the package maintainer which source must not be changed merely to hide a
contradiction; it does not allow a consumer to guess which value to use.

## 4. Required check catalogue

### 4.1 Package and artifact checks

| ID | Required check |
|---|---|
| `PKG-001` | Verify its expected digest, then decode the package description with the declared adapter and map it losslessly to the abstract information model. |
| `PKG-002` | Validate required root fields, cardinalities, identifier uniqueness, enumerated values, and reference resolution. |
| `PKG-003` | Confirm that exactly one primary model is declared and that it is present in the model collection. |
| `PKG-004` | Normalize every package path and reject absolute paths, traversal, links escaping the package root, and conflicting path identities. |
| `PKG-005` | Confirm presence, exact byte size, and cryptographic digest of every indexed required packaged artifact. |
| `PKG-006` | Resolve every required external artifact immutably, verify its digest before use, and record access failures without substituting other bytes. |
| `PKG-007` | Confirm that every artifact declares format, version, reader, license, role, requirement level, and creation provenance. |
| `PKG-008` | Confirm that custom formats have an immutable specification and that executable decoding is explicitly declared. |
| `PKG-009` | Confirm that the Model Card, license, citation, primary graph, and verification artifacts are reachable from the package description. |
| `PKG-010` | Reject credentials, undeclared personal data, private host paths, and undeclared executable artifacts found in required package content. |
| `PKG-011` | When an authenticity record or signature is supplied, verify that it binds the package ID, version, and expected package-description digest to a declared trust identity. |

`PKG-R01` produces a warning when a published package provides no declared
authenticity mechanism or trusted distribution record.

An inaccessible optional artifact is a warning. An inaccessible required
artifact is an error even when inference could proceed using a local cache.

### 4.2 ONNX checks

| ID | Required check |
|---|---|
| `ONNX-001` | Parse every required graph and run the applicable ONNX structural checker. |
| `ONNX-002` | Compare declared and actual IR versions, opset imports, graph digests, external-data references, and custom operator imports. |
| `ONNX-003` | Compare every declared input and output name, direction, element type, rank, and fixed dimension with the graph. |
| `ONNX-004` | Resolve all ONNX external data using safe package-relative paths and verified bytes. |
| `ONNX-005` | Confirm that a tested implementation is declared for every required custom operator and execution provider. |
| `ONNX-006` | Load each required graph under every runtime configuration for which conformance is claimed, with undeclared fallback disabled. |

A structural checker result demonstrates graph well-formedness only. It is not
a verification or scientific-validation result.

### 4.3 Scientific interface checks

| ID | Required check |
|---|---|
| `SEM-001` | Confirm that every graph input and output resolves to exactly one tensor contract. |
| `SEM-002` | Confirm shape-rank-axis consistency, fixed dimension values, and compatible uses of each symbolic dimension. |
| `SEM-003` | Confirm that every axis and stacked channel has ordered scientific meaning, quantity, units, coordinate placement, and valid-value behavior. |
| `SEM-004` | Confirm explicit physical and graph-boundary units, including explicit dimensionless quantities and conversions. |
| `SEM-005` | Confirm explicit missing-value, mask, padding, NaN, infinity, clipping, and invalid-input behavior for every applicable tensor. |
| `SEM-006` | Confirm separate admissible, training, evaluation, and intended-use domains and reproducible membership rules. |
| `SEM-007` | Confirm that every spatial or temporal tensor resolves to complete coordinate, time, orientation, and location semantics. |
| `SEM-008` | Compare machine-readable interface semantics with the Model Card and all executable examples. |

Unit compatibility is checked dimensionally and conventionally. Identical unit
strings do not pass when sign, reference state, accumulation interval, or point
versus average semantics conflict.

### 4.4 Transformation and process checks

| ID | Required check |
|---|---|
| `FLOW-001` | Validate every process graph as an acyclic typed graph for one invocation and determine an unambiguous execution order. |
| `FLOW-002` | Trace every required physical input to a graph input and every graph output to a required physical output. |
| `FLOW-003` | Confirm type, shape, axis, unit, domain, and state-version compatibility on every edge. |
| `FLOW-004` | Confirm that every external operation has an exact algorithm or formula, ordered inputs and outputs, parameters, preconditions, postconditions, and failure behavior. |
| `FLOW-005` | Confirm that each required transformation is applied exactly once on an applicable path and is not duplicated inside and outside ONNX. |
| `FLOW-006` | Resolve every parameter to a literal or verified artifact locator and check dtype, shape, axes, units, broadcast, and derivation provenance. |
| `FLOW-007` | Confirm that fitted transformation parameters use only the declared training partition, or that transductive use is disclosed in domains, limitations, and evidence. |
| `FLOW-008` | Confirm that examples and coupling contracts invoke the same process graph without hidden scientific operations. |

### 4.5 Known-answer verification

| ID | Required check |
|---|---|
| `VER-001` | Confirm that at least one case uses nominal nonzero values and identifies exact model, artifact, process-graph, and runtime versions. |
| `VER-002` | Decode all case values using declared formats and locators and check their dtype, logical shape, axes, units, and physical- or graph-space designation. |
| `VER-003` | Execute the complete declared process graph, including preprocessing and postprocessing, under each runtime configuration claimed by the case. |
| `VER-004` | Compare each output using the declared comparison method and per-quantity absolute, relative, or statistical tolerances and their rationale. |
| `VER-005` | Record observed errors and pass or fail for each output; an aggregate pass must not hide a failed required output. |
| `VER-006` | Execute additional cases for dynamic shapes, masks, boundaries, alternative modes, multiple graphs, and stochastic behavior whenever applicable. |
| `VER-007` | Deliberately perturb at least one mapping-sensitive condition and confirm that the verification procedure detects it. |

The perturbation in `VER-007` should target a credible integration error such
as axis permutation, channel swap, skipped transform, wrong parameter artifact,
or wrong state-update order. It must not modify released package bytes.

### 4.6 Scientific evidence and claim checks

| ID | Required check |
|---|---|
| `SCI-001` | Resolve each claim to its model version, process graph, operating mode, intended-use domain, runtime scope, and evidence status. |
| `SCI-002` | Confirm that every numerical result declares cases or data, independent-case count, metric formula, units, weighting, masks, aggregation, reduction order, code, configuration, uncertainty-estimation method and resampling unit, and result artifact. |
| `SCI-003` | Confirm baseline identity and reference-data uncertainty or explicitly record their absence and effect as a limitation. |
| `SCI-004` | Confirm that acceptance criteria and their predeclared or retrospective status are explicit. |
| `SCI-005` | Confirm that failed, inconclusive, excluded, boundary, stress, and relevant out-of-domain results are represented without being silently dropped. |
| `SCI-006` | Confirm that each out-of-domain result identifies what lies outside training support and the reproducible rule used to determine it. |
| `SCI-007` | Confirm source-data, split, training, hyperparameter and trial selection, checkpoint-selection, evaluation, ONNX-export, and source-parity provenance with immutable versions. |
| `SCI-008` | Confirm that the independent split unit and leakage controls are stated and consistent with the claimed independence. |
| `SCI-009` | Confirm that release status and wording do not exceed the evidence status or evaluated domain. |
| `SCI-010` | For experimental or observational data, confirm instrument or sensor identity, calibration, acquisition, measurement uncertainty, corrections, and quality flags. |

Recomputing every scientific study is not a core conformance requirement. The
validator must, however, verify evidence integrity and internal consistency and
run any evaluation reproduction explicitly required by an applicable profile
or conformance claim.

### 4.7 Model Card checks

| ID | Required check |
|---|---|
| `CARD-001` | Confirm all required Model Card sections in the core specification are present and non-placeholder. |
| `CARD-002` | Confirm that the opening identifies scientific purpose, modeled system, predicted quantities, release status, and intended-use boundary. |
| `CARD-003` | Confirm that the package map covers every required artifact and agrees with the package description. |
| `CARD-004` | Confirm that the quick start includes the complete physical-input-to-physical-output path and a validity check. |
| `CARD-005` | Confirm that input/output tables cover every ONNX interface value and stacked channel. |
| `CARD-006` | Confirm supported uses, unsupported uses, limitations, observed failures, unknowns, and failure detectability. |
| `CARD-007` | Confirm validation tables link every claim to domains, metrics, results, baselines, criteria, uncertainty, and evidence. |
| `CARD-008` | Confirm provenance, runtime, license, citation, maintenance, change history, and responsible contact. |

A link to a detailed artifact is acceptable only when the Model Card summarizes
the fact needed for correct use, gives the exact locator, and identifies the
authoritative source.

## 5. Profile checks

The validator evaluates profile applicability from package content and
supported-use statements before comparing it with declared profiles.

| ID | Required check |
|---|---|
| `PROF-001` | Record a decision for every defined profile; declare `core` and at least one behavior profile applicable. |
| `PROF-002` | Detect and reject omission of an applicable profile. |
| `PROF-003` | Execute all additional requirements and cases in each applicable profile. |
| `PROF-004` | Confirm that profile scope identifies the applicable process graph, operating mode, domain, and runtime configurations. |
| `PROF-005` | Reject overall conformance when an applicable profile is failed or not tested. |

Profile-specific checks use these prefixes: `SPAT`, `ITER`, `COUP`, `MULTI`,
`CUST`, `EXT`, `STOCH`, `VALID`, `REPRO`, and `OPER`. A validator may define
more granular stable identifiers but must report their parent requirement from
the profiles specification.

## 6. Conformance report

The report must be machine-readable in a declared format and accompanied by a
short human-readable summary. It contains:

- report format and version;
- specification identifier and version;
- package identifier, version, root identity, and package-description digest;
- adapter identifier, version, and digest;
- validator name, version, configuration, and invocation time;
- declared and detected profiles with applicability reasons;
- runtime configurations, hardware, precision, fallback, and deterministic
  settings used by executable checks;
- one result per check with ID, status, severity, message, affected entity and
  artifact references, and supporting observations;
- verification-case results and per-output errors;
- warnings, checks not run, and the exact reason for each;
- overall conformance result; and
- signer or responsible organization when the report is released publicly.

The report must not embed credentials or machine-private absolute paths. A
public report may redact restricted external locations only when their
immutable identity and access conditions remain verifiable by authorized
users.

## 7. Meaning of conformance

The following results are distinct and must be reported separately:

1. `structurally conforming`: package, graph, references, and metadata are
   internally valid;
2. `verification passed`: the documented implementation reproduces the
   known-answer cases within tolerance;
3. `scientifically validated`: specified scientific claims passed predefined
   criteria in a bounded evaluated domain; and
4. `operationally accepted`: the operational profile and a named deployment
   acceptance process passed.

A package may satisfy the first two while its scientific evidence is failed or
inconclusive. It must then use `research` status and must not imply scientific
validation. No conformance result establishes safety or fitness outside the
explicit claim, domain, process graph, runtime, and operating mode.
