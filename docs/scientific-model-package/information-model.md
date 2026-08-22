# Abstract information model

Status: Normative, Draft 0.1

## 1. Purpose

Every machine-readable package description must map without loss to the
entities and constraints in this document. The mapping may use any declared
serialization. Field names in a serialization may differ from the names below,
provided that their meaning and cardinality are preserved.

The mapping definition must cover every core entity and field, preserve order,
identifier and reference identity, numeric precision, and the distinction among
absent, not applicable, not evaluated, and explicitly unknown values. It must
identify the adapter or mapping version and digest and define how decoding and
cardinality errors are reported.

## 2. Conventions

Cardinalities have these meanings:

| Cardinality | Meaning |
|---|---|
| `1` | exactly one |
| `0..1` | zero or one |
| `1..*` | one or more |
| `0..*` | zero or more |

An identifier is unique within a package and matches
`[A-Za-z][A-Za-z0-9._-]*`. A package identifier is a globally unique,
persistent string. An identifier is not a display label.

A reference resolves to exactly one entity in the same package, or to an
external entity with an immutable URI and integrity or version identity.
Dangling and ambiguous references are errors.

Values use these scalar types: Boolean, integer, finite decimal number, string,
timestamp, duration, URI, version, identifier, and digest. Tensor dimensions
are positive integers or symbolic dimensions.

Unknown extension fields must be namespace-qualified. An extension cannot
change the meaning of a core field.

### 2.1 SymbolicDimension

A `SymbolicDimension` has an identifier, scope, minimum and maximum, permitted
values or divisibility constraints when applicable, and equality relationships
to other dimensions. A runtime shape must satisfy every declared constraint.
An unbounded maximum must be explicit and must not be interpreted as evidence
that every size is operationally supported.

## 3. Package

The root `Package` entity contains:

| Field | Cardinality | Requirement |
|---|---:|---|
| `specification_id` | 1 | Identifier of this specification set |
| `specification_version` | 1 | Version against which conformance is claimed |
| `package_id` | 1 | Persistent identifier for the model across releases |
| `package_version` | 1 | Identifier for this exact release |
| `release_status` | 1 | `research`, `validated`, `operational`, or `deprecated` |
| `release_date` | 1 | Release timestamp or date with time-zone convention |
| `model_card` | 1 | Reference to the Model Card artifact |
| `license` | 1..* | License identifiers or artifact references with scope |
| `citation` | 1 | Preferred citation as data or an artifact reference |
| `primary_model` | 1 | Reference to the primary `ModelGraph` |
| `models` | 1..* | Contained `ModelGraph` entities |
| `custom_operator_contracts` | 0..* | Contained `CustomOperatorContract` entities |
| `artifacts` | 1..* | Every indexed artifact other than the package description itself |
| `tensors` | 1..* | Contained `TensorContract` entities |
| `axes` | 0..* | Contained `Axis` entities |
| `coordinate_systems` | 0..* | Contained `CoordinateSystem` entities |
| `symbolic_dimensions` | 0..* | Declared dynamic-dimension constraints |
| `quantities` | 1..* | Contained `Quantity` entities |
| `domains` | 1..* | Contained `Domain` entities |
| `transformations` | 0..* | Contained `Transformation` entities |
| `process_graphs` | 1..* | Effective inference and deployment workflows |
| `coupling_contracts` | 0..* | Iteration or coupling rules |
| `verification_cases` | 1..* | Known-answer cases |
| `claims` | 1..* | Scientific or operational claims, including proposed claims |
| `evidence` | 1..* | Evidence or an explicit not-evaluated record for each claim |
| `provenance` | 1 | `Provenance` entity |
| `runtime_configurations` | 1..* | Tested execution configurations |
| `profile_decisions` | 1..* | Applicability, status, scope, and evidence for each profile |
| `extensions` | 0..* | Namespace-qualified extension objects |

`primary_model` must occur in `models`. Package-relative paths are resolved
from the declared package root after path normalization. The package description
does not contain its own digest; its expected digest is a validation input from
outside its digest closure.

## 4. Artifact

An `Artifact` describes bytes or an externally identified resource. It does not
imply a storage format.

| Field | Cardinality | Requirement |
|---|---:|---|
| `id` | 1 | Package-unique identifier |
| `role` | 1..* | Interpretation, preprocessing, inference, postprocessing, verification, evaluation, provenance, documentation, executable, or a defined role |
| `location` | 1 | Safe package-relative path or immutable external URI |
| `requirement` | 1 | Required, conditional with condition, recommended, or optional |
| `format` | 1 | Unambiguous format name and version |
| `media_type` | 0..1 | Registered or package-defined media type |
| `logical_type` | 1 | Document, scalar, tensor, table, grid, mesh, graph, executable, archive, or declared extension |
| `locators` | 0..* | Rules for selecting values, records, members, or byte ranges |
| `numerical_description` | 0..1 | Dtype, shape, axes, units, storage order, and byte order |
| `encoding` | 0..1 | Compression, packing, chunking, missing-value, and decoding rules |
| `reader` | 1 | Published format specification or compatible reader name and version |
| `byte_size` | 1 for packaged bytes | Exact size of the artifact |
| `digests` | 1..* | Algorithm and digest of the exact bytes or defined external resource |
| `license` | 1 | License or inherited package-license reference, including scope |
| `access` | 0..1 | Authentication-free access instructions or stated restriction |
| `created_by` | 1 | Provenance activity or documented creation procedure |
| `relationships` | 0..* | Derived-from, component-of, companion-of, replaces, or equivalent-to references |
| `alternatives` | 0..* | Equivalent artifacts, authority, and equivalence tolerance |
| `executable_risk` | 1 when executable | Trust, ABI, permissions, and isolation requirements |

For a custom format, `reader` resolves to a packaged specification or an
immutable external specification. Data decoding must not require execution of
embedded code unless the artifact is explicitly declared executable.

## 5. ModelGraph

Each `ModelGraph` contains:

| Field | Cardinality | Requirement |
|---|---:|---|
| `id` | 1 | Package-unique identifier |
| `artifact` | 1 | Reference to an ONNX artifact |
| `role` | 1 | Primary, component, ensemble member, fallback, or defined role |
| `onnx_ir_version` | 1 | ONNX IR version read from the graph |
| `opset_imports` | 1..* | Domain and opset version pairs |
| `external_data` | 0..* | Required ONNX external-data artifact references |
| `custom_operators` | 0..* | `CustomOperatorContract` references |
| `inputs` | 1..* | References to input `TensorContract` entities |
| `outputs` | 1..* | References to output `TensorContract` entities |
| `source_export` | 1 | Export activity in provenance |

The declared signature must exactly match the distributed ONNX graph for names,
directions, element types, ranks, and fixed dimensions. Symbolic dimensions are
declared even when the graph uses different symbol names.

The Draft 0.1 core interface covers dense ONNX tensor values. A sparse tensor,
sequence, map, or optional graph-interface value requires a namespace-qualified
extension defining equivalent scientific contracts and conformance checks.
Omitting that extension is an error.

### 5.1 CustomOperatorContract

A `CustomOperatorContract` identifies an operator domain, name, semantic
version, exact input/output semantics, attribute meanings and defaults,
supported dtypes and shapes, numerical behavior, error behavior, ABI, and
implementation artifact references. Each implementation reference identifies
the supported runtime, provider, hardware, build procedure, license, and digest.
The contract also references an independent operator-level verification case.

## 6. TensorContract

A `TensorContract` binds an ONNX interface value to scientific meaning.

| Field | Cardinality | Requirement |
|---|---:|---|
| `id` | 1 | Package-unique identifier |
| `model` | 1 | Owning `ModelGraph` reference |
| `onnx_name` | 1 | Exact case-sensitive ONNX value name |
| `direction` | 1 | Input or output |
| `role` | 1..* | State, forcing, boundary, coordinate, parameter, mask, diagnostic, increment, tendency, prediction, uncertainty, or defined role |
| `element_type` | 1 | ONNX tensor element type |
| `rank` | 1 | Number of logical dimensions |
| `shape` | 1 | Ordered fixed or symbolic dimensions |
| `axes` | 1 | Ordered axis references, one per dimension; empty only for rank zero |
| `channels` | 1..* | Ordered `QuantityBinding` entities |
| `graph_units` | 1 | ONNX-boundary units, or `mixed` with units on every channel |
| `physical_units` | 1 | Physical units, or `mixed` with units on every channel |
| `admissible_domain` | 1 | Domain accepted by the interface and external transformations |
| `training_domain` | 1 | Training-data support or explicit not-applicable with reason |
| `evaluation_domain` | 1 | Scientific-evaluation support or explicit not-evaluated |
| `intended_use_domain` | 1 | Conditions for which use is claimed |
| `preprocessing` | 0..* | Ordered transformation references for an input |
| `postprocessing` | 0..* | Ordered transformation references for an output |
| `invalid_value_behavior` | 1 | Rejection, clipping, propagation, replacement, or undefined behavior |
| `missing_value_behavior` | 1 | Mask, sentinel, NaN, padding, and interaction rules |

`axes` and `shape` are expressed in ONNX logical order, independent of the
consumer language's memory layout. The admissible, training, evaluation, and
intended-use domains are separate references. Equality may be declared, but
must not be inferred from an omitted definition.

### 6.1 QuantityBinding

A `QuantityBinding` contains a quantity reference, an ordered channel selector,
physical and graph units when the tensor has mixed units, valid range,
coordinate or vertical location, staggering, sign and reference conventions,
applicable transformations, and missing-value behavior.

Selectors in one tensor must not overlap unless the overlap and its meaning are
explicit. Together they cover every scientifically interpreted channel. Unused
or reserved channels are identified.

## 7. Scientific coordinates and quantities

### 7.1 Quantity

A `Quantity` has an identifier, display name, definition, symbol when used,
unit convention, dimensionality, sign convention, reference state when
relevant, and distinctions such as instantaneous versus accumulated, point
value versus average, or absolute value versus anomaly.

### 7.2 Axis

An `Axis` has an identifier, meaning, index origin, fixed or symbolic length,
coordinate reference, orientation, ordering, periodicity, boundary behavior,
staggering or mesh location, and convention for missing or padded positions.

### 7.3 CoordinateSystem

A `CoordinateSystem` identifies the spatial or temporal reference system,
datum or epoch, calendar, units, topology, grid or mesh artifact, connectivity,
cell or node placement, seams and poles, vertical-coordinate coefficients, and
transformations required to interpret positions.

## 8. Domain

A `Domain` describes a scientifically meaningful set of cases, not merely a
tensor range.

| Field | Cardinality | Requirement |
|---|---:|---|
| `id` | 1 | Package-unique identifier |
| `kind` | 1..* | Admissible, training, evaluation, intended-use, out-of-domain, or stress-test |
| `definition` | 1 | Plain-language scientific definition |
| `constraints` | 1..* | Quantity ranges, categories, geometry, time, resolution, fidelity, boundary and forcing conditions |
| `selection_rule` | 1 | Reproducible membership rule |
| `data_or_case_set` | 0..* | Dataset, split, or case references |
| `sample_unit` | 0..1 | Independent unit used for counts and splitting |
| `size` | 0..1 | Number of independent cases and samples when known |
| `relationship` | 0..* | Subset, overlap, disjointness, shift, or extrapolation relative to another domain |
| `unsupported_regions` | 0..* | Explicit holes or exclusions |

An out-of-domain definition names the variables or structures outside training
support and the rule used to establish that fact.

## 9. Transformation

A `Transformation` represents one operation inside or outside an ONNX graph.

| Field | Cardinality | Requirement |
|---|---:|---|
| `id` | 1 | Package-unique identifier |
| `operation` | 1 | Defined operation name |
| `location` | 1 | Inside a named graph or external |
| `inputs` | 1..* | Typed data-flow references |
| `outputs` | 1..* | Typed data-flow references |
| `definition` | 1 | Formula or unambiguous algorithm and order |
| `parameters` | 0..* | Literal values or artifact locators |
| `preconditions` | 0..* | Units, domains, shapes, masks, or state requirements |
| `postconditions` | 0..* | Units, domains, shapes, masks, or guarantees |
| `fitted_on` | 0..1 | Training partition used to fit parameters |
| `implementation` | 0..* | Reference implementations and tested versions |
| `numerical_behavior` | 1 | Precision, rounding, determinism, and error handling |

An operation encoded in ONNX must not also occur externally on the same path.
Fitted parameters refer to the training partition unless transductive use is
explicitly declared and reflected in limitations and evidence.

## 10. Effective process graph

A `ProcessGraph` is a directed graph of typed nodes and edges specifying the
complete mapping from physical inputs to physical outputs. Nodes may represent:

- artifact decode;
- unit conversion or scientific transformation;
- feature construction;
- coordinate conversion or regridding;
- ONNX inference using a named `ModelGraph`;
- postprocessing, correction, or diagnostic calculation;
- verification comparison; or
- input, output, and external-state boundaries.

Every node has an identifier, operation or entity reference, typed input and
output ports, preconditions, postconditions, and failure behavior. Every edge
connects compatible ports and states the tensor, quantity, units, axes, shape,
and state version carried across it.

For one inference invocation, the graph is acyclic and has an unambiguous
topological order. Every required physical input and output is connected. Every
external transformation occurs exactly once on each applicable path. No
required behavior may exist only in an unexplained example helper.

Iteration applies an acyclic process graph under a `CouplingContract`; it is not
represented by hiding a cycle in one invocation.

## 11. CouplingContract

An iterative or externally coupled package has a `CouplingContract` containing:

- process graph and participating model references;
- state, forcing, boundary, increment, tendency, diagnostic, and output maps;
- initialization and restart procedures;
- step duration, timestamps, forcing time, output time, and update order;
- exchange cadence, interpolation, synchronization, and conservation rules;
- corrections or fixers and the point at which each is applied;
- random-state evolution when applicable;
- termination, invalid-state detection, failure handling, and fallback;
- supported and evaluated rollout horizon; and
- persistent state required between invocations.

## 12. VerificationCase

A `VerificationCase` contains a case identifier, purpose, process graph, model
and artifact digests, runtime configuration, physical- or graph-space inputs,
expected outputs, exact mappings and locators, comparison methods, per-quantity
tolerances, and expected nondeterministic variation.

The result records observed values or digests, comparison errors, pass or fail
for every output, runtime identity, and execution time. Passing under one
runtime configuration does not imply support for another.

## 13. Claim and Evidence

A `Claim` contains an identifier, exact statement, scope, intended-use domain,
applicable profile, model version, decision relevance, and status. Permitted
statuses are `proposed`, `supported`, `not-supported`, `rejected`, and
`withdrawn`.

An `Evidence` entity contains:

- claim references and exact package and model versions;
- evaluation domain and case or dataset membership;
- number and definition of independent cases;
- metric definitions, units, weighting, masks, aggregation, and reduction;
- results with uncertainty or variability, estimation method, interval level,
  and independent or resampling unit;
- baseline identities and results;
- acceptance criteria and whether each was declared before evaluation;
- reference-data uncertainty and known bias;
- code, configuration, runtime, and result artifact references;
- observed failures and exclusions; and
- conclusion: passed, failed, inconclusive, or not evaluated.

Absence of evaluation is represented by an evidence entity with conclusion
`not evaluated`; failed or unknown results are not omitted. A conformance result
must not be used as evidence of scientific accuracy.

## 14. Provenance

`Provenance` records entities, activities, and responsible agents for:

- source simulations, experiments, observations, or analyses;
- instrument or sensor identity, calibration, acquisition, measurement
  uncertainty, corrections, and quality flags for experimental or observational
  sources;
- dataset assembly, filtering, quality control, and target construction;
- train, validation, test, and out-of-domain split construction;
- transformation-parameter estimation;
- training, checkpoint selection, and evaluation;
- ONNX export, graph transformation, and source-model parity checking; and
- package assembly and release approval.

Each activity records immutable input and output references, code and
configuration versions, time, responsible agent, numerical environment,
randomness controls, and failed or excluded inputs. Dataset splits record their
independent unit and leakage controls.

## 15. RuntimeConfiguration

A `RuntimeConfiguration` identifies runtime implementation and version,
execution provider, hardware class, precision, threading, fallback behavior,
custom-operator implementations, external-data handling, deterministic
settings, resource limits, and supported input shapes. It states which
verification and evaluation cases ran under that configuration.

## 16. Authority and consistency

The distributed ONNX bytes are authoritative for graph structure and signature.
The machine-readable package description is authoritative for entity
relationships and identifiers. Declared evidence artifacts are authoritative
for detailed numerical results. The Model Card is authoritative for human
instructions, claims, limitations, and intended use.

The sources must agree. A contradiction is a conformance error; authority
determines what must be corrected, not which value a consumer silently chooses.

## 17. Release identity

Any change to graph bytes, external ONNX data, required transformation
parameters, tensor semantics, effective process graph, or scientific meaning
requires a new package version. Changes to fixed tensor names, element types,
ranks, dimensions, units, or process ordering are interface changes and are
marked incompatible.

A correction that does not alter executable or scientific behavior still
requires a new package version when it changes packaged bytes or digests. The
change log classifies the release as behavior-changing, interface-incompatible,
evidence-only, or documentation-only.
