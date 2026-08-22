# Conformance profiles

Status: Normative, Draft 0.1

## 1. Profile selection

Every package records an applicability decision for every profile in this
document. `core` is always applicable, and every profile whose applicability
rule is true must be declared applicable and satisfied. A package must not mark
an applicable profile as inapplicable to avoid its requirements. More than one
profile may apply.

At least one behavior profile is required:

- `single-pass` for independent invocations whose outputs are not reused as
  model state;
- `iterative` when an output is used in a later invocation of the package; or
- `coupled` when the package exchanges state, forcing, tendencies, fluxes, or
  boundary values with another model or system.

`coupled` may be combined with `iterative`. A package offering more than one
supported operating mode declares the profiles for every offered mode and
identifies the applicable process graph and evidence for each mode.

## 2. Core profile

Applicability: every package.

In addition to the core specification, the package must:

- map without loss to the abstract information model;
- declare a release status and every applicable profile;
- provide an effective process graph from physical inputs to physical outputs;
- provide at least one executable known-answer verification case;
- identify scientific claims, including claims that are proposed, rejected, or
  not supported;
- state the evidence status for each claim and explicitly mark unevaluated
  intended-use conditions;
- distinguish interface admissibility, training support, evaluation support,
  and intended-use support; and
- state limitations and failure-detection behavior.

A core-conforming research package may contain inconclusive or failed scientific
evidence. It remains conforming only if its claims and release status do not
overstate that evidence.

## 3. Behavior profiles

### 3.1 Single-pass

Applicability: the supported use consists of independent process-graph
invocations.

The package must define input independence assumptions, batch semantics,
output reference time or condition, and whether repeated invocations with the
same inputs are deterministic. Evaluation must cover the claimed input domain
without relying on feedback from previous predictions.

### 3.2 Iterative

Applicability: any output, increment, tendency, or derived result becomes input
state for a later invocation.

The package must provide a `CouplingContract` that defines initialization,
state update, forcing cadence, time conventions, corrections, persistent state,
restart, invalid-state detection, and termination. It must also provide:

- a minimal executable rollout example;
- verification cases for the first step and at least one later step;
- evaluation at every claimed rollout horizon, not only one-step accuracy;
- drift, stability, accumulated error, and relevant conservation or balance
  diagnostics;
- sensitivity to initial conditions and numerical precision when material; and
- the longest evaluated horizon and the conditions under which failure was
  observed.

### 3.3 Coupled

Applicability: the package exchanges data with an external model, solver,
instrument, controller, or assimilation system as part of a supported use.

The package must provide a `CouplingContract` that defines both sides of every
exchange, units, coordinates, signs, timestamps, cadence, interpolation,
synchronization, state ownership, and failure handling. It must also provide:

- a reference coupling sequence or driver that exposes every exchange;
- cold-start and restart verification cases;
- conservation or budget closure across the coupling boundary when relevant;
- evaluation in the coupled configuration for every coupled scientific claim;
- behavior for missing, late, stale, rejected, or out-of-domain exchanged data;
  and
- a statement of which external-system versions were tested.

Uncoupled evaluation cannot by itself support a coupled-performance claim.

## 4. Structural profiles

### 4.1 Spatial

Applicability: any tensor represents values on physical points, cells, faces,
edges, grid positions, mesh entities, or spatial modes.

The package must declare the coordinate reference system, topology, axis
orientation, indexing, periodicity, seams, poles, staggering or entity
location, and coordinate artifacts. Regridding or interpolation must specify
the source and destination spaces, algorithm, boundary behavior, missing-data
behavior, and conservation properties. Verification must detect at least one
axis permutation or orientation error and, when applicable, a mask or boundary
error.

### 4.2 Multi-graph

Applicability: more than one ONNX graph is required for a supported process
graph.

Every graph must have its own signature and role. The effective process graph
must define invocation order, data flow, shared state, conditional branches,
and partial-failure behavior. Verification must execute every required graph
and every branch required for the claimed use. An ensemble must define member
identity, aggregation, weighting, and missing-member behavior.

### 4.3 Custom-operator

Applicability: a required ONNX graph imports an operator outside the standard
ONNX operator domains supported without package-specific code.

The package must identify each operator domain, name, version, semantics, ABI,
implementation artifact, build or installation procedure, supported runtime
and hardware configurations, license, and digest. A reference calculation must
test each custom operator independently, and a package verification case must
test it in the full graph. The Model Card must identify the additional trust
and portability implications.

### 4.4 External-artifact

Applicability: an artifact required for inference or interpretation is not
contained in the package bytes.

Every external artifact must use an immutable identifier, declared access
conditions, exact version and digest, license, expected size, and failure
behavior when unavailable. A mutable URL, branch, tag that can be replaced, or
latest-version selector is not sufficient. Conformance testing must verify the
retrieved bytes before use.

## 5. Statistical profiles

### 5.1 Stochastic

Applicability: predictions depend on random sampling, randomized operators,
latent draws, dropout, nondeterministic generation, or an ensemble interpreted
as a predictive distribution.

The package must declare:

- the source and scope of randomness;
- seed format, state initialization, state advancement, and reproducibility
  limits;
- whether samples are independent, exchangeable, correlated, or conditioned;
- the number of samples required for each claimed statistic;
- the meaning of distributional outputs and their parameterization; and
- aggregation and uncertainty-interval procedures.

Verification must compare invariant statistics or declared deterministic draws
with statistically justified tolerances. Scientific evaluation must include
calibration, coverage, sharpness or dispersion, and failure of uncertainty
estimates where relevant. A deterministic tolerance must not be imposed on a
stochastic output without a fixed and portable randomness contract.

## 6. Evidence and lifecycle profiles

### 6.1 Scientifically validated (`scientifically-validated`)

Applicability: the release uses `validated` or `operational`, or the Model Card
states that the model is validated, qualified, reliable, accurate enough, or
fit for a scientific purpose.

For every validated claim, the package must provide:

- an exact and falsifiable claim with a bounded intended-use domain;
- an acceptance criterion established before evaluation, or a conspicuous
  declaration that retrospective criteria were used;
- evaluation on data or cases independent of training and checkpoint
  selection;
- appropriate baselines and domain-relevant physical diagnostics;
- metric definitions, uncertainty, independent-case counts, and detailed
  reproducible results;
- reference-data or source-simulation uncertainty and known bias;
- negative, boundary, stress, and out-of-domain results relevant to the claim;
- the authority that accepted the evidence and the decision date; and
- a conclusion of passed, failed, or inconclusive for each criterion.

Only claims with passed criteria may be described as validated. Validation is
limited to the intersection of the evaluated domain, runtime configurations,
process graph, and operating mode in the evidence.

### 6.2 Training reproducible (`training-reproducible`)

Applicability: the package claims that training or model construction is
reproducible.

The package must provide immutable identities for all accessible source data,
split membership or a deterministic split procedure, preprocessing,
architecture, initialization, loss, optimizer, schedule, precision, seeds,
software environment, training configuration, stopping rule, checkpoint
selection, export procedure, and source checkpoint. Restricted inputs must be
identified with a reproducible access procedure.

The package must state whether reproduction means bitwise identity, graph and
weight identity, or agreement within declared statistical and numerical
tolerances. A rebuild report must compare a clean rebuild with the released
model under that definition.

### 6.3 Operational

Applicability: the release is deployed in a workflow where model output affects
an ongoing scientific, engineering, safety, regulatory, or resource decision,
or uses the release status `operational`.

The package must also satisfy the `scientifically-validated` profile. It must
define:

- deployment acceptance and release authority;
- monitored inputs, outputs, physical constraints, drift indicators, and alert
  thresholds;
- out-of-domain detection and the action taken on detection;
- fallback or safe-state behavior and the conditions that trigger it;
- runtime health checks and a scheduled verification procedure;
- incident reporting, rollback, security update, and model withdrawal
  procedures;
- version compatibility and migration rules for upstream and downstream
  systems; and
- support lifetime, review interval, and deprecation dates.

Operational conformance applies only to the tested deployment configuration. It
does not transfer automatically to a different runtime, hardware class,
precision, coupling partner, or monitored domain.

## 7. Declaring profiles

The Model Card and machine-readable package description must list:

| Profile | Applicability decision | Process graph or mode | Status | Evidence or reason |
|---|---|---|---|---|

`Status` is `conforming`, `not-conforming`, or `not-tested`. A package cannot
claim overall conformance while an applicable profile is `not-conforming` or
`not-tested`. A profile that is not applicable must include a short, testable
reason rather than merely being omitted.
