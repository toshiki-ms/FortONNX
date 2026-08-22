# ThermalColumn-Step: format-neutral worked example

Status: Informative

ThermalColumn-Step is an invented model used only to demonstrate the required
scientific contract. It advances a one-dimensional temperature column by one
60-second step under a prescribed surface heat flux. The values below are
illustrative; no executable model or scientific claim is distributed here.

Non-ONNX artifacts are identified by abstract format tokens `F-META`,
`F-ARRAY`, and `F-TABLE`. A real package replaces each token with its chosen
format name, version, reader, locator rules, and byte-level digest. The tokens
do not prescribe a serialization.

## 1. Package identity

| Field | Value |
|---|---|
| Specification ID | `scientific-onnx-model-package` |
| Specification version | `0.1-draft` |
| Package ID | `org.example.thermal-column-step` |
| Package version | `0.3.0` |
| Release status | `research` |
| Primary model | `thermal-step-graph` |
| Profiles | `core`, `iterative`, `spatial` |
| Intended use | Controlled numerical experiments over the evaluated column configurations |
| Unsupported use | Safety decisions, phase-changing regimes, or steps beyond the evaluated horizon |

## 2. Logical package contents

| Artifact ID | Logical path | Role | Format | Internal locator | Reader | Integrity |
|---|---|---|---|---|---|---|
| `card` | `README.md` | Model Card | Markdown, declared version | Whole document | Text reader | `<digest>` |
| `index` | `package-description` | Package description | `F-META` | Root package entity | `<declared adapter>` | `<supplied externally>` |
| `graph` | `models/thermal-step.onnx` | Primary graph | ONNX, declared IR version | Whole graph | `<tested runtime>` | `<digest>` |
| `norm` | `parameters/normalization-data` | Transform parameters | `F-ARRAY` | `temperature.mean`, `temperature.scale`, `flux.scale` | `<declared reader>` | `<digest>` |
| `vertical` | `coordinates/vertical-coordinate` | Cell-center and interface heights | `F-ARRAY` | `z_center`, `z_interface` | `<declared reader>` | `<digest>` |
| `case-input` | `verification/nominal-input` | Known-answer input | `F-ARRAY` | `temperature`, `surface_flux` | `<declared reader>` | `<digest>` |
| `case-output` | `verification/nominal-expected` | Known-answer output | `F-ARRAY` | `next_temperature` | `<declared reader>` | `<digest>` |
| `results` | `evaluation/rollout-results` | Detailed evaluation | `F-TABLE` | Claim and case identifiers | `<declared reader>` | `<digest>` |

For every `F-ARRAY` artifact, the real package also declares dtype, logical
shape, axis order, storage order, byte order, units, missing-value encoding,
and exact byte size. The abstract table is not sufficient as a physical
package index.

## 3. Model graph

`thermal-step-graph` refers to `models/thermal-step.onnx` and is the only ONNX
graph.

| ONNX name | Direction | Element type | Logical shape | Axes | Role |
|---|---|---|---|---|---|
| `temperature_normalized` | Input | `tensor(float)` | `[batch, level]` | `batch`, `level` | State |
| `surface_flux_normalized` | Input | `tensor(float)` | `[batch, 1]` | `batch`, `scalar` | Forcing |
| `temperature_increment_normalized` | Output | `tensor(float)` | `[batch, level]` | `batch`, `level` | State increment |

The graph declaration records its actual IR version, opset imports, signature,
export activity, and digest. `level` is fixed at 32 for this release; `batch`
is a positive symbolic dimension shared by all three tensors.

## 4. Quantities and coordinates

| Quantity | Definition | Physical units | Graph units | Sign/reference convention |
|---|---|---|---|---|
| `air_temperature` | Cell-average thermodynamic temperature | K | dimensionless | Absolute temperature |
| `surface_heat_flux` | Upward heat flux at the lower boundary | W m^-2 | dimensionless | Positive into the column |
| `temperature_increment` | End-of-step temperature minus start-of-step temperature | K | dimensionless | Positive is warming |

The `level` axis has 32 bottom-to-top cells. `z_center[level]` gives cell-center
height and `z_interface[0:33]` gives interfaces in metres above the lower
boundary. Values are cell averages. There is no periodic boundary. The upper
boundary has zero prescribed heat flux. The coordinate artifact identifies
the exact values, dtype, order, and digest.

Output timestamp is input timestamp plus 60 seconds. `surface_heat_flux` is a
constant average over the half-open interval from the input timestamp to the
output timestamp.

## 5. Domains

The four domains are separate even where their numerical ranges overlap.

| Domain ID | Kind | Membership rule |
|---|---|---|
| `interface-domain` | Admissible | Finite temperature in `[180, 340]` K at all 32 levels; finite flux in `[-100, 800]` W m^-2; exact declared vertical grid |
| `training-domain` | Training | Cases in split `train-v1`; temperatures in `[220, 315]` K, fluxes in `[0, 500]` W m^-2, no inversions stronger than the declared gradient threshold |
| `evaluation-domain` | Evaluation | Independent configurations in split `test-v1`, including flux endpoints and inversion stress cases; exact case membership is immutable |
| `intended-domain` | Intended use | Subset of `evaluation-domain` with temperature in `[230, 310]` K and flux in `[0, 400]` W m^-2 |
| `ood-high-flux` | Out of domain | Test configurations with flux in `(500, 800]` W m^-2, outside training support only in the forcing variable |

The independent split unit is a complete simulated column trajectory. No time
steps from one trajectory occur in more than one partition.

## 6. Transformations

The normalization artifact contains arrays `mu_T[level]`, `sigma_T[level]`, and
scalar `sigma_q`, fitted only on `train-v1`. All scales are strictly positive.

For batch item `b` and level `k`:

```text
x_T[b,k] = (T[b,k] - mu_T[k]) / sigma_T[k]
x_q[b,0] = q[b] / sigma_q
dT[b,k]  = y[b,k] * sigma_T[k]
T_next[b,k] = T[b,k] + dT[b,k]
```

No clipping occurs. A non-finite value or input outside `interface-domain`
causes rejection before inference. Normalization and state update occur outside
ONNX. They do not occur inside the graph.

The effective process graph is:

```text
physical T ---------> validate -----> normalize T ---+
                                                       +-> ONNX -> denormalize dT
physical surface q -> validate -----> normalize q ---+              |
physical T ----------------------------------------------------------+-> add -> T_next
                                                                        |
                                                                        +-> validate output
```

Every edge in the machine-readable process graph carries quantity, units,
dtype, logical shape, axes, and state version. `T` entering the addition is the
same state version used to form the graph input.

## 7. Iteration contract

1. Read `T_n` at timestamp `t_n` and the average surface flux for
   `[t_n, t_n + 60 s)`.
2. Execute the effective process graph once to obtain `T_(n+1)`.
3. Reject the step if any value is non-finite or outside the interface domain.
4. Use `T_(n+1)` as state for the next step; no hidden recurrent state is kept.
5. A restart consists of an admissible temperature column, timestamp, vertical
   coordinate identity, and forcing position.

The longest evaluated rollout is 360 steps under `intended-domain` forcing.
The Model Card does not support longer rollouts.

## 8. Verification cases

### `nominal-step`

- Model: `thermal-step-graph` with exact digest `<digest>`.
- Runtime: `runtime-cpu32`, float32, fallback disabled.
- Inputs: `case-input`, stored in physical space.
- Expected output: `case-output`, stored in physical space.
- Process: full validation, normalization, ONNX invocation, denormalization,
  state update, and output validation.
- Comparison: per-level
  `abs(actual - expected) <= 2e-5 K + 2e-6 * abs(expected)`.
- Expected nondeterministic variation: none for the declared configuration.

The input has nonzero, nonuniform temperature and nonzero flux. A negative
integration check swaps levels 3 and 4 before preprocessing and confirms that
the verification procedure fails.

### `boundary-flux`

This case exercises the maximum intended-use heat flux and checks the lower
boundary sign convention. A separate dynamic-batch case executes batch sizes 1
and 7.

These cases detect package integration errors. They are not evidence that the
learned dynamics are scientifically accurate.

## 9. Scientific claims and evidence

### Claim `rollout-rmse`

Within `intended-domain`, a 360-step rollout has final-time temperature RMSE no
greater than 0.60 K relative to the declared numerical reference.

| Item | Declared value |
|---|---|
| Independent cases | 120 complete trajectories from `test-v1` |
| Metric | Square root of the unweighted mean squared error over 32 levels, then arithmetic mean across trajectories |
| Result | 0.42 K with bootstrap 95% interval `[0.38, 0.47]` K |
| Baseline | Persistence, 1.31 K under the same calculation |
| Acceptance criterion | `<= 0.60 K`, declared before evaluation |
| Reference uncertainty | Discretization comparison contributes an estimated 0.05 K to final-time differences |
| Conclusion | Passed for the stated process graph, domain, horizon, and runtime scope |
| Detailed evidence | `results`, claim locator `rollout-rmse` |

### Claim `energy-drift`

The absolute column-energy drift after 360 steps is no greater than the
declared threshold after accounting for integrated boundary flux.

The illustrative result fails the criterion in 3 of 120 trajectories. The
claim status is `not-supported`, and the failure is listed in the Model Card.
It is not hidden by the passed RMSE claim.

### Out-of-domain result

`ood-high-flux` differs from training support in surface heat flux. RMSE rises
monotonically over the tested bins and exceeds the intended-use criterion above
650 W m^-2. No generalization claim is made for this domain.

The release remains `research` because the example has no actual released
artifacts and because the conservation claim failed. A real validated status
would require complete evidence, validation authority, and a dated decision.

## 10. Provenance

The abstract provenance graph records:

- numerical-solver revision, governing equations, boundary conditions,
  discretization, 32-level grid, time step, convergence study, and failed-run
  exclusions;
- immutable trajectory identities and exact partition membership;
- sampling and quality-control activities;
- training code, configuration, loss, optimizer, precision, seed, stopping
  rule, and checkpoint rule;
- source-checkpoint digest;
- exporter version, opset, options, and graph transformations;
- source-framework-to-ONNX comparison cases and maximum observed difference;
  and
- package assembly activity and responsible agent.

Every activity identifies its immutable inputs and outputs. Normalization
derivation points only to `train-v1`.

## 11. Runtime scope

| Configuration | Runtime/provider | Hardware | Precision | Determinism | Verified cases |
|---|---|---|---|---|---|
| `runtime-cpu32` | `<declared implementation/version>` | CPU class `<class>` | float32 | Deterministic with declared thread settings | All cases |
| `runtime-accelerator32` | `<declared implementation/version>` | Accelerator class `<class>` | float32 | Maximum observed repeat variation declared | Nominal and dynamic-batch cases |

Fallback is disabled in both configurations. No custom operators or external
ONNX data are used. Memory limits and performance results identify batch size,
hardware, warm-up, repetitions, and whether transformations are included.

## 12. Profile decisions

| Profile | Decision | Reason |
|---|---|---|
| `core` | Applicable | All packages |
| `iterative` | Applicable | Output temperature becomes next-step input |
| `spatial` | Applicable | Temperature is defined on vertical cells |
| `single-pass` | Not applicable to the claimed mode | Scientific claim uses a rollout |
| `coupled` | Not applicable | No external model exchanges data during a step |
| `multi-graph` | Not applicable | One ONNX graph is required |
| `custom-operator` | Not applicable | Only standard ONNX operators are imported |
| `external-artifact` | Not applicable | All required artifacts are packaged |
| `stochastic` | Not applicable | Model behavior is deterministic in the claimed configurations |
| `scientifically-validated` | Not applicable | Release does not claim validated status |
| `training-reproducible` | Not applicable | No reproducible-training claim is made |
| `operational` | Not applicable | No operational deployment is claimed |

The physical package would pass the applicable core, iterative, and spatial
checks before claiming overall conformance.
