# Scientific ONNX Model Package Specification

Specification identifier: `scientific-onnx-model-package`<br>
Specification version: `0.1-draft`<br>
Status: Draft

A conforming scientific ONNX model package supplies an executable ONNX model
together with the information and evidence needed to interpret, verify, and
evaluate its scientific behavior. The package may use any declared
serialization for data other than the ONNX graph.

## Normative documents

- [Core specification](specification.md) defines the package and Model Card
  requirements.
- [Abstract information model](information-model.md) defines the entities,
  identifiers, references, cardinalities, and scientific process graph that a
  machine-readable representation must preserve.
- [Profiles](profiles.md) defines additional requirements for spatial,
  iterative, coupled, stochastic, multi-graph, custom-operator, validated, and
  operational packages.
- [Conformance](conformance.md) defines the checks and the contents of a
  conformance report.

The words **must**, **must not**, **required**, and **prohibited** state
normative requirements. **Should** and **recommended** state requirements whose
omission is allowed only with a documented reason. **May** and **optional**
state permitted choices.

## Supporting documents

- [Model Card template](model-card-template.md) provides a user-facing starting
  point.
- [Worked example](reference-example.md) shows a complete abstract package
  without selecting a payload serialization.
- [Example package](../../examples/scientific_model_package/README.md) includes
  an independent validator for its declared direct-JSON mapping and CSV
  verification layout. Those implementations do not constrain other packages'
  metadata or scientific payload formats.

Supporting documents are informative. If they conflict with a normative
document, the normative document takes precedence.

## Conformance statement

A package is conforming only when it:

1. satisfies the core specification;
2. can be mapped without loss to the abstract information model;
3. declares and satisfies every applicable profile; and
4. passes all applicable required checks in the conformance specification.

Package conformance establishes that the scientific contract is complete,
consistent, traceable, and testable. It does not establish that the model is
scientifically accurate or suitable for a particular decision. Such claims
require the evidence and acceptance criteria defined by the `scientifically-validated` profile.

## Serialization and discovery

No manifest syntax or scientific payload format is prescribed. The Model Card
must identify the machine-readable package description by path or URI, format
and version, and adapter or published mapping. An automated validator is
invoked with those values, the package root, and an expected digest for the
exact package-description bytes.

The expected digest must come from a trusted distribution record, signature, or
validator invocation outside the package description's own digest closure. The
package description indexes every other required artifact but is not required
to contain its own digest. This rule avoids an impossible self-referential
digest. A digest establishes identity only relative to a trusted expected
digest; it does not by itself establish publisher authenticity.

A serialization profile may define automatic discovery rules, but those rules
are not part of the core specification. `README.md` remains the human-readable
package entry point.

## Extensions

Extensions must use a collision-resistant namespace and must declare whether
they are required to use the package. An extension may add information or
stricter constraints. It must not redefine a core field, weaken a core
requirement, or change the meaning of a declared profile.

## Versioning

The specification version identifies this document set. Draft versions may
change incompatibly. A future stable major version will change only when the
abstract information model or conformance meaning changes incompatibly; minor
versions may add optional fields, profiles, or checks.
