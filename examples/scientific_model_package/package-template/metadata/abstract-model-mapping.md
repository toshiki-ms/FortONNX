# Abstract information model mapping

Mapping identifier: `org.fortonnx.examples.direct-json`<br>
Mapping version: `1`

This package serializes the Scientific ONNX Model Package abstract information
model as UTF-8 JSON. This is a choice made by this example and is not a format
requirement of the specification.

## Root mapping

`package-description.json` contains one JSON object. Each top-level key has the
same name and meaning as the corresponding `Package` field in the abstract
information model.

| Abstract entity | JSON representation |
|---|---|
| Package | Root object |
| Artifact | Object in `artifacts` |
| ModelGraph | Object in `models` |
| TensorContract | Object in `tensors` |
| SymbolicDimension | Object in `symbolic_dimensions` |
| Axis | Object in `axes` |
| Quantity | Object in `quantities` |
| Domain | Object in `domains` |
| Transformation | Object in `transformations` |
| ProcessGraph | Object in `process_graphs` |
| VerificationCase | Object in `verification_cases` |
| Claim | Object in `claims` |
| Evidence | Object in `evidence` |
| Provenance | Object in `provenance` |
| RuntimeConfiguration | Object in `runtime_configurations` |
| ProfileDecision | Object in `profile_decisions` |

Entity collections are JSON arrays. Array order is authoritative whenever the
abstract model defines order. Object member order has no meaning.

## Values and references

- Abstract strings, identifiers, versions, URIs, timestamps, and durations are
  JSON strings.
- Abstract Booleans are JSON Booleans.
- Abstract integers and finite decimal numbers are JSON numbers. Non-finite
  values are prohibited.
- A local entity reference is the referenced entity's `id` encoded as a JSON
  string.
- A package-relative artifact location is a JSON string using `/` separators.
- A fixed tensor dimension is a positive JSON integer. A symbolic tensor
  dimension is its `SymbolicDimension.id` encoded as a JSON string.
- A digest is an object containing `algorithm` and lowercase hexadecimal
  `value` members.

The JSON `null` value is not used for a required abstract field. An optional
field is omitted only when it is absent. `not applicable`, `not evaluated`, and
`unknown` are explicit string or object values and are never represented by
omission or `null`.

## Artifact locators

CSV locators use an object with `header` and `rows` members. `header` lists the
ordered column names. `rows` is either an inclusive one-based row interval or
an explicit ordered list. JSON locators use an RFC-style JSON Pointer string.
Whole-file artifacts use the string `whole artifact`.

Tensor shapes and axes in the package description use ONNX logical order.
Storage order for CSV verification rows is declared separately in the artifact
entry.

## Bootstrap and integrity

The package description does not contain its own digest.
`package-description.sha256` is a detached digest record generated after the
description. The Model Card declares this mapping artifact and its digest.

Every artifact indexed by `package-description.json` contains an exact byte
size and SHA-256 digest. `conformance-report.json` and its digest are detached
validation results produced after the package description and are therefore
outside the indexed digest closure.

Both detached `.sha256` files use example digest-record layout 1: one UTF-8
line containing algorithm name, one ASCII space, lowercase hexadecimal digest,
one ASCII space, relative filename, and newline.

## Decoding errors

A decoder must reject:

- invalid UTF-8 or invalid JSON;
- duplicate object member names;
- non-finite numbers;
- a missing required field or wrong cardinality;
- duplicate entity identifiers;
- a reference resolving to zero or more than one entity;
- an absolute, escaping, or non-normalized package path; and
- a value whose JSON type does not match this mapping.
