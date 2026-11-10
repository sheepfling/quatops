# Supporting native evidence

This directory holds native observations that support shared mathematical or
geographic claims but are not named QuatOps converters. Converter-specific
captures and their pinned environments belong together under
[`experiments/converters`](../converters/README.md).

## Shared mathematical inputs

Read the [shared native capture inputs and helpers](shared/README.md) for the
corpus and capture-envelope conventions.

Converter acceptance packages use the canonical witness inputs in
[`experiments/interoperability`](../interoperability/README.md) or a
target-specific input recipe.

## Supporting captures

See the canonical [interoperability evidence atlas](../../docs/INTEROPERABILITY_EVIDENCE.md)
for captured evidence and remaining scope. Versioned reproduction guides:

- [PROJ 9.5.1](proj/9.5.1/README.md)
- [Sophus 1.24.6](sophus/1.24.6/README.md)
- [OpenXR SDK 1.1.65](openxr/1.1.65/README.md)

The OpenXR static-space capture directly loads the Khronos software test
runtime with synthetic time. It does not exercise the OpenXR loader path,
a real headset, a vendor runtime, tracking, or a QuatOps adapter.

Native outputs, specification-derived expectations, and independent formula
checks are different evidence types. Only selected, reviewed observations are
retained; broad passing sweeps remain reproducible discovery runs.
