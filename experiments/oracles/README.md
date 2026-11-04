# Supporting native evidence

This directory holds native observations that support shared mathematical or
geographic claims but are not named QuatOps converters. Converter-specific
captures and their pinned environments belong together under
[`experiments/converters`](../converters/README.md).

## Shared mathematical inputs

[`shared/orientation_inputs_v1.json`](shared/orientation_inputs_v1.json) is a
common input corpus for the shared mathematical oracle tests. Converter
acceptance packages use the smaller canonical witness set in
[`experiments/interoperability`](../interoperability/README.md), rather than
retaining one copy of this broad corpus per target.

## Supporting captures

- [PROJ 9.5.1](proj/9.5.1/README.md): WGS84 geodetic, ECEF, and ENU probes.
- [Sophus 1.24.6](sophus/1.24.6/README.md): SO(3) discovery; no named converter
  is accepted from this first capture.
- [OpenXR SDK 1.1.65](openxr/1.1.65/README.md): header ABI only; no runtime
  pose behavior is claimed.

Native outputs, specification-derived expectations, and independent formula
checks are different evidence types. Only selected, reviewed observations are
retained as goldens; broad passing sweeps remain reproducible discovery runs.
