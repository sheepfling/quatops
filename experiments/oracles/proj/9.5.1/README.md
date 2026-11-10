# PROJ 9.5.1 geographic capture

This versioned capture package records selected WGS84 geodetic/ECEF and
topocentric observations through pyproj. It is supporting frame evidence, not
a quaternion converter or a Cesium runtime. The frozen 11-case native record
is in [`tests/golden/oracles/proj/9.5.1`](../../../../tests/golden/oracles/proj/9.5.1/manifest.json).

The original native run used Python `3.13.5`, pyproj `3.7.2`, and PROJ `9.5.1`
on Linux x64. This package retains the input corpus, current exporter, and
pinned pyproj requirement for repeat runs. Its exporter was
reorganized for this versioned package after the selected observations were
captured; source hashes for the historical native run and current reproducer
are recorded separately in the fixture. The retained fixture keeps every
selected binary64 value, only the origin/basis/probe outputs needed per case,
and omits verified duplicate hexadecimal sidecars. The other former readbacks
were checked against their source points before removal.

From the repository root in PowerShell, create an isolated environment and
write a fresh capture outside the repository:

```powershell
$env:PYTHONPATH = (Get-Location).Path
$captureDir = Join-Path $env:TEMP 'quatops-captures\proj-9.5.1\latest'
uv run --locked --project experiments/converters/python --extra proj --python 3.13.14 `
  python experiments\oracles\proj\9.5.1\capture_proj.py `
  --repository-root . --geographic experiments\oracles\shared\geographic_inputs_v1.json --output $captureDir
```

The required `--geographic` input in the example is the shared
[`geographic_inputs_v1.json`](../../shared/geographic_inputs_v1.json). The
shared lock includes Pydantic for typed capture records and pyproj for the
native transforms. The exporter writes observations and provenance directly,
and validates the geographic input corpus with
the shared strict Pydantic schema, disables network access, and uses WGS84
without external datum grids. See
[`docs/oracles/proj.md`](../../../../docs/oracles/proj.md) for the measured
scope and numerical limits.
