# PROJ 9.5.1 geographic capture

This versioned capture package records selected WGS84 geodetic/ECEF and
topocentric observations through pyproj. It is supporting frame evidence, not
a quaternion converter or a Cesium runtime. The frozen 11-case native record
is in [`tests/golden/oracles/proj/9.5.1`](../../../../tests/golden/oracles/proj/9.5.1/manifest.json).

The original native run used Python `3.13.5`, pyproj `3.7.2`, and PROJ `9.5.1`
on Linux x64. This package retains the input corpus, capture helper, current
exporter, and pinned pyproj requirement for repeat runs. Its exporter was
reorganized for this versioned package after the selected observations were
captured; source hashes for the historical native run and current reproducer
are recorded separately in the fixture.

From the repository root in PowerShell, create an isolated environment and
write a fresh capture outside the repository:

```powershell
$envDir = Join-Path $env:TEMP 'quatops-captures\proj-9.5.1'
uv venv --python 3.13.14 $envDir
uv pip install --python (Join-Path $envDir 'Scripts\python.exe') -r experiments\oracles\proj\9.5.1\requirements.txt
$captureDir = Join-Path $env:TEMP 'quatops-captures\proj-9.5.1\latest'
& (Join-Path $envDir 'Scripts\python.exe') experiments\oracles\proj\9.5.1\capture_proj.py --output $captureDir
```

The local input file defaults to
[`geographic_inputs.json`](geographic_inputs.json). The exporter disables
network access and uses WGS84 without external datum grids. See
[`docs/oracles/proj.md`](../../../../docs/oracles/proj.md) for the measured
scope and numerical limits.
