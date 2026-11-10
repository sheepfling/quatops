# OpenUSD 26.11 converter capture

This pinned `usd-core==26.11` runtime reopens and evaluates the USDA stage
generated from explicit canonical-frame inputs. The fixture retains only the
three native stage observations used by the converter proof.

From the repository root, regenerate the input stage and capture the selected
native witnesses into a temporary directory:

```powershell
$env:PYTHONPATH = (Get-Location).Path
uv run --project experiments/converters/openusd/26.11 --locked --python 3.14.6 `
  python experiments/converters/openusd/26.11/create_stage.py --repository-root .
$captureDir = Join-Path $env:TEMP "quatops-openusd-26.11-capture"
uv run --project experiments/converters/openusd/26.11 --locked --python 3.14.6 `
  python experiments/converters/openusd/26.11/capture_native.py --repository-root . `
  --output $captureDir
```

The direct capture writes `witnesses.json` and `manifest.json`. The original
fixture and its historical provenance are retained separately.

The contract declares a right-handed canonical frame with +X right, +Y
forward, and +Z up, plus the OpenUSD stage frame and explicit axis map. The
native capture checks USDA serialization/reopen, `quatd` orient readback,
translate/orient op order, parent/child composition, `resetXformStack`, point
and direction actions, and inverse point recovery.

The fixture manifest records the runtime, binary, script, input, stage, and
lock hashes. Replay regenerates the input and stage and checks their captured
digests. The scope excludes time samples, scale/shear decomposition,
renderers, and application-specific import pipelines.
