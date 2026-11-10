# OpenXR 1.1.65 pose and frame evidence

This product keeps the OpenXR pose contract, the pinned header ABI probe, and
one small native test-runtime frame-chain witness. Source pins, run details,
and artifact hashes are in [`manifest.json`](manifest.json).

The experiment-owned [`semantic_contract.json`](semantic_contract.json)
records the versioned pose facts, LOCAL-space declaration, and target/base
relations exercised by the fixture. The verifier loads it through a strict
Pydantic model and checks its manifest SHA-256 before interpreting capture
fields. This package remains declared evidence, not a native-accepted QuatOps
converter.

## Captured contract

The SDK header probe records the size and field offsets of `XrQuaternionf` and
`XrPosef` in [`header_abi.json`](header_abi.json). The OpenXR specification
defines `XrQuaternionf` as a unit quaternion stored as `x, y, z, w`; `XrPosef`
contains orientation and position, with position in meters and orientation
applied before translation. `xrLocateSpace(target, base, time, ...)` returns
the pose of `target` in `base`, so the evidence labels that direction as
`base_from_target`.

[`witnesses/test_runtime_frame_chain.json`](witnesses/test_runtime_frame_chain.json)
retains three noncommuting orientations and translations for LOCAL spaces A,
B, and C. It contains native `xrLocateSpace` results for `A_from_B`,
`B_from_C`, and the direct `A_from_C` lookup. The test runtime reports valid
and tracked position and orientation for these static spaces. The verifier
checks each result against the assigned input poses and checks that composing
the two adjacent relations reproduces the direct relation and a transformed
point.

The pinned Khronos test runtime is built and loaded directly through its
runtime-negotiation entry point. It advertises and enables `XR_MND_headless`,
so no graphics binding or system runtime registration is used. The retained
observations are native OpenXR API outputs from that test runtime; the
separate SciPy checks are formula checks and do not execute QuatOps.

## Scope limits

The Khronos test runtime is software for API testing. Its space offsets are
static values supplied to `xrCreateReferenceSpace`; this capture does not
observe a headset, tracking system, or vendor runtime. The synthetic
`XrTime` value is accepted by this static test runtime and is not a production
display-time example. This product establishes OpenXR field and frame-chain
behavior for the pinned specification and test runtime; it does not claim a
validated QuatOps adapter or universal OpenXR world frame.

OpenXR specifies the handedness and natural axes for reference spaces:
`LOCAL` uses +X right, +Y up, and -Z forward, while the particular LOCAL origin
is runtime-selected. These specification facts are not observations of a
vendor runtime. No source model-forward axis is inferred from an engine or
runtime name.

## Reproduce the captures

Fetch `release-1.1.65` from both repositories in [`source.json`](source.json),
then compile and run from this directory. These commands use GCC 15.2 on
Windows x86-64 and `uv` for the pinned Python formula-check dependencies.

```powershell
$Sdk = 'C:/path/to/OpenXR-SDK'
$SdkSource = 'C:/path/to/OpenXR-SDK-Source'
$CaptureDir = Join-Path $env:TEMP 'quatops-openxr-1.1.65'
$env:PYTHONPATH = (Resolve-Path ..\..\..\..).Path
$CaptureBytes = 'import pathlib,subprocess,sys; pathlib.Path(sys.argv[1]).write_bytes(subprocess.check_output(sys.argv[2:]))'
New-Item -ItemType Directory -Force -Path $CaptureDir | Out-Null

g++ -std=c++17 -O2 -I (Join-Path $Sdk 'include') `
  capture_header_abi.cpp -o (Join-Path $CaptureDir 'capture_header_abi.exe')
python -c $CaptureBytes `
  (Join-Path $CaptureDir 'header_abi.json') `
  (Join-Path $CaptureDir 'capture_header_abi.exe')

g++ -std=c++17 -shared `
  -I (Join-Path $Sdk 'include') `
  -I (Join-Path $SdkSource 'src') `
  -I (Join-Path $SdkSource 'src/common') `
  (Join-Path $SdkSource 'src/tests/test_runtimes/runtime_test.cpp') `
  (Join-Path $SdkSource 'src/tests/test_runtimes/test_runtime.def') `
  -o (Join-Path $CaptureDir 'test_runtime.dll')
g++ -std=c++17 -O2 -I (Join-Path $Sdk 'include') `
  capture_test_runtime.cpp -o (Join-Path $CaptureDir 'capture_test_runtime.exe')
python -c $CaptureBytes `
  (Join-Path $CaptureDir 'test_runtime_capture.json') `
  (Join-Path $CaptureDir 'capture_test_runtime.exe') `
  (Join-Path $CaptureDir 'test_runtime.dll')

uv run --no-project --python 3.12 `
  --with-requirements requirements-reference.txt `
  python verify_test_runtime.py `
  (Join-Path $CaptureDir 'test_runtime_capture.json') `
  --retain-to witnesses/test_runtime_frame_chain.json
```

The `--retain-to` option validates the typed capture and writes it with the
repository's compact evidence JSON formatter. The complete successful native
stdout is hashed in the manifest; only the selected frame-chain witness is
retained. Primary references are the OpenXR 1.1.65
[`XrQuaternionf`](https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrQuaternionf.html),
[`XrPosef`](https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrPosef.html),
[`xrLocateSpace`](https://registry.khronos.org/OpenXR/specs/1.1/man/html/xrLocateSpace.html),
and [`XrReferenceSpaceType`](https://registry.khronos.org/OpenXR/specs/1.1/man/html/XrReferenceSpaceType.html)
pages, plus the pinned
[`Khronos test-runtime source`](https://github.com/KhronosGroup/OpenXR-SDK-Source/tree/release-1.1.65/src/tests/test_runtimes).
