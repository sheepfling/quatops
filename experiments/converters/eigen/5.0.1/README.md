# Eigen 5.0.1 quaternion-boundary capture

This auxiliary proof compiles a small C++ capture program against the official
Eigen 5.0.1 source archive. The program uses `Eigen::Quaternionf` and
`Eigen::Quaterniond` constructors, `coeffs()`, quaternion/vector products,
`toRotationMatrix()`, matrix-to-quaternion construction, and `AngleAxis`.
It imports and calls no QuatOps code.

The release archive URL and SHA-256 are pinned in
[`experiments/oracles/eigen/5.0.1/source.json`](../../../oracles/eigen/5.0.1/source.json).
Download and unpack it outside the repository, then run from the repository
root:

```powershell
$archive = Join-Path $env:TEMP 'eigen-5.0.1.tar.gz'
$extract = Join-Path $env:TEMP 'quatops-eigen-5.0.1'
Invoke-WebRequest -Uri 'https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz' -OutFile $archive
tar -xzf $archive -C $extract
uv sync --locked --project experiments/converters/python --extra eigen --python 3.13.14
uv run --locked --project experiments/converters/python --extra eigen --python 3.13.14 `
  python `
  experiments\converters\eigen\5.0.1\capture_native.py `
  --repository-root . `
  --eigen-root (Join-Path $extract 'eigen-5.0.1') `
  --eigen-archive $archive
```

The capture verifies the archive digest before compiling with `g++` in C++17
mode. Its Python 3.13 lock is shared across captures through separate extras;
other Python versions require a separately resolved environment. It keeps the
small selected witnesses and build provenance under
[`tests/golden/oracles/eigen/5.0.1/converter`](../../../../tests/golden/oracles/eigen/5.0.1/converter/manifest.json).
The compiler path/version and native executable digest are recorded in the
manifest. The harness retains one native `coeffs()` storage-order observation
per precision, alongside WXYZ coefficients and native actions for each selected
orientation. It emits the selected typed rows directly; the
historical producer source remains available by digest in the provenance
archive. Prepared JSON inputs
are generated in memory; the native input stream and shared canonical source
remain reproducible without a duplicate fixture file. Replay the frozen
native outputs with
[`test_eigen_converter.py`](../../../../tests/experiments/converters/eigen/test_eigen_converter.py).
