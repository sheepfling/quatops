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
.venv\Scripts\python.exe `
  experiments\converters\eigen\5.0.1\capture_native.py `
  --eigen-root (Join-Path $extract 'eigen-5.0.1') `
  --eigen-archive $archive
```

The capture verifies the archive digest before compiling with `g++` in C++17
mode. It keeps the small selected witnesses and build provenance under
`tests/golden/oracles/eigen/5.0.1/converter/`. The compiler path/version and
native executable digest are recorded in the manifest. Prepared JSON inputs
are generated in memory; the native input stream and shared canonical source
remain reproducible without a duplicate fixture file. Replay the frozen
native outputs with `tests/experiments/converters/eigen/test_eigen_converter.py`.
