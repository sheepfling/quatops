# Blender capture projects

Blender discovery projects and capture programs live in versioned directories.
Their default output is temporary. Retain only reviewed, discriminating native
witnesses under `tests/oracles/blender/<version>/` using
[`scripts/retain_engine_witnesses.py`](../../scripts/retain_engine_witnesses.py).

- [Blender 5.1.2](5.1.2/README.md): compact quaternion conversion witness and
  reproducible full discovery exporter.

These scripts are auxiliary evidence; they do not add a Blender dependency to
QuatOps or run inside package import.
