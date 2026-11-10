# Converter experiments

These auxiliary experiments exercise native converter boundaries and data
transports. They do not add runtime APIs under `src/`. The authoritative target
inventory, accepted evidence, and remaining scope are maintained in the
[interoperability evidence atlas](../../docs/INTEROPERABILITY_EVIDENCE.md).

## Where things live

- `experiments/converters/<target>/<version>/` contains target inputs,
  `semantic_contract.json`, generators, and native capture entry points.
- `tests/golden/{oracles,transports}/<target>/<version>/` contains selected
  native witnesses, contracts, and provenance.
- `tests/experiments/converters/<target>/` contains replay acceptance checks.
- `scripts/evidence_schema/semantic_serializers/` maps validated target
  captures, contracts, and inputs into semantic witnesses. MuJoCo and CSPICE
  share a serializer while retaining their target-specific capture models;
  `normalizer.py` validates the resulting shared evidence contract.
- `experiments/pyproject.toml` packages shared evidence helpers and Pydantic
  schemas and declares optional target dependency sets. Each pinned Python
  capture project selects its target extra and keeps a version-specific
  `uv.lock` beside the runner.

The optional groups use portable PEP 621 metadata. The `[tool.uv]` settings in
that file only express resolver-specific constraints between incompatible
native extras and point the local package at this checkout; they do not replace
the portable dependency declarations.

Run all frozen converter acceptance checks from the repository root with:

```shell
uv run --extra dev python -m pytest -q tests/experiments/converters
```

Each experiment manifest identifies its `semantic_contract.json` by path and
digest. The common pipeline verifies and loads that experiment-owned contract,
then passes typed inputs and native
observations to the experiment's semantic serializer. Native capture programs
do not import the common normalizer; the repository-side pipeline imports the
target adapter after the native runtime has exited. Capture exporters use
shared input files and typed evidence models where the target runtime permits
them. Version-specific locks keep native environments reproducible without
putting their dependencies in QuatOps' regular install. CI checks that each
lock matches current package metadata. The retained historical capture locks
predate the `alphus-trial` to `dev-experimental` extra rename; replacing the
new name with the old name in those lockfiles recovers their exact recorded
bytes. Native package versions and artifact hashes did not change in that
metadata repair.
`python -m scripts.validate_converter_evidence --repository-root .` checks the
normalized evidence in memory without generating a second data product.

To find a target runbook, inspect the README beside that target's capture
script. The directory names and optional dependency groups in
[`pyproject.toml`](../pyproject.toml) are the versioned source of truth; this
index intentionally does not repeat the changing target list.
