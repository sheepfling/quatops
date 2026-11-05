# Converter experiments

These auxiliary experiments exercise native converter boundaries and data
transports. They do not add runtime APIs under `src/`. The authoritative target
inventory, accepted evidence, and remaining scope are maintained in the
[interoperability evidence atlas](../../docs/INTEROPERABILITY_EVIDENCE.md).

## Where things live

- `experiments/converters/<target>/<version>/` contains target inputs,
  generators, and native capture entry points.
- `tests/golden/{oracles,transports}/<target>/<version>/` contains selected
  native witnesses, contracts, and provenance.
- `tests/experiments/converters/<target>/` contains replay acceptance checks.
- `experiments/pyproject.toml` packages shared evidence helpers and Pydantic
  schemas and declares optional target dependency sets. Each pinned Python
  capture project selects its target extra and keeps a version-specific
  `uv.lock` beside the runner.

Run all frozen converter acceptance checks from the repository root with:

```shell
uv run --extra dev python -m pytest -q tests/experiments/converters
```

Capture exporters use shared input files and typed evidence models where the
target runtime permits them. Version-specific locks keep native environments
reproducible without putting their dependencies in QuatOps' regular install.
The reduced evidence product is generated on
demand by [`crystallize_converter_evidence.py`](../../scripts/crystallize_converter_evidence.py);
the repository does not keep a second copy of its derived output.

To find a target runbook, inspect the README beside that target's capture
script. The directory names and optional dependency groups in
[`pyproject.toml`](../pyproject.toml) are the versioned source of truth; this
index intentionally does not repeat the changing target list.
