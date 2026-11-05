# Canonical and crystallized converter evidence

[`v1/canonical_inputs.json`](v1/canonical_inputs.json) defines the shared
right-handed source frame, active Hamilton WXYZ contract, eight unit
orientations, mixed-vector probe, and both noncommuting product orders. The
native capture programs use the same hand-authored inputs and call each pinned
target's own vector action; they do not import QuatOps or SciPy.

## Typed target packages and common product

Each target's contract, inputs, and native observations have strict Pydantic
models in `scripts/evidence_schema/targets.py`. Unknown fields fail validation.
The target-specific records are reduced into the shared typed models in
`scripts/evidence_schema/crystallized.py`; frame declarations and conversion
rules come from the corresponding target contract and are never inferred from
the product name.

Regenerate the compact proof product after changing a capture, contract, or
retention rule:

```powershell
.venv\Scripts\python.exe -m scripts.crystallize_converter_evidence --repository-root .
.venv\Scripts\python.exe -m scripts.crystallize_converter_evidence --repository-root . --output $env:TEMP\quatops-converters.json
.venv\Scripts\python.exe -m scripts.crystallize_converter_evidence --repository-root . --check --output $env:TEMP\quatops-converters.json
```

Validate the typed source packages independently:

```powershell
.venv\Scripts\python.exe -m scripts.validate_converter_evidence --repository-root .
```

The first command validates and reduces the typed source packages in memory; it
does not write a second checked-in copy of the captured values. Pass `--output`
when a portable JSON export is needed, and use `--check --output` to compare an
existing export with the current sources. The common `EvidenceProduct` stores
the shared canonical inputs once, selected native witnesses, normalized target
contracts, and source hashes/runtime provenance. It covers Unreal Engine,
Unity, Godot, Blender, Eigen, MuJoCo, OpenCV, ROS 2/tf2, SpiceyPy/CSPICE,
CesiumJS, glTF through Three.js, and OpenUSD. Godot 4.7 and 4.8 reference one
shared immutable witness set where their retained values match; their version
contracts and provenance remain separate. The raw versioned packages remain
the source for capture review and regeneration.

To add a target or native version, give its contract, inputs, and observations
target-specific Pydantic models, then add a reducer that selects only the
reviewed witnesses needed by the common contract. Reuse a witness-set ID only
when the selected native values are identical; keep version identity and
provenance on each record.

Full discovery sweeps remain outside the retained evidence. The versioned
selected native values, contracts, and provenance live under
`tests/golden/oracles/` and `tests/golden/transports/`; reusable inputs and
generated transport assets stay beside their experiment adapters. Identical
input and asset bytes are referenced from both sides rather than copied. The
reducer copies selected native results without regenerating them. The shared
mathematical property suite generates all 48 signed-axis maps in memory
instead of retaining a second map corpus.

If a selected value changes or a defect is found, retain the minimized input
and native re-execution under a new contract/witness version; do not silently
extend the compatibility claim for a previous build.
