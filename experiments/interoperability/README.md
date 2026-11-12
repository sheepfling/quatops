# Canonical converter evidence

[`v1/canonical_inputs.json`](v1/canonical_inputs.json) defines the shared
right-handed source frame, active Hamilton WXYZ contract, eight unit
orientations, mixed-vector probe, and both noncommuting product orders. The
native capture programs use the same hand-authored inputs and call each pinned
target's own vector action; they do not import QuatOps or SciPy.
The capture inputs include quarter turns for discovery, while the retained
quaternion witnesses select five orientations: identity, two unequal mixed
rotations, an antipode, and a non-principal half turn. The native-created
orientation and both product orders remain where applicable. Tests compare
the mixed native basis actions against all 48 signed-axis maps; this does not
assign a physical frame to a mathematics library.

## Typed target packages and normalization

Each target's contract, inputs, and native observations have strict Pydantic
models in `scripts/evidence_schema/converter_captures.py` and
`scripts/evidence_schema/engine_captures.py`. Unknown fields fail validation.
The target-specific records are reduced into the shared typed models in
`scripts/evidence_schema/crystallized.py`; frame declarations and conversion
rules come from the corresponding target contract and are never inferred from
the product name.

Validate and normalize the typed source packages after changing a capture,
contract, or retention rule:

```shell
python -m scripts.verify evidence
```

The validator checks the target-specific contracts and native witnesses and
normalizes them in memory. The retained source values are the evidence; no
generated evidence JSON is needed. The checks cover Unreal Engine,
Unity, Godot, Blender, Eigen, MuJoCo, OpenCV, ROS 2/tf2, SpiceyPy/CSPICE,
CesiumJS, glTF through Three.js, and OpenUSD. Godot 4.7 and 4.8 reference one
shared immutable witness set where their retained values match; their version
contracts and provenance remain separate. The versioned witness packages are
the source for capture review and regeneration.

To add a target or native version, give its contract, inputs, and observations
target-specific Pydantic models, then add a reducer that selects only the
reviewed witnesses needed by the common contract. Share a witness file only
when the selected native values are identical; keep version identity and
provenance on each run record.

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
