# Canonical interoperability inputs

[`v1/canonical_inputs.json`](v1/canonical_inputs.json) defines the shared
right-handed source frame, active Hamilton WXYZ contract, eight unit
orientations, mixed-vector probe, and both noncommuting product orders. The
native capture programs use the same hand-authored inputs and call each pinned
target's own vector action; they do not import QuatOps or SciPy.

Full engine discovery exports go to temporary paths. The versioned selected
native values are in `tests/oracles/<engine>/<version>/witnesses.json`, with
their contract and provenance in the same directory. The selector copies
chosen native results without regenerating them. The shared mathematical
property suite generates all 48 signed-axis maps in memory instead of
retaining a second map corpus.

If a selected value changes or a defect is found, retain the minimized input
and native re-execution under a new contract/witness version; do not silently
extend the compatibility claim for a previous build.
