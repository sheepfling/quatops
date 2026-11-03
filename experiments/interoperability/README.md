# Canonical interoperability inputs

This directory holds shared, versioned canonical inputs for converter captures.
Engine scripts consume the same hand-authored Hamilton WXYZ orientations and
mixed vectors, then ask their pinned native API to apply the mapped target
quaternion. The scripts do not import QuatOps or SciPy.

Version `v1` is in
[`canonical_inputs.json`](v1/canonical_inputs.json). It declares the
right-handed source frame, active column-vector action, unit inputs, two
noncommuting orientations, and the asymmetric vector used by
[CAP-300](../../docs/engines/CAP-300.md).

Native observations are added to each versioned engine fixture under
`tests/golden/<engine>/<version>/native_rotation.json`. Keep the target frame
map, native quaternion order, direct basis actions, mixed-vector action, and
noncommuting product inside the native capture. The independent package tests
then check both conversion directions and cross-engine action using the frozen
native results.

When the common inputs or their meanings change, make a new sibling corpus
version and retain the previous one. Engine versions should continue to live in
their own engine/version folders.
