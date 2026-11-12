# Engine proof methodology

The authoritative target inventory, version status, retained witnesses, and
remaining scope live in the [interoperability evidence atlas](../INTEROPERABILITY_EVIDENCE.md).
This page is the short index for the shared proof method:

- [Coordinate mapping](coordinate-mapping.md) defines the reusable basis and
  quaternion equations.
- Versioned [engine contracts and witnesses](../../tests/oracles/) retain the
  native values and provenance for each accepted build.

## Shared acceptance checks

Each selected capture checks canonical-to-native and native-to-canonical
conversion, the native action on the three basis vectors, and both orders of a
noncommuting product. Where a probe vector is captured, replay also checks its
physical action in the declared frames. Engine runtime observations remain
separate from SciPy formula checks; generated signed-axis property cases are
checked in memory and are not presented as native captures.

The Unity-to-Godot transfer is an optional pairwise observation from the early
cross-engine investigation. It does not define either adapter's acceptance
criteria or imply other pairwise engine support.

Keep target discoveries in the atlas and target-specific API facts in the
corresponding [engine guide](README.md); do not maintain a second status table
here.
