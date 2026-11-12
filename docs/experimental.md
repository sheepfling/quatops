# Experimental interoperability

## Scope and imports

The new frame-aware runtime APIs are opt-in through these namespaces:

- `quatops.experimental.converters`: `OrientationConverter`, scalar and batch
  conversion helpers, target values, and the converter/evidence registry.
- `quatops.experimental.semantics`: physical orientations, frame declarations,
  frame correspondences, and named frame conventions.
- `quatops.experimental.engines`: general orthogonal and two-frame mappings,
  plus Blender, Godot, Unity, and Unreal storage adapters.

These APIs ship in the ordinary wheel. An optional dependency extra installs
requirements; it does not control whether Python modules ship. The scalar
surface does not require NumPy. Batch conversion loads it when requested.

The existing scalar and NumPy quaternion backends, `SignedAxisMap`, and
`quatops.engines.numpy` remain outside this experimental surface. This boundary
preserves their existing contracts without declaring the alpha package stable.

## Compatibility and migration

Use the experimental paths in new code. Earlier PR paths under
`quatops.converters`, `quatops.semantics`, and the new engine adapter modules
remain deprecated compatibility facades. The new general frame-map exports
on `quatops.engines` are lazy compatibility accessors. Aliases preserve object
identity, so values made through either spelling can be used together.

The experimental API has no stable compatibility guarantee. Changes and
removals must be documented in the changelog. There is no scheduled removal
version for the facades in this change. A future removal should first identify
consumers and provide migration guidance, especially for serialized objects
whose Python module paths may have been persisted. Do not silently rewrite
user data or assume an import alias migrates every serialization format.

## Evidence is a separate dimension

An `accepted_converter` registry state records acceptance of the captured
operations for a particular target version and frame contract. It does not
make the Python API stable or establish universal engine, model, asset-import,
or hardware behavior. OpenXR remains a declared contract and Sophus remains
supporting-oracle evidence.

The normal test suite replays retained observations and independent numerical
references. It does not launch every native engine. Native recaptures must
state the actual runtime versions and operations exercised. Keep provenance,
source snapshots, manifests, and raw observations intact when changing runtime
code or normalizers. Never regenerate historical evidence merely to make a
new implementation pass.

## Promotion and removal

Promotion requires an explicit API review, current full CI, installed-package
checks, independent numerical coverage, and evidence appropriate to the claims
being promoted. Moving code does not automatically promote it or publish a
release. Release tags and package publication are separate decisions.

To retire this area, remove the experimental runtime, its compatibility
facades/lazy accessors, and its dedicated tests, examples, registry generation,
and CI smoke checks together. Verify that the core imports and core regression
suite still run with the experimental package absent. Keep the capture corpus
and provenance archived and labelled, rather than deleting the research.

The source-only `experiments/` tooling and its `quatops-experiments` project
are distinct from the installed `quatops.experimental` runtime. They are
retained as research and reproducibility tools, not stable runtime APIs.
