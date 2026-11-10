# Physical frame semantics

Experimental API: see the [compatibility and removal policy](experimental.md).

`quatops.experimental.semantics` describes what a rotation means independently of an engine,
transport, or quaternion coefficient layout. Its common form is a directed
orientation `q_parent_from_child` with an explicit parent frame and child
frame. The stored value is a standard active Hamilton WXYZ quaternion.

## Physical orientation value

`PhysicalOrientation` combines a `HamiltonQuaternion` with two declared
`CoordinateFrame` values:

- `parent_frame` is the frame on the left of `q_parent_from_child`.
- `child_frame` is the frame on the right.
- The action is `v_parent = R(q) @ v_child`.
- Both frame definitions retain names, axes, handedness, basis, and role.
- Parent and child frames must have matching handedness so the represented
  physical mapping is a proper rotation.
- The quaternion must be unit length within `1e-6`. `.normalized()` is an
  explicit operation; constructing the value does not silently normalize or
  change its sign. Direct quaternion-to-quaternion frame conversions preserve
  the global sign. A conversion through a principal rotation vector loses that
  information: `q` and `-q` represent the same vector, and converting back
  returns one canonical quaternion representative.

`PhysicalOrientation.compose()` checks that the intermediate frame
definitions agree before multiplying. Each input is checked against the unit
tolerance separately, so the product of two accepted near-unit values can fall
outside the tolerance. Composition then raises rather than silently
normalizing. Call `.normalized()` on the operands explicitly when normalization
before composition is intended. `PhysicalOrientation.inverse()` swaps the
parent and child frames. `PhysicalOrientationBatch` carries the same frame
definitions for array values; the NumPy gateway checks each quaternion's unit
length when converting a batch.

The lower-level `FramedOrientation` remains available for Hamilton values that
must preserve arbitrary magnitude or only carry frame labels. The physical
semantic type is the stricter application boundary.

## Frame conventions and references

`FrameConvention` is reusable axis-semantic data. Its `role`, positive axes,
handedness, and basis relative to a named `reference_convention` survive when
it is bound to a concrete `CoordinateFrame`.

The `reference` field on `CoordinateFrame` identifies the common physical
basis used to compare frames. Its optional `reference_convention` records the
axis semantics for that basis. A `FrameCorrespondence` derives the coordinate
map from the source and destination bases. Reference convention declarations
must match exactly, including the case where one is missing; shared reference
strings alone do not establish compatible axis semantics.

Use the standard convention's basis when binding a frame with those axes. A
frame provider can supply an origin- or epoch-specific basis together with its
reference convention:

```python
local_enu = WORLD_ENU.bind(
    "enu@site-17",
    reference=ECEF_FRAME.reference,
    basis_to_reference=enu_axes_in_ecef,
    reference_convention=ECEF_FRAME.reference_convention,
)
```

Here `enu_axes_in_ecef` is an `OrthogonalBasisMap` supplied by geospatial code
for the selected origin and ellipsoid. QuatOps defines the semantic boundary
and applies the map; it does not select a geographic origin or time.

## Common physical conventions

| Preset | Axis meaning | Basis reference |
| --- | --- | --- |
| `CANONICAL_WORLD_RFU` | right, forward, up | canonical world RFU |
| `CANONICAL_BODY_RFU` | right, forward, up | canonical body RFU |
| `ROS_BODY_FLU` | forward, left, up | canonical body RFU |
| `ROS_CAMERA_OPTICAL` | right, down, forward | canonical body RFU; camera mounting remains explicit |
| `AEROSPACE_BODY_FRD` | forward, right, down | canonical body RFU |
| `WORLD_ENU` | east, north, up | local ENU reference |
| `WORLD_NED` | north, east, down | the same local ENU reference |
| `ECEF_WORLD` | equatorial prime meridian, 90° east, north pole | ECEF |
| `OPENXR_REFERENCE_SPACE_RUB` | right, up, back (forward is -Z) | canonical world RFU; only standardized OpenXR VIEW, LOCAL, and LOCAL_FLOOR frames |

These presets describe physical axis semantics. They do not infer an engine's
world axes, an asset's model-forward direction, camera mounting, or a runtime
space's origin. Bind each convention to the actual frame identity in the
application.
`OPENXR_SPACE_RUB` remains a compatibility alias for
`OPENXR_REFERENCE_SPACE_RUB`.

[ROS REP-103](https://github.com/openrobotics/reps/blob/104970e7fb6f9a5bac2c0ec820d09ab443e38150/_posts/rep-0103.md)
describes FLU body, optical camera, ENU, and NED conventions. The
[OpenXR specification](https://registry.khronos.org/OpenXR/specs/1.1/html/xrspec.html)
defines +X right, +Y up, and -Z forward for its VIEW, LOCAL, and LOCAL_FLOOR
reference spaces. STAGE defines +Y up but lets the runtime orient its XZ
rectangle axes, and action spaces can have their own role-specific meanings.
The `openxr.view-space` role marks VIEW as a sensor frame, while
`openxr.reference-space` is limited to LOCAL and LOCAL_FLOOR world frames.
Their binders require the matching reference-space type, runtime identity,
session-scoped `XrSpace` and base-space identities, signed `XrTime`, and a
location with `XR_SPACE_LOCATION_ORIENTATION_VALID_BIT`. The tracked flag is
retained as a separate fact; valid does not mean currently tracked. The
binders reject STAGE and do not bind arbitrary `XrSpace` values. Cesium 1.124's
[`eastNorthUpToFixedFrame`](https://cesium.com/downloads/cesiumjs/releases/1.124/Build/Documentation/Transforms.html#eastNorthUpToFixedFrame)
depends on a geographic origin and ellipsoid. Inertial-to-fixed mappings
depend on an epoch, as described in the
[SPICE frame reference](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/frames.html).
Origin and epoch resolution belongs in a frame provider or caller-supplied
basis map.

## Executable role bindings

`SEMANTIC_CONTRACTS` is separate from quaternion representation profiles. Each
published role carries a pinned source, the source claim being applied, its
scope, exclusions, converter IDs, and any named execution evidence. The
generated [semantic assurance matrix](converter-registry.md#semantic-assurance-matrix)
shows representation evidence (R), role authority (S), required physical
alignment (B), and native action evidence (N) separately. R never implies S
or B. R/S/B/N are independent claims, not a single maturity score.

The source links identify the exact target documentation or standard section
used for each role; REP-103 is pinned to a repository commit. The validator
checks those source records against the registered role, converter family,
typed context, and named evidence artifacts. It does not fetch or interpret
remote documentation at build time. The recorded axis claims are also covered
by focused registry tests, while a reviewer can audit each linked primary
source. Native-engine evidence in the matrix is frozen output from the named
runtime, replayed by tests; replay does not relaunch that engine. SciPy-based
formula comparisons are mathematical reference checks and are kept distinct
from native captures.

`bind_semantic_role()` requires a typed `FrameAlignment` that names the
reference frame, basis, alignment purpose, provider, `alignment_source`, and
`alignment_evidence_reference`. The source says whether the physical
relationship came from an application declaration, calibration, survey, or
runtime relation. QuatOps validates the declaration's shape and consistency;
it cannot establish that an application declaration or calibration matches
the real installation. Conversion correctness is conditional on that supplied
physical relationship.

The authoritative axes are combined with the explicit alignment to produce a
`SemanticFrameBinding` containing the ready-to-use `CoordinateFrame`. For
Cesium, the supplied origin and ellipsoid establish the local ENU placement;
an object's body-forward alignment remains a separate caller-supplied frame.

The binding object validates that its frame axes, handedness, role, basis,
reference, alignment purpose, provider, and target family match the selected
contract. The binder rejects a mismatched alignment purpose or provider and a
basis whose determinant disagrees with the documented handedness. This is a
consistency check, not proof that the declared physical alignment is true. It
does not bind a model frame from an engine world-axis contract; the model's
axes remain an explicit application declaration.

Use `bind_cesium_enu_frame()` for the contextual geographic role. It requires a
typed longitude, latitude, height, and ellipsoid, computes local ENU axes in
the declared ECEF reference using the ellipsoid normal at Cesium's Cartesian
origin, and retains both the origin and full ECEF reference frame in the
binding. The height and ellipsoid can affect the computed up axis. This resolves
axes only; it does not construct a position or pose.

```python
from quatops.experimental.semantics import Ellipsoid, GeodeticOrigin, bind_cesium_enu_frame

site_enu = bind_cesium_enu_frame(
    name="site-17.enu",
    origin=GeodeticOrigin(-75.0, 40.0, 1_000.0),
    ellipsoid=Ellipsoid("WGS84", 6_378_137.0, 6_356_752.314245179),
)
```

## ECEF-from-body convenience

`ECEFBodyOrientation` is a convenient `q_ECEF_from_body` value using QuatOps'
right-handed ECEF frame and explicit RFU body frame. It wraps a Hamilton WXYZ
quaternion and contains no position or geodetic origin. Convert it to the
general semantic type with `to_physical_orientation()` before using the common
[target converter gateway](converters.md).

The ECEF convenience is one common physical meaning. Local scene, robot,
camera, and inertial orientations use `PhysicalOrientation` with their own
declared frame pair.
Semantic source claims are audited role by role in [the semantic source audit](semantic-source-audit.md).
