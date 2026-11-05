# Physical frame semantics

`quatops.semantics` describes what a rotation means independently of an engine,
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
map from the source and destination bases; when both frames name a reference
convention, those declarations must agree.

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
| `OPENXR_SPACE_RUB` | right, up, back (forward is -Z) | canonical world RFU |

These presets describe physical axis semantics. They do not infer an engine's
world axes, an asset's model-forward direction, camera mounting, or a runtime
space's origin. Bind each convention to the actual frame identity in the
application.

[ROS REP-103](https://github.com/ros-infrastructure/rep/blob/master/rep-0103.rst)
describes FLU body, optical camera, ENU, and NED conventions. The
[OpenXR specification](https://registry.khronos.org/OpenXR/specs/1.1/html/xrspec.html)
distinguishes `view`, `local`, and `stage` spaces; bind the actual space
instance. Cesium's
[`eastNorthUpToFixedFrame`](https://cesium.com/learn/cesiumjs/ref-doc/Transforms.html)
depends on a geographic origin and ellipsoid. Inertial-to-fixed mappings
depend on an epoch, as described in the
[SPICE frame reference](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/frames.html).
Origin and epoch resolution belongs in a frame provider or caller-supplied
basis map.

## ECEF-from-body convenience

`ECEFBodyOrientation` is a convenient `q_ECEF_from_body` value using QuatOps'
right-handed ECEF frame and explicit RFU body frame. It wraps a Hamilton WXYZ
quaternion and contains no position or geodetic origin. Convert it to the
general semantic type with `to_physical_orientation()` before using the common
[target converter gateway](converters.md).

The ECEF convenience is one common physical meaning. Local scene, robot,
camera, and inertial orientations use `PhysicalOrientation` with their own
declared frame pair.
