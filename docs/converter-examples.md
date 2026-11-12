# Converter integration examples

Experimental API: see the [compatibility and removal policy](experimental.md).

These examples show frame declarations at the call site. Target names select a
coefficient/API profile; they do not select world axes, asset axes, or an
origin. See [the converter contract](converters.md) for sign, input, and
precision behavior.

## One physical attitude in Unreal, Unity, and Godot

[`examples/converters/game_engines.py`](../examples/converters/game_engines.py)
is a runnable example. It declares the captured right-forward-up probe maps,
constructs one `PhysicalOrientation`, and converts it through three reusable
contexts. The printed tuples are values for the declared target fields. The
script checks round-trip orientation and the frame-aware vector identity
`R_target (B_body v) = B_world (R_source v)`. It does not launch those engines
or make claims about imported model axes; native acceptance is established by
the selected engine witnesses and converter tests.

Run it from the repository root:

```shell
python examples/converters/game_engines.py
```

## Receiving external quaternion values

Record the external API's algebra, coefficient order, active/passive direction,
and frame definitions before selecting a numerical module. Imports do not tag
tuples with those facts. Convert at the boundary, then use one chosen numerical
backend internally. For example, when an external contract explicitly supplies
an active Hamilton XYZW child-to-parent orientation:

```python
from quatops.python import hamilton
from quatops.experimental.semantics import PhysicalOrientation

x, y, z, w = received_xyzw
orientation = PhysicalOrientation.from_wxyz(
    hamilton.normalize((w, x, y, z)), parent_frame, child_frame
)
```

Normalization here is deliberate. Invert a passive transform first; a different
algebra needs its own explicit conversion. Use the semantic gateway for frame
and binding checks. These declarations describe the received data, rather than
following from the target's name.

## Observing a native object's forward and up directions

[`threejs_consumer.py`](../examples/converters/threejs_consumer.py) passes a
public gateway result into pinned Three.js 0.187.0, attaches asymmetric
forward/up markers to an object, and checks their observed world directions.
It also performs an explicit float32 storage cast and compares the native
action within its precision tolerance. Finally, it deliberately presents the
result under a different scene frame and checks the reverse gateway's error.

```shell
npm ci --prefix experiments/transports/gltf/2.0.0/threejs/0.187.0
python examples/converters/threejs_consumer.py
```

Use the development environment for this example's typed probe messages.
The [Three.js consumer page](engines/threejs.md) records its native API contract
and precise limits. This executable integration example does not extend any
retained evidence claim about imported mesh-front, camera mounting, or scenes
with scale/shear.

Choose comparisons for the boundary you need: quaternion-to-quaternion
conversion retains the representative sign; native storage can round its
coefficients; `q` and `-q` have equivalent rotation actions. Serialization may
need a deterministic representative, while trajectories need a sign policy
relative to adjacent samples. Neither exact coefficient equality nor
rotation-equivalent round trips alone establish continuity. The Rodrigues
boundary preserves rotation equivalence and loses the original sign.

## Local ENU attitude expressed in ECEF

A geospatial provider supplies the local ENU basis at the chosen origin. Its
columns are the ENU frame's unit axes expressed in ECEF. Rebuild the context if
the origin or reference epoch changes.

```python
from quatops.experimental.converters import OrientationConverter
from quatops.experimental.engines import OrthogonalBasisMap
from quatops.experimental.semantics import (
    CANONICAL_BODY_FRAME,
    ECEF_FRAME,
    WORLD_ENU,
    PhysicalOrientation,
)

# Obtain this origin-specific matrix from the selected geospatial provider.
enu_axes_in_ecef = OrthogonalBasisMap(enu_basis_matrix_from_provider)
local_enu = WORLD_ENU.bind(
    "site-17.enu",
    reference=ECEF_FRAME.reference,
    basis_to_reference=enu_axes_in_ecef,
    reference_convention=ECEF_FRAME.reference_convention,
)
body = CANONICAL_BODY_FRAME
attitude = PhysicalOrientation.from_wxyz(
    (0.8, 0.2, -0.4, 0.4), parent_frame=local_enu, child_frame=body
)

cesium = OrientationConverter(
    "cesium.quaternion",
    source_parent=local_enu,
    source_child=body,
    target_parent=ECEF_FRAME,
    target_child=body,
)
ecef_from_body_xyzw = cesium.to_target(attitude).values
```

The context maps the parent frame from local ENU to ECEF and keeps the body
frame explicit. It does not choose the origin or compute the geodetic basis.
See the [Cesium converter proof](oracles/cesium.md) for captured origins and
native ENU/ECEF action checks.

## Writing a glTF node rotation field

For a static node, convert its frame-labelled orientation and write the
returned XYZW values to `node.rotation`:

```python
native = gltf_converter.to_target(node_orientation)
node_document["rotation"] = list(native.values)
```

`gltf_converter` must bind the source and target node frames used by the
application. This field assignment covers the rotation value; it does not
serialize a complete scene or animation. The
[glTF/Three.js converter proof](transports/gltf.md) exercises a QuatOps-produced
static hierarchy through the independent loader.

A quaternion field assignment and a world-orientation extraction are different
operations. The current gateway does not decompose arbitrary scene matrices.
Scale, mirrored parents, shear, transform order, and import settings need
separate handling before a full transform can be interpreted as a rotation.
