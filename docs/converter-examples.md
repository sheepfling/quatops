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
