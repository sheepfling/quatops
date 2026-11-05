"""Convert one explicitly framed attitude to Unreal, Unity, and Godot fields.

The frame maps below are application declarations matching the captured
right-forward-up probe frames. They are not inferred from an engine name and
do not define imported model axes.
"""

from quatops.converters import OrientationConverter
from quatops.engines import SignedAxisMap
from quatops.python import hamilton
from quatops.semantics import (
    CANONICAL_BODY_RFU,
    CANONICAL_WORLD_RFU,
    CoordinateFrame,
    PhysicalOrientation,
)

SOURCE_WORLD = CANONICAL_WORLD_RFU.bind(
    "application.world.rfu", "application.world.axes"
)
SOURCE_BODY = CANONICAL_BODY_RFU.bind("asset.body.rfu", "asset.body.axes")
ORIENTATION = PhysicalOrientation.from_wxyz(
    (0.8, 0.2, -0.4, 0.4), SOURCE_WORLD, SOURCE_BODY
)
PROBE_BODY = (0.31, -0.72, 0.48)

TARGETS = (
    ("unreal.fquat", ("y", "x", "z"), ("forward", "right", "up")),
    ("unity.quaternion", ("x", "z", "y"), ("right", "up", "forward")),
    ("godot.quaternion", ("x", "-z", "y"), ("right", "up", "back")),
)


def main() -> None:
    for converter_id, target_axes, axis_names in TARGETS:
        basis = SignedAxisMap(target_axes)
        source_to_target = basis.inverse()
        handedness = "right" if basis.determinant == 1 else "left"
        target_world = CoordinateFrame(
            f"{converter_id}.world",
            SOURCE_WORLD.reference,
            axis_names,
            basis,
            handedness,
            role="world",
            reference_convention=SOURCE_WORLD.reference_convention,
        )
        target_body = CoordinateFrame(
            f"{converter_id}.body",
            SOURCE_BODY.reference,
            axis_names,
            basis,
            handedness,
            role="body",
            reference_convention=SOURCE_BODY.reference_convention,
        )
        converter = OrientationConverter(
            converter_id,
            SOURCE_WORLD,
            SOURCE_BODY,
            target_world,
            target_body,
        )
        native = converter.to_target(ORIENTATION)
        restored = converter.from_target(native)
        assert hamilton.rotations_equivalent(
            restored.quaternion_wxyz, ORIENTATION.quaternion_wxyz
        )
        native_wxyz = (native.values[3], *native.values[:3])
        expected_parent_vector = source_to_target.vector(
            hamilton.rotate(ORIENTATION.quaternion_wxyz, PROBE_BODY)
        )
        actual_parent_vector = hamilton.rotate(
            native_wxyz, source_to_target.vector(PROBE_BODY)
        )
        assert all(
            abs(actual - expected) <= 2e-14
            for actual, expected in zip(actual_parent_vector, expected_parent_vector)
        )
        print(f"{converter_id}: {native.values}")


if __name__ == "__main__":
    main()
