"""Exercise public consumer APIs from an isolated installed wheel."""

from __future__ import annotations

import sys
from dataclasses import replace


def scalar_consumer() -> None:
    """Check scalar action, binding copies, and distinct calibration contexts."""
    import quatops
    from quatops.experimental.converters import (
        OrientationConverter,
        accepted_converters,
    )
    from quatops.experimental.engines import SignedAxisMap
    from quatops.experimental.semantics import (
        CANONICAL_BODY_RFU,
        CANONICAL_WORLD_RFU,
        CoordinateFrame,
        FrameAlignment,
        PhysicalOrientation,
        bind_semantic_role,
    )
    from quatops.python import hamilton

    assert len(accepted_converters()) == 14
    assert "numpy" not in sys.modules
    world = CANONICAL_WORLD_RFU.bind("wheel.world", "wheel.world.reference")
    body = CANONICAL_BODY_RFU.bind("wheel.body", "wheel.body.reference")
    basis = SignedAxisMap(("x", "z", "y"))
    alignment = FrameAlignment(
        world,
        basis,
        "scene_alignment",
        "application",
        "application_declaration",
        "app://wheel/scene-v1",
    )
    target_world = bind_semantic_role(
        "unity.world-axes", name="wheel.scene", alignment=alignment
    )
    target_body = CoordinateFrame(
        "wheel.model",
        body.reference,
        ("right", "up", "forward"),
        basis,
        "left",
        "body",
        body.reference_convention,
    )
    converter = OrientationConverter(
        "unity.quaternion", world, body, target_world, target_body
    )
    orientation = PhysicalOrientation.from_wxyz((0.8, 0.2, -0.4, 0.4), world, body)
    native = converter.to_target(orientation)
    copied = replace(converter)
    assert copied.target_semantic_bindings == converter.target_semantic_bindings
    assert copied.from_target(native).quaternion_wxyz == orientation.quaternion_wxyz
    native_wxyz = (native.values[3], *native.values[:3])
    forward = (0.0, 1.0, 0.0)
    expected = basis.vector(hamilton.rotate(orientation.quaternion_wxyz, forward))
    observed = hamilton.rotate(native_wxyz, basis.vector(forward))
    assert all(abs(a - b) < 1e-14 for a, b in zip(expected, observed))
    calibration = replace(
        target_world,
        alignment=replace(
            alignment,
            alignment_source="calibration",
            alignment_evidence_reference="calibration://wheel/revision-2",
        ),
    )
    revised = replace(converter, target_parent=calibration)
    assert len({converter: "original", revised: "revised"}) == 2
    try:
        revised.from_target(native)
    except ValueError as error:
        assert "parent semantic binding" in str(error)
    else:
        raise AssertionError("a different calibration accepted the old native value")
    assert "numpy" not in sys.modules
    print(f"Installed scalar consumer passed: QuatOps {quatops.__version__}")


def numpy_consumer() -> None:
    """Check the installed batch boundary with float32 component-first values."""
    import numpy as np

    from quatops.experimental.converters import OrientationConverter
    from quatops.experimental.semantics import (
        CANONICAL_BODY_RFU,
        CANONICAL_WORLD_RFU,
        PhysicalOrientationBatch,
    )

    world = CANONICAL_WORLD_RFU.bind("wheel.world", "wheel.world.reference")
    body = CANONICAL_BODY_RFU.bind("wheel.body", "wheel.body.reference")
    converter = OrientationConverter("unity.quaternion", world, body, world, body)
    values = np.asarray(
        ((0.8, 0.2, -0.4, 0.4), (0.8, 0.2, -0.4, 0.4)), dtype=np.float32
    ).T
    native = converter.to_target_batch(
        PhysicalOrientationBatch(values, world, body), component_axis=0
    )
    assert native.component_axis == 0
    assert isinstance(native.values, np.ndarray)
    assert native.values.dtype == np.float32
    np.testing.assert_array_equal(converter.from_target_batch(native).values, values)
    print("Installed NumPy consumer passed: float32 component-first round trip")


if __name__ == "__main__":
    if sys.argv[1:] == ["--numpy"]:
        numpy_consumer()
    else:
        scalar_consumer()
