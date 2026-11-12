"""Check a public QuatOps boundary by observing a pinned Three.js object.

Run after installing development dependencies and the pinned Three.js fixture:
    npm ci --prefix experiments/transports/gltf/2.0.0/threejs/0.187.0
    python examples/converters/threejs_consumer.py
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, FiniteFloat

from quatops.experimental.converters import OrientationConverter
from quatops.experimental.engines import SignedAxisMap
from quatops.experimental.semantics import (
    CANONICAL_BODY_RFU,
    CANONICAL_WORLD_RFU,
    CoordinateFrame,
    PhysicalOrientation,
)
from quatops.python import hamilton

Vector3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
Quaternion4 = tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]


class ProbeRequest(BaseModel):
    """Explicit native quaternion and independently declared model markers."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    quaternion_xyzw: Quaternion4
    forward_tip_xyz: Vector3
    up_tip_xyz: Vector3


class NativeObservation(BaseModel):
    """Resolved output from exactly the pinned native object probe."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["0.187.0"]
    quaternion_xyzw: Quaternion4
    forward_xyz: Vector3
    up_xyz: Vector3
    float32_quaternion_xyzw: Quaternion4
    float32_forward_xyz: Vector3
    float32_up_xyz: Vector3


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    native_module = (
        root
        / "experiments/transports/gltf/2.0.0/threejs/0.187.0/node_modules/three/build/three.module.js"
    )
    node = shutil.which("node")
    if node is None or not native_module.is_file():
        raise SystemExit(
            "Install Node.js and run npm ci in the pinned Three.js experiment directory."
        )
    world = CANONICAL_WORLD_RFU.bind("app.world.rfu", "app.world.axes")
    body = CANONICAL_BODY_RFU.bind("arrow.body.rfu", "arrow.body.axes")
    # Application declarations: source +Y is forward and +Z is up. This
    # particular scene/model uses -Z forward and +Y up, independently of its API.
    basis = SignedAxisMap(("x", "-z", "y"))
    scene = CoordinateFrame(
        "app.scene",
        world.reference,
        ("right", "up", "back"),
        basis,
        "right",
        "world",
        world.reference_convention,
    )
    model = CoordinateFrame(
        "arrow.model",
        body.reference,
        ("right", "up", "back"),
        basis,
        "right",
        "body",
        body.reference_convention,
    )
    converter = OrientationConverter("threejs.quaternion", world, body, scene, model)
    orientation = PhysicalOrientation.from_wxyz((0.8, 0.2, -0.4, 0.4), world, body)
    native = converter.to_target(orientation)
    x, y, z, w = native.values
    request = ProbeRequest(
        quaternion_xyzw=(x, y, z, w),
        forward_tip_xyz=(0, 0, -2),
        up_tip_xyz=(0, 0.5, 0),
    )
    result = subprocess.run(
        [node, str(Path(__file__).with_suffix(".mjs")), str(native_module)],
        input=request.model_dump_json(),
        capture_output=True,
        text=True,
        check=True,
    )
    observation = NativeObservation.model_validate_json(result.stdout)
    assert observation.quaternion_xyzw == native.values
    assert observation.float32_quaternion_xyzw != native.values
    for source_direction, observed, observed_float32 in (
        ((0.0, 1.0, 0.0), observation.forward_xyz, observation.float32_forward_xyz),
        ((0.0, 0.0, 1.0), observation.up_xyz, observation.float32_up_xyz),
    ):
        expected = basis.inverse().vector(
            hamilton.rotate(orientation.quaternion_wxyz, source_direction)
        )
        assert all(abs(a - b) <= 2e-14 for a, b in zip(observed, expected))
        assert all(abs(a - b) <= 3e-7 for a, b in zip(observed_float32, expected))
    restored = converter.from_target(
        replace(native, values=observation.float32_quaternion_xyzw)
    )
    assert hamilton.rotations_equivalent(
        restored.quaternion_wxyz, orientation.quaternion_wxyz, atol=3e-7
    )
    try:
        converter.from_target(
            replace(native, to_frame=replace(scene, name="another.scene"))
        )
    except ValueError as error:
        assert "frame definitions" in str(error)
        print(f"Deliberate frame misuse rejected: {error}")
    else:
        raise AssertionError("the gateway accepted a different scene frame")
    print(
        f"Three.js {observation.version}: native forward/up markers and float32 assignment passed"
    )


if __name__ == "__main__":
    main()
