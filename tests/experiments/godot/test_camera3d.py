"""Replay pinned native camera observations; this test does not launch Godot."""

import hashlib
import json
import shutil

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.spatial.transform import Rotation

from scripts.evidence_schema.godot_camera import GodotCameraCapture, load_camera_capture
from tests.support.paths import REPOSITORY_ROOT


@pytest.mark.parametrize("version", ["4.7-stable", "4.8-dev1"])
def test_camera3d_native_action(version):
    directory = REPOSITORY_ROOT / "tests/golden/godot/camera3d" / version
    manifest = json.loads((directory / "provenance.json").read_text())
    raw = (directory / "witnesses.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest["witnesses_sha256"]
    for path_key, hash_key in [
        ("capture_source", "capture_sha256"),
        ("semantic_contract", "semantic_contract_sha256"),
    ]:
        assert (
            hashlib.sha256(
                (REPOSITORY_ROOT / manifest[path_key]).read_bytes()
            ).hexdigest()
            == manifest[hash_key]
        )
    capture = GodotCameraCapture.model_validate_json(raw)
    assert capture.version == manifest["version"]
    assert capture.build_hash == manifest["build_hash"]
    assert [case.id for case in capture.cases] == ["0_0", "1_2", "2_1"]
    rotations = [
        [0.0, 0.0, 0.0, 1.0],
        [0.2, 0.4, 0.4, 0.8],
        [-0.3, 0.6164414002968976, -0.2, 0.7],
    ]
    for case in capture.cases:
        parent_index, child_index = (int(index) for index in case.id.split("_"))
        np.testing.assert_allclose(
            case.parent_xyzw, rotations[parent_index], atol=1e-7, rtol=0
        )
        np.testing.assert_allclose(
            case.camera_xyzw, rotations[child_index], atol=1e-7, rtol=0
        )
        parent = Rotation.from_quat(case.parent_xyzw)
        child = Rotation.from_quat(case.camera_xyzw)
        world = parent * child
        for name, axis in [
            ("right", [1, 0, 0]),
            ("up", [0, 1, 0]),
            ("forward", [0, 0, -1]),
        ]:
            np.testing.assert_allclose(
                getattr(case.local_axes, name), child.apply(axis), atol=2e-6, rtol=0
            )
            for observed in [case.global_axes, case.camera_axes]:
                np.testing.assert_allclose(
                    getattr(observed, name), world.apply(axis), atol=2e-6, rtol=0
                )
        np.testing.assert_allclose(
            case.center_ray, world.apply([0, 0, -1]), atol=2e-6, rtol=0
        )
        origin = np.array([1.25, -2.5, 4.0]) + parent.apply([-0.5, 1.75, 2.25])
        np.testing.assert_allclose(case.global_origin, origin, atol=2e-6, rtol=0)
        np.testing.assert_allclose(case.camera_origin, origin, atol=2e-6, rtol=0)
    assert not np.allclose(
        capture.cases[1].global_axes.forward, capture.cases[2].global_axes.forward
    )


def test_camera_schema_rejects_unknown_fields():
    path = REPOSITORY_ROOT / "tests/golden/godot/camera3d/4.7-stable/witnesses.json"
    data = json.loads(path.read_text())
    data["calibrated"] = True
    with pytest.raises(ValidationError):
        GodotCameraCapture.model_validate(data)


@pytest.mark.parametrize("version", ["4.7-stable", "4.8-dev1"])
def test_registered_camera_loader(version):
    path = REPOSITORY_ROOT / "tests/golden/godot/camera3d" / version / "witnesses.json"
    assert load_camera_capture(REPOSITORY_ROOT, path).version.startswith(version)


@pytest.mark.parametrize("mutation", ["action", "axes", "native"])
def test_camera_rejects_semantic_or_numeric_forgery_with_updated_hash(
    tmp_path, mutation
):
    relative = "tests/golden/godot/camera3d/4.7-stable"
    shutil.copytree(REPOSITORY_ROOT / relative, tmp_path / relative)
    shutil.copytree(
        REPOSITORY_ROOT / "experiments/godot/camera3d",
        tmp_path / "experiments/godot/camera3d",
    )
    provenance = tmp_path / relative / "provenance.json"
    manifest = json.loads(provenance.read_text())
    if mutation == "native":
        path = tmp_path / relative / "witnesses.json"
        data = json.loads(path.read_text())
        data["cases"][1]["center_ray"][0] += 0.1
        key = "witnesses_sha256"
    else:
        path = tmp_path / manifest["semantic_contract"]
        data = json.loads(path.read_text())
        if mutation == "action":
            data["action"] = "passive"
        else:
            data["world_axes"] = ["up", "right", "backward"]
        key = "semantic_contract_sha256"
    path.write_text(json.dumps(data), encoding="utf-8")
    manifest[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        load_camera_capture(tmp_path, tmp_path / relative / "witnesses.json")
