"""SDK-free checks for validation of direct native capture outputs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from experiments.converters.cesium.inputs import build_inputs, serialize_inputs
from experiments.converters.gltf.inputs import build_artifacts, serialize_artifact
from experiments.run_converter_capture import validate_capture
from scripts.evidence_schema.utilities import sha256_file
from tests.support.paths import REPOSITORY_ROOT


@pytest.mark.parametrize("target", ["gltf", "cesium"])
def test_direct_capture_validates_selected_witnesses_and_provenance(
    tmp_path: Path, target: str
) -> None:
    root = REPOSITORY_ROOT
    version = "2.0.0" if target == "gltf" else "1.124.0"
    relative = Path("experiments/converters") / target / version
    source = tmp_path / relative
    source.mkdir(parents=True)
    script = "capture_threejs.mjs" if target == "gltf" else "capture_native.mjs"
    for name in ("semantic_contract.json", script):
        shutil.copyfile(root / relative / name, source / name)
    helper = tmp_path / "experiments/converters/shared/numeric_evidence.mjs"
    helper.parent.mkdir(parents=True)
    shutil.copyfile(root / helper.relative_to(tmp_path), helper)
    dependency_relative = (
        Path("experiments/transports/gltf/2.0.0/threejs/0.187.0/package-lock.json")
        if target == "gltf"
        else relative / "distribution.json"
    )
    dependency = tmp_path / dependency_relative
    dependency.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / dependency_relative, dependency)
    if target == "gltf":
        recipe, asset = build_artifacts(root)
        input_bytes = serialize_artifact(recipe.model_dump(mode="json"))
        asset_path = source / "quatops_hierarchy.gltf"
        asset_path.write_bytes(serialize_artifact(asset))
        fixture = (
            root
            / "tests/golden/transports/gltf/2.0.0/threejs/0.187.0/converter/witnesses.json"
        )
        target_name, package_version = "Three.js GLTFLoader", "0.187.0"
    else:
        input_bytes = serialize_inputs(build_inputs(root))
        fixture = root / "tests/golden/transports/cesium/1.124.0/witnesses.json"
        target_name, package_version = "CesiumJS", "1.124.0"
        installer = source.parent / "distribution.py"
        shutil.copyfile(
            root / "experiments/converters/cesium/distribution.py", installer
        )
    inputs = source / "converter_inputs.json"
    inputs.write_bytes(input_bytes)
    capture = tmp_path / "capture"
    capture.mkdir()
    witnesses = capture / "witnesses.json"
    shutil.copyfile(fixture, witnesses)
    if target == "gltf":
        document = json.loads(witnesses.read_text(encoding="utf-8"))
        document["native_entry_points"][2] = "Vector3.applyMatrix4"
        witnesses.write_text(json.dumps(document), encoding="utf-8")
    manifest = {
        "target": {"name": target_name, "package_version": package_version},
        "semantic_contract": {"sha256": sha256_file(source / "semantic_contract.json")},
        "observations": {
            "sha256": sha256_file(witnesses),
            "count": 2 if target == "gltf" else 5,
        },
        "exporter": {"sha256": sha256_file(source / script)},
        "numeric_helper_sha256": sha256_file(helper),
    }
    if target == "gltf":
        manifest["package_lock_sha256"] = sha256_file(dependency)
        manifest["asset"] = {
            "input_sha256": sha256_file(inputs),
            "sha256": sha256_file(asset_path),
        }
    else:
        manifest["distribution_sha256"] = sha256_file(dependency)
        manifest["distribution_installer_sha256"] = sha256_file(installer)
        manifest["target"]["distribution_integrity"] = json.loads(
            dependency.read_text(encoding="utf-8")
        )["integrity"]
        manifest["inputs"] = {"sha256": sha256_file(inputs)}
    manifest_path = capture / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    validate_capture(target, source, capture)

    if target == "cesium":
        manifest["target"]["distribution_integrity"] = "wrong"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="distribution integrity"):
            validate_capture(target, source, capture)
        manifest["target"]["distribution_integrity"] = json.loads(
            dependency.read_text(encoding="utf-8")
        )["integrity"]
        pin_bytes = dependency.read_bytes()
        pin = json.loads(pin_bytes)
        pin["version"] = "1.124.1"
        dependency.write_text(json.dumps(pin), encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="version differs from the contract"):
            validate_capture(target, source, capture)
        dependency.write_bytes(pin_bytes)

    document = json.loads(witnesses.read_text(encoding="utf-8"))
    if target == "gltf":
        document["observations"][1]["outputs"].pop("world_probe_point_xyz")
    else:
        document["observations"][1]["outputs"].pop("local_basis_and_probe_images")
    witnesses.write_text(json.dumps(document), encoding="utf-8")
    manifest["observations"]["sha256"] = sha256_file(witnesses)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        validate_capture(target, source, capture)
