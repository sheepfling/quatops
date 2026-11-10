"""Evidence bundles omit generated inputs after local recapture."""

from pathlib import Path

from scripts.package_evidence_bundle import included_files


def test_generated_gltf_inputs_are_excluded_but_witnesses_are_included(
    tmp_path: Path,
) -> None:
    asset_dir = tmp_path / "experiments/transports/gltf/2.0.0/assets"
    asset_dir.mkdir(parents=True)
    for name in ("rotation_transport.gltf", "rotation_transport.bin", "build_asset.py"):
        (asset_dir / name).write_text("input", encoding="utf-8")
    cesium_dir = tmp_path / "experiments/converters/cesium/1.124.0"
    cesium_dir.mkdir(parents=True)
    (cesium_dir / "converter_inputs.json").write_text("{}", encoding="utf-8")
    gltf_dir = tmp_path / "experiments/converters/gltf/2.0.0"
    gltf_dir.mkdir(parents=True)
    for name in ("converter_inputs.json", "quatops_hierarchy.gltf"):
        (gltf_dir / name).write_text("{}", encoding="utf-8")
    usd_dir = tmp_path / "experiments/converters/openusd/26.11"
    usd_dir.mkdir(parents=True)
    for name in ("converter_inputs.json", "quatops_hierarchy.usda"):
        (usd_dir / name).write_text("input", encoding="utf-8")
    witness = tmp_path / "tests/golden/native/witnesses.json"
    witness.parent.mkdir(parents=True)
    witness.write_text("{}", encoding="utf-8")
    unity = tmp_path / "experiments/unity/6000.5.2f1"
    (unity / "ProjectSettings").mkdir(parents=True)
    (unity / "ProjectSettings/ProjectVersion.txt").write_text("version")
    (unity / "ProjectSettings/GraphicsSettings.asset").write_text("generated")
    (unity / "Logs").mkdir()
    (unity / "Logs/editor.log").write_text("generated")
    (unity / "UserSettings").mkdir()
    (unity / "UserSettings/editor.asset").write_text("generated")
    (unity / "capture.log").write_text("generated")
    unreal_config = tmp_path / "experiments/unreal/5.7.4/Config"
    unreal_config.mkdir(parents=True)
    (unreal_config / "DefaultEngine.ini").write_text("generated")
    egg_info = tmp_path / "experiments/quatops_experiments.egg-info"
    egg_info.mkdir()
    (egg_info / "PKG-INFO").write_text("generated")

    assert {
        path.relative_to(tmp_path).as_posix() for path in included_files(tmp_path)
    } == {
        "experiments/transports/gltf/2.0.0/assets/build_asset.py",
        "experiments/unity/6000.5.2f1/ProjectSettings/ProjectVersion.txt",
        "tests/golden/native/witnesses.json",
    }
