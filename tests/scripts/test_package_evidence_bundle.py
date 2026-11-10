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

    assert {
        path.relative_to(tmp_path).as_posix() for path in included_files(tmp_path)
    } == {
        "experiments/transports/gltf/2.0.0/assets/build_asset.py",
        "tests/golden/native/witnesses.json",
    }
