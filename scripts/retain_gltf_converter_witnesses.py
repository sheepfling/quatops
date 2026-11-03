"""Retain the minimal glTF asset and native Three.js loader observations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/converters/gltf/2.0.0"
DESTINATION = ROOT / "tests/golden/transports/gltf/2.0.0/threejs/0.187.0/converter"
THREE_ROOT = ROOT / "experiments/quatops_capture_wave2/js/node_modules/three"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    args = parser.parse_args()
    capture = args.capture_dir.resolve()
    native_manifest = json.loads((capture / "manifest.json").read_text(encoding="utf-8"))
    observations = json.loads((capture / "observations.json").read_text(encoding="utf-8"))
    if len(observations) != 2 or {row["node"] for row in observations} != {"QuatOpsParent", "QuatOpsChild"}:
        raise ValueError("expected native observations for the two converter-generated nodes")

    DESTINATION.mkdir(parents=True, exist_ok=True)
    for source, name in ((HERE / "converter_inputs.json", "inputs.json"), (HERE / "quatops_hierarchy.gltf", "quatops_hierarchy.gltf")):
        shutil.copyfile(source, DESTINATION / name)
    witnesses_path = DESTINATION / "witnesses.json"
    witnesses_path.write_text(json.dumps({"schema_version": "1.0", "evidence_kind": native_manifest["evidence_kind"], "observations": observations}, indent=2) + "\n", encoding="utf-8", newline="\n")
    contract = {
        "contract_id": "quatops-gltf-static-hierarchy-v1",
        "target": "glTF 2.0 loaded by Three.js GLTFLoader 0.187.0",
        "algebra": "Hamilton active quaternion",
        "source_storage": "QuatOps WXYZ",
        "serialized_storage": "glTF XYZW",
        "canonical_frame": {"handedness": "right", "positive_x": "right", "positive_y": "forward", "positive_z": "up"},
        "asset_frame": {"handedness": "right", "positive_x": "right", "positive_y": "up", "positive_z": "backward", "forward": "-Z"},
        "destination_axes_from_source": ["x", "z", "-y"],
        "transform_semantics": "node local to parent; world = parent world * child local; column-vector action",
        "scope": ["static node TRS serialization", "hierarchy composition", "native loader local/world matrices", "point action and inverse point round trip"],
        "excludes": ["animation", "mesh model-forward convention", "compression extensions", "other GLTFLoader versions"],
    }
    contract_path = DESTINATION / "contract.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8", newline="\n")
    (DESTINATION / "manifest.json").write_text(json.dumps({
        "schema_version": "1.0", "review_state": "QuatOps-generated asset loaded by pinned native GLTFLoader; replay tests passed",
        "target": native_manifest["target"], "execution": native_manifest["execution"],
        "provenance": {
            "input_sha256": sha256(DESTINATION / "inputs.json"),
            "asset_sha256": sha256(DESTINATION / "quatops_hierarchy.gltf"),
            "native_observations_sha256": sha256(witnesses_path),
            "full_run_manifest_sha256": sha256(capture / "manifest.json"),
            "full_observation_sha256": sha256(capture / "observations.json"),
            "generator_sha256": sha256(HERE / "create_asset.py"),
            "native_loader_script_sha256": sha256(HERE / "capture_threejs.mjs"),
            "package_lock_sha256": native_manifest["package_lock_sha256"],
            "contract_sha256": sha256(contract_path),
        },
        "selection": {"retained_observations": len(observations), "native_observations": native_manifest["observations"]["count"],
                      "policy": "one native local/world hierarchy witness per serialized node"},
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"retained glTF converter asset and {len(observations)} Three.js loader witnesses at {DESTINATION}")


if __name__ == "__main__":
    main()
