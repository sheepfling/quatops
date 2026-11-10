"""Generate the glTF converter input and static hierarchy."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.converters.gltf.inputs import build_artifacts, serialize_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    root = parser.parse_args().repository_root.resolve(strict=True)
    target = root / "experiments/converters/gltf/2.0.0"
    inputs, asset = build_artifacts(root)
    (target / "converter_inputs.json").write_bytes(
        serialize_artifact(inputs.model_dump(mode="json"))
    )
    (target / "quatops_hierarchy.gltf").write_bytes(serialize_artifact(asset))
    print(f"wrote QuatOps inputs and glTF asset: {target / 'quatops_hierarchy.gltf'}")


if __name__ == "__main__":
    main()
