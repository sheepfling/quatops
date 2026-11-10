"""Generate the OpenUSD converter input and transform stack."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.converters.openusd.inputs import build_artifacts, serialize_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    root = parser.parse_args().repository_root.resolve(strict=True)
    target = root / "experiments/converters/openusd/26.11"
    inputs, stage = build_artifacts(root)
    (target / "converter_inputs.json").write_bytes(serialize_inputs(inputs))
    (target / "quatops_hierarchy.usda").write_bytes(stage.encode("utf-8"))
    print(f"wrote QuatOps converter stage: {target / 'quatops_hierarchy.usda'}")


if __name__ == "__main__":
    main()
