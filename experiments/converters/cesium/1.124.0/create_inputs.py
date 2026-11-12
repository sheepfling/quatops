"""Generate Cesium's converter inputs from the shared source cases."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.converters.cesium.inputs import build_inputs, serialize_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    root = parser.parse_args().repository_root.resolve(strict=True)
    output = root / "experiments/converters/cesium/1.124.0/converter_inputs.json"
    inputs = build_inputs(root)
    output.write_bytes(serialize_inputs(inputs))
    print(
        f"wrote {len(inputs.cases)} QuatOps-generated Cesium converter inputs: {output}"
    )


if __name__ == "__main__":
    main()
