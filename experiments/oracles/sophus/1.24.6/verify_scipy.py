"""Compare selected native Sophus witnesses with independent SciPy rotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import scipy
from retain_witnesses import (
    DEFAULT_PACKAGE,
    load_recipe_manifest,
    load_semantic_config,
    load_shared_witness,
    parse_witness,
    reduce_capture,
)
from scipy.spatial.transform import Rotation

from scripts.evidence_schema.primitives import finite_max_abs
from scripts.evidence_schema.sophus import SophusDiscoveryWitnessSet


def verify(
    witnesses: SophusDiscoveryWitnessSet,
    tolerance: float,
    package: Path = DEFAULT_PACKAGE,
) -> float:
    """Check retained native coefficient, basis-action and SO(3) observations."""
    maximum = 0.0
    load_semantic_config(package)
    input_path = package / "capture_inputs.tsv"
    input_bytes = input_path.read_bytes()
    if (
        hashlib.sha256(input_bytes).hexdigest()
        != load_recipe_manifest(package).artifacts.input_sha256
    ):
        raise ValueError(
            "pinned Sophus input TSV hash differs from the recorded corpus"
        )
    input_rows = [
        line.split()
        for line in input_bytes.decode("utf-8").splitlines()
        if line and not line.startswith("#")
    ]

    def precision(values: Sequence[str | float]) -> bytes:
        numbers = tuple(float(value) for value in values)
        code = "f" if witnesses.scalar_profile == "float32" else "d"
        return struct.pack(f"={len(numbers)}{code}", *numbers)

    corpus = {
        (row[0], row[1]): precision(row[2:]) for row in input_rows if row[0] != "probe"
    }
    probe = next(precision(row[1:]) for row in input_rows if row[0] == "probe")
    if precision(witnesses.probe_xyz) != probe or any(
        precision(row[1]) != corpus[("quaternion", row[0])]
        for row in witnesses.quaternion_construction
    ):
        raise ValueError("selected witness inputs differ from the pinned Sophus corpus")

    def compare(actual: object, expected: object, label: str) -> None:
        nonlocal maximum
        with np.errstate(over="ignore", invalid="ignore"):
            difference = np.asarray(actual, dtype=float) - np.asarray(
                expected, dtype=float
            )
        error = finite_max_abs(difference.flat, label)
        maximum = max(maximum, error)
        if error > tolerance:
            raise AssertionError(f"{label} exceeds {tolerance}: {error}")

    def rotation(row: tuple, expected: Rotation) -> None:
        case_id, xyzw, basis, probe = row
        native = np.asarray(xyzw, dtype=float)
        reference = expected.as_quat()
        compare(
            native,
            reference if np.dot(native, reference) >= 0.0 else -reference,
            f"{case_id} coefficients",
        )
        compare(basis, expected.apply(np.eye(3)), f"{case_id} basis actions")
        if probe is not None:
            compare(probe, expected.apply(witnesses.probe_xyz), f"{case_id} probe")

    constructors = {row[0]: row for row in witnesses.quaternion_construction}
    for case_id, wxyz, norm, xyzw, basis, probe in constructors.values():
        values = np.asarray(wxyz, dtype=float)
        compare([norm], [np.linalg.norm(values)], f"{case_id} input norm")
        expected = Rotation.from_quat(values[[1, 2, 3, 0]])
        rotation((case_id, xyzw, basis, probe), expected)
    if (
        np.dot(
            constructors["mixed_unequal"][3], constructors["mixed_unequal_antipode"][3]
        )
        >= 0
    ):
        raise AssertionError("native antipodes must retain opposite coefficient signs")

    a = Rotation.from_quat(constructors["mixed_unequal"][3])
    b = Rotation.from_quat(constructors["nonunit"][3])
    expected_products = (
        ("mixed_unequal_times_nonunit", a * b),
        ("nonunit_times_mixed_unequal", b * a),
    )
    for row, (case_id, expected) in zip(witnesses.composition, expected_products):
        if row[0] != case_id:
            raise ValueError("native composition order differs from the input contract")
        rotation(row, expected)
    return maximum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--capture", type=Path, help="full native capture to reduce and verify"
    )
    group.add_argument(
        "--witnesses", type=Path, help="retained selected native witnesses"
    )
    parser.add_argument("--output", type=Path, help="optional comparison report")
    args = parser.parse_args()
    package = args.package_dir.resolve(strict=True)
    config = load_semantic_config(package)
    if args.capture is not None:
        witnesses = reduce_capture(args.capture.read_bytes(), config, package)
    else:
        requested = parse_witness(args.witnesses.read_bytes())
        witnesses = load_shared_witness(requested.scalar_profile, package)
        if json.dumps(requested.model_dump(mode="json"), sort_keys=True) != json.dumps(
            witnesses.model_dump(mode="json"), sort_keys=True
        ):
            raise ValueError(
                "requested witness differs from the recorded shared payload"
            )
    profile = next(
        item
        for item in config.scalar_profiles
        if item.precision == witnesses.scalar_profile
    )
    maximum = verify(witnesses, profile.scipy_tolerance, package)
    report = json.dumps(
        {
            "result": "pass",
            "native_target": profile.witness_target,
            "reference": f"SciPy {scipy.__version__}",
            "quatops_executed": False,
            "absolute_tolerance": profile.scipy_tolerance,
            "max_abs_error": maximum,
        }
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8", newline="\n")
    print(report)


if __name__ == "__main__":
    main()
