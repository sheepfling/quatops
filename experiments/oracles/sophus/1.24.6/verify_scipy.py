"""Compare selected native Sophus witnesses with independent SciPy rotations."""

from __future__ import annotations

import argparse
import json
import struct
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import scipy
from retain_witnesses import _PINNED_INPUT_SHA256, load_semantic_config, reduce_capture
from scipy.spatial.transform import Rotation

from scripts.evidence_schema.primitives import finite_max_abs
from scripts.evidence_schema.sophus import SophusDiscoveryWitnessSet
from scripts.evidence_schema.utilities import sha256_file


def verify(witnesses: SophusDiscoveryWitnessSet, tolerance: float) -> float:
    """Check retained native coefficient, basis-action and SO(3) observations."""
    maximum = 0.0
    input_path = Path(__file__).with_name("capture_inputs.tsv")
    if sha256_file(input_path) != _PINNED_INPUT_SHA256:
        raise ValueError(
            "pinned Sophus input TSV hash differs from the recorded corpus"
        )
    input_rows = [
        line.split()
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]

    def precision(values: Sequence[str | float]) -> tuple[float, ...]:
        numbers = tuple(float(value) for value in values)
        if "SO3f" in witnesses.target:
            return tuple(
                struct.unpack("=f", struct.pack("=f", value))[0] for value in numbers
            )
        return numbers

    corpus = {
        (row[0], row[1]): precision(row[2:]) for row in input_rows if row[0] != "probe"
    }
    probe = next(precision(row[1:]) for row in input_rows if row[0] == "probe")
    if (
        precision(witnesses.probe_xyz) != probe
        or any(
            precision(row[1]) != corpus[("quaternion", row[0])]
            for row in witnesses.quaternion_construction
        )
        or any(
            precision(row[1]) != corpus[("tangent", row[0])]
            for row in witnesses.exp_log
        )
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
    for case_id, wxyz, norm, xyzw, basis, log, probe in constructors.values():
        values = np.asarray(wxyz, dtype=float)
        compare([norm], [np.linalg.norm(values)], f"{case_id} input norm")
        expected = Rotation.from_quat(values[[1, 2, 3, 0]])
        rotation((case_id, xyzw, basis, probe), expected)
        compare(
            Rotation.from_rotvec(log).as_matrix(),
            expected.as_matrix(),
            f"{case_id} log action",
        )
        if case_id != "half_turn_axis":
            compare(log, expected.as_rotvec(), f"{case_id} log")
    if (
        np.dot(
            constructors["mixed_unequal"][3], constructors["mixed_unequal_antipode"][3]
        )
        >= 0
    ):
        raise AssertionError("native antipodes must retain opposite coefficient signs")

    tangents: dict[str, Rotation] = {}
    for case_id, tangent, xyzw, basis, log, probe in witnesses.exp_log:
        expected = Rotation.from_rotvec(tangent)
        rotation((case_id, xyzw, basis, probe), expected)
        compare(log, expected.as_rotvec(), f"{case_id} log")
        tangents[case_id] = expected

    a, b = tangents["mixed_a"], tangents["mixed_b"]
    expected_products = (
        ("mixed_a_times_mixed_b", a * b),
        ("mixed_b_times_mixed_a", b * a),
    )
    for row, (case_id, expected) in zip(witnesses.composition, expected_products):
        if row[0] != case_id:
            raise ValueError("native composition order differs from the input contract")
        rotation(row, expected)
    if witnesses.inverse[0] != "inverse_mixed_a":
        raise ValueError("native inverse case differs from the input contract")
    rotation(witnesses.inverse, a.inv())

    # The perturbation input is in the pinned corpus; it is not an extra golden.
    delta = Rotation.from_rotvec(corpus[("tangent", "perturbation")])
    expected_deltas = (
        ("delta_times_mixed_a", delta * a),
        ("mixed_a_times_delta", a * delta),
    )
    for row, (case_id, expected) in zip(witnesses.perturbation_side, expected_deltas):
        if row[0] != case_id:
            raise ValueError(
                "native perturbation order differs from the input contract"
            )
        rotation(row, expected)
    return maximum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--capture", type=Path, help="full native capture to reduce and verify"
    )
    group.add_argument(
        "--witnesses", type=Path, help="retained selected native witnesses"
    )
    parser.add_argument("--output", type=Path, help="optional comparison report")
    args = parser.parse_args()
    config = load_semantic_config()
    witnesses = (
        reduce_capture(args.capture.read_bytes(), config)
        if args.capture is not None
        else SophusDiscoveryWitnessSet.model_validate_json(
            args.witnesses.read_bytes(), strict=True
        )
    )
    profile = next(
        item
        for item in config.scalar_profiles
        if item.witness_target == witnesses.target
    )
    maximum = verify(witnesses, profile.scipy_tolerance)
    report = json.dumps(
        {
            "result": "pass",
            "native_target": witnesses.target,
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
