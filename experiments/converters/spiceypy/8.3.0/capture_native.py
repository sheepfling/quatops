"""Capture selected CSPICE quaternion-boundary witnesses; imports no QuatOps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import spiceypy as spice  # pyright: ignore[reportMissingModuleSource]

from scripts.direct_quaternion_capture import DirectCaptureResult, run_direct_capture
from scripts.evidence_schema.converter_captures import (
    SPICEYPY_CONSTRUCTOR_CASE_ID,
    SpiceyPyCapture,
    SpiceyPyConstructedOrientation,
    SpiceyPyInputs,
    SpiceyPyObservations,
    SpiceyPyOrientation,
    TargetIdentity,
)
from scripts.evidence_schema.shared_inputs import (
    SharedCanonicalInputCapture,
    SharedQuaternionInput,
)
from scripts.evidence_schema.spiceypy_capture import (
    validate_spiceypy_capture_matches_inputs,
)


def capture_quaternion(
    case_id: str,
    q: np.ndarray,
    probe: np.ndarray,
    *,
    lhs: np.ndarray | None = None,
    rhs: np.ndarray | None = None,
) -> SpiceyPyOrientation:
    matrix = np.asarray(spice.q2m(q), dtype=np.float64)
    return SpiceyPyOrientation(
        case_id=case_id,
        native_input_wxyz=q.tolist(),
        native_mxv_basis_xyz=[
            np.asarray(spice.mxv(matrix, vector), dtype=np.float64).tolist()
            for vector in np.eye(3)
        ],
        native_mxv_probe_xyz=(
            np.asarray(spice.mxv(matrix, probe), dtype=np.float64).tolist()
            if case_id == "mixed_a"
            else None
        ),
        native_lhs_wxyz=lhs.tolist() if lhs is not None else None,
        native_rhs_wxyz=rhs.tolist() if rhs is not None else None,
        product_order="lhs * rhs; rhs acts first" if lhs is not None else None,
    )


def capture(
    spec: SpiceyPyInputs,
    shared: SharedCanonicalInputCapture,
    selected: list[SharedQuaternionInput],
) -> DirectCaptureResult:
    probe = np.asarray(shared.probe_vector_xyz, dtype=np.float64)
    canonical_to_native = [
        capture_quaternion(
            row.case_id, np.asarray(row.canonical_wxyz, dtype=np.float64), probe
        )
        for row in selected
    ]
    ctor = spec.native_constructed_orientation
    axis = np.asarray(ctor.axis_xyz, dtype=np.float64)
    matrix = np.asarray(spice.axisar(axis, float(ctor.angle_radians)), dtype=np.float64)
    qnative = np.asarray(spice.m2q(matrix), dtype=np.float64)
    q2m = np.asarray(spice.q2m(qnative), dtype=np.float64)
    native_created = SpiceyPyConstructedOrientation(
        case_id=SPICEYPY_CONSTRUCTOR_CASE_ID,
        native_constructor="axisar(axis, angle) -> m2q(matrix)",
        native_m2q_wxyz=qnative.tolist(),
        native_mxv_basis_xyz=[
            np.asarray(spice.mxv(q2m, vector), dtype=np.float64).tolist()
            for vector in np.eye(3)
        ],
    )
    pair = {
        row.case_id: np.asarray(row.canonical_wxyz, dtype=np.float64)
        for row in selected
    }
    a, b = pair["mixed_a"], pair["mixed_b"]
    composition = [
        capture_quaternion(
            case_id,
            np.asarray(spice.qxq(lhs, rhs), dtype=np.float64),
            probe,
            lhs=lhs,
            rhs=rhs,
        )
        for case_id, lhs, rhs in (
            ("mixed_a_times_mixed_b", a, b),
            ("mixed_b_times_mixed_a", b, a),
        )
    ]
    witnesses = SpiceyPyCapture(
        schema_version="1.0",
        evidence_kind="native_runtime_selected_converter_witnesses",
        target=TargetIdentity(
            name="SpiceyPy / CSPICE",
            wrapper_version=spice.__version__,
            toolkit_version=spice.tkvrsn("TOOLKIT"),
        ),
        observations=SpiceyPyObservations(
            canonical_to_native=canonical_to_native,
            native_created=[native_created],
            composition=composition,
        ),
    )
    validate_spiceypy_capture_matches_inputs(witnesses, spec, shared)
    library = Path(spice.libspice._name)
    return DirectCaptureResult(witnesses, library)


if __name__ == "__main__":
    run_direct_capture(Path(__file__), SpiceyPyInputs, capture)
