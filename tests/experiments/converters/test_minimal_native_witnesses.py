"""The reduced witnesses distinguish signed-axis frame-map mistakes."""

from itertools import permutations, product

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.evidence_json import read_evidence_document
from scripts.evidence_schema.validation import validate_evidence_packages
from tests.support.paths import REPOSITORY_ROOT


@pytest.mark.parametrize("target", ["Eigen", "MuJoCo", "OpenCV", "SpiceyPy / CSPICE"])
def test_mixed_native_actions_discriminate_signed_axis_maps(target):
    """Two mixed actions distinguish the assigned identity probe map up to ±B."""
    inputs = read_evidence_document(
        REPOSITORY_ROOT / "experiments/interoperability/v1/canonical_inputs.json"
    )
    cases = {row["case_id"]: row["canonical_wxyz"] for row in inputs["orientations"]}
    capture = next(
        c for c in validate_evidence_packages(REPOSITORY_ROOT) if c.target == target
    )
    for precision in {
        w.precision
        for w in capture.serialization.witnesses
        if w.kind in ("orientation", "rotation_vector")
    }:
        witnesses = {
            w.case_id.split("@")[0]: w
            for w in capture.serialization.witnesses
            if w.kind in ("orientation", "rotation_vector") and w.precision == precision
        }
        matches = []
        for axes in permutations(range(3)):
            for signs in product((-1, 1), repeat=3):
                basis = np.diag(signs) @ np.eye(3)[list(axes)]
                axial = round(np.linalg.det(basis)) * basis
                for case_id in ("mixed_a", "mixed_b"):
                    q = np.asarray(cases[case_id])
                    result = Rotation.from_quat(np.r_[axial @ q[1:], q[0]]).as_matrix()
                    native = np.asarray(witnesses[case_id].basis_images_xyz).T
                    if not np.allclose(
                        result,
                        native,
                        rtol=0,
                        atol=4e-6 if precision == "float32" else 4e-12,
                    ):
                        break
                else:
                    matches.append(axial)
        assert len(matches) == 2
        for match in matches:
            np.testing.assert_array_equal(match, np.eye(3))
