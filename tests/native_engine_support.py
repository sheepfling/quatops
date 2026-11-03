"""Shared QuatOps helpers for comparisons with native engine captures."""

import json
from pathlib import Path

import numpy as np

from tests.oracle_support import (
    COEFFICIENT_SIGN,
    from_wxyz,
    orient_vectors,
    to_wxyz,
    vectors_to_trailing,
)


def load_versioned_engine_witness(repository_root, engine, version):
    """Load a version contract and resolve its immutable witness reference."""
    base = Path(repository_root) / "tests" / "oracles" / engine / version
    contract = json.loads((base / "contract.json").read_text(encoding="utf-8"))
    witness_path = (base / contract["provenance"]["witness_file"]).resolve()
    witness = json.loads(witness_path.read_text(encoding="utf-8"))
    target = witness.get("target", {})
    if "version" not in target:
        target = contract["engine"]
    fixture = {
        **witness,
        "target": target,
        "contract": witness.get("contract", contract["conversion"]),
        "provenance": witness.get("provenance", contract["provenance"]),
    }
    return contract, fixture, witness_path


def from_engine_xyzw(module, xyzw, is_jpl, is_numpy, component_first):
    """Load a native Hamilton XYZW value into a public QuatOps backend."""
    wxyz = np.asarray(xyzw, dtype=float)[[3, 0, 1, 2]]
    if is_jpl:
        wxyz = wxyz * COEFFICIENT_SIGN
    return from_wxyz(module, wxyz, is_numpy, component_first)


def from_euler(module, angles, sequence, degrees, component_first):
    values = orient_vectors(np.asarray(angles, dtype=float), component_first)
    return module.from_euler(values, sequence, degrees=degrees)


def to_hamilton_xyzw(module, value, is_jpl, is_numpy, component_first):
    wxyz = np.asarray(to_wxyz(module, value, is_numpy, component_first), dtype=float)
    wxyz = vectors_to_trailing(wxyz, component_first).reshape((-1, 4))[0]
    if is_jpl:
        wxyz = wxyz * COEFFICIENT_SIGN
    return wxyz[[1, 2, 3, 0]]


def rotate_basis(module, value):
    basis = np.eye(3, dtype=float)
    return np.stack(
        [
            np.asarray(module.rotate(value, tuple(axis)), dtype=float).reshape(3)
            for axis in basis
        ]
    )


def vector_xyz(value):
    if not isinstance(value, dict):
        return np.asarray(value, dtype=float)
    return np.asarray((value["x"], value["y"], value["z"]), dtype=float)


def quaternion_xyzw(value):
    if "xyzw" in value:
        return np.asarray(value["xyzw"], dtype=float)
    return np.asarray((value["x"], value["y"], value["z"], value["w"]), dtype=float)


def quaternion_coefficient_error(lhs_xyzw, rhs_xyzw):
    """Return coefficient distance up to the equivalent global quaternion sign."""
    lhs = np.asarray(lhs_xyzw, dtype=float)
    rhs = np.asarray(rhs_xyzw, dtype=float)
    return min(np.linalg.norm(lhs - rhs), np.linalg.norm(lhs + rhs))


def matrix_rows(values):
    return np.stack([vector_xyz(row) for row in values])


def basis_images(values):
    return np.stack([vector_xyz(axis) for axis in values])
