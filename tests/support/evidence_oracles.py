"""Independent SciPy rotation and native numeric-envelope test helpers."""

import numpy as np
from scipy.spatial.transform import Rotation


def matrix_from_wxyz(quaternion):
    """Interpret Hamilton WXYZ using SciPy's XYZW interface."""
    q = np.asarray(quaternion, dtype=float)
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def native_values(item):
    """Unwrap stored native numeric envelopes without recomputing their values."""
    if isinstance(item, dict) and "values" in item and "dtype" in item:
        return item["values"]
    if isinstance(item, dict):
        return {key: native_values(value) for key, value in item.items()}
    if isinstance(item, list):
        return [native_values(value) for value in item]
    return item
