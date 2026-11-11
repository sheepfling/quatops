"""Independent SciPy rotation and native numeric-envelope test helpers."""

import numpy as np
from scipy.spatial.transform import Rotation


def ecef_origin(longitude_degrees, latitude_degrees, height_m):
    """Calculate WGS84 geodetic coordinates independently of capture code."""
    longitude, latitude = np.deg2rad([longitude_degrees, latitude_degrees])
    semi_major = 6_378_137.0
    flattening = 1.0 / 298.257223563
    eccentricity_squared = flattening * (2.0 - flattening)
    prime_vertical = semi_major / np.sqrt(
        1.0 - eccentricity_squared * np.sin(latitude) ** 2
    )
    return np.array(
        [
            (prime_vertical + height_m) * np.cos(latitude) * np.cos(longitude),
            (prime_vertical + height_m) * np.cos(latitude) * np.sin(longitude),
            (prime_vertical * (1.0 - eccentricity_squared) + height_m)
            * np.sin(latitude),
        ]
    )


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
