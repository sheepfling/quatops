"""Type surface for the untyped MuJoCo bindings used by the capture."""

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias

_Float64Array: TypeAlias = NDArray[np.float64]

__file__: str

def mj_versionString() -> str: ...
def mju_axisAngle2Quat(
    quaternion: _Float64Array, axis: _Float64Array, angle: float
) -> None: ...
def mju_mulQuat(
    product: _Float64Array, lhs: _Float64Array, rhs: _Float64Array
) -> None: ...
def mju_rotVecQuat(
    result: _Float64Array, vector: _Float64Array, quaternion: _Float64Array
) -> None: ...
