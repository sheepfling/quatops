"""Installed-wheel NumPy layout type checks with the optional NumPy extra."""

from typing import Any, assert_type

from numpy.typing import NDArray

from quatops.hamilton import numpy_wxyz as hw
from quatops.hamilton import numpy_xyzw as hx
from quatops.jpl import numpy_wxyz as jw
from quatops.jpl import numpy_xyzw as jx

q = (1.0, 0.0, 0.0, 0.0)
assert_type(hw.rotate(q, (1.0, 0.0, 0.0)), NDArray[Any])
assert_type(hx.to_axis_angle((0.0, 0.0, 0.0, 1.0)), NDArray[Any])
assert_type(jw.from_hamilton(q), NDArray[Any])
assert_type(jx.to_hamilton((0.0, 0.0, 0.0, 1.0)), NDArray[Any])

from quatops.numpy.hamilton import wxyz as backend_hw
from quatops.numpy.hamilton import xyzw as backend_hx
from quatops.numpy.jpl import wxyz as backend_jw
from quatops.numpy.jpl import xyzw as backend_jx

assert_type(backend_hw.rotate(q, (1.0, 0.0, 0.0)), NDArray[Any])
assert_type(backend_hx.to_axis_angle((0.0, 0.0, 0.0, 1.0)), NDArray[Any])
assert_type(backend_jw.from_hamilton(q), NDArray[Any])
assert_type(backend_jx.to_hamilton((0.0, 0.0, 0.0, 1.0)), NDArray[Any])
