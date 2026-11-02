"""Installed-wheel NumPy layout type checks with the optional NumPy extra."""

from typing import Any, assert_type

from numpy.typing import NDArray

from quatops.hamilton import numpy_wxyz as hw
from quatops.hamilton import numpy_xyzw as hx
from quatops.jpl import numpy_wxyz as jw
from quatops.jpl import numpy_xyzw as jx

q = (1.0, 0.0, 0.0, 0.0)
assert_type(hw.rotate(q, (1.0, 0.0, 0.0)), NDArray[Any])
assert_type(hw.add(q, q), NDArray[Any])
assert_type(hw.subtract(q, q), NDArray[Any])
assert_type(hw.negate(q), NDArray[Any])
assert_type(hw.scale(q, 2), NDArray[Any])
assert_type(hw.divide(lhs=q, rhs=q), NDArray[Any])
assert_type(hx.to_axis_angle((0.0, 0.0, 0.0, 1.0)), NDArray[Any])
assert_type(jw.from_hamilton(q), NDArray[Any])
assert_type(jx.to_hamilton((0.0, 0.0, 0.0, 1.0)), NDArray[Any])

from quatops.hamilton import numpy_wxyz_components_first as hw_first
from quatops.hamilton import numpy_xyzw_components_first as hx_first
from quatops.jpl import numpy_wxyz_components_first as jw_first
from quatops.jpl import numpy_xyzw_components_first as jx_first
from quatops.numpy.hamilton import wxyz as backend_hw
from quatops.numpy.hamilton import xyzw as backend_hx
from quatops.numpy.hamilton.components_first import wxyz as backend_hw_first
from quatops.numpy.hamilton.components_first import xyzw as backend_hx_first
from quatops.numpy.jpl import wxyz as backend_jw
from quatops.numpy.jpl import xyzw as backend_jx
from quatops.numpy.jpl.components_first import wxyz as backend_jw_first
from quatops.numpy.jpl.components_first import xyzw as backend_jx_first

assert_type(backend_hw.rotate(q, (1.0, 0.0, 0.0)), NDArray[Any])
assert_type(backend_hx.to_axis_angle((0.0, 0.0, 0.0, 1.0)), NDArray[Any])
assert_type(backend_jw.from_hamilton(q), NDArray[Any])
assert_type(backend_jx.to_hamilton((0.0, 0.0, 0.0, 1.0)), NDArray[Any])
assert_type(backend_hw_first.rotate(q, (1.0, 0.0, 0.0)), NDArray[Any])
assert_type(backend_hx_first.to_axis_angle((0.0, 0.0, 0.0, 1.0)), NDArray[Any])
assert_type(backend_jw_first.from_hamilton(q), NDArray[Any])
assert_type(backend_jx_first.to_hamilton((0.0, 0.0, 0.0, 1.0)), NDArray[Any])
assert_type(backend_hw.dot(q, q), NDArray[Any])
assert_type(backend_hw.norm_squared(q), NDArray[Any])
assert_type(backend_hw.left_divide(q, q), NDArray[Any])
assert_type(backend_hw.right_divide(q, q), NDArray[Any])
assert_type(backend_hw.rotation_distance(q, q), NDArray[Any])
assert_type(backend_hw.rotations_equivalent(q, q), NDArray[Any])
assert_type(backend_hw.canonicalize(q), NDArray[Any])
assert_type(backend_hw.from_rotation_matrix(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))), NDArray[Any])
assert_type(backend_hw.from_two_vectors((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), NDArray[Any])
assert_type(backend_hw.tan(q), NDArray[Any])
assert_type(backend_hw_first.from_rotation_matrix(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))), NDArray[Any])
assert_type(hw_first.rotate(q, (1.0, 0.0, 0.0)), NDArray[Any])
assert_type(hx_first.to_axis_angle((0.0, 0.0, 0.0, 1.0)), NDArray[Any])
assert_type(jw_first.from_hamilton(q), NDArray[Any])
assert_type(jx_first.to_hamilton((0.0, 0.0, 0.0, 1.0)), NDArray[Any])
