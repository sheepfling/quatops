"""Installed-wheel typing_tests smoke test; checked by mypy, Pyright, and ty."""

from typing_extensions import assert_type

from quatops import dataclass, namedtuple, scalar

q = (1.0, 0.0, 0.0, 0.0)
assert_type(scalar.norm(q), float)
assert_type(scalar.add(q, q), tuple[float, float, float, float])
assert_type(scalar.subtract(q, q), tuple[float, float, float, float])
assert_type(scalar.negate(q), tuple[float, float, float, float])
assert_type(scalar.scale(q, 2), tuple[float, float, float, float])
assert_type(scalar.divide(lhs=q, rhs=q), tuple[float, float, float, float])
assert_type(scalar.rotate(q, (1.0, 0.0, 0.0)), tuple[float, float, float])
assert_type(namedtuple.inverse(namedtuple.from_wxyz(q)), namedtuple.Quaternion)
assert_type(dataclass.conjugate(dataclass.from_wxyz(q)), dataclass.Quaternion)
assert_type(scalar.to_axis_angle(q), tuple[float, float, float, float])
assert_type(scalar.to_rotvec(q), tuple[float, float, float])
assert_type(namedtuple.to_axis_angle(q), namedtuple.AxisAngle)
assert_type(namedtuple.to_rotvec(q), namedtuple.RotVec)
assert_type(namedtuple.from_axis_angle(namedtuple.AxisAngle(1, 0, 0, 0)), namedtuple.Quaternion)
assert_type(dataclass.to_axis_angle(q), dataclass.AxisAngle)
assert_type(dataclass.to_rotvec(q), dataclass.RotVec)
assert_type(dataclass.from_rotvec(dataclass.RotVec(0, 0, 0)), dataclass.Quaternion)

from quatops.hamilton import dataclass as h_dataclass
from quatops.hamilton import namedtuple as h_namedtuple
from quatops.hamilton import scalar as h_scalar
from quatops.jpl import dataclass as j_dataclass
from quatops.jpl import namedtuple as j_namedtuple
from quatops.jpl import scalar as j_scalar

assert_type(h_scalar.to_wxyz(q), tuple[float, float, float, float])
assert_type(h_namedtuple.from_wxyz(q), h_namedtuple.Quaternion)
assert_type(h_dataclass.to_rotvec(q), h_dataclass.RotVec)
assert_type(j_scalar.from_hamilton(q), tuple[float, float, float, float])
assert_type(j_namedtuple.from_hamilton(q), j_namedtuple.Quaternion)
assert_type(j_dataclass.to_rotvec((0, 0, 0, 1)), j_dataclass.RotVec)

from quatops import types as quat_types
from quatops.python import hamilton as python_hamilton
from quatops.python import jpl as python_jpl

assert_type(python_hamilton.multiply(lhs=q, rhs=q), tuple[float, float, float, float])
assert_type(python_jpl.divide(q, q), tuple[float, float, float, float])
assert_type(python_hamilton.to_rotvec(q), tuple[float, float, float])
assert_type(python_jpl.from_hamilton(q), tuple[float, float, float, float])
assert_type(quat_types.HamiltonQuaternion(*q), h_dataclass.Quaternion)
assert_type(quat_types.JPLQuaternion(0, 0, 0, 1), j_dataclass.Quaternion)
