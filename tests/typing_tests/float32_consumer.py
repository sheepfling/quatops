"""Static acceptance of real scalar inputs beyond Python's built-in float."""

from typing import SupportsFloat, assert_type

import numpy as np

from quatops import dataclass, namedtuple, scalar
from quatops.hamilton import dataclass as hamilton_dataclass
from quatops.hamilton import namedtuple as hamilton_namedtuple
from quatops.hamilton import scalar as hamilton_scalar
from quatops.jpl import dataclass as jpl_dataclass
from quatops.jpl import namedtuple as jpl_namedtuple
from quatops.jpl import scalar as jpl_scalar
from quatops.python import hamilton as python_hamilton
from quatops.python import jpl as python_jpl
from quatops.types import (
    hamilton_dataclass as types_hamilton_dataclass,
)
from quatops.types import (
    hamilton_namedtuple as types_hamilton_namedtuple,
)
from quatops.types import (
    jpl_dataclass as types_jpl_dataclass,
)
from quatops.types import (
    jpl_namedtuple as types_jpl_namedtuple,
)

q = (np.float32(1.0), np.float32(0.1), np.float32(-0.2), np.float32(0.3))
mixed_q = (np.float32(1.0), 0.1, np.float64(-0.2), 0.3)
real_value: SupportsFloat = np.float32(0.5)

assert_type(scalar.normalize(q), tuple[float, float, float, float])
assert_type(scalar.normalize(mixed_q), tuple[float, float, float, float])
assert_type(scalar.scale(q, real_value), tuple[float, float, float, float])
assert_type(hamilton_scalar.multiply(q, q), tuple[float, float, float, float])
assert_type(hamilton_scalar.scale(q, real_value), tuple[float, float, float, float])
assert_type(jpl_scalar.from_wxyz(q), tuple[float, float, float, float])
assert_type(jpl_scalar.scale(q, real_value), tuple[float, float, float, float])
assert_type(python_hamilton.scale(q, real_value), tuple[float, float, float, float])
assert_type(python_jpl.from_rotation_vector(q[1:]), tuple[float, float, float, float])
assert_type(python_jpl.scale(q, real_value), tuple[float, float, float, float])
assert_type(namedtuple.from_wxyz(q), namedtuple.Quaternion)
assert_type(namedtuple.scale(q, real_value), namedtuple.Quaternion)
assert_type(hamilton_namedtuple.multiply(q, q), hamilton_namedtuple.Quaternion)
assert_type(hamilton_namedtuple.scale(q, real_value), hamilton_namedtuple.Quaternion)
assert_type(jpl_namedtuple.from_rotation_vector(q[1:]), jpl_namedtuple.Quaternion)
assert_type(jpl_namedtuple.scale(q, real_value), jpl_namedtuple.Quaternion)
assert_type(types_hamilton_namedtuple.from_wxyz(q), types_hamilton_namedtuple.Quaternion)
assert_type(types_hamilton_namedtuple.scale(q, real_value), types_hamilton_namedtuple.Quaternion)
assert_type(types_jpl_namedtuple.multiply(q, q), types_jpl_namedtuple.Quaternion)
assert_type(types_jpl_namedtuple.scale(q, real_value), types_jpl_namedtuple.Quaternion)
assert_type(dataclass.from_wxyz(q), dataclass.Quaternion)
assert_type(dataclass.scale(q, real_value), dataclass.Quaternion)
assert_type(hamilton_dataclass.multiply(q, q), hamilton_dataclass.Quaternion)
assert_type(hamilton_dataclass.scale(q, real_value), hamilton_dataclass.Quaternion)
assert_type(jpl_dataclass.from_rotation_vector(q[1:]), jpl_dataclass.Quaternion)
assert_type(jpl_dataclass.scale(q, real_value), jpl_dataclass.Quaternion)
assert_type(types_hamilton_dataclass.from_wxyz(q), types_hamilton_dataclass.Quaternion)
assert_type(types_hamilton_dataclass.scale(q, real_value), types_hamilton_dataclass.Quaternion)
assert_type(types_jpl_dataclass.multiply(q, q), types_jpl_dataclass.Quaternion)
assert_type(types_jpl_dataclass.scale(q, real_value), types_jpl_dataclass.Quaternion)
