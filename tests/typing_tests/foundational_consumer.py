"""Static return-type checks for the foundational scalar API additions."""

from typing import assert_type

from quatops import dataclass, namedtuple, scalar
from quatops.hamilton import dataclass as hamilton_dataclass
from quatops.hamilton import namedtuple as hamilton_namedtuple
from quatops.hamilton import scalar as hamilton_scalar
from quatops.jpl import dataclass as jpl_dataclass
from quatops.jpl import namedtuple as jpl_namedtuple
from quatops.jpl import scalar as jpl_scalar

q = (1.0, 0.0, 0.0, 0.0)
matrix = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
vector = (1.0, 0.0, 0.0)

assert_type(scalar.tan(q), tuple[float, float, float, float])
assert_type(scalar.dot(q, q), float)
assert_type(scalar.norm_squared(q), float)
assert_type(scalar.left_divide(q, q), tuple[float, float, float, float])
assert_type(scalar.right_divide(q, q), tuple[float, float, float, float])
assert_type(scalar.rotation_distance(q, q), float)
assert_type(scalar.rotations_equivalent(q, q), bool)
assert_type(scalar.canonicalize(q), tuple[float, float, float, float])
assert_type(scalar.from_rotation_matrix(matrix), tuple[float, float, float, float])
assert_type(scalar.from_two_vectors(vector, vector), tuple[float, float, float, float])

assert_type(hamilton_scalar.tan(q), tuple[float, float, float, float])
assert_type(hamilton_scalar.dot(q, q), float)
assert_type(hamilton_scalar.rotation_distance(q, q), float)
assert_type(hamilton_scalar.rotations_equivalent(q, q), bool)
assert_type(jpl_scalar.tan(q), tuple[float, float, float, float])
assert_type(jpl_scalar.dot(q, q), float)
assert_type(jpl_scalar.rotation_distance(q, q), float)
assert_type(jpl_scalar.rotations_equivalent(q, q), bool)

assert_type(namedtuple.tan(q), namedtuple.Quaternion)
assert_type(namedtuple.left_divide(q, q), namedtuple.Quaternion)
assert_type(namedtuple.canonicalize(q), namedtuple.Quaternion)
assert_type(namedtuple.from_rotation_matrix(matrix), namedtuple.Quaternion)
assert_type(namedtuple.from_two_vectors(vector, vector), namedtuple.Quaternion)
assert_type(namedtuple.dot(q, q), float)
assert_type(namedtuple.rotation_distance(q, q), float)
assert_type(namedtuple.rotations_equivalent(q, q), bool)

assert_type(hamilton_namedtuple.tan(q), hamilton_namedtuple.Quaternion)
assert_type(hamilton_namedtuple.left_divide(q, q), hamilton_namedtuple.Quaternion)
assert_type(hamilton_namedtuple.canonicalize(q), hamilton_namedtuple.Quaternion)
assert_type(hamilton_namedtuple.from_rotation_matrix(matrix), hamilton_namedtuple.Quaternion)
assert_type(hamilton_namedtuple.from_two_vectors(vector, vector), hamilton_namedtuple.Quaternion)
assert_type(hamilton_namedtuple.dot(q, q), float)
assert_type(hamilton_namedtuple.rotation_distance(q, q), float)
assert_type(hamilton_namedtuple.rotations_equivalent(q, q), bool)

assert_type(jpl_namedtuple.tan(q), jpl_namedtuple.Quaternion)
assert_type(jpl_namedtuple.left_divide(q, q), jpl_namedtuple.Quaternion)
assert_type(jpl_namedtuple.canonicalize(q), jpl_namedtuple.Quaternion)
assert_type(jpl_namedtuple.from_rotation_matrix(matrix), jpl_namedtuple.Quaternion)
assert_type(jpl_namedtuple.from_two_vectors(vector, vector), jpl_namedtuple.Quaternion)
assert_type(jpl_namedtuple.dot(q, q), float)
assert_type(jpl_namedtuple.rotation_distance(q, q), float)
assert_type(jpl_namedtuple.rotations_equivalent(q, q), bool)

assert_type(dataclass.tan(q), dataclass.Quaternion)
assert_type(dataclass.right_divide(q, q), dataclass.Quaternion)
assert_type(dataclass.canonicalize(q), dataclass.Quaternion)
assert_type(dataclass.from_rotation_matrix(matrix), dataclass.Quaternion)
assert_type(dataclass.from_two_vectors(vector, vector), dataclass.Quaternion)
assert_type(dataclass.norm_squared(q), float)
assert_type(dataclass.rotation_distance(q, q), float)
assert_type(dataclass.rotations_equivalent(q, q), bool)

assert_type(hamilton_dataclass.tan(q), hamilton_dataclass.Quaternion)
assert_type(hamilton_dataclass.right_divide(q, q), hamilton_dataclass.Quaternion)
assert_type(hamilton_dataclass.canonicalize(q), hamilton_dataclass.Quaternion)
assert_type(hamilton_dataclass.from_rotation_matrix(matrix), hamilton_dataclass.Quaternion)
assert_type(hamilton_dataclass.from_two_vectors(vector, vector), hamilton_dataclass.Quaternion)
assert_type(hamilton_dataclass.norm_squared(q), float)
assert_type(hamilton_dataclass.rotation_distance(q, q), float)
assert_type(hamilton_dataclass.rotations_equivalent(q, q), bool)

assert_type(jpl_dataclass.tan(q), jpl_dataclass.Quaternion)
assert_type(jpl_dataclass.right_divide(q, q), jpl_dataclass.Quaternion)
assert_type(jpl_dataclass.canonicalize(q), jpl_dataclass.Quaternion)
assert_type(jpl_dataclass.from_rotation_matrix(matrix), jpl_dataclass.Quaternion)
assert_type(jpl_dataclass.from_two_vectors(vector, vector), jpl_dataclass.Quaternion)
assert_type(jpl_dataclass.norm_squared(q), float)
assert_type(jpl_dataclass.rotation_distance(q, q), float)
assert_type(jpl_dataclass.rotations_equivalent(q, q), bool)
