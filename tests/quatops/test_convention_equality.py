"""Typed quaternion equality respects convention and representation identity."""


from quatops import dataclass, namedtuple
from quatops.jpl import dataclass as jpl_dataclass
from quatops.jpl import namedtuple as jpl_namedtuple


class ConventionEqualityTests:
    def test_namedtuple_quaternions_compare_only_with_the_same_class(self):
        hamilton = namedtuple.Quaternion(1, 2, 3, 4)
        same_hamilton = namedtuple.Quaternion(1, 2, 3, 4)
        jpl = jpl_namedtuple.Quaternion(1, 2, 3, 4)

        assert hamilton == same_hamilton
        assert hash(hamilton) == hash(same_hamilton)
        assert hamilton != jpl
        assert jpl != hamilton
        assert hamilton != tuple(hamilton)
        assert tuple(hamilton) != hamilton

    def test_dataclass_quaternions_compare_only_with_the_same_class(self):
        hamilton = dataclass.Quaternion(1, 2, 3, 4)
        same_hamilton = dataclass.Quaternion(1, 2, 3, 4)
        jpl = jpl_dataclass.Quaternion(1, 2, 3, 4)

        assert hamilton == same_hamilton
        assert hamilton != jpl
        assert jpl != hamilton
        assert hamilton != tuple(hamilton)
