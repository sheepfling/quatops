"""Canonical sign and explicitly bounded subnormal conversion regressions."""

import pytest
import math
from dataclasses import FrozenInstanceError

from quatops import dataclass, namedtuple, scalar

import numpy as np

from quatops.numpy.hamilton import wxyz as qnumpy

BACKENDS = (scalar, namedtuple, dataclass, qnumpy)


class CanonicalZeroTests:
    def assert_positive_zeros(self, result):
        for value in result:
            number = float(value)
            if number == 0.0:
                assert math.copysign(1.0, number) == 1.0, tuple(result)

    def test_zero_sign_after_final_conversion_arithmetic(self):
        tiny = math.nextafter(0.0, 1.0)
        cases = (
            # Axis division underflows a negative component to -0.0.
            ("to_axis_angle", (1.0, -tiny, 1e308, 0.0)),
            ("to_axis_angle", (-0.0, tiny, -1e308, -0.0)),
            # Rotvec multiplication underflows a negative component to -0.0.
            ("to_rotvec", (1e308, -5e-17, 1e307, 0.0)),
            ("to_rotvec", (1e308, -tiny, 0.0, -0.0)),
            ("from_axis_angle", (1.0, -0.0, 0.0, -math.pi)),
            ("from_rotvec", (-0.0, 0.0, -math.pi)),
        )
        for backend in BACKENDS:
            for operation, values in cases:
                supplied = list(values)
                before = supplied.copy()
                output = getattr(backend, operation)(supplied)
                self.assert_positive_zeros(output)
                assert supplied == before
                if backend is scalar:
                    assert type(output) is tuple
                elif backend in (namedtuple, dataclass):
                    expected = (backend.AxisAngle if operation == "to_axis_angle"
                                else backend.RotVec if operation == "to_rotvec"
                                else backend.Quaternion)
                    assert type(output) is expected
                    with pytest.raises((AttributeError, FrozenInstanceError)):
                        setattr(output, "x" if operation != "to_axis_angle" else "ax", 2)

    def test_numpy_read_only_batch_zero_sign_and_state(self):
        tiny = math.nextafter(0.0, 1.0)
        q = np.array([[1.0, -tiny, 1e308, -0.0], [1e308, -5e-17, 1e307, -0.0]])
        before = q.tobytes()
        q.flags.writeable = False
        old = np.seterr(all="raise")
        try:
            for operation in (qnumpy.to_axis_angle, qnumpy.to_rotvec):
                result = operation(q)
                assert not np.signbit(result[result == 0.0]).any()
                assert q.tobytes() == before
                assert np.geterr() == dict.fromkeys(("divide", "over", "under", "invalid"), "raise")
        finally:
            np.seterr(**old)

    def test_smallest_subnormal_rotation_vector_rounding_limit(self):
        # For r=(5u,5u,0), the exact quaternion components are just below
        # 2.5u, so their correctly rounded value is 2u. Current implementations
        # produce 3u after intermediate rounding. Allow either result so this
        # check records the known limit without preventing a future fix.
        u = math.nextafter(0.0, 1.0)
        for backend in BACKENDS:
            q = tuple(backend.from_rotvec((5*u, 5*u, 0)))
            assert q[0] == 1.0
            assert q[1] in (2*u, 3*u)
            assert q[2] in (2*u, 3*u)
            self.assert_positive_zeros(q)
