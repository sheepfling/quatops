"""Compare Hamilton and JPL operation costs across public representations.

Run through ``.venv/bin/python experiments/run.py convention-benchmark``. Timings
include public validation and output construction and are workload-specific.
"""

import platform
import sys
import timeit
from typing import Callable

import numpy as np

from quatops.hamilton import dataclass as hd
from quatops.hamilton import namedtuple as hn
from quatops.hamilton import scalar as hs
from quatops.jpl import dataclass as jd
from quatops.jpl import namedtuple as jn
from quatops.jpl import scalar as js
from quatops.numpy.hamilton import xyzw as ha
from quatops.numpy.jpl import xyzw as ja


def compare(
    label: str,
    hamilton: Callable[..., object],
    hargs: tuple,
    jpl: Callable[..., object],
    jargs: tuple,
    count: int,
    scale: float,
    unit: str,
) -> None:
    """Print median paired convention timings and the JPL/Hamilton ratio."""
    ham_timer = timeit.Timer(
        "operation(*args)",
        globals={
            "operation": hamilton,
            "args": hargs,
        },
    )
    jpl_timer = timeit.Timer(
        "operation(*args)",
        globals={
            "operation": jpl,
            "args": jargs,
        },
    )
    ham_values: list[float] = []
    jpl_values: list[float] = []
    for index in range(5):
        first, second = (
            (ham_timer, jpl_timer) if index % 2 == 0 else (jpl_timer, ham_timer)
        )
        first_value = first.timeit(number=count)
        second_value = second.timeit(number=count)
        if index % 2 == 0:
            ham_values.append(first_value)
            jpl_values.append(second_value)
        else:
            jpl_values.append(first_value)
            ham_values.append(second_value)
    ham = sorted(ham_values)[2] * scale / count
    jpl_time = sorted(jpl_values)[2] * scale / count
    print(
        f"{label}: H={ham:.3f} {unit} J={jpl_time:.3f} {unit} J/H={jpl_time / ham:.2f}x"
    )


def main() -> None:
    """Run matched scalar and NumPy workloads for both conventions."""
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform(), "NumPy:", np.__version__)
    hq, hr, vector = (
        (0.91, 0.11, 0.31, -0.20),
        (0.97, -0.14, 0.08, 0.17),
        (1.0, -2.0, 3.0),
    )
    jq, jr = js.from_hamilton(hq), js.from_hamilton(hr)
    rotation_matrix = hs.to_rotation_matrix(hq)
    homogeneous_matrix = hs.to_homogeneous_matrix(hq, (1.0, 2.0, 3.0))
    vector_pairs = (
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)),
    )
    scalar_backends = (
        ("tuple", hs, js, tuple, tuple),
        (
            "namedtuple",
            hn,
            jn,
            lambda q: hn.Quaternion(*q),
            lambda q: jn.Quaternion(*q),
        ),
        ("dataclass", hd, jd, lambda q: hd.Quaternion(*q), lambda q: jd.Quaternion(*q)),
    )
    for name, ham, jpl, make_h, make_j in scalar_backends:
        qh, rh, qj, rj = make_h(hq), make_h(hr), make_j(jq), make_j(jr)
        workloads = (
            ("add", (qh, rh), (qj, rj)),
            ("subtract", (qh, rh), (qj, rj)),
            ("scale", (qh, 0.7), (qj, 0.7)),
            ("multiply", (qh, rh), (qj, rj)),
            ("divide", (qh, rh), (qj, rj)),
            ("norm", (qh,), (qj,)),
            ("normalize", (qh,), (qj,)),
            ("inverse", (qh,), (qj,)),
            ("rotate", (qh, vector), (qj, vector)),
            ("exp", (qh,), (qj,)),
            ("log", (qh,), (qj,)),
            ("sin", (qh,), (qj,)),
            ("sinh", (qh,), (qj,)),
            ("cosh", (qh,), (qj,)),
            ("tanh", (qh,), (qj,)),
            ("tan", (qh,), (qj,)),
            ("power", (qh, 0.5), (qj, 0.5)),
            ("asin", (qh,), (qj,)),
            ("acos", (qh,), (qj,)),
            ("atan", (qh,), (qj,)),
            ("asinh", (qh,), (qj,)),
            ("acosh", (qh,), (qj,)),
            ("atanh", (qh,), (qj,)),
            ("expm1", (qh,), (qj,)),
            ("log1p", (qh,), (qj,)),
            ("log2", (qh,), (qj,)),
            ("log10", (qh,), (qj,)),
            ("power_quaternion", (qh, rh), (qj, rj)),
            ("slerp", (qh, rh, 0.35), (qj, rj, 0.35)),
            ("nlerp", (qh, rh, 0.35), (qj, rj, 0.35)),
            ("rotation_distance", (qh, rh), (qj, rj)),
            ("quaternion_derivative", (qh, vector), (qj, vector)),
            ("to_rotation_matrix", (qh,), (qj,)),
            ("from_rotation_matrix", (rotation_matrix,), (rotation_matrix,)),
            ("from_rotation_vector", (vector,), (vector,)),
            (
                "from_two_vectors",
                ((1.0, -2.0, 3.0), (-3.0, 1.0, 2.0)),
                ((1.0, -2.0, 3.0), (-3.0, 1.0, 2.0)),
            ),
            ("integrate_angular_velocity", (qh, vector, 0.01), (qj, vector, 0.01)),
            ("from_euler", ((0.2, -0.3, 0.1),), ((0.2, -0.3, 0.1),)),
            ("to_euler", (qh,), (qj,)),
            ("from_mrp", ((0.1, -0.2, 0.05),), ((0.1, -0.2, 0.05),)),
            ("to_mrp", (qh,), (qj,)),
            ("swing_twist", (qh, (0.0, 0.0, 1.0)), (qj, (0.0, 0.0, 1.0))),
            ("to_homogeneous_matrix", (qh, (1.0, 2.0, 3.0)), (qj, (1.0, 2.0, 3.0))),
            ("from_homogeneous_matrix", (homogeneous_matrix,), (homogeneous_matrix,)),
            ("align_vectors", vector_pairs, vector_pairs),
            ("rotation_mean", ((qh, rh),), ((qj, rj),)),
            ("rotation_covariance", ((qh, rh),), ((qj, rj),)),
        )
        for operation, hargs, jargs in workloads:
            hfunction = getattr(ham, operation)
            jfunction = getattr(jpl, operation)
            count = (
                600 if operation in ("rotation_mean", "rotation_covariance") else 3000
            )
            compare(
                f"{name} {operation}",
                hfunction,
                hargs,
                jfunction,
                jargs,
                count,
                1e6,
                "us/call",
            )

    hq_xyzw, hr_xyzw = hs.to_xyzw(hq), hs.to_xyzw(hr)
    qh, rh = np.tile(hq_xyzw, (10000, 1)), np.tile(hr_xyzw, (10000, 1))
    qj, rj = np.tile(jq, (10000, 1)), np.tile(jr, (10000, 1))
    vectors = np.tile(vector, (10000, 1))
    for operation, hargs, jargs in (
        ("multiply", (qh, rh), (qj, rj)),
        ("rotate", (qh, vectors), (qj, vectors)),
    ):
        hfunction = getattr(ha, operation)
        jfunction = getattr(ja, operation)
        compare(
            f"NumPy {operation}",
            hfunction,
            hargs,
            jfunction,
            jargs,
            10,
            1e3,
            "ms/10k pairs",
        )


if __name__ == "__main__":
    main()
