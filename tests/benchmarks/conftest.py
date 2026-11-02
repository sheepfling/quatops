"""Show numerical-library versions alongside benchmark results."""

from importlib.metadata import PackageNotFoundError, version

import numpy
import quaternion
import scipy


def pytest_report_header():
    try:
        numba_version = version("numba")
    except PackageNotFoundError:
        numba_label = "Numba unavailable (pure NumPy path)"
    else:
        numba_label = f"Numba {numba_version} (batch acceleration available)"
    return (
        f"benchmark dependencies: NumPy {numpy.__version__}, "
        f"numpy-quaternion {quaternion.__version__}, SciPy {scipy.__version__}, "
        f"{numba_label}"
    )
