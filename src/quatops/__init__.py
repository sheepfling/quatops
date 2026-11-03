"""Quaternion operations with explicitly selected conventions and layouts.

Choose a fully specified backend module such as ``quatops.python.hamilton`` or
``quatops.numpy.hamilton.xyzw``. Importing this root package does not import
NumPy.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

# setuptools-scm supplies the installed distribution version at build time.
try:
    __version__ = _distribution_version("quatops")
except PackageNotFoundError:
    # A source checkout may not have installed distribution metadata yet.
    __version__ = "0+unknown"
