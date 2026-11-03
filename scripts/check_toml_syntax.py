"""Parse every TOML path passed on the command line."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9 and 3.10
    import tomli as tomllib


def main(paths: list[str]) -> int:
    for path in paths:
        with Path(path).open("rb") as stream:
            tomllib.load(stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
