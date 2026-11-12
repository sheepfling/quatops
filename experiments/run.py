"""Run a QuatOps experiment with source paths set up automatically.

Examples from the repository root::

    .venv/bin/python experiments/run.py --list
    .venv/bin/python experiments/run.py check-homogeneous
    .venv/bin/python experiments/run.py rotation-benchmark
"""

from __future__ import annotations

import argparse
import importlib.util
import runpy
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
COMMANDS = {
    "check-homogeneous": ("check_homogeneous.py", ("numpy",)),
    "homogeneous-scalar": ("homogeneous_scalar.py", ()),
    "homogeneous-numpy": ("homogeneous_numpy.py", ("numpy",)),
    "homogeneous-benchmark": ("bench_homogeneous.py", ("numpy",)),
    "rotation-benchmark": ("rotation_benchmark.py", ("numpy", "scipy")),
    "numpy-layout-benchmark": ("numpy_layout_benchmark.py", ("numpy",)),
    "convention-benchmark": ("convention_benchmark.py", ("numpy",)),
}


def main(argv: Optional[Sequence[str]] = None) -> int:  # noqa: UP045
    parser = argparse.ArgumentParser(description="Run a QuatOps experiment.")
    parser.add_argument(
        "--list", action="store_true", help="list available experiments"
    )
    parser.add_argument("experiment", nargs="?", choices=tuple(COMMANDS))
    parser.add_argument(
        "modules",
        nargs="*",
        help="optional module names for check-homogeneous (default: both prototypes)",
    )
    args = parser.parse_args(argv)
    if args.list:
        for name, (script, requirements) in COMMANDS.items():
            extra = (
                " (requires " + ", ".join(requirements) + ")" if requirements else ""
            )
            print(f"{name}: {script}{extra}")
        return 0
    if not args.experiment:
        parser.print_help()
        return 2

    script, requirements = COMMANDS[args.experiment]
    missing = [name for name in requirements if importlib.util.find_spec(name) is None]
    if missing:
        print(
            "Missing optional dependencies: "
            + ", ".join(missing)
            + ". Install them with: python3 -m pip install -e '.[experiments]'",
            file=sys.stderr,
        )
        return 2

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(EXPERIMENTS))
    sys.argv = [str(EXPERIMENTS / script)]
    if args.experiment == "check-homogeneous":
        sys.argv.extend(args.modules)
    runpy.run_path(str(EXPERIMENTS / script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
