"""Run the pytest benchmarks and print an explicit winner per group.

Run from the repository root with ``.venv/bin/python
experiments/one_off_benchmark.py`` for scalar cases, or add
``--include-vectorized`` to include 10k NumPy workloads. Benchmark cases live in
``tests/benchmarks/`` so correctness checks and timings exercise the same
callables. Use ``--json results.json`` to save a machine-readable report.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy
import pytest
import pytest_benchmark
import quaternion
import scipy

ROOT = Path(__file__).resolve().parents[1]
TEST_FILES = (
    ROOT / "tests" / "benchmarks" / "hamilton" / "test_one_off.py",
    ROOT / "tests" / "benchmarks" / "jpl" / "test_one_off_jpl.py",
)
VECTOR_TEST_FILE = (
    ROOT / "tests" / "benchmarks" / "quatops" / "test_public_operations.py"
)
WINNER_TOLERANCE = 0.01


def _case_label(test_name: str, group: str) -> str:
    """Use compact backend names in reports while retaining convention labels."""
    case = test_name.removeprefix("test_")
    case = case.removeprefix(group.replace(" ", "_") + "_")
    for convention, label in (("hamilton", "Hamilton"), ("jpl", "JPL")):
        prefix = f"{convention}_numpy_"
        for operation in ("multiply_10k_", "rotate_10k_"):
            full_prefix = prefix + operation
            if case.startswith(full_prefix):
                return f"{label} {case.removeprefix(full_prefix)}"
    if group.startswith("scalar "):
        operation = group.removeprefix("scalar ") + "_"
        for convention, label in (("hamilton", "Hamilton"), ("jpl", "JPL")):
            prefix = f"{convention}_scalar_"
            if case.startswith(prefix):
                backend = case.removeprefix(prefix).removeprefix(operation)
                return f"{label} {backend}"
    for convention, label in (("hamilton", "Hamilton"), ("jpl", "JPL")):
        for api in ("scalar", "numpy"):
            prefix = f"{convention}_{api}_"
            if case.startswith(prefix):
                return f"{label} {case.removeprefix(prefix)}"
    return case


def _run_pytest(
    min_rounds: int, max_time: float, test_files: tuple[Path, ...] = TEST_FILES
) -> dict[str, Any]:
    """Run selected performance benchmarks and return pytest-benchmark JSON."""
    with tempfile.TemporaryDirectory(prefix="quatops-benchmark-") as temp_dir:
        json_path = Path(temp_dir) / "benchmark.json"
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-m",
            "benchmark",
            "--benchmark-only",
            f"--benchmark-min-rounds={min_rounds}",
            f"--benchmark-max-time={max_time}",
            "--benchmark-warmup=on",
            f"--benchmark-json={json_path}",
            *(str(test_file) for test_file in test_files),
        ]
        result = subprocess.run(
            command, check=False, cwd=ROOT, capture_output=True, text=True
        )
        if result.returncode:
            print(result.stdout, end="", file=sys.stderr)
            print(result.stderr, end="", file=sys.stderr)
            raise SystemExit(result.returncode)
        return json.loads(json_path.read_text(encoding="utf-8"))


def _winner_report(data: dict[str, Any]) -> dict[str, Any]:
    """Build grouped median results and select the fastest median per group."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for benchmark in data["benchmarks"]:
        grouped[benchmark["group"]].append(benchmark)

    results: dict[str, Any] = {}
    for group, benchmarks in sorted(grouped.items()):
        rows = []
        for benchmark in benchmarks:
            case = _case_label(benchmark["name"], group)
            rows.append(
                {
                    "backend": case,
                    "min_us": benchmark["stats"]["min"] * 1e6,
                    "median_us": benchmark["stats"]["median"] * 1e6,
                    "mean_us": benchmark["stats"]["mean"] * 1e6,
                    "stddev_us": benchmark["stats"]["stddev"] * 1e6,
                    "rounds": benchmark["stats"]["rounds"],
                }
            )
        rows.sort(key=lambda row: row["median_us"])
        fastest = rows[0]["median_us"]
        for row in rows:
            row["relative_to_winner"] = row["median_us"] / fastest
            row["within_one_percent_of_winner"] = (
                row["relative_to_winner"] <= 1.0 + WINNER_TOLERANCE
            )
        results[group] = {
            "winner": rows[0]["backend"],
            "co_winners": [
                row["backend"] for row in rows if row["within_one_percent_of_winner"]
            ],
            "results": rows,
        }
    return results


def _print_report(report: dict[str, Any]) -> None:
    """Print environment, winners, and relative medians in readable tables."""
    environment = report["environment"]
    title = (
        "QuatOps performance benchmark"
        if report["settings"].get("include_vectorized")
        else "QuatOps one-off benchmark"
    )
    print(title)
    print(
        f"Python {environment['python']}; NumPy {environment['numpy']}; "
        f"SciPy {environment['scipy']}; numpy-quaternion {environment['numpy_quaternion']}; "
        f"pytest-benchmark {environment['pytest_benchmark']}"
    )
    print(f"Platform: {environment['platform']}")
    print(
        f"Settings: {report['settings']['min_rounds']} rounds minimum, "
        f"{report['settings']['max_time']} s max per case, warmup enabled"
    )
    print("Backends within 1% of the fastest median are marked co-winners.")
    for group, result in report["groups"].items():
        tied = [name for name in result["co_winners"] if name != result["winner"]]
        tie_note = f"; within 1%: {', '.join(tied)}" if tied else ""
        print(f"\n{group} — WINNER: {result['winner']}{tie_note}")
        backend_width = max(16, *(len(row["backend"]) for row in result["results"]))
        print(
            f"{'Backend':<{backend_width}} {'Median (us)':>12} "
            f"{'vs winner':>12} {'Rounds':>8}"
        )
        for row in result["results"]:
            if row["backend"] == result["winner"]:
                label = "  WINNER"
            elif row["within_one_percent_of_winner"]:
                label = "  CO-WINNER"
            else:
                label = ""
            print(
                f"{row['backend']:<{backend_width}} {row['median_us']:>12.3f} "
                f"{row['relative_to_winner']:>11.2f}x{label} "
                f"{row['rounds']:>8}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-rounds", type=int, default=9)
    parser.add_argument("--max-time", type=float, default=0.2)
    parser.add_argument(
        "--include-vectorized",
        action="store_true",
        help="also compare 10k NumPy operations with SciPy and numpy-quaternion",
    )
    parser.add_argument("--json", type=Path, help="write the report to this path")
    args = parser.parse_args(argv)
    if args.min_rounds < 1 or args.max_time <= 0:
        parser.error(
            "--min-rounds must be positive and --max-time must be greater than zero"
        )

    test_files = TEST_FILES
    if args.include_vectorized:
        test_files += (VECTOR_TEST_FILE,)
    plugin_data = _run_pytest(args.min_rounds, args.max_time, test_files)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": numpy.__version__,
            "numpy_quaternion": quaternion.__version__,
            "scipy": scipy.__version__,
            "pytest": pytest.__version__,
            "pytest_benchmark": pytest_benchmark.__version__,
            "platform": platform.platform(),
        },
        "settings": {
            "min_rounds": args.min_rounds,
            "max_time": args.max_time,
            "include_vectorized": args.include_vectorized,
        },
        "winner_tolerance_percent": WINNER_TOLERANCE * 100,
        "groups": _winner_report(plugin_data),
    }
    _print_report(report)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved report: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
