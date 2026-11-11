"""Run repository verification locally or in a clean Linux container."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

PYTHON = sys.executable


def run(*command: str, cwd: Path, env: dict[str, str] | None = None) -> None:
    """Run one check without a shell, stopping when it fails."""
    print(f"+ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def module(root: Path, env: dict[str, str], name: str, *arguments: str) -> None:
    run(PYTHON, "-m", name, *arguments, cwd=root, env=env)


def check_environment() -> dict[str, str]:
    """Make the optional Pyright Node distribution available to native JS tests."""
    from scripts.platform_probe import executable_suffix, is_windows

    env = os.environ.copy()
    spec = importlib.util.find_spec("nodejs_wheel")
    if spec is not None and spec.origin is not None:
        package = Path(spec.origin).parent
        node_dir = package if is_windows() else package / "bin"
        node = node_dir / f"node{executable_suffix()}"
        if node.is_file():
            env["PATH"] = f"{node_dir}{os.pathsep}{env.get('PATH', '')}"
    return env


def evidence(root: Path, env: dict[str, str]) -> None:
    """Validate maintained evidence and the generated registry."""
    for name, extra in (
        ("scripts.validate_evidence_json", ()),
        ("scripts.validate_converter_evidence", ()),
        ("scripts.render_converter_registry", ("--check",)),
    ):
        module(root, env, name, "--repository-root", str(root), *extra)


def pytest(root: Path, *arguments: str, env: dict[str, str]) -> None:
    """Give each test invocation an isolated temporary directory."""
    with tempfile.TemporaryDirectory(prefix="quatops-verification-") as directory:
        module(
            root,
            env,
            "pytest",
            "-q",
            *arguments,
            "--basetemp",
            str(Path(directory) / "pytest"),
        )


def ci(root: Path, env: dict[str, str]) -> None:
    """Run the checks shared with the CI Python-version matrix."""
    evidence(root, env)
    pytest(root, env=env)
    module(
        root,
        env,
        "mypy",
        "tests/typing_tests/foundational_consumer.py",
        "tests/typing_tests/consumer.py",
        "tests/typing_tests/float32_consumer.py",
    )
    module(root, env, "ruff", "check", "src", "tests", "examples", "experiments")
    module(
        root,
        env,
        "ruff",
        "check",
        "--select",
        "B,C4,E4,PERF,RUF,SIM,UP",
        "experiments/oracles",
        "experiments/converters",
        "scripts",
    )
    module(root, env, "ruff", "format", "--check", "examples", "experiments", "scripts")


def typecheck(root: Path, env: dict[str, str]) -> None:
    module(root, env, "pyright", "--pythonpath", PYTHON)


def lock(root: Path, env: dict[str, str]) -> None:
    module(
        root, env, "uv", "lock", "--check", "--project", "experiments/converters/python"
    )


def all_checks(root: Path, env: dict[str, str]) -> None:
    ci(root, env)
    typecheck(root, env)
    lock(root, env)
    module(root, env, "build", "--quiet")


def engine_checks(root: Path, env: dict[str, str]) -> None:
    """Replay native engine witnesses and independently generated basis maps."""
    pytest(
        root,
        "tests/quatops/experimental/engines/test_canonical_interoperability.py",
        "tests/quatops/experimental/engines/test_engine_basis_oracle.py",
        "tests/experiments/engines/shared/test_engine_proof_profiles.py",
        env=env,
    )


def interoperability_checks(root: Path, env: dict[str, str]) -> None:
    pytest(
        root,
        "tests/quatops/experimental",
        "tests/experiments/engines",
        "tests/experiments/converters",
        env=env,
    )


def benchmark_checks(
    root: Path, env: dict[str, str], *, converters_only: bool = False
) -> None:
    """Run opt-in benchmarks and keep the full report in the system temp folder."""
    target = (
        "tests/benchmarks/quatops/test_orientation_converter.py"
        if converters_only
        else "tests/benchmarks"
    )
    arguments = [
        target,
        "-m",
        "benchmark",
        "--benchmark-only",
        "--benchmark-group-by=group",
        "--benchmark-columns=median",
    ]
    if not converters_only:
        arguments.append(
            f"--benchmark-json={Path(tempfile.gettempdir()) / 'quatops-benchmarks.json'}"
        )
    pytest(root, *arguments, env=env)


def converter_benchmarks(root: Path, env: dict[str, str]) -> None:
    benchmark_checks(root, env, converters_only=True)


def linux_checks(root: Path) -> None:
    """Run the committed checkout in Docker, preserving Git modes and attributes."""
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        raise ValueError("Commit local changes before running Linux verification")
    with tempfile.TemporaryDirectory(prefix="quatops-linux-check-") as directory:
        bundle = Path(directory) / "source.bundle"
        run("git", "bundle", "create", str(bundle), "HEAD", cwd=root)
        run(
            "docker",
            "run",
            "--rm",
            "--mount",
            f"type=bind,source={bundle},target=/tmp/source.bundle,readonly",
            "--mount",
            f"type=bind,source={root / 'scripts/verify.py'},target=/tmp/verify.py,readonly",
            "python:3.12-slim",
            "python",
            "/tmp/verify.py",
            "--repository-root",
            "/tmp",
            "--bootstrap",
            cwd=root,
        )


def bootstrap_linux() -> None:
    """Install tooling in the disposable container and run the checked-out script."""
    temporary = Path("/tmp")
    root = temporary / "quatops"
    run("apt-get", "update", "-qq", cwd=temporary)
    run(
        "apt-get",
        "install",
        "-y",
        "-qq",
        "--no-install-recommends",
        "git",
        cwd=temporary,
    )
    run("git", "clone", "/tmp/source.bundle", str(root), cwd=temporary)
    venv = root / ".venv"
    run(PYTHON, "-m", "venv", str(venv), cwd=root)
    python = str(venv / "bin/python")
    run(
        python,
        "-m",
        "pip",
        "install",
        "--quiet",
        "-e",
        ".[dev,typecheck]",
        "build",
        "uv==0.11.28",
        cwd=root,
    )
    run(python, "-m", "scripts.verify", "--repository-root", str(root), "all", cwd=root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument(
        "scope",
        nargs="?",
        default="all",
        choices=(
            "all",
            "ci",
            "evidence",
            "engines",
            "interoperability",
            "benchmarks",
            "converter-benchmark",
            "typecheck",
            "lock",
            "linux",
        ),
        help="Select checks to run (default: all). The linux scope runs them in Docker.",
    )
    parser.add_argument("--bootstrap", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args()
    scope = options.scope
    try:
        if options.bootstrap:
            bootstrap_linux()
        else:
            root = options.repository_root.resolve(strict=True)
            if not (root / "pyproject.toml").is_file():
                raise ValueError(f"Not a repository root: {root}")
            if scope == "linux":
                linux_checks(root)
            else:
                checks = {
                    "all": all_checks,
                    "ci": ci,
                    "evidence": evidence,
                    "engines": engine_checks,
                    "interoperability": interoperability_checks,
                    "benchmarks": benchmark_checks,
                    "converter-benchmark": converter_benchmarks,
                    "typecheck": typecheck,
                    "lock": lock,
                }
                checks[scope](root, check_environment())
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        return 1
    if not options.bootstrap:
        print(f"Verification passed: {scope}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
