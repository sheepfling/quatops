"""The shared verifier must fail closed before launching unsafe or partial checks."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import verify
from tests.support.paths import REPOSITORY_ROOT


def test_linux_verification_rejects_uncommitted_changes(tmp_path: Path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "uncommitted.txt").write_text("uncommitted", encoding="utf-8")
    with pytest.raises(ValueError, match="Commit local changes"):
        verify.linux_checks(tmp_path)


def test_ci_stops_when_a_check_fails(tmp_path: Path, monkeypatch):
    commands = []

    def fail(*command, **_kwargs):
        commands.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(verify, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        verify.ci(tmp_path, {})
    assert len(commands) == 1


def test_ci_mapping_keeps_each_tool_check(tmp_path: Path, monkeypatch):
    calls = []
    monkeypatch.setattr(verify, "evidence", lambda root, env: calls.append("evidence"))
    monkeypatch.setattr(verify, "pytest", lambda root, **kwargs: calls.append("pytest"))
    monkeypatch.setattr(
        verify, "module", lambda root, env, name, *args: calls.append((name, args))
    )
    verify.ci(tmp_path, {})
    assert calls[:2] == ["evidence", "pytest"]
    tool_calls = calls[2:]
    assert [name for name, _args in tool_calls] == ["mypy", "ruff", "ruff", "ruff"]
    assert all(name in tool_calls[1][1] for name in ("src", "tests", "experiments"))
    assert "--select" in tool_calls[2][1] and "scripts" in tool_calls[2][1]
    assert tool_calls[3][1][:2] == ("format", "--check")


def test_verifier_discovers_checkout_from_another_working_directory(tmp_path: Path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPOSITORY_ROOT)
    result = subprocess.run(
        [sys.executable, "-m", "scripts.verify", "evidence"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Verification passed: evidence" in result.stdout


def test_verifier_rejects_an_explicit_subdirectory(tmp_path: Path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPOSITORY_ROOT)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.verify",
            "--repository-root",
            str(REPOSITORY_ROOT / "experiments"),
            "evidence",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Not a repository root" in result.stderr


def test_pytest_runs_use_separate_temporary_directories(tmp_path: Path, monkeypatch):
    basetemps = []

    def capture(*command, **_kwargs):
        basetemps.append(Path(command[command.index("--basetemp") + 1]))
        assert basetemps[-1].parent.is_dir()

    monkeypatch.setattr(verify, "run", capture)
    verify.pytest(tmp_path, env={})
    verify.pytest(tmp_path, env={})
    assert basetemps[0] != basetemps[1]


def test_benchmark_report_uses_the_system_temporary_directory(
    tmp_path: Path, monkeypatch
):
    calls = []

    def capture(root, *arguments, env):
        calls.append((root, arguments, env))

    monkeypatch.setattr(verify, "pytest", capture)
    monkeypatch.setattr(verify.tempfile, "gettempdir", lambda: str(tmp_path))
    verify.benchmark_checks(tmp_path, {})
    assert f"--benchmark-json={tmp_path / 'quatops-benchmarks.json'}" in calls[0][1]
