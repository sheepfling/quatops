"""The shared verifier must fail closed before launching unsafe or partial checks."""

import subprocess
from pathlib import Path

import pytest

from scripts import verify


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
