"""Verify wheel contents and run consumers in a fresh, isolated environment."""

from __future__ import annotations

import argparse
import tempfile
import venv
from pathlib import Path
from zipfile import ZipFile

from scripts.verify import run


def check_wheel_contents(wheel: Path) -> None:
    """Reject local Python and Numba caches in a distributable wheel."""
    with ZipFile(wheel) as archive:
        caches = [
            name
            for name in archive.namelist()
            if name.endswith((".nbc", ".nbi", ".pyc")) or "/__pycache__/" in name
        ]
    if caches:
        raise ValueError(f"Wheel includes runtime caches: {caches}")


def check_installed_wheel(wheel: Path) -> None:
    """Install exactly this wheel, checking scalar before installing NumPy."""
    check_wheel_contents(wheel)
    consumer = Path(__file__).with_name("wheel_consumer.py")
    with tempfile.TemporaryDirectory(prefix="quatops-installed-wheel-") as directory:
        working = Path(directory)
        environment = working / "venv"
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(str(environment))
        python = Path(builder.ensure_directories(str(environment)).env_exe)
        run(
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            str(wheel.resolve()),
            cwd=working,
        )
        run(str(python), "-I", str(consumer), cwd=working)
        run(
            str(python),
            "-m",
            "pip",
            "install",
            "--only-binary=:all:",
            "numpy>=1.20",
            cwd=working,
        )
        run(str(python), "-I", str(consumer), "--numpy", cwd=working)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    check_installed_wheel(parser.parse_args().wheel.resolve(strict=True))


if __name__ == "__main__":
    main()
