"""Read retained source files, whether standalone or inside a source bundle."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def read_provenance_source(repository_root: Path, reference: str) -> bytes:
    """Read a repository source path or ``archive.zip#member/path`` reference."""
    archive_reference, separator, member_name = reference.partition("#")
    if not separator:
        return (repository_root / archive_reference).read_bytes()
    member_path = PurePosixPath(member_name)
    if not member_name or member_path.is_absolute() or ".." in member_path.parts:
        raise ValueError(f"invalid provenance archive member {member_name!r}")
    with ZipFile(repository_root / archive_reference) as archive:
        return archive.read(member_path.as_posix())
