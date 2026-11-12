"""Fetch the pinned Cesium distribution and extract its standalone Node bundle."""

from __future__ import annotations

import base64
import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from scripts.evidence_json import read_evidence_document
from scripts.evidence_schema.primitives import NativeCaptureModel


class CesiumDistribution(NativeCaptureModel):
    """The upstream archive and digest used to reproduce a Cesium capture."""

    name: Literal["cesium"]
    version: str
    url: str
    integrity: Annotated[str, Field(pattern=r"^sha512-[A-Za-z0-9+/]{86}==$")]


def install_distribution(pin_path: Path, destination: Path) -> CesiumDistribution:
    """Download and verify the archive before extracting the two required files."""
    pin = CesiumDistribution.model_validate(
        read_evidence_document(pin_path), strict=True
    )
    archive_path = destination / "cesium.tgz"
    digest = hashlib.sha512()
    with (
        urllib.request.urlopen(pin.url, timeout=120) as response,
        archive_path.open("wb") as archive,
    ):
        for block in iter(lambda: response.read(1024 * 1024), b""):
            digest.update(block)
            archive.write(block)
    integrity = "sha512-" + base64.b64encode(digest.digest()).decode("ascii")
    if integrity != pin.integrity:
        raise ValueError("Cesium distribution integrity does not match its pin")

    with tarfile.open(archive_path, "r:gz") as archive:
        for member_name, output_name in (
            ("package/package.json", "package.json"),
            ("package/Build/Cesium/index.cjs", "index.cjs"),
        ):
            member = archive.getmember(member_name)
            if not member.isfile():
                raise ValueError(f"Cesium archive member is not a file: {member_name}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read Cesium archive member: {member_name}")
            with source, (destination / output_name).open("wb") as output:
                shutil.copyfileobj(source, output)
    metadata = read_evidence_document(destination / "package.json")
    if (metadata.get("name"), metadata.get("version")) != (pin.name, pin.version):
        raise ValueError("Cesium distribution identity does not match its pin")
    return pin
