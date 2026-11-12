"""The Cesium capture uses only verified files from its pinned archive."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import tarfile

import pytest

from experiments.converters.cesium import distribution


def test_distribution_verifies_archive_before_using_its_bundle(tmp_path, monkeypatch):
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for name, content in (
            ("package/package.json", b'{"name":"cesium","version":"1.124.0"}'),
            ("package/Build/Cesium/index.cjs", b"module.exports = {};"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    data = payload.getvalue()
    pin = tmp_path / "distribution.json"
    pin.write_text(
        json.dumps(
            {
                "name": "cesium",
                "version": "1.124.0",
                "url": "https://example.invalid/cesium.tgz",
                "integrity": "sha512-"
                + base64.b64encode(hashlib.sha512(data).digest()).decode("ascii"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        distribution.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(data)
    )
    output = tmp_path / "verified"
    output.mkdir()
    assert distribution.install_distribution(pin, output).version == "1.124.0"
    assert (output / "index.cjs").read_bytes() == b"module.exports = {};"

    monkeypatch.setattr(
        distribution.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(data + b"tampered"),
    )
    rejected = tmp_path / "rejected"
    rejected.mkdir()
    with pytest.raises(ValueError, match="integrity"):
        distribution.install_distribution(pin, rejected)
    assert not (rejected / "index.cjs").exists()
