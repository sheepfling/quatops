"""Evidence file digests preserve the exact byte stream."""

from __future__ import annotations

import hashlib

import pytest

from scripts.evidence_schema.utilities import sha256_file


@pytest.mark.parametrize("payload", [b"", bytes(range(256)) * 4097])
def test_sha256_file_handles_empty_and_large_binary_files(tmp_path, payload):
    path = tmp_path / "capture.bin"
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()
