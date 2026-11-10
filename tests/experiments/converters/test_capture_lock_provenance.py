"""Check that historical capture locks remain losslessly recoverable."""

import hashlib
import json

import pytest

from tests.support.paths import REPOSITORY_ROOT


@pytest.mark.parametrize(
    "target,version,manifest,field",
    [
        (
            "openusd",
            "26.11",
            "tests/golden/transports/openusd/26.11/converter/manifest.json",
            "lockfile_sha256",
        ),
    ],
)
def test_historical_lock_bytes_are_recoverable(target, version, manifest, field):
    recorded = json.loads((REPOSITORY_ROOT / manifest).read_text(encoding="utf-8"))
    lock = REPOSITORY_ROOT / "experiments/converters" / target / version / "uv.lock"
    historical = lock.read_bytes().replace(b"dev-experimental", b"alphus-trial")
    assert hashlib.sha256(historical).hexdigest() == recorded["provenance"][field]
