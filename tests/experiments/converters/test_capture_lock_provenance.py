"""Check that historical capture locks remain losslessly recoverable."""

import hashlib
import json

import pytest

from tests.support.paths import REPOSITORY_ROOT


@pytest.mark.parametrize(
    "target,version,manifest,field",
    [
        (
            "mujoco",
            "3.17.0",
            "tests/golden/oracles/mujoco/3.17.0/converter/manifest.json",
            "pinned_lockfile_sha256",
        ),
        (
            "opencv",
            "4.13.0",
            "tests/golden/oracles/opencv/4.13.0/converter/manifest.json",
            "lockfile_sha256",
        ),
        (
            "openusd",
            "26.11",
            "tests/golden/transports/openusd/26.11/converter/manifest.json",
            "lockfile_sha256",
        ),
        (
            "spiceypy",
            "8.3.0",
            "tests/golden/oracles/spiceypy/8.3.0/converter/manifest.json",
            "pinned_lockfile_sha256",
        ),
    ],
)
def test_historical_lock_bytes_are_recoverable(target, version, manifest, field):
    recorded = json.loads((REPOSITORY_ROOT / manifest).read_text(encoding="utf-8"))
    lock = REPOSITORY_ROOT / "experiments/converters" / target / version / "uv.lock"
    historical = lock.read_bytes().replace(b"dev-experimental", b"alphus-trial")
    assert hashlib.sha256(historical).hexdigest() == recorded["provenance"][field]
