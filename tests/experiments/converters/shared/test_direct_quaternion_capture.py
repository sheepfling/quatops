"""The shared runner must reject invalid recipes before calling native code."""

import sys

import pytest

from scripts.direct_quaternion_capture import run_direct_capture
from scripts.evidence_schema.converter_captures import MuJoCoInputs, SpiceyPyInputs
from tests.support.paths import REPOSITORY_ROOT


@pytest.mark.parametrize(
    "target,version,input_type",
    [("mujoco", "3.17.0", MuJoCoInputs), ("spiceypy", "8.3.0", SpiceyPyInputs)],
)
def test_duplicate_input_keys_are_rejected_before_native_execution(
    tmp_path, monkeypatch, target, version, input_type
):
    source = REPOSITORY_ROOT / "experiments/converters" / target / version
    original = (source / "converter_inputs.json").read_text(encoding="utf-8")
    payload = original.replace(
        '"schema_version": "1.0",',
        '"schema_version": "1.0", "schema_version": "1.0",',
    )
    input_path = tmp_path / "inputs.json"
    input_path.write_text(payload, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "capture",
            "--repository-root",
            str(REPOSITORY_ROOT),
            "--inputs",
            str(input_path),
        ],
    )

    def unexpected_capture(*_args):
        pytest.fail("invalid input reached the native capture")

    with pytest.raises(ValueError):
        run_direct_capture(source / "capture_native.py", input_type, unexpected_capture)
