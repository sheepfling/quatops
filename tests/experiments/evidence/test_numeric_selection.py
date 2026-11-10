"""Selected JavaScript native values preserve finite numbers and signed zero."""

import math
import shutil
import subprocess

import pytest

from scripts.evidence_json import parse_evidence_document
from tests.support.paths import REPOSITORY_ROOT


def test_javascript_capture_serializes_signed_zero_and_rejects_nonfinite() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node 24 is required for the native JavaScript capture")
    support = subprocess.run(
        [node, "-p", "typeof JSON.rawJSON"], check=True, capture_output=True, text=True
    )
    if support.stdout.strip() != "function":
        pytest.skip("Node with JSON.rawJSON support is required")
    helper = REPOSITORY_ROOT / "experiments/converters/shared/numeric_evidence.mjs"
    program = """
        const {numeric, stringifyEvidence} = await import(process.argv[1]);
        try { numeric([Infinity]); throw new Error('accepted Infinity'); }
        catch (error) { if (!(error instanceof TypeError)) throw error; }
        process.stdout.write(stringifyEvidence(numeric([-0, Number.MIN_VALUE, 1.5])));
    """
    result = subprocess.run(
        [node, "--input-type=module", "-e", program, helper.as_uri()],
        check=True,
        capture_output=True,
        text=True,
    )
    decoded = parse_evidence_document(result.stdout)
    assert math.copysign(1.0, decoded["values"][0]) == -1.0
    assert decoded["values"][1:] == [5e-324, 1.5]
