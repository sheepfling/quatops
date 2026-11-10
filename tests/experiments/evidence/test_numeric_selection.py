"""Selected JavaScript native values preserve finite numbers and signed zero."""

import math
import shutil
import subprocess

import pytest

from scripts.evidence_json import dump_evidence_json, parse_evidence_document
from scripts.retain_cesium_converter_witnesses import _vectors
from scripts.retain_gltf_converter_witnesses import select_numeric
from tests.support.paths import REPOSITORY_ROOT


def test_selectors_preserve_values_and_signed_zero() -> None:
    item = {"values": [[-0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], "dtype": "binary64"}
    assert _vectors(item, start=-1) == {
        "values": [[3.0, 4.0, 5.0]],
        "dtype": "binary64",
    }
    selected = select_numeric(item, 0)
    assert selected == {"values": [-0.0, 1.0, 2.0], "dtype": "binary64"}
    decoded = parse_evidence_document(dump_evidence_json(selected))
    assert math.copysign(1.0, decoded["values"][0]) == -1.0


@pytest.mark.parametrize(
    "item,selection,error",
    [
        ({"values": [[1.0]], "dtype": "binary32"}, 0, ValueError),
        ({"values": 1.0, "dtype": "binary64"}, 0, TypeError),
        ({"values": [[1.0]], "dtype": "binary64", "shape": [1, 1]}, 0, TypeError),
        ({"values": [[1.0]], "dtype": "binary64"}, True, TypeError),
        ({"values": [[1.0]], "dtype": "binary64"}, 1, IndexError),
        ({"values": [[1.0], [2.0, 3.0]], "dtype": "binary64"}, 0, ValueError),
        ({"values": [float("nan")], "dtype": "binary64"}, 0, ValueError),
        ({"values": [{"nonfinite": "nan"}], "dtype": "binary64"}, 0, TypeError),
    ],
)
def test_selection_rejects_invalid_inputs(item, selection, error) -> None:
    with pytest.raises(error):
        select_numeric(item, selection)


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
