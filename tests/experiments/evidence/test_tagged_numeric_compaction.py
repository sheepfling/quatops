"""Nonfinite and signed-zero tags remain intact through generic JSON compaction."""

from __future__ import annotations

import json

from experiments.oracles.shared.capture_io import numeric
from scripts.evidence_json import compact_numeric_evidence, parse_evidence_payload


def test_scientific_nonfinite_and_signed_zero_tags_survive_parse_and_compaction():
    tagged = numeric([float("nan"), -0.0, 2.0])
    parsed = parse_evidence_payload(json.dumps(tagged, allow_nan=False))
    compacted = compact_numeric_evidence(parsed)
    assert compacted["values"][0] == {"nonfinite": "nan"}
    assert compacted["values"][1] == 0.0
    assert compacted["ieee754_hex_be"][1] == "8000000000000000"
    assert "shape" not in compacted
