# Shared native-capture inputs and helpers

[`orientation_inputs_v1.json`](orientation_inputs_v1.json) contains only the
selected inputs needed to replay retained native witnesses. The broader
discovery sweep was removed after selection; its SHA-256 remains in the
corpus' `retention.source_discovery_corpus_sha256` field. Native outputs stay
with target-specific retained fixtures.

[`capture_io.py`](capture_io.py) provides a small standard-library capture
envelope for native Python wrappers. It records input, output, source, runtime,
precision, and binary identities without implementing rotation formulas or
expected values.
