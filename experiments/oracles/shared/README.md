# Shared native-capture inputs and helpers

[`orientation_inputs_v1.json`](orientation_inputs_v1.json) is the frozen input
corpus used by versioned native captures. It contains inputs only; native
outputs stay with the target-specific retained fixtures.

[`capture_io.py`](capture_io.py) provides a small standard-library capture
envelope for native Python wrappers. It records input, output, source, runtime,
precision, and binary identities without implementing rotation formulas or
expected values.
