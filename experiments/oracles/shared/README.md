# Shared native-capture inputs and helpers

[`orientation_inputs_v1.json`](orientation_inputs_v1.json) is a shared input
corpus for mathematical oracle tests and selected ROS 2/tf2 cases. The broader
discovery outputs are not retained per target; converter packages refer to
the smaller selected set under `experiments/interoperability/v1/`.

[`capture_io.py`](capture_io.py) provides a small standard-library capture
envelope for native Python wrappers. It records input, output, source, runtime,
precision, and binary identities without implementing rotation formulas or
expected values.
