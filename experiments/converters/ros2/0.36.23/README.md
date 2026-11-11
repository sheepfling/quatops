# ROS 2 Jazzy / tf2 0.36.23 converter capture

This package preserves the native `BufferCore` capture source, CMake recipe,
container pin, and capture wrapper for the selected two-direction frame-chain
proof. Inputs come from the explicit [frame-chain recipe](chain_inputs.json).
The committed witness file retains the exact native values used by the
replay test; broad capture output belongs in a temporary directory.
The current exporter emits only the two selected lookups, shared edge inputs,
and the forward message, round trip, and point action. The wrapper validates
and tags finite binary64 values without changing them, including signed zero.
It omits the old temporary matrix and direct-composition observations. The
original fixture and its historical provenance remain unchanged; its native
producer sources are archived in `experiments/provenance/reproduction-v1.zip`.
The message-to-transform check accepts either quaternion sign and float64
rounding within `2e-12`; the stored native values are never normalized or
rewritten by that comparison.

The capture requires Docker with Linux containers. From the repository root:

```shell
python experiments/converters/ros2/0.36.23/run_capture.py --repository-root .
```

The runner builds the pinned image, compares native values to the retained
witnesses, and writes the result to a temporary directory. Pass `--output` to
select a new or empty directory. The capture writes `witnesses.json` and a
run-specific `manifest.json`; its observation count is two. The historical fixture
records the nine-observation capture from which it was originally selected.
The comparison checks all selected values, including signed zero,
against that historical fixture and records its digest in the new run manifest.
The retained fixture
and QuatOps replay are in
[`tests/golden/oracles/ros2/0.36.23`](../../../../tests/golden/oracles/ros2/0.36.23/manifest.json)
and [`tests/experiments/converters/ros2/test_ros_tf2_converter.py`](../../../../tests/experiments/converters/ros2/test_ros_tf2_converter.py).
