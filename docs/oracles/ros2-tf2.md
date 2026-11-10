# ROS 2 Jazzy / tf2 frame-chain proof

## Native runtime capture

The capture compiled and executed a C++20 exporter against the pinned
`ros:jazzy-ros-base` image digest
`sha256:8eed0a4669bbf43ee331878a4c39bd8ebddea8e3312a58b4fae8edb14526b617`.
It used ROS 2 Jazzy, `tf2` 0.36.23, `tf2_geometry_msgs` 0.36.23, and
`geometry_msgs` 5.3.8, with 64-bit `tf2Scalar`. The native `BufferCore` capture
ran in Linux x86-64 under WSL2; it did not use DDS or a live ROS network.
The Dockerfile, runner, C++ source, and selected native observations are
versioned in the workspace.

The initial native run produced 102 observations. The retained set contains
the two lookups for one nontrivial, same-timestamp chain:
`world <- base <- sensor`. Both edges have nonzero translation and rotation;
their rotations do not commute. The native output records
`lookupTransform(target, source)` in `target_from_source` direction, so the
selected lookups are `world_from_sensor` and `sensor_from_world`.

## Explicit frame and message contract

The canonical source is declared right-handed as `+X right/east`, `+Y
forward/north`, `+Z up`. The fixture declares separate maps for ROS world,
base, and optical sensor coordinates:

| Frame | Canonical coordinates mapped to frame coordinates |
| --- | --- |
| world | `(+X, +Y, +Z)` |
| base | `(+Y, -X, +Z)` |
| optical sensor | `(+X, -Z, +Y)` |

These are explicit test-frame declarations, not assumptions inferred from ROS
frame names. Quaternion component order, Hamilton algebra, active source-point
to target-point action, and transform direction are recorded in the
[versioned contract](../../experiments/converters/ros2/0.36.23/semantic_contract.json).
The observations also retain the `TransformStamped` parent/child IDs, stamp,
translation and quaternion fields, plus the native `tf2::fromMsg` round trip.

## QuatOps replay and limits

[`test_ros_tf2_converter.py`](../../tests/experiments/converters/ros2/test_ros_tf2_converter.py) converts
the native edge rotations and translations into the declared canonical
frames, composes the two labelled edges, and compares that result with each
native forward/reverse lookup. It also compares native transformed points,
message fields, and the message-to-tf2 round trip. The fixture keeps the two
selected exact native observations and their hashes; the broader passing
capture remains discovery output.

This proves the static frame-chain and message-field boundary for the tested
ROS 2/tf2 build. It does not cover DDS/CDR serialization, live publishers,
time interpolation/extrapolation, dynamic transforms, or covariance messages.
The Python replay and its SciPy/reference checks do not start ROS; native
runtime execution and mathematical replay are separate evidence.
The [versioned capture runbook](../../experiments/converters/ros2/0.36.23/README.md)
contains the Docker image, C++ build, capture, and retention commands.
