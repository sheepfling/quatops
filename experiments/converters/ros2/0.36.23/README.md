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

The capture requires Docker Desktop with Linux containers. From the repository
root, build the pinned image and run the capture into a temporary host folder:

```powershell
$image = 'quatops-ros2-tf2:jazzy'
$captureDir = Join-Path $env:TEMP 'quatops-captures\ros2-tf2-0.36.23'
New-Item -ItemType Directory -Force $captureDir | Out-Null
docker build --tag $image --file experiments\converters\ros2\0.36.23\Dockerfile experiments\converters\ros2\0.36.23
$containerCommand = @'
source /opt/ros/jazzy/setup.bash
colcon build \
  --base-paths experiments/converters/ros2/0.36.23/native \
  --build-base /tmp/ros-build \
  --install-base /tmp/ros-install
source /tmp/ros-install/setup.bash
PYTHONPATH=/workspace /opt/quatops-evidence/bin/python experiments/converters/ros2/0.36.23/capture_native.py \
  --repository-root /workspace \
  --inputs experiments/converters/ros2/0.36.23/chain_inputs.json \
  --output /captures/full \
  --executable /tmp/ros-install/quatops_ros_capture/lib/quatops_ros_capture/capture_ros \
  --compare-witnesses tests/golden/oracles/ros2/0.36.23/witnesses.json \
  --runtime-image ros:jazzy-ros-base@sha256:8eed0a4669bbf43ee331878a4c39bd8ebddea8e3312a58b4fae8edb14526b617
'@
docker run --rm `
  --volume "${PWD}:/workspace" `
  --volume "${captureDir}:/captures" `
  $image `
  $containerCommand
```

The capture writes `witnesses.json` and a run-specific `manifest.json` directly
under `/captures/full`. Its observation count is two; the historical fixture
records the nine-observation capture from which it was originally selected.
The optional comparison checks all selected values, including signed zero,
against that historical fixture and records its digest in the new run manifest.
The retained fixture
and QuatOps replay are in
[`tests/golden/oracles/ros2/0.36.23`](../../../../tests/golden/oracles/ros2/0.36.23/manifest.json)
and [`tests/experiments/converters/ros2/test_ros_tf2_converter.py`](../../../../tests/experiments/converters/ros2/test_ros_tf2_converter.py).
