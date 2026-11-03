# Running the captures

Run commands from the extracted bundle root unless noted. Use a fresh output directory for every native execution. Never overwrite the supplied captures or regenerate goldens automatically after a test failure.

## 1. Replay the completed evidence

The exact Python versions used are recorded in verification/runtime_inventory.json. A reproducibility requirements file is supplied; it is not a cryptographic dependency lock and the native manifests remain the authoritative record of what actually ran.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-replay.txt
python -m tools.verify_results
python -m pytest -q
node js/test_capture_io.mjs
```

The comparison script writes a new derived report. Archive a copy before changing reference-library versions. Raw captures are not rewritten by replay. Absolute provenance paths in original manifests are mapped by tests to their corresponding extracted bundle paths; the original strings are preserved as evidence.

## 2. Repeat installed-library captures

Install requirements-native-executed.txt in a separate environment. The delivery environment happened to contain both OpenCV GUI/headless distribution metadata at the same version; the recipe selects headless only. Record the actual resolved cv2 binary instead of assuming those environments are bit-identical.

```bash
python -m exporters.capture_opencv --output runs/local-opencv
python -m exporters.capture_gltf_trimesh --output runs/local-trimesh
python -m exporters.capture_gltf_vtk --output runs/local-vtk
python -m exporters.capture_proj --output runs/local-proj
```

Optional VTK offscreen warnings must remain in stderr logs. Import/sampling does not require rendering an image. PROJ network is explicitly disabled by its exporter.

The provided verifier selects the four supplied run folders. To compare a new capture, add its manifest/path explicitly; do not replace a supplied run or silently treat a different runtime as the original version.

## 3. Cesium, Three.js and Khronos validator

Node 22 or later is the chosen capture environment. The requested top-level pins in js/package.json are Cesium 1.124.0, Three.js 0.187.0 and gltf-validator 2.0.0-dev.3.10. These are a proposed capture baseline, not a statement that all are latest. Package installation and compatibility were not verified here.

A transitive lockfile could not be generated because downloads failed. In a network-enabled environment, resolve once, inspect and retain the generated lockfile, then use npm ci for subsequent executions:

```bash
cd js
npm install --ignore-scripts
npm ls --all --json > resolved-before-capture.json
# Retain package-lock.json and review the resolved dependency graph.
npm ci --ignore-scripts
node capture_cesium.mjs --output ../runs/local-cesium
node capture_three.mjs --output ../runs/local-three
node validate_gltf.mjs --output ../runs/local-khronos-validation
cd ..
```

The runtime exporters refuse to run without package-lock.json. They record Node/V8, dependency graph, package integrity and actual JavaScript entry hashes. Math captures need neither an ion token nor a browser renderer. There is no Cesium Model/glTF import, CZML, time-dependent ICRF/EOP, GPU or browser parity claim.

Run the validator first when promoting glTF inputs; fix any errors as a *new version* of the assets and preserve old inputs/runs rather than silently editing their hashes. Native importer behavior and validator acceptance are different evidence types.

## 4. OpenUSD

In a separate environment containing official pxr bindings, resolve and pin the installed distribution. `usd-core==26.11` was identified as an available candidate during research, but it was not installed or tested here. Existing vendor builds are also suitable when their identities are retained.

```bash
python -m pip install 'usd-core==26.11'
python -c 'from pxr import Usd; print(Usd.GetVersion())'
python -m exporters.capture_usd --output runs/local-openusd
```

This exporter must successfully parse the bundled USDA files; their text checks alone are not native validation. It covers Gf float/double quaternions and UsdGeom stack evaluation. In-memory rotate-op inputs use degrees. Gf matrices are captured with row-vector action and native row arrays. Extraction through Gf.Rotation returns double precision and is labelled separately even when the input was Matrix3f.

If an installed version lacks a particular API, record it as unsupported or adjust a separately versioned exporter; do not fill its output with Python formulas.

## 5. ROS 2 / tf2

Prepared for a pinned ROS 2 Jazzy environment with a C++20 compiler, colcon, ament_cmake, tf2, tf2_geometry_msgs, geometry_msgs and nlohmann_json. Source compilation has not been performed. The package maintainer field is a placeholder to replace before publishing the package.

In an appropriate ROS shell:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --base-paths ros2/quatops_ros_capture \
  --build-base build/ros --install-base install/ros \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/ros/setup.bash
python -m exporters.run_ros \
  --executable install/ros/lib/quatops_ros_capture/capture_ros \
  --output runs/local-ros-tf2
```

Retain the full compiler output and exact ROS installation/container digest externally alongside the run. The wrapper resolves tf2's installed package.xml, records the compiled executable hash, and keeps native stdout/stderr. The baseline wrapper requires native tf2Scalar to be double; a float build needs distinct native/message precision handling rather than silent widening.

No ROS daemon, live transform broadcaster or robot is needed: the executable inserts a deterministic static frame chain directly into BufferCore. It captures geometry_msgs conversion, not serialized DDS/CDR bytes. Interpret lookupTransform(target, source, time) as the transform acting on source coordinates to produce target coordinates; verify both directions and nonzero translations.

## 6. Other prepared priorities

In separately pinned environments with the actual native libraries available:

```bash
python -m exporters.capture_spice --output runs/local-cspice
python -m exporters.capture_mujoco --output runs/local-mujoco
```

Choose and freeze SpiceyPy/CSPICE and MuJoCo versions before running; none are claimed installed by this delivery. The exporters were syntax-checked only. CSPICE capture covers products, matrix conversions and axis-angle; Euler and qdq2av remain pending. MuJoCo capture covers derivatives, integration, differences and native integration Jacobians. No rate-frame claim is promoted until its native results are analyzed.

## 7. Promotion

Preserve raw values, signs, bits, stdout, stderr, dependency identities and hashes. Validate manifest/observation schemas. Add operation-specific independent comparisons. Separate native inverse quirks from forward accuracy. Then integrate QuatOps replay and record representation coverage independently. Only after review should a subset become accepted_for_scope.
