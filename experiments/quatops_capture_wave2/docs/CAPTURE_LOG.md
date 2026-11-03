# Capture ledger — Wave 2

The previous plan remains the baseline. This is an additive implementation pass, not a claim that the existing Unreal/Unity/Godot evidence has been audited. No repository changes or QuatOps replay were performed.

## Requested targets

| ID | Priority now | Target | Actual state | Next gate |
|---|---|---|---|---|
| CAP-100 | P1 user-selected | ROS 2 / tf2 | C++ package and Python runner written; C++ compilation pending | Build in a pinned ROS environment, record actual tf2/package identities, execute native frame chain and inspect directions |
| CAP-095 | P1 user-selected | Cesium | JavaScript exporter written and syntax-checked; no Cesium package/runtime run | Resolve npm lockfile, execute local-HPR and geographic profiles, compare against supporting PROJ/reference data |
| CAP-120 | P1 user-selected | OpenUSD | Python exporter and two USDA input stages written; no pxr execution | Open stages natively, capture Gf float/double and UsdGeom operation stacks, confirm row-action and op order |
| CAP-110 | P1 user-selected | glTF / GLB | Six scenarios in 12 assets; VTK and Trimesh captured; comparison pass | Run Khronos validator and prepared Three.js loader; integrate frozen outputs into QuatOps |

## Supporting and other priorities

| ID/profile | State | Scope and remaining work |
|---|---|---|
| CAP-080/OpenCV Rodrigues | Captured; reference checked; not accepted for QuatOps | 264 observations; preserve 26 inverse-boundary diagnostics; C++ cv::Quat exp/log/trig remains pending |
| CAP-096/PROJ geographic support | Captured; reference checked; not a Cesium oracle run | 11 locations with ECEF/ENU points; compare conventions only after Cesium executes, especially exact poles |
| CAP-090/Three.js | Exporter syntax-checked | Six Euler orders, multiplication, matrices, GLTFLoader and sampled scene transforms; npm resolution/run pending |
| CAP-050/CSPICE | Exporter syntax-checked | Product, q2m/m2q, axisar/raxisa subset; native execution, Euler and qdq2av still pending |
| CAP-070/MuJoCo | Exporter syntax-checked | Quaternion derivative/integration/difference and native integration Jacobians; native execution pending |
| CAP-040/Blender | Still planned | No new Blender capture in this bundle |
| CAP-060/061 Eigen/Sophus | Still planned | No headers/build/runtime supplied or executed |
| CAP-010/020/030 engine extensions | Existing reported baseline unchanged | No new Unreal, Unity, Godot run or artifact audit |
| CAP-130/OpenXR | Still planned | Pose-contract work; no runtime/device data |
| CAP-140/reference audit | Partial supporting work only | SciPy used for this comparison stage; no current QuatOps tests or numpy-quaternion audit |

## Native observations, not multiplied by representations

| Run directory | Selected/accounted case IDs | Executed case IDs | Successful observations | Not run |
|---|---:|---:|---:|---:|
| opencv-4.13.0-first-pass | 135 | 132 | 264 | 6 |
| vtk-9.6.2-gltf | 12 asset encodings | 12 | 112 | 0 |
| trimesh-4.11.1-static | 12 asset encodings | 12 | 12 | 0 |
| proj-geographic | 11 | 11 | 11 | 0 |
| **Total observations** | Do not sum as unique shared cases | — | **399** | **6** |

The legacy manifest field `independent_input_count` counts selected case IDs, including excluded OpenCV inputs and alternative asset encodings. The table above clarifies executed inputs. The 12 assets represent six scenarios; VTK and Trimesh loading the same asset are distinct consumer observations, not different inputs.

## Completion policy

`exporter_written` / `syntax_checked` is not `native_captured`. `native_captured` is not `accepted_for_scope`. Promotion requires native binding review, explicit precision/frame conventions, raw integrity, independent comparisons, discrepancy resolution or explicit scope restrictions, and QuatOps replay across relevant representations.

Native manifests are immutable observations from the initial capture. This separate ledger records later checks without rewriting their original `captured_not_reviewed` status. No task is auto-promoted because its documentation sounds complete.

## Next execution batch

1. Run the prepared ROS/tf2, Cesium and OpenUSD exporters on environments containing their native dependencies.
2. Run the Khronos validator and Three.js loader on the exact bundled assets; retain npm lockfile and package integrity.
3. Review OpenCV inverse-boundary diagnostics without changing the forward accuracy policy.
4. Integrate validated observations into QuatOps and record representation replay counts separately.
5. Execute the prepared CSPICE/MuJoCo subsets; then continue Blender and Eigen/Sophus when their environments are available.
