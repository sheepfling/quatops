# Findings and scope

## Executed glTF consumers

Twelve self-contained assets were actually imported by VTK 9.6.2 and Trimesh 4.11.1. VTK additionally sampled five animation scenarios in each encoding at ten absolute times: 100 samples. VTK supplied 12 static and 100 animated observations; Trimesh supplied 12 static scene-graph observations.

The 124 observations pass independent matrix comparisons. VTK maximum absolute element error is `2.399441831335558e-7`; Trimesh is `8.881784197001252e-16`. VTK exposes matrices through double accessors, but that does not prove double arithmetic for its animation internals. glTF animation buffers are binary32.

The assets test a translated, rotated parent; equivalent child TRS and independently authored matrix forms; STEP and LINEAR rotation interpolation; and normalized component-Hermite CUBICSPLINE with nonuniform times and nonzero tangents. Globally negating every key and tangent preserves the orientation trajectory. Negating only the middle key and its tangents intentionally changes the cubic trajectory. The reference evaluator preserves these signs instead of canonicalizing them away.

VTK ran in an offscreen render-window context without calling Render(). Its log contains an unavailable X-server warning; imports and sampling completed. This is not renderer, shader, GPU, image, or visual validation. A native actor's matrix was observed, not an invented node-to-actor name mapping. Static equivalent children are compared as an unordered set. Trimesh is static-only in this capture.

Trimesh's bundled JSON-schema validator accepted the inputs. **The Khronos glTF Validator has not run.** Neither schema validation nor these selected assets establish general glTF conformance. Prepared Three.js GLTFLoader observations are not yet available.

## OpenCV Rodrigues

264 observations: 124 vector inputs and eight independent matrices in each of binary32 and binary64. Three invalid matrices per precision were deliberately excluded and recorded as six `not_run` rows; they are not failures or passes.

Forward matrices agree with independent SciPy conversion at max element errors `2.979167001981864e-8` (binary32) and `4.440892098500626e-16` (binary64). Six random binary64 forward Jacobians were checked using central differences at 1e-4, 1e-5 and 1e-6. The largest best-step error is `2.6320778889754592e-11`. The native 3x9 Jacobian is transposed for comparison with a 9x3 derivative of the row-major flattened matrix. It is not an SO(3) tangent Jacobian.

The native matrix->rotation-vector->matrix reconstruction has **26 binary64 boundary diagnostics** above the 1e-12 diagnostic threshold, around zero and pi. Maximum element residual is approximately 1e-6. This is retained as a native inverse/branch behavior observation, not a diagnosis of a bug, not a QuatOps failure, and not grounds to weaken all conversion tolerances. The float32 maximum residual is also about 1e-6; the selected binary32 threshold is less stringent.

The residual is a matrix-element difference, not a quoted angular error. Inspect exact input bits and native returned vectors in the JSONL before deciding whether QuatOps should imitate any branch. General quaternion exp/log/trig via C++ cv::Quat is still unimplemented here.

## Cesium preparation and PROJ supporting evidence

Cesium's local heading/pitch/roll contract uses heading about negative Z, pitch about negative Y, and roll about positive X, in radians. The local active matrix to investigate is Rz(-heading) Ry(-pitch) Rx(roll). That is distinct from placing an orientation on the globe: the default local east/north/up frame is then mapped into Earth-fixed coordinates.

The Cesium exporter records local Quaternion.fromHeadingPitchRoll separately from Transforms.headingPitchRollQuaternion, the ENU-to-ECEF matrix, and the model-to-ECEF transform. It captures points and directions separately so the ECEF translation cannot be accidentally applied to a direction. Zero local HPR should not be interpreted as an identity quaternion in the global ECEF basis.

The PROJ run is supporting evidence, **not a Cesium run**. Eleven geographic locations cover equatorial, oblique, high-altitude, antimeridian, near-pole and exact-pole cases. The native WGS84/ECEF and inverse topocentric transforms agree with separate analytic evaluation to at most `9.313225746154785e-10 m`; native local-coordinate round trips have at most `6.343396918850885e-10 m` residual. This measures numerical consistency, not surveying accuracy.

At an exact pole geographic east is not uniquely defined without an additional convention. PROJ's longitude-defined local orientation and Cesium's native fallback must not be assumed equal. Capture and classify, rather than changing one library's data until it matches.

## ROS and OpenUSD: preparation, not results

ROS source uses setRPY, native quaternion/matrix/action APIs, geometry_msgs field conversion, and a deterministic static BufferCore world->base->sensor chain. Both target-from-source and reverse lookups are included. There is no DDS, serialized CDR wire message, live robot, dynamic timestamp interpolation or ROS network claim. C++ compilation is pending.

OpenUSD source covers Gf.Quatf/Quatd, matrix extraction, vector action, ordered rotate ops and stage evaluation. The two authored USDA files differ only in upAxis metadata. They exercise orient versus rotate operations, reordered stacks, resetXformStack, a pivot/inverse-pivot pair, and orientation time samples. Gf's native row-vector matrices are retained; comparison adapters transpose only where an explicit column-action comparison requires it. Metadata is not silently turned into a basis conversion. Native parsing, precision behavior and outputs remain unverified until pxr runs.

## Independence and remaining uncertainty

No exporter imports QuatOps or SciPy; mathematical comparisons live in tools/verify_results.py. Native outputs remain raw with coefficient signs, dtypes and IEEE bits. The glTF assets are authored inputs, not supposed native answers. VTK and Trimesh are distinct consumers; the planned Three.js loader is not an independent implementation from Three.js's own math library. PROJ and Cesium must likewise remain separately identified.

No QuatOps adapter, test, repository branch, or installed engine from the previous reports was run in this wave. The packet is implementation-ready evidence and exporter work, not a declaration of product compatibility.
