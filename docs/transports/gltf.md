# glTF 2.0 rotation transport evidence

## Retained transport asset and loader observations

The retained source asset exercises independent TRS and matrix transforms,
parent/child composition, and `STEP`, `LINEAR`, and `CUBICSPLINE` rotation
animation on nonuniform key times. Its standard-library-only builder,
`.gltf`, and binary buffer are indexed in the
[versioned transport package](../../experiments/transports/gltf/2.0.0/README.md).
The Khronos validator reports no issues for this asset; validation establishes
format validity, while the named loader runs establish implementation behavior.

Three.js `GLTFLoader` `0.187.0` loads this asset in Node.js without a browser,
renderer, or GPU. Its retained capture records node matrices, matrix layout,
animation samples, and runtime/package/source identity. The asset is also
imported by Blender `5.1.2`, whose frozen capture records the importer build,
source hashes, scene matrices, F-curve profiles, selected samples, and axis
probes.

The Blender importer maps the tested glTF axes to Blender as `+X -> +X`,
`+Y -> +Z`, and `+Z -> -Y`, a proper rotation. In this pinned version,
`LINEAR` quaternion curves use component-linear F-curves, and `CUBICSPLINE`
inputs are represented by automatic Bezier curves; selected native midpoint
values differ from the glTF reference interpolation. Two step-key samples
retain a narrow subframe boundary behavior. These are Blender `5.1.2` observations and do not
change the format contract or Three.js results. Details and native values are
in the [Blender capture notes](../../experiments/transports/gltf/2.0.0/blender/5.1.2/README.md)
and [frozen output](../../tests/golden/transports/gltf/2.0.0/blender/5.1.2/native_rotation.json).

For glTF, rotational `LINEAR` interpolation is specified to use SLERP, and
`CUBICSPLINE` data uses in-tangent, value, and out-tangent triples with
normalization of the resulting quaternion. See the
[Khronos glTF 2.0 animation sampler rules](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-animation-sampler).

## QuatOps static-file converter proof

The separate accepted converter fixture contains a QuatOps-generated,
two-node `.gltf` hierarchy. Its explicit canonical frame is `+X right`,
`+Y forward`, `+Z up`; the declared glTF basis map is `(+X, +Z, -Y)`. Three.js
`GLTFLoader` `0.187.0` loads the serialized artifact. Two selected native node
witnesses preserve coefficients, local and world matrices, point and direction
actions, inverse point transforms, and noncommuting parent/child composition.

The asset, exact native observations, contract, and hashes are in the
[converter fixture](../../tests/golden/transports/gltf/2.0.0/threejs/0.187.0/converter/manifest.json).
The [capture package](../../experiments/converters/gltf/2.0.0/README.md)
retains the generator, loader script, dependency lock, and replay instructions.
Replay is covered by
[test_gltf_converter_roundtrip.py](../../tests/experiments/converters/gltf/test_gltf_converter_roundtrip.py).

## Limits

The converter proof covers static transforms for this declared basis map. It
does not claim an animation exporter, universal model-forward axis, behavior
for every loader/import setting, or full conformance beyond the retained
validator and loader evidence.
