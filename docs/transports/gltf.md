# glTF 2.0 rotation transport proof

## Evidence set

The initial asset and named loader runs are indexed under
[`experiments/transports/gltf/2.0.0`](../../experiments/transports/gltf/2.0.0/README.md).
Wave 2 adds six scenarios as twelve paired `.gltf`/`.glb` assets: a static
hierarchy, `STEP`, `LINEAR`, `CUBICSPLINE`, globally negated cubic keys and
tangents, and one locally negated cubic key and tangents. Cubic inputs use
nonuniform times and nonzero tangents. The original and Wave 2 assets remain
separate inputs.

The original asset and all twelve Wave 2 files pass Khronos glTF Validator
`2.0.0-dev.3.10` with zero reported errors, warnings, infos, or hints.
Validation checks the asset format; loader captures and numerical tests
provide separate animation evidence. Wave 2's per-asset validator reports are
in [the versioned fixture directory](../../tests/golden/transports/gltf/2.0.0/khronos-validator/2.0.0-dev.3.10/wave2/).

## Three.js GLTFLoader 0.187.0

The named Three.js Node capture agrees with the glTF 2.0 animation equations:
`STEP` holds the previous key until the next key, rotational `LINEAR` uses
shortest-path SLERP, and `CUBICSPLINE` evaluates the time-scaled Hermite
components then normalizes the quaternion. The Khronos specification says
rotation `LINEAR` should use SLERP and requires the sampled rotation result to
be normalized; it defines cubic key data as in-tangent, value, and out-tangent
triples ([animation sampler rules](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#reference-animation-sampler)).

The capture records Node/V8 and Three.js versions, original asset hashes,
adapted in-memory JSON hash, lockfile/exporter hashes, matrix values and all
sampled coefficients. The external `.bin` bytes are unchanged when embedded as
a data URI for Node's loader.

The Wave 2 Three.js 0.187.0 run adds all six Euler orders and imports all twelve
new assets. It records 12 static scene observations plus 100 animation samples
over the five animated scenarios and both encodings. An independent TRS,
STEP/SLERP, and normalized cubic-Hermite evaluator agrees with all 112 loader
samples to a maximum matrix-element difference of `3.43e-8`. Globally negating
all cubic keys/tangents preserves the measured trajectory; negating only one
key/tangent pair changes it. The native output and manifest are under
[`wave2`](../../tests/golden/transports/gltf/2.0.0/threejs/0.187.0/wave2/).

## VTK 9.6.2 and Trimesh 4.11.1

The Wave 2 VTK importer capture contains twelve static observations and 100
animation samples. The Trimesh capture is static-only and contains twelve
observations. Against the same independent matrix evaluator, the maximum
element differences are `2.40e-7` for VTK and `8.89e-16` for Trimesh. These
were native library runs in the packet's pinned Linux environment; Windows
replay checked their frozen outputs without rerunning those native libraries.
Each raw manifest and observation file is in its versioned
[VTK](../../tests/golden/transports/gltf/2.0.0/vtk/9.6.2/manifest.json) and
[Trimesh](../../tests/golden/transports/gltf/2.0.0/trimesh/4.11.1/manifest.json)
folder.

## Blender 5.1.2 importer

The Blender capture confirms this specific importer maps glTF's source
translation axes to Blender as `X -> X`, `Y -> Z`, `Z -> -Y` (determinant +1).
Its equivalent TRS and matrix nodes agree to roughly `6e-8` in world matrices.
These probes establish the loader's frame map for this asset; they do not
establish an imported model's semantic forward axis.

The captured `LINEAR` quaternion curves interpolate components and leave the
midpoint coefficients non-unit (norms about `0.9520` and `0.8290`). The
`CUBICSPLINE` data is reduced to key values and represented as automatic Bezier
F-curves; its midpoints differ from the glTF Hermite evaluation. A subframe
probe 1e-6 seconds before a `STEP` key also evaluates to the next key, whereas a
probe 1e-5 seconds before it retains the previous key. These are Blender
5.1.2 importer/evaluator observations and are preserved as such, rather than
being blended into the Three.js or format-level claims. The exact Blender
runtime and importer source hashes are in its
[frozen capture](../../tests/golden/transports/gltf/2.0.0/blender/5.1.2/native_rotation.json).

## Boundaries

These captures cover the named loaders and these deterministic assets. They do
not assert behavior for every application, compression extension, asset
import setting, animation mixer, model-forward convention, renderer, or full
glTF conformance beyond the recorded validator reports.
