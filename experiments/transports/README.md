# Rotation transport captures

Transport proof assets stay separate from the engine/library math captures.
Pin the format version, each loader/version, source assets, import options and
runtime. A format specification, validator report, and loader output are
different evidence types.

- [glTF 2.0](gltf/2.0.0/README.md): deterministic TRS/matrix and rotation
  animation assets with named loader runs.
- [OpenUSD 26.11](openusd/26.11/README.md): `Quatf`/`Quatd` math and saved
  transform-stack evaluation through `UsdGeom`.
