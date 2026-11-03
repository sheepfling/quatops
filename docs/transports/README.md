# Rotation transports

Transport fixtures distinguish a file-format contract from behavior in a
named loader. The source assets, spec validation report, loader runs and
comparison tests all have separate identities.

The cross-engine/oracle/transport conventions and proof status are summarized
in the [interoperability evidence atlas](../INTEROPERABILITY_EVIDENCE.md).

- [glTF 2.0](gltf.md): frozen TRS/matrix hierarchy and rotation animation
  loaded by Three.js 0.187.0, Blender 5.1.2, VTK 9.6.2, and Trimesh 4.11.1;
  Khronos validation is recorded separately.
- [OpenUSD 26.11](openusd.md): `Gf` quaternion behavior and a saved
  `UsdGeom` transform stack.
