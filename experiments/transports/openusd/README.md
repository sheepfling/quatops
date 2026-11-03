# OpenUSD transport oracle

This target records `Gf` quaternion math and `UsdGeomXformable` transform-stack
evaluation from a pinned OpenUSD Python distribution. Each release owns its
environment, input stage, exporter, and frozen native observations.

- [26.11](26.11/README.md): `Quatf`/`Quatd`, `Gf.Rotation`, and a saved USDA
  hierarchy with orient/rotate/translate ops.
