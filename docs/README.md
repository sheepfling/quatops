# Documentation map

Start with the [project README](../README.md) for installation, a backend
chooser, and short examples. Use the [API contract](API.md) as the source of
truth for current behavior and public signatures.

The verification and experiment guides are for contributors and researchers;
package users can stay with the README and API contract.

## For contributors and coding agents

1. Read the README and choose a fully specified convention and representation
   module before writing an example or changing an operation.
2. Read the relevant API section before changing mathematical behavior.
3. Set up the local environment and run the checks listed in
   [Verification](VERIFICATION.md). Use the repository `.venv`.
4. Use [the experiment guide](../experiments/README.md) for runnable scripts.

## Reference and research

- [Interoperability evidence atlas](INTEROPERABILITY_EVIDENCE.md): consolidated
  conventions, native engine/oracle captures, transport/loader results, proof
  layers, and remaining gaps.
- [API contract](API.md): conventions, storage, public operations, validation,
  numerical behavior, and known limits. This is the current contract.
- [Game-engine references](engines/README.md): separate notes for Unreal,
  Unity, Godot, and Blender Euler APIs and coordinate-basis conversions.
- [External math oracles](oracles/README.md): pinned Three.js, CSPICE, and
  MuJoCo runtimes, separate from QuatOps' implementation and test dependencies.
- [Transport proofs](transports/README.md): format contracts and named loader
  observations, including glTF animations through Three.js and Blender and
  OpenUSD transform stacks.
- [Verification](VERIFICATION.md): setup commands, recorded test evidence, and
  oracle coverage. Rerun its commands to verify a newer checkout.
- [Oracle coverage](ORACLE_COVERAGE.md): completed coverage groups and their
  corresponding test files.
- [Rotation precision investigation](ROTATION_NUMERICS.md): known numerical
  limits and measurements from a historical investigation.
- [Homogeneous rotation experiment](HOMOGENEOUS_INVESTIGATION.md): rejected
  research prototypes and why they were not promoted into the library.
- [Changelog](../CHANGELOG.md): alpha release contents and compatibility note.
