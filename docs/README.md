# Documentation map

Start with the [project README](../README.md) for installation, a backend
chooser, and short examples. Use the [API contract](API.md) as the source of
truth for current behavior and public signatures.

## For contributors and coding agents

1. Read the README and choose a fully specified convention and representation
   module before writing an example or changing an operation.
2. Read the relevant API section before changing mathematical behavior.
3. Set up the local environment and run the checks listed in
   [Verification](VERIFICATION.md). Use the repository `.venv`.
4. Use [the experiment guide](../experiments/README.md) for runnable scripts.

## Reference and research

- [API contract](API.md): conventions, storage, public operations, validation,
  numerical behavior, and known limits. This is the current contract.
- [Verification](VERIFICATION.md): setup commands, recorded test evidence, and
  oracle coverage. Rerun its commands to verify a newer checkout.
- [Oracle coverage](ORACLE_COVERAGE.md): completed coverage groups and their
  corresponding test files.
- [Rotation precision investigation](ROTATION_NUMERICS.md): known numerical
  limits and measurements from a historical investigation.
- [Homogeneous rotation experiment](HOMOGENEOUS_INVESTIGATION.md): rejected
  research prototypes and why they were not promoted into the library.
- [Axis-angle and rotation-vector proposal](PROPOSED_ROTATION_REPRESENTATIONS.md):
  historical pre-implementation proposal. Do not use its old package names,
  examples, or open questions as current API guidance.
- [Changelog](../CHANGELOG.md): alpha release contents and compatibility note.
