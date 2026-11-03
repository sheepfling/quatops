# Documentation

## For users

- [Project README](../README.md): installation, short examples, and backend
  selection.
- [API contract](API.md): conventions, representations, layouts, operations,
  validation, and numerical limits.
- [Game-engine guides](engines/README.md): Unreal, Unity, Godot, and Blender
  API notes, coordinate maps, and quaternion trajectory conversion.
- [Interoperability evidence atlas](INTEROPERABILITY_EVIDENCE.md): consolidated
  engine, math-oracle, transport, and capture-layer findings with open gaps.
- [Changelog](../CHANGELOG.md): release and compatibility notes.

## For contributors

- [Verification](contributing/VERIFICATION.md): development setup, routine
  checks, and how to run the test suite.
- [Oracle coverage](contributing/ORACLE_COVERAGE.md): independent references
  and the tests that exercise them.
- [Experiment and benchmark tools](../experiments/README.md): commands,
  workloads, and local result output.
- [Engine integration workflow](contributing/ENGINE_INTEGRATIONS.md):
  implementation and validation guidance.
- [Engine agent guidance](contributing/ENGINE_AGENT_GUIDE.md): required
  convention reasoning and evidence boundaries for code changes.
- [Math-library oracles](oracles/README.md): native quaternion and coordinate APIs.
- [Rotation transports](transports/README.md): format and named-loader evidence.

## Open evidence work

Unreal, Unity, Godot, and Blender have versioned native convention and
converter witnesses. Remaining evidence gaps are narrower:

- [Extreme float64 rotation accuracy](https://github.com/sheepfling/quatops/issues/6)
- OpenXR runtime behavior, including `xrLocateSpace` and reference-space relations.
- Broader Sophus scalar, Jacobian, and derivative contracts.

See the [interoperability evidence atlas](INTEROPERABILITY_EVIDENCE.md) for
the accepted scope and explicit limits of each converter.
