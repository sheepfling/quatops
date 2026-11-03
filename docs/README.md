# Documentation

## For users

- [Project README](../README.md): installation, short examples, and backend
  selection.
- [API contract](API.md): conventions, representations, layouts, operations,
  validation, and numerical limits.
- [Game-engine guides](engines/README.md): Unreal, Unity, and Godot API notes,
  coordinate maps, and quaternion trajectory conversion.
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

## Open investigation

Work that needs new discovery or native runtime evidence is tracked as GitHub
issues rather than recorded benchmark reports:

- [Extreme float64 rotation accuracy](https://github.com/sheepfling/quatops/issues/6)
- [Unreal native fixtures](https://github.com/sheepfling/quatops/issues/4)
- [Unity and Godot runtime verification](https://github.com/sheepfling/quatops/issues/7)
