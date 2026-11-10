# Evidence bundle

Build the review bundle with:

```powershell
uv run python -m scripts.package_evidence_bundle
```

The archive includes the maintained documentation, semantic and numerical
implementation, capture/replay scripts, manifests and frozen evidence, the
complete test tree (including `tests.support`), and packaging/validation
helpers. It is intended to replay the supported Python semantic and converter
suite from an extracted copy. It is not an engine installation image: engine
installations, `node_modules`, generated build directories, caches, and virtual
environments are excluded. Native captures are retained as immutable artifacts;
the bundle does not claim those engine runtimes were re-executed during bundle
creation.

`docs/semantic-source-audit.md` records the manual review of each promoted role's
axes, source scope, and exclusions against its cited primary document.
