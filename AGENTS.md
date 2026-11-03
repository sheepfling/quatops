# Repository agent guidance

When working on engine coordinate conversions, game-library adapters, or
rotation trajectories, read
[`docs/contributing/ENGINE_AGENT_GUIDE.md`](docs/contributing/ENGINE_AGENT_GUIDE.md)
before changing code or making convention claims. Keep user-facing engine API
facts on their pages under `docs/engines/` and shared math in
`coordinate-mapping.md`.

The source-frame contract for this work is an explicit right-handed frame.
Never infer its forward/up directions, the target asset's model-forward axis,
or whether a stored transform is active or passive from an engine name alone.
Report SciPy formula checks separately from tests run inside a pinned engine
runtime.
