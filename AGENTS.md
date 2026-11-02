# Repository agent guidance

When working on engine coordinate conversions, game-library adapters, or
rotation trajectories, read [`docs/engines/AGENT_GUIDE.md`](docs/engines/AGENT_GUIDE.md)
before changing code or making convention claims. Keep each engine's API facts
on its own page under `docs/engines/` and the shared math in
`coordinate-mapping.md`.

The source-frame contract for this work is an explicit right-handed frame.
Never infer its forward/up directions, the target asset's model-forward axis,
or whether a stored transform is active or passive from an engine name alone.
Report SciPy formula checks separately from tests run inside a pinned engine
runtime.
