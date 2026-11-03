# Godot captures

Native capture projects are versioned separately:

- [Godot 4.7 stable](4.7-stable/README.md)
- [Godot 4.8-dev1](4.8-dev1/README.md)

Both projects use the shared [`capture.gd`](capture.gd) exporter. It writes full
discovery output to a temporary path. Retain only
reviewed converter witnesses under `tests/oracles/godot/<version>/`; later
versions use sibling contracts and reuse witness files only after the runtime
outputs have been compared.
