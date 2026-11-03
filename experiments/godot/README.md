# Godot captures

Native capture projects are versioned separately:

- [Godot 4.7 stable](4.7-stable/README.md)
- [Godot 4.8-dev1](4.8-dev1/README.md)

Both projects use the shared [`capture.gd`](capture.gd) exporter, which records
the selected quaternion converter witnesses. Each runtime has a versioned
contract under `tests/oracles/godot/<version>/`; their identical selected
outputs are stored once under `tests/oracles/godot/shared/` after both native
versions have been captured and compared.
