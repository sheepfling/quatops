# Blender native captures

The Blender capture uses a versioned directory and writes its native runtime
output under the matching version in `tests/golden/blender/`. Keep every
Blender build in its own sibling folder.

- [Blender 5.1.2](5.1.2/README.md): 522 all-order Euler captures, quaternion
  and matrix conversions, axis-angle and exponential-map routes, SLERP,
  rotation difference, swing-twist, Euler continuity trajectories, and scene
  object rotation modes.

These scripts are auxiliary evidence. They do not add a Blender dependency to
QuatOps or run inside the package test process.
