# Three.js oracle

Three.js is captured as a versioned math-library oracle. The Node exporter has
no renderer, browser, GPU, QuatOps, NumPy, or SciPy dependency. Each pinned
release lives under `threejs/<revision>/`; transport loader captures remain a
separate target even when they use the same package.

## Captured release

- [0.187.0](0.187.0/README.md): Euler orders and boundaries, quaternion
  algebra/normalization, axis-angle, independent matrices, vector action,
  shortest SLERP, two-vector alignment, Euler reorder, and Object3D hierarchy.

See the version page for the command and known API limits. The dataset is
native JavaScript `Number` output and does not represent a float32 runtime.
