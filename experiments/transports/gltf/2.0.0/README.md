# glTF 2.0 transport fixtures

The assets exercise independent TRS and matrix transform forms, parent/child
composition, and `STEP`, `LINEAR`, and `CUBICSPLINE` quaternion animation on
nonuniform key times. The spec does not itself run a sampler; each named loader
has a distinct version directory and frozen observations.

## Frozen asset

- [`rotation_transport.gltf`](assets/rotation_transport.gltf)
- [`rotation_transport.bin`](assets/rotation_transport.bin)
- [`build_asset.py`](assets/build_asset.py) rebuilds the two source files using
  Python's standard library only. It imports no QuatOps, SciPy, or NumPy code.

## Loaders

- [Three.js GLTFLoader 0.187.0](threejs/0.187.0/README.md)
- [Blender glTF importer 5.1.2](blender/5.1.2/README.md)
- VTK 9.6.2 importer ([capture](vtk/9.6.2/README.md))
- Trimesh 4.11.1 importer ([capture](trimesh/4.11.1/README.md))

The Three.js and Blender records include the original authored transport
asset. Wave 2 adds a separate twelve-file asset corpus and captures for
Three.js, VTK, and Trimesh; see the
[Wave 2 replay note](../../../quatops_capture_wave2/LOCAL_REPLAY.md).
