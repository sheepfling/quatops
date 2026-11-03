"""Capture the installed VTK glTF importer, including sampled actor transforms."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import vtk
import vtkmodules.vtkIOImport as io_import

from shared.capture_io import ROOT, Capture, Observation, contract, numeric, parser, sha256


def actors(importer: Any) -> list[dict[str, Any]]:
    result = []
    collection = importer.GetImportedActors()
    for i in range(collection.GetNumberOfItems()):
        actor = collection.GetItemAsObject(i)
        matrix = actor.GetMatrix()
        mapper = actor.GetMapper()
        mapper.Update()
        data = mapper.GetInput()
        result.append({"native_collection_index": i,
            "matrix_rows": numeric([[matrix.GetElement(r,c) for c in range(4)] for r in range(4)]),
            "mapper_points": numeric([list(data.GetPoint(j)) for j in range(data.GetNumberOfPoints())]),
            "point_storage_type": data.GetPoints().GetData().GetDataTypeAsString()})
    return result


def main() -> None:
    cli = parser(__doc__ or "VTK glTF")
    cli.add_argument("--assets", type=Path, default=ROOT / "assets/gltf")
    cli.add_argument("--sampling", type=Path, default=ROOT / "assets/sampling.json")
    args = cli.parse_args()
    times = json.loads(args.sampling.read_text(encoding="utf-8"))["times_seconds"]
    cap = Capture("gltf_vtk", "VTK vtkGLTFImporter", vtk.vtkVersion.GetVTKVersion(), args.corpus, args.output,
        Path(__file__), {"input": "glTF JSON; animation/geometry FLOAT binary32 buffers",
        "arithmetic": "mixed vendor implementation; not inferred from Python float",
        "output": "vtkMatrix4x4 accessor double; animation internals may be float",
        "serialized": "binary64 accessor values with bits; original asset hash retained"},
        oracle_family="vtk", native_binary=Path(io_import.__file__),
        limitations=["No rendering or GPU output captured.",
            "Actor collection indices are retained; no unstated node-name correspondence.",
            "Animation internal precision is not promoted to float64 by accessor type.",
            "No QuatOps adapter replay."])
    cap.artifacts.append((args.sampling.resolve(), "sample_times"))
    for path in sorted(args.assets.iterdir()):
        if path.suffix not in {".gltf", ".glb"}:
            continue
        importer = vtk.vtkGLTFImporter()
        importer.SetFileName(str(path.resolve()))
        # No Render() call. The window is only an importer-owned scene container.
        window = vtk.vtkRenderWindow()
        window.SetOffScreenRendering(1)
        importer.SetRenderWindow(window)
        messages: list[str] = []
        importer.AddObserver("ErrorEvent", lambda obj, event: messages.append(str(event)))
        ok = importer.Update()
        if messages or importer.GetImportedActors().GetNumberOfItems() == 0:
            raise RuntimeError(f"VTK import failed for {path.name}: {messages}; Update={ok}")
        count = importer.GetNumberOfAnimations()
        base_input = {"asset": path.name, "sha256": sha256(path)}
        native_contract = contract(units="metres, seconds", order="asset xyzw", frame="actor local to renderer/world")
        cap.add(Observation(path.name, "gltf-static-load", "vtkGLTFImporter.Update; vtkActor.GetMatrix", native_contract,
            base_input, {"actors": actors(importer), "animation_count": count,
                         "animation_support_level": int(importer.GetAnimationSupportLevel())}))
        for animation in range(count):
            importer.EnableAnimation(animation)
            importer.SetInterpolateBetweenTimeSteps(True)
            for time in times:
                succeeded = importer.UpdateAtTimeValue(time)
                if not succeeded:
                    raise RuntimeError(f"VTK animation evaluation failed: {path.name} {time}")
                cap.add(Observation(path.name, f"gltf-animation-{animation}-t{time:.9f}",
                    "vtkGLTFImporter.EnableAnimation; UpdateAtTimeValue; vtkActor.GetMatrix", native_contract,
                    {**base_input, "animation_index": animation, "time_seconds": numeric(time),
                     "interpolate_between_time_steps": True}, {"actors": actors(importer)}))
            importer.DisableAnimation(animation)
        cap.artifacts.append((path.resolve(), "input_asset"))
        window.Finalize()
    print(cap.finish(["FORMAT", "FRAME", "MATRIX", "TRAJECTORY"]))


if __name__ == "__main__":
    main()
