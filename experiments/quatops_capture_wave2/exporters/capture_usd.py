"""OpenUSD Gf quaternion and UsdGeom transform-stack exporter.

Prepared and syntax-checked; requires native pxr bindings before execution.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from pxr import Gf, Usd, UsdGeom

from shared.capture_io import ROOT, Capture, Observation, contract, numeric, parser


def coefficients(q: Any) -> list[float]:
    return [float(q.GetReal()), *map(float, q.GetImaginary())]


def matrix_rows(matrix: Any, size: int) -> list[list[float]]:
    return [[float(matrix[r][c]) for c in range(size)] for r in range(size)]


def main() -> None:
    cli = parser(__doc__ or "OpenUSD capture")
    cli.add_argument("--assets", type=Path, default=ROOT / "assets/usd")
    args = cli.parse_args()
    data = json.loads(args.corpus.read_text(encoding="utf-8"))
    binary = next(iter(Path(Gf.__file__).parent.glob("_gf*.so")), None)
    cap = Capture("openusd", "OpenUSD Gf and UsdGeom", ".".join(map(str, Usd.GetVersion())),
        args.corpus,args.output,Path(__file__),
        {"input":"Quatf/Quatd native coefficients; degree-based XformOps",
         "arithmetic":"Gf float/double profiles; XformCache returns Matrix4d",
         "output":"per-field binary32/binary64; extraction through Gf.Rotation explicitly double",
         "serialized":"JSON with explicit IEEE-754 bits"},native_binary=binary,
        limitations=["No renderer, asset importer, constraints, skeletons or half-precision profile.",
            "Changing upAxis metadata is tested without silently transforming coordinate values.",
            "Matrix extraction through Gf.Rotation returns a double quaternion, even for Matrix3f input.",
            "No QuatOps replay is performed by this exporter."])
    row_contract = contract(units="dimensionless",order="wxyz",action="row vectors; rows preserved",frame="native Gf numeric action")
    for label, Qt, Vt, Mt in (("binary32",Gf.Quatf,Gf.Vec3f,Gf.Matrix3f),("binary64",Gf.Quatd,Gf.Vec3d,Gf.Matrix3d)):
        for item in data["quaternion_inputs"]:
            if item["domain"] != "unit_rotation":
                continue
            w,x,y,z=item["wxyz"]
            q=Qt(w,Vt(x,y,z));m=Mt(q)
            recovered=m.ExtractRotation().GetQuat()
            cap.add(Observation(item["case_id"],f"gf-action-matrix-{label}",
                "Gf.Quatf/d; Matrix3f/d(quat); Quat.Transform; Matrix.ExtractRotation().GetQuat()",row_contract,
                {"quaternion_wxyz":numeric(coefficients(q),label)},
                {"matrix_rows":numeric(matrix_rows(m,3),label),
                 "basis_and_probe_images":numeric([list(q.Transform(Vt(*v))) for v in data["vectors_xyz"]],label),
                 "extracted_quaternion_wxyz":numeric(coefficients(recovered),"binary64")}))
        for item in data["matrix_inputs"]:
            if item["domain"] != "proper_rotation":
                continue
            # The corpus describes column action. Transpose the INPUT explicitly for Gf row action.
            source=item["rows"]
            native_rows=[[source[c][r] for c in range(3)] for r in range(3)]
            m=Mt(*[v for row in native_rows for v in row])
            q=m.ExtractRotation().GetQuat()
            cap.add(Observation(item["case_id"],f"gf-independent-matrix-{label}","Gf.Matrix3f/d.ExtractRotation().GetQuat()",row_contract,
                {"matrix_rows":numeric(matrix_rows(m,3),label),"input_adapter":"transpose corpus column-action matrix"},
                {"quaternion_wxyz":numeric(coefficients(q))}))
        for pair in data["multiplication_pairs"]:
            a,b=(Qt(v[0],Vt(*v[1:])) for v in (pair["a_wxyz"],pair["b_wxyz"]))
            cap.add(Observation(pair["case_id"],f"gf-products-{label}","Gf.Quatf/d operator*",row_contract,
                {"a_wxyz":numeric(coefficients(a),label),"b_wxyz":numeric(coefficients(b),label)},
                {"ab_wxyz":numeric(coefficients(a*b),label),"ba_wxyz":numeric(coefficients(b*a),label)}))
    stage=Usd.Stage.CreateInMemory()
    for order in data["euler_orders"]:
        prim=UsdGeom.Xform.Define(stage,"/Order"+order)
        op=prim.AddXformOp(getattr(UsdGeom.XformOp,"TypeRotate"+order),UsdGeom.XformOp.PrecisionDouble)
        for item in data["euler_inputs"]:
            degrees=Gf.Vec3d(*(math.degrees(v) for v in item["xyz"]))
            op.Set(degrees)
            m=prim.GetLocalTransformation()
            cap.add(Observation(item["case_id"],f"usd-rotate-{order}",f"UsdGeom.XformOp.TypeRotate{order}; GetLocalTransformation",
                contract(units="degrees",order="wxyz",action="row vectors",frame="local transform"),
                {"axis_labelled_xyz_degrees":numeric(list(op.Get())),"xformOpOrder":list(prim.GetXformOpOrderAttr().Get())},
                {"matrix_rows":numeric(matrix_rows(m,4)),"quaternion_wxyz":numeric(coefficients(m.ExtractRotation().GetQuat()))}))
    times=json.loads((ROOT/"assets/sampling.json").read_text())["times_seconds"]
    for path in sorted(args.assets.glob("*.usda")):
        stage=Usd.Stage.Open(str(path.resolve()))
        if not stage:
            raise RuntimeError(f"OpenUSD could not open {path}")
        cap.artifacts.append((path.resolve(),"input_stage"))
        for time in times:
            cache=UsdGeom.XformCache(Usd.TimeCode(time))
            for prim in stage.Traverse():
                if not prim.IsA(UsdGeom.Xform):
                    continue
                xform=UsdGeom.Xformable(prim)
                local=xform.GetLocalTransformation(Usd.TimeCode(time))
                world=cache.GetLocalToWorldTransform(prim)
                cap.add(Observation(f"{path.name}:{prim.GetPath()}",f"xform-stack-t{time:.9f}",
                    "Usd.Stage.Open; UsdGeom.XformCache.GetLocalToWorldTransform; Xformable.GetLocalTransformation",
                    contract(units="metres; time codes (1/sec in input)",order="wxyz",action="row vectors",frame="prim local to stage world"),
                    {"stage":path.name,"prim":str(prim.GetPath()),"time_code":numeric(time),
                     "upAxis":str(UsdGeom.GetStageUpAxis(stage)),"metersPerUnit":numeric(UsdGeom.GetStageMetersPerUnit(stage)),
                     "xformOpOrder":list(xform.GetXformOpOrderAttr().Get()),"reset_stack":xform.GetResetXformStack()},
                    {"local_matrix_rows":numeric(matrix_rows(local,4)),"world_matrix_rows":numeric(matrix_rows(world,4)),
                     "world_point":numeric(list(world.Transform(Gf.Vec3d(2,-3,5)))),
                     "world_direction":numeric(list(world.TransformDir(Gf.Vec3d(0.3,-0.2,0.7))))}))
    print(cap.finish(["ALGEBRA","EULER","MATRIX","ACTION","FRAME","FORMAT","TRAJECTORY"]))


if __name__ == "__main__":
    main()
