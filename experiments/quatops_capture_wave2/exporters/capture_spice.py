"""Prepared CSPICE-through-SpiceyPy baseline: real native calls, no mission kernels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import spiceypy as spice
from spiceypy.utils.libspicehelper import libspice

from shared.capture_io import Capture, Observation, contract, numeric, parser


def main() -> None:
    args=parser(__doc__ or "CSPICE capture").parse_args()
    corpus=json.loads(args.corpus.read_text(encoding="utf-8"))
    library=Path(str(libspice._name)).resolve()
    cap=Capture("cspice","CSPICE via SpiceyPy",spice.tkvrsn("TOOLKIT"),args.corpus,args.output,Path(__file__),
        {"input":"binary64", "arithmetic":"SpiceDouble", "output":"binary64", "serialized":"JSON plus IEEE-754 bits"},
        oracle_family="cspice",dependencies=[{"name":"SpiceyPy","version":spice.__version__}],native_binary=library,
        limitations=["Baseline only: qxq/q2m/m2q/axisar/raxisa.","Euler and qdq2av frame-derivative profiles remain pending.",
            "NASA/JPL provenance does not change the native SPICE/Hamilton product into negative-cross-product algebra."])
    native_contract=contract(order="wxyz",frame="SPICE native matrix numeric action; application frame labels require a separate profile")
    for item in corpus["multiplication_pairs"]:
        a,b=np.asarray(item["a_wxyz"]),np.asarray(item["b_wxyz"])
        cap.add(Observation(item["case_id"],"qxq-both-orders","spiceypy.qxq -> CSPICE qxq_c",native_contract,
            {"a_wxyz":numeric(a.tolist()),"b_wxyz":numeric(b.tolist())},
            {"ab_wxyz":numeric(spice.qxq(a,b).tolist()),"ba_wxyz":numeric(spice.qxq(b,a).tolist())}))
    for item in corpus["quaternion_inputs"]:
        if item["domain"]!="unit_rotation":
            continue
        q=np.asarray(item["wxyz"],dtype=np.float64);m=spice.q2m(q)
        cap.add(Observation(item["case_id"],"q2m-m2q","spiceypy.q2m; spiceypy.m2q",native_contract,
            {"quaternion_wxyz":numeric(q.tolist())},
            {"matrix_rows":numeric(m.tolist()),"readback_wxyz":numeric(spice.m2q(m).tolist()),
             "basis_and_probe_images":numeric([spice.mxv(m,v).tolist() for v in corpus["vectors_xyz"]])}))
    for item in corpus["matrix_inputs"]:
        if item["domain"]!="proper_rotation":
            continue
        m=np.asarray(item["rows"],dtype=np.float64);q=spice.m2q(m);axis,angle=spice.raxisa(m)
        cap.add(Observation(item["case_id"],"independent-matrix","spiceypy.m2q; spiceypy.raxisa",native_contract,
            {"matrix_rows":numeric(m.tolist())},
            {"quaternion_wxyz":numeric(q.tolist()),"axis_xyz":numeric(axis.tolist()),"angle_radians":numeric(angle)}))
    for item in corpus["axis_angle_inputs"]:
        m=spice.axisar(item["axis_xyz"],item["angle_radians"])
        cap.add(Observation(item["case_id"],"axisar","spiceypy.axisar",native_contract,
            {"axis_xyz":numeric(item["axis_xyz"]),"angle_radians":numeric(item["angle_radians"])},
            {"matrix_rows":numeric(m.tolist())}))
    print(cap.finish(["ALGEBRA","ACTION","MATRIX","AXIS_ROTVEC"]))


if __name__ == "__main__":
    main()
