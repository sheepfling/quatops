"""Prepared MuJoCo native quaternion integration/differentiation baseline."""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from shared.capture_io import Capture, Observation, contract, numeric, parser


def main() -> None:
    args=parser(__doc__ or "MuJoCo capture").parse_args()
    corpus=json.loads(args.corpus.read_text(encoding="utf-8"))
    directory=Path(mujoco.__file__).parent
    library=next(iter(directory.glob("libmujoco.so*")),None)
    cap=Capture("mujoco","MuJoCo",mujoco.mj_versionString(),args.corpus,args.output,Path(__file__),
        {"input":"NumPy binary64 buffers", "arithmetic":"vendor mjtNum build; accepted dtype recorded",
         "output":"binary64 buffers", "serialized":"JSON plus IEEE-754 bits"},
        native_binary=library,dependencies=[{"name":"mujoco Python bindings","version":mujoco.__version__}],
        limitations=["Low-level quaternion functions only; no dynamics, model or viewer.",
            "Angular-rate frame must be checked using nonidentity cases before adapter acceptance.",
            "Native mjd_* tangent Jacobians are not four-coefficient derivatives."])
    native_contract=contract(order="wxyz",frame="native rate/difference frame under investigation; do not silently label world or body")
    for item in corpus["integration_inputs"]:
        initial=np.asarray(item["initial_wxyz"],dtype=np.float64)
        velocity=np.asarray(item["omega_xyz"],dtype=np.float64)
        dt=float(item["dt_seconds"])
        q=initial.copy();derivative=np.empty(4);difference=np.empty(3)
        mujoco.mju_derivQuat(derivative,initial,velocity)
        mujoco.mju_quatIntegrate(q,velocity,dt)
        mujoco.mju_subQuat(difference,q,initial)
        dquat=np.empty((3,3));dvel=np.empty((3,3));dscale=np.empty(3)
        mujoco.mjd_quatIntegrate(velocity,dt,dquat,dvel,dscale)
        cap.add(Observation(item["case_id"],"integration-and-derivatives",
            "mju_derivQuat; mju_quatIntegrate; mju_subQuat; mjd_quatIntegrate",native_contract,
            {"initial_wxyz":numeric(initial.tolist()),"angular_velocity_xyz":numeric(velocity.tolist()),"dt_seconds":numeric(dt)},
            {"integrated_wxyz":numeric(q.tolist()),"derivative_wxyz":numeric(derivative.tolist()),"difference_xyz":numeric(difference.tolist()),
             "tangent_jacobian_initial":numeric(dquat.tolist()),"tangent_jacobian_velocity":numeric(dvel.tolist()),
             "tangent_derivative_scale":numeric(dscale.tolist())}))
    for pair in corpus["multiplication_pairs"]:
        a=np.asarray(pair["a_wxyz"],dtype=np.float64);b=np.asarray(pair["b_wxyz"],dtype=np.float64)
        ab=np.empty(4);ba=np.empty(4)
        mujoco.mju_mulQuat(ab,a,b);mujoco.mju_mulQuat(ba,b,a)
        cap.add(Observation(pair["case_id"],"products","mju_mulQuat",native_contract,
            {"a_wxyz":numeric(a.tolist()),"b_wxyz":numeric(b.tolist())},
            {"ab_wxyz":numeric(ab.tolist()),"ba_wxyz":numeric(ba.tolist())}))
    print(cap.finish(["ALGEBRA","DIFFERENTIAL"]))


if __name__ == "__main__":
    main()
