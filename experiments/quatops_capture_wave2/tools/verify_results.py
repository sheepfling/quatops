"""Separate verification stage: SciPy and glTF-spec calculations, never native goldens."""
from __future__ import annotations

import base64
from collections import Counter
import json
from pathlib import Path
import struct
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import scipy

from shared.capture_io import ROOT, write_json


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def value(blob: dict[str, Any]) -> np.ndarray:
    return np.asarray(blob["values"], dtype=np.float64)


def asset_header(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.suffix == ".gltf":
        doc = json.loads(path.read_text(encoding="utf-8"))
        binary = base64.b64decode(doc["buffers"][0]["uri"].split(",", 1)[1])
        return doc, binary
    raw = path.read_bytes()
    magic, version, total = struct.unpack_from("<III", raw)
    if (magic, version, total) != (0x46546C67, 2, len(raw)):
        raise ValueError("Invalid GLB header")
    length, kind = struct.unpack_from("<II", raw, 12)
    if kind != 0x4E4F534A:
        raise ValueError("Missing JSON chunk")
    doc = json.loads(raw[20:20+length])
    bin_length, bin_kind = struct.unpack_from("<II", raw, 20+length)
    if bin_kind != 0x004E4942:
        raise ValueError("Missing BIN chunk")
    binary = raw[28+length:28+length+bin_length]
    return doc, binary


def accessor(doc: dict[str, Any], binary: bytes, index: int) -> np.ndarray:
    access = doc["accessors"][index]
    view = doc["bufferViews"][access["bufferView"]]
    width = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[access["type"]]
    dtype = {5126: "<f4", 5123: "<u2"}[access["componentType"]]
    start = view.get("byteOffset", 0) + access.get("byteOffset", 0)
    return np.frombuffer(binary, dtype=dtype, count=access["count"]*width, offset=start).reshape(-1, width).astype(np.float64)


def animated_quaternion(doc: dict[str, Any], binary: bytes, time: float) -> np.ndarray:
    sampler = doc["animations"][0]["samplers"][0]
    times = accessor(doc, binary, sampler["input"]).ravel()
    out = accessor(doc, binary, sampler["output"])
    mode = sampler["interpolation"]
    i = max(0, min(int(np.searchsorted(times, time, side="right"))-1, len(times)-2))
    if mode == "CUBICSPLINE":
        packed = out.reshape(len(times), 3, 4)
        if time <= times[0]:
            q = packed[0, 1]
        elif time >= times[-1]:
            q = packed[-1, 1]
        else:
            dt = times[i+1]-times[i]
            u = (time-times[i])/dt
            q = ((2*u**3-3*u*u+1)*packed[i,1] + (u**3-2*u*u+u)*dt*packed[i,2]
                + (-2*u**3+3*u*u)*packed[i+1,1] + (u**3-u*u)*dt*packed[i+1,0])
        return q / np.linalg.norm(q)
    if mode == "STEP":
        return out[min(int(np.searchsorted(times, time, side="right"))-1, len(times)-1)]
    return Slerp(times, Rotation.from_quat(out))([min(max(time, times[0]), times[-1])]).as_quat()[0]


def node_local(node: dict[str, Any], quaternion: np.ndarray | None = None) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4,4,order="F")
    matrix = np.eye(4)
    q = quaternion if quaternion is not None else np.asarray(node.get("rotation", [0,0,0,1]))
    matrix[:3,:3] = Rotation.from_quat(q).as_matrix() @ np.diag(node.get("scale", [1,1,1]))
    matrix[:3,3] = node.get("translation", [0,0,0])
    return matrix


def expected_world(asset: Path, time: float | None = None) -> list[np.ndarray]:
    doc, binary = asset_header(asset)
    parent = node_local(doc["nodes"][0])
    result = []
    for index in doc["nodes"][0]["children"]:
        q = animated_quaternion(doc,binary,time) if time is not None and index == 1 and doc.get("animations") else None
        result.append(parent @ node_local(doc["nodes"][index],q))
    return result


def verify_opencv() -> dict[str, Any]:
    records = rows(ROOT / "runs/opencv-4.13.0-first-pass/observations.jsonl")
    maxima: dict[str,float] = {}
    discrepancies = []
    for rec in records:
        if rec["status"] != "ok":
            continue
        inputs, outputs = rec["native_inputs"], rec["native_outputs"]
        op = rec["operation_id"]
        precision = op.rsplit("-",1)[-1]
        tol = 1e-6 if precision == "binary32" else 1e-12
        if "rotation_vector" in inputs:
            rv = value(inputs["rotation_vector"]).ravel()
            reference = Rotation.from_rotvec(rv).as_matrix()
            error = float(np.max(np.abs(value(outputs["matrix_rows"])-reference)))
            key = f"forward_matrix_{precision}"
            maxima[key] = max(maxima.get(key,0),error)
            if error > tol:
                discrepancies.append({"case_id":rec["case_id"],"operation":key,"max_abs_error":error})
            native_initial = value(outputs["matrix_rows"])
        else:
            native_initial = value(inputs["matrix_rows"])
        residual = float(np.max(np.abs(value(outputs["reconstructed_matrix_rows"])-native_initial)))
        key = f"native_inverse_roundtrip_{precision}"
        maxima[key] = max(maxima.get(key,0),residual)
        if residual > tol:
            discrepancies.append({"case_id":rec["case_id"],"operation":key,"max_abs_error":residual,
                "classification":"native inverse/branch residual; not a forward-conversion tolerance override"})
    derivative = []
    subset = [r for r in records if r["case_id"].startswith("rodrigues-random") and r["operation_id"].endswith("binary64")][:6]
    for rec in subset:
        rv = value(rec["native_inputs"]["rotation_vector"]).ravel()
        jac = value(rec["native_outputs"]["jacobian_native"]).T
        errors = []
        for step in (1e-4,1e-5,1e-6):
            columns = []
            for axis in np.eye(3):
                plus = Rotation.from_rotvec(rv+step*axis).as_matrix().ravel()
                minus = Rotation.from_rotvec(rv-step*axis).as_matrix().ravel()
                columns.append((plus-minus)/(2*step))
            errors.append({"step":step,"max_abs_error":float(np.max(np.abs(jac-np.stack(columns,axis=1))))})
        derivative.append({"case_id":rec["case_id"],"input_dimensions":3,"output_dimensions":9,
            "native_layout":"3x9 transposed for 9x3 row-major flattened matrix derivative", "steps":errors})
    return {"ok_observations":sum(r["status"]=="ok" for r in records),"maxima":maxima,
            "discrepancies":discrepancies,"jacobian_checks":derivative}


def verify_gltf() -> dict[str, Any]:
    checked = []
    for backend, folder in (("trimesh","trimesh-4.11.1-static"),("vtk","vtk-9.6.2-gltf")):
        for rec in rows(ROOT / f"runs/{folder}/observations.jsonl"):
            inputs, outputs = rec["native_inputs"],rec["native_outputs"]
            time = float(value(inputs["time_seconds"])) if "time_seconds" in inputs else None
            expected = expected_world(ROOT / "assets/gltf" / inputs["asset"],time)
            if backend == "vtk":
                actual = [value(a["matrix_rows"]) for a in outputs["actors"]]
                # Static fixture has two geometrically equivalent children; do not invent actor->node names.
                errors = [min(float(np.max(np.abs(m-e))) for e in expected) for m in actual]
            else:
                actual = [value(n["world_matrix_rows"]) for n in outputs["nodes"] if n["node_name"] in {"Probe","ProbeMatrix"}]
                errors = [min(float(np.max(np.abs(m-e))) for e in expected) for m in actual]
            if len(actual) != len(expected):
                raise AssertionError(f"Actor/node count mismatch in {inputs['asset']}")
            checked.append({"consumer":backend,"asset":inputs["asset"],"operation":rec["operation_id"],
                "time_seconds":time,"max_abs_matrix_error":max(errors),"pass":max(errors)<1e-6})
    return {"observation_count":len(checked),"max_abs_matrix_error":max(r["max_abs_matrix_error"] for r in checked),
        "pass_count":sum(r["pass"] for r in checked),"failure_count":sum(not r["pass"] for r in checked),"checks":checked}


def verify_proj() -> dict[str, Any]:
    checks=[]
    for rec in rows(ROOT / "runs/proj-geographic/observations.jsonl"):
        inputs,outputs=rec["native_inputs"],rec["native_outputs"]
        lon=np.deg2rad(float(value(inputs["longitude_degrees"])))
        lat=np.deg2rad(float(value(inputs["latitude_degrees"])))
        h=float(value(inputs["height_m"]))
        a=float(value(inputs["ellipsoid_semi_major_m"]))
        f=1/float(value(inputs["ellipsoid_inverse_flattening"]))
        e2=f*(2-f);radius=a/np.sqrt(1-e2*np.sin(lat)**2)
        origin=np.array([(radius+h)*np.cos(lat)*np.cos(lon),(radius+h)*np.cos(lat)*np.sin(lon),(radius*(1-e2)+h)*np.sin(lat)])
        enu=np.array([[-np.sin(lon),-np.sin(lat)*np.cos(lon),np.cos(lat)*np.cos(lon)],
                      [np.cos(lon),-np.sin(lat)*np.sin(lon),np.cos(lat)*np.sin(lon)],
                      [0,np.cos(lat),np.sin(lat)]])
        expected=value(inputs["local_probe_points"])@enu.T+origin
        residual=float(np.max(np.abs(value(outputs["ecef_probe_points"])-expected)))
        roundtrip=float(np.max(np.abs(value(outputs["local_readback_points"])-value(inputs["local_probe_points"]))))
        checks.append({"case_id":rec["case_id"],"analytic_point_max_abs_error_m":residual,
                       "roundtrip_max_abs_error_m":roundtrip,"pass":residual<1e-7 and roundtrip<1e-7})
    return {"native_location_count":len(checks),"checks":checks,
            "max_abs_error_m":max(x["analytic_point_max_abs_error_m"] for x in checks),
            "max_roundtrip_error_m":max(x["roundtrip_max_abs_error_m"] for x in checks),
            "cesium_executed":False}


def main() -> None:
    report = {"evidence_kind":"derived_reference", "quatops_executed":False,
        "references":{"scipy":scipy.__version__,"numpy":np.__version__,"gltf":"2.0 normalized component Hermite for cubic; explicit source inputs"},
        "opencv":verify_opencv(),"gltf":verify_gltf(),"proj":verify_proj()}
    write_json(ROOT / "verification/comparison_report.json",report)
    print(json.dumps({"opencv_maxima":report["opencv"]["maxima"],"opencv_discrepancies":report["opencv"]["discrepancies"],
        "gltf_max":report["gltf"]["max_abs_matrix_error"],"gltf_failures":report["gltf"]["failure_count"]},indent=2))


if __name__ == "__main__":
    main()
