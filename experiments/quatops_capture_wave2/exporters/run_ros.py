"""Run the separately built native tf2 exporter and wrap its unmodified observations."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from typing import Any

from shared.capture_io import Capture, Observation, ROOT, contract, numeric, parser, sha256


def version_for(package: str) -> tuple[str, Path]:
    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        path=Path(prefix)/"share"/package/"package.xml"
        if path.is_file():
            version=ET.parse(path).getroot().findtext("version")
            if version:
                return version,path
    raise RuntimeError(f"Cannot resolve {package} package identity. Source the ROS installation first.")


def pack(value: Any, label: str) -> Any:
    if isinstance(value,dict):
        return {key:pack(item,label) for key,item in value.items()}
    if isinstance(value,list) and (not value or isinstance(value[0],(int,float,list))):
        return numeric(value,label)
    if isinstance(value,float):
        return numeric(value,label)
    return value


def main() -> None:
    cli=parser(__doc__ or "ROS capture")
    cli.add_argument("--executable",type=Path,required=True)
    args=cli.parse_args()
    version,package=version_for("tf2")
    cap=Capture("ros2_tf2","ROS 2 tf2",version,args.corpus,args.output,Path(__file__),
        {"input":"native tf2Scalar; geometry_msgs double fields", "arithmetic":"tf2Scalar as measured by compiled executable",
         "output":"native scalar and message doubles; inspect native_scalar_bytes", "serialized":"JSON plus native scalar IEEE bits"},
        dependencies=[{"name":"ROS_DISTRO","version":os.environ.get("ROS_DISTRO")},{"name":"tf2 package.xml","sha256":sha256(package)}],
        native_binary=args.executable,
        limitations=["Static same-time BufferCore chain only; no DDS/live network, interpolation or timestamp extrapolation.",
            "Message field conversion is captured, not DDS/CDR wire serialization.","No QuatOps replay."])
    run=subprocess.run([str(args.executable.resolve()),str(args.corpus.resolve())],text=True,capture_output=True,check=False)
    stdout=args.output/"native-stdout.json";stderr=args.output/"native-stderr.log"
    stdout.write_text(run.stdout);stderr.write_text(run.stderr)
    if run.returncode:
        raise RuntimeError(f"Native ROS exporter exited {run.returncode}; preserved stdout/stderr in {args.output}")
    result=json.loads(run.stdout)
    label={4:"binary32",8:"binary64"}.get(result["native_scalar_bytes"])
    if label is None:
        raise ValueError("Unknown native scalar size")
    if result["native_scalar_bytes"] != 8:
        raise ValueError("This wrapper's message bit schema currently requires double tf2Scalar; capture 32-bit/message fields separately")
    for item in result["observations"]:
        cap.add(Observation(item["case_id"],item["operation_id"],item["native_entry_point"],
            contract(order="xyzw",frame="lookup target_from_source; transform maps source point coordinates into target"),
            pack(item["native_inputs"],label),pack(item["native_outputs"],label)))
    cap.artifacts.extend([(stdout,"native_process_output"),(stderr,"native_process_stderr"),
        (ROOT/"ros2/quatops_ros_capture/src/capture_ros.cpp","native_exporter_source"),
        (ROOT/"ros2/quatops_ros_capture/CMakeLists.txt","build_recipe")])
    print(cap.finish(["EULER","ALGEBRA","MATRIX","ACTION","FRAME","FORMAT"]))


if __name__ == "__main__":
    main()
