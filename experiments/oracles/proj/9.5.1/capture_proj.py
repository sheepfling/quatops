"""Capture PROJ ENU/ECEF transforms as support for the Cesium frame profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyproj
import pyproj._transformer as transformer
from pyproj import Transformer, network
from pyproj.enums import TransformDirection

from scripts.evidence_schema.proj_capture import (
    ProjCapture,
    ProjCaptureContract,
    ProjObservation,
)
from scripts.evidence_schema.shared_inputs import GeographicInputCorpus
from scripts.evidence_schema.utilities import sha256_file


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--repository-root", type=Path, required=True)
    cli.add_argument(
        "--geographic",
        type=Path,
        required=True,
    )
    cli.add_argument(
        "--output", type=Path, required=True, help="New, empty capture directory"
    )
    cli.add_argument(
        "--compare-witness",
        type=Path,
        help="Proposed shared witness to compare exactly",
    )
    cli.add_argument("--proj-version", help="Required native PROJ runtime version")
    cli.add_argument("--pyproj-version", help="Required pyproj wrapper version")
    args = cli.parse_args()

    if (args.proj_version is None) != (args.pyproj_version is None):
        cli.error("--proj-version and --pyproj-version must be supplied together")
    requested_proj = args.proj_version or "9.5.1"
    requested_pyproj = args.pyproj_version or "3.7.2"

    network.set_network_enabled(False)
    input_bytes = args.geographic.read_bytes()
    data = GeographicInputCorpus.model_validate_json(input_bytes, strict=True)
    native_binary = Path(transformer.__file__)
    if (pyproj.__version__, pyproj.proj_version_str) != (
        requested_pyproj,
        requested_proj,
    ):
        raise ValueError(
            f"Capture requires pyproj {requested_pyproj} and PROJ {requested_proj}; "
            f"found pyproj {pyproj.__version__} and PROJ {pyproj.proj_version_str}"
        )
    if native_binary.suffix not in {".pyd", ".so"} or not native_binary.is_file():
        raise FileNotFoundError(
            f"Cannot identify pyproj's native extension: {native_binary}"
        )
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite capture: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    observations: list[ProjObservation] = []

    geocentric = Transformer.from_crs(4979, 4978, always_xy=True)
    for item in data.locations:
        longitude = item.longitude_degrees
        latitude = item.latitude_degrees
        height = item.height_m
        origin = geocentric.transform(longitude, latitude, height, errcheck=True)
        pipeline = (
            "+proj=topocentric +ellps=WGS84 "
            f"+lon_0={longitude:.17g} +lat_0={latitude:.17g} +h_0={height:.17g}"
        )
        local = Transformer.from_pipeline(pipeline)
        probes = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            data.local_point_xyz_m,
        ]
        ecef_points = [
            local.transform(*point, direction=TransformDirection.INVERSE, errcheck=True)
            for point in probes
        ]
        probe_readback = local.transform(*ecef_points[-1], errcheck=True)
        observation = ProjObservation(
            case_id=item.case_id,
            origin_ecef_m=list(origin),
            enu_endpoints_ecef_m=[list(point) for point in ecef_points[1:4]],
            probe_ecef_m=list(ecef_points[-1]),
            readback_enu_m=list(probe_readback),
        )
        observations.append(observation)

    capture = ProjCapture(
        schema_version="2.0",
        operation_id="geographic-enu-ecef",
        native_entry_point=(
            "pyproj.Transformer.from_crs(4979,4978); "
            "from_pipeline(topocentric); transform"
        ),
        contract=ProjCaptureContract(
            units="geographic degrees; Cartesian metres",
            quaternion_order=None,
            matrix_action="column vectors",
            frame_direction="forward ECEF -> ENU; inverse ENU -> ECEF",
        ),
        dtype="binary64",
        status="ok",
        diagnostic=None,
        observations=observations,
    )
    rows = args.output / "observations.jsonl"
    capture_bytes = capture.to_jsonl().encode("utf-8")
    rows.write_bytes(capture_bytes)
    comparison_sha = None
    if args.compare_witness is not None:
        proposed = args.compare_witness.read_bytes()
        actual = capture_bytes
        comparison_sha = hashlib.sha256(proposed).hexdigest()
        actual_sha = hashlib.sha256(actual).hexdigest()
        print(
            f"Proposed witness SHA-256: {comparison_sha}; capture SHA-256: {actual_sha}"
        )
        if proposed != actual:
            raise ValueError(
                "Native capture does not exactly match the proposed witness"
            )

    manifest = {
        "evidence_kind": "native_proj_geographic_capture",
        "was_run": True,
        "capture_operations_completed": True,
        "utc": datetime.now(timezone.utc).isoformat(),
        "pyproj_version": pyproj.__version__,
        "proj_version": pyproj.proj_version_str,
        "python": sys.version,
        "platform": platform.platform(),
        "native_binary": str(native_binary),
        "native_binary_sha256": sha256_file(native_binary),
        "native_binary_note": (
            "SHA-256 of the pyproj _transformer extension, "
            "not a hash of every dynamically linked PROJ library."
        ),
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "exporter_sha256": sha256_file(Path(__file__)),
        "digest_helper_sha256": sha256_file(
            args.repository_root / "scripts/evidence_schema/utilities.py"
        ),
        "lockfile_sha256": sha256_file(
            args.repository_root / "experiments/converters/python/uv.lock"
        ),
        "observations_sha256": hashlib.sha256(capture_bytes).hexdigest(),
        "observation_count": len(observations),
        "scope": "WGS84 ENU/ECEF point transforms; network disabled; not Cesium or quaternion execution",
    }
    if comparison_sha is not None:
        manifest["compared_witness_exact"] = True
    path = args.output / "manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(path)


if __name__ == "__main__":
    main()
