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
    ProjCaptureContract,
    ProjNativeInputs,
    ProjNativeOutputs,
    ProjNumericBasisImages,
    ProjNumericVector3,
    ProjObservation,
)
from scripts.evidence_schema.shared_inputs import GeographicInputCorpus


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
    args = cli.parse_args()

    network.set_network_enabled(False)
    data = GeographicInputCorpus.model_validate_json(
        args.geographic.read_bytes(), strict=True
    )
    native_binary = Path(transformer.__file__)
    if (pyproj.__version__, pyproj.proj_version_str) != ("3.7.2", "9.5.1"):
        raise ValueError("This capture requires pyproj 3.7.2 and PROJ 9.5.1")
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
            native_inputs=ProjNativeInputs(),
            native_outputs=ProjNativeOutputs(
                origin_ecef_m=ProjNumericVector3(values=list(origin), dtype="binary64"),
                enu_basis_endpoints_ecef_m=ProjNumericBasisImages(
                    values=[list(point) for point in ecef_points[1:4]],
                    dtype="binary64",
                ),
                probe_endpoint_ecef_m=ProjNumericVector3(
                    values=list(ecef_points[-1]), dtype="binary64"
                ),
                probe_roundtrip_local_xyz_m=ProjNumericVector3(
                    values=list(probe_readback), dtype="binary64"
                ),
            ),
            status="ok",
            diagnostic=None,
            schema_version="1.0",
        )
        observations.append(observation)

    rows = args.output / "observations.jsonl"
    rows.write_text(
        "".join(
            json.dumps(row.model_dump(mode="json"), allow_nan=False) + "\n"
            for row in observations
        ),
        encoding="utf-8",
        newline="\n",
    )

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "evidence_kind": "native_proj_geographic_capture",
        "utc": datetime.now(timezone.utc).isoformat(),
        "pyproj_version": pyproj.__version__,
        "proj_version": pyproj.proj_version_str,
        "python": sys.version,
        "platform": platform.platform(),
        "native_binary": str(native_binary),
        "native_binary_sha256": digest(native_binary),
        "input_sha256": digest(args.geographic),
        "exporter_sha256": digest(Path(__file__)),
        "lockfile_sha256": digest(
            args.repository_root / "experiments/converters/python/uv.lock"
        ),
        "observations_sha256": digest(rows),
        "observation_count": len(observations),
        "scope": "WGS84 ENU/ECEF point transforms; network disabled; not Cesium or quaternion execution",
    }
    path = args.output / "manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(path)


if __name__ == "__main__":
    main()
