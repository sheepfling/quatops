"""Capture PROJ ENU/ECEF transforms as support for the Cesium frame profile."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyproj
from pyproj import CRS, Transformer, network
from pyproj.enums import TransformDirection

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from experiments.oracles.shared.capture_io import (
        Capture,
        Observation,
        contract,
        numeric,
    )

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--geographic", type=Path, default=HERE / "geographic_inputs.json")
    cli.add_argument(
        "--output", type=Path, required=True, help="New, empty capture directory"
    )
    args = cli.parse_args()

    network.set_network_enabled(False)
    data = json.loads(args.geographic.read_text(encoding="utf-8"))
    module_dir = Path(pyproj.__file__).parent
    native_binary = next(iter(module_dir.glob("_transformer*")), None)
    capture = Capture(
        "proj_geographic",
        "PROJ via pyproj",
        pyproj.proj_version_str,
        args.geographic,
        args.output,
        Path(__file__),
        {
            "input": "binary64 geographic degrees/metres",
            "arithmetic": "native PROJ double",
            "output": "binary64",
            "serialized": "JSON plus IEEE bits",
        },
        oracle_family="proj",
        native_binary=native_binary,
        dependencies=[{"name": "pyproj", "version": pyproj.__version__}],
        limitations=[
            "This is PROJ execution, not Cesium execution.",
            "Local Cartesian point transforms only; no quaternion/HPR function is inferred.",
            "At geographic poles longitude fixes a convention; Cesium may choose another fallback.",
            "Network disabled; WGS84 ellipsoid only, with no external datum grids.",
        ],
    )
    capture.artifacts.append((args.geographic.resolve(), "geographic_input_corpus"))

    geocentric = Transformer.from_crs(4979, 4978, always_xy=True)
    ellipsoid = CRS.from_epsg(4979).ellipsoid
    for item in data["locations"]:
        longitude = item["longitude_degrees"]
        latitude = item["latitude_degrees"]
        height = item["height_m"]
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
            data["local_point_xyz_m"],
        ]
        ecef_points = [
            local.transform(*point, direction=TransformDirection.INVERSE, errcheck=True)
            for point in probes
        ]
        local_readback = [
            local.transform(*point, errcheck=True) for point in ecef_points
        ]
        capture.add(
            Observation(
                item["case_id"],
                "geographic-enu-ecef",
                "pyproj.Transformer.from_crs(4979,4978); from_pipeline(topocentric); transform",
                contract(
                    units="geographic degrees; Cartesian metres",
                    frame="forward ECEF -> ENU; inverse ENU -> ECEF",
                ),
                {
                    "longitude_degrees": numeric(longitude),
                    "latitude_degrees": numeric(latitude),
                    "height_m": numeric(height),
                    "pipeline": pipeline,
                    "always_xy": True,
                    "ellipsoid_semi_major_m": numeric(ellipsoid.semi_major_metre),
                    "ellipsoid_inverse_flattening": numeric(
                        ellipsoid.inverse_flattening
                    ),
                    "local_probe_points": numeric(probes),
                },
                {
                    "origin_ecef_m": numeric(list(origin)),
                    "ecef_probe_points": numeric(
                        [list(point) for point in ecef_points]
                    ),
                    "local_readback_points": numeric(
                        [list(point) for point in local_readback]
                    ),
                },
            )
        )

    print(capture.finish(["FRAME"]))


if __name__ == "__main__":
    main()
