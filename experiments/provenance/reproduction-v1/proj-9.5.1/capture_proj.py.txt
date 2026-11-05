"""Capture PROJ ENU/ECEF transforms as support for the Cesium frame profile."""

from __future__ import annotations

import argparse
import json
from importlib import resources
from pathlib import Path

import pyproj
from pyproj import Transformer, network
from pyproj.enums import TransformDirection

from experiments.oracles.shared.capture_io import (
    Capture,
    Observation,
    contract,
    numeric,
)


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
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
    data = json.loads(args.geographic.read_text(encoding="utf-8"))
    module_files = resources.files("pyproj")
    native_binary = next(
        (
            Path(str(resource))
            for resource in module_files.iterdir()
            if resource.name.startswith("_transformer")
        ),
        None,
    )
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
            "serialized": "round-trip binary64 JSON values without per-value hex sidecars",
        },
        oracle_family="proj",
        native_binary=native_binary,
        dependencies=[{"name": "pyproj", "version": pyproj.__version__}],
        limitations=[
            "This is PROJ execution, not Cesium execution.",
            "Local Cartesian point transforms only; no quaternion/HPR function is inferred.",
            "At geographic poles longitude fixes a convention; Cesium may choose another fallback.",
            "Network disabled; WGS84 ellipsoid only, with no external datum grids.",
            "Retains the origin, three local-axis endpoints, one probe, and its inverse readback per location.",
        ],
    )
    capture.artifacts.append((args.geographic.resolve(), "geographic_input_corpus"))

    geocentric = Transformer.from_crs(4979, 4978, always_xy=True)
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
        probe_readback = local.transform(*ecef_points[-1], errcheck=True)
        capture.add(
            Observation(
                item["case_id"],
                "geographic-enu-ecef",
                "pyproj.Transformer.from_crs(4979,4978); from_pipeline(topocentric); transform",
                contract(
                    units="geographic degrees; Cartesian metres",
                    frame="forward ECEF -> ENU; inverse ENU -> ECEF",
                ),
                {},
                {
                    "origin_ecef_m": numeric(list(origin), include_bits=False),
                    "enu_basis_endpoints_ecef_m": numeric(
                        [list(point) for point in ecef_points[1:4]],
                        include_bits=False,
                    ),
                    "probe_endpoint_ecef_m": numeric(
                        list(ecef_points[-1]), include_bits=False
                    ),
                    "probe_roundtrip_local_xyz_m": numeric(
                        list(probe_readback), include_bits=False
                    ),
                },
            )
        )

    print(capture.finish(["FRAME"], observation_role="selected_native_observations"))


if __name__ == "__main__":
    main()
