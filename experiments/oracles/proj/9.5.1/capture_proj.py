"""Capture PROJ ENU/ECEF transforms as support for the Cesium frame profile."""

from __future__ import annotations

import argparse
from importlib import resources
from pathlib import Path

import pyproj
from pyproj import Transformer, network
from pyproj.enums import TransformDirection

from experiments.oracles.shared.capture_io import (
    Capture,
    Observation,
    contract,
)
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
        typed_observation = ProjObservation(
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
        capture.add(
            Observation(
                typed_observation.case_id,
                typed_observation.operation_id,
                typed_observation.native_entry_point,
                contract(
                    units=typed_observation.contract.units,
                    order=typed_observation.contract.quaternion_order,
                    action=typed_observation.contract.matrix_action,
                    frame=typed_observation.contract.frame_direction,
                ),
                typed_observation.native_inputs.model_dump(mode="json"),
                typed_observation.native_outputs.model_dump(mode="json"),
                status=typed_observation.status,
                diagnostic=typed_observation.diagnostic,
                schema_version=typed_observation.schema_version,
            )
        )

    print(capture.finish(["FRAME"], observation_role="selected_native_observations"))


if __name__ == "__main__":
    main()
