"""Generate converter inputs using QuatOps, with explicit local ENU frames."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from quatops.engines import QuaternionFrameMap, SignedAxisMap
from quatops.python import hamilton
from scripts.evidence_json import read_evidence_document
from scripts.evidence_schema.shared_inputs import (
    GeographicInputCorpus,
    SharedCanonicalInputCapture,
)
from scripts.evidence_schema.targets import CesiumInputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    repository_root = parser.parse_args().repository_root.resolve(strict=True)
    target_dir = repository_root / "experiments/converters/cesium/1.124.0"
    witness_inputs = (
        repository_root / "experiments/interoperability/v1/canonical_inputs.json"
    )
    geographic_inputs = (
        repository_root / "experiments/oracles/shared/geographic_inputs_v1.json"
    )
    output = target_dir / "converter_inputs.json"
    canonical = SharedCanonicalInputCapture.model_validate(
        read_evidence_document(witness_inputs), strict=True
    )
    geographic = GeographicInputCorpus.model_validate(
        read_evidence_document(geographic_inputs), strict=True
    )
    by_id = {row.case_id: row.canonical_wxyz for row in canonical.orientations}
    required_orientations = {"identity", "mixed_a", "mixed_b", "antipode_mixed_a"}
    missing_orientations = required_orientations.difference(by_id)
    if missing_orientations:
        raise ValueError(
            f"canonical input is missing Cesium cases: {sorted(missing_orientations)}"
        )
    identity = SignedAxisMap(("x", "y", "z"))
    to_cesium_local = QuaternionFrameMap(identity, identity)
    selection = (
        ("identity", "geo-00", (0.0, 0.0, 0.0)),
        ("mixed_a", "geo-03", (12.5, -8.25, 4.0)),
        ("mixed_b", "geo-04", (-3.0, 6.0, 9.0)),
        ("antipode_mixed_a", "geo-09", (0.5, 1.0, -2.0)),
    )
    locations = {row.case_id: row for row in geographic.locations}
    required_locations = {"geo-00", "geo-03", "geo-04", "geo-09"}
    missing_locations = required_locations.difference(locations)
    if missing_locations:
        raise ValueError(
            f"geographic input is missing Cesium cases: {sorted(missing_locations)}"
        )
    cases = []
    for orientation_id, location_id, translation in selection:
        source = by_id[orientation_id]
        cesium_wxyz = to_cesium_local.to_destination_wxyz(source)
        xyzw = (cesium_wxyz[1], cesium_wxyz[2], cesium_wxyz[3], cesium_wxyz[0])
        location = locations[location_id]
        cases.append(
            {
                "case_id": f"{orientation_id}@{location_id}",
                "canonical_orientation_case": orientation_id,
                "location_case": location_id,
                "canonical_wxyz": list(source),
                "quatops_cesium_local_xyzw": list(xyzw),
                "local_translation_xyz_m": list(translation),
                "longitude_degrees": location.longitude_degrees,
                "latitude_degrees": location.latitude_degrees,
                "height_m": location.height_m,
            }
        )
    mixed_composition = hamilton.multiply(by_id["mixed_a"], by_id["mixed_b"])
    composed = to_cesium_local.to_destination_wxyz(mixed_composition)
    cases.append(
        {
            "case_id": "mixed_a_times_mixed_b@geo-03",
            "canonical_orientation_case": "mixed_a * mixed_b",
            "location_case": "geo-03",
            "canonical_wxyz": list(mixed_composition),
            "quatops_cesium_local_xyzw": [
                composed[1],
                composed[2],
                composed[3],
                composed[0],
            ],
            "local_translation_xyz_m": [0.0, 0.0, 0.0],
            "longitude_degrees": locations["geo-03"].longitude_degrees,
            "latitude_degrees": locations["geo-03"].latitude_degrees,
            "height_m": locations["geo-03"].height_m,
        }
    )

    payload = {
        "schema_version": "1.0",
        "kind": "quatops_generated_converter_inputs",
        "canonical_input_sha256": hashlib.sha256(
            witness_inputs.read_bytes()
        ).hexdigest(),
        "geographic_input_sha256": hashlib.sha256(
            geographic_inputs.read_bytes()
        ).hexdigest(),
        "frame_contract": {
            "canonical_local": {
                "handedness": "right",
                "positive_x": "east/right",
                "positive_y": "north/forward",
                "positive_z": "up",
            },
            "cesium_local": {
                "handedness": "right",
                "positive_x": "east",
                "positive_y": "north",
                "positive_z": "ellipsoid normal",
            },
            "map_canonical_to_cesium_enu": ["x", "y", "z"],
            "destination_local_api": "Cesium Quaternion XYZW; active; Matrix3 column-vector action",
            "world_placement": "WGS84 ENU-to-ECEF from Cesium Transforms.eastNorthUpToFixedFrame",
        },
        "probe_vectors_xyz": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            canonical.probe_vector_xyz,
        ],
        "cases": cases,
    }
    typed_payload = CesiumInputs.model_validate(payload, strict=True)
    output.write_text(
        json.dumps(typed_payload.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {len(cases)} QuatOps-generated Cesium converter inputs: {output}")


if __name__ == "__main__":
    main()
