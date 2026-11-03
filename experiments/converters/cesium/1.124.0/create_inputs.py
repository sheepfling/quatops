"""Generate converter inputs using QuatOps, with explicit local ENU frames."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from quatops.engines import QuaternionFrameMap, SignedAxisMap
from quatops.python import hamilton

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
WITNESS_INPUTS = ROOT / "experiments/interoperability/v1/canonical_inputs.json"
GEOGRAPHIC_INPUTS = (
    ROOT / "experiments/converters/cesium/1.124.0/geographic_inputs.json"
)
OUTPUT = Path(__file__).with_name("converter_inputs.json")


def main() -> None:
    from scripts.evidence_json import read_evidence_document

    canonical = read_evidence_document(WITNESS_INPUTS)
    geographic = read_evidence_document(GEOGRAPHIC_INPUTS)
    by_id = {row["case_id"]: row["canonical_wxyz"] for row in canonical["orientations"]}
    identity = SignedAxisMap(("x", "y", "z"))
    to_cesium_local = QuaternionFrameMap(identity, identity)
    selection = (
        ("identity", "geo-00", (0.0, 0.0, 0.0)),
        ("quarter_x", "geo-03", (1.25, -2.5, 3.75)),
        ("quarter_y", "geo-04", (-7.0, 3.5, 2.0)),
        ("quarter_z", "geo-04", (2.0, 5.0, -1.0)),
        ("mixed_a", "geo-03", (12.5, -8.25, 4.0)),
        ("mixed_b", "geo-04", (-3.0, 6.0, 9.0)),
        ("antipode_mixed_a", "geo-09", (0.5, 1.0, -2.0)),
        ("nonprincipal_halfturn", "geo-00", (4.0, -1.0, 2.0)),
    )
    locations = {row["case_id"]: row for row in geographic["locations"]}
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
                "longitude_degrees": location["longitude_degrees"],
                "latitude_degrees": location["latitude_degrees"],
                "height_m": location["height_m"],
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
            "longitude_degrees": locations["geo-03"]["longitude_degrees"],
            "latitude_degrees": locations["geo-03"]["latitude_degrees"],
            "height_m": locations["geo-03"]["height_m"],
        }
    )

    payload = {
        "schema_version": "1.0",
        "kind": "quatops_generated_converter_inputs",
        "canonical_input_sha256": hashlib.sha256(
            WITNESS_INPUTS.read_bytes()
        ).hexdigest(),
        "geographic_input_sha256": hashlib.sha256(
            GEOGRAPHIC_INPUTS.read_bytes()
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
            canonical["probe_vector_xyz"],
        ],
        "cases": cases,
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"wrote {len(cases)} QuatOps-generated Cesium converter inputs: {OUTPUT}")


if __name__ == "__main__":
    main()
