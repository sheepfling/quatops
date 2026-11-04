"""Load and crystallize native game-engine quaternion witness packages."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Literal

from scripts.evidence_json import read_evidence_document
from scripts.evidence_schema.crystallized import (
    CanonicalInputs,
    CaptureProvenance,
    CompositionWitness,
    CoordinateFrame,
    CrystallizationHeader,
    CrystallizedCapture,
    CrystallizedEvidence,
    CrystallizedWitnessSet,
    DependencyPin,
    EvidenceTarget,
    EvidenceWitness,
    OrientationWitness,
    PrecisionTolerance,
    QuaternionBoundaryContract,
)
from scripts.evidence_schema.engine_targets import (
    EngineCaptureDocument,
    EngineComposition,
    EngineContractDocument,
    EngineOrientation,
    NativeVector3,
    Vector3Object,
)

_RETAINED_CASES = frozenset(
    {
        "identity",
        "quarter_x",
        "quarter_y",
        "quarter_z",
        "mixed_a",
        "antipode_mixed_a",
        "nonprincipal_halfturn",
    }
)

ENGINE_VERSIONS = (
    ("unreal", "5.7.4"),
    ("unity", "6000.5.2f1"),
    ("godot", "4.7-stable"),
    ("godot", "4.8-dev1"),
    ("blender", "5.1.2"),
)


class LoadedEnginePackage:
    """Typed pair of a versioned convention contract and its native capture."""

    def __init__(
        self,
        contract_path: Path,
        witness_path: Path,
        contract: EngineContractDocument,
        capture: EngineCaptureDocument,
    ) -> None:
        self.contract_path = contract_path
        self.witness_path = witness_path
        self.contract = contract
        self.capture = capture


def load_engine_package(
    repository_root: Path, engine: str, version: str
) -> LoadedEnginePackage:
    """Validate one engine contract and the native witness file it names."""
    directory = repository_root / "tests" / "oracles" / engine / version
    contract_path = directory / "contract.json"
    raw_contract = read_evidence_document(contract_path)
    contract = EngineContractDocument.model_validate(raw_contract, strict=True)
    witness_name = contract.provenance.witness_file or "witnesses.json"
    witness_path = (directory / witness_name).resolve()
    witness_document = read_evidence_document(witness_path)
    capture_document = {
        **witness_document,
        "target": witness_document.get("target", raw_contract["engine"]),
        "contract": witness_document.get("contract", raw_contract["conversion"]),
        "provenance": witness_document.get("provenance", raw_contract["provenance"]),
    }
    capture = EngineCaptureDocument.model_validate(capture_document, strict=True)

    if contract.engine.engine != capture.target.engine:
        raise ValueError(f"{directory}: contract and capture engine names differ")
    if contract.engine.api != capture.target.api:
        raise ValueError(f"{directory}: contract and capture API identities differ")
    if contract.engine.version and capture.target.version not in (
        None,
        contract.engine.version,
    ):
        raise ValueError(f"{directory}: contract and capture versions differ")
    if contract.conversion.contract_id != capture.contract.contract_id:
        raise ValueError(f"{directory}: contract and capture contract IDs differ")
    if capture.canonical_interop.contract_id != contract.conversion.contract_id:
        raise ValueError(f"{directory}: canonical interop contract ID differs")
    if (
        capture.canonical_interop.native_quaternion_order
        != contract.engine.native_quaternion_order
    ):
        raise ValueError(f"{directory}: native quaternion order differs")
    if (
        capture.canonical_interop.frame_map_destination_axes_from_source
        != contract.conversion.destination_axes_from_source
    ):
        raise ValueError(f"{directory}: native frame map differs")
    if contract.provenance.witness_sha256:
        actual_witness_hash = hashlib.sha256(witness_path.read_bytes()).hexdigest()
        if actual_witness_hash != contract.provenance.witness_sha256:
            raise ValueError(
                f"{directory}: native witness digest differs from contract"
            )
    return LoadedEnginePackage(contract_path, witness_path, contract, capture)


def _vector(value: NativeVector3) -> list[float]:
    if isinstance(value, Vector3Object):
        return [value.x, value.y, value.z]
    return value


def _handedness_from_map(
    axis_map: list[str],
) -> Literal["right-handed", "left-handed"]:
    axes = [value.lstrip("-") for value in axis_map]
    inversions = sum(
        axes[left] > axes[right] for left in range(3) for right in range(left + 1, 3)
    )
    sign = -1 if inversions % 2 else 1
    if sum(value.startswith("-") for value in axis_map) % 2:
        sign *= -1
    return "right-handed" if sign > 0 else "left-handed"


def _opposite_axis_meaning(value: str) -> str:
    opposites = {
        "right": "left",
        "left": "right",
        "east": "west",
        "west": "east",
        "forward": "backward",
        "backward": "forward",
        "north": "south",
        "south": "north",
        "up": "down",
        "down": "up",
    }
    if value not in opposites:
        raise ValueError(f"cannot invert declared axis meaning {value!r}")
    return opposites[value]


def _native_axis_labels(package: LoadedEnginePackage) -> tuple[str, str, str]:
    frame = package.contract.conversion.native_frame
    labels = (frame.positive_x, frame.positive_y, frame.positive_z)
    if all(label is not None for label in labels):
        return labels  # type: ignore[return-value]
    declaration = frame.declared_probe_semantics
    if declaration is None:
        raise ValueError("native frame must declare positive axis meanings")
    match = re.search(r"\+X\s+([^,]+),\s*\+Y\s+([^,]+),\s*\+Z\s+(.+)", declaration)
    if match is None:
        raise ValueError("native probe frame declaration is not parseable")
    return (
        match.group(1).strip(),
        match.group(2).strip(),
        re.split(r"\s+for\s+", match.group(3).strip(), maxsplit=1)[0],
    )


def _validate_frame_map(package: LoadedEnginePackage) -> tuple[str, str, str]:
    conversion = package.contract.conversion
    axis_map = conversion.destination_axes_from_source
    normalized_axes = [axis.lstrip("-") for axis in axis_map]
    if (
        len(axis_map) != 3
        or any(axis not in {"x", "y", "z", "-x", "-y", "-z"} for axis in axis_map)
        or sorted(normalized_axes) != ["x", "y", "z"]
    ):
        raise ValueError("native frame map must be a signed-axis permutation")
    canonical = conversion.canonical_frame
    declared_labels = _native_axis_labels(package)
    canonical_labels = (
        canonical.positive_x,
        canonical.positive_y,
        canonical.positive_z,
    )
    for destination_index, axis in enumerate(axis_map):
        source_index = "xyz".index(axis.lstrip("-"))
        expected = canonical_labels[source_index]
        if axis.startswith("-"):
            expected = _opposite_axis_meaning(expected)
        if declared_labels[destination_index] != expected:
            raise ValueError(
                f"declared native axis {destination_index} disagrees with signed map"
            )
    derived_handedness = _handedness_from_map(axis_map)
    declared_handedness = package.contract.conversion.native_frame.handedness
    if declared_handedness is not None:
        expected_hand = (
            "right-handed"
            if declared_handedness.lower() in {"right", "right-handed"}
            else "left-handed"
            if declared_handedness.lower() in {"left", "left-handed"}
            else None
        )
        if expected_hand != derived_handedness:
            raise ValueError(
                "declared native handedness disagrees with signed-axis map"
            )
    return declared_labels


def _validate_probe_vectors(
    package: LoadedEnginePackage, canonical_inputs: CanonicalInputs
) -> None:
    conversion = package.contract.conversion
    if len(canonical_inputs.probe_vector_xyz) != 3:
        raise ValueError("shared canonical input must include a three-vector probe")
    for profile, cases, _composition, _reverse in _profile_cases(package):
        tolerance = conversion.native_precision_tolerance.get(profile)
        if tolerance is None:
            raise ValueError(f"engine contract lacks tolerance for {profile}")
        probe_cases = [row for row in cases if row.case_id == "mixed_a"]
        if (
            len(probe_cases) != 1
            or probe_cases[0].native_rotated_probe_vector_xyz is None
        ):
            raise ValueError(
                f"engine {profile} witnesses must retain one mixed-vector action"
            )


def _precision_profiles(package: LoadedEnginePackage) -> list[str]:
    return package.contract.engine.native_scalar_profiles


def _profile_cases(
    package: LoadedEnginePackage,
) -> list[tuple[str, list[EngineOrientation], EngineComposition, EngineComposition]]:
    interop = package.capture.canonical_interop
    if interop.precision_captures is not None:
        return [
            (
                profile,
                capture.orientation_cases,
                capture.composition,
                capture.reverse_composition,
            )
            for profile, capture in sorted(interop.precision_captures.items())
        ]
    if (
        interop.orientation_cases is None
        or interop.composition is None
        or interop.reverse_composition is None
    ):
        raise ValueError("engine capture lacks either profiles or flat witnesses")
    if len(package.contract.engine.native_scalar_profiles) != 1:
        raise ValueError("flat engine capture must declare one precision profile")
    return [
        (
            package.contract.engine.native_scalar_profiles[0],
            interop.orientation_cases,
            interop.composition,
            interop.reverse_composition,
        )
    ]


def _orientation_witnesses(
    package: LoadedEnginePackage, canonical_inputs: CanonicalInputs
) -> list[EvidenceWitness]:
    identity = package.contract.engine
    shared = {
        row.case_id: row.coefficients_wxyz for row in canonical_inputs.orientations
    }
    output: list[EvidenceWitness] = []
    for precision, cases, composition, reverse in _profile_cases(package):
        seen_case_ids: set[str] = set()
        for row in cases:
            if row.case_id not in _RETAINED_CASES:
                continue
            if row.case_id not in shared:
                raise ValueError(
                    f"{identity.engine} {row.case_id} is absent from shared inputs"
                )
            seen_case_ids.add(row.case_id)
            probe_image = row.native_rotated_probe_vector_xyz
            if row.case_id == "mixed_a" and probe_image is None:
                raise ValueError(f"{identity.engine} mixed_a lacks its probe action")
            output.append(
                OrientationWitness(
                    kind="orientation",
                    case_id=row.case_id,
                    precision=precision,
                    coefficient_order=identity.native_quaternion_order,
                    coefficients=row.native_quaternion,
                    basis_images_xyz=[
                        _vector(value)
                        for value in row.native_rotated_positive_basis_vectors_xyz
                    ],
                    probe_image_xyz=(
                        _vector(probe_image)
                        if row.case_id == "mixed_a" and probe_image is not None
                        else None
                    ),
                )
            )
        if seen_case_ids != _RETAINED_CASES:
            raise ValueError(
                f"{identity.engine} {precision} witness IDs differ from the retained set"
            )
        expected_orders = (
            (
                canonical_inputs.composition_left_case_id,
                canonical_inputs.composition_right_case_id,
                canonical_inputs.product_order,
            ),
            (
                canonical_inputs.composition_right_case_id,
                canonical_inputs.composition_left_case_id,
                canonical_inputs.product_order,
            ),
        )
        for row, expected in zip((composition, reverse), expected_orders):
            if (row.left_case_id, row.right_case_id, row.product_order) != expected:
                raise ValueError(
                    "native composition cases differ from shared product orders"
                )
            if row.left_case_id not in shared or row.right_case_id not in shared:
                raise ValueError(
                    "native composition references an unknown canonical case"
                )
            output.append(
                CompositionWitness(
                    kind="composition",
                    case_id=f"{row.left_case_id}_times_{row.right_case_id}",
                    left_case_id=row.left_case_id,
                    right_case_id=row.right_case_id,
                    product_order=row.product_order,
                    precision=precision,
                    coefficient_order=identity.native_quaternion_order,
                    coefficients=row.native_product_quaternion,
                    basis_images_xyz=[
                        _vector(value)
                        for value in row.native_rotated_positive_basis_vectors_xyz
                    ],
                )
            )
    return output


def crystallize_engine_package(
    package: LoadedEnginePackage,
    *,
    repository_root: Path,
    canonical_inputs: CanonicalInputs,
) -> CrystallizedCapture:
    """Convert the selected native engine proof into a common quaternion record."""
    contract = package.contract
    engine = contract.engine
    conversion = contract.conversion
    declared_labels = _validate_frame_map(package)
    _validate_probe_vectors(package, canonical_inputs)
    handedness = _handedness_from_map(conversion.destination_axes_from_source)
    source = conversion.canonical_frame
    if source.handedness.lower() not in {"right", "right-handed"}:
        raise ValueError("engine interoperability source frame must be right-handed")
    if conversion.canonical_action != "active":
        raise ValueError("engine native witness contract must declare active rotations")
    if (
        conversion.canonical_algebra != canonical_inputs.contract.algebra
        or conversion.canonical_order != canonical_inputs.contract.coefficient_order
        or conversion.canonical_action != canonical_inputs.contract.rotation_action
    ):
        raise ValueError(
            "engine canonical quaternion contract differs from shared inputs"
        )
    if (
        source.positive_x != canonical_inputs.source_frame.positive_x
        or source.positive_y != canonical_inputs.source_frame.positive_y
        or source.positive_z != canonical_inputs.source_frame.positive_z
    ):
        raise ValueError("engine source frame differs from shared canonical inputs")
    native_frame = CoordinateFrame(
        name=f"{engine.engine} explicit native probe frame",
        handedness=handedness,
        positive_x=declared_labels[0],
        positive_y=declared_labels[1],
        positive_z=declared_labels[2],
    )
    common_contract = QuaternionBoundaryContract(
        kind="quaternion_boundary",
        canonical_algebra=conversion.canonical_algebra,
        canonical_order=conversion.canonical_order,
        rotation_action=conversion.canonical_action,
        source_frame=CoordinateFrame(
            name="QuatOps canonical frame",
            handedness=(
                "right-handed"
                if source.handedness.lower() in {"right", "right-handed"}
                else "left-handed"
            ),
            positive_x=source.positive_x,
            positive_y=source.positive_y,
            positive_z=source.positive_z,
        ),
        destination_frame=native_frame,
        destination_axes_from_source=conversion.destination_axes_from_source,
        native_algebra=engine.native_algebra,
        native_order=engine.native_quaternion_order,
        native_vector_action=engine.native_vector_action,
        conversion_rule=conversion.conversion_rule,
        inverse_rule=conversion.inverse_rule,
        input_domain=conversion.input_domain,
        tolerances=[
            PrecisionTolerance(profile=profile, absolute=tolerance, relative=0.0)
            for profile, tolerance in sorted(
                conversion.native_precision_tolerance.items()
            )
        ],
        scope=[
            conversion.scope,
            "native basis actions for selected orientations",
            "both orders of a noncommuting native product",
        ],
        excludes=conversion.excludes,
    )
    provenance = contract.provenance
    if provenance.capture_source is None or provenance.capture_source_sha256 is None:
        raise ValueError("engine capture contract lacks adapter provenance")
    adapter_path = repository_root / Path(provenance.capture_source)
    checked_in_hash = (
        hashlib.sha256(adapter_path.read_bytes()).hexdigest()
        if adapter_path.is_file()
        else None
    )
    witness_hash = hashlib.sha256(package.witness_path.read_bytes()).hexdigest()
    canonical_hash = provenance.canonical_input_sha256
    if canonical_hash is None:
        raise ValueError("engine contract lacks canonical input digest")
    runtime_version = (
        provenance.native_build_record.python_version
        if provenance.native_build_record is not None
        else None
    )
    native_runtime = engine.version or "version specified by per-version contract"
    if runtime_version:
        native_runtime += f"; embedded Python {runtime_version}"
    if provenance.capture_method:
        native_runtime += f"; {provenance.capture_method}"
    capture_provenance = CaptureProvenance(
        capture_adapter=provenance.capture_source,
        capture_adapter_sha256=provenance.capture_source_sha256,
        input_sha256=canonical_hash,
        contract_sha256=hashlib.sha256(package.contract_path.read_bytes()).hexdigest(),
        witnesses_sha256=witness_hash,
        native_runtime=native_runtime,
        checked_in_adapter_sha256=(
            checked_in_hash
            if checked_in_hash != provenance.capture_source_sha256
            else None
        ),
        dependencies=(
            [DependencyPin(name="embedded Python", version=runtime_version)]
            if runtime_version
            else []
        ),
    )
    target = EvidenceTarget(
        product=engine.engine,
        version=engine.version or contract.engine.version or "shared",
        api=engine.api,
        native_precision=_precision_profiles(package),
        build_id=str(engine.build) if engine.build is not None else None,
    )
    witness_set_id = (
        "godot-shared-native-witness-v1"
        if package.contract_path.parent.name.startswith("4.")
        and package.contract_path.parent.parent.name == "godot"
        else f"{package.contract_path.parent.parent.name}-native-witness-v1"
    )
    header = CrystallizationHeader(
        evidence_id=f"{engine.engine}:{target.version}",
        layer="engine-quaternion",
        target=target,
        contract_id=conversion.contract_id,
        contract=common_contract,
        witness_set_id=witness_set_id,
        provenance=capture_provenance,
    )
    return CrystallizedCapture(
        record=CrystallizedEvidence(
            schema_version=1,
            evidence_id=header.evidence_id,
            layer=header.layer,
            target=header.target,
            contract_id=header.contract_id,
            contract=header.contract,
            witness_set_id=header.witness_set_id,
            provenance=header.provenance,
        ),
        witness_set=CrystallizedWitnessSet(
            schema_version=1,
            witness_set_id=header.witness_set_id,
            witnesses=_orientation_witnesses(package, canonical_inputs),
        ),
    )
