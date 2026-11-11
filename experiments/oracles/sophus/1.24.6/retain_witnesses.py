"""Validate direct Sophus native observations and write typed witnesses."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, model_validator

from scripts.evidence_format import dump_evidence_json
from scripts.evidence_json import EvidencePayload, read_evidence_document
from scripts.evidence_schema.primitives import NativeCaptureModel
from scripts.evidence_schema.sophus import (
    CONSTRUCTION_CASES,
    CONSTRUCTOR_COLUMNS,
    ROTATION_COLUMNS,
    ConstructorRow,
    RotationRow,
    SophusBindingManifest,
    SophusDiscoveryWitnessSet,
    SophusRecipeManifest,
    SophusSemanticConfig,
    SophusSourcePin,
    Vector3,
)
from scripts.evidence_schema.utilities import sha256_file
from scripts.evidence_schema.witness_references import read_evidence_file

DEFAULT_PACKAGE = Path(__file__).resolve().parent


class NativeCapture(NativeCaptureModel):
    """Only native values selected for the permanent witness set."""

    capture_schema: Literal["quatops-sophus-direct-v2"]
    target: Literal["Sophus SO3d", "Sophus SO3f"]
    probe_xyz: Vector3
    quaternion_construction: Annotated[
        tuple[ConstructorRow, ...], Field(min_length=8, max_length=8)
    ]
    composition: Annotated[tuple[RotationRow, ...], Field(min_length=2, max_length=2)]

    @model_validator(mode="after")
    def selected_cases_are_complete(self) -> NativeCapture:
        expected = (
            (self.quaternion_construction, CONSTRUCTION_CASES),
            (
                self.composition,
                ("mixed_unequal_times_nonunit", "nonunit_times_mixed_unequal"),
            ),
        )
        if any(tuple(row[0] for row in rows) != ids for rows, ids in expected):
            raise ValueError("Sophus native case IDs or order differ from the contract")
        if any(
            (row[-1] is not None) != (row[0] == "mixed_unequal")
            for row in self.quaternion_construction
        ):
            raise ValueError("Sophus native probe selection differs from the contract")
        if any(row[-1] is None for row in self.composition):
            raise ValueError("Sophus product probe selection differs")
        a, b = self.quaternion_construction[4][3], self.quaternion_construction[5][3]
        if sum(x * y for x, y in zip(a, b)) >= 0:
            raise ValueError("antipodal constructor storage must have opposite signs")
        return self

    def validate_pinned_inputs(self, path: Path, expected_sha256: str) -> None:
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ValueError(
                "pinned Sophus input TSV hash differs from the recorded corpus"
            )
        rows = [
            line.split()
            for line in data.decode("utf-8").splitlines()
            if line and not line.startswith("#")
        ]
        inputs = {
            (row[0], row[1]): tuple(float(x) for x in row[2:])
            for row in rows
            if row[0] != "probe"
        }
        probe = next(
            tuple(float(x) for x in row[1:]) for row in rows if row[0] == "probe"
        )

        def precision(values: tuple[float, ...]) -> bytes:
            code = "f" if self.target == "Sophus SO3f" else "d"
            return struct.pack(f"={len(values)}{code}", *values)

        if precision(self.probe_xyz) != precision(probe):
            raise ValueError("native probe differs from pinned TSV")
        for row in self.quaternion_construction:
            if precision(row[1]) != precision(inputs[("quaternion", row[0])]):
                raise ValueError(f"native input differs from pinned TSV: {row[0]}")


def load_recipe_manifest(package: Path = DEFAULT_PACKAGE) -> SophusRecipeManifest:
    return SophusRecipeManifest.model_validate(
        read_evidence_document(package / "manifest.json"), strict=True
    )


def load_semantic_config(package: Path = DEFAULT_PACKAGE) -> SophusSemanticConfig:
    package = package.resolve(strict=True)
    root = package.parents[3]
    path = package / "semantic_contract.json"
    manifest = load_recipe_manifest(package)
    reference = manifest.semantic_contract
    if reference.file != path.relative_to(root).as_posix():
        raise ValueError("experiment semantic contract differs from manifest")
    contract_bytes = read_evidence_file(
        root, reference.file, package, reference.sha256, context="Sophus contract"
    )
    config = SophusSemanticConfig.model_validate_json(contract_bytes, strict=True)
    source_path = package / "source.json"
    source_bytes = read_evidence_file(
        root,
        source_path.relative_to(root).as_posix(),
        package,
        manifest.artifacts.current_source_sha256,
        context="Sophus source pin",
    )
    source = SophusSourcePin.model_validate_json(source_bytes, strict=True)
    if (
        package.name != config.version
        or source.product != config.target
        or source.version != config.version
        or f"{source.dependency.product} {source.dependency.version}"
        != config.dependency
    ):
        raise ValueError(
            "Sophus source pin differs from the versioned semantic contract"
        )
    return config


def _preserve_negative_zero(payload: bytes) -> str:
    decoder = json.JSONDecoder(
        parse_int=lambda text: -0.0 if text == "-0" else int(text)
    )
    value = EvidencePayload.model_validate(
        decoder.decode(payload.decode("utf-8")), strict=True
    ).root
    return json.dumps(value, allow_nan=False)


def parse_native_capture(capture_bytes: bytes) -> NativeCapture:
    """Preserve JSON's integer-spelled negative zero before typed validation."""
    return NativeCapture.model_validate_json(
        _preserve_negative_zero(capture_bytes), strict=True
    )


def parse_witness(witness_bytes: bytes) -> SophusDiscoveryWitnessSet:
    """Preserve signed zeros in a shared witness before typed validation."""
    return SophusDiscoveryWitnessSet.model_validate_json(
        _preserve_negative_zero(witness_bytes), strict=True
    )


def reduce_capture(
    capture_bytes: bytes,
    config: SophusSemanticConfig,
    package: Path = DEFAULT_PACKAGE,
) -> SophusDiscoveryWitnessSet:
    if config != load_semantic_config(package):
        raise ValueError("Sophus configuration differs from the selected package")
    capture = parse_native_capture(capture_bytes)
    capture.validate_pinned_inputs(
        package / "capture_inputs.tsv",
        load_recipe_manifest(package).artifacts.input_sha256,
    )
    if capture.probe_xyz != tuple(config.probe_xyz):
        raise ValueError("native capture probe differs from the experiment contract")
    profile = next(
        (p for p in config.scalar_profiles if p.capture_target == capture.target), None
    )
    if profile is None:
        raise ValueError(f"semantic config has no profile for {capture.target!r}")
    return SophusDiscoveryWitnessSet(
        schema_version="4.0",
        evidence_kind="selected_native_discovery_witnesses",
        scalar_profile=profile.precision,
        probe_xyz=capture.probe_xyz,
        quaternion_construction_columns=CONSTRUCTOR_COLUMNS,
        quaternion_construction=capture.quaternion_construction,
        rotation_columns=ROTATION_COLUMNS,
        composition=capture.composition,
    )


def verify_exact_capture(
    capture_bytes: bytes,
    witness_bytes: bytes,
    config: SophusSemanticConfig,
    package: Path = DEFAULT_PACKAGE,
) -> SophusDiscoveryWitnessSet:
    """Compare native capture and shared values exactly, including signed zeros."""
    captured = reduce_capture(capture_bytes, config, package)
    witness = parse_witness(witness_bytes)
    if dump_evidence_json(captured.model_dump(mode="json")) != dump_evidence_json(
        witness.model_dump(mode="json")
    ):
        raise ValueError("native capture differs from the shared witness payload")
    return witness


def validate_witness_binding(
    manifest: SophusBindingManifest,
    config: SophusSemanticConfig,
    precision: Literal["float32", "float64"],
    witness_bytes: bytes,
) -> SophusDiscoveryWitnessSet:
    """Validate a version's recorded association; this attestation cannot prove execution."""
    artifacts = manifest.artifacts
    if precision not in artifacts.scalar_profiles:
        raise ValueError("Sophus scalar profile is not recorded for this version")
    reference = artifacts.scalar_profiles[precision]
    run = artifacts.direct_capture
    path = PurePosixPath(reference.selected_witness_file)
    expected_name = f"{precision}-{reference.selected_witness_sha256}.json"
    expected_dir = PurePosixPath("experiments/oracles/sophus/shared/witnesses")
    names = [
        profile.capture_target.removeprefix("Sophus ")
        for profile in config.scalar_profiles
    ]
    expected_targets = {
        f"Sophus {config.version} {' and '.join(order)} / {config.dependency}"
        for order in (names, names[::-1])
    }
    declared = {profile.precision for profile in config.scalar_profiles}
    if (
        path.parent != expected_dir
        or path.name != expected_name
        or hashlib.sha256(witness_bytes).hexdigest()
        != reference.selected_witness_sha256
        or set(artifacts.scalar_profiles) != declared
        or set(run.raw_capture_sha256) != declared
        or manifest.target not in expected_targets
        or run.product_version != config.version
        or run.dependency != config.dependency
        or run.input_sha256 != artifacts.input_sha256
        or run.raw_capture_sha256.get(precision) != reference.raw_capture_sha256
    ):
        raise ValueError(
            "Sophus version, native run, or shared witness binding differs"
        )
    witness = parse_witness(witness_bytes)
    if witness.scalar_profile != precision or witness.probe_xyz != tuple(
        config.probe_xyz
    ):
        raise ValueError("shared witness profile or probe differs from the contract")
    return witness


def load_shared_witness(
    precision: Literal["float32", "float64"],
    package: Path = DEFAULT_PACKAGE,
) -> SophusDiscoveryWitnessSet:
    """Load the recorded payload after checking its version-specific association."""
    if precision not in {"float32", "float64"}:
        raise ValueError("unsupported Sophus scalar profile")
    package = package.resolve(strict=True)
    root = package.parents[3]
    manifest = SophusBindingManifest.model_validate(
        read_evidence_document(package / "manifest.json"), strict=True
    )
    config = load_semantic_config(package)
    run = manifest.artifacts.direct_capture
    read_evidence_file(
        root,
        run.capture_source_file,
        root / "experiments/oracles/sophus",
        run.capture_source_sha256,
        context="Sophus capture source",
    )
    if run.input_sha256 != sha256_file(package / "capture_inputs.tsv"):
        raise ValueError("Sophus recorded capture input differs")
    if precision not in manifest.artifacts.scalar_profiles:
        raise ValueError("Sophus scalar profile is not recorded for this version")
    reference = manifest.artifacts.scalar_profiles[precision]
    shared = (root / "experiments/oracles/sophus/shared/witnesses").resolve(strict=True)
    witness_bytes = read_evidence_file(
        root,
        reference.selected_witness_file,
        shared,
        reference.selected_witness_sha256,
        context="Sophus witness",
    )
    return validate_witness_binding(manifest, config, precision, witness_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--compare", type=Path)
    args = parser.parse_args()
    capture_bytes = args.capture.read_bytes()
    package = args.package_dir.resolve(strict=True)
    config = load_semantic_config(package)
    if args.compare is not None:
        witness_bytes = args.compare.read_bytes()
        selected = verify_exact_capture(capture_bytes, witness_bytes, config, package)
        print(
            json.dumps(
                {
                    "result": "exact",
                    "declared_product_version": config.version,
                    "declared_dependency": config.dependency,
                    "execution_verified": False,
                    "scalar_profile": selected.scalar_profile,
                    "native_capture_sha256": hashlib.sha256(capture_bytes).hexdigest(),
                    "witness_sha256": hashlib.sha256(witness_bytes).hexdigest(),
                }
            )
        )
        return
    selected = reduce_capture(capture_bytes, config, package)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        dump_evidence_json(selected.model_dump(mode="json")),
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
