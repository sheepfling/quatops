"""Validate direct Sophus native observations and write typed witnesses."""

from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from scripts.evidence_format import dump_evidence_json
from scripts.evidence_json import read_evidence_document
from scripts.evidence_schema.primitives import (
    NativeCaptureModel,
    SemanticContractReference,
)
from scripts.evidence_schema.sophus import (
    CONSTRUCTION_CASES,
    CONSTRUCTOR_COLUMNS,
    EXP_LOG_CASES,
    EXP_LOG_COLUMNS,
    ROTATION_COLUMNS,
    ConstructorRow,
    ExpLogRow,
    RotationRow,
    SophusDiscoveryWitnessSet,
    SophusSemanticConfig,
    Vector3,
)
from scripts.evidence_schema.utilities import sha256_file

_PINNED_INPUT_SHA256 = (
    "713fece6dbad561b864c3f625c740190f92210318af4fa09f4467930a04a3048"
)


class NativeCapture(NativeCaptureModel):
    """Only native values selected for the permanent witness set."""

    capture_schema: Literal["quatops-sophus-direct-v1"]
    target: Literal["Sophus SO3d", "Sophus SO3f"]
    probe_xyz: Vector3
    quaternion_construction: Annotated[
        tuple[ConstructorRow, ...], Field(min_length=8, max_length=8)
    ]
    exp_log: Annotated[tuple[ExpLogRow, ...], Field(min_length=6, max_length=6)]
    composition: Annotated[tuple[RotationRow, ...], Field(min_length=2, max_length=2)]
    inverse: RotationRow
    perturbation_side: Annotated[
        tuple[RotationRow, ...], Field(min_length=2, max_length=2)
    ]

    @model_validator(mode="after")
    def selected_cases_are_complete(self) -> NativeCapture:
        expected = (
            (self.quaternion_construction, CONSTRUCTION_CASES),
            (self.exp_log, EXP_LOG_CASES),
            (self.composition, ("mixed_a_times_mixed_b", "mixed_b_times_mixed_a")),
            ((self.inverse,), ("inverse_mixed_a",)),
            (self.perturbation_side, ("delta_times_mixed_a", "mixed_a_times_delta")),
        )
        if any(tuple(row[0] for row in rows) != ids for rows, ids in expected):
            raise ValueError("Sophus native case IDs or order differ from the contract")
        for rows, probe_case in (
            (self.quaternion_construction, "mixed_unequal"),
            (self.exp_log, "mixed_a"),
        ):
            if any((row[-1] is not None) != (row[0] == probe_case) for row in rows):
                raise ValueError(
                    "Sophus native probe selection differs from the contract"
                )
        if any(row[-1] is None for row in self.composition) or any(
            row[-1] is not None for row in (self.inverse, *self.perturbation_side)
        ):
            raise ValueError("Sophus product or inverse probe selection differs")
        a, b = self.quaternion_construction[4][3], self.quaternion_construction[5][3]
        if sum(x * y for x, y in zip(a, b)) >= 0:
            raise ValueError("antipodal constructor storage must have opposite signs")
        return self

    def validate_pinned_inputs(self, path: Path) -> None:
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != _PINNED_INPUT_SHA256:
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

        def precision(values: tuple[float, ...]) -> tuple[float, ...]:
            if self.target == "Sophus SO3f":
                return tuple(
                    struct.unpack("=f", struct.pack("=f", v))[0] for v in values
                )
            return values

        if precision(self.probe_xyz) != precision(probe):
            raise ValueError("native probe differs from pinned TSV")
        for kind, records in (
            ("quaternion", self.quaternion_construction),
            ("tangent", self.exp_log),
        ):
            for row in records:
                if precision(row[1]) != precision(inputs[(kind, row[0])]):
                    raise ValueError(f"native input differs from pinned TSV: {row[0]}")


def load_semantic_config() -> SophusSemanticConfig:
    package = Path(__file__).resolve().parent
    root = package.parents[3]
    path = package / "semantic_contract.json"
    manifest = read_evidence_document(package / "manifest.json")
    reference = SemanticContractReference.model_validate(
        manifest.get("semantic_contract"), strict=True
    )
    if (
        reference.file != path.relative_to(root).as_posix()
        or sha256_file(path) != reference.sha256
    ):
        raise ValueError("experiment semantic contract differs from manifest")
    return SophusSemanticConfig.model_validate(
        read_evidence_document(path), strict=True
    )


def reduce_capture(
    capture_bytes: bytes, config: SophusSemanticConfig
) -> SophusDiscoveryWitnessSet:
    capture = NativeCapture.model_validate_json(capture_bytes, strict=True)
    capture.validate_pinned_inputs(Path(__file__).with_name("capture_inputs.tsv"))
    if capture.probe_xyz != tuple(config.probe_xyz):
        raise ValueError("native capture probe differs from the experiment contract")
    profile = next(
        (p for p in config.scalar_profiles if p.capture_target == capture.target), None
    )
    if profile is None:
        raise ValueError(f"semantic config has no profile for {capture.target!r}")
    return SophusDiscoveryWitnessSet(
        schema_version="2.0",
        evidence_kind="selected_native_discovery_witnesses",
        target=profile.witness_target,
        probe_frame=config.probe_frame.description,
        probe_xyz=capture.probe_xyz,
        raw_capture_sha256=hashlib.sha256(capture_bytes).hexdigest(),
        quaternion_construction_columns=CONSTRUCTOR_COLUMNS,
        quaternion_construction=capture.quaternion_construction,
        exp_log_columns=EXP_LOG_COLUMNS,
        exp_log=capture.exp_log,
        rotation_columns=ROTATION_COLUMNS,
        composition=capture.composition,
        inverse=capture.inverse,
        perturbation_side=capture.perturbation_side,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    selected = reduce_capture(args.capture.read_bytes(), load_semantic_config())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        dump_evidence_json(selected.model_dump(mode="json")),
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
