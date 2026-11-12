"""Strict schema for PROJ's retained geographic-frame observations."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    StrictBool,
    StrictInt,
    model_validator,
)

from scripts.evidence_schema.primitives import SHA256, NativeCaptureModel
from scripts.evidence_schema.shared_inputs import GeographicInputCorpus
from scripts.evidence_schema.witness_references import read_evidence_file

FiniteVector3 = Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]
FiniteBasisImages = Annotated[list[FiniteVector3], Field(min_length=3, max_length=3)]


def _native_integer(value: str) -> float | int:
    """Retain the sign of JSON's integer-form negative zero."""
    return -0.0 if value == "-0" else int(value)


class ProjCaptureContract(NativeCaptureModel):
    """Exact units and transform direction used by the PROJ experiment."""

    units: Literal["geographic degrees; Cartesian metres"]
    quaternion_order: None
    matrix_action: Literal["column vectors"]
    frame_direction: Literal["forward ECEF -> ENU; inverse ENU -> ECEF"]


class ProjObservation(NativeCaptureModel):
    """Native observations for one input, resolved by ID in the shared corpus."""

    case_id: str
    origin_ecef_m: FiniteVector3
    enu_endpoints_ecef_m: FiniteBasisImages
    probe_ecef_m: FiniteVector3
    readback_enu_m: FiniteVector3


class ProjCaptureHeader(NativeCaptureModel):
    """Common metadata for all rows in one native capture."""

    schema_version: Literal["2.0"]
    operation_id: Literal["geographic-enu-ecef"]
    native_entry_point: Literal[
        "pyproj.Transformer.from_crs(4979,4978); from_pipeline(topocentric); transform"
    ]
    contract: ProjCaptureContract
    dtype: Literal["binary64"]
    status: Literal["ok"]
    diagnostic: None


class ProjCapture(ProjCaptureHeader):
    """Typed native capture with one shared contract and binary64 declaration."""

    observations: list[ProjObservation] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_cases(self) -> ProjCapture:
        if len({row.case_id for row in self.observations}) != len(self.observations):
            raise ValueError("PROJ capture contains duplicate case IDs")
        return self

    @classmethod
    def from_jsonl(cls, text: str) -> ProjCapture:
        """Read one shared header followed by typed native observation rows."""
        lines = text.splitlines()
        if not lines:
            raise ValueError("PROJ capture requires a header and observations")
        header = ProjCaptureHeader.model_validate_json(lines[0], strict=True)
        decoder = json.JSONDecoder(parse_int=_native_integer)
        rows = [
            ProjObservation.model_validate(decoder.decode(line), strict=True)
            for line in lines[1:]
        ]
        return cls(
            schema_version=header.schema_version,
            operation_id=header.operation_id,
            native_entry_point=header.native_entry_point,
            contract=header.contract,
            dtype=header.dtype,
            status=header.status,
            diagnostic=header.diagnostic,
            observations=rows,
        )

    def to_jsonl(self) -> str:
        """Write one shared header followed by lossless numeric observations."""
        header = self.model_dump(mode="json", exclude={"observations"})
        rows = [json.dumps(header, allow_nan=False, separators=(",", ":"))]
        rows.extend(
            json.dumps(
                row.model_dump(mode="json"), allow_nan=False, separators=(",", ":")
            )
            for row in self.observations
        )
        return "\n".join(rows) + "\n"


class _ManifestModel(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)


class _Dependency(_ManifestModel):
    name: str
    version: str


class _Product(_ManifestModel):
    name: Literal["PROJ via pyproj"]
    version: str
    dependencies: list[_Dependency]


class _Execution(_ManifestModel):
    was_run: StrictBool
    exit_code: StrictInt

    @model_validator(mode="after")
    def successful(self) -> _Execution:
        if not self.was_run or self.exit_code != 0:
            raise ValueError("PROJ native run must have completed successfully")
        return self


class _Reproduction(_ManifestModel):
    input_path: str
    input_sha256: SHA256
    exporter_path: str
    exporter_sha256: SHA256


class _CompactRun(_Execution):
    capture_operations_completed: StrictBool
    utc: str = Field(min_length=1)
    pyproj_version: str
    proj_version: str
    python: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    native_binary_sha256: SHA256
    native_binary_note: str = Field(min_length=1)
    input_sha256: SHA256
    exporter_sha256: SHA256
    observations_sha256: SHA256
    compared_witness_exact: StrictBool
    observation_count: Annotated[StrictInt, Field(ge=1)]

    @model_validator(mode="after")
    def completed(self) -> _CompactRun:
        if not self.capture_operations_completed or not self.compared_witness_exact:
            raise ValueError(
                "PROJ native capture and exact witness comparison must complete"
            )
        return self


class _Artifact(_ManifestModel):
    path: str
    sha256: SHA256
    role: str


class _VersionManifest(_ManifestModel):
    product: _Product
    execution: _Execution
    reproduction: _Reproduction
    compact_reproduction: _CompactRun
    artifacts: list[_Artifact]


def load_proj_capture(repository_root: Path, version_dir: Path) -> ProjCapture:
    """Load shared evidence only through an executed version's typed manifest."""
    root = repository_root.resolve(strict=True)
    evidence_root = (root / "tests/golden/oracles/proj").resolve(strict=True)
    version = version_dir.resolve(strict=True)
    if version.parent != evidence_root:
        raise ValueError("PROJ version directory is outside the evidence directory")
    manifest = _VersionManifest.model_validate_json(
        (version / "manifest.json").read_bytes(), strict=True
    )
    run = manifest.compact_reproduction
    if manifest.product.version != version.name or run.proj_version != version.name:
        raise ValueError("PROJ version does not match the recorded native run")
    pyproj_versions = [
        dependency.version
        for dependency in manifest.product.dependencies
        if dependency.name == "pyproj"
    ]
    if pyproj_versions != [run.pyproj_version]:
        raise ValueError("pyproj version differs from the recorded native run")
    artifacts = [
        artifact
        for artifact in manifest.artifacts
        if artifact.role == "selected_native_observations"
    ]
    if len(artifacts) != 1:
        raise ValueError("PROJ manifest requires exactly one witness reference")
    artifact = artifacts[0]
    reproduction = manifest.reproduction
    if (
        artifact.sha256 != run.observations_sha256
        or reproduction.input_sha256 != run.input_sha256
        or reproduction.exporter_sha256 != run.exporter_sha256
    ):
        raise ValueError("PROJ manifest digests differ from the recorded native run")
    if PurePosixPath(artifact.path).name != f"observations-{artifact.sha256}.jsonl":
        raise ValueError("PROJ witness filename must contain its complete digest")
    witness = read_evidence_file(
        version, artifact.path, evidence_root, artifact.sha256, context="PROJ witness"
    )
    inputs = read_evidence_file(
        root,
        reproduction.input_path,
        root,
        reproduction.input_sha256,
        context="PROJ input corpus",
    )
    read_evidence_file(
        root,
        reproduction.exporter_path,
        root,
        reproduction.exporter_sha256,
        context="PROJ exporter",
    )
    capture = ProjCapture.from_jsonl(witness.decode("utf-8"))
    corpus = GeographicInputCorpus.model_validate_json(inputs, strict=True)
    if len(capture.observations) != run.observation_count or {
        row.case_id for row in capture.observations
    } != {item.case_id for item in corpus.locations}:
        raise ValueError("PROJ witness cases differ from the recorded corpus or run")
    return capture
