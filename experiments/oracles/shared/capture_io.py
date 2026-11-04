"""Raw capture envelopes. No rotation algorithms or comparison oracles belong here."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import struct
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict

NumericType = Literal["binary32", "binary64"]
ContractUnits = Literal["radians", "geographic degrees; Cartesian metres"]
QuaternionOrder = Literal["wxyz", "xyzw"]
MatrixAction = Literal["column vectors"]
FrameDirection = Literal[
    "active rotation in one numeric basis",
    "forward ECEF -> ENU; inverse ENU -> ECEF",
    "lookup target_from_source; transform maps source point coordinates into target",
]


class CaptureContract(BaseModel):
    """Typed convention fields embedded in a raw observation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    units: ContractUnits | None = "radians"
    quaternion_order: QuaternionOrder | None = None
    matrix_action: MatrixAction | None = "column vectors"
    frame_direction: FrameDirection | None = "active rotation in one numeric basis"


class ContractRecord(TypedDict):
    """Serialized literal-only contract fields stored in each observation."""

    units: ContractUnits | None
    quaternion_order: QuaternionOrder | None
    matrix_action: MatrixAction | None
    frame_direction: FrameDirection | None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric(
    values: Any, dtype: NumericType = "binary64", *, include_bits: bool = True
) -> dict[str, Any]:
    """Serialize a numeric array, retaining post-binding bits and negative zero.

    NumPy is deliberately not required. Foreign vector/matrix objects must first
    be converted to explicit lists by the exporter that knows their layout.
    """
    code = ">f" if dtype == "binary32" else ">d"
    bits: list[str] = []

    def visit(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return [visit(item) for item in value]
        native = struct.unpack(code, struct.pack(code, float(value)))[0]
        if include_bits:
            bits.append(struct.pack(code, native).hex())
        if not math.isfinite(native):
            return {
                "nonfinite": "nan"
                if math.isnan(native)
                else ("+inf" if native > 0 else "-inf")
            }
        return native

    def shape(value: Any) -> list[int]:
        if not isinstance(value, (list, tuple)):
            return []
        if not value:
            return [0]
        child = shape(value[0])
        if any(shape(item) != child for item in value):
            raise ValueError("Ragged numeric arrays are not allowed")
        return [len(value), *child]

    dimensions = shape(values)
    result = visit(values)
    result = {
        "values": result,
        "dtype": dtype,
        "shape": dimensions,
    }
    if include_bits:
        result["ieee754_hex_be"] = bits
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


@dataclass(frozen=True, slots=True)
class Observation:
    case_id: str
    operation_id: str
    native_entry_point: str
    contract: ContractRecord
    native_inputs: dict[str, Any]
    native_outputs: dict[str, Any]
    status: str = "ok"
    diagnostic: str | None = None
    schema_version: str = "1.0"


@dataclass(slots=True)
class Capture:
    oracle_id: str
    product_name: str
    version: str
    corpus_path: Path
    output_dir: Path
    exporter_path: Path
    precision: dict[str, Any]
    oracle_family: str | None = None
    dependencies: list[dict[str, Any]] = field(default_factory=list)
    native_binary: Path | None = None
    limitations: list[str] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    artifacts: list[tuple[Path, str]] = field(default_factory=list)
    started_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite capture: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add(self, observation: Observation) -> None:
        if observation.status == "ok" and not observation.native_outputs:
            raise ValueError("An ok observation must have native output")
        self.observations.append(observation)

    def finish(
        self,
        profiles: list[str],
        *,
        exit_code: int = 0,
        observation_role: str = "raw_native_observations",
    ) -> Path:
        rows = self.output_dir / "observations.jsonl"
        rows.write_text(
            "".join(
                json.dumps(asdict(item), allow_nan=False) + "\n"
                for item in self.observations
            ),
            encoding="utf-8",
        )
        counts = Counter(item.status for item in self.observations)
        artifacts = [(rows, observation_role), *self.artifacts]
        manifest = {
            "schema_version": "1.0",
            "capture_id": self.output_dir.name,
            "oracle_id": self.oracle_id,
            "oracle_family": self.oracle_family or self.oracle_id,
            "evidence_kind": "native_runtime",
            "product": {
                "name": self.product_name,
                "version": self.version,
                "source_revision": None,
                "binary_sha256": sha256(self.native_binary)
                if self.native_binary
                else None,
                "binary_identity_note": str(self.native_binary)
                if self.native_binary
                else "Not exposed; see dependency identities and limitations",
                "dependencies": self.dependencies,
            },
            "execution": {
                "was_run": True,
                "utc": self.started_utc,
                "argv": [sys.executable, *sys.argv],
                "exit_code": exit_code,
                "platform": platform.platform(),
                "architecture": platform.machine(),
                "runtime": sys.version,
                "compiler_flags": [],
                "compiler_flags_note": "Vendor native flags not inferred from host Python",
            },
            "precision": self.precision,
            "exporter": {
                "path": str(self.exporter_path),
                "sha256": sha256(self.exporter_path),
                "commit": None,
            },
            "corpus": {
                "path": str(self.corpus_path),
                "sha256": sha256(self.corpus_path),
            },
            "artifacts": [
                {
                    "path": str(path.relative_to(self.output_dir))
                    if path.is_relative_to(self.output_dir)
                    else str(path),
                    "sha256": sha256(path),
                    "role": role,
                }
                for path, role in artifacts
            ],
            "coverage": {
                "profiles": profiles,
                "independent_input_count": len(
                    {item.case_id for item in self.observations}
                ),
                "native_observation_count": counts.get("ok", 0),
                "quatops_replay_count": 0,
                "status_counts": dict(counts),
                "record_count": len(self.observations),
            },
            "review": {
                "state": "captured_not_reviewed",
                "reviewer": None,
                "limitations": self.limitations,
            },
        }
        path = self.output_dir / "manifest.json"
        write_json(path, manifest)
        return path


def contract(
    *,
    units: ContractUnits | None = "radians",
    order: QuaternionOrder | None = None,
    action: MatrixAction | None = "column vectors",
    frame: FrameDirection | None = "active rotation in one numeric basis",
) -> ContractRecord:
    model = CaptureContract(
        units=units,
        quaternion_order=order,
        matrix_action=action,
        frame_direction=frame,
    )
    return {
        "units": model.units,
        "quaternion_order": model.quaternion_order,
        "matrix_action": model.matrix_action,
        "frame_direction": model.frame_direction,
    }
