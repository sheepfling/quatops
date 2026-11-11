"""Shared runner for independent native direct-quaternion captures."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np

from scripts.evidence_format import dump_evidence_json
from scripts.evidence_json import read_evidence_document
from scripts.evidence_schema.converter_captures import (
    DirectLibraryInputBase,
    MuJoCoCapture,
    SpiceyPyCapture,
)
from scripts.evidence_schema.primitives import NATIVE_RETAINED_CASE_IDS
from scripts.evidence_schema.shared_inputs import (
    SharedCanonicalInputCapture,
    SharedQuaternionInput,
    resolve_orientation_selection,
)
from scripts.evidence_schema.utilities import sha256_file


@dataclass(frozen=True)
class DirectCaptureResult:
    """The native result and the actual binary that produced it."""

    witnesses: MuJoCoCapture | SpiceyPyCapture
    native_binary: Path


InputT = TypeVar("InputT", bound=DirectLibraryInputBase)


def run_direct_capture(
    capture_script: Path,
    input_type: type[InputT],
    capture: Callable[
        [InputT, SharedCanonicalInputCapture, list[SharedQuaternionInput]],
        DirectCaptureResult,
    ],
) -> None:
    """Select inputs and record provenance around target-specific native calls."""
    parser = argparse.ArgumentParser(description=capture_script.stem)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output-dir", type=Path)
    options = parser.parse_args()
    root = options.repository_root.resolve(strict=True)
    script = capture_script.resolve(strict=True)
    target_dir = script.parent
    target_dir.relative_to(
        root
    )  # Require the producer to belong to the declared checkout.
    input_path = (options.inputs or target_dir / "converter_inputs.json").resolve()
    inputs = input_type.model_validate(read_evidence_document(input_path), strict=True)
    shared_path = root / inputs.shared_canonical_inputs
    shared = SharedCanonicalInputCapture.model_validate(
        read_evidence_document(shared_path), strict=True
    )
    selected = [
        row
        for row in resolve_orientation_selection(
            shared, inputs.retained_case_ids, required_case_ids={"mixed_a", "mixed_b"}
        )
        if row.case_id in NATIVE_RETAINED_CASE_IDS
    ]
    if tuple(row.case_id for row in selected) != NATIVE_RETAINED_CASE_IDS:
        raise ValueError(
            "native orientation selection differs from the retained recipe"
        )

    result = capture(inputs, shared, selected)
    binary = result.native_binary.resolve(strict=True)
    output = (
        options.output_dir
        or Path(tempfile.mkdtemp(prefix=f"quatops-{target_dir.parent.name}-capture-"))
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    witnesses_path = output / "witnesses.json"
    witnesses_path.write_text(
        dump_evidence_json(result.witnesses.model_dump(mode="json", exclude_none=True)),
        encoding="utf-8",
        newline="\n",
    )
    contract = target_dir / "semantic_contract.json"
    lockfile = root / "experiments/converters/python/uv.lock"
    target_name = target_dir.parent.name
    manifest = {
        "schema_version": "1.0",
        "semantic_contract": {
            "file": contract.relative_to(root).as_posix(),
            "sha256": sha256_file(contract),
        },
        "review_state": "Native capture; repository replay acceptance is recorded by tests",
        "execution": {
            "was_run": True,
            "utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "numpy_version": np.__version__,
        },
        "target": {
            **result.witnesses.target.model_dump(mode="json", exclude_none=True),
            "native_scalar_dtype": "float64",
        },
        "selection": {
            "retained_orientation_cases": len(selected),
            "native_created_orientations": 1,
            "noncommuting_product_orders": 2,
            "native_basis_images_per_orientation": 3,
            "native_mixed_probe_case_id": "mixed_a",
        },
        "native_binary": {"path": str(binary), "sha256": sha256_file(binary)},
        "provenance": {
            "capture_script_sha256": sha256_file(script),
            "shared_capture_script_sha256": sha256_file(Path(__file__)),
            "converter_input_spec_sha256": sha256_file(input_path),
            "selected_inputs_sha256": sha256_file(shared_path),
            "witnesses_sha256": sha256_file(witnesses_path),
            "lockfile_sha256": sha256_file(lockfile),
            "invocation_template": (
                f"uv run --locked --project experiments/converters/python --extra {target_name} "
                f"--python 3.13.14 python {script.relative_to(root).as_posix()} "
                "--repository-root <repository>"
            ),
        },
    }
    (output / "manifest.json").write_text(
        dump_evidence_json(manifest), encoding="utf-8", newline="\n"
    )
    print(
        json.dumps({"output_dir": str(output), "target": result.witnesses.target.name})
    )
