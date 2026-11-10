"""Render or check the documentation view of the converter registry."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from quatops.experimental.converters import CONVERTERS, TARGET_DECLARATIONS
from quatops.experimental.semantics.assurance import (
    SemanticEvidenceArtifact,
    semantic_assurance_rows,
    validate_semantic_assurance,
)
from quatops.experimental.semantics.roles import SemanticContract


def _table_value(value: str) -> str:
    if any(character in value for character in "|\r\n"):
        raise ValueError(
            "converter registry values cannot contain Markdown table delimiters"
        )
    return value


def render_registry() -> str:
    lines = [
        "# Converter and evidence-target registry",
        "",
        "This generated view comes from `quatops.experimental.converters.CONVERTERS` and",
        "`TARGET_DECLARATIONS`. Representation evidence and semantic authority",
        "are separate; role-specific frame contracts are listed below. Edit",
        "the registry definitions, then regenerate",
        "this page with `python -m scripts.render_converter_registry --repository-root .`.",
        "",
        "| ID | Target API | Version | Role | Representation / order | Representation evidence | Semantic authority | Evidence state | Scope |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for declaration in TARGET_DECLARATIONS.values():
        profile = CONVERTERS.get(declaration.converter_id or "")
        layout = "—"
        if profile is not None:
            layout = f"{profile.representation} / {profile.component_order or '—'}"
        values = (
            declaration.target_id,
            f"{declaration.target}: {declaration.api}",
            declaration.version,
            declaration.role,
            layout,
            profile.representation_evidence if profile is not None else "—",
            profile.semantic_authority if profile is not None else "—",
            declaration.state,
            declaration.scope,
        )
        cells = " | ".join(_table_value(value) for value in values)
        lines.append(f"| {cells} |")
    from quatops.experimental.semantics.roles import SEMANTIC_CONTRACTS

    lines.extend(
        [
            "",
            "## Role-specific semantic contracts",
            "",
            "These contracts describe named roles and their required context; they do not make a generic quaternion field intrinsically world-, body-, or asset-oriented.",
            "",
            "| ID | Authority | Frame role | Description | Target family | Source kind | Source / version / section | Axes / handedness | Binding context |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for contract in SEMANTIC_CONTRACTS.values():
        values = (
            contract.contract_id,
            contract.authority,
            contract.frame_role,
            contract.role,
            contract.target_family or "—",
            contract.source_kind,
            " / ".join(
                item
                for item in (
                    contract.source_url or contract.application_source or "",
                    contract.source_version or "",
                    contract.source_section or "",
                )
                if item
            ),
            (
                f"{contract.positive_axes}; {contract.handedness}"
                if contract.positive_axes is not None
                else "—"
            ),
            contract.binding_kind
            + (
                f"; alignment={contract.alignment_purpose}"
                if contract.alignment_purpose
                else ""
            ),
        )
        lines.append("| " + " | ".join(_table_value(value) for value in values) + " |")
    lines.extend(
        [
            "",
            "## Semantic assurance matrix",
            "",
            "This matrix separates four claims. R is representation evidence; S is the scoped role source; B is an explicit alignment input whose physical truth remains caller/runtime supplied; N reports the native-proof scope. `numeric_action` covers quaternion/vector or static hierarchy behavior, `role_specific_action` exercises the named semantic role, and `runtime_space_relation` exercises a native named-space relation. Adapter conformance is stated separately. None of these proves physical asset or sensor alignment. R does not imply S or B.",
            "",
            "| Role | R: representation | S: role authority | B: physical alignment | N: native action | Scope and exclusions | Evidence IDs |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in semantic_assurance_rows():
        contract = cast(SemanticContract, row["contract"])
        evidence = cast(tuple[SemanticEvidenceArtifact, ...], row["evidence"])
        source_reference = contract.source_url or contract.application_source or "—"
        if contract.source_url is not None:
            source_reference = (
                f"[{contract.source_version} source]({contract.source_url})"
            )
        values = (
            contract.contract_id,
            str(row["representation"]),
            (
                f"{contract.source_kind}: {source_reference}; "
                f"{contract.source_section or '—'}; "
                f"claim: {contract.source_claim or 'application-defined'}"
            ),
            str(row["binding"]),
            str(row["native"]),
            f"{contract.assurance_scope or '—'} Excludes: {'; '.join(contract.exclusions) or '—'}",
            "; ".join(
                f"[{item.evidence_id}](../{item.artifact_path})" for item in evidence
            )
            or "—",
        )
        lines.append("| " + " | ".join(_table_value(value) for value in values) + " |")
    lines.extend(
        [
            "",
            "Representation-only profiles without a matching semantic role remain coefficient/action converters over caller-supplied frame definitions; their target name never supplies physical axes. Native evidence IDs identify immutable artifacts; each N entry names the narrow proof scope and whether adapter conformance was replayed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        required=True,
        help="repository root that contains docs/converter-registry.md",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the committed registry page differs from the source registry",
    )
    args = parser.parse_args()
    root = args.repository_root.resolve(strict=True)
    validate_semantic_assurance(str(root))
    output_path = root / "docs" / "converter-registry.md"
    rendered = render_registry()
    if args.check:
        if (
            not output_path.is_file()
            or output_path.read_text(encoding="utf-8") != rendered
        ):
            print("converter registry page is stale; regenerate it with this script")
            return 1
        print("converter registry page is current")
        return 0

    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        output.write(rendered)
    print(f"wrote {output_path.relative_to(root).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
