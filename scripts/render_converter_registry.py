"""Render or check the documentation view of the converter registry."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
from textwrap import fill
from typing import cast

from quatops.experimental.converters import CONVERTERS, TARGET_DECLARATIONS
from quatops.experimental.semantics.assurance import (
    SemanticEvidenceArtifact,
    semantic_assurance_rows,
    validate_semantic_assurance,
)
from quatops.experimental.semantics.evidence_claims import SemanticEvidenceClaim
from quatops.experimental.semantics.roles import SemanticContract


def _wrapped(label: str, value: str) -> str:
    return fill(
        f"{label}{value}",
        width=100,
        subsequent_indent="  ",
        break_long_words=False,
        break_on_hyphens=False,
    )


def _cell(value: str) -> str:
    if "|" in value or "\n" in value or "\r" in value:
        raise ValueError("registry table cells cannot contain pipes or newlines")
    return escape(value, quote=False)


def render_registry() -> str:
    lines = [
        "# Converter and evidence-target registry",
        "",
        "This generated view comes from the experimental converter declarations.",
        "Representation evidence and semantic authority are recorded separately.",
        "Regenerate it with `python -m scripts.render_converter_registry --repository-root .`.",
        "",
        "## Target profiles",
        "",
        "| Profile | Target, API, and version | Role and representation | Evidence, authority, and state | Scope |",
        "| --- | --- | --- | --- | --- |",
    ]
    for declaration in TARGET_DECLARATIONS.values():
        profile = CONVERTERS.get(declaration.converter_id or "")
        representation = (
            f"{profile.representation} / {profile.component_order or '—'}"
            if profile is not None
            else "—"
        )
        evidence = profile.representation_evidence if profile is not None else "—"
        authority = profile.semantic_authority if profile is not None else "—"
        cells = (
            f"`{declaration.target_id}`",
            f"{declaration.target}: {declaration.api}; {declaration.version}",
            f"{declaration.role}; {representation}",
            f"evidence={evidence}; semantic authority={authority}; state={declaration.state}",
            declaration.scope,
        )
        lines.append("| " + " | ".join(_cell(cell) for cell in cells) + " |")

    lines.extend(
        [
            "",
            "## Semantic assurance matrix",
            "",
            "R identifies representation evidence. S records the declared role source. B is a",
            "caller- or runtime-supplied physical alignment. N states the native proof scope.",
            "Adapter conformance is reported separately; these claims do not establish physical",
            "asset or sensor alignment. Primary sources are indexed in the",
            "[semantic source audit](semantic-source-audit.md).",
            "Every native claim names an operation, context, version, and retained witness IDs.",
            "Repository verification checks those against the validated capture contents.",
            "Profile replay applies only to the listed operation and scope; it does not prove",
            "the entire role. LOCAL_FLOOR and VIEW have",
            "no native execution claim in the current fixtures.",
            "",
        ]
    )
    for row in semantic_assurance_rows():
        contract = cast(SemanticContract, row["contract"])
        evidence = cast(tuple[SemanticEvidenceArtifact, ...], row["evidence"])
        claims = cast(tuple[SemanticEvidenceClaim, ...], row["claims"])
        source = (
            " / ".join(
                value
                for value in (
                    contract.source_version,
                    contract.source_section,
                    contract.application_source,
                )
                if value
            )
            or "—"
        )
        axes = (
            f"{contract.positive_axes}; {contract.handedness}"
            if contract.positive_axes is not None
            else "—"
        )
        binding = contract.binding_kind
        if contract.alignment_purpose:
            binding += f"; alignment={contract.alignment_purpose}"
        replayed_ids = {
            item.evidence_id for item in evidence if item.adapter_conformance_checked
        }
        conformance = (
            "; ".join(
                f"{claim.converter_id}: {claim.operation} [{claim.scope}] @ {claim.version}"
                for claim in claims
                if claim.evidence_id in replayed_ids
            )
            or "not demonstrated"
        )
        evidence_links = (
            "; ".join(
                f"[{item.evidence_id}](../{item.artifact_path})" for item in evidence
            )
            or "—"
        )
        lines.extend(
            [
                f"### `{contract.contract_id}`",
                "",
                _wrapped(
                    "- Declaration: ",
                    f"{contract.role}; authority={contract.authority}; "
                    f"frame role={contract.frame_role or '—'}; "
                    f"target={contract.target_family or '—'}",
                ),
                _wrapped("- Source: ", f"{contract.source_kind}; {source}"),
                _wrapped("- Axes and binding: ", f"{axes}; {binding}"),
                _wrapped("- Claim: ", contract.source_claim or "application-defined"),
                _wrapped("- Scope: ", contract.assurance_scope or "—"),
                _wrapped("- Exclusions: ", "; ".join(contract.exclusions) or "—"),
                _wrapped(
                    "- Assurance: ",
                    f"R: {row['representation']}; S: {row['source']}; "
                    f"B: {row['binding']}; N: {row['native']}",
                ),
                _wrapped("- Adapter conformance: ", conformance),
                _wrapped(
                    "- Witnesses: ",
                    "; ".join(
                        f"{claim.evidence_id} / {claim.operation} [{claim.scope}]: "
                        + ", ".join(claim.witness_ids)
                        for claim in claims
                    )
                    or "-",
                ),
                _wrapped("- Evidence: ", evidence_links),
                "",
            ]
        )
    lines.extend(
        [
            "Representation-only profiles remain coefficient/action converters over",
            "caller-supplied frame definitions. A target name never supplies physical axes.",
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
