"""Render or check the documentation view of the converter registry."""

from __future__ import annotations

import argparse
from pathlib import Path

from quatops.experimental.converters import CONVERTERS, TARGET_DECLARATIONS


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
        "`TARGET_DECLARATIONS`. Edit the registry definitions, then regenerate",
        "this page with `python -m scripts.render_converter_registry --repository-root .`.",
        "",
        "These APIs are experimental. Evidence acceptance is not API stability;",
        "see [the experimental policy](experimental.md).",
        "",
        "| ID | Target API | Version | Role | Representation / order | Evidence state | Scope |",
        "| --- | --- | --- | --- | --- | --- | --- |",
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
            declaration.state,
            declaration.scope,
        )
        cells = " | ".join(_table_value(value) for value in values)
        lines.append(f"| {cells} |")
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
