"""Generate the compact, target-neutral converter evidence data product."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from scripts.evidence_schema.product import (
        build_evidence_product,
        render_evidence_product,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="write the generated product to this path; omitted to report a summary only",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the existing product differs from regenerated evidence",
    )
    args = parser.parse_args()

    product = build_evidence_product(ROOT)
    if args.check:
        if args.output is None:
            raise SystemExit("--check requires --output")
        rendered = render_evidence_product(product)
        output_path = args.output.resolve()
        if not output_path.is_file():
            raise SystemExit(f"crystallized product does not exist: {output_path}")
        existing = output_path.read_text(encoding="utf-8")
        if existing != rendered:
            raise SystemExit(f"crystallized product is stale: {output_path}")
        print(f"PASS {output_path}: {len(rendered.encode('utf-8'))} bytes")
        return

    witness_count = sum(len(row.witnesses) for row in product.witness_sets)
    if args.output is None:
        print(
            f"Validated {len(product.records)} target/version records, "
            f"{len(product.witness_sets)} shared witness sets, "
            f"{witness_count} witnesses; no generated file retained"
        )
        return

    rendered = render_evidence_product(product)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(rendered.encode("utf-8"))
    print(
        f"Wrote {output_path}: {len(product.records)} target/version records, "
        f"{len(product.witness_sets)} shared witness sets, {witness_count} witnesses, "
        f"{len(rendered.encode('utf-8'))} bytes"
    )


if __name__ == "__main__":
    main()
