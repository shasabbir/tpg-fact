from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_scv.publication_candidates import (
    audit_publication_candidates,
    build_publication_candidates,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and audit selector-ready publication candidate packets."
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[2])
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("experiments/configs/publication_confirmation.yaml"),
    )
    parser.add_argument(
        "--retrieval-result",
        type=Path,
        default=Path("experiments/outputs/publication_confirmation/retrieval"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/publication_confirmation/retrieval_candidates"),
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    resolve = lambda value: value if value.is_absolute() else project_root / value
    output = resolve(args.output).resolve()
    manifest = build_publication_candidates(
        project_root,
        resolve(args.protocol).resolve(),
        resolve(args.retrieval_result).resolve(),
        output,
    )
    audit = audit_publication_candidates(output)
    (output / "retrieval.audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "counts": manifest["counts"],
                "packet_ceiling": manifest["packet_ceiling"],
                "runtime": manifest["runtime"],
                "audit": audit,
            },
            indent=2,
        )
    )
    if audit["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

