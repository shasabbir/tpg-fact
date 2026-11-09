from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_scv.publication_retrieval import (
    audit_publication_retrieval,
    run_publication_retrieval,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the test-sealed publication retrieval fusion experiment."
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[2])
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("experiments/configs/publication_confirmation.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/publication_confirmation/retrieval"),
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    protocol = args.protocol if args.protocol.is_absolute() else project_root / args.protocol
    output = args.output if args.output.is_absolute() else project_root / args.output
    result = run_publication_retrieval(project_root, protocol.resolve(), output.resolve())
    audit = audit_publication_retrieval(output.resolve())
    audit_path = output.resolve() / "retrieval.audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "gate": result["gate"],
                "selected_text_rrf_weight": result["selected_text_rrf_weight"],
                "selected_image_siglip_weight": result["selected_image_siglip_weight"],
                "validation": result["population_results"]["validation"],
                "audit": audit,
            },
            indent=2,
        )
    )
    if audit["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

