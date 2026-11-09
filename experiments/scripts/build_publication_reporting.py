from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_scv.publication_reporting import build_publication_reporting


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the compact publication-confirmation result bundle."
    )
    parser.add_argument(
        "--retrieval-root", type=Path, default=Path("outputs/publication_confirmation/retrieval")
    )
    parser.add_argument(
        "--selector-root", type=Path, default=Path("outputs/publication_confirmation/selector_base")
    )
    parser.add_argument(
        "--completeness-root",
        type=Path,
        default=Path("outputs/publication_confirmation/completeness_switch"),
    )
    parser.add_argument(
        "--verifier-root", type=Path, default=Path("outputs/publication_confirmation/verifier")
    )
    parser.add_argument(
        "--research-root", type=Path, default=Path("../research/publication_confirmation")
    )
    parser.add_argument(
        "--generated-root", type=Path, default=Path("../paper/generated")
    )
    parser.add_argument(
        "--figure-root", type=Path, default=Path("../paper/figures")
    )
    args = parser.parse_args()
    result = build_publication_reporting(
        args.retrieval_root,
        args.selector_root,
        args.completeness_root,
        args.verifier_root,
        args.research_root,
        args.generated_root,
        args.figure_root,
    )
    print(
        json.dumps(
            {
                "profile": result["profile"],
                "status": result["status"],
                "formal_test_annotations_accessed": result[
                    "formal_test_annotations_accessed"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

