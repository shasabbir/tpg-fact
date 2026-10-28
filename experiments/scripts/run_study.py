from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_scv.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DACS-SCV from data to audited predictions.")
    parser.add_argument("--profile", choices=("smoke", "pilot", "formal"), default="smoke")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(
        json.dumps(
            run_pipeline(
                args.project_root.resolve(),
                profile=args.profile,
                download=args.download,
                device=args.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
