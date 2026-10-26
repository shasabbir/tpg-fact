from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data.preflight import run_preflight
from .smoke import run_smoke


def smoke_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/smoke"))
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.output), indent=2))


def preflight_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/data_preflight.json"))
    args = parser.parse_args()
    result = run_preflight()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)
