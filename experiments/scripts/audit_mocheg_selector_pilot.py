from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_scv.selection.mocheg_audit import audit_mocheg_selector_pilot


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit saved MOCHEG selector outputs.")
    parser.add_argument(
        "--features", type=Path, default=Path("outputs/mocheg/pilot_features.npz")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/mocheg/selector_pilot")
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("outputs/mocheg/selector_pilot/audit.json"),
    )
    args = parser.parse_args()
    result = audit_mocheg_selector_pilot(args.features, args.output_dir)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
