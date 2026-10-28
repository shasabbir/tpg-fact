from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True, check=False
    )
    result = {
        "profile": "mocheg_scientific_logic_full_test_suite",
        "status": "pass" if completed.returncode == 0 else "fail",
        "exit_code": completed.returncode,
        "command": [sys.executable, "-m", "pytest", "-q"],
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    target = Path("outputs/tests/full_suite.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "exit_code": result["exit_code"]}, indent=2))
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
