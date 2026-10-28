from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_native_command_guard_rejects_nonzero_exit() -> None:
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("Windows PowerShell is unavailable")

    experiments_root = Path(__file__).resolve().parents[1]
    guard = experiments_root / "scripts" / "native_command_guard.ps1"
    escaped_guard = str(guard).replace("'", "''")
    escaped_python = sys.executable.replace("'", "''")
    command = (
        f". '{escaped_guard}'; "
        "Invoke-CheckedNative -Description 'regression probe' -Command { "
        f"& '{escaped_python}' -c 'import sys; sys.exit(7)' "
        "}"
    )

    completed = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "regression probe failed with exit code 7" in (
        completed.stdout + completed.stderr
    )
