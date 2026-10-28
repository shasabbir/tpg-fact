from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


def system_ram_bytes() -> int | None:
    if os.name == "nt":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_physical)
        return None
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def nvidia_gpus() -> list[dict[str, object]]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    completed = subprocess.run(
        [
            executable,
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        name, memory_mib, driver = [part.strip() for part in line.split(",", 2)]
        rows.append(
            {
                "name": name,
                "memory_mib": int(memory_mib),
                "driver_version": driver,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a TRACE-SCV workstation.")
    parser.add_argument("--storage-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    storage = shutil.disk_usage(args.storage_root.resolve())
    ram = system_ram_bytes()
    gpus = nvidia_gpus()
    checks = {
        "logical_cpu_threads_at_least_16": (os.cpu_count() or 0) >= 16,
        "system_ram_at_least_48_gib": ram is not None and ram >= 48 * 2**30,
        "free_storage_at_least_300_gib": storage.free >= 300 * 2**30,
        "cuda_gpu_detected": bool(gpus),
        "gpu_vram_at_least_20_gib": any(
            int(gpu["memory_mib"]) >= 20 * 1024 for gpu in gpus
        ),
    }
    try:
        import torch

        torch_status: dict[str, object] = {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        }
    except ImportError:
        torch_status = {"installed": False, "cuda_available": False}
    checks["pytorch_cuda_available"] = bool(torch_status["cuda_available"])

    result = {
        "status": "pass" if all(checks.values()) else "failed",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpu_threads": os.cpu_count(),
        "system_ram_gib": round(ram / 2**30, 2) if ram else None,
        "storage_root": str(args.storage_root.resolve()),
        "free_storage_gib": round(storage.free / 2**30, 2),
        "gpus": gpus,
        "torch": torch_status,
        "checks": checks,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.strict and result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
