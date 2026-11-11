from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


def _local_hardware() -> dict[str, object]:
    try:
        import torch

        devices = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "bytes": int(torch.cuda.get_device_properties(index).total_memory),
            }
            for index in range(torch.cuda.device_count())
        ]
        return {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": devices,
        }
    except Exception as error:  # pragma: no cover - environment dependent
        return {"cuda_available": False, "error": f"{type(error).__name__}: {error}"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit feasibility of the pinned official AVerImaTeC judge.")
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/external_confirmation/averimatec/official_evaluator_feasibility.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    info = HfApi().model_info(args.model, files_metadata=True)
    weight_files = [
        sibling
        for sibling in info.siblings
        if sibling.rfilename.endswith((".safetensors", ".bin")) and sibling.size
    ]
    weight_bytes = sum(int(sibling.size) for sibling in weight_files)
    hardware = _local_hardware()
    aggregate_vram = sum(int(device["bytes"]) for device in hardware.get("cuda_devices", []))
    fits_weights_in_aggregate_vram = weight_bytes <= aggregate_vram
    status = (
        "weights_fit_in_aggregate_vram"
        if fits_weights_in_aggregate_vram
        else "not_feasible_as_published_on_local_gpu_without_offload_or_quantization"
    )
    audit = {
        "status": status,
        "model": args.model,
        "model_revision": info.sha,
        "weight_files": len(weight_files),
        "weight_bytes": weight_bytes,
        "local_hardware": hardware,
        "fits_weights_in_aggregate_vram": fits_weights_in_aggregate_vram,
        "published_launcher": {
            "gpus": 2,
            "host_memory_gb": 50,
            "precision": "bfloat16",
            "source": "pinned shared-task scripts/eval_offline.sh and prepare_submission/eval_offline.py",
        },
        "official_evaluation_executed": False,
        "replacement_score_labeled_official": False,
        "test_annotations_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
