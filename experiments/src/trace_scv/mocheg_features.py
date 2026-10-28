from __future__ import annotations

import gc
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from trace_scv.data.io import sha256_file
from trace_scv.image_io import load_rgb_on_white
from trace_scv.features import HashEmbedder


@dataclass(frozen=True, slots=True)
class MochegEncoderConfig:
    text_model: str = "BAAI/bge-m3"
    text_revision: str = "5617a9f61b028005a4858fdac845db406aefb181"
    vision_model: str = "google/siglip2-base-patch16-224"
    vision_revision: str = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
    text_batch_size: int = 32
    image_batch_size: int = 64
    max_text_tokens: int = 512
    device: str = "auto"


class MochegEncoder(Protocol):
    def encode_bge(self, values: Sequence[str]) -> np.ndarray: ...

    def encode_siglip(
        self, texts: Sequence[str], image_paths: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def identity(self) -> dict[str, object]: ...


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


class TransformersMochegEncoder:
    """Pinned frozen BGE-M3 and SigLIP2 encoder for workstation execution."""

    def __init__(self, config: MochegEncoderConfig) -> None:
        self.config = config

    def _runtime(self):
        try:
            import torch
            from transformers import AutoModel, AutoProcessor, AutoTokenizer
        except ImportError as error:  # pragma: no cover - environment guidance
            raise RuntimeError("install trace-scv[formal] before feature extraction") from error
        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot access it")
        return torch, AutoModel, AutoProcessor, AutoTokenizer, device

    def encode_bge(self, values: Sequence[str]) -> np.ndarray:
        torch, AutoModel, _, AutoTokenizer, device = self._runtime()
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.text_model, revision=self.config.text_revision
        )
        model = AutoModel.from_pretrained(
            self.config.text_model,
            revision=self.config.text_revision,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device).eval()
        rows: list[np.ndarray] = []
        for start in range(0, len(values), self.config.text_batch_size):
            batch = tokenizer(
                list(values[start : start + self.config.text_batch_size]),
                padding=True,
                truncation=True,
                max_length=self.config.max_text_tokens,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=device == "cuda"
                ):
                    hidden = model(**batch).last_hidden_state
                    mask = batch["attention_mask"].unsqueeze(-1)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            rows.append(pooled.float().cpu().numpy())
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return _normalize(np.concatenate(rows, axis=0))

    def encode_siglip(
        self, texts: Sequence[str], image_paths: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        torch, AutoModel, AutoProcessor, _, device = self._runtime()
        processor = AutoProcessor.from_pretrained(
            self.config.vision_model, revision=self.config.vision_revision
        )
        model = AutoModel.from_pretrained(
            self.config.vision_model,
            revision=self.config.vision_revision,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device).eval()
        text_rows: list[np.ndarray] = []
        for start in range(0, len(texts), self.config.text_batch_size):
            batch = processor(
                text=list(texts[start : start + self.config.text_batch_size]),
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                embedded = model.get_text_features(**batch)
            text_rows.append(embedded.float().cpu().numpy())

        image_rows: list[np.ndarray] = []
        for start in range(0, len(image_paths), self.config.image_batch_size):
            images = [
                load_rgb_on_white(path)
                for path in image_paths[start : start + self.config.image_batch_size]
            ]
            try:
                batch = processor(images=images, return_tensors="pt").to(device)
                with torch.inference_mode():
                    embedded = model.get_image_features(**batch)
                image_rows.append(embedded.float().cpu().numpy())
            finally:
                for image in images:
                    image.close()
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return (
            _normalize(np.concatenate(text_rows, axis=0)),
            _normalize(np.concatenate(image_rows, axis=0)),
        )

    def identity(self) -> dict[str, object]:
        return asdict(self.config)


class HashMochegEncoder:
    """Dependency-light backend for end-to-end software validation only."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.embedder = HashEmbedder(dim)

    def encode_bge(self, values: Sequence[str]) -> np.ndarray:
        return np.asarray(self.embedder.texts(list(values)), dtype=np.float32)

    def encode_siglip(
        self, texts: Sequence[str], image_paths: Sequence[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        text = np.asarray(self.embedder.texts(list(texts)), dtype=np.float32)
        images = np.asarray(
            self.embedder.images([(path, "") for path in image_paths]),
            dtype=np.float32,
        )
        return text, images

    def identity(self) -> dict[str, object]:
        return {
            "backend": "hash_smoke_only",
            "dim": self.dim,
            "scientific_results_allowed": False,
        }


def _read_packets(paths: Sequence[Path]) -> list[dict[str, object]]:
    packets: list[dict[str, object]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            packets.extend(json.loads(line) for line in handle if line.strip())
    return packets


def build_mocheg_pair_features(
    packet_paths: Sequence[Path],
    output_path: Path,
    *,
    encoder: MochegEncoder,
) -> dict[str, object]:
    """Build matched item-level features without exposing verdict labels to inputs."""
    started = time.perf_counter()
    encoder_identity = encoder.identity()
    peak_gpu_memory_bytes: int | None = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch = None
    packets = _read_packets(packet_paths)
    claim_texts = list(dict.fromkeys(str(row["claim"]) for row in packets))
    text_contents = list(
        dict.fromkeys(
            str(candidate["content"])
            for row in packets
            for candidate in row["candidates"]
            if candidate["available"] and candidate["modality"] == "text"
        )
    )
    image_paths = list(
        dict.fromkeys(
            str(candidate["media_path"])
            for row in packets
            for candidate in row["candidates"]
            if candidate["available"] and candidate["modality"] == "image"
        )
    )
    bge_values = [*claim_texts, *text_contents]
    bge_embeddings = encoder.encode_bge(bge_values)
    bge = dict(zip(bge_values, bge_embeddings, strict=True))
    siglip_text, siglip_image = encoder.encode_siglip(claim_texts, image_paths)
    siglip_claims = dict(zip(claim_texts, siglip_text, strict=True))
    siglip_images = dict(zip(image_paths, siglip_image, strict=True))

    features: list[list[float]] = []
    gold: list[int] = []
    metadata: list[dict[str, object]] = []
    claim_candidate_counts = {
        str(row["claim_id"]): sum(bool(item["available"]) for item in row["candidates"])
        for row in packets
    }
    for row in packets:
        claim_id = str(row["claim_id"])
        claim = str(row["claim"])
        available_count = claim_candidate_counts[claim_id]
        for candidate in row["candidates"]:
            if not candidate["available"]:
                continue
            modality = str(candidate["modality"])
            if modality == "text":
                content = str(candidate["content"])
                similarity = float(np.dot(bge[claim], bge[content]))
                candidate_tokens = len(content.split())
            elif modality == "image":
                media_path = str(candidate["media_path"])
                similarity = float(
                    np.dot(siglip_claims[claim], siglip_images[media_path])
                )
                candidate_tokens = 0
            else:
                raise ValueError(f"unsupported MOCHEG modality: {modality}")
            features.append(
                [
                    similarity,
                    similarity * similarity,
                    float(modality == "text"),
                    float(modality == "image"),
                    math.log1p(candidate_tokens) / 10.0,
                    math.log1p(len(claim.split())) / 10.0,
                    math.log1p(available_count) / 10.0,
                ]
            )
            gold.append(int(bool(candidate["gold"])))
            metadata.append(
                {
                    "claim_id": claim_id,
                    "evidence_id": candidate["evidence_id"],
                    "modality": modality,
                    "split": row["split"],
                    "gold": bool(candidate["gold"]),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=np.asarray(features, dtype=np.float32),
        gold=np.asarray(gold, dtype=np.int8),
    )
    sidecar = output_path.with_suffix(".json")
    sidecar_payload = {
        "feature_names": [
            "semantic_similarity",
            "semantic_similarity_squared",
            "is_text",
            "is_image",
            "log_text_token_count",
            "log_claim_token_count",
            "log_available_candidate_count",
        ],
        "encoder": encoder_identity,
        "packet_files": {
            str(path): sha256_file(path) for path in packet_paths
        },
        "records": metadata,
        "formal_test_annotations_accessed": False,
    }
    sidecar.write_text(
        json.dumps(sidecar_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if torch is not None and torch.cuda.is_available():
        peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated())
    implementation_path = Path(__file__).resolve()
    data_implementation_path = implementation_path.parent / "data" / "mocheg.py"
    manifest = {
        "status": "pass",
        "rows": len(metadata),
        "gold_rows": int(sum(gold)),
        "claims": len({row["claim_id"] for row in metadata}),
        "feature_width": len(features[0]) if features else 0,
        "features_sha256": sha256_file(output_path),
        "features_bytes": output_path.stat().st_size,
        "sidecar_sha256": sha256_file(sidecar),
        "sidecar_bytes": sidecar.stat().st_size,
        "encoder": encoder_identity,
        "packet_files": {
            str(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in packet_paths
        },
        "implementation_files": {
            str(implementation_path): sha256_file(implementation_path),
            str(data_implementation_path): sha256_file(data_implementation_path),
        },
        "runtime": {
            "wall_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": getattr(torch, "__version__", None),
            "torch_cuda": getattr(getattr(torch, "version", None), "cuda", None),
            "cuda_device": (
                torch.cuda.get_device_name(0)
                if torch is not None and torch.cuda.is_available()
                else None
            ),
        },
        "formal_test_annotations_accessed": False,
        "scientific_results_allowed": (
            encoder_identity.get("scientific_results_allowed", True)
        ),
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
