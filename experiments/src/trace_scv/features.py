from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from trace_scv.data.io import read_packets
from trace_scv.image_io import load_rgb_on_white
from trace_scv.schema import EvidenceItem


NODE_TYPES = {"padding": 0, "text": 1, "caption": 2, "table": 3, "figure": 4, "image": 5}
LABELS = {"refuted": 0, "supported": 1}
ID_TO_LABEL = {value: key for key, value in LABELS.items()}
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class Embedder(Protocol):
    dim: int

    def text(self, value: str) -> np.ndarray: ...

    def image(self, path: str | None, fallback: str) -> np.ndarray: ...

    def texts(self, values: list[str]) -> list[np.ndarray]: ...

    def images(self, values: list[tuple[str | None, str]]) -> list[np.ndarray]: ...


class HashEmbedder:
    """Deterministic dependency-light backend for tests and CPU pilots."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def text(self, value: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = _TOKEN_RE.findall(value.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            integer = int.from_bytes(digest, "little")
            vector[integer % self.dim] += 1.0 if integer & 1 else -1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def image(self, path: str | None, fallback: str) -> np.ndarray:
        vector = self.text(fallback)
        if not path or not Path(path).exists():
            return vector
        try:
            image = load_rgb_on_white(path)
            rgb = image.resize((64, 64))
            pixels = np.asarray(rgb, dtype=np.float32) / 255.0
            rgb.close()
            image.close()
            stats = np.concatenate(
                [
                    pixels.mean(axis=(0, 1)),
                    pixels.std(axis=(0, 1)),
                    *[
                        np.histogram(pixels[:, :, channel], bins=16, range=(0, 1), density=True)[0]
                        for channel in range(3)
                    ],
                ]
            ).astype(np.float32)
            stats /= max(float(np.linalg.norm(stats)), 1e-8)
            vector[: len(stats)] += stats[: self.dim]
            vector /= max(float(np.linalg.norm(vector)), 1e-8)
        except Exception:
            return vector
        return vector

    def texts(self, values: list[str]) -> list[np.ndarray]:
        return [self.text(value) for value in values]

    def images(self, values: list[tuple[str | None, str]]) -> list[np.ndarray]:
        return [self.image(path, fallback) for path, fallback in values]


class HuggingFaceEmbedder:
    """Frozen BGE text and CLIP vision encoder used by formal runs."""

    def __init__(
        self,
        dim: int = 512,
        text_model: str = "BAAI/bge-small-en-v1.5",
        vision_model: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("Install trace-scv[formal] for Hugging Face features") from error
        self.torch = torch
        self.dim = dim
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.text_model = AutoModel.from_pretrained(text_model).to(self.device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained(vision_model)
        self.clip_model = CLIPModel.from_pretrained(vision_model).to(self.device).eval()

    def _fit(self, values: np.ndarray) -> np.ndarray:
        output = np.zeros(self.dim, dtype=np.float32)
        length = min(self.dim, values.shape[-1])
        output[:length] = values[:length]
        output /= max(float(np.linalg.norm(output)), 1e-8)
        return output

    def _fit_batch(self, values: np.ndarray) -> np.ndarray:
        output = np.zeros((values.shape[0], self.dim), dtype=np.float32)
        length = min(self.dim, values.shape[-1])
        output[:, :length] = values[:, :length]
        norms = np.linalg.norm(output, axis=1, keepdims=True)
        return output / np.maximum(norms, 1e-8)

    def text(self, value: str) -> np.ndarray:
        return self.texts([value])[0]

    def texts(self, values: list[str]) -> list[np.ndarray]:
        rows: list[np.ndarray] = []
        for start in range(0, len(values), self.batch_size):
            batch = self.tokenizer(
                values[start : start + self.batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with self.torch.inference_mode():
                hidden = self.text_model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            rows.extend(self._fit_batch(pooled.float().cpu().numpy()))
        return rows

    def image(self, path: str | None, fallback: str) -> np.ndarray:
        return self.images([(path, fallback)])[0]

    def images(self, values: list[tuple[str | None, str]]) -> list[np.ndarray]:
        output: list[np.ndarray | None] = [None] * len(values)
        valid: list[tuple[int, str]] = []
        fallback_indices: list[int] = []
        for index, (path, _) in enumerate(values):
            if not path or not Path(path).exists():
                fallback_indices.append(index)
                continue
            valid.append((index, path))
        for start in range(0, len(valid), self.batch_size):
            chunk = valid[start : start + self.batch_size]
            loaded: list[tuple[int, object]] = []
            for index, path in chunk:
                try:
                    loaded.append((index, load_rgb_on_white(path)))
                except Exception:
                    fallback_indices.append(index)
            if not loaded:
                continue
            batch = self.clip_processor(
                images=[image for _, image in loaded], return_tensors="pt"
            ).to(self.device)
            with self.torch.inference_mode():
                embedded = self.clip_model.get_image_features(**batch)
            fitted = self._fit_batch(embedded.float().cpu().numpy())
            for (index, image), row in zip(loaded, fitted, strict=True):
                output[index] = row
                image.close()
        if fallback_indices:
            text_fallbacks = [
                (index, values[index][1])
                for index in fallback_indices
                if values[index][1].strip()
            ]
            if text_fallbacks:
                fallback_rows = self.texts([value for _, value in text_fallbacks])
                for (index, _), row in zip(text_fallbacks, fallback_rows, strict=True):
                    output[index] = row
            for index in fallback_indices:
                if output[index] is None:
                    output[index] = np.zeros(self.dim, dtype=np.float32)
        return [row if row is not None else np.zeros(self.dim, dtype=np.float32) for row in output]


def make_embedder(backend: str, dim: int) -> Embedder:
    if backend == "hash":
        return HashEmbedder(dim)
    if backend == "hf":
        return HuggingFaceEmbedder(dim)
    raise ValueError(f"unknown feature backend: {backend}")


@dataclass(frozen=True)
class FeatureStore:
    claim_features: np.ndarray
    evidence_features: np.ndarray
    visual_features: np.ndarray
    visual_mask: np.ndarray
    evidence_mask: np.ndarray
    gold_mask: np.ndarray
    node_types: np.ndarray
    adjacency: np.ndarray
    labels: np.ndarray
    claim_ids: list[str]
    group_ids: list[str]
    evidence_ids: list[list[str]]
    evidence_text: list[list[str]]


def build_feature_store(
    packet_path: Path,
    output_path: Path,
    *,
    backend: str = "hash",
    dim: int = 256,
    max_candidates: int = 12,
    embedder: Embedder | None = None,
) -> dict:
    packets = read_packets(packet_path)
    embedder = embedder or make_embedder(backend, dim)
    count = len(packets)
    claim_features = np.zeros((count, dim), dtype=np.float32)
    evidence_features = np.zeros((count, max_candidates, dim), dtype=np.float32)
    visual_features = np.zeros((count, max_candidates, dim), dtype=np.float32)
    visual_mask = np.zeros((count, max_candidates), dtype=np.bool_)
    evidence_mask = np.zeros((count, max_candidates), dtype=np.bool_)
    gold_mask = np.zeros((count, max_candidates), dtype=np.float32)
    node_types = np.zeros((count, max_candidates), dtype=np.int64)
    adjacency = np.zeros((count, max_candidates, max_candidates), dtype=np.float32)
    labels = np.zeros(count, dtype=np.int64)
    claim_ids: list[str] = []
    group_ids: list[str] = []
    evidence_ids: list[list[str]] = []
    evidence_text: list[list[str]] = []
    claim_values = list(dict.fromkeys(packet.claim_text for packet in packets))
    text_cache = dict(zip(claim_values, embedder.texts(claim_values), strict=True))
    unique_evidence: dict[tuple[str, str | None, str], EvidenceItem] = {}
    for packet in packets:
        for item in packet.evidence[:max_candidates]:
            unique_evidence.setdefault((item.modality, item.media_path, item.content), item)
    evidence_keys = list(unique_evidence)
    evidence_cache = dict(
        zip(
            evidence_keys,
            embedder.texts([unique_evidence[key].content for key in evidence_keys]),
            strict=True,
        )
    )
    visual_keys = [
        key
        for key, item in unique_evidence.items()
        if item.modality in {"figure", "image", "table"}
        and item.media_path
        and Path(item.media_path).is_file()
    ]
    visual_cache = dict(
        zip(
            visual_keys,
            embedder.images([(unique_evidence[key].media_path, "") for key in visual_keys]),
            strict=True,
        )
    )

    for row_index, packet in enumerate(packets):
        claim_features[row_index] = text_cache[packet.claim_text]
        labels[row_index] = LABELS[packet.label]
        claim_ids.append(packet.claim_id)
        group_ids.append(packet.group_id or packet.claim_id)
        row_ids: list[str] = []
        row_text: list[str] = []
        for item_index, item in enumerate(packet.evidence[:max_candidates]):
            cache_key = (item.modality, item.media_path, item.content)
            evidence_features[row_index, item_index] = evidence_cache[cache_key]
            if cache_key in visual_cache:
                visual_features[row_index, item_index] = visual_cache[cache_key]
                visual_mask[row_index, item_index] = True
            evidence_mask[row_index, item_index] = True
            gold_mask[row_index, item_index] = float(item.gold)
            node_types[row_index, item_index] = NODE_TYPES[item.modality]
            row_ids.append(item.evidence_id)
            row_text.append(item.content)
        valid = int(evidence_mask[row_index].sum())
        adjacency[row_index, :valid, :valid] = 1.0
        evidence_ids.append(row_ids)
        evidence_text.append(row_text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        claim_features=claim_features,
        evidence_features=evidence_features,
        visual_features=visual_features,
        visual_mask=visual_mask,
        evidence_mask=evidence_mask,
        gold_mask=gold_mask,
        node_types=node_types,
        adjacency=adjacency,
        labels=labels,
    )
    sidecar = output_path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "backend": backend,
                "dim": dim,
                "max_candidates": max_candidates,
                "dual_channel": True,
                "visual_available_items": int(visual_mask.sum()),
                "claim_ids": claim_ids,
                "group_ids": group_ids,
                "evidence_ids": evidence_ids,
                "evidence_text": evidence_text,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {"count": count, "backend": backend, "dim": dim, "path": str(output_path)}


def load_feature_store(path: Path) -> FeatureStore:
    arrays = np.load(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    evidence_features = arrays["evidence_features"]
    visual_features = (
        arrays["visual_features"]
        if "visual_features" in arrays.files
        else np.zeros_like(evidence_features)
    )
    visual_mask = (
        arrays["visual_mask"]
        if "visual_mask" in arrays.files
        else np.zeros(evidence_features.shape[:2], dtype=np.bool_)
    )
    return FeatureStore(
        claim_features=arrays["claim_features"],
        evidence_features=evidence_features,
        visual_features=visual_features,
        visual_mask=visual_mask,
        evidence_mask=arrays["evidence_mask"],
        gold_mask=arrays["gold_mask"],
        node_types=arrays["node_types"],
        adjacency=arrays["adjacency"],
        labels=arrays["labels"],
        claim_ids=meta["claim_ids"],
        group_ids=meta["group_ids"],
        evidence_ids=meta["evidence_ids"],
        evidence_text=meta["evidence_text"],
    )
