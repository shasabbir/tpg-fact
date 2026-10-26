from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np


def cosine_rank(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
    document_ids: Sequence[str],
) -> list[tuple[str, float]]:
    """Rank precomputed embeddings with deterministic cosine similarity."""
    query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    documents = np.asarray(document_embeddings, dtype=np.float32)
    if documents.ndim != 2 or documents.shape[0] != len(document_ids):
        raise ValueError("document_embeddings and document_ids must align")
    if query.shape[1] != documents.shape[1]:
        raise ValueError("query and document embeddings must have the same width")
    query /= np.linalg.norm(query, axis=1, keepdims=True).clip(min=1e-12)
    documents = documents / np.linalg.norm(documents, axis=1, keepdims=True).clip(
        min=1e-12
    )
    scores = documents @ query[0]
    order = np.argsort(-scores, kind="stable")
    return [(str(document_ids[index]), float(scores[index])) for index in order]


class BgeM3Encoder:
    """Lazy BGE-M3 adapter; heavyweight packages load only when instantiated."""

    def __init__(self, model_name: str = "BAAI/bge-m3", revision: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError("Install the 'retrieval' extra for BGE-M3") from error
        self.model = SentenceTransformer(model_name, revision=revision)

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                list(texts),
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )


class BgeReranker:
    """Cross-encoder reranker for the fixed candidate set."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        revision: str | None = None,
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError("Install the 'retrieval' extra for reranking") from error
        self.model = CrossEncoder(model_name, revision=revision)

    def rank(
        self,
        query: str,
        documents: Sequence[tuple[str, str]],
        batch_size: int = 16,
    ) -> list[tuple[str, float]]:
        scores = self.model.predict(
            [(query, text) for _, text in documents],
            batch_size=batch_size,
            show_progress_bar=False,
        )
        ranked = sorted(
            zip((doc_id for doc_id, _ in documents), scores, strict=True),
            key=lambda item: (-float(item[1]), item[0]),
        )
        return [(doc_id, float(score)) for doc_id, score in ranked]


class Siglip2ImageEncoder:
    """Lazy SigLIP2 image adapter for local, license-compliant image files."""

    def __init__(
        self,
        model_name: str = "google/siglip2-so400m-patch14-384",
        revision: str | None = None,
        device: str | None = None,
    ):
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError("Install the 'retrieval' extra for SigLIP2") from error
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name, revision=revision)
        self.model = AutoModel.from_pretrained(model_name, revision=revision).to(self.device)
        self.model.eval()

    def encode(
        self,
        image_paths: Sequence[str | Path],
        batch_size: int = 8,
    ) -> np.ndarray:
        try:
            from PIL import Image
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError("Install the 'retrieval' extra for image loading") from error
        outputs: list[np.ndarray] = []
        for start in range(0, len(image_paths), batch_size):
            paths = image_paths[start : start + batch_size]
            images = []
            for path in paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                embeddings = self.model.get_image_features(**inputs)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            outputs.append(embeddings.detach().cpu().numpy().astype(np.float32))
        if not outputs:
            return np.empty((0, 0), dtype=np.float32)
        return np.concatenate(outputs, axis=0)
