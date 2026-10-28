from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from trace_scv.mocheg_full_corpus import (
    FullCorpusRetrievalConfig,
    _ranking_metrics,
    _top_indices,
    build_full_corpus_features,
    import_full_media_from_lfs_cache,
)


def test_top_indices_are_descending_and_stable() -> None:
    values = np.asarray([0.1, 0.9, 0.9, 0.2])
    assert _top_indices(values, 3).tolist() == [1, 2, 3]


def test_ranking_metrics_penalize_unretrieved_gold() -> None:
    result = _ranking_metrics(["a", "x", "b"], {"a", "b", "missing"})
    assert result["recall_at_5"] == 2.0 / 3.0
    assert result["map_at_60"] < 1.0
    assert result["ndcg_at_60"] < 1.0


def test_ranking_metrics_use_full_gold_denominator() -> None:
    result = _ranking_metrics(["a", "b"], {"a", "b", "c", "d"})
    assert result["recall_at_50"] == 0.5
    assert result["map_at_60"] == 0.5


class _TinyEncoder:
    def encode_bge(self, values):
        rows = []
        for value in values:
            rows.append([float("alpha" in value), float("beta" in value)])
        result = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        return result / np.maximum(norms, 1.0)

    def encode_siglip(self, texts, image_paths):
        return (
            np.asarray([[1.0, 0.0] if "alpha" in text else [0.0, 1.0] for text in texts]),
            np.asarray([[1.0, 0.0] for _ in image_paths]),
        )

    def identity(self):
        return {"backend": "tiny-scientific-test"}


def test_full_corpus_builder_keeps_unretrieved_gold_in_denominator(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = tmp_path / "dataset"
    processed = tmp_path / "processed"
    corpus = dataset / "metadata" / "corpus" / "Corpus3.csv"
    corpus.parent.mkdir(parents=True)
    with corpus.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "claim_id",
                "relevant_document_id",
                "Snopes URL",
                "Link URL",
                "Archive Url",
                "Origin Document",
            ],
        )
        writer.writeheader()
        writer.writerow({"claim_id": "source", "relevant_document_id": "doc", "Origin Document": "alpha. beta."})
    processed.mkdir()
    packets = {
        "train": {"claim_id": "1", "claim": "alpha", "split": "train", "candidates": [{"available": True}]},
        "validation": {"claim_id": "2", "claim": "beta", "split": "validation", "candidates": [{"available": True}]},
    }
    for split, packet in packets.items():
        (processed / f"{split}_pilot_packets.jsonl").write_text(json.dumps(packet) + "\n", encoding="utf-8")
    image = dataset / "media" / "train" / "objects" / "a.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"not-opened-by-tiny-encoder")
    media_manifest = {
        "status": "pass",
        "downloaded": [{"basename": "a.jpg", "sha256": "same", "mirror": "train", "relative_path": "objects/a.jpg"}],
    }
    (dataset / "media" / "full_development_media_manifest.json").write_text(json.dumps(media_manifest), encoding="utf-8")
    for split, folder, claim_id in (("train", "train", "1"), ("validation", "val", "2")):
        root = dataset / "metadata" / folder
        root.mkdir(parents=True, exist_ok=True)
        for name, document_id in (("text_evidence_qrels_sentence_level.csv", "source-doc-0"), ("img_evidence_qrels.csv", "a.jpg")):
            (root / name).write_text(
                f"TOPIC,ITERATION,DOCUMENT#,RELEVANCY,evidence_id\n{claim_id},0,{document_id},1,proof\n",
                encoding="utf-8",
            )
    monkeypatch.setattr(
        "trace_scv.mocheg_full_corpus._nltk_sentence_splitter",
        lambda _root: (lambda text: [part.strip() for part in text.split(".") if part.strip()], {"test": "split"}),
    )
    output = tmp_path / "features.npz"
    manifest = build_full_corpus_features(
        dataset,
        processed,
        output,
        encoder=_TinyEncoder(),
        config=FullCorpusRetrievalConfig(first_stage_text_documents=1, retained_text_candidates=1, retained_image_candidates=1, audit_ranking_depth=2),
    )
    sidecar = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert manifest["claims"] == 2
    assert sidecar["full_gold_evidence_ids_by_claim"]["1"] == ["image:a.jpg", "text:source-doc-0"]


def test_lfs_import_uses_content_addressed_objects(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "dataset"
    cache = tmp_path / "cache"
    payload = b"image-bytes"
    oid = hashlib.sha256(payload).hexdigest()
    for mirror in ("train", "validation"):
        source = cache / f"{mirror}.git" / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
        source.parent.mkdir(parents=True)
        source.write_bytes(payload)

    monkeypatch.setattr(
        "trace_scv.mocheg_full_corpus.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=f"{oid} - images/folder/example.jpg\n"
        ),
    )
    result = import_full_media_from_lfs_cache(dataset, cache)
    assert result["status"] == "pass"
    assert result["imported"] == 2
    assert result["missing_count"] == 0
    assert len(list((dataset / "media").rglob("*.jpg"))) == 2
