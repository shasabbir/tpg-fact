from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trace_scv.mocheg_features import (
    MochegEncoderConfig,
    HashMochegEncoder,
    build_mocheg_pair_features,
)


class FakeEncoder:
    def encode_bge(self, values):
        return np.asarray(
            [[1.0, 0.0] if "match" in value else [0.0, 1.0] for value in values],
            dtype=np.float32,
        )

    def encode_siglip(self, texts, image_paths):
        return (
            np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32),
            np.asarray([[1.0, 0.0] for _ in image_paths], dtype=np.float32),
        )

    def identity(self):
        return {"name": "fake"}


def test_pair_features_preserve_matched_candidate_identity(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"not-read-by-fake")
    packet = tmp_path / "packets.jsonl"
    rows = [
        {
            "claim_id": "1",
            "claim": "match claim",
            "label": "supported",
            "split": "train",
            "candidates": [
                {
                    "evidence_id": "text:1-2-0",
                    "modality": "text",
                    "gold": True,
                    "available": True,
                    "content": "match evidence",
                    "media_path": None,
                },
                {
                    "evidence_id": "image:a.png",
                    "modality": "image",
                    "gold": False,
                    "available": True,
                    "content": None,
                    "media_path": str(image),
                },
                {
                    "evidence_id": "image:missing.png",
                    "modality": "image",
                    "gold": False,
                    "available": False,
                    "content": None,
                    "media_path": None,
                },
            ],
        }
    ]
    packet.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    output = tmp_path / "features.npz"

    manifest = build_mocheg_pair_features([packet], output, encoder=FakeEncoder())
    values = np.load(output)
    sidecar = json.loads(output.with_suffix(".json").read_text())

    assert manifest["rows"] == 2
    assert manifest["gold_rows"] == 1
    assert values["features"].shape == (2, 7)
    assert manifest["encoder"] == {"name": "fake"}
    assert manifest["features_bytes"] == output.stat().st_size
    assert manifest["packet_files"][str(packet)]["sha256"]
    assert manifest["runtime"]["wall_seconds"] >= 0.0
    assert [row["evidence_id"] for row in sidecar["records"]] == [
        "text:1-2-0",
        "image:a.png",
    ]
    assert sidecar["formal_test_annotations_accessed"] is False


def test_hash_encoder_is_explicitly_smoke_only(tmp_path: Path) -> None:
    encoder = HashMochegEncoder(dim=32)
    text = encoder.encode_bge(["same text", "same text"])

    assert np.allclose(text[0], text[1])
    assert encoder.identity()["scientific_results_allowed"] is False


def test_siglip_processor_choice_is_explicitly_frozen() -> None:
    assert MochegEncoderConfig().vision_use_fast_processor is False
