from pathlib import Path

import numpy as np

from trace_scv.data.io import write_packets
from trace_scv.features import build_feature_store, load_feature_store
from trace_scv.pipeline import _synthetic_packets
from trace_scv.schema import ClaimPacket, EvidenceItem


def test_hash_feature_store_round_trip(tmp_path: Path) -> None:
    packets = tmp_path / "train.jsonl"
    write_packets(packets, _synthetic_packets("train", 4))
    output = tmp_path / "train.npz"
    result = build_feature_store(packets, output, backend="hash", dim=32, max_candidates=4)
    store = load_feature_store(output)
    assert result["count"] == 4
    assert store.claim_features.shape == (4, 32)
    assert store.evidence_features.shape == (4, 4, 32)
    assert store.visual_features.shape == (4, 4, 32)
    assert store.visual_mask.shape == (4, 4)
    assert store.visual_mask.sum() == 0
    assert store.gold_mask.sum() == 4


class _DistinctChannelEmbedder:
    dim = 4

    def text(self, value: str) -> np.ndarray:
        return self.texts([value])[0]

    def image(self, path: str | None, fallback: str) -> np.ndarray:
        return self.images([(path, fallback)])[0]

    def texts(self, values: list[str]) -> list[np.ndarray]:
        return [np.full(self.dim, len(value), dtype=np.float32) for value in values]

    def images(self, values: list[tuple[str | None, str]]) -> list[np.ndarray]:
        return [np.full(self.dim, 99.0, dtype=np.float32) for _ in values]


def test_visual_item_retains_semantic_and_pixel_channels(tmp_path: Path) -> None:
    media = tmp_path / "figure.bin"
    media.write_bytes(b"available-media")
    packets = tmp_path / "packets.jsonl"
    write_packets(
        packets,
        [
            ClaimPacket(
                claim_id="c1",
                claim_text="claim",
                label="supported",
                split="train",
                dataset="unit",
                group_id="paper-1",
                evidence=(
                    EvidenceItem(
                        "figure:1",
                        "figure",
                        "visible caption",
                        media_path=str(media),
                        gold=True,
                    ),
                ),
            )
        ],
    )
    output = tmp_path / "features.npz"
    build_feature_store(
        packets,
        output,
        backend="hash",
        dim=4,
        max_candidates=2,
        embedder=_DistinctChannelEmbedder(),
    )
    store = load_feature_store(output)
    assert store.evidence_features[0, 0].tolist() == [15.0] * 4
    assert store.visual_features[0, 0].tolist() == [99.0] * 4
    assert store.visual_mask[0, 0]
