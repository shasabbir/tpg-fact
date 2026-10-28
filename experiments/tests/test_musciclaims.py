import json
from pathlib import Path

from trace_scv.data.io import read_packets
from trace_scv.data.musciclaims import prepare_musciclaims


def test_prepare_musciclaims(tmp_path: Path) -> None:
    root = tmp_path / "source"
    (root / "paper_figures").mkdir(parents=True)
    (root / "paper_figures" / "bio_p1_figure_1.jpg").write_bytes(b"not-an-image")
    (root / "paper_figures" / "bio_p1_figure_2.jpg").write_bytes(b"not-an-image")
    row = {
        "claim_id": "c1",
        "claim_text": "The figure reports an increase.",
        "label_2class": "SUPPORT",
        "paper_id": "p1",
        "associated_figure_filepath": "paper_figures/bio_p1_figure_1.jpg",
        "associated_figure_panels": ["A"],
        "caption": "Values increase over time.",
    }
    (root / "test_set.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    output = tmp_path / "external_test.jsonl"
    manifest = prepare_musciclaims(root, output)
    packets = read_packets(output)
    assert manifest["count"] == 1
    assert packets[0].label == "supported"
    assert len(packets[0].evidence) == 2
    assert packets[0].gold_evidence_ids == frozenset({"figure:bio_p1_figure_1.jpg"})
