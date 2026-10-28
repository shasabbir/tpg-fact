import json
from pathlib import Path

from trace_scv.data.io import read_packets
from trace_scv.data.sciver import build_sciver_packet, prepare_sciver


def _paper() -> dict:
    return {
        "title": "Example",
        "abstract": "An example paper.",
        "sections": [
            {"section_id": "1", "section_name": "Methods", "text": "The sample has 20 items."},
            {"section_id": "2", "section_name": "Results", "text": "Accuracy was 91 percent."},
        ],
        "appendix": [],
        "tables": {"1": {"caption": "Scores", "table_html": "<table><tr><td>91</td></tr></table>"}},
        "image_paths": {},
    }


def test_exact_child_section_precedes_empty_parent(tmp_path: Path) -> None:
    paper = _paper()
    paper["sections"] = [
        {"section_id": "4", "section_name": "Experiments", "text": ""},
        {
            "section_id": "4.3",
            "section_name": "Synthetic Volumes",
            "text": "The exact annotated evidence.",
        },
    ]
    row = {
        "paperid": "paper-exact-child",
        "claim": "The synthetic volume experiment succeeded.",
        "label": True,
        "section": ["4.3"],
        "type": "text",
        "request_id": 1,
    }
    packet = build_sciver_packet(row, paper, tmp_path, "train", 0)
    gold = [item for item in packet.evidence if item.gold]
    assert [item.evidence_id for item in gold] == ["text:4.3"]
    assert "The exact annotated evidence" in gold[0].content


def test_missing_annotated_section_adds_gold_abstract_before_distractors(
    tmp_path: Path,
) -> None:
    paper = _paper()
    row = {
        "paperid": "paper-missing-child",
        "claim": "A claim with an unavailable annotated section.",
        "label": False,
        "section": ["9.9"],
        "type": "text",
        "request_id": 2,
    }
    packet = build_sciver_packet(row, paper, tmp_path, "train", 0)
    gold = [item for item in packet.evidence if item.gold]
    assert [item.evidence_id for item in gold] == ["abstract"]
    assert gold[0].metadata["fallback"] == "missing_annotated_evidence"
    assert any(not item.gold for item in packet.evidence)


def test_prepare_sciver_preserves_official_test(tmp_path: Path) -> None:
    root = tmp_path / "source"
    (root / "papers").mkdir(parents=True)
    rows = []
    for index in range(12):
        paper_id = f"paper-{index}"
        (root / "papers" / f"{paper_id}.json").write_text(json.dumps(_paper()), encoding="utf-8")
        rows.append(
            {
                "paperid": paper_id,
                "claim": "Accuracy was 91 percent.",
                "label": index % 2 == 0,
                "section": ["2"],
                "type": "table",
                "item": "1",
                "request_id": index,
                "image_path": "",
            }
        )
    (root / "valset.json").write_text(json.dumps(rows[:10]), encoding="utf-8")
    (root / "testset.json").write_text(json.dumps(rows[10:]), encoding="utf-8")
    output = tmp_path / "processed"
    manifest = prepare_sciver(root, output, seed=13)
    train = read_packets(output / "train.jsonl")
    validation = read_packets(output / "validation.jsonl")
    test = read_packets(output / "test.jsonl")
    assert len(test) == 2
    assert {packet.group_id for packet in train}.isdisjoint(
        {packet.group_id for packet in validation}
    )
    assert manifest["splits"]["test"]["count"] == 2
    assert manifest["group_intersections"] == {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }
    assert manifest["data_quality"]["duplicate_claim_ids"] == 0
    assert manifest["data_quality"]["splits"]["test"]["table_items"] == 2
    assert (
        manifest["data_quality"]["splits"]["test"]["required_visual_media_coverage"]
        is None
    )
    assert (
        manifest["data_quality"]["cross_split_duplicates"]["train_test"]
        ["exact_content_fingerprint_overlap"]
        >= 1
    )
    duplicate = manifest["data_quality"]["cross_split_duplicate_details"][
        "train_test"
    ]["exact_content"][0]
    assert duplicate["fingerprint"]
    assert duplicate["left"][0]["claim_id"]
    assert duplicate["right"][0]["evidence_id"]
    assert "gold" in duplicate["left"][0]
    assert manifest["data_quality"]["release_gate"]["pass"] is False
    assert all(packet.gold_evidence_ids for packet in [*train, *validation, *test])
