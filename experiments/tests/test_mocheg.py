import csv
import json
from pathlib import Path

import pytest

from trace_scv.data.mocheg import (
    MochegClaim,
    MochegQrel,
    _classify_missing_media,
    _audit_downloaded_images,
    assert_mocheg_development_sealed,
    build_mocheg_candidate_sets,
    load_mocheg_claims,
    load_mocheg_qrels,
    normalize_mocheg_label,
    parse_mocheg_sentence_document_id,
    reconstruct_mocheg_sentence_texts,
)


def test_aggregated_train_schema_normalizes_lists(tmp_path: Path) -> None:
    path = tmp_path / "train.json"
    path.write_text(
        json.dumps(
            [
                {
                    "claim_id": "10",
                    "claim": "A claim",
                    "evidence_id": ["2", "1"],
                    "text_evidence": ["second", "first"],
                    "label": "supported",
                    "ruling_outline": "Reason",
                    "snopes_url": ["https://example.test/claim"],
                    "image": ["10-proof-b.jpg", "10-proof-a.jpg"],
                }
            ]
        ),
        encoding="utf-8",
    )

    claims = load_mocheg_claims(path, split="train")

    assert len(claims) == 1
    assert claims[0].label == "supported"
    assert [item.evidence_id for item in claims[0].text_evidence] == ["1", "2"]
    assert claims[0].source_image_ids == ("10-proof-a.jpg", "10-proof-b.jpg")


def test_evidence_row_validation_schema_groups_claim(tmp_path: Path) -> None:
    path = tmp_path / "validation.json"
    common = {
        "claim_id": "20",
        "claim": "Another claim",
        "label": "not enough information",
        "ruling_outline": "Reason",
        "snopes_url": "https://example.test/claim-20",
        "image": ["20-proof-a.jpg"],
    }
    path.write_text(
        json.dumps(
            [
                {**common, "evidence_id": "8", "text_evidence": "text eight"},
                {**common, "evidence_id": "9", "text_evidence": "text nine"},
            ]
        ),
        encoding="utf-8",
    )

    claims = load_mocheg_claims(path, split="validation")

    assert len(claims) == 1
    assert claims[0].label == "nei"
    assert len(claims[0].text_evidence) == 2
    assert claims[0].source_image_ids == ("20-proof-a.jpg",)


def test_conflicting_claim_label_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "validation.json"
    path.write_text(
        json.dumps(
            [
                {
                    "claim_id": "1",
                    "claim": "Claim",
                    "evidence_id": "1",
                    "text_evidence": "one",
                    "label": "supported",
                },
                {
                    "claim_id": "1",
                    "claim": "Claim",
                    "evidence_id": "2",
                    "text_evidence": "two",
                    "label": "refuted",
                },
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="label"):
        load_mocheg_claims(path, split="validation")


def test_qrels_are_typed_and_validated(tmp_path: Path) -> None:
    path = tmp_path / "qrels.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["TOPIC", "ITERATION", "DOCUMENT#", "RELEVANCY", "evidence_id"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "TOPIC": "3",
                "ITERATION": "0",
                "DOCUMENT#": "candidate.jpg",
                "RELEVANCY": "1",
                "evidence_id": "proof.jpg",
            }
        )

    rows = load_mocheg_qrels(path)

    assert len(rows) == 1
    assert rows[0].claim_id == "3"
    assert rows[0].relevant is True
    assert rows[0].evidence_id == "proof.jpg"


def test_development_seal_rejects_test_directory(tmp_path: Path) -> None:
    (tmp_path / "metadata" / "test").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="formal-test"):
        assert_mocheg_development_sealed(tmp_path)


def test_media_audit_detects_readable_duplicate_images(tmp_path: Path) -> None:
    pillow = pytest.importorskip("PIL.Image")
    media = tmp_path / "train" / "images"
    media.mkdir(parents=True)
    first = media / "a.png"
    second = media / "b.png"
    image = pillow.new("RGB", (12, 8), "white")
    image.save(first)
    image.save(second)
    rows = [
        {
            "mirror": "train",
            "relative_path": "images/a.png",
            "sha256": "same",
        },
        {
            "mirror": "train",
            "relative_path": "images/b.png",
            "sha256": "same",
        },
    ]

    audit = _audit_downloaded_images(tmp_path, rows)

    assert audit["readable"] == 2
    assert audit["unique_sha256"] == 1
    assert audit["exact_duplicate_occurrences"] == 1
    assert audit["unreadable"] == []


def test_missing_negative_media_does_not_break_gold_gate() -> None:
    claim = MochegClaim(
        "1", "Claim", "supported", (), ("proof.jpg",), "", (), "train"
    )
    audit = _classify_missing_media(
        claims=[claim],
        qrels=[
            MochegQrel("1", "proof.jpg", True, "proof.jpg"),
            MochegQrel("1", "missing-negative.jpg", False, "proof.jpg"),
        ],
        missing_names=["missing-negative.jpg"],
    )

    assert audit["missing_negative_candidate_rows"] == 1
    assert audit["missing_positive_candidate_rows"] == 0
    assert audit["gold_complete"] is True


def test_missing_positive_media_breaks_gold_gate() -> None:
    claim = MochegClaim("1", "Claim", "supported", (), (), "", (), "train")
    audit = _classify_missing_media(
        claims=[claim],
        qrels=[MochegQrel("1", "missing-proof.jpg", True, "gold.jpg")],
        missing_names=["missing-proof.jpg"],
    )

    assert audit["missing_positive_candidate_rows"] == 1
    assert audit["gold_complete"] is False


def test_candidate_sets_deduplicate_and_qualify_modalities() -> None:
    claim = MochegClaim("1", "Claim", "supported", (), (), "", (), "train")
    duplicate_text = MochegQrel("1", "1-2-3", True, None)
    candidates = build_mocheg_candidate_sets(
        [claim],
        text_qrels=[duplicate_text, duplicate_text],
        image_qrels=[MochegQrel("1", "image.jpg", False, "proof.jpg")],
    )

    assert [row.evidence_id for row in candidates["1"]] == [
        "image:image.jpg",
        "text:1-2-3",
    ]
    assert sum(row.gold for row in candidates["1"]) == 1


def test_candidate_sets_reject_conflicting_relevance() -> None:
    claim = MochegClaim("1", "Claim", "supported", (), (), "", (), "train")

    with pytest.raises(ValueError, match="conflicting relevance"):
        build_mocheg_candidate_sets(
            [claim],
            text_qrels=[
                MochegQrel("1", "1-2-3", True, None),
                MochegQrel("1", "1-2-3", False, None),
            ],
            image_qrels=[],
        )


def test_sentence_document_id_parsing() -> None:
    assert parse_mocheg_sentence_document_id("120182-261423-7") == (
        "120182",
        "261423",
        7,
    )
    with pytest.raises(ValueError, match="sentence index"):
        parse_mocheg_sentence_document_id("120182-261423-last")


def test_sentence_reconstruction_matches_official_duplicate_policy(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "Corpus3.csv"
    with corpus.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["claim_id", "relevant_document_id", "Origin Document"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "claim_id": "1",
                "relevant_document_id": "2",
                "Origin Document": "<p>first|second</p>",
            }
        )
        writer.writerow(
            {
                "claim_id": "1",
                "relevant_document_id": "2",
                "Origin Document": "replacement",
            }
        )

    qrels = [
        MochegQrel("9", "1-2-0", True, None),
        MochegQrel("9", "1-2-1", False, None),
    ]
    records, audit = reconstruct_mocheg_sentence_texts(
        corpus,
        qrels,
        sentence_splitter=lambda text: text.split("|"),
    )

    assert records["1-2-0"].content == "replacement"
    assert records["1-2-1"].content == "second"
    assert audit["missing_sentence_ids"] == 0
    assert audit["duplicate_source_rows"] == 1
    assert audit["conflicting_source_rows"] == 1
    assert audit["conflicting_sentence_assignments"] == 1


def test_sentence_reconstruction_reports_missing_ids(tmp_path: Path) -> None:
    corpus = tmp_path / "Corpus3.csv"
    corpus.write_text(
        "claim_id,relevant_document_id,Origin Document\n1,2,only sentence\n",
        encoding="utf-8",
    )
    records, audit = reconstruct_mocheg_sentence_texts(
        corpus,
        [MochegQrel("9", "1-2-3", True, None)],
        sentence_splitter=lambda text: [text],
    )

    assert records == {}
    assert audit["missing_document_ids"] == ["1-2-3"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("support", "supported"),
        ("refuted", "refuted"),
        ("NEI", "nei"),
        ("not enough information", "nei"),
    ],
)
def test_label_mapping(raw: str, expected: str) -> None:
    assert normalize_mocheg_label(raw) == expected
