from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prepare_averimatec import _best_label_blind_chunk, _lexical_features
from trace_scv.data.averimatec import (
    candidate_identifier,
    claim_identifier,
    gold_source_urls,
    load_annotations,
    normalize_source_url,
    train_role,
)
from trace_scv.external_reporting import paired_claim_bootstrap
from trace_scv.external_selection import (
    enforce_official_output_constraints,
    select_evidence_indices,
)
from trace_scv.external_study import (
    freeze_train_only_models,
    run_locked_development,
    write_jsonl,
)


def test_url_normalization_removes_tracking_and_fragment() -> None:
    assert normalize_source_url("HTTPS://www.Example.com/a/?utm_source=x&b=2#frag") == (
        "https://example.com/a?b=2"
    )


def test_claim_and_candidate_identifiers_are_deterministic() -> None:
    row = {"claim_images": ["abc#CLAIM#0.jpg"], "claim_text": "claim"}
    assert claim_identifier(row, 4) == "abc"
    first = candidate_identifier("abc", "https://example.com", "text", "content", 0)
    second = candidate_identifier("abc", "https://example.com", "text", "content", 0)
    assert first == second
    assert len(first) == 64


def test_gold_sources_ignore_metadata_pseudo_source() -> None:
    row = {
        "questions": [
            {
                "answers": [
                    {"source_url": "Metadata"},
                    {"source_url": "https://example.com/evidence/"},
                ]
            }
        ]
    }
    assert gold_source_urls(row) == {"https://example.com/evidence"}


def test_test_loader_rejects_outcome_bearing_fields(tmp_path: Path) -> None:
    path = tmp_path / "test.json"
    path.write_text(json.dumps([{"claim_text": "x", "label": "Refuted"}]), encoding="utf-8")
    with pytest.raises(RuntimeError, match="outcome-bearing"):
        load_annotations(path, split="test")


def test_candidate_features_do_not_use_gold_questions() -> None:
    candidates = [
        {
            "content": "The released evidence contains a distinctive answer phrase.",
            "normalized_source_url": "https://example.com/evidence",
            "branch": "text_related",
        }
    ]
    first = {
        "claim_text": "A generic claim",
        "metadata": {"transcription": ""},
        "questions": [{"question": "distinctive answer phrase"}],
    }
    second = {
        "claim_text": "A generic claim",
        "metadata": {"transcription": ""},
        "questions": [{"question": "unrelated wording"}],
    }
    assert _lexical_features(first, candidates) == _lexical_features(second, candidates)


def test_store_chunking_is_label_blind_and_respects_character_cap() -> None:
    row = {"url2text": ["noise", "target phrase " * 200]}
    selected = _best_label_blind_chunk(row, {"target", "phrase"})
    assert len(selected) <= 1500
    assert "target phrase" in selected


def test_train_roles_are_repeatable_and_valid() -> None:
    assert train_role("claim-1", 13) == train_role("claim-1", 13)
    assert train_role("claim-1", 13) in {
        "fit",
        "calibration",
        "model_selection",
        "internal_holdout",
    }


def test_expected_f1_and_threshold_obey_ten_item_cap() -> None:
    probabilities = [0.9 - 0.01 * index for index in range(30)]
    assert len(select_evidence_indices(probabilities, method="expected_f1")) <= 10
    assert len(
        select_evidence_indices(probabilities, method="threshold", threshold=0.1)
    ) <= 10


def test_official_output_constraints_reject_missing_url_and_long_text() -> None:
    with pytest.raises(ValueError, match="source URL"):
        enforce_official_output_constraints([{"source_url": "", "text": "evidence"}])
    with pytest.raises(ValueError, match="exceeds"):
        enforce_official_output_constraints(
            [{"source_url": "https://example.com", "text": "x " * 1501}]
        )


def test_paired_bootstrap_is_claim_aligned_and_deterministic() -> None:
    left = [{"claim_id": "a", "f1": 1.0}, {"claim_id": "b", "f1": 0.5}]
    right = [{"claim_id": "b", "f1": 0.0}, {"claim_id": "a", "f1": 0.5}]
    first = paired_claim_bootstrap(left, right, samples=100, seed=7)
    second = paired_claim_bootstrap(left, right, samples=100, seed=7)
    assert first == second
    assert first["mean"] == 0.5


def test_external_protocol_keeps_test_submission_locked() -> None:
    import yaml

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs/external_confirmation/averimatec.yaml").read_text(encoding="utf-8")
    )
    assert config["study"]["test_submission_allowed"] is False
    assert config["study"]["test_labels_allowed"] is False
    assert config["selection"]["maximum_selected_items"] == 10
    assert config["retrieval"]["gold_questions_allowed_for_candidate_scoring"] is False


def test_synthetic_train_freeze_and_locked_development_are_separate(tmp_path: Path) -> None:
    candidates: list[dict[str, object]] = []
    full_gold: dict[str, list[str]] = {}
    for split, claims in (("train", 80), ("validation", 4)):
        for claim_index in range(claims):
            claim_id = f"{split}-{claim_index}"
            key = f"{split}:{claim_id}"
            gold_url = f"https://evidence.example/{claim_id}"
            full_gold[key] = [gold_url]
            for candidate_index in range(3):
                positive = candidate_index == 0
                candidates.append(
                    {
                        "split": split,
                        "claim_id": claim_id,
                        "evidence_id": f"{claim_id}-{candidate_index}",
                        "normalized_source_url": (
                            gold_url if positive else f"https://noise.example/{claim_id}/{candidate_index}"
                        ),
                        "modality": "text" if candidate_index < 2 else "image",
                        "gold": positive,
                        "features": [float(positive), 1.0 - candidate_index / 3.0],
                    }
                )
    candidate_path = tmp_path / "candidates.jsonl"
    gold_path = tmp_path / "gold.json"
    write_jsonl(candidate_path, candidates)
    gold_path.write_text(json.dumps(full_gold), encoding="utf-8")
    frozen = freeze_train_only_models(
        candidate_path,
        gold_path,
        seeds=(13,),
        fixed_budgets=(1, 3),
        thresholds=(0.3, 0.5),
        text_widths=(2,),
        image_widths=(1,),
        maximum_items=3,
        maximum_fit_negatives_per_claim=2,
        learning_rate=0.05,
        epochs=20,
        l2=1e-3,
    )
    assert frozen["status"] == "frozen_train_only"
    assert frozen["development_outcomes_accessed"] is False
    summary, rows = run_locked_development(
        candidate_path,
        gold_path,
        frozen,
        maximum_items=3,
        bootstrap_samples=100,
    )
    assert summary["status"] == "checked_locked_development"
    assert summary["claims"] == 4
    assert rows
    assert summary["test_annotations_accessed"] is False
