from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.export_averimatec_outputs import _attach_frozen_verdicts
from scripts.prepare_averimatec import (
    _best_label_blind_chunk,
    _lexical_features,
    _read_json_or_jsonl,
)
from trace_scv.data.averimatec import (
    OFFICIAL_LABELS,
    candidate_identifier,
    claim_identifier,
    gold_evidence_targets,
    gold_image_names,
    gold_source_urls,
    load_annotations,
    normalize_source_url,
    train_role,
)
from trace_scv.external_retrieval import (
    ExternalRetrievalConfig,
    bm25_scores,
    rank_external_candidates,
    reciprocal_rank_fusion,
)
from trace_scv.external_reporting import paired_claim_bootstrap
from trace_scv.external_output import build_official_predictions
from trace_scv.external_selection import (
    enforce_official_output_constraints,
    select_evidence_indices,
)
from trace_scv.external_study import (
    freeze_train_only_models,
    run_locked_development,
    write_jsonl,
)
from trace_scv.external_verifier import (
    aggregate_evidence_features,
    classification_metrics,
    fit_external_mlp_verifier,
    fit_external_verifier,
)


def test_url_normalization_removes_tracking_and_fragment() -> None:
    assert normalize_source_url("HTTPS://www.Example.com/a/?utm_source=x&b=2#frag") == (
        "https://example.com/a?b=2"
    )


def test_jsonl_reader_preserves_unicode_line_separator_inside_record(tmp_path: Path) -> None:
    path = tmp_path / "store.json"
    path.write_text(
        '{"claim_id":"1","url2text":["left\u2028right"]}\n'
        '{"claim_id":"1","url2text":["next"]}\n',
        encoding="utf-8",
    )
    rows = _read_json_or_jsonl(path)
    assert len(rows) == 2
    assert rows[0]["url2text"] == ["left\u2028right"]


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


def test_gold_targets_keep_url_and_image_modalities_separate() -> None:
    row = {
        "questions": [
            {
                "answers": [
                    {
                        "source_url": "https://example.com/evidence/",
                        "image_answers": ["folder/Answer.JPG"],
                    }
                ]
            }
        ]
    }
    assert gold_image_names(row) == {"answer.jpg"}
    assert gold_evidence_targets(row) == {
        "url:https://example.com/evidence",
        "image:answer.jpg",
    }


def test_test_loader_rejects_outcome_bearing_fields(tmp_path: Path) -> None:
    path = tmp_path / "test.json"
    path.write_text(json.dumps([{"claim_text": "x", "label": "Refuted"}]), encoding="utf-8")
    with pytest.raises(RuntimeError, match="outcome-bearing"):
        load_annotations(path, split="test")


def test_sanitized_annotation_view_removes_outcomes(tmp_path: Path) -> None:
    path = tmp_path / "train.json"
    path.write_text(
        json.dumps(
            [
                {
                    "claim_text": "claim",
                    "claim_images": ["c1#CLAIM#0.jpg"],
                    "label": "Supported",
                    "questions": [],
                    "justification": "because",
                }
            ]
        ),
        encoding="utf-8",
    )
    row = load_annotations(path, split="train", include_outcomes=False)[0]
    assert "label" not in row
    assert "questions" not in row
    assert "justification" not in row
    assert row["gold_evidence_targets"] == []


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


def test_external_bm25_and_rrf_are_deterministic() -> None:
    scores = bm25_scores("target phrase", ["noise", "target phrase target"])
    assert scores[1] > scores[0]
    order, values = reciprocal_rank_fusion(([1, 0], [0, 1]), constant=60)
    assert order.tolist() == [0, 1]
    assert values[0] == values[1]


class _TinyExternalEncoder:
    def _encode(self, values: list[str]) -> object:
        import numpy as np

        rows = []
        for value in values:
            lowered = value.casefold()
            row = np.asarray(
                [
                    float("target" in lowered),
                    float("noise" in lowered),
                    float("image" in lowered),
                    1.0,
                ],
                dtype=np.float32,
            )
            rows.append(row / np.linalg.norm(row))
        return np.stack(rows)

    def encode_bge(self, values: list[str]) -> object:
        return self._encode(values)

    def encode_siglip(self, texts: list[str], image_paths: list[str]) -> tuple[object, object]:
        return self._encode(texts), self._encode(image_paths)

    def identity(self) -> dict[str, object]:
        return {"backend": "tiny_test", "scientific_results_allowed": False}


def test_multimodal_retrieval_ranks_real_image_rows_without_outcomes(tmp_path: Path) -> None:
    claim = {
        "split": "train",
        "claim_id": "c1",
        "claim_text": "target image",
        "transcription": "",
    }
    image = tmp_path / "target-image.jpg"
    image.write_bytes(b"test-only")
    candidates = {
        "train:c1": [
            {
                "split": "train",
                "claim_id": "c1",
                "claim_ordinal": 0,
                "evidence_id": "t0",
                "modality": "text",
                "branch": "text_related",
                "content": "noise",
                "content_word_count": 1,
            },
            {
                "split": "train",
                "claim_id": "c1",
                "claim_ordinal": 0,
                "evidence_id": "t1",
                "modality": "text",
                "branch": "image_related",
                "content": "target phrase",
                "content_word_count": 2,
            },
            {
                "split": "train",
                "claim_id": "c1",
                "claim_ordinal": 0,
                "evidence_id": "i0",
                "modality": "image",
                "branch": "image_candidate",
                "content": "target image",
                "content_word_count": 2,
                "media_path": str(image),
            },
        ]
    }
    ranked, manifest = rank_external_candidates(
        [claim],
        candidates,
        encoder=_TinyExternalEncoder(),
        config=ExternalRetrievalConfig(text_depth=2, image_depth=1, rerank_depth=2),
        include_cross_encoder=False,
    )
    text_rows = [row for row in ranked if row["modality"] == "text"]
    image_rows = [row for row in ranked if row["modality"] == "image"]
    assert text_rows[0]["evidence_id"] == "t1"
    assert image_rows[0]["evidence_id"] == "i0"
    assert image_rows[0]["siglip_score"] > 0
    assert manifest["gold_questions_used_for_retrieval"] is False
    assert manifest["scientific_results_allowed"] is False


def test_multimodal_retrieval_rejects_outcome_fields() -> None:
    with pytest.raises(ValueError, match="outcome-bearing"):
        rank_external_candidates(
            [
                {
                    "split": "train",
                    "claim_id": "c1",
                    "claim_text": "claim",
                    "transcription": "",
                    "label": "Refuted",
                }
            ],
            {},
            encoder=_TinyExternalEncoder(),
            config=ExternalRetrievalConfig(),
            include_cross_encoder=False,
        )


def test_external_four_class_verifier_and_feature_aggregation() -> None:
    import numpy as np

    labels = [
        "Supported",
        "Supported",
        "Refuted",
        "Refuted",
        "Not Enough Evidence",
        "Not Enough Evidence",
        "Conflicting Evidence/Cherrypicking",
        "Conflicting Evidence/Cherrypicking",
    ]
    features = np.asarray(
        [
            [2.0, 0.0, 0.0, 0.0],
            [1.8, 0.1, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.1, 1.8, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.1, 1.8, 0.0],
            [0.0, 0.0, 0.0, 2.0],
            [0.1, 0.0, 0.0, 1.8],
        ],
        dtype=np.float64,
    )
    verifier = fit_external_verifier(features, labels, epochs=300)
    predictions = [
        verifier.labels[int(index)]
        for index in verifier.predict_probabilities(features).argmax(axis=1)
    ]
    assert classification_metrics(labels, predictions)["macro_f1"] > 0.9
    mlp = fit_external_mlp_verifier(features, labels, seed=13, hidden_units=8, epochs=300)
    mlp_predictions = [
        mlp.labels[int(index)]
        for index in mlp.predict_probabilities(features).argmax(axis=1)
    ]
    assert classification_metrics(labels, mlp_predictions)["macro_f1"] > 0.9
    aggregate = aggregate_evidence_features(
        [
            {
                "evidence_id": "a",
                "features": [1.0, 0.0],
                "modality": "text",
                "source_domain": "example.com",
            },
            {
                "evidence_id": "b",
                "features": [0.0, 1.0],
                "modality": "image",
                "source_domain": "",
            },
        ],
        ["a", "b"],
    )
    assert aggregate.shape == (10,)


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


def test_official_output_constraints_reject_empty_and_long_evidence() -> None:
    with pytest.raises(ValueError, match="neither text nor images"):
        enforce_official_output_constraints([{"text": "", "images": []}])
    with pytest.raises(ValueError, match="exceeds"):
        enforce_official_output_constraints(
            [{"source_url": "https://example.com", "text": "x " * 1501}]
        )


def test_official_export_uses_pinned_evidence_schema() -> None:
    predictions, diagnostics = build_official_predictions(
        [{"claim_id": "c1"}],
        [
            {
                "claim_id": "c1",
                "evidence_id": "e1",
                "modality": "text",
                "content": "evidence text",
                "source_url": "https://example.com",
                "evaluation_id": "url:https://example.com",
                "retrieval_rank": 1,
            }
        ],
        [
            {
                "split": "validation",
                "claim_id": "c1",
                "seed": 13,
                "policy": "completeness_aware_switch",
                "selected_evidence_ids": ["e1"],
                "predicted_verdict": "Supported",
            }
        ],
    )
    assert predictions == [
        {
            "evidence": [{"text": "evidence text", "images": []}],
            "justification": "",
            "questions": [],
            "verdict": "Supported",
        }
    ]
    assert diagnostics[0]["official_evidence_score_computed"] is False


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
            claim_label = (
                "Supported",
                "Refuted",
                "Not Enough Evidence",
                "Conflicting Evidence/Cherrypicking",
            )[claim_index % 4]
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
                        "claim_label": claim_label,
                        "features": [float(positive), 1.0 - candidate_index / 3.0],
                    }
                )
    train_candidate_path = tmp_path / "train_candidates.jsonl"
    train_gold_path = tmp_path / "train_gold.json"
    validation_candidate_path = tmp_path / "validation_candidates.jsonl"
    validation_gold_path = tmp_path / "validation_gold.json"
    write_jsonl(
        train_candidate_path,
        [row for row in candidates if row["split"] == "train"],
    )
    train_gold_path.write_text(
        json.dumps({key: value for key, value in full_gold.items() if key.startswith("train:")}),
        encoding="utf-8",
    )
    write_jsonl(
        validation_candidate_path,
        [row for row in candidates if row["split"] == "validation"],
    )
    validation_gold_path.write_text(
        json.dumps(
            {key: value for key, value in full_gold.items() if key.startswith("validation:")}
        ),
        encoding="utf-8",
    )
    frozen = freeze_train_only_models(
        train_candidate_path,
        train_gold_path,
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
    assert frozen["seeds"][0]["completeness_model"]["kind"] == "constant"
    assert "completeness_aware_switch" in frozen["seeds"][0]["internal_holdout"]
    assert frozen["seeds"][0]["shared_verifier"]["status"] == "frozen_train_only"
    summary, rows = run_locked_development(
        validation_candidate_path,
        validation_gold_path,
        frozen,
        maximum_items=3,
        bootstrap_samples=100,
    )
    assert summary["status"] == "checked_locked_development"
    assert summary["claims"] == 4
    assert rows
    assert summary["test_annotations_accessed"] is False
    assert "completeness_aware_switch" in summary["aggregate_mean_over_seed_then_claim"]
    assert "completeness_switch_minus_threshold" in summary["paired_claim_bootstrap"]
    assert summary["shared_verifier"]["status"] == "checked_locked_development"
    fixed_rows = [row for row in rows if row["policy"] == "fixed"]
    exported_rows = _attach_frozen_verdicts(candidates, fixed_rows, frozen, seed=13)
    assert len(exported_rows) == 4
    assert all(row["predicted_verdict"] in OFFICIAL_LABELS for row in exported_rows)
    assert all(len(row["verdict_probabilities"]) == 4 for row in exported_rows)


def test_train_freeze_rejects_development_rows(tmp_path: Path) -> None:
    candidate_path = tmp_path / "mixed_candidates.jsonl"
    gold_path = tmp_path / "mixed_gold.json"
    write_jsonl(
        candidate_path,
        [
            {
                "split": "train",
                "claim_id": "train-1",
                "evidence_id": "train-evidence",
                "modality": "text",
                "gold": True,
                "claim_label": "Supported",
                "features": [1.0, 0.0],
            },
            {
                "split": "validation",
                "claim_id": "validation-1",
                "evidence_id": "validation-evidence",
                "modality": "text",
                "gold": True,
                "claim_label": "Refuted",
                "features": [1.0, 0.0],
            },
        ],
    )
    gold_path.write_text(
        json.dumps(
            {
                "train:train-1": ["url:https://example.test/train"],
                "validation:validation-1": ["url:https://example.test/validation"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="train-only candidate artifact"):
        freeze_train_only_models(
            candidate_path,
            gold_path,
            seeds=(13,),
            fixed_budgets=(1,),
            thresholds=(0.5,),
            text_widths=(1,),
            image_widths=(1,),
            maximum_items=1,
            maximum_fit_negatives_per_claim=1,
            learning_rate=0.05,
            epochs=2,
            l2=1e-3,
        )
