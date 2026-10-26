from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path

import numpy as np
import yaml

from data.io import read_jsonl, write_json, write_jsonl
from data.schema import ClaimRecord, EvidenceRecord
from data.splits import split_name
from evaluation.interventions import (
    inject_post_cutoff_verdict,
    inject_syndicated_copies,
    remove_earliest_source,
)
from evaluation.metrics import classification_metrics
from retrieval.dense import BgeM3Encoder
from training.dataset import EmbeddingStore, build_examples
from training.engine import TrainingSettings, predict_from_checkpoint


DEFAULT_VARIANTS = ("vanilla_rag", "time_filtered_rag", "provenance_pool", "full")
SCENARIOS = (
    "duplicate_1",
    "duplicate_5",
    "duplicate_10",
    "post_cutoff_verdict",
    "remove_earliest_source",
    "dependent_conflict",
    "independent_conflict",
    "caption_swap",
    "historical_image_reuse",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen checkpoints on deterministic evidence interventions"
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--claims", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--text-features", required=True)
    parser.add_argument("--image-features")
    parser.add_argument("--matrix-root", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--conflict-manifest")
    parser.add_argument("--variant", action="append")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--skip-synthetic-encoding",
        action="store_true",
        help="Diagnostic only; use zero vectors for injected verdict documents",
    )
    return parser.parse_args()


def _variant_options(variant: str) -> dict[str, bool]:
    return {
        "enforce_time_filter": variant not in {"vanilla_rag", "no_temporal_filter"},
        "use_provenance": variant not in {"vanilla_rag", "time_filtered_rag", "no_provenance"},
        "use_evidence_text": variant != "no_text_evidence",
        "use_visual_evidence": variant != "no_visual_evidence",
        "use_temporal_features": variant != "no_temporal_features",
    }


def _scenario_records(
    claims: list[ClaimRecord],
    donor_claims: list[ClaimRecord],
    evidence_by_id: dict[str, EvidenceRecord],
    conflict_donors: dict[str, str] | None = None,
) -> tuple[
    dict[str, list[ClaimRecord]],
    dict[str, list[EvidenceRecord]],
    dict[str, str],
    dict[tuple[str, str], str],
]:
    scenario_claims: dict[str, list[ClaimRecord]] = {name: [] for name in SCENARIOS}
    scenario_evidence: dict[str, list[EvidenceRecord]] = {name: [] for name in SCENARIOS}
    aliases: dict[str, str] = {}
    image_aliases: dict[tuple[str, str], str] = {}
    for claim in claims:
        original = [
            evidence_by_id[item_id]
            for item_id in claim.evidence_ids
            if item_id in evidence_by_id
        ]
        original.sort(key=lambda item: (item.archived_at, item.evidence_id))
        variants: dict[str, list[EvidenceRecord]] = {
            "post_cutoff_verdict": inject_post_cutoff_verdict(claim, original),
            "remove_earliest_source": remove_earliest_source(original),
        }
        label_donors = sorted(
            (
                candidate
                for candidate in donor_claims
                if candidate.claim_id != claim.claim_id
                and candidate.normalized_label != claim.normalized_label
            ),
            key=lambda candidate: (candidate.claim_time, candidate.claim_id),
        )
        donor_claim = label_donors[0] if label_donors else None
        audited_donor_id = (conflict_donors or {}).get(claim.claim_id)
        donor_evidence = evidence_by_id.get(audited_donor_id) if audited_donor_id else (
            next(
                (
                    evidence_by_id[item_id]
                    for item_id in donor_claim.evidence_ids
                    if item_id in evidence_by_id
                ),
                None,
            )
            if donor_claim
            else None
        )
        if donor_evidence is not None:
            published = min(donor_evidence.published_at, claim.decision_cutoff - timedelta(days=1))
            archived = min(donor_evidence.archived_at, claim.decision_cutoff - timedelta(hours=1))
            if archived < published:
                archived = published
            target_cluster = (
                original[0].source_cluster_id
                if original and original[0].source_cluster_id
                else f"intervention-cluster:{claim.claim_id}"
            )
            for name, cluster, links in (
                (
                    "dependent_conflict",
                    target_cluster,
                    (original[0].url,) if original else (),
                ),
                (
                    "independent_conflict",
                    f"intervention-independent:{claim.claim_id}",
                    (),
                ),
            ):
                conflict_id = f"{claim.claim_id}__{name}"
                conflict = replace(
                    donor_evidence,
                    evidence_id=conflict_id,
                    url=f"https://{name}.invalid/{claim.claim_id}",
                    published_at=published,
                    archived_at=archived,
                    source_cluster_id=cluster,
                    outgoing_links=links,
                    timestamp_confidence="confirmed",
                    timestamp_source="synthetic_intervention",
                    dependency_signals=(f"synthetic_{name}",),
                )
                variants[name] = [*original, conflict]
                aliases[conflict_id] = donor_evidence.evidence_id
        else:
            variants["dependent_conflict"] = list(original)
            variants["independent_conflict"] = list(original)
        if original:
            for copies in (1, 5, 10):
                name = f"duplicate_{copies}"
                variants[name] = inject_syndicated_copies(
                    original, original[0].evidence_id, copies
                )
                for item in variants[name]:
                    if "__copy_" in item.evidence_id:
                        aliases[item.evidence_id] = original[0].evidence_id
        else:
            for copies in (1, 5, 10):
                variants[f"duplicate_{copies}"] = []
        variants["caption_swap"] = list(original)
        variants["historical_image_reuse"] = list(original)
        caption_donors = sorted(
            (
                candidate
                for candidate in claims
                if candidate.claim_id != claim.claim_id
                and candidate.claim_id in {item.claim_id for item in donor_claims}
            ),
            key=lambda candidate: candidate.claim_id,
        )
        caption_donor = caption_donors[0] if caption_donors else donor_claim
        historical_donors = sorted(
            (
                candidate
                for candidate in donor_claims
                if candidate.claim_id != claim.claim_id
                and candidate.event_id != claim.event_id
                and candidate.claim_time < claim.claim_time
            ),
            key=lambda candidate: (candidate.claim_time, candidate.claim_id),
        )
        historical_donor = historical_donors[0] if historical_donors else caption_donor
        for name, rows in variants.items():
            scenario_evidence[name].extend(rows)
            modified_claim = replace(
                claim, evidence_ids=tuple(item.evidence_id for item in rows)
            )
            image_donor = (
                caption_donor
                if name == "caption_swap"
                else historical_donor
                if name == "historical_image_reuse"
                else None
            )
            if image_donor is not None:
                modified_claim = replace(
                    modified_claim,
                    claim_image_url=image_donor.claim_image_url,
                    image_sha256=image_donor.image_sha256,
                    image_phash=image_donor.image_phash,
                )
                image_aliases[(name, claim.claim_id)] = image_donor.claim_id
            scenario_claims[name].append(modified_claim)
    return scenario_claims, scenario_evidence, aliases, image_aliases


def _store_for_scenario(
    base: EmbeddingStore,
    records: list[EvidenceRecord],
    aliases: dict[str, str],
    image_aliases: dict[tuple[str, str], str],
    scenario: str,
    encoded: dict[str, np.ndarray],
) -> EmbeddingStore:
    text = dict(base.text)
    image = dict(base.image)
    for item in records:
        source = aliases.get(item.evidence_id)
        if source:
            text[item.evidence_id] = np.asarray(text[source], dtype=np.float32)
            if source in image:
                image[item.evidence_id] = np.asarray(image[source], dtype=np.float32)
        elif item.evidence_id in encoded:
            text[item.evidence_id] = encoded[item.evidence_id]
    for (name, claim_id), donor_id in image_aliases.items():
        if name == scenario and donor_id in image:
            image[claim_id] = np.asarray(image[donor_id], dtype=np.float32)
    return EmbeddingStore(
        text=text,
        image=image,
        text_dim=base.text_dim,
        image_dim=base.image_dim,
    )


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    variants = tuple(args.variant or DEFAULT_VARIANTS)
    all_claims = [ClaimRecord.from_dict(row) for row in read_jsonl(args.claims)]
    claims = [claim for claim in all_claims if split_name(claim) == "test"]
    evidence = [EvidenceRecord.from_dict(row) for row in read_jsonl(args.evidence)]
    if not claims or not evidence:
        raise ValueError("Test claims and evidence must be non-empty")
    evidence_by_id = {item.evidence_id: item for item in evidence}
    conflict_donors: dict[str, str] | None = None
    if args.conflict_manifest:
        rows = read_jsonl(args.conflict_manifest)
        conflict_donors = {}
        for row in rows:
            claim_id = str(row.get("claim_id", ""))
            donor_id = str(row.get("donor_evidence_id", ""))
            verified = row.get("semantic_conflict_verified") is True
            if not claim_id or not donor_id or not verified:
                raise ValueError(
                    "Every conflict-manifest row needs claim_id, donor_evidence_id, "
                    "and semantic_conflict_verified=true"
                )
            if claim_id in conflict_donors:
                raise ValueError(f"Duplicate conflict-manifest claim_id: {claim_id}")
            if donor_id not in evidence_by_id:
                raise ValueError(f"Unknown donor evidence: {donor_id}")
            conflict_donors[claim_id] = donor_id
        missing = sorted({claim.claim_id for claim in claims} - set(conflict_donors))
        if missing:
            raise ValueError(f"Conflict manifest is missing test claims: {missing[:5]}")
    scenario_claims, scenario_evidence, aliases, image_aliases = _scenario_records(
        claims, all_claims, evidence_by_id, conflict_donors
    )
    base_store = EmbeddingStore.from_npz(args.text_features, args.image_features)
    injected = {
        item.evidence_id: item.title
        for item in scenario_evidence["post_cutoff_verdict"]
        if item.evidence_id.endswith("__post_cutoff_verdict")
    }
    if args.skip_synthetic_encoding:
        encoded = {
            item_id: np.zeros(base_store.text_dim, dtype=np.float32)
            for item_id in injected
        }
    else:
        retrieval = config["retrieval"]
        encoder = BgeM3Encoder(
            retrieval["dense_text_model"], revision=retrieval["dense_text_revision"]
        )
        matrix = encoder.encode(list(injected.values()))
        encoded = {
            item_id: matrix[index]
            for index, item_id in enumerate(injected)
        }
    stores = {
        name: _store_for_scenario(
            base_store,
            scenario_evidence[name],
            aliases,
            image_aliases,
            name,
            encoded,
        )
        for name in SCENARIOS
    }
    truth = [claim.normalized_label for claim in claims]
    matrix_root = Path(args.matrix_root).resolve()
    output_root = Path(args.output_directory).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, object]] = []
    for variant in variants:
        seed_roots = sorted((matrix_root / variant).glob("seed_*"))
        if not seed_roots:
            raise FileNotFoundError(f"No completed seed runs for {variant}")
        for seed_root in seed_roots:
            summary = json.loads((seed_root / "run_summary.json").read_text(encoding="utf-8"))
            settings = TrainingSettings(**summary["settings"])
            original_rows = {
                str(row["claim_id"]): row for row in read_jsonl(seed_root / "predictions.jsonl")
            }
            for scenario in SCENARIOS:
                examples = build_examples(
                    scenario_claims[scenario],
                    scenario_evidence[scenario],
                    stores[scenario],
                    **_variant_options(variant),
                )
                predictions = predict_from_checkpoint(
                    examples,
                    settings,
                    summary["checkpoint"],
                    temperature=float(summary["temperature"]),
                    abstention_threshold=float(summary["abstention_threshold"]),
                    device_name=args.device,
                )
                destination = output_root / "predictions" / variant / seed_root.name / f"{scenario}.jsonl"
                write_jsonl(destination, predictions)
                predicted = [str(row["predicted_label"]) for row in predictions]
                metrics = classification_metrics(truth, predicted)
                changes = [
                    str(row["predicted_label"])
                    != str(original_rows[str(row["claim_id"])]["predicted_label"])
                    for row in predictions
                ]
                confidence_shift = [
                    float(row["answer_confidence"])
                    - float(original_rows[str(row["claim_id"])]["answer_confidence"])
                    for row in predictions
                ]
                runs.append(
                    {
                        "variant": variant,
                        "seed": settings.seed,
                        "scenario": scenario,
                        "macro_f1": float(metrics["macro_f1"]),
                        "balanced_accuracy": float(metrics["balanced_accuracy"]),
                        "prediction_change_rate": float(np.mean(changes)),
                        "mean_confidence_shift": float(np.mean(confidence_shift)),
                        "predictions": str(destination),
                    }
                )
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in runs:
        grouped[(str(row["variant"]), str(row["scenario"]))].append(row)
    aggregate = [
        {
            "variant": variant,
            "scenario": scenario,
            "seed_count": len(rows),
            **{
                metric: _mean_std([float(row[metric]) for row in rows])
                for metric in (
                    "macro_f1",
                    "balanced_accuracy",
                    "prediction_change_rate",
                    "mean_confidence_shift",
                )
            },
        }
        for (variant, scenario), rows in sorted(grouped.items())
    ]
    report = {
        "status": "checked_intervention_evaluation",
        "claims": str(Path(args.claims).resolve()),
        "evidence": str(Path(args.evidence).resolve()),
        "matrix_root": str(matrix_root),
        "variants": list(variants),
        "scenarios": list(SCENARIOS),
        "test_claims": len(claims),
        "diagnostic_skip_synthetic_encoding": bool(args.skip_synthetic_encoding),
        "conflict_donor_policy": (
            "audited_manifest" if conflict_donors is not None else "automatic_opposite_label_diagnostic"
        ),
        "conflict_manifest": str(Path(args.conflict_manifest).resolve()) if args.conflict_manifest else None,
        "runs": runs,
        "aggregate": aggregate,
    }
    write_json(output_root / "intervention_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
