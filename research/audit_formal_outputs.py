"""Independent audit of a completed DACS-SCV formal output directory.

The script recomputes primary endpoints from per-claim records, verifies the
artifact and source-code ledgers, checks the frozen-ranking invariants, and
replays the deterministic grouped-bootstrap aggregation from the preserved
implementation. It never changes the returned formal archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean


SEEDS = (13, 37, 73, 101, 137)
TRAINED_VARIANTS = ("claim_only", "all_pool", "relevance", "flat", "trace_no_cf")
DERIVED_VARIANTS = (
    "fixed_top1",
    "fixed_top2",
    "dacs_scv",
    "dacs_no_valid_count",
    "dacs_no_visual_count",
)
ALL_VARIANTS = TRAINED_VARIANTS + DERIVED_VARIANTS
TOLERANCE = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def close(actual: float, expected: float, label: str, failures: list[str]) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=TOLERANCE):
        failures.append(f"{label}: expected {expected!r}, recomputed {actual!r}")


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def label_macro_f1(gold: list[str], predicted: list[str]) -> float:
    labels = sorted(set(gold) | set(predicted))
    scores = []
    for label in labels:
        true_positive = sum(g == label and p == label for g, p in zip(gold, predicted, strict=True))
        false_positive = sum(g != label and p == label for g, p in zip(gold, predicted, strict=True))
        false_negative = sum(g == label and p != label for g, p in zip(gold, predicted, strict=True))
        precision = safe_div(true_positive, true_positive + false_positive)
        recall = safe_div(true_positive, true_positive + false_negative)
        scores.append(safe_div(2.0 * precision * recall, precision + recall))
    return fmean(scores) if scores else 0.0


def citation_row(predicted: list[str], gold: list[str]) -> tuple[float, float, float]:
    predicted_set, gold_set = set(predicted), set(gold)
    overlap = len(predicted_set & gold_set)
    precision = safe_div(overlap, len(predicted_set))
    recall = safe_div(overlap, len(gold_set))
    return precision, recall, safe_div(2.0 * precision * recall, precision + recall)


def citation_metrics(records: list[dict]) -> dict:
    rows = [citation_row(row["predicted_citations"], row["gold_citations"]) for row in records]
    predicted_counts = [len(set(row["predicted_citations"])) for row in records]
    gold_counts = [len(set(row["gold_citations"])) for row in records]
    return {
        "precision": fmean(row[0] for row in rows),
        "recall": fmean(row[1] for row in rows),
        "f1": fmean(row[2] for row in rows),
        "mae": fmean(abs(left - right) for left, right in zip(predicted_counts, gold_counts, strict=True)),
        "exact_match": fmean(float(left == right) for left, right in zip(predicted_counts, gold_counts, strict=True)),
        "mean_predicted": fmean(predicted_counts),
        "mean_gold": fmean(gold_counts),
    }


def ranking_metrics(records: list[dict], k: int = 10) -> dict:
    recalls, average_precisions, ndcgs = [], [], []
    for row in records:
        target = set(row["gold_citations"])
        hits = [int(item in target) for item in row["ranked_citations"][:k]]
        recalls.append(safe_div(sum(hits), len(target)))
        precisions = [sum(hits[: index + 1]) / (index + 1) for index, hit in enumerate(hits) if hit]
        average_precisions.append(safe_div(sum(precisions), len(target)))
        dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
        ideal = sum(1.0 / math.log2(index + 2) for index in range(min(len(target), k)))
        ndcgs.append(safe_div(dcg, ideal))
    return {
        "recall@10": fmean(recalls),
        "map@10": fmean(average_precisions),
        "ndcg@10": fmean(ndcgs),
    }


def probabilistic_metrics(records: list[dict], bins: int = 10) -> dict:
    labels = sorted({row["gold_label"] for row in records})
    brier, nll, confidences, correctness = [], [], [], []
    for row in records:
        probabilities = row["probabilities"]["clean"]
        target = row["gold_label"]
        brier.append(sum((float(probabilities.get(label, 0.0)) - float(label == target)) ** 2 for label in labels))
        nll.append(-math.log(max(float(probabilities.get(target, 0.0)), 1e-12)))
        predicted = max(labels, key=lambda label: float(probabilities.get(label, 0.0)))
        confidences.append(float(probabilities[predicted]))
        correctness.append(float(predicted == target))
    ece = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [
            member
            for member, value in enumerate(confidences)
            if lower <= value and (value < upper or (index == bins - 1 and value <= upper))
        ]
        if members:
            ece += len(members) / len(records) * abs(
                fmean(correctness[member] for member in members)
                - fmean(confidences[member] for member in members)
            )
    return {"brier": fmean(brier), "nll": fmean(nll), "ece": ece}


def verify_endpoint_records(
    variant: str,
    seed: int,
    records: list[dict],
    metrics: dict,
    failures: list[str],
) -> dict:
    prefix = f"{variant}/seed-{seed}"
    require(len(records) == 2000, f"{prefix}: expected 2,000 records", failures)
    claim_ids = [row["claim_id"] for row in records]
    require(len(set(claim_ids)) == len(claim_ids), f"{prefix}: duplicate claim IDs", failures)
    require(len({row["group_id"] for row in records}) == 786, f"{prefix}: expected 786 source groups", failures)
    for row in records:
        predicted = max(row["probabilities"]["clean"], key=row["probabilities"]["clean"].get)
        require(predicted == row["predicted_label"], f"{prefix}/{row['claim_id']}: label/probability mismatch", failures)
        if "predicted_citation_count" in row:
            require(
                int(row["predicted_citation_count"]) == len(row["predicted_citations"]),
                f"{prefix}/{row['claim_id']}: citation-count mismatch",
                failures,
            )

    gold = [row["gold_label"] for row in records]
    predicted = [row["predicted_label"] for row in records]
    verdict = label_macro_f1(gold, predicted)
    citation = citation_metrics(records)
    ranking = ranking_metrics(records)
    probabilistic = probabilistic_metrics(records)
    close(verdict, metrics["verdict_macro_f1"], f"{prefix}/verdict_macro_f1", failures)
    for key in ("precision", "recall", "f1"):
        close(citation[key], metrics["citation_macro"][key], f"{prefix}/citation_{key}", failures)
    for key in ("mae", "exact_match", "mean_predicted", "mean_gold"):
        close(citation[key], metrics["citation_cardinality"][key], f"{prefix}/count_{key}", failures)
    for key in ("recall@10", "map@10", "ndcg@10"):
        close(ranking[key], metrics["evidence_ranking"][key], f"{prefix}/{key}", failures)
    for key in ("brier", "nll", "ece"):
        close(probabilistic[key], metrics["probabilistic"][key], f"{prefix}/{key}", failures)
    return {
        "variant": variant,
        "seed": seed,
        "verdict_macro_f1": verdict,
        "citation_macro": {key: citation[key] for key in ("precision", "recall", "f1")},
        "citation_cardinality": {key: citation[key] for key in ("mae", "exact_match", "mean_predicted", "mean_gold")},
        "_comparison_records": records,
    }


def compare_nested(actual: object, expected: object, prefix: str, failures: list[str]) -> None:
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f"{prefix}: expected mapping", failures)
        if not isinstance(actual, dict):
            return
        require(set(actual) == set(expected), f"{prefix}: key mismatch", failures)
        for key in expected:
            if key in actual:
                compare_nested(actual[key], expected[key], f"{prefix}.{key}", failures)
    elif isinstance(expected, list):
        require(actual == expected, f"{prefix}: list mismatch", failures)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        close(float(actual), float(expected), prefix, failures)
    else:
        require(actual == expected, f"{prefix}: expected {expected!r}, got {actual!r}", failures)


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Extracted formal output directory")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    project_root = arguments.project_root.resolve()
    failures: list[str] = []

    inventory = read_json(root / "audit" / "artifact_inventory.json")
    listed = set(inventory["files"])
    require(inventory["file_count"] == len(listed) == 252, "artifact inventory count mismatch", failures)
    for relative, expected in inventory["files"].items():
        path = root / Path(relative)
        require(path.is_file(), f"missing artifact: {relative}", failures)
        if path.is_file():
            require(path.stat().st_size == expected["bytes"], f"byte mismatch: {relative}", failures)
            require(sha256_file(path) == expected["sha256"], f"hash mismatch: {relative}", failures)
    actual_files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    require(actual_files - listed == {"audit/artifact_inventory.json"}, "unexpected unlisted artifacts", failures)

    manifest = read_json(root / "audit" / "run_manifest.json")
    summary = read_json(root / "summary.json")
    require(manifest["profile"] == summary["profile"] == "formal", "profile is not formal", failures)
    require(summary["formal_run_completed"] is True, "formal completion flag is false", failures)
    gate = manifest["dataset_manifests"]["manifest.json"]["data_quality"]["release_gate"]
    require(gate == {"no_group_overlap": True, "no_exact_visual_overlap": True, "no_gold_content_overlap": True, "pass": True}, "release gate failed", failures)
    require(manifest["dataset_manifests"]["manifest.json"]["splits"]["test"]["count"] == 2000, "official-test count mismatch", failures)
    for relative, expected in manifest["code_hashes"].items():
        path = project_root / Path(relative)
        require(path.is_file(), f"missing frozen code: {relative}", failures)
        if path.is_file():
            require(sha256_file(path) == expected, f"frozen-code hash mismatch: {relative}", failures)

    results: list[dict] = []
    records_by_variant_seed: dict[tuple[str, int], list[dict]] = {}
    metrics_by_variant_seed: dict[tuple[str, int], dict] = {}
    for variant in TRAINED_VARIANTS:
        for seed in SEEDS:
            evaluation = root / variant / f"seed-{seed}" / "evaluation"
            records = read_jsonl(evaluation / "comparison_records.jsonl")
            metrics = read_json(evaluation / "metrics.json")
            results.append(verify_endpoint_records(variant, seed, records, metrics, failures))
            records_by_variant_seed[(variant, seed)] = records
            metrics_by_variant_seed[(variant, seed)] = metrics
    for seed in SEEDS:
        for variant in DERIVED_VARIANTS:
            evaluation = root / variant / f"seed-{seed}" / "evaluation"
            records = read_jsonl(evaluation / "comparison_records.jsonl")
            metrics = read_json(evaluation / "metrics.json")
            results.append(verify_endpoint_records(variant, seed, records, metrics, failures))
            records_by_variant_seed[(variant, seed)] = records
            metrics_by_variant_seed[(variant, seed)] = metrics

    identity_checks = 0
    for seed in SEEDS:
        base = records_by_variant_seed[("trace_no_cf", seed)]
        for variant in ("fixed_top1", "fixed_top2", "dacs_scv", "dacs_no_valid_count", "dacs_no_visual_count"):
            candidate = records_by_variant_seed[(variant, seed)]
            require(len(base) == len(candidate), f"{variant}/seed-{seed}: alignment length mismatch", failures)
            for left, right in zip(base, candidate, strict=True):
                require(left["claim_id"] == right["claim_id"], f"{variant}/seed-{seed}: claim order changed", failures)
                require(left["predicted_label"] == right["predicted_label"], f"{variant}/seed-{seed}: verdict changed", failures)
                require(left["probabilities"]["clean"] == right["probabilities"]["clean"], f"{variant}/seed-{seed}: verdict probabilities changed", failures)
                require(left["ranked_citations"] == right["ranked_citations"], f"{variant}/seed-{seed}: evidence ranking changed", failures)
                require(right["predicted_citations"] == right["ranked_citations"][: len(right["predicted_citations"])], f"{variant}/seed-{seed}: emitted set is not a ranking prefix", failures)
                identity_checks += 1
        for row in base:
            require(row["predicted_citations"] == row["ranked_citations"][:3], f"trace_no_cf/seed-{seed}: not fixed top-three", failures)
        for variant, count in (("fixed_top1", 1), ("fixed_top2", 2)):
            for row in records_by_variant_seed[(variant, seed)]:
                require(row["predicted_citations"] == row["ranked_citations"][:count], f"{variant}/seed-{seed}: fixed-prefix mismatch", failures)

    for variant in ALL_VARIANTS:
        rows = [metrics_by_variant_seed[(variant, seed)] for seed in SEEDS]
        aggregate = summary["summary"][variant]
        close(fmean(row["verdict_macro_f1"] for row in rows), aggregate["verdict_macro_f1_mean"], f"summary/{variant}/verdict", failures)
        close(fmean(row["citation_macro"]["f1"] for row in rows), aggregate["citation_macro_f1_mean"], f"summary/{variant}/citation", failures)
        close(fmean(row["citation_cardinality"]["mae"] for row in rows), aggregate["citation_count_mae_mean"], f"summary/{variant}/count_mae", failures)
        close(fmean(row["citation_cardinality"]["exact_match"] for row in rows), aggregate["citation_count_exact_match_mean"], f"summary/{variant}/count_exact", failures)

    sys.path.insert(0, str(project_root / "experiments" / "src"))
    from trace_scv.evaluation.statistics import (  # pylint: disable=import-outside-toplevel
        paired_group_bootstrap_comparisons,
        seed_level_comparisons,
    )

    recomputed_seed = seed_level_comparisons(results, reference="dacs_scv")
    recomputed_bootstrap = paired_group_bootstrap_comparisons(results, reference="dacs_scv", samples=10000)
    compare_nested(recomputed_seed, summary["paired_seed_comparisons"], "paired_seed_comparisons", failures)
    compare_nested(recomputed_bootstrap, summary["paired_group_bootstrap_comparisons"], "paired_group_bootstrap_comparisons", failures)

    fixed_family_names = ("fixed_top1", "fixed_top2", "trace_no_cf")
    fixed_effects = {
        name: dict(summary["paired_group_bootstrap_comparisons"][name]["citation_macro_f1_difference"])
        for name in fixed_family_names
    }
    family_adjusted = holm_adjust({name: value["p_raw"] for name, value in fixed_effects.items()})
    for name, adjusted in family_adjusted.items():
        fixed_effects[name]["p_holm_fixed_budget_family"] = adjusted
    adaptive_superiority = all(fixed_effects[name]["mean"] > 0.0 for name in fixed_family_names)

    output = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "artifact_checks": {
            "declared_files": len(listed),
            "hash_failures": sum("hash mismatch" in failure for failure in failures),
            "code_files": len(manifest["code_hashes"]),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "test_claims": 2000,
            "source_groups": 786,
            "bootstrap_samples": 10000,
            "release_gate_pass": gate["pass"],
        },
        "record_checks": {
            "variant_seed_runs": len(ALL_VARIANTS) * len(SEEDS),
            "endpoint_records_recomputed": len(ALL_VARIANTS) * len(SEEDS) * 2000,
            "paired_identity_rows_checked": identity_checks,
            "verdict_and_ranking_identity": not any("verdict" in failure or "ranking" in failure for failure in failures),
        },
        "primary_fixed_budget_effects": fixed_effects,
        "adaptive_superiority_pass": adaptive_superiority,
        "interpretation": (
            "The prespecified adaptive-superiority claim passes."
            if adaptive_superiority
            else "The prespecified adaptive-superiority claim fails because at least one fixed budget has a non-positive DACS-SCV effect."
        ),
    }
    rendered = json.dumps(output, indent=2)
    if arguments.output:
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
