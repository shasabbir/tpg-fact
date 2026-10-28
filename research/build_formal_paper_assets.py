"""Generate manuscript tables and the formal-results figure from the frozen ZIP."""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import matplotlib.pyplot as plt
import numpy as np


SEEDS = (13, 37, 73, 101, 137)
METHODS = (
    ("majority", "Majority"),
    ("claim_only", "Claim-only"),
    ("all_pool", "All-pool"),
    ("relevance", "Relevance"),
    ("flat", "Flat selector"),
    ("trace_no_cf", "Graph-fixed top-3"),
    ("fixed_top1", "Graph-fixed top-1"),
    ("fixed_top2", "Graph-fixed top-2"),
    ("dacs_scv", "DACS-SCV"),
    ("dacs_no_valid_count", r"DACS w/o valid count"),
    ("dacs_no_visual_count", r"DACS w/o visual count"),
)


def load_json(archive: zipfile.ZipFile, name: str) -> dict:
    return json.loads(archive.read(name).decode("utf-8"))


def load_jsonl(archive: zipfile.ZipFile, name: str) -> list[dict]:
    return [json.loads(line) for line in archive.read(name).decode("utf-8").splitlines() if line]


def metrics_for(archive: zipfile.ZipFile, variant: str) -> list[dict]:
    if variant == "majority":
        return [load_json(archive, "majority/evaluation/metrics.json")]
    return [
        load_json(archive, f"{variant}/seed-{seed}/evaluation/metrics.json")
        for seed in SEEDS
    ]


def average(rows: list[dict], getter) -> float:
    return fmean(float(getter(row)) for row in rows)


def citation_f1(row: dict) -> float:
    predicted, gold = set(row["predicted_citations"]), set(row["gold_citations"])
    overlap = len(predicted & gold)
    if not predicted or not gold or not overlap:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(gold)
    return 2.0 * precision * recall / (precision + recall)


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    output, running = {}, 0.0
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * pvalues[key]))
        output[key] = running
    return output


def write_dataset_table(manifest: dict, target: Path) -> None:
    splits = manifest["dataset_manifests"]["manifest.json"]["splits"]
    quality = manifest["dataset_manifests"]["manifest.json"]["data_quality"]["splits"]
    rows = []
    for key, label in (("train", "Train"), ("validation", "Validation"), ("test", "Official test")):
        row, qrow = splits[key], quality[key]
        visual_gold = qrow["gold_modality_counts"]["figure"] + qrow["gold_modality_counts"]["table"]
        rows.append(
            f"{label} & {row['count']:,} & {row['groups']:,} & "
            f"{row['labels']['supported']:,} & {row['labels']['refuted']:,} & "
            f"{qrow['gold_evidence_items']:,} & {visual_gold:,} \\\\"
        )
    target.write_text(
        "\n".join(
            [
                r"\begin{table}[t]",
                r"\centering",
                r"\caption{Audited SciVer partitions. Visual gold counts include annotated figures and tables.}",
                r"\label{tab:data-audit}",
                r"\scriptsize",
                r"\setlength{\tabcolsep}{2.5pt}",
                r"\begin{tabular}{@{}lrrrrrr@{}}",
                r"\toprule",
                r"Split & Claims & Src. IDs & Sup. & Ref. & Gold & Visual \\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_main_table(archive: zipfile.ZipFile, target: Path) -> dict[str, dict[str, float]]:
    aggregates = {}
    for variant, _ in METHODS:
        rows = metrics_for(archive, variant)
        aggregates[variant] = {
            "verdict": average(rows, lambda row: row["verdict_macro_f1"]),
            "precision": average(rows, lambda row: row["citation_macro"]["precision"]),
            "recall": average(rows, lambda row: row["citation_macro"]["recall"]),
            "citation": average(rows, lambda row: row["citation_macro"]["f1"]),
            "mae": average(rows, lambda row: row["citation_cardinality"]["mae"]),
            "exact": average(rows, lambda row: row["citation_cardinality"]["exact_match"]),
            "predicted_count": average(rows, lambda row: row["citation_cardinality"]["mean_predicted"]),
        }
    best_citation = max(value["citation"] for value in aggregates.values())
    best_mae = min(value["mae"] for value in aggregates.values())
    best_exact = max(value["exact"] for value in aggregates.values())
    body = []
    for variant, label in METHODS:
        value = aggregates[variant]
        citation = f"{value['citation']:.4f}"
        mae = f"{value['mae']:.4f}"
        exact = f"{value['exact']:.4f}"
        if math.isclose(value["citation"], best_citation):
            citation = rf"\textbf{{{citation}}}"
        if math.isclose(value["mae"], best_mae):
            mae = rf"\textbf{{{mae}}}"
        if math.isclose(value["exact"], best_exact):
            exact = rf"\textbf{{{exact}}}"
        body.append(
            f"{label} & {value['verdict']:.4f} & {value['precision']:.4f} & "
            f"{value['recall']:.4f} & {citation} & {mae} & {exact} \\\\"
        )
    target.write_text(
        "\n".join(
            [
                r"\begin{table*}[t]",
                r"\centering",
                r"\caption{Official-test results. Learned systems report the mean over five frozen seeds; Majority is deterministic. P/R/F1 are claim-level macro citation metrics. Exact is citation-count exact match. Bold marks the best value in each citation/count column, not a superiority claim.}",
                r"\label{tab:main-results}",
                r"\small",
                r"\setlength{\tabcolsep}{5pt}",
                r"\begin{tabular}{@{}lrrrrrr@{}}",
                r"\toprule",
                r"Method & Verdict F1 & Cite P & Cite R & Cite F1 & Count MAE & Exact \\",
                r"\midrule",
                *body[:5],
                r"\midrule",
                *body[5:8],
                r"\midrule",
                *body[8:],
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table*}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return aggregates


def write_effect_table(summary: dict, target: Path) -> dict[str, dict]:
    mapping = (
        ("fixed_top1", "Fixed top-1"),
        ("fixed_top2", "Fixed top-2"),
        ("trace_no_cf", "Fixed top-3"),
    )
    effects = {
        key: dict(summary["paired_group_bootstrap_comparisons"][key]["citation_macro_f1_difference"])
        for key, _ in mapping
    }
    adjusted = holm_adjust({key: value["p_raw"] for key, value in effects.items()})
    rows = []
    for key, label in mapping:
        value = effects[key]
        value["p_holm_fixed_family"] = adjusted[key]
        rows.append(
            f"{label} & {value['mean']:+.4f} & [{value['lower']:+.4f}, {value['upper']:+.4f}] & "
            f"{adjusted[key]:.4f} \\\\"
        )
    target.write_text(
        "\n".join(
            [
                r"\begin{table}[t]",
                r"\centering",
                r"\caption{Paired DACS-SCV minus fixed-budget citation-F1 effects on the official test. Intervals use 10,000 source-paper-grouped bootstrap samples; $p_{\mathrm{H}}$ applies Holm correction to these three prespecified comparisons.}",
                r"\label{tab:fixed-effects}",
                r"\scriptsize",
                r"\setlength{\tabcolsep}{3pt}",
                r"\begin{tabular}{@{}lrrr@{}}",
                r"\toprule",
                r"Comparator & $\Delta$F1 & 95\% CI & $p_{\mathrm{H}}$ \\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return effects


def build_analysis_figure(archive: zipfile.ZipFile, target: Path) -> dict:
    variants = ("fixed_top1", "fixed_top2", "trace_no_cf", "dacs_scv")
    labels = ("Top-1", "Top-2", "Top-3", "DACS")
    seed_scores = defaultdict(list)
    stratum_scores = defaultdict(list)
    counts = {}
    for variant in variants:
        for seed in SEEDS:
            metrics = load_json(archive, f"{variant}/seed-{seed}/evaluation/metrics.json")
            seed_scores[variant].append(float(metrics["citation_macro"]["f1"]))
            records = load_jsonl(archive, f"{variant}/seed-{seed}/evaluation/comparison_records.jsonl")
            for bucket in ("1", "2", "3+"):
                subset = [
                    row
                    for row in records
                    if (len(row["gold_citations"]) == int(bucket) if bucket != "3+" else len(row["gold_citations"]) >= 3)
                ]
                counts[bucket] = len(subset)
                stratum_scores[(variant, bucket)].append(fmean(citation_f1(row) for row in subset))

    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.7), constrained_layout=True)
    colors = ("#4C78A8", "#F58518", "#54A24B", "#B279A2")
    for variant, label, color in zip(variants, labels, colors, strict=True):
        axes[0].plot(SEEDS, seed_scores[variant], marker="o", linewidth=1.5, markersize=3.5, label=label, color=color)
    axes[0].set_title("(a) Variation across frozen seeds")
    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("Citation macro-F1")
    axes[0].set_xticks(SEEDS)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=2, frameon=False, fontsize=7)

    buckets = ("1", "2", "3+")
    x_values = np.arange(len(buckets))
    width = 0.19
    for index, (variant, label, color) in enumerate(zip(variants, labels, colors, strict=True)):
        values = [fmean(stratum_scores[(variant, bucket)]) for bucket in buckets]
        axes[1].bar(x_values + (index - 1.5) * width, values, width=width, label=label, color=color)
    axes[1].set_title("(b) Performance by required evidence count")
    axes[1].set_xlabel("Gold citation count (claims)")
    axes[1].set_ylabel("Citation macro-F1")
    axes[1].set_xticks(x_values, [f"{bucket}\n($n={counts[bucket]}$)" for bucket in buckets])
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_axisbelow(True)
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)
    return {
        "seed_scores": dict(seed_scores),
        "stratum_scores": {
            f"{variant}/{bucket}": fmean(values)
            for (variant, bucket), values in stratum_scores.items()
        },
        "stratum_counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--paper-root", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path)
    arguments = parser.parse_args()
    paper_root = arguments.paper_root.resolve()
    tables = paper_root / "tables"
    figures = paper_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(arguments.archive) as archive:
        summary = load_json(archive, "summary.json")
        manifest = load_json(archive, "audit/run_manifest.json")
        write_dataset_table(manifest, tables / "dataset_statistics.tex")
        aggregates = write_main_table(archive, tables / "main_results.tex")
        effects = write_effect_table(summary, tables / "primary_effects.tex")
        analysis = build_analysis_figure(archive, figures / "formal_analysis.pdf")
    if arguments.audit_output:
        arguments.audit_output.write_text(
            json.dumps({"aggregates": aggregates, "effects": effects, "analysis": analysis}, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
