from __future__ import annotations

import csv
import json
from dataclasses import fields
from pathlib import Path
from statistics import fmean
from typing import Iterable, Sequence

import numpy as np

from trace_scv.q1_phase2_reporting import (
    _write_robustness_reporting,
    bootstrap_two_sided_p,
    holm_adjust,
)
from trace_scv.selection.calibration import (
    binary_brier_score,
    binary_expected_calibration_error,
    binary_nll,
)
from trace_scv.selection.probability_calibration import (
    ProbabilityCalibrator,
    apply_calibrator,
)


METHODS = ("fixed", "threshold", "count", "cefes", "size_penalized", "rc_cefes")
METHOD_LABELS = {
    "fixed": "Fixed top-5",
    "threshold": "Calibrated threshold",
    "count": "Learned count",
    "cefes": "Expected-F1",
    "size_penalized": "Size-penalized expected-F1",
    "rc_cefes": "Size + redundancy expected-F1",
}

REACHABILITY_LABELS = {
    "none": "No gold reachable",
    "partial": "Partially reachable",
    "complete": "Complete gold reachable",
}


def _read_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _model_from_dict(value: dict[str, object]) -> ProbabilityCalibrator:
    allowed = {field.name for field in fields(ProbabilityCalibrator)}
    cleaned = {key: value[key] for key in allowed if key in value}
    for key in ("coefficients", "thresholds", "values"):
        cleaned[key] = tuple(map(float, cleaned.get(key, ())))
    return ProbabilityCalibrator(**cleaned)


def _reachability_stratum(reachable_count: int, gold_count: int) -> str:
    if gold_count <= 0:
        raise ValueError("Reachability strata require a nonempty gold set.")
    if reachable_count < 0 or reachable_count > gold_count:
        raise ValueError("Reachable gold count must lie between zero and gold count.")
    if reachable_count == 0:
        return "none"
    if reachable_count == gold_count:
        return "complete"
    return "partial"


def _reachability_reporting(
    selector_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Summarize selector performance by locked-validation reachability."""

    claim_values: dict[str, dict[str, object]] = {}
    methods = ("fixed", "threshold", "cefes")
    for seed_value in (13, 37, 73, 101, 137):
        claim_rows = [
            row
            for row in _read_jsonl(
                selector_root / f"seed_{seed_value}_claim_records.jsonl"
            )
            if row["split"] == "validation"
        ]
        fixed_rows = {
            str(row["key"]): row for row in claim_rows if row["method"] == "fixed"
        }
        reachable: dict[str, int] = {}
        for row in _read_jsonl(
            selector_root / f"seed_{seed_value}_candidate_records.jsonl"
        ):
            if row["split"] != "validation" or not bool(row["gold"]):
                continue
            key = f"validation:{row['claim_id']}"
            reachable[key] = reachable.get(key, 0) + 1

        by_method: dict[str, dict[str, float]] = {method: {} for method in methods}
        for row in claim_rows:
            method = str(row["method"])
            if method in by_method:
                by_method[method][str(row["key"])] = float(row["f1"])

        for key, fixed_row in fixed_rows.items():
            gold_count = int(fixed_row["gold_count"])
            reachable_count = reachable.get(key, 0)
            stratum = _reachability_stratum(reachable_count, gold_count)
            item = claim_values.setdefault(
                key,
                {
                    "claim_key": key,
                    "gold_count": gold_count,
                    "reachable_count": reachable_count,
                    "stratum": stratum,
                    "oracle_f1": 2.0 * reachable_count / (gold_count + reachable_count),
                    **{method: [] for method in methods},
                },
            )
            if (
                item["gold_count"] != gold_count
                or item["reachable_count"] != reachable_count
                or item["stratum"] != stratum
            ):
                raise ValueError(f"Retrieval reachability changed across seeds for {key}.")
            for method in methods:
                item[method].append(by_method[method][key])

    detail_rows: list[dict[str, object]] = []
    for key in sorted(claim_values):
        item = claim_values[key]
        detail_rows.append(
            {
                "claim_key": key,
                "stratum": item["stratum"],
                "gold_count": item["gold_count"],
                "reachable_count": item["reachable_count"],
                "oracle_f1": item["oracle_f1"],
                **{
                    f"{method}_f1": fmean(item[method])
                    for method in methods
                },
            }
        )

    summary_rows: list[dict[str, object]] = []
    for stratum in ("none", "partial", "complete"):
        rows = [row for row in detail_rows if row["stratum"] == stratum]
        summary_rows.append(
            {
                "stratum": stratum,
                "label": REACHABILITY_LABELS[stratum],
                "claims": len(rows),
                "claim_share": len(rows) / len(detail_rows),
                "oracle_f1": fmean(float(row["oracle_f1"]) for row in rows),
                "fixed_f1": fmean(float(row["fixed_f1"]) for row in rows),
                "threshold_f1": fmean(float(row["threshold_f1"]) for row in rows),
                "expected_f1": fmean(float(row["cefes_f1"]) for row in rows),
            }
        )
    return summary_rows, detail_rows


def _aligned_effect(
    selector_root: Path,
    left: str,
    right: str,
    *,
    samples: int,
    seed: int,
) -> dict[str, object]:
    effects: dict[str, list[float]] = {}
    for seed_value in (13, 37, 73, 101, 137):
        rows = list(_read_jsonl(selector_root / f"seed_{seed_value}_claim_records.jsonl"))
        by_method: dict[str, dict[str, float]] = {}
        for row in rows:
            by_method.setdefault(str(row["method"]), {})[str(row["key"])] = float(row["f1"])
        for key, left_value in by_method[left].items():
            effects.setdefault(key, []).append(left_value - by_method[right][key])
    keys = sorted(effects)
    values = np.asarray([fmean(effects[key]) for key in keys], dtype=np.float64)
    generator = np.random.default_rng(seed)
    bootstrap: list[float] = []
    remaining = samples
    while remaining:
        current = min(512, remaining)
        indices = generator.integers(0, len(values), size=(current, len(values)))
        bootstrap.extend(values[indices].mean(axis=1).tolist())
        remaining -= current
    return {
        "left": left,
        "right": right,
        "mean": float(values.mean()),
        "lower": float(np.quantile(bootstrap, 0.025)),
        "upper": float(np.quantile(bootstrap, 0.975)),
        "claims": len(keys),
        "samples": samples,
    }


def _calibration_reporting(
    selector: dict[str, object], selector_root: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    candidate_metrics: dict[str, list[dict[str, float]]] = {}
    reliability_rows: list[dict[str, object]] = []
    for seed_row in selector["seeds"]:
        seed = int(seed_row["seed"])
        validation = [
            row
            for row in _read_jsonl(selector_root / f"seed_{seed}_candidate_records.jsonl")
            if row["split"] == "validation"
        ]
        raw = np.asarray([float(row["raw_probability"]) for row in validation])
        calibrated = np.asarray(
            [float(row["calibrated_probability"]) for row in validation]
        )
        labels = np.asarray([int(bool(row["gold"])) for row in validation])
        modalities = np.asarray([str(row["modality"]) for row in validation])
        for modality in ("text", "image"):
            indices = np.flatnonzero(modalities == modality)
            report = seed_row["calibration_report"][modality]
            for candidate in report["candidates"]:
                kind = str(candidate["kind"])
                model = _model_from_dict(candidate["model"])
                values = apply_calibrator(raw[indices], model)
                candidate_metrics.setdefault(kind, []).append(
                    {
                        "brier": binary_brier_score(values, labels[indices]),
                        "nll": binary_nll(values, labels[indices]),
                        "ece": binary_expected_calibration_error(values, labels[indices]),
                    }
                )
        bin_indices = np.minimum((calibrated * 10).astype(int), 9)
        for bin_index in range(10):
            indices = np.flatnonzero(bin_indices == bin_index)
            if len(indices):
                reliability_rows.append(
                    {
                        "seed": seed,
                        "bin": bin_index,
                        "count": len(indices),
                        "mean_probability": float(np.mean(calibrated[indices])),
                        "observed_rate": float(np.mean(labels[indices])),
                    }
                )
    calibration_rows = []
    for kind, rows in candidate_metrics.items():
        calibration_rows.append(
            {
                "calibrator": kind,
                "brier": fmean(row["brier"] for row in rows),
                "nll": fmean(row["nll"] for row in rows),
                "ece": fmean(row["ece"] for row in rows),
            }
        )
    calibration_rows.sort(key=lambda row: (float(row["brier"]), str(row["calibrator"])))
    return calibration_rows, reliability_rows


def _write_tables(
    root: Path,
    final: dict[str, object],
    method_rows: Sequence[dict[str, object]],
    effect_rows: Sequence[dict[str, object]],
    calibration_rows: Sequence[dict[str, object]],
    partition_rows: Sequence[dict[str, object]],
    verifier_rows: Sequence[dict[str, object]],
    reachability_rows: Sequence[dict[str, object]],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    population = final["population"]
    packet = final["retained_packet"]
    (root / "population_flow.tex").write_text(
        """\\begin{table}[t]\n\\centering\n\\caption{Evaluation population and reachable-evidence boundary.}\n\\label{tab:population-flow}\n\\begin{tabular}{lr}\n\\toprule\nPopulation or boundary & Value \\\\\n\\midrule\n"""
        + f"Released train and validation claims & {population['released_claims']:,} \\\\\n"
        + f"Claims with nonempty complete gold & {population['eligible_claims']:,} \\\\\n"
        + f"Locked validation claims & {population['validation_claims']:,} \\\\\n"
        + f"Eligible claims with no reachable gold & {int(packet['eligible_claims_with_zero_reachable_gold']):,} \\\\\n"
        + f"Packet-oracle evidence F1 & {packet['packet_oracle_f1']:.4f} \\\\\n"
        + f"Complete-gold packet coverage & {100 * packet['complete_gold_claim_coverage']:.2f}\\% \\\\\n"
        + "\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8",
    )
    (root / "operating_depth_retrieval.tex").write_text(
        """\\begin{table}[t]\n\\centering\n\\caption{Retrieval at the packet depths received by every selector.}\n\\label{tab:operating-retrieval}\n\\begin{tabular}{lrr}\n\\toprule\nModality & Retained depth & Complete-gold recall \\\\\n\\midrule\n"""
        + f"Text & 40 & {packet['text_recall_at_40']:.4f} \\\\\n"
        + f"Image & 20 & {packet['image_recall_at_20']:.4f} \\\\\n"
        + "\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8",
    )
    retrieval = final["retrieval_matrix"]
    retrieval_rows = (
        ("Text", "Dense sentence", retrieval["text"]["dense_sentence"]),
        ("Text", "BM25 sentence", retrieval["text"]["bm25_sentence"]),
        ("Text", "Dense + BM25 RRF", retrieval["text"]["dense_bm25_rrf"]),
        (
            "Text",
            "RRF + cross-encoder",
            retrieval["text"]["dense_bm25_rrf_cross_encoder"],
        ),
        ("Image", "SigLIP2", retrieval["image"]["siglip"]),
        ("Image", "Filename dense", retrieval["image"]["filename_dense"]),
        ("Image", "SigLIP2 + filename RRF", retrieval["image"]["siglip_filename_rrf"]),
    )
    retrieval_lines = [
        f"{modality} & {label} & {metrics['recall_at_5']:.4f} & "
        f"{metrics['recall_at_10']:.4f} & {metrics['recall_at_20']:.4f} & "
        f"{metrics['recall_at_50']:.4f} & {metrics['map_at_100']:.4f} & "
        f"{metrics['ndcg_at_100']:.4f} \\\\"
        for modality, label, metrics in retrieval_rows
    ]
    (root / "retrieval_matrix.tex").write_text(
        """\\begin{table*}[t]
\\centering
\\caption{Complete-gold retrieval quality before packet truncation. Recall includes unretrieved released-gold items in the denominator; MAP and nDCG use the depth-100 ranked lists.}
\\label{tab:retrieval-matrix}
\\small
\\setlength{\\tabcolsep}{5pt}
\\begin{tabular}{llrrrrrr}
\\toprule
Modality & Retriever & R@5 & R@10 & R@20 & R@50 & MAP@100 & nDCG@100 \\\\
\\midrule
"""
        + "\n".join(retrieval_lines)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n",
        encoding="utf-8",
    )
    calibration_lines = [
        f"{row['calibrator'].replace('_', ' ').title()} & {row['brier']:.4f} & {row['nll']:.4f} & {row['ece']:.4f} \\\\"
        for row in calibration_rows
    ]
    (root / "calibration_summary.tex").write_text(
        """\\begin{table}[t]\n\\centering\n\\caption{Validation calibration averaged across modalities and deterministic partitions.}\n\\label{tab:calibration}\n\\begin{tabular}{lrrr}\n\\toprule\nMap & Brier $\\downarrow$ & NLL $\\downarrow$ & ECE $\\downarrow$ \\\\\n\\midrule\n"""
        + "\n".join(calibration_lines)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8",
    )
    method_lines = [
        f"{row['label']} & {row['precision']:.4f} & {row['recall']:.4f} & {row['f1']:.4f} & {row['exact']:.4f} & {row['selected_count']:.2f} \\\\"
        for row in method_rows
    ]
    (root / "selector_means.tex").write_text(
        """\\begin{table*}[t]\n\\centering\n\\caption{Locked-validation evidence-set results over 1,490 claims. Values are means across five deterministic training-role partitions.}\n\\label{tab:selector-means}\n\\begin{tabular}{lrrrrr}\n\\toprule\nMethod & Precision & Recall & F1 & Exact match & Selected \\\\\n\\midrule\n"""
        + "\n".join(method_lines)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n",
        encoding="utf-8",
    )
    partition_lines = [
        f"{row['partition']} & {row['fixed_f1']:.4f} & {row['threshold_f1']:.4f} & "
        f"{row['expected_f1']:.4f} & {row['expected_minus_fixed']:+.4f} & "
        f"{row['expected_precision']:.4f} & {row['expected_recall']:.4f} & "
        f"{row['expected_selected']:.2f} \\\\"
        for row in partition_rows
    ]
    (root / "partition_selector_results.tex").write_text(
        """\\begin{table*}[t]
\\centering
\\caption{Partition-level locked-validation evidence selection. Every row contains the same 1,490 validation claims; partitions change only the training, calibration, and model-selection roles.}
\\label{tab:partition-selector-results}
\\small
\\setlength{\\tabcolsep}{5pt}
\\begin{tabular}{rrrrrrrr}
\\toprule
Partition & Fixed F1 & Threshold F1 & Expected-F1 & Difference & Precision & Recall & Selected \\\\
\\midrule
"""
        + "\n".join(partition_lines)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n",
        encoding="utf-8",
    )
    effect_lines = [
        f"{row['label']} & {row['mean']:+.4f} & [{row['lower']:+.4f}, {row['upper']:+.4f}] & {row['holm_p']:.4f} \\\\"
        for row in effect_rows
    ]
    (root / "selector_effects.tex").write_text(
        """\\begin{table}[t]\n\\centering\n\\caption{Claim-aligned evidence-F1 effects against fixed top-5.}\n\\label{tab:selector-effects-detail}\n\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{lrrr}\n\\toprule\nMethod & Mean & 95\\% interval & Holm $p$ \\\\\n\\midrule\n"""
        + "\n".join(effect_lines)
        + "\n\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n",
        encoding="utf-8",
    )
    verifier_lines = [
        f"{row['seed']} & {row['fixed']:.4f} & {row['threshold']:.4f} & {row['expected_f1']:.4f} & {row['difference']:+.4f} \\\\"
        for row in verifier_rows
    ]
    (root / "verifier_effects.tex").write_text(
        """\\begin{table}[t]\n\\centering\n\\caption{Shared-verifier macro-F1 by deterministic partition.}\n\\label{tab:verifier-effects}\n\\begin{tabular}{lrrrr}\n\\toprule\nPartition & Fixed & Threshold & Expected-F1 & Difference \\\\\n\\midrule\n"""
        + "\n".join(verifier_lines)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8",
    )
    reachability_lines = [
        f"{row['label']} & {row['claims']:,} ({100 * row['claim_share']:.1f}\\%) & "
        f"{row['oracle_f1']:.3f} & {row['fixed_f1']:.3f} & "
        f"{row['threshold_f1']:.3f} & {row['expected_f1']:.3f} \\\\"
        for row in reachability_rows
    ]
    (root / "reachability_strata.tex").write_text(
        """\\begin{table*}[t]
\\centering
\\caption{Locked-validation performance by packet reachability. The packet oracle selects every reachable released-gold item. Selector values average claim-level F1 across five deterministic training-role partitions.}
\\label{tab:reachability-strata}
\\begin{tabular}{lrrrrr}
\\toprule
Reachability stratum & Claims (share) & Packet oracle & Fixed top-5 & Threshold & Expected-F1 \\\\
\\midrule
"""
        + "\n".join(reachability_lines)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n",
        encoding="utf-8",
    )


def _write_figures(
    root: Path,
    effect_rows: Sequence[dict[str, object]],
    reliability_rows: Sequence[dict[str, object]],
    verifier: dict[str, object],
    final: dict[str, object],
) -> None:
    import matplotlib.pyplot as plt

    root.mkdir(parents=True, exist_ok=True)
    labels = [str(row["label"]) for row in effect_rows]
    means = np.asarray([float(row["mean"]) for row in effect_rows])
    lower = np.asarray([float(row["lower"]) for row in effect_rows])
    upper = np.asarray([float(row["upper"]) for row in effect_rows])
    figure, axis = plt.subplots(figsize=(6.6, 3.2))
    y = np.arange(len(labels))
    axis.errorbar(
        means,
        y,
        xerr=[means - lower, upper - means],
        fmt="o",
        color="#24557a",
        capsize=3,
    )
    axis.axvline(0.0, color="black", linestyle=":", linewidth=1)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Claim-level evidence-F1 difference vs. fixed top-5")
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(root / f"corrected_selector_effects.{suffix}", bbox_inches="tight")
    plt.close(figure)


    totals: dict[int, int] = {}
    probability_sums: dict[int, float] = {}
    observed_sums: dict[int, float] = {}
    for row in reliability_rows:
        index = int(row["bin"])
        count = int(row["count"])
        totals[index] = totals.get(index, 0) + count
        probability_sums[index] = probability_sums.get(index, 0.0) + count * float(
            row["mean_probability"]
        )
        observed_sums[index] = observed_sums.get(index, 0.0) + count * float(
            row["observed_rate"]
        )
    x = np.asarray([probability_sums[index] / totals[index] for index in sorted(totals)])
    y_values = np.asarray([observed_sums[index] / totals[index] for index in sorted(totals)])
    figure, axis = plt.subplots(figsize=(4.2, 4.0))
    axis.plot([0, 1], [0, 1], color="#777777", linestyle="--", label="Ideal")
    axis.plot(x, y_values, marker="o", color="#1b8078", label="Selected map")
    axis.set_xlim(0, max(0.3, float(x.max()) + 0.03))
    axis.set_ylim(0, max(0.3, float(y_values.max()) + 0.03))
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed evidence rate")
    axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(root / f"calibration_reliability.{suffix}", bbox_inches="tight")
    plt.close(figure)

    aggregate = verifier["aggregate_seed_aligned_rc_cefes_minus_fixed"]
    figure, axis = plt.subplots(figsize=(5.0, 2.1))
    mean = float(aggregate["mean"])
    lo = float(aggregate["lower"])
    hi = float(aggregate["upper"])
    axis.errorbar(mean, [0], xerr=[[mean - lo], [hi - mean]], fmt="o", color="#1b8078", capsize=4)
    axis.axvline(0.0, color="black", linestyle=":", linewidth=1)
    axis.axvline(float(verifier["noninferiority_margin"]), color="#c03d4b", linestyle="--", linewidth=1)
    axis.set_yticks([0], ["Expected-F1 vs. fixed"])
    axis.set_xlabel("Shared-verifier macro-F1 difference")
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(root / f"corrected_verifier_effect.{suffix}", bbox_inches="tight")
    plt.close(figure)

    packet = final["retained_packet"]
    methods = {str(row["method"]): row for row in final["method_means"]}
    zero_share = (
        float(packet["eligible_claims_with_zero_reachable_gold"])
        / float(packet["eligible_claims"])
    )
    complete_share = float(packet["complete_gold_claim_coverage"])
    partial_share = 1.0 - zero_share - complete_share
    palette = {"blue": "#24557A", "teal": "#2A7F78", "orange": "#C56A2D"}

    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.2))
    figure.subplots_adjust(wspace=0.48)
    left = 0.0
    for value, label, color in (
        (zero_share, "No gold", "#A23B3B"),
        (partial_share, "Partial", palette["orange"]),
        (complete_share, "Complete", palette["teal"]),
    ):
        axes[0].barh([0], [value], left=[left], color=color, height=0.5, label=label)
        if value > 0.08:
            axes[0].text(
                left + value / 2,
                0,
                f"{100 * value:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
            )
        left += value
    axes[0].set_xlim(0, 1)
    axes[0].set_yticks([])
    axes[0].set_xlabel("Share of 10,970 eligible claims")
    axes[0].set_title("A  Retrieval boundary", loc="left", fontweight="bold")
    axes[0].legend(
        frameon=False,
        fontsize=7,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.42),
    )

    policy_order = ("fixed", "cefes", "threshold")
    policy_labels = ("Fixed top-5", "Expected-F1", "Threshold")
    policy_values = [float(methods[key]["f1"]) for key in policy_order]
    axes[1].barh(
        policy_labels,
        policy_values,
        color=["#8A99A8", palette["blue"], palette["teal"]],
    )
    axes[1].axvline(
        float(packet["packet_oracle_f1"]),
        color=palette["orange"],
        linestyle="--",
        linewidth=1.4,
    )
    for index, value in enumerate(policy_values):
        axes[1].text(value + 0.006, index, f"{value:.3f}", va="center", fontsize=8)
    axes[1].text(
        float(packet["packet_oracle_f1"]) - 0.004,
        2.55,
        "packet oracle 0.356",
        ha="right",
        va="top",
        fontsize=7,
        color=palette["orange"],
    )
    axes[1].set_xlim(0, 0.40)
    axes[1].set_xlabel("Evidence-set F1")
    axes[1].set_title("B  Policy comparison", loc="left", fontweight="bold")
    axes[1].invert_yaxis()

    point_order = ("cefes", "fixed", "threshold", "count")
    point_labels = {
        "cefes": "Expected-F1",
        "fixed": "Fixed",
        "threshold": "Threshold",
        "count": "Count",
    }
    for key, marker, color in zip(
        point_order,
        ("o", "s", "D", "X"),
        (palette["blue"], "#8A99A8", palette["teal"], "#A23B3B"),
    ):
        row = methods[key]
        x_value = float(row["selected_count"])
        y_value = float(row["precision"])
        axes[2].scatter(x_value, y_value, s=45, marker=marker, color=color, zorder=3)
        axes[2].annotate(
            point_labels[key],
            (x_value, y_value),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    axes[2].set_xlim(0, 35)
    axes[2].set_ylim(0, 0.43)
    axes[2].set_xlabel("Items returned per claim")
    axes[2].set_ylabel("Evidence precision")
    axes[2].set_title("C  Review-cost tradeoff", loc="left", fontweight="bold")
    axes[2].grid(True, color="#E5E8EB", linewidth=0.6, zorder=0)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(root / f"study_summary.{suffix}", bbox_inches="tight")
    plt.close(figure)


def _write_robustness_absolute_table(
    research_root: Path,
    generated_root: Path,
    robustness: dict[str, object],
) -> None:
    labels = {
        "baseline": "Baseline",
        "width_10": "Width 10",
        "width_20": "Width 20",
        "width_40": "Width 40",
        "text_only": "Text only",
        "image_only": "Image only",
        "remove_gold_25": "Remove 25\\% gold",
        "remove_gold_50": "Remove 50\\% gold",
        "promote_5_distractors": "Promote 5 distractors",
        "duplicate_top_3": "Duplicate top 3",
        "temperature_0.5": "Temperature 0.5",
        "temperature_2.0": "Temperature 2.0",
    }
    grouped: dict[str, dict[str, list[dict[str, float]]]] = {}
    for row in robustness["seed_condition_method_metrics"]:
        condition = str(row["condition"])
        method = str(row["method"])
        if method not in {"fixed", "threshold", "cefes"}:
            continue
        grouped.setdefault(condition, {}).setdefault(method, []).append(row)
    rows: list[dict[str, object]] = []
    for condition_value in robustness["conditions"]:
        condition = str(condition_value)
        by_method = grouped[condition]
        fixed_f1 = fmean(float(row["f1"]) for row in by_method["fixed"])
        threshold_f1 = fmean(float(row["f1"]) for row in by_method["threshold"])
        expected_f1 = fmean(float(row["f1"]) for row in by_method["cefes"])
        rows.append(
            {
                "condition": condition,
                "label": labels[condition].replace("\\%", "%"),
                "fixed_f1": fixed_f1,
                "threshold_f1": threshold_f1,
                "expected_f1": expected_f1,
                "expected_selected": fmean(
                    float(row["selected_count"]) for row in by_method["cefes"]
                ),
                "difference": expected_f1 - fixed_f1,
            }
        )
    _write_csv(research_root / "robustness_absolute.csv", rows)
    lines = [
        f"{labels[str(row['condition'])]} & {row['fixed_f1']:.4f} & "
        f"{row['threshold_f1']:.4f} & {row['expected_f1']:.4f} & "
        f"{row['expected_selected']:.2f} & {row['difference']:+.4f} \\\\"
        for row in rows
    ]
    (generated_root / "robustness_absolute.tex").write_text(
        """\\begin{table*}[t]
\\centering
\\caption{Absolute evidence F1 and expected-F1 set size under controlled perturbations. Values average the five deterministic partitions; the perturbation analysis is descriptive.}
\\label{tab:robustness-absolute}
\\small
\\begin{tabular}{lrrrrr}
\\toprule
Condition & Fixed F1 & Threshold F1 & Expected-F1 & Expected items & Difference \\\\
\\midrule
"""
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n",
        encoding="utf-8",
    )


def build_calibration_corrected_reporting(
    retrieval_root: Path,
    selector_root: Path,
    verifier_root: Path,
    robustness_root: Path,
    research_root: Path,
    generated_root: Path,
    figure_root: Path,
) -> dict[str, object]:
    selector = json.loads((selector_root / "summary.json").read_text(encoding="utf-8"))
    verifier = json.loads((verifier_root / "summary.json").read_text(encoding="utf-8"))
    retrieval = json.loads((retrieval_root / "retrieval.manifest.json").read_text(encoding="utf-8"))
    robustness = (
        json.loads((robustness_root / "summary.json").read_text(encoding="utf-8"))
        if (robustness_root / "summary.json").is_file()
        else None
    )
    method_rows = [
        {
            "method": method,
            "label": METHOD_LABELS[method],
            **{
                metric: fmean(float(seed["validation"][method][metric]) for seed in selector["seeds"])
                for metric in ("precision", "recall", "f1", "jaccard", "exact", "selected_count")
            },
        }
        for method in METHODS
    ]
    aggregate = selector["aggregate_seed_aligned_vs_fixed"]
    raw_p = {
        method: bootstrap_two_sided_p(aggregate[method]["samples"])
        for method in aggregate
    }
    adjusted = holm_adjust(raw_p)
    effect_rows = [
        {
            "method": method,
            "label": METHOD_LABELS[method],
            "mean": aggregate[method]["mean"],
            "lower": aggregate[method]["lower"],
            "upper": aggregate[method]["upper"],
            "raw_p": raw_p[method],
            "holm_p": adjusted[method],
        }
        for method in aggregate
    ]
    calibration_rows, reliability_rows = _calibration_reporting(selector, selector_root)
    reachability_rows, reachability_detail = _reachability_reporting(selector_root)
    threshold_minus_cefes = _aligned_effect(
        selector_root, "threshold", "cefes", samples=10_000, seed=20261104
    )
    verifier_rows = [
        {
            "seed": row["seed"],
            "fixed": row["metrics"]["fixed"]["macro_f1"],
            "threshold": row["metrics"]["threshold"]["macro_f1"],
            "expected_f1": row["metrics"]["cefes"]["macro_f1"],
            "difference": row["metrics"]["cefes"]["macro_f1"]
            - row["metrics"]["fixed"]["macro_f1"],
        }
        for row in verifier["seeds"]
    ]
    partition_rows = []
    for seed_row in selector["seeds"]:
        validation = seed_row["validation"]
        partition_rows.append(
            {
                "partition": int(seed_row["seed"]),
                "fixed_f1": float(validation["fixed"]["f1"]),
                "threshold_f1": float(validation["threshold"]["f1"]),
                "expected_f1": float(validation["cefes"]["f1"]),
                "expected_minus_fixed": float(validation["cefes"]["f1"])
                - float(validation["fixed"]["f1"]),
                "expected_precision": float(validation["cefes"]["precision"]),
                "expected_recall": float(validation["cefes"]["recall"]),
                "expected_selected": float(validation["cefes"]["selected_count"]),
            }
        )
    packet = selector["retained_packet_metrics"]
    final = {
        "profile": "calibration_corrected_evidence_selection",
        "status": "complete",
        "population": {
            "released_claims": selector["evaluation_population"]["claims_before_complete_gold_filter"],
            "eligible_claims": selector["evaluation_population"]["claims"],
            "claims_without_released_gold": selector["evaluation_population"]["claims_excluded_without_released_gold"],
            "validation_claims": int(selector["seeds"][0]["partition_claims"]["validation"]),
        },
        "retained_packet": packet,
        "retrieval_matrix": retrieval["retrieval_metrics"],
        "calibration": calibration_rows,
        "validation_reachability": reachability_rows,
        "method_means": method_rows,
        "effects_vs_fixed": effect_rows,
        "threshold_minus_expected_f1": threshold_minus_cefes,
        "verifier_aggregate": {
            key: value
            for key, value in verifier["aggregate_seed_aligned_rc_cefes_minus_fixed"].items()
            if key != "samples"
        },
        "verifier_noninferiority_margin": verifier["noninferiority_margin"],
        "verifier_noninferiority_pass": verifier["aggregate_noninferiority_lower_bound_above_margin"],
        "robustness": robustness,
        "formal_test_annotations_accessed": False,
        "formal_test_unlocked": False,
        "conclusion": (
            "Prior-aware calibration makes expected-F1 selection better than fixed top-5, "
            "but calibrated thresholding remains strongest and verifier non-inferiority is not established."
        ),
    }
    _write_json(research_root / "final_results.json", final)
    _write_csv(research_root / "selector_means.csv", method_rows)
    _write_csv(research_root / "selector_effects.csv", effect_rows)
    _write_csv(research_root / "calibration_summary.csv", calibration_rows)
    _write_csv(research_root / "calibration_reliability.csv", reliability_rows)
    _write_csv(research_root / "verifier_effects.csv", verifier_rows)
    _write_csv(research_root / "partition_selector_results.csv", partition_rows)
    _write_csv(research_root / "validation_reachability.csv", reachability_rows)
    _write_csv(
        research_root / "validation_reachability_claims.csv", reachability_detail
    )
    _write_tables(
        generated_root,
        final,
        method_rows,
        effect_rows,
        calibration_rows,
        partition_rows,
        verifier_rows,
        reachability_rows,
    )
    _write_figures(figure_root, effect_rows, reliability_rows, verifier, final)
    if robustness is not None:
        _write_robustness_reporting(
            robustness_root / "summary.json",
            research_root,
            generated_root,
            figure_root,
        )
        _write_robustness_absolute_table(
            research_root,
            generated_root,
            robustness,
        )
    return final
