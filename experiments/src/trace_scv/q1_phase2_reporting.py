from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Iterable, Sequence

import numpy as np

from trace_scv.data.io import sha256_file


SEEDS = (13, 37, 73, 101, 137)
METHODS = ("fixed", "threshold", "count", "cefes", "size_penalized", "rc_cefes")
METHOD_LABELS = {
    "fixed": "Fixed top-5",
    "threshold": "Calibrated threshold",
    "count": "Learned count",
    "cefes": "CEFES",
    "size_penalized": "Size-penalized CEFES",
    "rc_cefes": "RC-CEFES",
}


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda name: (p_values[name], name))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        value = min(1.0, (total - rank) * p_values[name])
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def bootstrap_two_sided_p(samples: Sequence[float]) -> float:
    values = np.asarray(samples, dtype=np.float64)
    nonpositive = (np.count_nonzero(values <= 0.0) + 1) / (len(values) + 1)
    nonnegative = (np.count_nonzero(values >= 0.0) + 1) / (len(values) + 1)
    return float(min(1.0, 2.0 * min(nonpositive, nonnegative)))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _scientific_artifact_comparison(
    original_root: Path,
    clean_root: Path,
) -> dict[str, object]:
    retrieval_names = (
        "rankings.npz",
        "claims.jsonl",
        "candidate_features.npz",
        "candidate_records.jsonl",
        "oracle_records.jsonl",
        "retrieval_metrics.json",
        "full_gold.json",
    )
    selector_names = tuple(
        f"seed_{seed}_{suffix}"
        for seed in SEEDS
        for suffix in ("candidate_records.jsonl", "claim_records.jsonl")
    )
    verifier_names = tuple(
        f"seed_{seed}_{suffix}"
        for seed in SEEDS
        for suffix in ("verifier.json", "verdict_records.jsonl")
    )
    groups = {
        "retrieval": retrieval_names,
        "selector": selector_names,
        "verifier": verifier_names,
    }
    rows: list[dict[str, object]] = []
    for group, names in groups.items():
        for name in names:
            left = original_root / group / name
            right = clean_root / group / name
            left_hash = sha256_file(left)
            right_hash = sha256_file(right)
            rows.append(
                {
                    "group": group,
                    "artifact": name,
                    "original_sha256": left_hash,
                    "clean_sha256": right_hash,
                    "exact_match": left_hash == right_hash,
                }
            )
    left_prejoin = json.loads(
        (original_root / "retrieval" / "rankings.prejoin.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    right_prejoin = json.loads(
        (clean_root / "retrieval" / "rankings.prejoin.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for value in (left_prejoin, right_prejoin):
        value["rankings"]["path"] = Path(value["rankings"]["path"]).name
        value["claims_sidecar"]["path"] = Path(value["claims_sidecar"]["path"]).name
    audit_paths = (
        original_root / "retrieval" / "retrieval.audit.json",
        original_root / "selector" / "audit.json",
        original_root / "verifier" / "audit.json",
        clean_root / "retrieval" / "retrieval.audit.json",
        clean_root / "selector" / "audit.json",
        clean_root / "verifier" / "audit.json",
    )
    audits_pass = all(
        json.loads(path.read_text(encoding="utf-8"))["status"] == "pass"
        for path in audit_paths
    )
    return {
        "profile": "q1_phase2_clean_reproduction",
        "status": "pass"
        if all(bool(row["exact_match"]) for row in rows)
        and left_prejoin == right_prejoin
        and audits_pass
        else "failed",
        "exact_artifact_count": len(rows),
        "artifacts": rows,
        "prejoin_manifest_semantic_match": left_prejoin == right_prejoin,
        "expected_prejoin_difference": "output path strings only",
        "all_independent_audits_pass": audits_pass,
        "formal_test_annotations_accessed": False,
    }


def _latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def _write_latex_tables(
    generated_root: Path,
    retrieval_rows: Sequence[dict[str, object]],
    method_rows: Sequence[dict[str, object]],
    effect_rows: Sequence[dict[str, object]],
    verifier_rows: Sequence[dict[str, object]],
    gate_rows: Sequence[dict[str, object]],
) -> None:
    generated_root.mkdir(parents=True, exist_ok=True)
    retrieval_lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Modality & Variant & Recall@50 & MAP@100 \\",
        r"\midrule",
    ]
    retrieval_lines.extend(
        f"{row['modality']} & {_latex_escape(str(row['variant_label']))} & "
        f"{float(row['recall_at_50']):.4f} & {float(row['map_at_100']):.4f} \\\\"
        for row in retrieval_rows
    )
    retrieval_lines.extend((r"\bottomrule", r"\end{tabular}"))
    (generated_root / "retrieval_matrix.tex").write_text(
        "\n".join(retrieval_lines) + "\n", encoding="utf-8"
    )

    method_lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Precision & Recall & F1 & Selected \\",
        r"\midrule",
    ]
    method_lines.extend(
        f"{row['label']} & {float(row['precision']):.4f} & {float(row['recall']):.4f} & "
        f"{float(row['f1']):.4f} & {float(row['selected_count']):.2f} \\\\"
        for row in method_rows
    )
    method_lines.extend((r"\bottomrule", r"\end{tabular}"))
    (generated_root / "selector_means.tex").write_text(
        "\n".join(method_lines) + "\n", encoding="utf-8"
    )

    effect_lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method vs. fixed & Mean & 95\% lower & 95\% upper & Holm $p$ \\",
        r"\midrule",
    ]
    effect_lines.extend(
        f"{_latex_escape(str(row['label']))} & {float(row['mean']):+.4f} & {float(row['lower']):+.4f} & "
        f"{float(row['upper']):+.4f} & {float(row['holm_p']):.4f} \\\\"
        for row in effect_rows
    )
    effect_lines.extend((r"\bottomrule", r"\end{tabular}"))
    (generated_root / "selector_effects.tex").write_text(
        "\n".join(effect_lines) + "\n", encoding="utf-8"
    )

    verifier_lines = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"Seed & Fixed & RC-CEFES & Difference & 95\% interval \\",
        r"\midrule",
    ]
    verifier_lines.extend(
        f"{int(row['seed'])} & {float(row['fixed_macro_f1']):.4f} & "
        f"{float(row['rc_macro_f1']):.4f} & {float(row['difference']):+.4f} & "
        f"[{float(row['lower']):+.4f}, {float(row['upper']):+.4f}] \\\\"
        for row in verifier_rows
    )
    verifier_lines.extend((r"\bottomrule", r"\end{tabular}"))
    (generated_root / "verifier_effects.tex").write_text(
        "\n".join(verifier_lines) + "\n", encoding="utf-8"
    )

    gate_lines = [
        r"\begin{tabular}{p{0.76\columnwidth}c}",
        r"\toprule",
        r"Condition & Status \\",
        r"\midrule",
    ]
    gate_lines.extend(
        f"{row['label']} & {'Pass' if row['pass'] else r'\textbf{Fail}'} \\\\"
        for row in gate_rows
    )
    gate_lines.extend((r"\bottomrule", r"\end{tabular}"))
    (generated_root / "final_gate.tex").write_text(
        "\n".join(gate_lines) + "\n", encoding="utf-8"
    )


def _write_figures(
    figure_root: Path,
    retrieval_rows: Sequence[dict[str, object]],
    effect_rows: Sequence[dict[str, object]],
    verifier_aggregate: dict[str, object],
) -> None:
    import matplotlib.pyplot as plt

    figure_root.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150})

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), constrained_layout=True)
    for axis, modality, baseline in zip(
        axes, ("Text", "Image"), (0.3616018276517159, 0.21384297520661158), strict=True
    ):
        rows = [row for row in retrieval_rows if row["modality"] == modality]
        labels = [str(row["short_label"]) for row in rows]
        values = [float(row["recall_at_50"]) for row in rows]
        axis.bar(range(len(rows)), values, color="#3E6D9C")
        axis.axhline(baseline, color="#A23B72", linestyle="--", linewidth=1.2, label="Frozen baseline")
        axis.set_xticks(range(len(rows)), labels, rotation=25, ha="right")
        axis.set_ylim(0.0, max(0.42, max(values) + 0.05))
        axis.set_ylabel("Complete-gold Recall@50")
        axis.set_title(f"{modality} retrieval")
        axis.legend(frameon=False, fontsize=8)
    for suffix in ("pdf", "png"):
        fig.savefig(figure_root / f"retrieval_matrix.{suffix}", bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.6, 3.3), constrained_layout=True)
    y = np.arange(len(effect_rows))
    means = np.asarray([float(row["mean"]) for row in effect_rows])
    lower = np.asarray([float(row["lower"]) for row in effect_rows])
    upper = np.asarray([float(row["upper"]) for row in effect_rows])
    axis.errorbar(
        means,
        y,
        xerr=np.vstack((means - lower, upper - means)),
        fmt="o",
        color="#234E70",
        capsize=3,
    )
    axis.axvline(0.0, color="black", linestyle=":", linewidth=1)
    axis.set_yticks(y, [str(row["label"]) for row in effect_rows])
    axis.invert_yaxis()
    axis.set_xlabel("Claim-level evidence-F1 difference vs. fixed")
    for suffix in ("pdf", "png"):
        fig.savefig(figure_root / f"selector_effects.{suffix}", bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(4.6, 2.7), constrained_layout=True)
    mean = float(verifier_aggregate["mean"])
    lower_value = float(verifier_aggregate["lower"])
    upper_value = float(verifier_aggregate["upper"])
    axis.errorbar(
        [mean], [0], xerr=[[mean - lower_value], [upper_value - mean]],
        fmt="o", color="#2A9D8F", capsize=4,
    )
    axis.axvline(0.0, color="black", linestyle=":", linewidth=1)
    axis.axvline(-0.01, color="#A23B72", linestyle="--", linewidth=1.2)
    axis.set_yticks([0], ["RC-CEFES vs. fixed"])
    axis.set_xlabel("Verdict macro-F1 difference")
    for suffix in ("pdf", "png"):
        fig.savefig(figure_root / f"verifier_noninferiority.{suffix}", bbox_inches="tight")
    plt.close(fig)


def _write_robustness_reporting(
    robustness_path: Path,
    research_root: Path,
    generated_root: Path,
    figure_root: Path,
) -> None:
    import matplotlib.pyplot as plt

    robustness = json.loads(robustness_path.read_text(encoding="utf-8"))
    labels = {
        "baseline": "Baseline",
        "width_10": "Width 10",
        "width_20": "Width 20",
        "width_40": "Width 40",
        "text_only": "Text only",
        "image_only": "Image only",
        "remove_gold_25": "Remove 25% gold",
        "remove_gold_50": "Remove 50% gold",
        "promote_5_distractors": "Promote 5 distractors",
        "duplicate_top_3": "Duplicate top 3",
        "temperature_0.5": "Temperature 0.5",
        "temperature_2.0": "Temperature 2.0",
    }
    rows = [
        {"condition": condition, "label": labels[condition], **values}
        for condition, values in robustness["rc_cefes_minus_fixed_seed_aligned"].items()
    ]
    _write_csv(research_root / "robustness_effects.csv", rows)
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Condition & Mean & 95\% lower & 95\% upper \\",
        r"\midrule",
    ]
    lines.extend(
        f"{_latex_escape(str(row['label']))} & {float(row['mean']):+.4f} & {float(row['lower']):+.4f} & "
        f"{float(row['upper']):+.4f} \\\\"
        for row in rows
    )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    (generated_root / "robustness_effects.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    fig, axis = plt.subplots(figsize=(6.3, 4.8), constrained_layout=True)
    y = np.arange(len(rows))
    means = np.asarray([float(row["mean"]) for row in rows])
    lower = np.asarray([float(row["lower"]) for row in rows])
    upper = np.asarray([float(row["upper"]) for row in rows])
    colors = ["#C44536" if row["condition"] == "promote_5_distractors" else "#264653" for row in rows]
    for index in range(len(rows)):
        axis.errorbar(
            means[index], y[index],
            xerr=[[means[index] - lower[index]], [upper[index] - means[index]]],
            fmt="o", color=colors[index], capsize=3,
        )
    axis.axvline(0.0, color="black", linestyle=":", linewidth=1)
    axis.set_yticks(y, [str(row["label"]) for row in rows])
    axis.invert_yaxis()
    axis.set_xlabel("Expected-F1 minus fixed top-5 evidence F1")
    for suffix in ("pdf", "png"):
        fig.savefig(figure_root / f"robustness_effects.{suffix}", bbox_inches="tight")
    plt.close(fig)


def build_q1_phase2_reporting(
    original_root: Path,
    clean_root: Path,
    research_root: Path,
    generated_root: Path,
    figure_root: Path,
) -> dict[str, object]:
    retrieval = json.loads((original_root / "retrieval" / "retrieval.manifest.json").read_text(encoding="utf-8"))
    selector = json.loads((original_root / "selector" / "summary.json").read_text(encoding="utf-8"))
    verifier = json.loads((original_root / "verifier" / "summary.json").read_text(encoding="utf-8"))
    reproduction = _scientific_artifact_comparison(original_root, clean_root)

    retrieval_labels = {
        "dense_sentence": ("Dense sentence", "Dense"),
        "bm25_sentence": ("BM25 sentence", "BM25"),
        "dense_bm25_rrf": ("Dense--BM25 RRF", "RRF"),
        "dense_bm25_rrf_cross_encoder": ("RRF + cross-encoder", "RRF+CE"),
        "siglip": ("SigLIP2", "SigLIP2"),
        "filename_dense": ("Filename BGE-M3", "Filename"),
        "siglip_filename_rrf": ("SigLIP2--filename RRF", "RRF"),
    }
    retrieval_rows: list[dict[str, object]] = []
    for modality_key, modality_label in (("text", "Text"), ("image", "Image")):
        for variant, metrics in retrieval["retrieval_metrics"][modality_key].items():
            label, short = retrieval_labels[variant]
            retrieval_rows.append(
                {
                    "modality": modality_label,
                    "variant": variant,
                    "variant_label": label,
                    "short_label": short,
                    "recall_at_5": metrics["recall_at_5"],
                    "recall_at_10": metrics["recall_at_10"],
                    "recall_at_20": metrics["recall_at_20"],
                    "recall_at_50": metrics["recall_at_50"],
                    "map_at_100": metrics["map_at_100"],
                    "ndcg_at_100": metrics["ndcg_at_100"],
                }
            )

    method_rows: list[dict[str, object]] = []
    for method in METHODS:
        method_rows.append(
            {
                "method": method,
                "label": METHOD_LABELS[method],
                **{
                    metric: fmean(float(seed["validation"][method][metric]) for seed in selector["seeds"])
                    for metric in ("precision", "recall", "f1", "jaccard", "selected_count")
                },
            }
        )

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

    verifier_rows = [
        {
            "seed": seed["seed"],
            "fixed_macro_f1": seed["metrics"]["fixed"]["macro_f1"],
            "rc_macro_f1": seed["metrics"]["rc_cefes"]["macro_f1"],
            "difference": seed["rc_cefes_minus_fixed_bootstrap"]["mean"],
            "lower": seed["rc_cefes_minus_fixed_bootstrap"]["lower"],
            "upper": seed["rc_cefes_minus_fixed_bootstrap"]["upper"],
        }
        for seed in verifier["seeds"]
    ]
    verifier_aggregate = verifier["aggregate_seed_aligned_rc_cefes_minus_fixed"]
    verifier_aggregate_summary = {
        key: value for key, value in verifier_aggregate.items() if key != "samples"
    }

    gate_rows = [
        {"key": "retrieval_text_gain", "label": "Text Recall@50 gain at least +0.10", "pass": selector["retrieval_gate"]["text_gain_at_least_0_10"]},
        {"key": "retrieval_image_gain", "label": "Image Recall@50 gain at least +0.10", "pass": selector["retrieval_gate"]["image_gain_at_least_0_10"]},
        {"key": "scorer_map", "label": "Selected scorer improves MAP in every seed", "pass": selector["scorer_gate"]["positive_map_effect_each_seed"]},
        {"key": "calibration", "label": "Calibrated Brier score is non-worse in every seed", "pass": selector["scorer_gate"]["calibrated_brier_nonworse_each_seed"]},
        {"key": "selector_mean", "label": "Mean RC-CEFES evidence-F1 gain at least +0.015", "pass": selector["superiority_gate"]["mean_gain_at_least_0_015"]},
        {"key": "selector_each_seed", "label": "RC-CEFES evidence effect positive in every seed", "pass": selector["superiority_gate"]["positive_each_seed"]},
        {"key": "selector_interval", "label": "Seed-aligned evidence interval lower bound above zero", "pass": selector["superiority_gate"]["aggregate_interval_lower_positive"]},
        {"key": "set_size", "label": "Mean selected/gold evidence ratio no more than 2", "pass": selector["superiority_gate"]["mean_selected_to_gold_ratio_no_more_than_2"]},
        {"key": "verifier", "label": "Verifier interval lower bound above -0.01", "pass": verifier["aggregate_noninferiority_lower_bound_above_margin"]},
        {"key": "reproduction", "label": "Clean end-to-end reproduction", "pass": reproduction["status"] == "pass"},
    ]
    final_gate_pass = all(bool(row["pass"]) for row in gate_rows)
    final_results = {
        "profile": "q1_phase2_final_results",
        "status": "complete",
        "retrieval": {
            "text_recall_at_50": selector["retrieval_gate"]["text_recall_at_50"],
            "text_gain": selector["retrieval_gate"]["text_gain"],
            "image_recall_at_50": selector["retrieval_gate"]["image_recall_at_50"],
            "image_gain": selector["retrieval_gate"]["image_gain"],
            "packet_oracle_f1": retrieval["packet_ceiling"]["mean_oracle_f1"],
            "complete_gold_claim_coverage": retrieval["packet_ceiling"]["complete_gold_claim_coverage"],
        },
        "evaluation_population": selector["evaluation_population"],
        "method_means": method_rows,
        "selector_effects_vs_fixed": effect_rows,
        "mean_rc_cefes_minus_fixed_f1": selector["mean_rc_cefes_minus_fixed_f1"],
        "mean_selected_to_gold_ratio": selector["mean_selected_to_gold_ratio"],
        "verifier_aggregate": verifier_aggregate_summary,
        "verifier_noninferiority_margin": verifier["noninferiority_margin"],
        "gate": gate_rows,
        "final_gate_pass": final_gate_pass,
        "formal_test_unlocked": False,
        "formal_test_annotations_accessed": False,
        "conclusion": (
            "The preregistered gate fails. Image retrieval and evidence scoring improve, "
            "but the joint retrieval, selector-superiority, and verifier non-inferiority "
            "conditions are not all satisfied."
        ),
    }
    _write_json(research_root / "reproduction_manifest.json", reproduction)
    _write_json(research_root / "final_results.json", final_results)
    _write_csv(research_root / "retrieval_matrix.csv", retrieval_rows)
    _write_csv(research_root / "selector_means.csv", method_rows)
    _write_csv(research_root / "selector_effects.csv", effect_rows)
    _write_csv(research_root / "verifier_effects.csv", verifier_rows)
    _write_latex_tables(
        generated_root, retrieval_rows, method_rows, effect_rows, verifier_rows, gate_rows
    )
    _write_figures(figure_root, retrieval_rows, effect_rows, verifier_aggregate)
    robustness_path = original_root / "robustness" / "summary.json"
    if robustness_path.is_file():
        _write_robustness_reporting(
            robustness_path, research_root, generated_root, figure_root
        )
    return final_results
