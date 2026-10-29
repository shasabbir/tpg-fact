from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


SEEDS = (13, 37, 73)
LABELS = ("supported", "refuted", "nei")
GOLD_SIZE_BINS = ("1--2", "3--5", "6+")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Recompute manuscript diagnostics from frozen MOCHEG development artifacts."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "experiments" / "outputs" / "mocheg" / "reporting",
    )
    parser.add_argument(
        "--figure-dir", type=Path, default=repo_root / "paper" / "figures"
    )
    parser.add_argument(
        "--preview-dir", type=Path, default=repo_root / "tmp" / "reporting_previews"
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evidence_metrics(
    selected_ids: Iterable[str], gold_ids: Iterable[str]
) -> dict[str, float]:
    selected = set(selected_ids)
    gold = set(gold_ids)
    true_positive = len(selected & gold)
    precision = safe_div(true_positive, len(selected))
    recall = safe_div(true_positive, len(gold))
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    union = len(selected | gold)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact": float(selected == gold),
        "jaccard": safe_div(true_positive, union),
        "selected_count": float(len(selected)),
        "gold_count": float(len(gold)),
    }


def average_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: fmean(float(row[key]) for row in rows) for key in keys}


def class_metrics(
    rows: list[dict[str, Any]], method: str
) -> dict[str, float]:
    f1_values: list[float] = []
    output: dict[str, float] = {}
    correct = 0
    for row in rows:
        correct += int(row["gold_label"] == row["methods"][method]["predicted_label"])
    for label in LABELS:
        true_positive = sum(
            row["gold_label"] == label
            and row["methods"][method]["predicted_label"] == label
            for row in rows
        )
        false_positive = sum(
            row["gold_label"] != label
            and row["methods"][method]["predicted_label"] == label
            for row in rows
        )
        false_negative = sum(
            row["gold_label"] == label
            and row["methods"][method]["predicted_label"] != label
            for row in rows
        )
        precision = safe_div(true_positive, true_positive + false_positive)
        recall = safe_div(true_positive, true_positive + false_negative)
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        output[f"{label}_f1"] = f1
        f1_values.append(f1)
    output["macro_f1"] = fmean(f1_values)
    output["accuracy"] = safe_div(correct, len(rows))
    return output


def gold_size_bin(count: int) -> str:
    if count <= 2:
        return "1--2"
    if count <= 5:
        return "3--5"
    return "6+"


def gold_modality(ids: Iterable[str]) -> str:
    modalities = {
        "image" if evidence_id.startswith("image:") else "text"
        for evidence_id in ids
    }
    if len(modalities) > 1:
        return "mixed"
    if modalities:
        return next(iter(modalities))
    return "none"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def save_figure(fig: plt.Figure, pdf_path: Path, preview_dir: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path)
    fig.savefig(preview_dir / f"{pdf_path.stem}.png", dpi=220)
    plt.close(fig)


def plot_retrieval(
    retrieval: dict[str, dict[str, float]], figure_dir: Path, preview_dir: Path
) -> None:
    cutoffs = (5, 10, 50)
    x = np.arange(len(cutoffs))
    fig, ax = plt.subplots(figsize=(3.35, 2.05))
    styles = {
        "text": {"color": "#356AA0", "marker": "o", "linestyle": "-"},
        "image": {"color": "#D17A22", "marker": "s", "linestyle": "--"},
    }
    for modality in ("text", "image"):
        values = [retrieval[modality][f"recall_at_{cutoff}"] for cutoff in cutoffs]
        ax.plot(x, values, label=modality.capitalize(), **styles[modality])
        for x_value, value in zip(x, values, strict=True):
            ax.annotate(
                f"{value:.3f}",
                (x_value, value),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
    ax.set_xticks(x, [str(cutoff) for cutoff in cutoffs])
    ax.set_xlabel("Retrieval cutoff $k$")
    ax.set_ylabel("Complete-gold Recall@$k$")
    ax.set_ylim(0.0, 0.42)
    ax.grid(axis="y", color="#D7D7D7", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    save_figure(fig, figure_dir / "full_corpus_retrieval.pdf", preview_dir)


def plot_paired_effects(
    effects: list[dict[str, Any]], figure_dir: Path, preview_dir: Path
) -> None:
    evidence = [row for row in effects if row["endpoint"] == "evidence_f1"]
    verdict = [row for row in effects if row["endpoint"] == "verdict_macro_f1"]
    fig, axes = plt.subplots(1, 2, figsize=(7.10, 2.45), gridspec_kw={"wspace": 0.42})
    color_map = {"fixed": "#356AA0", "count": "#D17A22"}

    evidence_labels = []
    for index, row in enumerate(evidence):
        y_value = len(evidence) - 1 - index
        color = color_map[row["comparator"]]
        axes[0].errorbar(
            row["mean"],
            y_value,
            xerr=[[row["mean"] - row["lower"]], [row["upper"] - row["mean"]]],
            fmt="o" if row["comparator"] == "fixed" else "s",
            color=color,
            markerfacecolor="white" if row["comparator"] == "count" else color,
            markeredgewidth=1.0,
            capsize=2.5,
        )
        evidence_labels.append(f"{row['comparator'].capitalize()}, seed {row['seed']}")
    axes[0].set_yticks(
        np.arange(len(evidence)), list(reversed(evidence_labels))
    )
    axes[0].set_xlabel(r"CEFES $-$ comparator evidence F1")
    axes[0].set_xlim(-0.075, 0.015)

    verdict_labels = []
    for index, row in enumerate(verdict):
        y_value = len(verdict) - 1 - index
        axes[1].errorbar(
            row["mean"],
            y_value,
            xerr=[[row["mean"] - row["lower"]], [row["upper"] - row["mean"]]],
            fmt="o",
            color="#356AA0",
            capsize=2.5,
        )
        verdict_labels.append(f"Seed {row['seed']}")
    axes[1].set_yticks(np.arange(len(verdict)), list(reversed(verdict_labels)))
    axes[1].set_xlabel(r"CEFES $-$ fixed verdict macro-F1")
    axes[1].set_xlim(-0.035, 0.14)

    for label, ax in zip(("(a) Evidence selection", "(b) Shared verifier"), axes, strict=True):
        ax.axvline(0.0, color="#333333", linestyle=":", linewidth=0.9)
        ax.grid(axis="x", color="#D7D7D7", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.0, 1.04, label, transform=ax.transAxes, ha="left", va="bottom")
    save_figure(fig, figure_dir / "paired_development_effects.pdf", preview_dir)


def plot_set_size_mechanism(
    subgroup_rows: list[dict[str, Any]], figure_dir: Path, preview_dir: Path
) -> None:
    rows = [row for row in subgroup_rows if row["group_type"] == "gold_size"]
    fig, axes = plt.subplots(1, 2, figsize=(7.10, 2.55), gridspec_kw={"wspace": 0.30})
    x = np.arange(len(GOLD_SIZE_BINS))
    count_series = (
        ("Gold evidence", "mean_gold_count", "#333333", "D", "-"),
        ("Fixed", "fixed_mean_selected", "#356AA0", "s", "--"),
        ("Count control", "dacs_mean_selected", "#B98519", "^", "-."),
        ("CEFES", "cefes_mean_selected", "#D17A22", "o", "-"),
    )
    for label, key, color, marker, linestyle in count_series:
        means = [
            fmean(
                float(row[key])
                for row in rows
                if row["group"] == group
            )
            for group in GOLD_SIZE_BINS
        ]
        axes[0].plot(
            x,
            means,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            markerfacecolor="white" if label in {"Fixed", "Count control"} else color,
        )
    axes[0].set_xticks(x, GOLD_SIZE_BINS)
    axes[0].set_xlabel("Gold-set size")
    axes[0].set_ylabel("Mean evidence count")
    axes[0].set_ylim(0, 44)
    axes[0].legend(frameon=False, ncol=2, loc="upper left")

    seed_styles = {
        13: ("#356AA0", "o", "-"),
        37: ("#D17A22", "s", "--"),
        73: ("#6B7C3D", "^", "-."),
    }
    for seed in SEEDS:
        color, marker, linestyle = seed_styles[seed]
        effects = [
            next(
                float(row["cefes_minus_fixed_f1"])
                for row in rows
                if row["seed"] == seed and row["group"] == group
            )
            for group in GOLD_SIZE_BINS
        ]
        axes[1].plot(
            x,
            effects,
            label=f"Seed {seed}",
            color=color,
            marker=marker,
            linestyle=linestyle,
        )
    axes[1].axhline(0.0, color="#333333", linestyle=":", linewidth=0.9)
    axes[1].set_xticks(x, GOLD_SIZE_BINS)
    axes[1].set_xlabel("Gold-set size")
    axes[1].set_ylabel(r"CEFES $-$ fixed evidence F1")
    axes[1].set_ylim(-0.135, 0.025)
    axes[1].legend(frameon=False, loc="lower right")

    for label, ax in zip(("(a) Set-size mismatch", "(b) Descriptive effect"), axes, strict=True):
        ax.grid(axis="y", color="#D7D7D7", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.0, 1.04, label, transform=ax.transAxes, ha="left", va="bottom")
    save_figure(fig, figure_dir / "set_size_mechanism.pdf", preview_dir)


def write_latex_tables(generated_dir: Path, reporting: dict[str, Any]) -> None:
    generated_dir.mkdir(parents=True, exist_ok=True)
    retrieval = reporting["retrieval"]
    retrieval_lines = []
    for modality in ("text", "image"):
        row = retrieval[modality]
        retrieval_lines.append(
            f"{modality.capitalize()} & {row['recall_at_5']:.5f} & "
            f"{row['recall_at_10']:.5f} & {row['recall_at_50']:.5f} & "
            f"{row['map_at_60']:.5f} & {row['ndcg_at_60']:.5f} \\\\"
        )
    retrieval_table = r"""\begin{table}[!t]
\centering
\caption{Complete-gold retrieval quality for the frozen full-corpus artifact. MAP and nDCG use the retained depth of 60.}
\label{tab:full-corpus-retrieval}
\small
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Modality & R@5 & R@10 & R@50 & MAP@60 & nDCG@60 \\
\midrule
""" + "\n".join(retrieval_lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (generated_dir / "retrieval_table.tex").write_text(retrieval_table, encoding="utf-8")

    selector_lines = []
    for row in reporting["selector_method_means"]:
        selector_lines.append(
            f"{row['method']} & {row['precision']:.5f} & {row['recall']:.5f} & "
            f"{row['f1']:.5f} & {row['jaccard']:.5f} & "
            f"{row['mean_selected']:.2f} & {row['exact']:.3f} \\\\"
        )
    selector_table = r"""\begin{table*}[!t]
\centering
\caption{Full-corpus evidence-selection performance averaged over seeds 13, 37 and 73. Metrics are claim-macro; all methods receive the same frozen candidates within a seed.}
\label{tab:full-corpus-method-means}
\small
\setlength{\tabcolsep}{7pt}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
Method & Precision & Recall & F1 & Jaccard & Mean $|\widehat{E}|$ & Exact \\
\midrule
""" + "\n".join(selector_lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    (generated_dir / "selector_mean_table.tex").write_text(selector_table, encoding="utf-8")

    calibration_lines = []
    for row in reporting["calibration_by_seed"]:
        calibration_lines.append(
            f"{row['seed']} & {row['applied_temperature']:.2f} & "
            f"{row['calibrated_brier']:.5f} & {row['calibrated_nll']:.5f} & "
            f"{row['calibrated_ece']:.5f} \\\\"
        )
    calibration_table = r"""\begin{table}[!t]
\centering
\caption{Full-corpus calibration diagnostics. The training-only guard rejected temperature scaling in every seed, so calibrated and raw values are identical.}
\label{tab:full-corpus-calibration}
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}rrrrr@{}}
\toprule
Seed & Temp. & Brier & NLL & ECE \\
\midrule
""" + "\n".join(calibration_lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (generated_dir / "calibration_table.tex").write_text(calibration_table, encoding="utf-8")

    verifier_lines = []
    for row in reporting["verifier_method_means"]:
        verifier_lines.append(
            f"{row['method']} & {row['accuracy']:.5f} & {row['supported_f1']:.5f} & "
            f"{row['refuted_f1']:.5f} & {row['nei_f1']:.5f} & {row['macro_f1']:.5f} \\\\"
        )
    verifier_table = r"""\begin{table*}[!t]
\centering
\caption{Shared-verifier diagnostics averaged over three seeds. One balanced checkpoint per seed is applied to every selector. These are secondary diagnostics, not replacements for evidence quality.}
\label{tab:full-corpus-verifier-means}
\small
\setlength{\tabcolsep}{8pt}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Selector & Accuracy & Supported F1 & Refuted F1 & NEI F1 & Macro-F1 \\
\midrule
""" + "\n".join(verifier_lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    (generated_dir / "verifier_mean_table.tex").write_text(verifier_table, encoding="utf-8")

    selector_detail_lines = []
    for row in reporting["selector_by_seed"]:
        selector_detail_lines.append(
            f"{row['seed']} & {row['method']} & {row['precision']:.5f} & "
            f"{row['recall']:.5f} & {row['f1']:.5f} & {row['jaccard']:.5f} & "
            f"{row['mean_selected']:.2f} \\\\"
        )
    selector_detail_table = r"""\begin{table*}[!t]
\centering
\caption{Per-seed full-corpus selector metrics underlying Table~\ref{tab:full-corpus-method-means}. Exact evidence-set match is zero for every listed row.}
\label{tab:full-corpus-selector-detail}
\small
\setlength{\tabcolsep}{7pt}
\begin{tabular}{@{}rlrrrrr@{}}
\toprule
Seed & Method & Precision & Recall & F1 & Jaccard & Mean $|\widehat{E}|$ \\
\midrule
""" + "\n".join(selector_detail_lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""

    verifier_detail_lines = []
    for row in reporting["verifier_by_seed"]:
        verifier_detail_lines.append(
            f"{row['seed']} & {row['method']} & {row['accuracy']:.5f} & "
            f"{row['supported_f1']:.5f} & {row['refuted_f1']:.5f} & "
            f"{row['nei_f1']:.5f} & {row['macro_f1']:.5f} \\\\"
        )
    verifier_detail_table = r"""\begin{table*}[!t]
\centering
\caption{Per-seed shared-verifier diagnostics underlying Table~\ref{tab:full-corpus-verifier-means}.}
\label{tab:full-corpus-verifier-detail}
\small
\setlength{\tabcolsep}{7pt}
\begin{tabular}{@{}rlrrrrr@{}}
\toprule
Seed & Selector & Accuracy & Supported F1 & Refuted F1 & NEI F1 & Macro-F1 \\
\midrule
""" + "\n".join(verifier_detail_lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""

    subgroup_detail_lines = []
    gold_rows = [
        row for row in reporting["descriptive_subgroups"] if row["group_type"] == "gold_size"
    ]
    for group in GOLD_SIZE_BINS:
        rows = [row for row in gold_rows if row["group"] == group]
        subgroup_detail_lines.append(
            f"{group} & {rows[0]['claims']} & "
            f"{fmean(row['mean_gold_count'] for row in rows):.2f} & "
            f"{fmean(row['cefes_mean_selected'] for row in rows):.2f} & "
            f"{fmean(row['fixed_mean_selected'] for row in rows):.2f} & "
            f"{fmean(row['dacs_mean_selected'] for row in rows):.2f} & "
            f"{fmean(row['cefes_minus_fixed_f1'] for row in rows):+.5f} & "
            f"{fmean(row['cefes_minus_dacs_f1'] for row in rows):+.5f} \\\\"
        )
    subgroup_table = r"""\begin{table*}[!t]
\centering
\caption{Post hoc descriptive analysis by gold-set size, averaged over seeds. Effects were not part of the confirmatory gate and receive no confirmatory interpretation.}
\label{tab:gold-size-diagnostics}
\small
\setlength{\tabcolsep}{6pt}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
Gold size & Claims & Mean gold & CEFES selected & Fixed selected & Count selected & CEFES$-$fixed & CEFES$-$count \\
\midrule
""" + "\n".join(subgroup_detail_lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    (generated_dir / "appendix_diagnostic_tables.tex").write_text(
        selector_detail_table + "\n" + verifier_detail_table + "\n" + subgroup_table,
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    mocheg_root = repo_root / "experiments" / "outputs" / "mocheg"
    selector_root = mocheg_root / "full_corpus_selector_trial_002"
    verifier_root = mocheg_root / "full_corpus_matched_verifier_trial_002"
    dacs_root = mocheg_root / "full_corpus_dacs_trial_002"

    source_paths = [
        mocheg_root / "full_corpus_features.audit.json",
        selector_root / "summary.json",
        selector_root / "audit.json",
        verifier_root / "summary.json",
        verifier_root / "audit.json",
        dacs_root / "summary.json",
        dacs_root / "audit.json",
        mocheg_root / "development_gate.json",
    ]
    for seed in SEEDS:
        source_paths.extend(
            [
                selector_root / f"seed_{seed}_claim_records.jsonl",
                dacs_root / f"seed_{seed}_dacs_records.jsonl",
                verifier_root / f"seed_{seed}_verdict_records.jsonl",
            ]
        )
    missing = [path for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing reporting inputs: {missing}")

    retrieval_audit = load_json(mocheg_root / "full_corpus_features.audit.json")
    selector_summary = load_json(selector_root / "summary.json")
    verifier_summary = load_json(verifier_root / "summary.json")
    dacs_summary = load_json(dacs_root / "summary.json")
    gate = load_json(mocheg_root / "development_gate.json")

    if retrieval_audit["status"] != "pass":
        raise ValueError("Full-corpus retrieval audit is not passing.")
    if gate["formal_test_annotations_accessed"] or gate["formal_test_unlocked"]:
        raise ValueError("Formal-test seal is not intact; reporting build stopped.")

    selector_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    verifier_rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []
    paired_effects: list[dict[str, Any]] = []

    selector_seed_map = {int(row["seed"]): row for row in selector_summary["seeds"]}
    dacs_seed_map = {int(row["seed"]): row for row in dacs_summary["seeds"]}
    verifier_seed_map = {int(row["seed"]): row for row in verifier_summary["seeds"]}

    for seed in SEEDS:
        selector_seed = selector_seed_map[seed]
        validation = selector_seed["validation"]
        for method_key, display_name in (
            ("cefes", "CEFES"),
            ("fixed", "Fixed"),
            ("threshold_0_5", "Threshold 0.5"),
        ):
            metrics = validation[method_key]
            selector_rows.append(
                {
                    "seed": seed,
                    "method": display_name,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "exact": metrics["exact"],
                    "jaccard": metrics["jaccard"],
                    "mean_selected": metrics["mean_selected"],
                }
            )

        dacs_records = load_jsonl(dacs_root / f"seed_{seed}_dacs_records.jsonl")
        dacs_metrics = [
            evidence_metrics(row["selected_evidence_ids"], row["gold_evidence_ids"])
            for row in dacs_records
        ]
        mean_dacs_metrics = average_metric_rows(dacs_metrics)
        stored_dacs_f1 = float(dacs_seed_map[seed]["dacs_evidence_f1"])
        if not math.isclose(mean_dacs_metrics["f1"], stored_dacs_f1, abs_tol=1e-12):
            raise ValueError(f"DACS F1 mismatch for seed {seed}")
        selector_rows.append(
            {
                "seed": seed,
                "method": "Count control",
                "precision": mean_dacs_metrics["precision"],
                "recall": mean_dacs_metrics["recall"],
                "f1": mean_dacs_metrics["f1"],
                "exact": mean_dacs_metrics["exact"],
                "jaccard": mean_dacs_metrics["jaccard"],
                "mean_selected": mean_dacs_metrics["selected_count"],
            }
        )

        calibration = validation["calibration"]
        guard = selector_seed["calibration_guard"]
        calibration_rows.append(
            {
                "seed": seed,
                "temperature_retained": bool(guard["temperature_retained"]),
                "applied_temperature": guard["applied_temperature"],
                "raw_brier": calibration["raw"]["brier"],
                "calibrated_brier": calibration["calibrated"]["brier"],
                "raw_nll": calibration["raw"]["nll"],
                "calibrated_nll": calibration["calibrated"]["nll"],
                "raw_ece": calibration["raw"]["ece"],
                "calibrated_ece": calibration["calibrated"]["ece"],
            }
        )

        selector_effect = validation["cefes_minus_fixed_bootstrap"]
        paired_effects.append(
            {
                "endpoint": "evidence_f1",
                "comparator": "fixed",
                "seed": seed,
                "mean": selector_effect["mean"],
                "lower": selector_effect["lower"],
                "upper": selector_effect["upper"],
                "bootstrap_samples": selector_effect["sample_count"],
            }
        )
        dacs_effect = dacs_seed_map[seed]["cefes_minus_dacs_bootstrap"]
        paired_effects.append(
            {
                "endpoint": "evidence_f1",
                "comparator": "count",
                "seed": seed,
                "mean": dacs_effect["mean"],
                "lower": dacs_effect["lower"],
                "upper": dacs_effect["upper"],
                "bootstrap_samples": dacs_effect["sample_count"],
            }
        )

        claim_records = load_jsonl(selector_root / f"seed_{seed}_claim_records.jsonl")
        dacs_by_claim = {str(row["claim_id"]): row for row in dacs_records}
        grouped: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
        for row in claim_records:
            claim_id = str(row["claim_id"])
            dacs_row = dacs_by_claim[claim_id]
            gold_count = len(row["gold_evidence_ids"])
            values = {
                "gold_count": float(gold_count),
                "cefes_mean_selected": float(row["selected_count"]),
                "fixed_mean_selected": float(validation["fixed"]["mean_selected"]),
                "dacs_mean_selected": float(dacs_row["predicted_count"]),
                "cefes_f1": float(row["f1"]),
                "fixed_f1": float(row["fixed_f1"]),
                "dacs_f1": float(dacs_row["dacs_f1"]),
                "cefes_minus_fixed_f1": float(row["cefes_minus_fixed_f1"]),
                "cefes_minus_dacs_f1": float(dacs_row["cefes_minus_dacs_f1"]),
            }
            grouped[("gold_size", gold_size_bin(gold_count))].append(values)
            grouped[("modality", gold_modality(row["gold_evidence_ids"]))].append(values)
        for (group_type, group), group_values in sorted(grouped.items()):
            means = average_metric_rows(group_values)
            subgroup_rows.append(
                {
                    "seed": seed,
                    "group_type": group_type,
                    "group": group,
                    "claims": len(group_values),
                    "mean_gold_count": means["gold_count"],
                    "cefes_mean_selected": means["cefes_mean_selected"],
                    "fixed_mean_selected": means["fixed_mean_selected"],
                    "dacs_mean_selected": means["dacs_mean_selected"],
                    "cefes_f1": means["cefes_f1"],
                    "fixed_f1": means["fixed_f1"],
                    "dacs_f1": means["dacs_f1"],
                    "cefes_minus_fixed_f1": means["cefes_minus_fixed_f1"],
                    "cefes_minus_dacs_f1": means["cefes_minus_dacs_f1"],
                }
            )

        verdict_records = load_jsonl(verifier_root / f"seed_{seed}_verdict_records.jsonl")
        verifier_seed = verifier_seed_map[seed]
        for method_key, display_name in (
            ("cefes", "CEFES"),
            ("fixed", "Fixed"),
            ("threshold", "Threshold 0.5"),
        ):
            metrics = class_metrics(verdict_records, method_key)
            stored = verifier_seed["validation"][method_key]
            if not math.isclose(metrics["macro_f1"], stored["macro_f1"], abs_tol=1e-12):
                raise ValueError(f"Verifier macro-F1 mismatch for seed {seed}, {method_key}")
            if not math.isclose(metrics["accuracy"], stored["accuracy"], abs_tol=1e-12):
                raise ValueError(f"Verifier accuracy mismatch for seed {seed}, {method_key}")
            verifier_rows.append(
                {"seed": seed, "method": display_name, **metrics}
            )
        verifier_effect = verifier_seed["cefes_minus_fixed_macro_f1_bootstrap"]
        paired_effects.append(
            {
                "endpoint": "verdict_macro_f1",
                "comparator": "fixed",
                "seed": seed,
                "mean": verifier_effect["mean"],
                "lower": verifier_effect["lower"],
                "upper": verifier_effect["upper"],
                "bootstrap_samples": verifier_effect["sample_count"],
            }
        )

    selector_method_means = []
    for method in ("CEFES", "Fixed", "Threshold 0.5", "Count control"):
        method_rows = [row for row in selector_rows if row["method"] == method]
        selector_method_means.append(
            {
                "method": method,
                **{
                    key: fmean(float(row[key]) for row in method_rows)
                    for key in (
                        "precision",
                        "recall",
                        "f1",
                        "exact",
                        "jaccard",
                        "mean_selected",
                    )
                },
            }
        )

    verifier_method_means = []
    for method in ("CEFES", "Fixed", "Threshold 0.5"):
        method_rows = [row for row in verifier_rows if row["method"] == method]
        verifier_method_means.append(
            {
                "method": method,
                **{
                    key: fmean(float(row[key]) for row in method_rows)
                    for key in (
                        "accuracy",
                        "supported_f1",
                        "refuted_f1",
                        "nei_f1",
                        "macro_f1",
                    )
                },
            }
        )

    reporting = {
        "profile": "mocheg_frozen_development_reporting_expansion",
        "status": "pass",
        "scope": "post-freeze recomputation and descriptive reporting only",
        "formal_test_annotations_accessed": False,
        "formal_test_unlocked": False,
        "source_hashes": {
            str(path.relative_to(repo_root)).replace("\\", "/"): sha256(path)
            for path in source_paths
        },
        "retrieval": retrieval_audit["recomputed"]["retrieval_metrics"],
        "selector_by_seed": selector_rows,
        "selector_method_means": selector_method_means,
        "calibration_by_seed": calibration_rows,
        "dacs_by_seed": [
            {
                "seed": int(row["seed"]),
                "count_mae": row["count_mae"],
                "count_exact_match": row["count_exact_match"],
            }
            for row in dacs_summary["seeds"]
        ],
        "verifier_by_seed": verifier_rows,
        "verifier_method_means": verifier_method_means,
        "paired_effects": paired_effects,
        "descriptive_subgroups": subgroup_rows,
        "development_gate": gate,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results_expansion.json").write_text(
        json.dumps(reporting, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "selector_by_seed.csv", selector_rows)
    write_csv(args.output_dir / "selector_method_means.csv", selector_method_means)
    write_csv(args.output_dir / "calibration_by_seed.csv", calibration_rows)
    write_csv(args.output_dir / "verifier_by_seed.csv", verifier_rows)
    write_csv(args.output_dir / "verifier_method_means.csv", verifier_method_means)
    write_csv(args.output_dir / "paired_effects.csv", paired_effects)
    write_csv(args.output_dir / "descriptive_subgroups.csv", subgroup_rows)

    write_latex_tables(repo_root / "paper" / "generated", reporting)

    configure_matplotlib()
    plot_retrieval(reporting["retrieval"], args.figure_dir, args.preview_dir)
    plot_paired_effects(paired_effects, args.figure_dir, args.preview_dir)
    plot_set_size_mechanism(subgroup_rows, args.figure_dir, args.preview_dir)

    print(json.dumps({
        "status": "pass",
        "output": str(args.output_dir / "results_expansion.json"),
        "figures": [
            str(args.figure_dir / "full_corpus_retrieval.pdf"),
            str(args.figure_dir / "paired_development_effects.pdf"),
            str(args.figure_dir / "set_size_mechanism.pdf"),
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
