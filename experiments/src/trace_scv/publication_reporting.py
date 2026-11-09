from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


METHOD_ORDER = (
    ("fixed", "Fixed top-5"),
    ("threshold", "Calibrated threshold"),
    ("expected_f1", "Expected-F1"),
    ("uncalibrated_expected_f1", "Uncalibrated expected-F1"),
    ("completeness_switch", "Completeness-aware switch"),
    ("count", "Learned count"),
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _interval_without_samples(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "samples"}


def _mean_metrics(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(rows)
    keys = ("precision", "recall", "f1", "selected_count")
    return {key: mean(float(row[key]) for row in rows) for key in keys}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _latex_escape(value: str) -> str:
    return value.replace("%", r"\%").replace("_", r"\_")


def _build_selector_means(
    selector: dict[str, Any], completeness: dict[str, Any]
) -> dict[str, dict[str, float]]:
    base_aliases = {
        "fixed": "fixed",
        "threshold": "threshold",
        "expected_f1": "cefes",
        "count": "count",
    }
    result = {
        target: _mean_metrics(seed["validation"][source] for seed in selector["seeds"])
        for target, source in base_aliases.items()
    }
    for method in ("uncalibrated_expected_f1", "completeness_switch"):
        result[method] = _mean_metrics(
            seed["validation"][method] for seed in completeness["seeds"]
        )
    return result


def _write_latex_tables(
    generated_root: Path,
    retrieval_rows: list[dict[str, Any]],
    selector_rows: list[dict[str, Any]],
    verifier_rows: list[dict[str, Any]],
    verifier_effect: dict[str, Any],
) -> None:
    generated_root.mkdir(parents=True, exist_ok=True)
    retrieval_body = "\n".join(
        f"{_latex_escape(row['system'])} & {row['text_recall_at_40']:.4f} & "
        f"{row['image_recall_at_20']:.4f} & {row['packet_oracle_f1']:.4f} & "
        f"{100.0 * row['complete_gold_coverage']:.2f} & "
        f"{int(row['zero_reachable_claims']):,} \\\\" for row in retrieval_rows
    )
    (generated_root / "publication_retrieval.tex").write_text(
        """\\begin{table*}[t]
\\centering
\\caption{Retrieval repair at the retained packet of 40 text and 20 image items. Coverage is the percentage of eligible claims whose complete released-gold set is present.}
\\label{tab:publication-retrieval}
\\small
\\begin{tabular}{lrrrrr}
\\toprule
Packet & Text R@40 & Image R@20 & Oracle F1 & Complete (\\%) & Zero reachable \\\\
\\midrule
"""
        + retrieval_body
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n",
        encoding="utf-8",
    )

    selector_body = "\n".join(
        f"{_latex_escape(row['method'])} & {row['precision']:.4f} & "
        f"{row['recall']:.4f} & {row['f1']:.4f} & "
        f"{row['selected_count']:.2f} \\\\" for row in selector_rows
    )
    (generated_root / "publication_selectors.tex").write_text(
        """\\begin{table}[t]
\\centering
\\caption{Matched selector results on 1,490 locked validation claims, averaged over five deterministic training-role partitions.}
\\label{tab:publication-selectors}
\\small
\\setlength{\\tabcolsep}{3.5pt}
\\begin{tabular}{lrrrr}
\\toprule
Policy & Prec. & Recall & F1 & Items \\\\
\\midrule
"""
        + selector_body
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8",
    )

    verifier_body = "\n".join(
        f"{_latex_escape(row['verifier'])} & {row['threshold_macro_f1']:.4f} & "
        f"{row['switch_macro_f1']:.4f} & {row['difference']:+.4f} \\\\"
        for row in verifier_rows
    )
    (generated_root / "publication_verifier.tex").write_text(
        """\\begin{table}[t]
\\centering
\\caption{Shared-verifier macro-F1 averaged over five seeds. The aggregate MLP switch effect has a seed-aligned paired 95\\% interval shown below the table.}
\\label{tab:publication-verifier}
\\small
\\begin{tabular}{lrrr}
\\toprule
Verifier & Threshold & Switch & $\\Delta$ \\\\
\\midrule
"""
        + verifier_body
        + "\n\\bottomrule\n\\end{tabular}\n"
        + f"\\vspace{{1mm}}\\par\\footnotesize MLP $\\Delta$ 95\\% CI: "
        + f"[{verifier_effect['lower']:.4f}, {verifier_effect['upper']:.4f}]; "
        + "non-inferiority margin $=-0.01$.\n\\end{table}\n",
        encoding="utf-8",
    )


def _write_summary_figure(
    figure_root: Path,
    retrieval_rows: list[dict[str, Any]],
    selector_rows: list[dict[str, Any]],
    verifier_rows: list[dict[str, Any]],
    verifier_effect: dict[str, Any],
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    figure_root.mkdir(parents=True, exist_ok=True)
    colors = {
        "blue": "#24557A",
        "teal": "#2A7F78",
        "orange": "#C56A2D",
        "gray": "#9AA1A8",
        "red": "#A23B3B",
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.15), constrained_layout=True)

    metric_names = ("Text R@40", "Image R@20", "Oracle F1", "Complete")
    metric_keys = (
        "text_recall_at_40",
        "image_recall_at_20",
        "packet_oracle_f1",
        "complete_gold_coverage",
    )
    x = np.arange(len(metric_names))
    width = 0.34
    for offset, row, color in zip(
        (-width / 2, width / 2), retrieval_rows, (colors["gray"], colors["blue"])
    ):
        axes[0].bar(
            x + offset,
            [row[key] for key in metric_keys],
            width,
            label=row["system"],
            color=color,
        )
    axes[0].set_xticks(x, metric_names, rotation=22, ha="right")
    axes[0].set_ylim(0, 0.5)
    axes[0].set_ylabel("Proportion / macro-F1")
    axes[0].set_title("A  Retrieval repair")
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    axes[0].grid(axis="y", alpha=0.2)

    selector_plot = [
        row
        for row in selector_rows
        if row["key"] in {"fixed", "threshold", "expected_f1", "completeness_switch"}
    ]
    selector_colors = [colors["gray"], colors["blue"], colors["orange"], colors["teal"]]
    bars = axes[1].bar(
        np.arange(len(selector_plot)),
        [row["f1"] for row in selector_plot],
        color=selector_colors,
    )
    axes[1].set_xticks(
        np.arange(len(selector_plot)),
        ["Top-5", "Threshold", "Expected-\nF1", "Completeness\nswitch"],
        fontsize=7.5,
    )
    axes[1].set_ylim(0.15, 0.195)
    axes[1].set_ylabel("Evidence macro-F1")
    axes[1].set_title("B  Matched selection")
    axes[1].grid(axis="y", alpha=0.2)
    for bar, row in zip(bars, selector_plot):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            row["f1"] + 0.0007,
            f"{row['f1']:.3f}\n{row['selected_count']:.1f} items",
            ha="center",
            va="bottom",
            fontsize=6.8,
        )

    verifier = verifier_rows[0]
    axes[2].errorbar(
        verifier_effect["mean"],
        0,
        xerr=[
            [verifier_effect["mean"] - verifier_effect["lower"]],
            [verifier_effect["upper"] - verifier_effect["mean"]],
        ],
        fmt="o",
        color=colors["teal"],
        capsize=3,
        label="MLP aggregate",
    )
    axes[2].axvline(0, color="black", linewidth=0.8)
    axes[2].axvline(-0.01, color=colors["red"], linestyle="--", linewidth=1.0)
    axes[2].set_xlim(-0.025, 0.02)
    axes[2].set_yticks([])
    axes[2].set_xlabel("Switch minus threshold macro-F1")
    axes[2].set_title("C  Shared-verifier safeguard")
    axes[2].grid(axis="x", alpha=0.2)
    axes[2].text(-0.0095, 0.17, "NI margin", color=colors["red"], fontsize=7)
    axes[2].text(
        verifier_effect["mean"],
        -0.14,
        f"{verifier_effect['mean']:+.4f}\n"
        f"[{verifier_effect['lower']:.4f}, {verifier_effect['upper']:.4f}]",
        ha="center",
        va="top",
        fontsize=7.2,
    )
    axes[2].set_ylim(-0.35, 0.35)
    for suffix in ("pdf", "png"):
        figure.savefig(
            figure_root / f"publication_confirmation_summary.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def build_publication_reporting(
    retrieval_root: Path,
    selector_root: Path,
    completeness_root: Path,
    verifier_root: Path,
    research_root: Path,
    generated_root: Path,
    figure_root: Path,
) -> dict[str, Any]:
    source_paths = {
        "retrieval": retrieval_root / "retrieval_results.json",
        "retrieval_audit": retrieval_root / "retrieval.audit.json",
        "selector": selector_root / "summary.json",
        "selector_audit": selector_root / "selector.audit.json",
        "completeness": completeness_root / "summary.json",
        "completeness_audit": completeness_root / "completeness.audit.json",
        "verifier": verifier_root / "summary.json",
        "verifier_audit": verifier_root / "verifier.audit.json",
    }
    inputs = {key: _load_json(path) for key, path in source_paths.items()}
    for key in ("retrieval_audit", "selector_audit", "completeness_audit", "verifier_audit"):
        if inputs[key]["status"] != "pass":
            raise ValueError(f"required audit did not pass: {key}")
    for key in ("retrieval", "selector", "completeness", "verifier"):
        if inputs[key].get("formal_test_annotations_accessed") is not False:
            raise ValueError(f"formal-test access flag is not false: {key}")

    retrieval = inputs["retrieval"]
    selector = inputs["selector"]
    completeness = inputs["completeness"]
    verifier = inputs["verifier"]

    validation = retrieval["population_results"]["validation"]
    all_eligible = retrieval["population_results"]["all_eligible"]
    retrieval_rows = []
    for key, label in (
        ("baseline", "Frozen packet"),
        ("selected_weighted_fusion", "Weighted fusion"),
    ):
        values = validation[key]
        retrieval_rows.append(
            {
                "key": key,
                "system": label,
                "text_recall_at_40": values["text_recall_at_40"],
                "image_recall_at_20": values["image_recall_at_20"],
                "packet_oracle_f1": values["packet_oracle_f1"],
                "complete_gold_coverage": values["complete_gold_claim_coverage"],
                "zero_reachable_claims": int(
                    values["eligible_claims_with_zero_reachable_gold"]
                ),
            }
        )

    selector_means = _build_selector_means(selector, completeness)
    selector_rows = [
        {"key": key, "method": label, **selector_means[key]}
        for key, label in METHOD_ORDER
    ]

    verifier_means = {
        key: mean(float(seed["metrics"][key]["macro_f1"]) for seed in verifier["seeds"])
        for key in (
            "mlp_threshold",
            "mlp_switch",
            "linear_threshold",
            "linear_switch",
        )
    }
    verifier_rows = [
        {
            "verifier": "Two-layer MLP",
            "threshold_macro_f1": verifier_means["mlp_threshold"],
            "switch_macro_f1": verifier_means["mlp_switch"],
            "difference": verifier_means["mlp_switch"] - verifier_means["mlp_threshold"],
        },
        {
            "verifier": "Linear control",
            "threshold_macro_f1": verifier_means["linear_threshold"],
            "switch_macro_f1": verifier_means["linear_switch"],
            "difference": verifier_means["linear_switch"]
            - verifier_means["linear_threshold"],
        },
    ]
    verifier_effect = _interval_without_samples(
        verifier["aggregate_seed_aligned_switch_minus_threshold"]
    )

    result = {
        "profile": "publication_confirmation_reporting_v1",
        "status": "complete",
        "source_sha256": {key: _sha256(path) for key, path in source_paths.items()},
        "population": {
            "retrieval_eligible_claims": int(all_eligible["selected_weighted_fusion"]["eligible_claims"]),
            "locked_validation_claims": int(validation["selected_weighted_fusion"]["eligible_claims"]),
            "training_role_partitions": len(selector["seeds"]),
        },
        "retrieval": {
            "selected_text_rrf_weight": retrieval["selected_text_rrf_weight"],
            "selected_image_siglip_weight": retrieval["selected_image_siglip_weight"],
            "validation": validation,
            "all_eligible": all_eligible,
            "gate": retrieval["gate"],
        },
        "selectors": {
            "means": selector_means,
            "threshold_vs_fixed": _interval_without_samples(
                selector["aggregate_seed_aligned_vs_fixed"]["threshold"]
            ),
            "expected_f1_vs_fixed": _interval_without_samples(
                selector["aggregate_seed_aligned_vs_fixed"]["cefes"]
            ),
            "switch_vs_threshold": _interval_without_samples(
                completeness["aggregate_switch_vs_threshold"]
            ),
            "superiority_gate": completeness["superiority_gate"],
        },
        "verifier": {
            "means": verifier_means,
            "switch_vs_threshold": verifier_effect,
            "noninferiority_margin": verifier["noninferiority_margin"],
            "aggregate_noninferiority_pass": verifier["aggregate_noninferiority_pass"],
            "each_seed_noninferiority_pass": verifier["each_seed_noninferiority_pass"],
        },
        "external_confirmation": {
            "performed": False,
            "reason": "selector superiority and per-seed verifier safeguards failed; formal test not authorized",
        },
        "formal_test_annotations_accessed": False,
    }

    research_root.mkdir(parents=True, exist_ok=True)
    (research_root / "final_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(research_root / "retrieval_results.csv", retrieval_rows)
    _write_csv(research_root / "selector_results.csv", selector_rows)
    _write_csv(research_root / "verifier_results.csv", verifier_rows)
    _write_latex_tables(
        generated_root,
        retrieval_rows,
        selector_rows,
        verifier_rows,
        verifier_effect,
    )
    _write_summary_figure(
        figure_root,
        retrieval_rows,
        selector_rows,
        verifier_rows,
        verifier_effect,
    )
    return result

