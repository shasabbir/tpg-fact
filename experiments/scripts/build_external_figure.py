from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "blue": "#24557A",
    "teal": "#2A7F78",
    "orange": "#C56A2D",
    "red": "#A23B3B",
    "green": "#26734D",
    "gray": "#66717E",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the external-confirmation summary figure.")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/external_confirmation/averimatec/reporting/external_results.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../paper/figures/external_confirmation_summary.pdf"),
    )
    args = parser.parse_args()
    result = json.loads(args.results.read_text(encoding="utf-8"))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.25), constrained_layout=True)

    # A: reachability before selection.
    labels = ("R@10", "R@20", "R@50", "R@100", "Pool")
    keys = ("recall_at_10", "recall_at_20", "recall_at_50", "recall_at_100", "pool_recall")
    train = result["reachability"]["train"]["metrics"]
    validation = result["reachability"]["validation"]["metrics"]
    x = np.arange(len(labels))
    width = 0.36
    axes[0].bar(x - width / 2, [100 * train[key] for key in keys], width, color=COLORS["blue"], label="Train")
    axes[0].bar(
        x + width / 2,
        [100 * validation[key] for key in keys],
        width,
        color=COLORS["teal"],
        label="Development",
    )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Exact-source recall (%)")
    axes[0].set_ylim(0, 30)
    axes[0].grid(axis="y", alpha=0.22, linewidth=0.6)
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].set_title("A  Reachability before selection", loc="left", fontweight="bold")

    # B: evidence F1 versus review-set size.
    policy_style = {
        "fixed": ("Fixed", COLORS["gray"], "o"),
        "calibrated_threshold": ("Threshold", COLORS["blue"], "s"),
        "expected_f1": ("Expected-F1", COLORS["teal"], "D"),
        "uncalibrated_expected_f1": ("Uncalibrated", COLORS["red"], "X"),
        "completeness_aware_switch": ("Switch", COLORS["orange"], "^"),
    }
    offsets = {
        "fixed": (4, -11),
        "calibrated_threshold": (4, 5),
        "expected_f1": (4, 5),
        "uncalibrated_expected_f1": (-49, 5),
        "completeness_aware_switch": (-34, -12),
    }
    for policy, (label, color, marker) in policy_style.items():
        row = result["selector"][policy]
        axes[1].scatter(
            row["selected_count"],
            row["f1"],
            s=48,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        axes[1].annotate(
            label,
            (row["selected_count"], row["f1"]),
            xytext=offsets[policy],
            textcoords="offset points",
            fontsize=7.5,
        )
    axes[1].set_xlabel("Mean selected items")
    axes[1].set_ylabel("Exact-source F1")
    axes[1].set_xlim(0.4, 10.5)
    axes[1].set_ylim(0.04, 0.112)
    axes[1].grid(alpha=0.22, linewidth=0.6)
    axes[1].set_title("B  Quality--size trade-off", loc="left", fontweight="bold")

    # C: selector and verifier paired effects.
    comparison_rows = [
        ("Threshold - fixed", result["comparisons"]["threshold_minus_fixed"], False),
        ("Expected-F1 - threshold", result["comparisons"]["expected_f1_minus_threshold"], False),
        (
            "Calibrated - uncalibrated",
            result["comparisons"]["calibrated_minus_uncalibrated_expected_f1"],
            True,
        ),
        ("Switch - threshold", result["comparisons"]["completeness_switch_minus_threshold"], False),
        (
            "Verifier: switch - threshold",
            result["verifier"]["paired_switch_minus_threshold"],
            False,
        ),
    ]
    y = np.arange(len(comparison_rows))[::-1]
    for y_value, (_, row, passed) in zip(y, comparison_rows, strict=True):
        lower_key = "lower_adjusted" if "lower_adjusted" in row else "lower"
        upper_key = "upper_adjusted" if "upper_adjusted" in row else "upper"
        mean = float(row["mean"])
        lower = float(row[lower_key])
        upper = float(row[upper_key])
        axes[2].errorbar(
            mean,
            y_value,
            xerr=np.asarray([[mean - lower], [upper - mean]]),
            fmt="o",
            color=COLORS["green"] if passed else COLORS["red"],
            ecolor=COLORS["green"] if passed else COLORS["gray"],
            capsize=2.5,
            markersize=4.5,
            linewidth=1.2,
        )
    axes[2].axvline(0.0, color="#222222", linewidth=0.8, linestyle="--")
    axes[2].axvline(-0.01, color=COLORS["orange"], linewidth=0.8, linestyle=":")
    axes[2].set_yticks(y, [row[0] for row in comparison_rows])
    axes[2].set_xlabel("Paired effect")
    axes[2].set_xlim(-0.045, 0.095)
    axes[2].grid(axis="x", alpha=0.22, linewidth=0.6)
    axes[2].set_title("C  Prespecified effects", loc="left", fontweight="bold")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
