"""Render the fixed-family failure decomposition and development reliability."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, (failure, reliability) = plt.subplots(
        1, 2, figsize=(7.2, 2.8), layout="constrained", gridspec_kw={"width_ratios": [1.2, 1]}
    )

    stages = report["stages"]["fixed"]["combined"]
    keys = ("absent", "unusable", "outside_pool", "outside_packet", "omitted", "selected")
    labels = ("Absent from indexed store", "Present, text unusable", "Outside ranked pool",
              "Outside scorer packet", "Omitted by fixed policy", "Selected")
    shares = [100 * stages[key] / sum(stages.values()) for key in keys]
    failure.barh(labels, shares, color=["#24557A"] * 4 + ["#C56A2D", "#2A7F78"])
    failure.bar_label(failure.containers[0], fmt="%.1f%%", padding=3, fontsize=8)
    failure.invert_yaxis()
    failure.set_xlim(0, max(shares) * 1.25)
    failure.set_xlabel("Exact targets across five seeds (%)")
    failure.set_title("(a) Where exact targets were lost", loc="left", fontsize=10)

    reliability.plot([0, 1], [0, 1], "--", color="#66717E", linewidth=1, label="Ideal")
    for key, label, color, marker in (
        ("raw", "Raw", "#24557A", "o"), ("calibrated", "Calibrated", "#2A7F78", "s")
    ):
        bins = report["calibration"]["combined"][key]["reliability_pooled"]
        populated = [entry for entry in bins if entry["count"]]
        reliability.scatter(
            [entry["confidence"] for entry in populated],
            [entry["frequency"] for entry in populated],
            s=[12 + 80 * (entry["count"] / sum(b["count"] for b in populated)) ** 0.55
               for entry in populated],
            color=color, alpha=0.8, marker=marker, label=label,
        )
    reliability.set(xlim=(-0.04, 1.04), ylim=(-0.04, 1.04),
                    xlabel="Mean predicted probability", ylabel="Observed exact-match rate")
    reliability.set_title("(b) Candidate reliability", loc="left", fontsize=10)
    reliability.legend(frameon=False, loc="upper left", fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
