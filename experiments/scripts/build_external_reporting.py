from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from trace_scv.data.io import sha256_file
from trace_scv.external_reporting import (
    apply_holm_to_intervals,
    paired_claim_bootstrap,
    summarize_claim_rows,
)
from trace_scv.external_study import read_jsonl
from trace_scv.external_verifier import (
    OFFICIAL_LABELS,
    classification_metrics,
    seed_aligned_macro_f1_bootstrap,
)


POLICIES = (
    "fixed",
    "calibrated_threshold",
    "expected_f1",
    "uncalibrated_expected_f1",
    "completeness_aware_switch",
)
METRICS = ("precision", "recall", "f1", "exact", "selected_count")
COMPARISONS = {
    "threshold_minus_fixed": ("calibrated_threshold", "fixed", 91_013),
    "expected_f1_minus_threshold": ("expected_f1", "calibrated_threshold", 91_037),
    "calibrated_minus_uncalibrated_expected_f1": (
        "expected_f1",
        "uncalibrated_expected_f1",
        91_073,
    ),
    "completeness_switch_minus_threshold": (
        "completeness_aware_switch",
        "calibrated_threshold",
        91_101,
    ),
}


def _assert_close(actual: Any, expected: Any, path: str = "root") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError(f"mapping mismatch at {path}")
        for key in expected:
            _assert_close(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"list mismatch at {path}")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _assert_close(left, right, f"{path}[{index}]")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(f"numeric mismatch at {path}: {actual} != {expected}")
        return
    if actual != expected:
        raise AssertionError(f"value mismatch at {path}: {actual!r} != {expected!r}")


def _averaged_claim_rows(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    for policy in POLICIES:
        by_claim: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            if str(row["policy"]) == policy:
                by_claim[str(row["claim_id"])].append(row)
        result[policy] = [
            {
                "claim_id": claim_id,
                **{
                    metric: fmean(float(row[metric]) for row in claim_rows)
                    for metric in METRICS
                },
            }
            for claim_id, claim_rows in sorted(by_claim.items())
        ]
    return result


def _recompute_verifier(
    rows: list[dict[str, object]], summary: dict[str, object]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    table_rows: list[dict[str, object]] = []
    aligned_runs: list[tuple[list[str], list[str], list[str]]] = []
    for seed_row in summary["seeds"]:
        seed = int(seed_row["seed"])
        predictions: dict[str, tuple[list[str], list[str]]] = {}
        for policy in ("calibrated_threshold", "completeness_aware_switch"):
            selected = sorted(
                (
                    row
                    for row in rows
                    if int(row["seed"]) == seed and str(row["policy"]) == policy
                ),
                key=lambda row: str(row["claim_id"]),
            )
            gold = [str(row["gold_verdict"]) for row in selected]
            predicted = [str(row["predicted_verdict"]) for row in selected]
            metrics = classification_metrics(gold, predicted)
            _assert_close(
                metrics,
                seed_row["shared_verifier"]["metrics"][policy],
                f"verifier.seed_{seed}.{policy}",
            )
            predictions[policy] = (gold, predicted)
            record: dict[str, object] = {
                "seed": seed,
                "policy": policy,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
            for label in OFFICIAL_LABELS:
                record[f"{label}_f1"] = metrics["per_class"][label]["f1"]
            table_rows.append(record)
        threshold_gold, threshold_predictions = predictions["calibrated_threshold"]
        switch_gold, switch_predictions = predictions["completeness_aware_switch"]
        if threshold_gold != switch_gold:
            raise AssertionError(f"verifier labels are not aligned for seed {seed}")
        aligned_runs.append((threshold_gold, switch_predictions, threshold_predictions))

    aggregate = seed_aligned_macro_f1_bootstrap(
        aligned_runs,
        samples=10_000,
        seed=291_013,
    )
    _assert_close(
        aggregate,
        summary["shared_verifier"]["aggregate_switch_minus_threshold_macro_f1"],
        "verifier.aggregate",
    )
    means: dict[str, dict[str, float]] = {}
    for policy in ("calibrated_threshold", "completeness_aware_switch"):
        policy_rows = [row for row in table_rows if row["policy"] == policy]
        mean_row = {
            key: fmean(float(row[key]) for row in policy_rows)
            for key in policy_rows[0]
            if key not in {"seed", "policy"}
        }
        means[policy] = mean_row
        table_rows.append({"seed": "mean", "policy": policy, **mean_row})
    return table_rows, {"means": means, "paired_switch_minus_threshold": aggregate}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _latex_table(results: dict[str, object]) -> str:
    labels = {
        "fixed": "Train-selected fixed top-$k$",
        "calibrated_threshold": "Calibrated threshold",
        "expected_f1": "Calibrated expected-F1",
        "uncalibrated_expected_f1": "Uncalibrated expected-F1",
        "completeness_aware_switch": "Completeness-aware switch",
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Locked AVerImaTeC development results averaged over five train-only seeds. Evidence scores use exact normalized source identity and are diagnostic, not the unavailable official semantic score.}",
        r"\label{tab:external_selector}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Policy & Precision & Recall & F1 & Exact & Items \\",
        r"\midrule",
    ]
    selector = results["selector"]
    for policy in POLICIES:
        row = selector[policy]
        lines.append(
            f"{labels[policy]} & {row['precision']:.4f} & {row['recall']:.4f} & "
            f"{row['f1']:.4f} & {row['exact']:.4f} & {row['selected_count']:.2f} \\\\" 
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independently recompute and report locked AVerImaTeC results."
    )
    parser.add_argument(
        "--locked-root",
        type=Path,
        default=Path("outputs/external_confirmation/averimatec/locked_development"),
    )
    parser.add_argument(
        "--train-reachability",
        type=Path,
        default=Path(
            "outputs/external_confirmation/averimatec/prepared_train/reachability.audit.json"
        ),
    )
    parser.add_argument(
        "--validation-reachability",
        type=Path,
        default=Path(
            "outputs/external_confirmation/averimatec/prepared_validation/reachability.audit.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/external_confirmation/averimatec/reporting"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite populated {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    summary_path = args.locked_root / "summary.json"
    records_path = args.locked_root / "claim_records.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = read_jsonl(records_path)
    if summary["status"] != "checked_locked_development":
        raise RuntimeError("locked development summary is not checked")
    if summary["test_annotations_accessed"] or summary["test_submission_performed"]:
        raise RuntimeError("test-access boundary was violated")
    if len(rows) != 152 * 5 * len(POLICIES):
        raise AssertionError(f"unexpected claim-record count: {len(rows)}")
    if {str(row["split"]) for row in rows} != {"validation"}:
        raise AssertionError("claim records are not validation-only")
    if sha256_file(records_path) != summary["artifacts"]["claim_records"]["sha256"]:
        raise AssertionError("claim-record hash does not match locked summary")

    averaged = _averaged_claim_rows(rows)
    selector = {policy: summarize_claim_rows(averaged[policy]) for policy in POLICIES}
    _assert_close(selector, summary["aggregate_mean_over_seed_then_claim"], "selector")
    intervals = apply_holm_to_intervals(
        {
            name: paired_claim_bootstrap(
                averaged[left],
                averaged[right],
                samples=10_000,
                seed=seed,
                family_size=len(COMPARISONS),
            )
            for name, (left, right, seed) in COMPARISONS.items()
        }
    )
    _assert_close(intervals, summary["paired_claim_bootstrap"], "comparisons")

    verifier_rows, verifier = _recompute_verifier(rows, summary)
    threshold = selector["calibrated_threshold"]
    expected = selector["expected_f1"]
    switch = selector["completeness_aware_switch"]
    size_reduction = 1.0 - expected["selected_count"] / threshold["selected_count"]
    _assert_close(
        size_reduction,
        summary["expected_f1_size_reduction_vs_threshold"],
        "expected_f1_size_reduction",
    )
    switch_seed_effects = [
        float(seed["metrics"]["completeness_aware_switch"]["f1"])
        - float(seed["metrics"]["calibrated_threshold"]["f1"])
        for seed in summary["seeds"]
    ]
    _assert_close(
        switch_seed_effects,
        summary["completeness_switch_seed_effects"],
        "switch_seed_effects",
    )

    h1 = intervals["threshold_minus_fixed"]
    h2 = intervals["expected_f1_minus_threshold"]
    h3 = intervals["calibrated_minus_uncalibrated_expected_f1"]
    switch_interval = intervals["completeness_switch_minus_threshold"]
    decisions = {
        "threshold_superiority": bool(h1["mean"] >= 0.010 and h1["lower_adjusted"] > 0.0),
        "expected_f1_noninferiority": bool(h2["lower_adjusted"] > -0.010),
        "expected_f1_size_reduction": bool(size_reduction >= 0.15),
        "expected_f1_compact_operating_point": bool(
            h2["lower_adjusted"] > -0.010 and size_reduction >= 0.15
        ),
        "calibration_benefit": bool(
            h3["mean"] >= 0.010 and h3["lower_adjusted"] > 0.0
        ),
        "completeness_switch_superiority": bool(
            switch_interval["mean"] >= 0.010
            and switch_interval["lower_adjusted"] > 0.0
            and all(value > 0.0 for value in switch_seed_effects)
            and switch["selected_count"] <= threshold["selected_count"]
        ),
        "verifier_noninferiority": bool(
            verifier["paired_switch_minus_threshold"]["lower"] > -0.010
        ),
    }
    if decisions["completeness_switch_superiority"]:
        blind_policy = "completeness_aware_switch"
    elif decisions["expected_f1_compact_operating_point"]:
        blind_policy = "expected_f1"
    elif decisions["threshold_superiority"]:
        blind_policy = "calibrated_threshold"
    else:
        blind_policy = "fixed"

    train_reachability = json.loads(args.train_reachability.read_text(encoding="utf-8"))
    validation_reachability = json.loads(
        args.validation_reachability.read_text(encoding="utf-8")
    )
    result = {
        "status": "checked_independent_recomputation",
        "selector": selector,
        "comparisons": intervals,
        "decisions": decisions,
        "blind_policy_family": blind_policy,
        "expected_f1_size_reduction_vs_threshold": size_reduction,
        "switch_seed_effects": switch_seed_effects,
        "verifier": verifier,
        "reachability": {
            "train": train_reachability,
            "validation": validation_reachability,
        },
        "official_semantic_evaluator": summary["official_semantic_evaluator"],
        "test_annotations_accessed": False,
        "test_submission_performed": False,
        "source_artifacts": {
            "summary": {"path": summary_path.as_posix(), "sha256": sha256_file(summary_path)},
            "claim_records": {"path": records_path.as_posix(), "sha256": sha256_file(records_path)},
            "train_reachability": {
                "path": args.train_reachability.as_posix(),
                "sha256": sha256_file(args.train_reachability),
            },
            "validation_reachability": {
                "path": args.validation_reachability.as_posix(),
                "sha256": sha256_file(args.validation_reachability),
            },
        },
    }
    results_path = args.output_root / "external_results.json"
    results_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    selector_rows = [{"policy": policy, **selector[policy]} for policy in POLICIES]
    _write_csv(args.output_root / "selector_results.csv", selector_rows)
    _write_csv(args.output_root / "verifier_results.csv", verifier_rows)
    (args.output_root / "external_results.tex").write_text(
        _latex_table(result), encoding="utf-8"
    )
    audit = {
        "status": "pass",
        "claim_records": len(rows),
        "claims": len({str(row["claim_id"]) for row in rows}),
        "seeds": len({int(row["seed"]) for row in rows}),
        "policies": len({str(row["policy"]) for row in rows}),
        "summary_recomputed": True,
        "bootstrap_recomputed": True,
        "verifier_recomputed": True,
        "results": {"path": results_path.as_posix(), "sha256": sha256_file(results_path)},
        "test_annotations_accessed": False,
        "test_submission_performed": False,
    }
    audit_path = args.output_root / "reporting.audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
