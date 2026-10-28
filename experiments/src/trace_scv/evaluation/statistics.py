from __future__ import annotations

import itertools
import math
from collections import defaultdict

import numpy as np


def _citation_f1(predicted: list[str], gold: list[str]) -> float:
    predicted_set, gold_set = set(predicted), set(gold)
    overlap = len(predicted_set & gold_set)
    if not predicted_set or not gold_set or overlap == 0:
        return 0.0
    precision = overlap / len(predicted_set)
    recall = overlap / len(gold_set)
    return 2.0 * precision * recall / (precision + recall)


def _macro_f1_from_confusions(confusions: np.ndarray) -> np.ndarray:
    true_positive = np.diagonal(confusions, axis1=-2, axis2=-1)
    false_positive = confusions.sum(axis=-2) - true_positive
    false_negative = confusions.sum(axis=-1) - true_positive
    denominator = 2.0 * true_positive + false_positive + false_negative
    scores = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=denominator > 0,
    )
    return scores.mean(axis=-1)


def _group_contributions(records: list[dict], groups: list[str], labels: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group_index = {group: index for index, group in enumerate(groups)}
    label_index = {label: index for index, label in enumerate(labels)}
    confusion = np.zeros((len(groups), len(labels), len(labels)), dtype=np.float64)
    citation_sum = np.zeros(len(groups), dtype=np.float64)
    counts = np.zeros(len(groups), dtype=np.float64)
    for row in records:
        group = group_index[str(row["group_id"])]
        confusion[group, label_index[row["gold_label"]], label_index[row["predicted_label"]]] += 1.0
        citation_sum[group] += _citation_f1(row["predicted_citations"], row["gold_citations"])
        counts[group] += 1.0
    return confusion, citation_sum, counts


def paired_group_bootstrap_comparisons(
    results: list[dict],
    *,
    reference: str = "trace_scv",
    samples: int = 10000,
    seed: int = 13,
    batch_size: int = 256,
) -> dict:
    """Paired paper-group bootstrap of mean-over-seed verdict and citation effects."""
    by_variant: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in results:
        if "_comparison_records" in row:
            by_variant[row["variant"]][int(row["seed"])] = row
    if reference not in by_variant or samples <= 0:
        return {}
    output: dict[str, dict] = {}
    raw = {"verdict_macro_f1": {}, "citation_macro_f1": {}}
    generator = np.random.default_rng(seed)

    for variant, seed_rows in by_variant.items():
        if variant == reference:
            continue
        shared_seeds = sorted(set(seed_rows) & set(by_variant[reference]))
        if not shared_seeds:
            continue
        first_reference = by_variant[reference][shared_seeds[0]]["_comparison_records"]
        groups = sorted({str(row["group_id"]) for row in first_reference})
        labels = sorted({str(row["gold_label"]) for row in first_reference})
        paired: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for shared_seed in shared_seeds:
            reference_records = by_variant[reference][shared_seed]["_comparison_records"]
            variant_records = seed_rows[shared_seed]["_comparison_records"]
            reference_ids = [row["claim_id"] for row in reference_records]
            if reference_ids != [row["claim_id"] for row in variant_records]:
                raise ValueError(f"unaligned comparison records for {variant}, seed {shared_seed}")
            ref_conf, ref_citation, group_counts = _group_contributions(
                reference_records, groups, labels
            )
            var_conf, var_citation, variant_counts = _group_contributions(
                variant_records, groups, labels
            )
            if not np.array_equal(group_counts, variant_counts):
                raise ValueError(f"group counts differ for {variant}, seed {shared_seed}")
            paired.append((ref_conf, var_conf, ref_citation, var_citation, group_counts))

        verdict_differences: list[np.ndarray] = []
        citation_differences: list[np.ndarray] = []
        probabilities = np.full(len(groups), 1.0 / len(groups))
        remaining = samples
        while remaining:
            current = min(batch_size, remaining)
            weights = generator.multinomial(len(groups), probabilities, size=current)
            verdict = np.zeros(current, dtype=np.float64)
            citation = np.zeros(current, dtype=np.float64)
            for ref_conf, var_conf, ref_citation, var_citation, group_counts in paired:
                ref_matrix = np.einsum("bg,gij->bij", weights, ref_conf, optimize=True)
                var_matrix = np.einsum("bg,gij->bij", weights, var_conf, optimize=True)
                verdict += _macro_f1_from_confusions(ref_matrix) - _macro_f1_from_confusions(var_matrix)
                denominator = weights @ group_counts
                citation += np.divide(
                    weights @ (ref_citation - var_citation),
                    denominator,
                    out=np.zeros(current, dtype=np.float64),
                    where=denominator > 0,
                )
            verdict_differences.append(verdict / len(paired))
            citation_differences.append(citation / len(paired))
            remaining -= current

        verdict_values = np.concatenate(verdict_differences)
        citation_values = np.concatenate(citation_differences)

        def interval(values: np.ndarray) -> dict[str, float]:
            p_raw = min(
                1.0,
                2.0
                * min(
                    (np.count_nonzero(values <= 0) + 1) / (len(values) + 1),
                    (np.count_nonzero(values >= 0) + 1) / (len(values) + 1),
                ),
            )
            return {
                "mean": float(values.mean()),
                "lower": float(np.quantile(values, 0.025)),
                "upper": float(np.quantile(values, 0.975)),
                "p_raw": float(p_raw),
            }

        output[variant] = {
            "paired_seeds": shared_seeds,
            "bootstrap_samples": samples,
            "verdict_macro_f1_difference": interval(verdict_values),
            "citation_macro_f1_difference": interval(citation_values),
        }
        raw["verdict_macro_f1"][variant] = output[variant]["verdict_macro_f1_difference"]["p_raw"]
        raw["citation_macro_f1"][variant] = output[variant]["citation_macro_f1_difference"]["p_raw"]

    for metric, values in raw.items():
        for variant, adjusted in holm_adjust(values).items():
            output[variant][f"{metric}_difference"]["p_holm"] = adjusted
    return output


def _exact_sign_flip_pvalue(differences: list[float]) -> float:
    if not differences:
        return 1.0
    observed = abs(float(np.mean(differences)))
    values = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        values.append(abs(float(np.mean([sign * value for sign, value in zip(signs, differences, strict=True)]))))
    return sum(value >= observed - 1e-12 for value in values) / len(values)


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        value = min(1.0, (count - rank) * pvalues[key])
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def seed_level_comparisons(results: list[dict], reference: str = "trace_scv") -> dict:
    by_variant: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in results:
        by_variant[row["variant"]][int(row["seed"])] = row
    if reference not in by_variant:
        return {}
    output: dict[str, dict] = {}
    raw_pvalues: dict[str, dict[str, float]] = {
        "verdict_macro_f1": {},
        "citation_macro_f1": {},
    }
    for variant, seed_rows in by_variant.items():
        if variant == reference:
            continue
        shared = sorted(set(seed_rows) & set(by_variant[reference]))
        if not shared:
            continue
        verdict = [
            by_variant[reference][seed]["verdict_macro_f1"] - seed_rows[seed]["verdict_macro_f1"]
            for seed in shared
        ]
        citation = [
            by_variant[reference][seed]["citation_macro"]["f1"]
            - seed_rows[seed]["citation_macro"]["f1"]
            for seed in shared
        ]
        output[variant] = {
            "paired_seeds": shared,
            "verdict_macro_f1_difference": {
                "mean": float(np.mean(verdict)),
                "sd": float(np.std(verdict, ddof=1)) if len(verdict) > 1 else 0.0,
                "p_raw": _exact_sign_flip_pvalue(verdict),
            },
            "citation_macro_f1_difference": {
                "mean": float(np.mean(citation)),
                "sd": float(np.std(citation, ddof=1)) if len(citation) > 1 else 0.0,
                "p_raw": _exact_sign_flip_pvalue(citation),
            },
        }
        raw_pvalues["verdict_macro_f1"][variant] = output[variant]["verdict_macro_f1_difference"]["p_raw"]
        raw_pvalues["citation_macro_f1"][variant] = output[variant]["citation_macro_f1_difference"]["p_raw"]
    for metric, values in raw_pvalues.items():
        for variant, adjusted in holm_adjust(values).items():
            output[variant][f"{metric}_difference"]["p_holm"] = adjusted
    return output
