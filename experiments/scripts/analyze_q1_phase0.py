from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import fmean

import matplotlib.pyplot as plt
import numpy as np


SEEDS = (13, 37, 73)
LABELS = ("nei", "refuted", "supported")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_mean_bootstrap(values: np.ndarray, *, seed: int, samples: int = 10_000) -> dict[str, object]:
    generator = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 512):
        count = min(512, samples - start)
        indices = generator.integers(0, len(values), size=(count, len(values)))
        draws[start : start + count] = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lower": float(np.quantile(draws, 0.025)),
        "median": float(np.quantile(draws, 0.5)),
        "upper": float(np.quantile(draws, 0.975)),
        "samples": samples,
    }


def sign_flip_pvalue(values: np.ndarray, *, seed: int, samples: int = 100_000) -> float:
    generator = np.random.default_rng(seed)
    observed = abs(float(values.mean()))
    extreme = 0
    remaining = samples
    while remaining:
        count = min(2048, remaining)
        signs = generator.choice((-1.0, 1.0), size=(count, len(values)))
        extreme += int(np.sum(np.abs((signs * values).mean(axis=1)) >= observed - 1e-15))
        remaining -= count
    return (extreme + 1) / (samples + 1)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        value = min(1.0, (total - rank) * pvalues[name])
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def macro_f1(gold: np.ndarray, predicted: np.ndarray) -> float:
    scores = []
    for label in LABELS:
        tp = np.sum((gold == label) & (predicted == label))
        fp = np.sum((gold != label) & (predicted == label))
        fn = np.sum((gold == label) & (predicted != label))
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return fmean(scores)


def verifier_seed_aligned_bootstrap(records_by_seed: dict[int, list[dict[str, object]]]) -> dict[str, object]:
    first = records_by_seed[SEEDS[0]]
    claim_ids = [str(row["claim_id"]) for row in first]
    by_seed = {
        seed: {str(row["claim_id"]): row for row in rows}
        for seed, rows in records_by_seed.items()
    }
    gold = np.asarray([str(row["gold_label"]) for row in first], dtype=object)
    left = {
        seed: np.asarray(
            [str(by_seed[seed][claim_id]["methods"]["cefes"]["predicted_label"]) for claim_id in claim_ids],
            dtype=object,
        )
        for seed in SEEDS
    }
    right = {
        seed: np.asarray(
            [str(by_seed[seed][claim_id]["methods"]["fixed"]["predicted_label"]) for claim_id in claim_ids],
            dtype=object,
        )
        for seed in SEEDS
    }
    point = fmean(macro_f1(gold, left[seed]) - macro_f1(gold, right[seed]) for seed in SEEDS)
    generator = np.random.default_rng(20261102)
    values = []
    for _ in range(10_000):
        indices = generator.integers(0, len(gold), size=len(gold))
        values.append(
            fmean(
                macro_f1(gold[indices], left[seed][indices])
                - macro_f1(gold[indices], right[seed][indices])
                for seed in SEEDS
            )
        )
    return {
        "mean": point,
        "lower": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper": float(np.quantile(values, 0.975)),
        "noninferiority_margin": -0.01,
        "noninferiority_pass": float(np.quantile(values, 0.025)) > -0.01,
        "samples": 10_000,
    }


def reliability_rows(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    rows = []
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        if np.any(mask):
            rows.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "count": int(mask.sum()),
                    "mean_probability": float(probabilities[mask].mean()),
                    "observed_frequency": float(labels[mask].mean()),
                }
            )
    return rows


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    root = repo / "experiments" / "outputs" / "mocheg"
    selector = root / "full_corpus_selector_trial_002"
    count = root / "full_corpus_dacs_trial_002"
    verifier = root / "full_corpus_matched_verifier_trial_002"
    output = repo / "research" / "q1_phase2"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    evidence_by_claim: dict[str, list[float]] = {}
    count_by_claim: dict[str, list[float]] = {}
    verifier_records: dict[int, list[dict[str, object]]] = {}
    reliability: dict[int, list[dict[str, float]]] = {}
    source_paths: list[Path] = []
    for seed in SEEDS:
        claim_path = selector / f"seed_{seed}_claim_records.jsonl"
        count_path = count / f"seed_{seed}_dacs_records.jsonl"
        candidate_path = selector / f"seed_{seed}_candidate_records.jsonl"
        verdict_path = verifier / f"seed_{seed}_verdict_records.jsonl"
        source_paths.extend((claim_path, count_path, candidate_path, verdict_path))
        for row in read_jsonl(claim_path):
            evidence_by_claim.setdefault(str(row["claim_id"]), []).append(
                float(row["cefes_minus_fixed_f1"])
            )
        for row in read_jsonl(count_path):
            count_by_claim.setdefault(str(row["claim_id"]), []).append(
                float(row["cefes_minus_dacs_f1"])
            )
        verifier_records[seed] = read_jsonl(verdict_path)
        candidates = [row for row in read_jsonl(candidate_path) if row["role"] == "validation"]
        probabilities = np.asarray([float(row["calibrated_probability"]) for row in candidates])
        labels = np.asarray([int(bool(row["gold"])) for row in candidates])
        reliability[seed] = reliability_rows(probabilities, labels)

    evidence_values = np.asarray([fmean(values) for _, values in sorted(evidence_by_claim.items())])
    count_values = np.asarray([fmean(values) for _, values in sorted(count_by_claim.items())])
    effects = {
        "cefes_minus_fixed": paired_mean_bootstrap(evidence_values, seed=13037),
        "cefes_minus_count": paired_mean_bootstrap(count_values, seed=37073),
    }
    raw_p = {
        "cefes_minus_fixed": sign_flip_pvalue(evidence_values, seed=130037),
        "cefes_minus_count": sign_flip_pvalue(count_values, seed=370073),
    }
    adjusted = holm(raw_p)
    for name in effects:
        effects[name]["sign_flip_p_raw"] = raw_p[name]
        effects[name]["sign_flip_p_holm"] = adjusted[name]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharex=True, sharey=True)
    for axis, seed in zip(axes, SEEDS, strict=True):
        rows = reliability[seed]
        axis.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
        axis.plot(
            [row["mean_probability"] for row in rows],
            [row["observed_frequency"] for row in rows],
            marker="o",
            linewidth=1.5,
        )
        axis.set_title(f"Seed {seed}")
        axis.set_xlabel("Mean predicted probability")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Observed evidence frequency")
    fig.tight_layout()
    reliability_pdf = figures / "frozen_full_corpus_reliability.pdf"
    reliability_png = figures / "frozen_full_corpus_reliability.png"
    fig.savefig(reliability_pdf, bbox_inches="tight")
    fig.savefig(reliability_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    result = {
        "profile": "q1_phase2_phase0_frozen_sync",
        "status": "pass",
        "scope": "post-freeze recomputation only",
        "aggregate_evidence_effects": effects,
        "verifier_seed_aligned": verifier_seed_aligned_bootstrap(verifier_records),
        "reliability": {str(seed): rows for seed, rows in reliability.items()},
        "artifacts": {
            "reliability_pdf": {"path": str(reliability_pdf), "sha256": sha256(reliability_pdf)},
            "reliability_png": {"path": str(reliability_png), "sha256": sha256(reliability_png)},
        },
        "source_hashes": {str(path.relative_to(repo)): sha256(path) for path in source_paths},
        "formal_test_unlocked": False,
        "formal_test_annotations_accessed": False,
    }
    target = output / "phase0_sync.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(target)}, indent=2))


if __name__ == "__main__":
    main()
