from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import fmean
from typing import Iterable

import numpy as np

from trace_scv.data.io import sha256_file


_TOLERANCE = 1e-12
_EPSILON = 1e-7


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _set_f1(predicted: set[str], gold: set[str]) -> float:
    denominator = len(predicted) + len(gold)
    return 2.0 * len(predicted & gold) / denominator if denominator else 1.0


def _brier(rows: Iterable[dict[str, object]], field: str) -> float:
    values = list(rows)
    return fmean((float(row[field]) - int(bool(row["gold"]))) ** 2 for row in values)


def _temperature_scale(probability: float, temperature: float) -> float:
    probability = min(1.0 - _EPSILON, max(_EPSILON, probability))
    logit = math.log(probability) - math.log1p(-probability)
    scaled = logit / temperature
    if scaled >= 0.0:
        inverse = math.exp(-scaled)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(scaled)
    return exponent / (1.0 + exponent)


def _close(left: float, right: float, tolerance: float = _TOLERANCE) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def _inventory(paths: Iterable[Path]) -> list[dict[str, object]]:
    return [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(set(path.resolve() for path in paths), key=str)
    ]


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_commit(root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def audit_mocheg_selector_pilot(
    feature_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Independently recompute saved pilot endpoints without rerunning the model."""
    feature_path = feature_path.resolve()
    output_dir = output_dir.resolve()
    sidecar_path = feature_path.with_suffix(".json")
    feature_manifest_path = feature_path.with_suffix(".manifest.json")
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    full_gold = {
        str(claim_id): {str(evidence_id) for evidence_id in evidence_ids}
        for claim_id, evidence_ids in sidecar.get(
            "full_gold_evidence_ids_by_claim", {}
        ).items()
    }
    arrays = np.load(feature_path, allow_pickle=False)

    discrepancies: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            discrepancies.append(message)

    require(
        sha256_file(feature_path) == summary["feature_sha256"],
        "summary feature hash mismatch",
    )
    require(
        sha256_file(sidecar_path) == summary["feature_sidecar_sha256"],
        "summary sidecar hash mismatch",
    )
    require(
        sha256_file(feature_manifest_path) == summary["feature_manifest_sha256"],
        "summary feature-manifest hash mismatch",
    )
    require(
        sha256_file(feature_path) == feature_manifest["features_sha256"],
        "feature manifest NPZ hash mismatch",
    )
    require(
        sha256_file(sidecar_path) == feature_manifest["sidecar_sha256"],
        "feature manifest sidecar hash mismatch",
    )
    require(bool(feature_manifest["scientific_results_allowed"]), "non-scientific features")
    require(not summary["formal_test_annotations_accessed"], "summary reports test access")
    require(
        not feature_manifest["formal_test_annotations_accessed"],
        "feature manifest reports test access",
    )
    require(
        not sidecar["formal_test_annotations_accessed"],
        "feature sidecar reports test access",
    )
    require(len(arrays["features"]) == feature_manifest["rows"], "feature row mismatch")
    require(len(arrays["gold"]) == feature_manifest["rows"], "gold row mismatch")
    require(len(sidecar["records"]) == feature_manifest["rows"], "sidecar row mismatch")

    audited_seeds: list[dict[str, object]] = []
    recomputed_gains: list[float] = []
    brier_not_worse: list[bool] = []
    artifact_paths = [feature_path, sidecar_path, feature_manifest_path, summary_path]
    for summary_seed in summary["seeds"]:
        seed = int(summary_seed["seed"])
        seed_path = output_dir / f"seed_{seed}.json"
        claim_path = output_dir / f"seed_{seed}_claim_records.jsonl"
        candidate_path = output_dir / f"seed_{seed}_candidate_records.jsonl"
        artifact_paths.extend((seed_path, claim_path, candidate_path))
        saved = json.loads(seed_path.read_text(encoding="utf-8"))
        candidates = _read_jsonl(candidate_path)
        claims = _read_jsonl(claim_path)
        require(saved == summary_seed, f"seed {seed} JSON differs from summary")
        require(
            sha256_file(claim_path) == saved["artifacts"]["claim_records"]["sha256"],
            f"seed {seed} claim-record hash mismatch",
        )
        require(
            sha256_file(candidate_path)
            == saved["artifacts"]["candidate_records"]["sha256"],
            f"seed {seed} candidate-record hash mismatch",
        )

        role_counts = Counter(str(row["role"]) for row in candidates)
        validation = [row for row in candidates if row["role"] == "validation"]
        require(
            len(validation) == summary["bounded_candidate_rows"]["validation"],
            f"seed {seed} validation candidate count mismatch",
        )
        require(
            role_counts["fit"] + role_counts["calibration"]
            == summary["bounded_candidate_rows"]["train"],
            f"seed {seed} train candidate count mismatch",
        )
        require(
            len({(str(row["claim_id"]), str(row["evidence_id"])) for row in candidates})
            == len(candidates),
            f"seed {seed} duplicate candidate identity",
        )

        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in validation:
            grouped[str(row["claim_id"])].append(row)
        cefes_f1: list[float] = []
        fixed_f1: list[float] = []
        for claim_id, rows in grouped.items():
            gold = full_gold.get(
                claim_id,
                {str(row["evidence_id"]) for row in rows if row["gold"]},
            )
            cefes = {
                str(row["evidence_id"]) for row in rows if row["selected_cefes"]
            }
            fixed = {
                str(row["evidence_id"]) for row in rows if row["selected_fixed"]
            }
            cefes_f1.append(_set_f1(cefes, gold))
            fixed_f1.append(_set_f1(fixed, gold))
            ordered = sorted(
                rows,
                key=lambda row: (
                    -float(row["calibrated_probability"]),
                    int(row["calibrated_rank"]),
                ),
            )
            require(
                [int(row["calibrated_rank"]) for row in ordered]
                == list(range(1, len(rows) + 1)),
                f"seed {seed} invalid calibrated ranks for claim {claim_id}",
            )

        recomputed_cefes = fmean(cefes_f1)
        recomputed_fixed = fmean(fixed_f1)
        gain = recomputed_cefes - recomputed_fixed
        recomputed_gains.append(gain)
        require(
            _close(recomputed_cefes, saved["validation"]["cefes"]["f1"]),
            f"seed {seed} CEFES F1 mismatch",
        )
        require(
            _close(recomputed_fixed, saved["validation"]["fixed"]["f1"]),
            f"seed {seed} fixed F1 mismatch",
        )

        raw_brier = _brier(validation, "raw_probability")
        calibrated_brier = _brier(validation, "calibrated_probability")
        brier_not_worse.append(calibrated_brier <= raw_brier + _TOLERANCE)
        require(
            _close(raw_brier, saved["validation"]["calibration"]["raw"]["brier"]),
            f"seed {seed} raw Brier mismatch",
        )
        require(
            _close(
                calibrated_brier,
                saved["validation"]["calibration"]["calibrated"]["brier"],
            ),
            f"seed {seed} calibrated Brier mismatch",
        )
        temperature = float(saved["temperature_scaler"]["temperature"])
        maximum_temperature_error = max(
            abs(
                _temperature_scale(float(row["raw_probability"]), temperature)
                - float(row["calibrated_probability"])
            )
            for row in candidates
        )
        require(
            maximum_temperature_error <= _TOLERANCE,
            f"seed {seed} calibrated probabilities do not match temperature",
        )

        bootstrap = saved["validation"]["cefes_minus_fixed_bootstrap"]
        samples = np.asarray(bootstrap["samples"], dtype=np.float64)
        require(len(samples) == bootstrap["sample_count"], f"seed {seed} bootstrap count")
        for field, quantile in (("lower", 0.025), ("median", 0.5), ("upper", 0.975)):
            require(
                _close(float(np.quantile(samples, quantile)), float(bootstrap[field])),
                f"seed {seed} bootstrap {field} mismatch",
            )
        require(_close(gain, bootstrap["mean"]), f"seed {seed} bootstrap mean mismatch")

        claim_by_id = {str(row["claim_id"]): row for row in claims}
        require(set(claim_by_id) == set(grouped), f"seed {seed} claim coverage mismatch")
        for claim_id, rows in grouped.items():
            require(
                _close(
                    _set_f1(
                        {
                            str(row["evidence_id"])
                            for row in rows
                            if row["selected_cefes"]
                        },
                        full_gold.get(
                            claim_id,
                            {
                                str(row["evidence_id"])
                                for row in rows
                                if row["gold"]
                            },
                        ),
                    ),
                    float(claim_by_id[claim_id]["f1"]),
                ),
                f"seed {seed} claim-record F1 mismatch for {claim_id}",
            )

        audited_seeds.append(
            {
                "seed": seed,
                "role_counts": dict(sorted(role_counts.items())),
                "validation_claims": len(grouped),
                "cefes_f1": recomputed_cefes,
                "fixed_f1": recomputed_fixed,
                "cefes_minus_fixed_f1": gain,
                "raw_brier": raw_brier,
                "calibrated_brier": calibrated_brier,
                "calibrated_brier_not_worse": brier_not_worse[-1],
                "maximum_temperature_transform_error": maximum_temperature_error,
                "bootstrap": {
                    "sample_count": len(samples),
                    "resampling_seed": bootstrap["resampling_seed"],
                    "lower": float(np.quantile(samples, 0.025)),
                    "median": float(np.quantile(samples, 0.5)),
                    "upper": float(np.quantile(samples, 0.975)),
                },
            }
        )

    recomputed_mean_gain = fmean(recomputed_gains)
    recomputed_numeric_gate = (
        recomputed_mean_gain >= float(summary["config"]["minimum_mean_gain"])
        and all(gain > 0.0 for gain in recomputed_gains)
        and all(brier_not_worse)
    )
    require(
        _close(recomputed_mean_gain, summary["mean_cefes_minus_fixed_f1"]),
        "summary mean gain mismatch",
    )
    require(
        recomputed_numeric_gate == summary["selector_numeric_criteria_met"],
        "selector numeric gate mismatch",
    )
    require(not summary["full_development_gate_pass"], "full gate unexpectedly passed")

    return {
        "status": "pass" if not discrepancies else "failed",
        "discrepancies": discrepancies,
        "feature_identity": {
            "sha256": sha256_file(feature_path),
            "rows": int(feature_manifest["rows"]),
            "claims": int(feature_manifest["claims"]),
            "scientific_results_allowed": bool(
                feature_manifest["scientific_results_allowed"]
            ),
            "encoder": feature_manifest["encoder"],
        },
        "seeds": audited_seeds,
        "recomputed": {
            "mean_cefes_minus_fixed_f1": recomputed_mean_gain,
            "positive_effect_each_seed": all(gain > 0.0 for gain in recomputed_gains),
            "calibrated_brier_not_worse_each_seed": all(brier_not_worse),
            "selector_numeric_criteria_met": recomputed_numeric_gate,
            "full_development_gate_pass": False,
            "formal_test_annotations_accessed": False,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "huggingface_hub": _package_version("huggingface-hub"),
            "pillow": _package_version("pillow"),
            "nltk": _package_version("nltk"),
            "source_commit": _git_commit(Path.cwd()),
        },
        "artifact_inventory": _inventory(artifact_paths),
    }
