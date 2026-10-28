from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean

import numpy as np

from trace_scv.data.io import sha256_file


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _set_f1(predicted: set[str], gold: set[str]) -> float:
    denominator = len(predicted) + len(gold)
    return 2.0 * len(predicted & gold) / denominator if denominator else 1.0


def audit_mocheg_dacs(output_dir: Path) -> dict[str, object]:
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    discrepancies: list[str] = []
    audited: list[dict[str, object]] = []
    tolerance = 1e-12
    for seed_row in summary["seeds"]:
        seed = int(seed_row["seed"])
        checkpoint = output_dir / f"seed_{seed}_dacs_count.json"
        records_path = output_dir / f"seed_{seed}_dacs_records.jsonl"
        if sha256_file(checkpoint) != seed_row["checkpoint"]["sha256"]:
            discrepancies.append(f"seed {seed}: checkpoint hash mismatch")
        if sha256_file(records_path) != seed_row["records"]["sha256"]:
            discrepancies.append(f"seed {seed}: records hash mismatch")
        rows = _read_jsonl(records_path)
        if len(rows) != int(seed_row["validation_claims"]):
            discrepancies.append(f"seed {seed}: validation claim count mismatch")
        dacs_f1: list[float] = []
        cefes_f1: list[float] = []
        effects: list[float] = []
        for row in rows:
            predicted = set(map(str, row["selected_evidence_ids"]))
            gold = set(map(str, row["gold_evidence_ids"]))
            dacs = _set_f1(predicted, gold)
            if len(predicted) != int(row["predicted_count"]):
                discrepancies.append(f"seed {seed}: predicted count does not match set")
            if abs(dacs - float(row["dacs_f1"])) > tolerance:
                discrepancies.append(f"seed {seed}: DACS F1 mismatch")
            dacs_f1.append(dacs)
            cefes_f1.append(float(row["cefes_f1"]))
            effects.append(float(row["cefes_f1"]) - dacs)
        if abs(fmean(dacs_f1) - float(seed_row["dacs_evidence_f1"])) > tolerance:
            discrepancies.append(f"seed {seed}: aggregate DACS F1 mismatch")
        bootstrap = seed_row["cefes_minus_dacs_bootstrap"]
        samples = np.asarray(bootstrap["samples"], dtype=np.float64)
        if len(samples) != int(bootstrap["sample_count"]):
            discrepancies.append(f"seed {seed}: bootstrap count mismatch")
        for key, quantile in (("lower", 0.025), ("median", 0.5), ("upper", 0.975)):
            if abs(float(np.quantile(samples, quantile)) - float(bootstrap[key])) > tolerance:
                discrepancies.append(f"seed {seed}: bootstrap {key} mismatch")
        if abs(fmean(effects) - float(bootstrap["mean"])) > tolerance:
            discrepancies.append(f"seed {seed}: effect mismatch")
        audited.append(
            {
                "seed": seed,
                "claims": len(rows),
                "dacs_evidence_f1": fmean(dacs_f1),
                "cefes_evidence_f1": fmean(cefes_f1),
                "cefes_minus_dacs_f1": fmean(effects),
            }
        )
    mean_dacs = fmean(float(row["dacs_evidence_f1"]) for row in audited)
    mean_effect = fmean(float(row["cefes_minus_dacs_f1"]) for row in audited)
    if abs(mean_dacs - float(summary["mean_dacs_evidence_f1"])) > tolerance:
        discrepancies.append("mean DACS F1 mismatch")
    if abs(mean_effect - float(summary["mean_cefes_minus_dacs_f1"])) > tolerance:
        discrepancies.append("mean CEFES-minus-DACS mismatch")
    if summary["formal_test_annotations_accessed"]:
        discrepancies.append("formal-test access flag is true")
    result = {
        "profile": "mocheg_dacs_independent_audit",
        "status": "pass" if not discrepancies else "fail",
        "discrepancies": discrepancies,
        "summary_sha256": sha256_file(summary_path),
        "seeds": audited,
        "recomputed": {
            "mean_dacs_evidence_f1": mean_dacs,
            "mean_cefes_minus_dacs_f1": mean_effect,
            "formal_test_annotations_accessed": False,
        },
    }
    (output_dir / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
