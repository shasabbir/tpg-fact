from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from trace_scv.external_reporting import set_metrics
from trace_scv.external_study import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit exact-source reachability in prepared AVerImaTeC candidates.")
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output_path = args.prepared_root / "reachability.audit.json"
    claim_path = args.prepared_root / "reachability.claims.jsonl"
    if (output_path.exists() or claim_path.exists()) and not args.overwrite:
        raise FileExistsError("refusing to overwrite an existing reachability audit")
    rows = read_jsonl(args.prepared_root / "candidate_records.jsonl")
    gold = json.loads((args.prepared_root / "full_gold_sources.json").read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(f"{row['split']}:{row['claim_id']}", []).append(row)
    claim_rows: list[dict[str, object]] = []
    depths = (10, 20, 50, 100)
    for key in sorted(gold):
        expected = set(map(str, gold[key]))
        if not expected:
            continue
        candidates = grouped.get(key, [])
        ranked = sorted(
            candidates,
            key=lambda row: (-float(row["features"][3]), str(row["evidence_id"])),
        )
        pool = {str(row["normalized_source_url"]) for row in ranked}
        record: dict[str, object] = {
            "key": key,
            "claim_id": key.split(":", 1)[1],
            "candidate_rows": len(candidates),
            "candidate_sources": len(pool),
            "gold_sources": len(expected),
            "pool_recall": len(pool & expected) / len(expected),
            "complete_pool_reachability": float(expected <= pool),
            "zero_pool_reachability": float(not (pool & expected)),
        }
        for depth in depths:
            selected = {str(row["normalized_source_url"]) for row in ranked[:depth]}
            record[f"recall_at_{depth}"] = len(selected & expected) / len(expected)
            record[f"complete_at_{depth}"] = float(expected <= selected)
        text = [row for row in ranked if row["modality"] == "text"][:100]
        image = [row for row in ranked if row["modality"] == "image"][:20]
        packet = {str(row["normalized_source_url"]) for row in text + image}
        record.update({f"packet_{name}": value for name, value in set_metrics(packet, expected).items()})
        claim_rows.append(record)
    write_jsonl(claim_path, claim_rows)
    metric_names = (
        "pool_recall",
        "complete_pool_reachability",
        "zero_pool_reachability",
        *(f"recall_at_{depth}" for depth in depths),
        *(f"complete_at_{depth}" for depth in depths),
        "packet_precision",
        "packet_recall",
        "packet_f1",
        "packet_exact",
        "packet_selected_count",
    )
    summary = {
        "status": "checked",
        "condition": args.condition,
        "claims_with_web_gold": len(claim_rows),
        "candidate_rows": len(rows),
        "mean_candidate_rows_per_claim": fmean(float(row["candidate_rows"]) for row in claim_rows),
        "metrics": {
            metric: fmean(float(row[metric]) for row in claim_rows) for metric in metric_names
        },
        "metric_scope": "normalized exact-source diagnostic; not the official semantic evidence score",
        "test_annotations_accessed": False,
    }
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
