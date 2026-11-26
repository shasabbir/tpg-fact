"""Analyze frozen AVerImaTeC development outputs without fitting or decoding."""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

from prepare_averimatec import (
    _build_image_index,
    _build_store_index,
    _content_segments,
    _read_json_or_jsonl,
    _source_url,
)
from trace_scv.data.averimatec import load_annotations, normalize_source_url
from trace_scv.data.io import sha256_file
from trace_scv.external_study import (
    _group,
    _packet_rows,
    _predict_by_claim,
    calibrator_from_dict,
    read_jsonl,
    scorer_from_dict,
)


STAGES = ("absent", "unusable", "outside_pool", "outside_packet", "omitted", "selected")
POLICIES = ("fixed", "calibrated_threshold", "expected_f1",
            "uncalibrated_expected_f1", "completeness_aware_switch")
MODALITIES = ("text", "image", "combined")


def average_precision(labels, probabilities):
    """Stepwise area under the precision-recall curve, grouping tied scores."""
    if not labels.any():
        return None
    order = np.argsort(-probabilities, kind="stable")
    scores = probabilities[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    positives = np.cumsum(labels[order])[ends]
    return float(np.dot(np.diff(positives, prepend=0), positives / (ends + 1)) / positives[-1])


def calibration(labels, probabilities):
    clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
    bins = np.minimum((probabilities * 10).astype(int), 9)
    reliability = []
    for index in range(10):
        mask = bins == index
        reliability.append({
            "count": int(mask.sum()),
            "confidence": float(probabilities[mask].mean()) if mask.any() else None,
            "frequency": float(labels[mask].mean()) if mask.any() else None,
        })
    return {
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "nll": float(np.mean(-labels * np.log(clipped) - (1 - labels) * np.log1p(-clipped))),
        "ece": sum(b["count"] * abs(b["confidence"] - b["frequency"])
                   for b in reliability if b["count"]) / len(labels),
        "ap": average_precision(labels, probabilities),
        "reliability": reliability,
    }


def calibration_report(grouped, predictions):
    candidates = [row for rows in grouped.values() for row in rows]
    modalities = np.array([row["modality"] for row in candidates])
    labels = np.array([int(row["gold"]) for row in candidates])
    report = {}
    for modality in MODALITIES:
        mask = modalities == modality if modality != "combined" else np.ones(len(labels), dtype=bool)
        y = labels[mask]
        prevalence = float(y.mean())
        report[modality] = {
            "rows_per_seed": len(y),
            "positives_per_seed": int(y.sum()),
            "prevalence_brier": prevalence * (1 - prevalence),
        }
        for name, index in (("raw", 0), ("calibrated", 1)):
            values = [np.concatenate(list(pair[index].values()))[mask] for pair in predictions.values()]
            by_seed = [calibration(y, p) for p in values]
            report[modality][name] = {
                metric: float(np.mean([entry[metric] for entry in by_seed]))
                if by_seed[0][metric] is not None else None
                for metric in ("brier", "nll", "ece", "ap")
            }
            report[modality][name]["reliability_pooled"] = calibration(
                np.tile(y, len(values)), np.concatenate(values)
            )["reliability"]
            report[modality][name]["by_seed"] = dict(zip(predictions, by_seed, strict=True))
    return report


def inventory(store, annotations, audits):
    store_index, image_index = _build_store_index(store), _build_image_index(store)
    found = {}
    inputs = []
    for ordinal, claim in enumerate(annotations):
        split_key = ("validation", ordinal)
        key = f"validation:{claim['claim_id']}"
        files = store_index.get(split_key, [])
        images = image_index.get(split_key, [])
        audit = audits[key]
        if len(images) != audit["image_candidates"]:
            raise ValueError(f"Image discovery mismatch: {key}")
        image_ids = {f"image:{p.name.casefold()}" for p in images}
        store_ids = set(image_ids)
        raw_count = 0
        text_rows = set()
        for branch, path in files:
            inputs.append((path.relative_to(store).as_posix(), sha256_file(path)))
            for row in _read_json_or_jsonl(path):
                url = normalize_source_url(_source_url(row))
                if url:
                    store_ids.add(f"url:{url}")
                    if _content_segments(row):
                        raw_count += 1
                        text_rows.add((branch, f"url:{url}"))
        observed = (len(files), raw_count, len(text_rows))
        expected = tuple(audit[k] for k in ("store_files", "raw_candidates", "deduplicated_candidates"))
        if observed != expected:
            raise ValueError(f"Text discovery mismatch: {key}")
        found[key] = (store_ids, image_ids | {target for _, target in text_rows})
    return found, sorted(inputs)


def decomposition(grouped, gold, found, seeds, predictions, frozen):
    counts = {policy: {modality: Counter({stage: 0 for stage in STAGES})
                       for modality in MODALITIES} for policy in POLICIES}
    complete = Counter()
    disagreements = {}
    verdicts = {p: Counter() for p in POLICIES}
    for seed, seed_row in seeds.items():
        for key, rows in grouped.items():
            targets = gold[key]
            store_ids, usable = found[key]
            pool = {row["evaluation_id"] for row in rows}
            kept, _ = _packet_rows(rows, predictions[seed][0][key],
                                   text_width=seed_row["text_packet_width"],
                                   image_width=seed_row["image_packet_width"])
            packet = {row["evaluation_id"] for row in kept}
            by_id = {row["evidence_id"]: row["evaluation_id"] for row in kept}
            if not packet <= pool <= usable <= store_ids:
                raise ValueError(f"Nested candidate inventories disagree: {key}")
            strata = "complete" if targets <= pool else "partial" if targets & pool else "zero"
            for stage, available in (("store", store_ids), ("usable", usable),
                                     ("pool", pool), ("packet", packet)):
                complete[stage] += targets <= available
            sets = {}
            for policy in POLICIES:
                record = frozen[seed, key, policy]
                selected = set(record["selected_source_urls"])
                if selected != {by_id[item] for item in record["selected_evidence_ids"]}:
                    raise ValueError(f"Frozen selection mismatch: {seed}, {key}, {policy}")
                if targets != set(record["gold_source_urls"]):
                    raise ValueError(f"Frozen gold mismatch: {seed}, {key}, {policy}")
                sets[policy] = selected
                complete[policy] += targets <= selected
                for stage, bucket in zip(STAGES, (
                    targets - store_ids, (targets & store_ids) - usable,
                    (targets & usable) - pool, (targets & pool) - packet,
                    (targets & packet) - selected, targets & selected,
                ), strict=True):
                    for target in bucket:
                        modality = "image" if target.startswith("image:") else "text"
                        counts[policy][modality][stage] += 1
                        counts[policy]["combined"][stage] += 1
                if "predicted_verdict" in record:
                    link = "overlap" if targets & selected else "no_overlap"
                    correctness = "correct" if record["gold_verdict"] == record["predicted_verdict"] else "incorrect"
                    verdicts[policy][f"{link}_{correctness}"] += 1
            for left, right in itertools.combinations(POLICIES, 2):
                a, b = sets[left], sets[right]
                key_label = f"{left} vs {right}"
                stats = disagreements.setdefault(key_label, {}).setdefault(strata, Counter())
                stats["pairs"] += 1
                stats["different"] += a != b
                stats["jaccard_sum"] += len(a & b) / len(a | b) if a | b else 1
    return {
        "stages": counts, "complete_claim_seed_counts": complete, "verdicts": verdicts,
        "disagreement": {
            label: {
                stratum: {"pairs": stats["pairs"], "different": stats["different"],
                          "mean_jaccard": stats["jaccard_sum"] / stats["pairs"]}
                for stratum, stats in strata.items()
            }
            for label, strata in disagreements.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/external_confirmation/averimatec"))
    parser.add_argument("--validation-root", type=Path, default=Path("data/public/averimatec/raw"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.validation_root / "val.json"
    store = args.validation_root / "knowledge_store/extracted/validation"
    prepared = args.root / "prepared_validation"
    paths = {
        "claim_records": args.root / "locked_development/claim_records.jsonl",
        "development_candidate_records": prepared / "candidate_records.jsonl",
        "development_full_gold_sources": prepared / "full_gold_sources.json",
        "train_only_freeze": args.root / "train_only.freeze.json",
    }
    summary = json.loads((args.root / "locked_development/summary.json").read_text(encoding="utf-8"))
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if any(hashes[name] != summary["artifacts"][name]["sha256"] for name in paths):
        raise ValueError("Frozen input hash mismatch")
    candidates = read_jsonl(paths["development_candidate_records"])
    records = read_jsonl(paths["claim_records"])
    if any(row["split"] != "validation" for row in candidates + records):
        raise ValueError("This report accepts development artifacts only")
    grouped = _group(candidates)
    gold = {key: set(ids) for key, ids in json.loads(
        paths["development_full_gold_sources"].read_text(encoding="utf-8")).items()}
    annotations = load_annotations(raw, split="validation")
    if gold != {f"validation:{row['claim_id']}": set(row["gold_evidence_targets"]) for row in annotations}:
        raise ValueError("Development annotation identity mismatch")
    freeze = json.loads(paths["train_only_freeze"].read_text(encoding="utf-8"))
    seeds = {entry["seed"]: entry for entry in freeze["seeds"]}
    frozen = {(row["seed"], f"validation:{row['claim_id']}", row["policy"]): row for row in records}
    if set(frozen) != set(itertools.product(seeds, gold, POLICIES)) or len(frozen) != len(records):
        raise ValueError("Incomplete or duplicate frozen policy records")
    if grouped.keys() != gold.keys() or any(not targets for targets in gold.values()):
        raise ValueError("Unexpected development population")
    if any(row["gold"] != (row["evaluation_id"] in gold[key])
           for key, rows in grouped.items() for row in rows):
        raise ValueError("Candidate label mismatch")
    predictions = {
        seed: _predict_by_claim(grouped, scorer_from_dict(entry["scorer"]),
                                calibrator_from_dict(entry["calibrator"]))
        for seed, entry in seeds.items()
    }
    audits = {f"validation:{row['claim_id']}": row for row in read_jsonl(prepared / "claim_audit.jsonl")}
    found, store_files = inventory(store, annotations, audits)
    result = {
        "scope": "post hoc development diagnostics using frozen scores and exact identities",
        "claims": len(gold), "seeds": list(seeds), "unique_claim_target_pairs": sum(map(len, gold.values())),
        "hashes": hashes | {"validation_annotations": sha256_file(raw),
                            "locked_summary": sha256_file(args.root / "locked_development/summary.json")},
        "store_text_file_hashes": dict(store_files),
        "calibration": calibration_report(grouped, predictions),
        **decomposition(grouped, gold, found, seeds, predictions, frozen),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
