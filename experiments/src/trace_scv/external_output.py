from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from trace_scv.data.averimatec import OFFICIAL_LABELS
from trace_scv.external_selection import enforce_official_output_constraints


def candidate_to_official_evidence(row: Mapping[str, object]) -> dict[str, object]:
    modality = str(row["modality"])
    text = str(row.get("content", ""))[:1500]
    images = [str(row["media_path"])] if modality == "image" else []
    evidence = {"text": text, "images": images}
    enforce_official_output_constraints([evidence])
    return evidence


def build_official_predictions(
    annotations: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    policy_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    candidates: dict[tuple[str, str], Mapping[str, object]] = {
        (str(row["claim_id"]), str(row["evidence_id"])): row for row in candidate_rows
    }
    policies = {str(row["claim_id"]): row for row in policy_rows}
    predictions: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for annotation in annotations:
        claim_id = str(annotation["claim_id"])
        if claim_id not in policies:
            raise ValueError(f"missing policy output for claim {claim_id}")
        policy = policies[claim_id]
        if str(policy.get("split")) != "validation":
            raise RuntimeError("official development exporter accepts validation rows only")
        verdict = str(policy.get("predicted_verdict", ""))
        if verdict not in OFFICIAL_LABELS:
            raise ValueError(f"missing or invalid predicted verdict for claim {claim_id}")
        selected_ids = list(map(str, policy["selected_evidence_ids"]))
        evidence: list[dict[str, object]] = []
        evidence_diagnostics: list[dict[str, object]] = []
        for evidence_id in selected_ids:
            key = (claim_id, evidence_id)
            if key not in candidates:
                raise ValueError(f"selected evidence {evidence_id} is absent for claim {claim_id}")
            candidate = candidates[key]
            evidence.append(candidate_to_official_evidence(candidate))
            evidence_diagnostics.append(
                {
                    "evidence_id": evidence_id,
                    "evaluation_id": str(candidate.get("evaluation_id", "")),
                    "source_url": str(candidate.get("source_url", "")),
                    "retrieval_rank": candidate.get("retrieval_rank"),
                    "modality": str(candidate["modality"]),
                }
            )
        enforce_official_output_constraints(evidence)
        predictions.append(
            {
                "evidence": evidence,
                "justification": "",
                "questions": [],
                "verdict": verdict,
            }
        )
        diagnostics.append(
            {
                "claim_id": claim_id,
                "policy": str(policy.get("policy", "")),
                "seed": policy.get("seed"),
                "evidence": evidence_diagnostics,
                "questions_generated": False,
                "justification_generated": False,
                "official_evidence_score_computed": False,
                "test_annotations_accessed": False,
            }
        )
    return predictions, diagnostics


def write_official_predictions(
    output_path: Path,
    diagnostic_path: Path,
    predictions: Sequence[Mapping[str, object]],
    diagnostics: Sequence[Mapping[str, object]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(list(predictions), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    diagnostic_path.write_text(
        json.dumps(list(diagnostics), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
