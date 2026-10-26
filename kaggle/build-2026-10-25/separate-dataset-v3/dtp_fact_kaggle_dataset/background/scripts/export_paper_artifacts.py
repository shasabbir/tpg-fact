from __future__ import annotations

import argparse
import json
from pathlib import Path


DISPLAY_NAMES = {
    "majority": "Majority class",
    "text_only": "Text only",
    "image_text": "Image--text fusion",
    "vanilla_rag": "Vanilla multimodal RAG",
    "time_filtered_rag": "Time-filtered RAG",
    "provenance_pool": "Provenance-aware pooling",
    "no_graph": "Cluster pooling (no graph)",
    "no_temporal_filter": "Without temporal filtering",
    "no_provenance": "Without provenance",
    "no_temporal_features": "Without temporal graph features",
    "no_text_evidence": "Without text evidence",
    "no_visual_evidence": "Without visual evidence",
    "no_sufficiency_gate": "Without sufficiency gate",
    "no_evidence_dropout": "Without evidence dropout",
    "full": "Full decision-time provenance model",
}
MAIN_VARIANTS = (
    "majority",
    "text_only",
    "image_text",
    "vanilla_rag",
    "time_filtered_rag",
    "provenance_pool",
    "full",
)
ABLATION_VARIANTS = (
    "no_temporal_filter",
    "no_provenance",
    "no_temporal_features",
    "no_graph",
    "no_text_evidence",
    "no_visual_evidence",
    "no_sufficiency_gate",
    "no_evidence_dropout",
    "full",
)


def _value(row: dict[str, object], metric: str) -> str:
    summary = row["aggregate"][metric]
    return f"{float(summary['mean']):.3f} $\\pm$ {float(summary['sample_std']):.3f}"


def _table(rows: dict[str, dict[str, object]], variants: tuple[str, ...], caption: str, label: str) -> str:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Method & Macro-F1 $\\uparrow$ & Bal. Acc. $\\uparrow$ & Brier $\\downarrow$ & ECE $\\downarrow$ & AURC $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for variant in variants:
        if variant not in rows:
            raise ValueError(f"Missing completed evaluation for {variant}")
        row = rows[variant]
        values = " & ".join(
            _value(row, metric)
            for metric in ("macro_f1", "balanced_accuracy", "brier", "ece", "aurc")
        )
        lines.append(f"{DISPLAY_NAMES[variant]} & {values} \\\\")
    lines.extend(("\\bottomrule", "\\end{tabular}", "\\end{table*}", ""))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX result tables only from a checked matrix summary"
    )
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--retrieval-report", required=True)
    parser.add_argument("--intervention-report", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.comparison).read_text(encoding="utf-8"))
    if report.get("status") != "checked_matrix_aggregation":
        raise ValueError("Comparison report has not passed matrix aggregation checks")
    rows = {str(row["variant"]): row for row in report["evaluations"]}
    dataset = json.loads(Path(args.dataset_manifest).read_text(encoding="utf-8"))
    if dataset.get("status") != "frozen_dataset":
        raise ValueError("Dataset manifest is not frozen")
    retrieval = json.loads(Path(args.retrieval_report).read_text(encoding="utf-8"))
    if retrieval.get("status") != "completed_frozen_retrieval":
        raise ValueError("Retrieval report is not complete")
    if retrieval.get("diagnostic_skip_reranker"):
        raise ValueError("Paper tables cannot use diagnostic retrieval without reranking")
    interventions = json.loads(Path(args.intervention_report).read_text(encoding="utf-8"))
    if interventions.get("status") != "checked_intervention_evaluation":
        raise ValueError("Intervention report is not complete")
    if interventions.get("diagnostic_skip_synthetic_encoding"):
        raise ValueError("Paper tables cannot use zero-vector intervention diagnostics")
    if interventions.get("conflict_donor_policy") != "audited_manifest":
        raise ValueError("Paper tables require an audited semantic-conflict donor manifest")
    output_root = Path(args.output_directory).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "generated_main_results.tex").write_text(
        _table(
            rows,
            MAIN_VARIANTS,
            "Controlled comparison on the event-disjoint 2025--2026 test set. Values are mean $\\pm$ sample standard deviation across the configured seeds; the deterministic majority baseline has zero seed variation.",
            "tab:main-results",
        ),
        encoding="utf-8",
        newline="\n",
    )
    (output_root / "generated_ablation_results.tex").write_text(
        _table(
            rows,
            ABLATION_VARIANTS,
            "Ablations of temporal validity, provenance, modalities, hierarchy, graph aggregation, and evidence dropout under the frozen protocol.",
            "tab:ablations",
        ),
        encoding="utf-8",
        newline="\n",
    )
    dataset_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Frozen corpus composition by temporal partition.}",
        "\\label{tab:dataset-composition}",
        "\\small",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Split & Claims & Events & Supported & Refuted & Insufficient \\\\",
        "\\midrule",
    ]
    for split in ("train", "validation", "test"):
        item = dataset["splits"][split]
        labels = item["labels"]
        dataset_lines.append(
            f"{split.title()} & {int(item['claims'])} & {int(item['events'])} & "
            f"{int(labels.get('supported', 0))} & {int(labels.get('refuted', 0))} & "
            f"{int(labels.get('insufficient_evidence', 0))} \\\\"
        )
    dataset_lines.extend(
        (
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        )
    )
    (output_root / "generated_dataset_summary.tex").write_text(
        "\n".join(dataset_lines), encoding="utf-8", newline="\n"
    )
    test_retrieval = retrieval["metrics_by_split"]["test"]
    retrieval_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Frozen hybrid retrieval on the event-disjoint test set.}",
        "\\label{tab:retrieval-results}",
        "\\small",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Retriever & Recall@10 $\\uparrow$ & nDCG@10 $\\uparrow$ \\\\",
        "\\midrule",
        f"BM25+BGE-M3+SigLIP2, BGE reranked & {float(test_retrieval['recall_at_10']):.3f} & {float(test_retrieval['ndcg_at_10']):.3f} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    (output_root / "generated_retrieval_results.tex").write_text(
        "\n".join(retrieval_lines), encoding="utf-8", newline="\n"
    )
    intervention_rows = {
        (str(row["variant"]), str(row["scenario"])): row
        for row in interventions["aggregate"]
    }
    robustness_lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Counterfactual evidence robustness. Change rate is relative to each seed's unmodified test prediction.}",
        "\\label{tab:intervention-results}",
        "\\small",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Intervention & Full F1 & Full change & Vanilla RAG F1 & Vanilla change \\\\",
        "\\midrule",
    ]
    scenario_names = {
        "duplicate_1": "One syndicated copy",
        "duplicate_5": "Five syndicated copies",
        "duplicate_10": "Ten syndicated copies",
        "post_cutoff_verdict": "Post-cutoff verdict",
        "remove_earliest_source": "Earliest source removed",
        "dependent_conflict": "Dependent conflict",
        "independent_conflict": "Independent conflict",
        "caption_swap": "Caption/image swap",
        "historical_image_reuse": "Historical image reuse",
    }
    for scenario, display in scenario_names.items():
        full = intervention_rows[("full", scenario)]
        vanilla = intervention_rows[("vanilla_rag", scenario)]
        robustness_lines.append(
            f"{display} & {float(full['macro_f1']['mean']):.3f} & "
            f"{float(full['prediction_change_rate']['mean']):.3f} & "
            f"{float(vanilla['macro_f1']['mean']):.3f} & "
            f"{float(vanilla['prediction_change_rate']['mean']):.3f} \\\\" 
        )
    robustness_lines.extend(("\\bottomrule", "\\end{tabular}", "\\end{table*}", ""))
    (output_root / "generated_intervention_results.tex").write_text(
        "\n".join(robustness_lines), encoding="utf-8", newline="\n"
    )
    macros = {
        "TestClaimCount": report["test_claims"],
        "TestEventCount": report["test_events"],
        "FrozenClaimCount": dataset["claim_count"],
        "FrozenEvidenceCount": dataset["evidence_count"],
        "TestRetrievalRecallAtTen": f"{float(test_retrieval['recall_at_10']):.3f}",
        "TestRetrievalNDCGAtTen": f"{float(test_retrieval['ndcg_at_10']):.3f}",
        "FullMacroFOne": f"{float(rows['full']['aggregate']['macro_f1']['mean']):.3f}",
        "FullMacroFOneStd": f"{float(rows['full']['aggregate']['macro_f1']['sample_std']):.3f}",
    }
    macro_text = "\n".join(
        f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()
    ) + "\n"
    (output_root / "generated_result_macros.tex").write_text(
        macro_text, encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "status": "paper_artifacts_generated",
                "output_directory": str(output_root),
                "files": sorted(path.name for path in output_root.glob("generated_*.tex")),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
