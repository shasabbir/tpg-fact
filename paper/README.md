# Main paper

The canonical manuscript is `paper.tex`; the compiled reading copy is `output.pdf`.
It reports a retrieval-conditioned, calibration-aware evidence-selection study
and uses the released MOCHEG validation set as a locked evaluation population.
The formal-test annotations remain sealed.

## Build

From this directory:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
bibtex paper
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
```

The build expects generated tables in `generated/`, PDF figures in `figures/`,
and the 51-entry cited bibliography in `references_used.bib`. The larger
`references.bib` remains the literature catalog. The final rendered-reference
audit is under `../research/publication_confirmation/reference_audit/`.

## Source of reported results

- Frozen confirmation configuration: `../experiments/configs/publication_confirmation.yaml`
- Retrieval outputs: `../experiments/outputs/publication_confirmation/retrieval/`
- Repaired candidate packet: `../experiments/outputs/publication_confirmation/retrieval_candidates/`
- Selector outputs: `../experiments/outputs/publication_confirmation/selector_base/`
- Completeness-switch outputs: `../experiments/outputs/publication_confirmation/completeness_switch/`
- Stronger-verifier outputs: `../experiments/outputs/publication_confirmation/verifier/`
- Compact reporting bundle: `../research/publication_confirmation/final_results.json`
- Prespecified gate reports: `../research/publication_confirmation/retrieval_gate.md`, `selector_gate.md`, and `verifier_gate.md`
- Publication readiness report: `../research/publication_confirmation/manuscript_readiness.md`
- Full diagnostic source bundle: `../research/calibration_corrected_study/`
- Reachability analysis: `../research/calibration_corrected_study/validation_reachability.csv`
- Partition-level selector analysis: `../research/calibration_corrected_study/partition_selector_results.csv`
- Absolute robustness analysis: `../research/calibration_corrected_study/robustness_absolute.csv`

The paper describes a controlled development study and does not present a
formal-test, state-of-the-art, deployment-ready, or external-generalization
claim.
