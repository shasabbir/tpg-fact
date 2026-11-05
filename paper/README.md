# Main paper

The canonical manuscript is `paper.tex`. It reports a retrieval-conditioned, calibration-aware evidence-selection study and uses the released MOCHEG validation set as a locked evaluation population. The paper includes a post hoc reachability-stratified diagnostic derived from the preserved claim and candidate records.

## Build

From this directory:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
bibtex paper
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
```

The build expects generated tables in `generated/`, PDF figures in `figures/`, and the 55-entry cited bibliography in `references_used.bib`. The larger `references.bib` remains the literature catalog. Figure and table source results are under `../research/calibration_corrected_study/`.

## Source of reported results

- Frozen configuration: `../experiments/configs/calibration_corrected_study.yaml`
- Selector outputs: `../experiments/outputs/calibration_corrected_study/selector/`
- Verifier outputs: `../experiments/outputs/calibration_corrected_study/verifier/`
- Robustness outputs: `../experiments/outputs/calibration_corrected_study/robustness/`
- Reporting bundle: `../research/calibration_corrected_study/final_results.json`
- Reachability analysis: `../research/calibration_corrected_study/validation_reachability.csv`
- Partition-level selector analysis: `../research/calibration_corrected_study/partition_selector_results.csv`
- Absolute robustness analysis: `../research/calibration_corrected_study/robustness_absolute.csv`
- Manuscript readiness audit: `../research/calibration_corrected_study/manuscript_readiness.md`

The official test annotations remain unopened. The paper describes a controlled development study and does not present a formal test claim.
