# Main paper

The canonical manuscript is `paper.tex`; the compiled reading copy is `output.pdf`.
The archived v1 build is `publication-v1-source.zip`. See `v1.md` for the release scope.
The current manuscript uses the Springer Nature double-column template for
the *International Journal of Multimedia Information Retrieval* (IJMIR).
It reports a retrieval-conditioned, calibration-aware evidence-selection study
on a locked MOCHEG population and in one independently frozen AVerImaTeC
organizer-store evaluation. Both formal-test annotation sets remain sealed.

## Build

From this directory:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
bibtex paper
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
```

Use an up-to-date LaTeX distribution with the AMS, appendix, and sttools
packages available. The build uses the included Springer Nature `sn-jnl.cls`
and `sn-basic.bst`, generated tables in `generated/`, PDF figures in `figures/`,
and the 52-entry cited bibliography in `references_used.bib`. The diagrams
are made with LaTeX and TikZ. The results plots are made with Matplotlib. The larger
`references.bib` remains the literature catalog. The final rendered-reference
audit for v1 is under `../research/external_confirmation/v1_reference_audit/`.

## IJMIR source bundle

`ijmir-source.zip` contains one editable manuscript, the cited bibliography,
the publisher class and bibliography style, and seven figures as `Fig1.pdf`
through `Fig7.pdf`. Build the extracted bundle with the commands above.
To recreate the bundle, export to a new or empty folder and archive its contents.

```powershell
python build_ijmir_source.py --out ../tmp/ijmir-upload
Compress-Archive -Path ../tmp/ijmir-upload/* -DestinationPath ijmir-source.zip -Force
```

Confirm both authors' contributions and the reviewer access arrangement
for the study code and supporting result files before submission.

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
- External freeze manifest: `../experiments/outputs/external_confirmation/averimatec/train_only.freeze.json`
- Locked external records: `../experiments/outputs/external_confirmation/averimatec/locked_development/claim_records.jsonl`
- Independently recomputed external report: `../experiments/outputs/external_confirmation/averimatec/reporting/external_results.json`
- External result narrative and artifact hashes: `../research/external_confirmation/results.md`
- External calibration, target loss, and policy disagreement: `../research/external_confirmation/one_shot_diagnostics.json`
- Regeneration instructions, evidence map, and venue checklist: `../research/external_confirmation/one_shot_delivery.md`

The claims are scoped to the reported development populations and frozen
evidence-selection protocols. AVerImaTeC reports exact-source diagnostics
and class-wise verifier results. Full Gemma semantic evaluation is deferred.
