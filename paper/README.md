# Manuscript

`paper.tex` is the IEEE journal-mode main document. Section prose is kept in `sections/`, generated result tables are in `tables/`, figures are in `figures/`, and citations are in `references.bib`.

Formal values are generated from `../dacs-scv-formal-outputs.zip` by `../research/build_formal_paper_assets.py`. The independent audit is implemented in `../research/audit_formal_outputs.py`; its checked findings are recorded in `../research/formal_results_audit_2026-10-28.md` and the claim--evidence register. Do not edit generated numerical tables by hand.

Compile locally from this directory with:

```powershell
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

The built-in Codex LaTeX compiler is used after material manuscript changes.
