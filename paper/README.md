# Manuscript

`paper.tex` is the IEEE journal-mode main document. Section prose is kept in `sections/`, the system diagram is in `figures/`, and citations are in `references.bib`.

The current manuscript is deliberately pre-results. Do not replace the Results section with numerical claims unless each value is generated from a frozen experiment artifact and entered in `../research/claim_evidence_register.csv`.

Compile locally from this directory with:

```powershell
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

The built-in Codex LaTeX compiler successfully compiled the manuscript on 2026-10-26.
