# Active manuscript

paper.tex is the IEEE journal-mode negative development manuscript for:

**Rethinking Evidence Selection in Multimodal Fact-Checking: Calibrated Adaptive Set Optimization Beyond Fixed Top-\(k\) Retrieval**

The source is intentionally synchronized with the active CEFES/MOCHEG study in
../plan.md. Section prose is in sections/, the editable method diagram is
figures/architecture.tex, and verified citations are in references.bib.

Current scientific status:

- data-access, sentence-recovery, media, and 400-claim packet audits pass;
- the exact expected-F1 decoder and calibration path are implemented and tested;
- the pinned BGE-M3/SigLIP2 full-corpus development run is checked and negative;
- the DACS control, matched verifier, and clean reproduction are checked;
- the complete Section 13 gate fails the two primary evidence-effect conditions;
- the official MOCHEG test remains sealed;
- no formal-test or superiority claim is available.

The old SciVer/DACS manuscript remains recoverable from Git history and its
audited archives. It is not the active paper.

Compile from this directory with:

    pdflatex paper.tex
    bibtex paper
    pdflatex paper.tex
    pdflatex paper.tex

After a material change, use the built-in LaTeX compiler and inspect the
rendered PDF. Never manually insert final result values: generate tables and
figures from frozen experiment artifacts after independent recomputation.
