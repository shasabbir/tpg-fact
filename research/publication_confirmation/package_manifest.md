# Publication foundation package

- Archive: `paper/publication-foundation-source.zip`
- Archive SHA-256: `F662CA3D72D0BAF1F81B99B9082E9BE398C0869318E4C1174E160002435CCFAB`
- Canonical PDF: `paper/output.pdf`
- PDF SHA-256: `DE638A175F14EA5E34B5B1BE021AEB94CF75E6C0879F58AE623A9D217435D03A`
- PDF pages: 18

## Included source

- `paper.tex`, `README.md`, `references_used.bib`, and `output.pdf`
- all manuscript files under `sections/`
- all generated tables under `generated/` required by the manuscript
- all code-native diagrams and PDF/PNG figure sources under `figures/`,
  including the external-confirmation summary

## Clean build check

The archive contents were copied to
`tmp/package-build-20261112-final/` and built with `pdflatex`, `bibtex`, and
repeated `pdflatex` passes. The clean build produced 18 pages with no undefined
citations or references, no overfull boxes, and no LaTeX warnings on the final
pass.

This is a complete publication-foundation source package, not a
venue-specific submission archive. Venue declarations, anonymity settings,
and page-limit conversion remain author decisions.

