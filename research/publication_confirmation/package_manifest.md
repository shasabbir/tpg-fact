# Publication foundation package

- Archive: `paper/publication-foundation-source.zip`
- Archive SHA-256: `DEA274F76091382D8898762B311115EEE229310BD39B6BD25EFF040151241E43`
- Canonical PDF: `paper/output.pdf`
- PDF SHA-256: `E417CEE5464EBD76DCE65891BB0B5537109DAF8E81331E57CA446998B4051180`
- PDF pages: 14

## Included source

- `paper.tex`, `README.md`, `references_used.bib`, and `output.pdf`
- all manuscript files under `sections/`
- the nine generated tables under `generated/` required by the manuscript
- `architecture.tex`, `policy_decoding.tex`, and the three PDF figures required
  by the build

## Clean build check

The archive contents were copied to
`tmp/package-test-20261111-final/` and built with `pdflatex`, `bibtex`, and
repeated `pdflatex` passes. The clean build produced 14 pages with no undefined
citations or references, no overfull boxes, and no LaTeX warnings on the final
pass.

This is a complete publication-foundation source package, not a
venue-specific submission archive. Venue declarations, anonymity settings,
and page-limit conversion remain author decisions.

