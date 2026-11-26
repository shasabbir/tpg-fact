# One-shot manuscript package verification

The source bundle at `paper/publication-one-shot-source.zip` was extracted
into a fresh directory. Its clean build produced 18 pages and matched the
canonical PDF's extracted text byte-for-byte. The final LaTeX log had zero
overfull boxes or undefined citations and references. All pages were visually
reviewed.

| Artifact | SHA-256 |
|---|---|
| `paper/output.pdf` | `74c5cb5cd81cf4618ca5c62b66a2d9e288940ba37e0e697ed856d0889d351b60` |
| `paper/publication-one-shot-source.zip` | `65a4e2b0e944621daf11381622115ca88b1ef5ad70275d73b88cf2288598f7dd` |
| `paper/figures/one_shot_external_diagnostics.pdf` | `e4b2ff1df79ff9a18b08114d39c701972159351454df113ebc420a4b2cabe179` |

All 163 tests and bytecode compilation passed. The rendered bibliography
audit reports 52 verified records. The report and figure replays matched
their working-copy hashes. The archive contains editable LaTeX, figures,
tables, and the aggregate external diagnostic supplement. Venue formatting
and author declarations remain open.
