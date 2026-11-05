# Manuscript change record

## 2026-11-04 enrichment

- Installed and applied the `research-paper-workflow` review, writing, methodology, evidence, reference, and delivery criteria.
- Reframed the title and abstract around the evaluated task, retrieval boundary, calibrated decisions, and bounded conclusion.
- Replaced the oversized author block with a compact IEEE journal author line and affiliation footnote.
- Rebuilt the architecture figure as a linear, editable TikZ data-flow diagram with explicit training and validation boundaries.
- Added an evidence-derived three-panel study summary covering reachability, policy performance, and the review-cost tradeoff.
- Added a post hoc reachability-stratified analysis from preserved candidate and claim records, plus tests that verify packet strata are invariant across deterministic partitions.
- Expanded the experimental design with the post hoc status, software environment, and deterministic execution details.
- Updated results, discussion, limitations, abstract, and conclusion to state that the expected-F1 gain is concentrated in complete-reachability claims.
- Added an online metadata reference audit. Human claim-support review and strict-policy conflicts remain explicit rather than being marked complete.
- Added a research brief, claim-evidence register, readiness assessment, and remaining-action list.

No formal test annotation was opened, no measured result was replaced by an estimate, and no unfavorable result was removed.

## 2026-11-04 depth restoration

- Compared the revised manuscript against `paper/output.pdf` page by page and used its scholarly density, comparison structure, and appendix depth as the presentation benchmark.
- Restored a six-part related-work taxonomy and expanded the comparison matrix from four to nine external systems plus the present study.
- Expanded the cited literature from 26 to 55 works while keeping citations tied to specific technical themes rather than adding an uncited bibliography quota.
- Added explicit scorer, calibration, and exact expected-F1 utility equations together with the decoder assumptions and decision scope.
- Added the full seven-retriever audit matrix and five-partition selector table to the main results.
- Added absolute robustness scores and selected-set sizes alongside paired effects so large relative changes cannot hide weak absolute operating points.
- Moved DOI-less model-card provenance to pinned operational footnotes, eliminating a nearly blank final reference page while retaining exact resource URLs.

All additions use the existing frozen retrieval, selector, verifier, and robustness records. No new empirical value was inferred from the reference manuscript.

## 2026-11-05 float placement correction

- Audited all 14 rendered pages against the source order and first in-text citations.
- Prevented Results floats from drifting into Discussion and Limitations by flushing the Results float queue before the section transition.
- Reordered the robustness figure declaration within Results and enlarged the full-width rendering so the final Results float page remains balanced and legible.
- Flushed appendix tables before the bibliography so Tables XI--XIV now precede the References section.
- Rebuilt the PDF and source archive; both the main and clean extracted builds contain 14 pages with no undefined-reference, overfull-box, or underfull-box warnings.

No prose, reported number, caption text, table content, citation, or scientific conclusion was changed.

## 2026-11-05 policy-decoding diagram

- Added one editable TikZ diagram to the Method section, where it is first needed.
- The diagram compares fixed top-5, calibrated thresholding, learned count, and expected-F1 on the same calibrated candidate ranking.
- It explicitly separates the changing set-decision rule from the shared candidate identities, order, and probabilities, and is labeled as procedural rather than measured.
- Recompiled and visually inspected all 14 pages; the new diagram remains legible at final column width and does not disturb the Results, appendix, or reference boundaries.
- Refreshed and clean-built the source archive with the new editable figure source included.

No experimental value, method behavior, result interpretation, or scientific conclusion was changed.

## 2026-11-05 prose humanization and author metadata

- Reviewed the abstract and all manuscript sections for staged contrasts, repetitive closers, inflated phrasing, and document-centered narration.
- Rewrote only the passages with clear style patterns; technical contrasts, qualifications, citations, equations, labels, and numerical results were preserved.
- Added the author details verified from the supplied reference PDF: Sabbir Hossain and Dr. Firoz Ahmed, with Firoz Ahmed identified as a professor and faculty member in the Department of Computer Science at American International University-Bangladesh.
- Confirmed exact pre/post agreement for 347 numerical tokens, 67 citation keys, 27 labels, and 17 cross-references.
- Recompiled and inspected all 14 pages. The expanded author block is legible, and no figure, table, appendix, or reference placement was disturbed.
- Refreshed the delivery PDF and clean-built the source archive without undefined-reference, overfull-box, underfull-box, or package warnings.

No experimental value, method behavior, result interpretation, citation, or scientific conclusion was changed.

## 2026-11-05 visible author block correction

- Replaced the compact author line and bottom-of-page affiliation footnotes with a visible two-column author block directly beneath the title.
- Transcribed the reference author information without paraphrasing: Sabbir Hossain; Dr. Firoz Ahmed; Professor, Faculty; Department of Computer Science; American International University-Bangladesh; Dhaka, Bangladesh; and both email addresses.
- Recompiled and visually inspected all 14 pages. The author block is centered and legible, and the changed title-page height does not create clipping, overlap, or misplaced floats elsewhere in the paper.
- Refreshed and clean-built the delivery PDF and source archive with no LaTeX, package, reference, overfull-box, or underfull-box warnings.

No scientific prose, experiment, result, citation, table, figure, or conclusion was changed.
