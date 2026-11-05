# Phase-3 Diagnostic Robustness Protocol

Frozen: 2026-11-03, after the primary Phase-2 result was known.

Status: exploratory mechanism analysis only. These interventions cannot change the
preregistered retrieval, selector, verifier, or formal-test decisions.

The saved validation candidate packets and calibrated probabilities are perturbed
without refitting. Each seed retains its frozen fixed budget, threshold, and
RC-CEFES penalties. Complete gold remains unchanged, so removed or duplicated
evidence is scored as unavailable or irrelevant exactly as it would be end to end.

Declared conditions:

- unchanged 60-item packet;
- packet widths 10, 20, and 40 by frozen final retrieval rank;
- text-only and image-only packets;
- deterministic removal of 25% and 50% of retrievable gold candidates;
- promotion of five existing non-gold candidates to the highest membership scores;
- duplication of the three highest-scoring candidates under new non-gold IDs while
  preserving source groups; and
- rank-preserving probability temperatures 0.5 (sharper) and 2.0 (flatter).

Reported methods are fixed, calibrated threshold, original CEFES, and RC-CEFES.
The diagnostic endpoint is complete-gold claim-level evidence F1, with a
claim-aligned five-seed bootstrap for RC-CEFES minus fixed. No condition is chosen
or removed based on its outcome.
