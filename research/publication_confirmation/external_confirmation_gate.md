# External Confirmation Gate

Date: 2026-11-09  
Decision: **not run; gate remains closed**

## Why the gate is closed

The publication-confirmation protocol does not authorize validation outcomes to
unlock the formal test. A one-shot formal-test run additionally requires
explicit author authorization. No such authorization has been given in this
execution, and the experimental prerequisites do not justify requesting an
exception:

- the selector-superiority gate failed;
- the completeness-aware switch improved evidence F1 by only +0.00138 over
  calibrated thresholding, with a 95% interval crossing zero and one negative
  partition;
- aggregate verifier non-inferiority passed, but only two of five seed-specific
  intervals cleared the frozen margin;
- the existing benchmark audit found no external dataset that supports an
  unchanged replication of the atomic multimodal evidence-set protocol.

The sealed MOCHEG formal-test annotations therefore remain unopened. The
publication-confirmation results are controlled development evidence on the
released validation population, not independent confirmation.

## Existing external-benchmark audit

The 2026-11-03 audit considered AVerImaTeC and MMCV. AVerImaTeC lacks a usable
annotated blind test and a stable shared atomic evidence corpus compatible with
the present packet. MMCV is the closest candidate, but it requires upstream
corpus reconstruction, a table modality, and a new hash-locked split. Adapting
either dataset would define a new protocol rather than replicate the frozen
one. See `research/q1_phase2/second_benchmark_audit.md` for the full audit.

## Publication consequence

The manuscript must:

1. describe the work as a single-benchmark controlled development study;
2. avoid generalization, state-of-the-art, and independent-confirmation claims;
3. report both the modest gains and the failed selector/per-seed safeguards;
4. identify an MMCV-specific preregistered replication as future work; and
5. state that formal-test annotations were not accessed.

The formal test can be reconsidered only after a new frozen decision record and
explicit author authorization. No result in this run should be retuned to make
that decision easier.

