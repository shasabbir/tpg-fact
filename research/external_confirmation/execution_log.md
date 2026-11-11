# Independent External Confirmation Execution Log

Date started: 2026-11-11
Protocol commit: `95d3389`
Checkpoint before execution: `b79ae39` (`v4`)

## Current status

The implementation and access audits are complete enough to run once the
official training knowledge-store text archive is available. The outcome-bearing
train-only fit and locked development comparison have not been run. No test
annotation has been opened, no test prediction has been generated, and no
leaderboard submission has been made.

This distinction is intentional: a completed software path is not an external
confirmation result.

## Completed work

- pinned the official dataset, baseline, and shared-task code revisions;
- downloaded and audited the released training and validation annotations;
- downloaded the released claim-image archive and verified annotation coverage;
- implemented deterministic identifiers, source normalization, grouped training
  roles, label-blind ranking output, post-freeze gold joining, calibrated
  selectors, claim-aligned bootstrap intervals, and locked-output guards;
- prevented gold questions from entering candidate features;
- enforced the official maximum of ten evidence items and the corrected
  1,500-character evidence limit;
- added a hidden-test field guard and kept the published test archive unopened;
- passed all 154 tests in the full project suite after the implementation changes; and
- audited the pinned official semantic evaluator without replacing it with an
  easier judge.

## Data observations

The released training split contains 793 claims and the validation split 152.
The verdict distribution is highly imbalanced: training contains 756 Refuted,
18 Not Enough Evidence, 13 Supported, and 6 Conflicting Evidence/Cherrypicking
claims; validation contains 141, 6, 4, and 1 respectively. Any verdict result
must therefore report macro and class-wise scores, not accuracy alone.

The annotation audit found 2,270 training questions and 432 validation
questions. Three training claims have no web-source URL after excluding
metadata pseudo-sources. These claims remain in the access audit but cannot
contribute a positive exact-source target.

## External access condition

The organizer distributes the static knowledge store through Google Drive.
Parallel downloads reached approximately 42% of the training image archive and
68% of the validation image archive before the host began returning its public
``Quota exceeded'' page for every file in the shared folder. The incomplete
files and transfer-control records were preserved; they were not treated as
valid archives.

As an audit-only fallback, the public `humane-lab/AVerImaTeC-Filled` Hugging
Face release was downloaded. Its validation text archive contains both
text-related branches (306 members, 5,193,221,644 uncompressed bytes). It is a
filled store released by a participating team, not the organizer's original
training store, so it is not silently substituted into the prespecified
matched train/development experiment.

The label-blind adapter produced 129,475 validation candidate rows (851.81 per
claim) from this fallback and hashed the pre-join rankings before exact-source
labels were attached. The full pool reaches 92.94% of normalized annotated
sources, completely reaches 84.21% of claims, and has two zero-reachable
claims. Claim-text lexical ranking is much weaker at shallow depths: mean
exact-source recall is 4.82% at 10, 8.00% at 20, 13.21% at 50, and 19.48% at
100. The declared 100-text/20-image-related packet reaches 61.11% recall but
only 0.0217 source-set F1 because it returns 109.10 unique sources on average.
These are access and reachability diagnostics for a participant-filled store,
not results from the frozen selector protocol and not the official semantic
evidence score.

## Official evaluator feasibility

The pinned evaluator uses `google/gemma-3-27b-it` in bfloat16. The current model
revision contains 54,864,980,440 bytes of weight files. The local workstation
has one RTX 4090 with 25,756,696,576 bytes of VRAM, whereas the organizer's
launcher requests two GPUs and 50 GB of host memory. The published evaluator
therefore does not fit on the local GPU as specified. CPU offload or
quantization would change the execution condition and has not been presented
as an official score.

## Completion boundary

Execution can proceed without changing the protocol after the official
training text archive is accessible. The required order is:

1. hash and extract the complete official training text store;
2. build and hash label-blind rankings;
3. join training gold sources only after the ranking hash exists;
4. freeze all train-only models and selection parameters;
5. run the locked 152-claim development comparison once;
6. report exact-source diagnostics separately from the unavailable official
   semantic judge; and
7. update the canonical manuscript only with checked results.

The blind test and leaderboard remain outside this execution boundary.
