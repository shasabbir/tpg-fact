# Independent External Confirmation Execution Log

Date started: 2026-11-11
Protocol commit: `95d3389`
Checkpoint before execution: `b79ae39` (`v4`)

## Current status

The organizer training and validation knowledge-store archives are now
complete, hash-verified, and extracted. The full official training store has
passed the pinned multimodal retrieval path, and all five train-only models and
selection decisions are frozen in an immutable manifest. One locked
development comparison has completed, its aggregates were independently
recomputed, and the canonical manuscript is being synchronized. The official
semantic judge remains unavailable. No test annotation has been opened, no
test prediction has been generated, and no leaderboard submission has been
made.

## Official-store resolution and train-only freeze

The manually supplied organizer archives resolved the data-access blocker.
Their identities are:

- training image store: 12,901,641,633 bytes,
  SHA-256 `4b4f3dcdc2cf12594403e6c7097674645d4254be06fab44924b1051e32b92d9b`;
- training text store: 6,716,122,829 bytes,
  SHA-256 `6c4e8e2c92e3c352e61c0ae447da0da619e18c8e9c33f269ddfee7310eba6eb9`;
- validation store: 9,445,438,165 bytes,
  SHA-256 `80dcf65fc91a586fd8a8ac171dd5e632c2465c207118dd419e29be6e07b5013c`.

Extraction reproduced the archive manifests exactly: training contains 1,586
JSON files and 81,817 JPEG images; validation contains 304 JSON files and
49,617 JPEG images. The participant-filled fallback extraction was moved out
of the organizer-store root before preparation, preventing source mixing.

The first full preparation attempt exposed a parser error rather than archive
corruption. Python's `str.splitlines()` treated U+2028/U+2029 characters inside
valid scraped JSON strings as JSONL boundaries. The reader now splits only on
physical LF delimiters, reports exact source locations for malformed records,
and has a regression test covering the organizer-store condition. No records
were removed or repaired.

The corrected official-train run ranked 529,580 text and 81,817 image
candidates for all 793 claims, retaining 79,247 text and 14,838 image rows.
The label-blind pre-join artifact contains 94,085 rows and has SHA-256
`2b4353ba7019926bd4913a47139a7d1a2fc9fe0d4e7cc2362c593056f185a8a2`.
Gold targets were joined only after that file was written and hashed.

Exact-source reachability remains a major empirical limitation: across 792
training claims with web-source gold, the retained pool reaches 22.47% of
targets, completely reaches 9.22% of claims, and has zero exact reachability
for 61.36% of claims. These are diagnostic normalized-source identities, not
the official semantic evidence score.

The five-seed train-only freeze completed without loading development rows.
Every seed selected a 50-text/10-image packet; fixed budgets, thresholds,
calibrators, completeness models, and shared verifiers were selected only from
the declared training roles. The freeze manifest has SHA-256
`36e4210a438a9c6c0051d6c4608b9d2fc04709aceaecfe5c9a694518bfa0e314`,
names clean code revision `218dbf1ac88630a06d5ad3cce3e1cacaeb84ddcc`,
and records `development_outcomes_accessed: false`. The implementation now
rejects mixed-split freeze inputs and accepts only validation-only artifacts in
the locked runner. After the reporting and export additions, the full suite
contains 163 passing tests and bytecode compilation succeeds.

## Completed work

- pinned the official dataset, baseline, and shared-task code revisions;
- downloaded and audited the released training and validation annotations;
- downloaded the released claim-image archive and verified annotation coverage;
- implemented deterministic identifiers, source normalization, grouped training
  roles, label-blind ranking output, post-freeze gold joining, calibrated
  selectors, claim-aligned bootstrap intervals, and locked-output guards;
- implemented claim-local BM25, pinned BGE-M3 dense ranking, equal-weight
  reciprocal-rank fusion, and the pinned BGE reranker;
- corrected the evidence model so reverse-image-search page text remains text
  evidence and actual candidate-image files are loaded as image evidence;
- implemented pinned SigLIP2 plus filename-BGE image ranking, genuine multimodal
  packet identities, and separate exact URL/image diagnostic targets;
- implemented the train-only completeness estimator and switch, a balanced
  linear verifier control, a deterministic two-layer four-class verifier, and
  a development-only official-schema exporter;
- added Bonferroni simultaneous bootstrap intervals alongside Holm-adjusted
  p-values for the four selector comparisons;
- prevented gold questions from entering candidate features;
- enforced the official maximum of ten evidence items and the corrected
  1,500-character evidence limit;
- added a hidden-test field guard and kept the published test archive unopened;
- passed the complete 163-test project suite after the final reporting changes; and
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

## Resolved external access history

The organizer distributes the static knowledge store through Google Drive.
Parallel downloads reached approximately 42% of the training image archive and
68% of the validation image archive before the host began returning its public
``Quota exceeded'' page for every file in the shared folder. The incomplete
files and transfer-control records were preserved; they were not treated as
valid archives.

On 2026-11-11 the canonical folder was queried again and still listed the
organizer's train text, train image, validation, and test archives. A direct
retry of the required train text file failed with Google's ``Too many users
have viewed or downloaded this file recently'' response. The test archive was
not opened.

A public Hugging Face registry search found only the official annotation/image
release, the participant-filled validation/test release, and a claim-image
reproduction. None contains the organizer's training knowledge store. A final
authenticated-download attempt could not proceed: Chrome's current Windows
cookie protection prevented CLI cookie import, no Firefox profile existed, and
local browser control was disabled in this Work environment. The user-supplied
organizer archives later resolved this gate; the earlier partial files remain
preserved only as an access-history record and were never treated as valid.

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

## Real-model software pilot

The pinned BGE-M3 and BGE reranker completed a CUDA pilot on two validation
claims from the participant-filled fallback store. The label-blind stage read
1,949 text candidates, retained 20 per claim, wrote and hashed 40 pre-join rows
(`65f44e58ef8c765e10b599dcd8a9cde20117e608e646a368474e2537e30c61bd`),
and joined gold targets only afterward. The pilot was repeated after enforcing
a sanitized pre-join annotation view and reproduced the same hash. One of the
two claims had no reachable target in the retained packet. This tiny, selected
software pilot is not a scientific result and is not used in the paper.

The pinned SigLIP2 model also encoded one released claim image on the RTX 4090,
producing aligned 768-dimensional text and image vectors. At that checkpoint
this validated model loading only; the later organizer-store run completed the
full image path and supersedes this pilot as scientific evidence.

## Official evaluator feasibility

The pinned evaluator uses `google/gemma-3-27b-it` in bfloat16. The current model
revision contains 54,864,980,440 bytes of weight files. The local workstation
has one RTX 4090 with 25,756,696,576 bytes of VRAM, whereas the organizer's
launcher requests two GPUs and 50 GB of host memory. The published evaluator
therefore does not fit on the local GPU as specified. CPU offload or
quantization would change the execution condition and has not been presented
as an official score.

## Pre-run completion boundary (satisfied)

The locked run followed this authorized order:

1. prepare validation-only label-blind rankings after the committed train-only
   freeze;
2. hash those rankings before joining development gold sources;
3. run the locked 152-claim development comparison exactly once;
4. report exact-source diagnostics separately from the unavailable official
   semantic judge; and
5. update the canonical manuscript only with checked results.

The blind test and leaderboard remain outside this execution boundary.

## Locked development completion

Completed on 2026-11-12. Validation-only preparation ranked 101,289 text
candidates and 49,617 actual images for 152 claims. The 18,240-row label-blind
pre-join artifact has SHA-256
`b24739c9e85763561cd248eae7b572f1dac2cce0956cfc578220c615bfe719c0`.
The exact-source audit found 25.48% target recall, 12.50% complete claim
reachability, and 59.87% zero-reachable claims. One 144-million-pixel organizer
image triggered a PIL safety warning but was processed without exclusion or
fallback.

The locked comparison ran once and produced 3,800 claim-policy-seed records.
Independent recomputation matched all selector aggregates, four paired
bootstrap comparisons, and shared-verifier results. Calibrated expected-F1
beats its uncalibrated ablation by +0.05316 exact-source F1 with a Bonferroni
simultaneous interval [+0.02207,+0.08852]. Thresholding does not beat the fixed
family (+0.00003 [-0.02341,+0.02404]). Expected-F1 exceeds thresholding by
+0.01714 [-0.00148,+0.03905] but selects 3.20 rather than 1.17 items, so the
compactness requirement fails. The switch gains +0.01555
[-0.00263,+0.03621] and selects 2.71 items, so its superiority gate fails.

Shared-verifier macro-F1 averages 0.1573 for threshold evidence and 0.1444 for
switch evidence. The switch-minus-threshold effect is -0.01293 with 95%
interval [-0.03242,+0.00687], failing the -0.010 non-inferiority safeguard.
The frozen blind rule therefore selects the train-selected fixed policy family.
The official semantic judge remains unavailable, and no blind-test prediction
or submission was generated.

The first fixed-family development export failed because verdict predictions
had been attached only to threshold and switch rows. Deterministic inference
from the already frozen shared verifier was added without refitting or changing
evidence. All five local development exports then passed the official schema;
they remain unsubmitted. Full details and evidence hashes are recorded in
`research/external_confirmation/results.md`.
