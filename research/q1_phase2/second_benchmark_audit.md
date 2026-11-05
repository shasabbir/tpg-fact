# Second-Benchmark Access and Compatibility Audit

Audit date: 2026-11-03

Decision: **no candidate qualifies for an unchanged confirmatory replication of
the MOCHEG evidence-set protocol.** The external-replication gate therefore
fails closed. No external test labels were used, and the paper must be framed as
a single-benchmark development study.

## Requirements

The frozen plan requires a research-compatible license, identifiable text and
image evidence, stable corpus access, released evidence identifiers, enough
disjoint supervision for fitting/calibration without test leakage, and a task
that supports the same complete-gold evidence-set metrics.

## AVerImaTeC

- Official dataset page: https://fever.ai/dataset/averimatec.html
- Dataset revision audited: `Rui4416/AVerImaTeC` at
  `7e77fb26a1ba86159ff5ba27b59f373b450d6fd0`.
- License: CC BY-NC 4.0.
- Strengths: 1,297 real-world image--text claims; released training and
  development annotations; question--answer evidence with source URLs and
  image-related fields; an evidence collection is advertised.
- Blocking incompatibilities: the official page still labels the blind test as
  TBD; the Hugging Face repository exposes train/development images but no
  usable annotated blind test; evidence is represented as question--answer
  support and open-web source URLs rather than a stable shared corpus of text
  and image item identifiers; and the official retrieval pipeline depends on
  an externally collected Google Search evidence store. Converting QA answers
  and URLs into MOCHEG-style atomic evidence items would define a new task and
  require a separately frozen protocol.

Result: licensed and relevant, but not a valid unchanged confirmatory benchmark.

## MMCV

- Official repository: https://github.com/mmcv-dataset/MMCV
- Repository revision audited: `904c1cd2101144fea69901da808368ea7d33e069`.
- License: CC BY-SA 4.0.
- Strengths: roughly 15,000 multi-hop claims with text, image, and table
  evidence identifiers linked to MultiModalQA; the release explicitly includes
  annotated evidence and describes evidence retrieval.
- Blocking incompatibilities: the public release is organized by hop count, not
  by an official train/calibration/development/test split; the paper's reported
  open-book evaluation supplies gold evidence rather than a label-blind
  candidate packet; reconstructing the global corpus requires the upstream
  MultiModalQA text/table/image resources; and tables are a required third
  modality not modeled by the frozen MOCHEG scorer or selector protocol. A
  deterministic split and table encoder would be a new benchmark adaptation,
  not an unchanged replication.

Result: the closest future replication candidate, but it requires a separately
preregistered three-modality protocol and cannot serve as fresh confirmation in
this completed run.

## Other candidates

Chart-centered datasets and multimodal claim-detection datasets were excluded
because they either provide the claim image itself rather than an external
retrieval corpus, or target check-worthiness/verdict prediction without atomic
gold evidence identifiers. They do not support the same evidence-set endpoint.

## Consequence

The completed study reports MOCHEG only. A future external study should freeze
an MMCV-specific corpus reconstruction, table representation, official or
hash-locked split, modality-aware scorer, and evidence identity mapping before
viewing its held-out outcomes. AVerImaTeC should be reconsidered when a stable
blind test and immutable evidence collection are publicly available.
