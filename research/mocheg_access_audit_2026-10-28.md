# MOCHEG Access and Mirror Audit

Audit date: 2026-10-28

Status: **development metadata and sentence-candidate identity passed; full media preflight remains open; official test content is not downloaded.**

## Authoritative sources

- Paper DOI: `10.1145/3539618.3591879`
- Official repository: `https://github.com/PLUM-Lab/Mocheg`
- Official archive directory: `http://nlplab1.cs.vt.edu/~menglong/project/multimodal/fact_checking/MOCHEG/dataset/latest_dataset/`
- Official archive: `mocheg_with_tweet_2023_03.tar.gz`
- Dataset license: CC BY 4.0
- Code license: Apache 2.0

The official server reported:

| Field | Value |
|---|---|
| Bytes | 74,304,681,291 |
| Approximate size | 69.2 GiB |
| Last modified | 2023-12-25 19:07:51 GMT |
| ETag | `114ce7754b-60d5a484602f7` |
| Byte ranges | supported |

The archive is authoritative but operationally inconvenient: it is one gzip stream, the local drive cannot safely retain both a 69.2 GiB archive and a complete extraction, and the observed server transfer rate was much lower than the mirror rate. It is retained as a recovery source, not the default development transport.

## Pinned transport mirrors

The following public Hugging Face repositories are unofficial transport mirrors. Their contents must be reconciled with official counts and identifiers before use. Their repository-card `MIT` license tag is not treated as authoritative; the official dataset's CC BY 4.0 terms govern the data.

| Role | Repository | Pinned revision | Reported storage | Files |
|---|---|---|---:|---:|
| Train | `vcmt794/mocheg-train` | `985e4206cf9990ff0a875473616be204cfbd09c3` | 14.084 GiB | 78,017 |
| Development | `vcmt794/mocheg-val` | `9dabf73ad41ddecafb3b525fb82a3de018e3e378` | 2.826 GiB | 12,267 |
| Formal test - sealed | `vcmt794/mocheg-test` | `a72342a67b6bb5e930f8a2aa5fc3b051690c0f74` | 5.369 GiB | 22,779 |
| Shared corpus | `tonyvu1289/mocheg` | `d9b15bf2b7d2be790c40efa960e8f05a58487ce2` | 2.224 GiB | 9,994 |

The test repository identity is recorded, but its claim JSON, qrels, labels, explanations, and gold evidence are not present locally and must remain unavailable to development profiles.

## Local development metadata

Only train/development metadata and the shared unlabeled text corpus were downloaded under `experiments/data/public/mocheg/metadata/`.

| File | Bytes | SHA-256 |
|---|---:|---|
| `train/train.json` | 71,445,924 | `B39B1E43735E2D09D9C28A34801A698F4D0EC2218ADE0EF2C31B8F8A8055307E` |
| `train/img_evidence_qrels.csv` | 5,186,489 | `F8AE493E0971C435D4B44D35C9D1418BB3C8DB4F29E8F972C7068F24490EF37C` |
| `train/text_evidence_qrels_article_level.csv` | 401,806 | `5F0D597E56E333203F423A3FA5815D2DFE06DB3C61985DCE205CA782A29FFC00` |
| `train/text_evidence_qrels_sentence_level.csv` | 8,419,098 | `7E214536B4292DDB84A83F7FE357CC0E82BDA0BA6BA6129177A08DFE297013D7` |
| `val/train.json` | 26,836,873 | `600F9D892C32A66266A0E685242C8900C2F2BABB14EEF37857533977383E27E0` |
| `val/img_evidence_qrels.csv` | 773,648 | `5627DA6E028580EAF6DB6AD5AD23B2CB049014586B32283D3F29DB9FCBA0BDEA` |
| `val/text_evidence_qrels_article_level.csv` | 87,723 | `25CBE802645284B9827F39B6F3A85898DF072CA1ABDA20724E4E4C4D4DF79A38` |
| `val/text_evidence_qrels_sentence_level.csv` | 1,665,188 | `94A5633B3A5685799BF9C0F4C3FD7933EF21A1334B7812823C6F89BF80FD623F` |
| `corpus/Corpus3.csv` | 409,258,619 | `7C32AF7BCF56060360914EF74D756EE9DF55EA420F55280961304C71FB8F2CC7` |

## Identity results

| Check | Train | Development | Expected | Result |
|---|---:|---:|---:|---|
| Unique claims | 11,669 | 1,490 | 11,669 / 1,490 | pass |
| Text evidence | 23,545 | 4,067 | 23,545 / 4,067 | pass |
| Unique gold-image evidence IDs | 8,927 | 1,178 | 8,927 / 1,178 | pass |
| Conflicting verdicts per claim | 0 | 0 | 0 | pass |

`Corpus3.csv` contains 91,822 rows, exactly matching the documented text-corpus size. It has 83,581 unique `relevant_document_id` values, 15,642 represented claim IDs, and no empty `Origin Document` cells.

The mirror is not schema-uniform: train JSON is aggregated one row per claim with list-valued evidence fields, while development JSON contains one row per evidence and repeats claim fields. The implemented adapter normalizes both forms into a common claim/evidence schema and rejects inconsistent claim text, labels, ruling outlines, or evidence mappings.

The real preflight exposed two release-quality limitations that are retained rather than hidden:

- 4,556 of the 23,545 training evidence IDs have blank text in the converted JSON, affecting 4,186 claims. None of those claims or evidence IDs appears in the train article/sentence qrels, so these records cannot be silently treated as usable gold text. They remain preserved as missing-content identifiers pending authoritative recovery or a declared exclusion rule.
- `Corpus3.csv` contains 8,221 repeated `(claim_id, relevant_document_id)` rows, including 717 keys whose normalized document text conflicts. Retrieval indexing must assign stable row/content identities instead of overwriting by document ID.

## Sentence-candidate reconstruction

The official repository constructs sentence IDs by removing literal `<p>` tags,
applying NLTK sentence tokenization, numbering retained sentences from zero, and
loading duplicate IDs with last-assignment-wins semantics. The local adapter
reproduces this recipe on demand without materializing unrelated sentences. It
uses NLTK 3.9.2 and records the exact Punkt resource hash.

| Check | Result |
|---|---:|
| Distinct train/development qrel sentence IDs | 378,864 |
| Recovered sentence IDs | 378,864 |
| Missing sentence IDs | 0 |
| Referenced source documents | 32,382 |
| Matched corpus rows | 35,057 |
| Duplicate source rows among referenced documents | 2,675 |
| Conflicting duplicate source rows | 133 |
| Conflicting duplicate sentence assignments | 2,188 |

The audited artifact is
`experiments/data/processed/mocheg_development/sentence_text.jsonl` (95,459,108
bytes; SHA-256
`3313f5d6de46ef38b6bfcb4cfb2b5028d2ca712ad9ad9197b190aa2b0edcb1eb`).
Its tokenizer resource SHA-256 is
`3efaaeecafe626e3462d4d5e800be9a087bd377eb6b29ac570095fb9df68f24e`.
This resolves the qrel-candidate text path; blank converted evidence snippets
outside the sentence qrels remain visible in the audit and are not imputed.

## Open media checks

- The train mirror contains 78,010 image files and the development mirror contains 12,260.
- All development gold-image IDs are represented by available train/development mirror files.
- Three train gold-image IDs are not represented by the train/development mirror filenames and require recovery or transparent exclusion.
- Retrieval qrels reference distractor images associated with other partitions. A development corpus may expose unlabeled, non-proof candidate images from all partitions, but development code must never load test proof images, qrels, labels, explanations, or test metrics.
- Full binary image readability, SHA-256 identities, perceptual duplicates, and missing/corrupt rates remain to be measured after development media acquisition.

## Bounded acquisition smoke test

The profile-aware downloader was tested on deterministic seed 13 with train claims `103679` and `1838` and development claims `4028` and `107199`.

| Check | Result |
|---|---:|
| Required images | 39 |
| Downloaded images | 39 |
| Readable images | 39 |
| Missing images | 0 |
| Unique SHA-256 values | 34 |
| Exact duplicate occurrences | 5 |
| Test annotations downloaded | no |

The smoke acquisition passed. The exact duplicates are expected evidence/candidate copies and demonstrate why duplicate-aware evaluation remains necessary.

## 400-claim development packet

The feasibility packet was expanded deterministically to 200 training and 200
development claims with seed 13. Windows-incompatible repository filenames are
stored under SHA-256-derived local object names while their original repository
paths remain in the manifest.

| Check | Train | Development |
|---|---:|---:|
| Claims | 200 | 200 |
| Qrel candidates | 6,465 | 9,358 |
| Available candidates | 6,243 | 9,046 |
| Unavailable image distractors | 222 | 312 |
| Unavailable text candidates | 0 | 0 |
| Unavailable gold candidates | 0 | 0 |
| Claims without released candidates | 33 | 0 |

All 1,612 downloaded images decode successfully; 1,255 SHA-256 values are
unique and 357 occurrences are exact duplicates. Every one of the 534 absent
filenames is a negative distractor residing outside the pinned train/development
mirrors. They are excluded identically for every selector. No positive candidate,
gold proof reference, or selected source image is missing. The packet gate passes
without accessing formal-test annotations.

## Decision

Use pinned mirrors for reproducible transport and the official server/archive as the authority and recovery fallback. Do not download or extract the 69.2 GiB archive now. The implemented development downloader fetches only pinned metadata and requested train/development image candidates, keeps formal-test annotations absent, and records revisions, hashes, missing assets, readability, and duplicates. If the mirror cannot provide the required candidate corpus or missing gold media within the declared rules, recover only the required assets from the official archive or revise the task before experimentation.
