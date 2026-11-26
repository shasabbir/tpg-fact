# Publication revision and reproducibility

Checkpoint `e5d31b0` records the revision plan on top of the preserved study (`ae2c9fd`). The external protocol and archived locked results were frozen at `b2075d0` and `11be256`. The revision adds analyses of frozen development data and corrects the manuscript's interpretation of timing, identity, and calibration.

## Evidence register

| Claim or output | Input and check |
|---|---|
| MOCHEG reachability, selector and verifier comparisons | `../publication_confirmation/final_results.json` and the frozen run directories listed in `../../paper/README.md`. The reporting script regenerated the same SHA-256 (`54743e47…`). |
| AVerImaTeC exact-source results | The independent reporting script recomputed 3,800 claim records, the paired bootstrap, and verifier aggregates. Its result matched SHA-256 `0751ce76…`. |
| External calibration and reliability | `one_shot_diagnostics.json`, field `calibration`. Candidate-weighted scores are averaged over seeds. Reliability bins pool candidate–seed rows. The prevalence reference uses development labels only to describe class imbalance. |
| First failure stage and policy disagreement | `one_shot_diagnostics.json`, fields `stages`, `complete_claim_seed_counts`, `verdicts`, and `disagreement`. Two inventory implementations agreed byte-for-byte. One called the original lexical discovery function, and the other checked whether the original text segmenter yielded content. Both checked discovery counts and the nested stage identities. |
| External plot | The repository contains `experiments/scripts/build_one_shot_figure.py` and `paper/figures/one_shot_external_diagnostics.pdf`. |

The derived report SHA-256 is `6edb2a782b889cbf478fb52e7e3552d6d0c93a6958a41b3077d04acca15167f6`. It includes the four frozen input hashes, the annotation hash, and hashes for the indexed validation JSON files. The figure uses percentages of target–seed instances. The reported 25.48% pool recall is averaged over claims, which weights targets differently from the stage counts.

Text-stage accounting uses normalized URLs within each claim's indexed entries. Image-stage accounting uses exact basenames. A record with a usable text segment can enter the ranker. The scorer selects a per-seed packet from the ranked pool using frozen raw scores. Verdict errors are counts at the claim–seed level conditional on observed overlap, since a verdict cannot be assigned to an individual gold identity.

## Regeneration

Use the workstation environment from `experiments/README.md`. The revision ran with Python 3.13.14, NumPy 2.5.4, PyTorch 2.14.1+cu130, Transformers 4.57.6, Accelerate 1.15.0, Matplotlib 3.11.2, Pillow 12.3.0, pytest 9.2.0, and bibtexparser 1.4.4. The frozen model and split identities are in `experiments/configs/external_confirmation/averimatec.yaml`, `experiments/configs/publication_confirmation.yaml`, and the associated freeze manifests.

Acquire MOCHEG from [the benchmark repository](https://github.com/PLUM-Lab/Mocheg) and AVerImaTeC from [the dataset release](https://huggingface.co/datasets/Rui4416/AVerImaTeC) and [organizer repository](https://github.com/abril4416/AVerImaTec_Shared_Task). Use the original extracted train and validation stores and preserve the relative paths in the configuration and execution log. The source archive carries the manuscript and aggregate diagnostics. The frozen claim-level outputs and benchmark content are separate inputs governed by the source terms.

From `experiments/` in the activated environment, choose a fresh output directory for each replay.

```powershell
python -m pytest -o addopts='' -q
python -m compileall -q scripts src
python scripts/build_publication_reporting.py --research-root ../tmp/replay/mocheg/research --generated-root ../tmp/replay/mocheg/tables --figure-root ../tmp/replay/mocheg/figures
python scripts/build_external_reporting.py --output-root ../tmp/replay/external
python scripts/build_external_figure.py --results ../tmp/replay/external/external_results.json --output ../tmp/replay/external_summary.pdf
python scripts/build_one_shot_diagnostics.py --output ../tmp/replay/one_shot_diagnostics.json
python scripts/build_one_shot_figure.py --report ../tmp/replay/one_shot_diagnostics.json --output ../tmp/replay/one_shot_external_diagnostics.pdf
```

Build the manuscript with the commands in `paper/README.md`. Its reporting appendix gives the operating conditions and freeze identifiers. `one_shot_reference_audit/` records the rendered DOI check. The review follows the existing DOI and support ledger and adds one paper on reverse-image retrieval for AVerImaTeC, checked at the proceedings record and the published paper's abstract and figure.

## Semantic evaluator preflight

The pinned organizer code is `f6f361d2fc5a0ab7d3ea282e2beced56b928e1bd`. Its published script targets test by default, and a development adapter would need to use only `val.json` and the frozen output paths. All 294 distinct development reference image names were found by basename in the extracted image directory. This is a path inventory, and scoring remains pending.

On 2026-11-25, the workstation had one RTX 4090 (24,564 MiB VRAM), 63.8 GiB of physical RAM, and roughly 45 GiB of available RAM at inspection. The model revision `005ad3404e59d6023443cb575daa05336842228a` has 54.86 GB of BF16 weights. The pinned configuration request returned HTTP 401, and the Google QAT Q4 weight endpoint (`17cf0f6ad611f1a57a1640daa57eb427d6e67ed6`) also returned 401. No Hugging Face credential was configured. CPU offload feasibility and runtime remain unmeasured. The official endpoint is recorded as unavailable under the current access and hardware conditions. Qwen was left out of this revision because the selected publication route uses the frozen exact-source endpoint.

### Authenticated follow-up

After the user granted access, the pinned configuration downloaded
successfully and the 12 BF16 shards began transferring. The
`experiments/scripts/run_averimatec_semantic_dev.py` preflight matched the
frozen input hashes and assembled 3,800 development outputs, covering 1,318
unique ordered evidence sets and 432 reference questions. All 294 referenced
image files resolve. The frozen selections contain zero image files, so
the eventual score evaluates textual selections against the organizer
references and cannot establish visual evidence recall.

The validation-only pilot uses claims at zero-based ordinals 34 and 29,
chosen before judging for one text-only reference and one image-bearing
reference. The runner preserves the organizer prompts and uses the pinned
checkpoint's generation settings with reproducible seeds per reference and
distinct prediction set. An empty-model device-map check found a placement
on 19 GiB GPU plus 37 GiB host RAM. Real fit and latency require loaded weights.
The runner records both rules because the pinned code and notebook use
image rating 9 and verdict recall greater than 0.4, whereas the published
task description specifies rating 8 and minimum recall 0.3. Raw judge
responses let either rule be checked. Upstream image-scoring exceptions fail
the run so they cannot silently count as perfect image matches.

The local launcher is `experiments/scripts/start_averimatec_semantic_pilot.ps1`.
It reads the read token from the ignored root `.env`, resumes missing
checkpoint files on D:, and writes pilot artifacts to
`tmp/averimatec-semantic-pilot/`. The current background job logs to
`tmp/averimatec-semantic-pilot-job/`. A pilot score is a feasibility check;
manuscript conclusions require a completed development run.

On 2026-11-26, the first launch log showed that the downloader returned a
snapshot path with only six of twelve shards present. The pilot stopped
before loading model weights. The revised launcher calls
`experiments/scripts/fetch_averimatec_gemma.py` to resume each missing file
over HTTP and verify the size and SHA-256 of every indexed shard. The
processor files are also verified against the pinned repository metadata.
The pinned processor accepted a real text prompt and a development reference
image locally. Real BF16 inference remains pending.

## Venue and author checklist

- Q1 candidate: *Information Processing & Management*. SCImago SJR 2025 lists Q1 in Information Systems and Library and Information Sciences. Publisher scope fits retrieval evaluation. The journal-specific guide returned 403 in this environment. Check its current author guide before venue conversion. [SCImago](https://www.scimagojr.com/journalsearch.php?q=12689&tip=sid) · [Elsevier scope](https://shop.elsevier.com/journals/subjects/social-sciences-and-humanities/social-sciences/library-and-information-science/information-search-and-retrieval) · [Author guide](https://www.sciencedirect.com/journal/information-processing-and-management/publish/guide-for-authors)
- Q2 fallback for the current evidence: *Journal of Information Science*. SCImago SJR 2025 lists Q2 in Information Systems and Q1 in Library and Information Sciences. Sage accepts LaTeX and specifies double-anonymized review, an abstract of at most 150 words, Sage Vancouver citations, and a typical 5,000–7,500 words. Confirm institutional quartile rules. [SCImago](https://www.scimagojr.com/journalsearch.php?q=12813&tip=sid) · [Sage author instructions](https://journals.sagepub.com/author-instructions/JIS)
- The canonical PDF is the authored reading copy. Prepare the anonymized manuscript and separate title page in the chosen venue's style. Confirm funding, conflicts, authorship and submission consent, data/code availability, and any required AI-use disclosure with the authors. The archive and code are prepared locally and repository distribution remains an author decision.

An independent human scientific read was unavailable. The final technical pass checked the manuscript against both regenerated benchmark summaries and the new outputs. Review the author declarations and selected venue formatting before submission.
