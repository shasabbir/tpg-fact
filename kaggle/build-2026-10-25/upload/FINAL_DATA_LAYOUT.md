# Required layout for the future audited final-data ZIP

Name the archive `dtp_fact_final_data_<date>.zip`. It must expand to one folder
containing:

```text
dtp_fact_final_data/
├── collection_manifest.json
├── conflict_donors.jsonl
├── image_manifest.csv
├── images/
├── pilot_claims_with_evidence.jsonl
├── pilot_evidence.jsonl
├── pilot_assessment.json
├── frozen/
│   ├── claims.jsonl
│   ├── evidence.jsonl
│   ├── train_claims.jsonl
│   ├── validation_claims.jsonl
│   ├── test_claims.jsonl
│   └── dataset_manifest.json
├── evidence_image_manifest.csv        # optional
├── evidence_images/                   # optional
└── evidence_text.jsonl                # optional
```

Every `local_path` in an image manifest must be relative to that manifest, for
example `images/claim_001.jpg`; Windows absolute paths are invalid on Kaggle.
`conflict_donors.jsonl` must contain one audited entry for every test claim with
`claim_id`, `donor_evidence_id`, and JSON boolean
`"semantic_conflict_verified": true`.

Do not include an API key, `.secrets`, unlicensed article bodies, or images that
cannot legally be placed in a private research dataset.

