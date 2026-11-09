# Retrieval Gate Report

Run: `PC-RETR-001`  
Date: 2026-11-09  
Status: **advancement gate passed; engineering target failed**

## Frozen selection

Weights were selected on the training `model_selection` role only. Validation
labels did not select the configuration.

- text: 0.50 dense--BM25 RRF rank + 0.50 cross-encoder rank;
- image: 0.50 SigLIP2 rank + 0.50 filename-text rank;
- packet: 40 text + 20 image candidates.

## Locked-validation result

| Metric | Frozen baseline | Weighted fusion | Difference |
|---|---:|---:|---:|
| Text Recall@40 | 0.33217 | 0.33931 | +0.00714 |
| Image Recall@20 | 0.32871 | 0.32871 | 0.00000 |
| Packet-oracle evidence F1 | 0.42863 | 0.43475 | +0.00612 |
| Complete-gold coverage | 5.30% | 5.91% | +0.60 percentage points |
| Claims with zero reachable gold | 211 | 207 | -4 |

All four advancement conditions pass: packet-oracle F1 improves, neither
modality's operating-depth recall decreases, and complete-gold coverage does
not decrease. The aspirational engineering target of an absolute +0.05
packet-oracle improvement and at least 8% complete-gold coverage does not pass.

The result therefore justifies evaluating the frozen selectors on the repaired
packet, but not presenting weighted fusion as a large retrieval advance. The
protocol's optional OCR/article-context branch remains closed because it was
allowed only if the advancement gate failed.

## Integrity

- Formal-test annotations accessed: **false**.
- Pre-gold ranking manifest SHA-256:
  `7871A9BD68F1C67DB7D72A52263DF9A4C1D907E332B929E682EAA0711BCC8F1D`.
- Weighted ranking archive SHA-256:
  `50C719010FA0A801520B538A4F0E009E9AFCAE079F1ED02C95D80E24780ACDCE`.
- Selected ranking archive SHA-256:
  `41182426C2B4ADD8BFD695D2C5407F55E2A2154B585880CA0E42F7C7F2DC6A53`.
- Result SHA-256:
  `33F36B7834989C5763E74CA4D2B6D5DBFA6F274964F365D079B6197A9CA6CE0D`.
- Audit SHA-256:
  `2A67BBD3EBF43BF53DF2CED88671D1DEA155A8F88869CBB63428C8BD775F2952`.
- Independent audit: **pass**.
- Wall time for the weighted-fusion computation and reporting: 9.51 seconds.

