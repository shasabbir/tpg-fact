# Selector Gate Report

Run: `PC-SELECT-001`  
Date: 2026-11-09  
Status: **complete; superiority gate failed**

## Matched validation results

All policies use the same repaired 40-text/20-image packet, scorer family,
calibration map, complete-gold denominator, and five deterministic training-role
partitions.

| Policy | Evidence F1 | Mean selected items | Interpretation |
|---|---:|---:|---|
| Fixed top-5 | 0.17119 | 5.00 | Simple fixed control |
| Calibrated threshold | 0.18637 | 5.26 | Strongest prespecified baseline |
| Expected-F1 | 0.18153 | 3.61 | More compact, but lower F1 than threshold |
| Learned count | 0.11219 | 31.49 | Clear negative control |
| Uncalibrated expected-F1 | 0.12884 | 24.90 | Demonstrates the calibration dependency |
| Completeness-aware switch | **0.18775** | 4.70 | Highest point estimate, but not a confirmed improvement |

Expected-F1 minus calibrated threshold is `-0.00484`, with a claim-aligned
95% interval of `[-0.00938, -0.00020]`. The repaired packet therefore does not
change the earlier ordering: thresholding remains stronger than expected-F1.

## Completeness-aware policy

The switch predicts whether the 60-item packet contains the full released-gold
set using the frozen ten-feature profile. It selects expected-F1 for predicted
complete packets and calibrated thresholding otherwise. Every partition chose
the prespecified cutoff `0.50`; expected-F1 was used for 37.8%--46.4% of
validation claims.

The switch improves on calibrated thresholding by `+0.00138` F1, with 95% CI
`[-0.00191, 0.00462]`. It returns 0.56 fewer items on average and increases
precision, but:

- the effect is below the declared `+0.010` practical threshold;
- the confidence interval crosses zero;
- one of five partition effects is negative.

The selector-superiority gate therefore fails. The result can support a
precision/review-burden tradeoff discussion, not a method-superiority claim.

## Integrity

- Base selector summary SHA-256:
  `F9D2E2220B564B18239E1351E6CC1874909C8E3E1EA6C80C751EB3EB669D06F9`.
- Base selector audit SHA-256:
  `B848B8186F1B323330C7F80FFC6BF789668E998076FD8CF05750913E8C92A170`.
- Completeness-switch summary SHA-256:
  `C18080026AE3EB7B363A829FA8BBF2DCEE4679F22F9C9ADBBE362FAAFEFBF8DF`.
- Completeness-switch audit SHA-256:
  `8F65302E2D0E7848A267A5D7A4610F5AE147E226E7A08F29A3714F07FBD54B42`.
- Independent audits: **pass**.
- Formal-test annotations accessed: **false**.

