# Manuscript argument map

This map links the manuscript's main claims to checked evidence and states the
boundary on each claim. The governing unit is the released MOCHEG
development evidence; formal-test annotations were not accessed.

| Question | Supported answer | Primary evidence | Boundary |
|---|---|---|---|
| How much evidence is reachable? | The frozen packet is a binding ceiling: after bounded weighted fusion, packet-oracle F1 is 0.3637 and complete-gold coverage is 4.74% over 10,970 eligible claims. | `retrieval_results.csv`, `retrieval_gate.md`, retrieval audit | The gain is modest; image Recall@20 is unchanged and the engineering target fails. |
| Does probability calibration matter? | Yes. Prior-aware maps reduce the dominant probability error and make threshold and expected-utility decisions meaningful. | frozen diagnostic bundle, calibration table and reliability figure | Calibration is measured on released development resources, not a fresh external population. |
| Which evidence-set policy is confirmed? | Calibrated thresholding is the strongest confirmed policy at 0.1864 evidence F1. Expected-F1 is a compact alternative at 0.1815 F1 and 3.61 items. | `selector_results.csv`, `selector_gate.md`, 10,000 claim-aligned bootstrap samples | Expected-F1 does not beat thresholding; the uncalibrated ablation fails badly. |
| Does packet-completeness routing improve selection? | It has the highest point estimate, 0.1877 F1 with 4.70 items, but the +0.00138 gain over thresholding is inconclusive. | completeness output, selector gate, partition records | Superiority is not established: the interval crosses zero and one partition is negative. |
| Is verdict quality preserved? | The stronger MLP verifier passes aggregate non-inferiority: +0.00434 macro-F1, 95% interval [-0.00277, 0.01130]. | `verifier_results.csv`, `verifier_gate.md`, 10,000 seed-aligned bootstrap samples | Only two of five seed-specific intervals clear the margin; the linear control reverses direction. |

## Contribution boundary

The contribution is an auditable empirical study of retrieval reachability,
operating-prior calibration, and matched evidence-set decisions. It does not
claim state-of-the-art performance, selector superiority, external
generalization, deployment readiness, or a formal-test result.

