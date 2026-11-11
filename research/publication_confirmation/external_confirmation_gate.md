# External Confirmation Gate

Date: 2026-11-12
Decision: **locked development run complete; blind gate remains closed**

## Completed independent evaluation

The organizer training and validation stores were hash-audited, the five
training-only seeds were frozen before development preparation, and the
152-claim comparison ran once. The locked 3,800-record bundle was independently
recomputed. Exact-source pool recall is 25.48%, and 59.87% of claims have no
reachable target. Of the four selector comparisons, only calibration clears
its prespecified simultaneous interval: calibrated expected-F1 improves over
its uncalibrated ablation by 0.05316 with interval [0.02207, 0.08852].

- threshold minus fixed: +0.00003, interval [-0.02341, 0.02404];
- expected-F1 minus threshold: +0.01714, but 3.20 versus 1.17 items;
- switch minus threshold: +0.01555, interval [-0.00263, 0.03621]; and
- switch verifier macro-F1 minus threshold: -0.01293, interval
  [-0.03242, 0.00687], failing the -0.010 safeguard.

The frozen blind rule therefore selects the train-selected fixed family. Five
local development exports pass the official schema and remain unsubmitted.

## Why the blind gate remains closed

Development outcomes do not authorize formal-test access. The user has not
authorized generation or submission of the one frozen blind file. The official
semantic evaluator is also unavailable as specified: its 54.86 GB of weights
do not fit the local 24.56 GiB RTX 4090, and the organizer launcher requests two
GPUs. Exact-source diagnostics are not mislabeled as the official score.

## Publication consequence

The manuscript therefore:

1. reports the external run as a diagnostic nonreplication;
2. avoids state-of-the-art, official-score, and universal-generalization claims;
3. reports reachability, calibration, failed selector gates, class-wise
   verifier behavior, and the unavailable judge; and
4. states that both formal-test annotation sets were not accessed.

The blind test can be reconsidered only with explicit author authorization.
The frozen selection rule, fixed-family choice, and one-submission limit must
not be changed after the locked development result.
