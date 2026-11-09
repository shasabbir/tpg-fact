# Stronger Shared-Verifier Gate

Date: 2026-11-09  
Run: `PC-VERIFIER-001`  
Protocol: `experiments/configs/publication_confirmation.yaml`  
Audit: **pass**  
Formal-test annotations accessed: **no**

## Decision

The aggregate verifier non-inferiority condition passed, but the stronger
per-seed safeguard failed. The completeness-aware switch therefore remains a
promising development result, not a stable end-to-end improvement claim.

## Frozen evaluation

The shared verifier was the prespecified two-layer multilayer perceptron with
one 64-unit ReLU hidden layer, inverse-frequency weighting, 300 full-batch Adam
epochs, learning rate 0.01, and L2 penalty 0.001. Each seed used the same model
architecture, fitting roles, selected evidence representations, and validation
claims for both selectors. A linear verifier was retained as a diagnostic
control. The non-inferiority margin was -0.010 macro-F1.

## Results

| Verifier and evidence policy | Mean validation macro-F1 |
|---|---:|
| MLP + calibrated threshold | 0.39108 |
| MLP + completeness-aware switch | **0.39542** |
| Linear control + calibrated threshold | **0.37874** |
| Linear control + completeness-aware switch | 0.37461 |

For the MLP, the aggregate switch-minus-threshold effect was +0.00434
macro-F1 with a seed-aligned paired 95% bootstrap interval of
[-0.00277, 0.01130] over 1,490 validation claims and five verifier seeds. The
lower bound exceeds the frozen -0.010 margin, so aggregate non-inferiority
passes. The interval includes zero and does not establish superiority.

| Seed | Threshold | Switch | Difference | 95% interval | Non-inferior? |
|---:|---:|---:|---:|---:|:---:|
| 13 | 0.39622 | 0.39621 | -0.00001 | [-0.01367, 0.01388] | no |
| 37 | 0.37357 | 0.39150 | +0.01793 | [0.00276, 0.03302] | yes |
| 73 | 0.39623 | 0.40716 | +0.01094 | [-0.00464, 0.02642] | yes |
| 101 | 0.40108 | 0.39457 | -0.00651 | [-0.02174, 0.00895] | no |
| 137 | 0.38829 | 0.38766 | -0.00063 | [-0.01406, 0.01281] | no |

## Interpretation

The switch is non-inferior after aggregating seeds, and its mean point estimate
is higher under the nonlinear verifier. However, three individual seeds have
lower confidence limits below the non-inferiority margin, the aggregate
interval crosses zero, and the linear control moves in the opposite direction.
These results show that verifier choice and seed variation remain meaningful
sources of uncertainty. The paper may report the point estimate and aggregate
non-inferiority result, but must also report the failed per-seed safeguard and
must not claim stable verifier superiority.

## Traceability

- Output: `experiments/outputs/publication_confirmation/verifier/summary.json`
- Audit: `experiments/outputs/publication_confirmation/verifier/verifier.audit.json`
- Summary SHA-256: `5457c3420e2ec7abe765f4c9a6ba5bd132db38ee4e5e018770aba76a807dd257`
- Runtime: 246.89 seconds on the recorded publication-confirmation hardware

