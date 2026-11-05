# Official MOCHEG Baseline Compatibility Audit

Date: 2026-11-02

Status: protocol-stage audit; no official-baseline result is claimed here.

## Available authoritative implementation

The official MOCHEG repository is `PLUM-Lab/Mocheg`. Its published environment uses Python 3.8.10, PyTorch 1.9.0, torchvision 0.10.0, and CUDA 10.2. The current workstation uses Python 3.13 and a substantially newer CUDA/PyTorch stack. Its original pretrained-model and dataset links must also be identity-checked before use.

## Compatibility decision

Running the old repository directly inside the Phase-2 environment would require material compatibility changes and would not be a faithful reproduction. Therefore:

1. the published baseline values remain contextual rather than entries in paired tests;
2. a reproduction, if attempted, must run in an isolated legacy-compatible environment;
3. every source or dependency modification must be disclosed;
4. the Phase-2 matched comparison uses the repository's own fixed, threshold, count, and CEFES controls on identical candidates; and
5. no manuscript statement may say the official baseline was reproduced until a separate artifact audit passes.

This decision prevents an incompatible port from being mislabeled as the official baseline while allowing the retrieval-conditioned study to proceed.
