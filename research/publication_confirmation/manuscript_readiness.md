# Publication readiness review

Reviewed: 2026-11-12

## Verdict

The study is internally coherent, reproducible from the recorded artifacts,
and substantially stronger as a full manuscript. The independently frozen
AVerImaTeC evaluation is complete and provides a useful nonreplication rather
than method superiority. The scientific manuscript is ready for venue-specific
review, but the submission package is not complete because the official
external semantic judge, blind-test authorization, venue rules, and author
declarations remain unresolved.

## Verified strengths

| Area | Evidence | Status |
|---|---|---|
| Implementation | Full suite: 163 tests passed; bytecode compilation and external reporting regeneration succeed | Pass |
| Formal-test protection | Access flag is false in run outputs, audits, and compact report | Pass |
| Retrieval analysis | 10,970 eligible claims; prespecified advancement gate passes | Pass, modest effect |
| Selector comparison | Six matched policies, five deterministic partitions, claim-aligned uncertainty | Pass |
| Stronger verifier | Shared MLP safeguard plus linear capacity control | Aggregate pass; seed-stability fail |
| Independent evaluation | One frozen five-seed AVerImaTeC organizer-store run; 3,800 locked records and independent recomputation | Pass; diagnostic nonreplication |
| Reference integrity | 51/51 cited records pass registry, ledger, and rendered-DOI checks | Pass |
| PDF build | 18 pages; no undefined citations/references or overfull boxes | Pass |
| Visual review | All 18 pages inspected; tables, equations, figures, floats, and references are legible | Pass |
| Claim discipline | Abstract, results, discussion, limitations, and conclusion report failed gates | Pass |

## Scientific assessment

The strongest result is the joint diagnosis: retrieval limits the attainable
evidence score, and prior-aware calibration materially changes adaptive set
decisions even when rank order is unchanged. Calibrated thresholding is the
best confirmed selector. Expected-F1 supplies a smaller review set, while the
completeness switch remains a plausible but unconfirmed direction.

The main weakness is not implementation quality; it is the evidence endpoint.
On AVerImaTeC, 59.87% of development claims have no exact reachable target and
the official Gemma-3-27B semantic evaluator cannot run under the prescribed
condition on the single local GPU. The exact-source result is therefore a
diagnostic rather than a shared-task score. It confirms calibration dependence
but contradicts compact expected-F1 selection, threshold superiority, switch
superiority, and verifier preservation. Broad generalization and method-
superiority claims remain out of scope.

## Remaining submission blockers

| Priority | Blocker | Required resolution |
|---:|---|---|
| 1 | Official external semantic score unavailable | Obtain suitable two-GPU evaluator hardware if the target venue requires the official score; otherwise retain the explicit limitation. |
| 1 | Blind test remains authorization-gated | Keep both tests sealed unless the authors explicitly authorize the one frozen prediction/submission. |
| 1 | Selector superiority and verifier safeguards fail | Keep the negative external result and restrained claims; do not advertise the switch as superior. |
| 1 | Author declarations are unknown | Authors must provide CRediT roles, funding, conflicts, data/code availability wording, acknowledgments, and any required ethics statement. |
| 2 | Venue not selected | Select the journal/conference, then apply its page limit, declaration order, anonymity rules, and supplementary-material policy. |
| 2 | Formal-test authorization absent | Keep annotations sealed unless the authors explicitly authorize a single frozen evaluation. |
| 3 | Final archival package | The generic clean-build archive is verified; create the venue-specific archive after venue conversion. |

## Recommendation

Use the current manuscript as a cross-benchmark measurement and negative-
results study. Its value is the independently frozen contradiction, not a new
adaptive-selector win. Venue conversion and author declarations are the next
required tasks; the official judge and blind test are optional only if their
absence is accepted and disclosed.

