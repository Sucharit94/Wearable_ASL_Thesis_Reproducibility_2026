# Stage 34E — Final Multi-Criteria Model Selection

## Decision rule

Models must first satisfy WER ≤ 0.10 and exact-sentence accuracy ≥ 0.75.
Eligible models are then compared under quality-priority, balanced and
efficiency-priority weight scenarios. Scores use min-max utility within
the eligible set. The recommendation is the eligible sub-one-million-
parameter model with the highest mean score across all three scenarios.
This is a transparent decision aid, not a claim that weights are universal.
If the operating system cannot expose peak RSS, the memory criterion is
marked unavailable and its weight is proportionally redistributed across
the remaining criteria; the benchmark still reports this limitation.
Models with an unresolved paired WER difference receive their common group
mean WER during scoring, so an insignificant numerical difference cannot
change the multi-criteria decision.

## Combined evidence

| Model | WER | Exact accuracy | Parameters | Batch-1 ms | Batch-16 samples/s | Quality eligible | Pareto | Role |
|---|---:|---:|---:|---:|---:|---|---|---|
| G1 | 0.001074 | 0.996486 | 4,791,688 | 12.817 | 119.10 | True | True | accuracy_reference |
| C1 | 0.067215 | 0.799595 | 392,488 | 2.183 | 1705.92 | True | True | eligible_comparator |
| C2 | 0.040050 | 0.864730 | 656,168 | 4.525 | 856.19 | True | True | recommended_lightweight |
| C3 | 0.444667 | 0.136486 | 237,352 | 2.090 | 1726.05 | False | False | performance_threshold_not_met |
| C4 | 0.040909 | 0.876622 | 348,604 | 24.897 | 41.73 | True | True | eligible_comparator |

## Weight sensitivity

| Scenario | Rank | Model | Score |
|---|---:|---|---:|
| quality_priority | 1 | G1 | 0.774920 |
| quality_priority | 2 | C2 | 0.514950 |
| quality_priority | 3 | C4 | 0.379053 |
| quality_priority | 4 | C1 | 0.299012 |
| balanced | 1 | G1 | 0.603836 |
| balanced | 2 | C2 | 0.600170 |
| balanced | 3 | C1 | 0.498025 |
| balanced | 4 | C4 | 0.398859 |
| efficiency_priority | 1 | C1 | 0.697531 |
| efficiency_priority | 2 | C2 | 0.689297 |
| efficiency_priority | 3 | G1 | 0.452164 |
| efficiency_priority | 4 | C4 | 0.369315 |

## Final interpretation

- Accuracy reference: G1.
- Recommended lightweight model: C2.
- The recommendation combines frozen recognition, uncertainty and
  same-machine efficiency evidence; it is not based on accuracy alone.
- C2 and C4 must not be separated by WER significance because Stage
  34C did not resolve their WER difference at 95% confidence.
