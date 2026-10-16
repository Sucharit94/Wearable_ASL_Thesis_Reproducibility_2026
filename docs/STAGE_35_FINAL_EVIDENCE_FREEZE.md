# Stage 35 — Final Known-Signer Evidence Freeze

## Frozen conclusion

G1 is the accuracy reference. C2 is the recommended lightweight model.
This conclusion is based on quality, uncertainty, paired statistics and
measured efficiency; accuracy is not used alone.

## Publication-ready model table

| Model | WER (95% CI) | Exact accuracy (95% CI) | SER | Parameters | CPU latency | Throughput | Role |
|---|---:|---:|---:|---:|---:|---:|---|
| G1 | 0.001074 [0.000608, 0.001658] | 0.996486 [0.994744, 0.997962] | 0.003514 | 4,791,688 | 12.817 ms | 119.10/s | accuracy_reference |
| C1 | 0.067215 [0.057684, 0.077463] | 0.799595 [0.774057, 0.824457] | 0.200405 | 392,488 | 2.183 ms | 1705.92/s | eligible_comparator |
| C2 | 0.040050 [0.033484, 0.046848] | 0.864730 [0.842954, 0.885984] | 0.135270 | 656,168 | 4.525 ms | 856.19/s | recommended_lightweight |
| C3 | 0.444667 [0.427566, 0.462208] | 0.136486 [0.111921, 0.161653] | 0.863514 | 237,352 | 2.090 ms | 1726.05/s | performance_threshold_not_met |
| C4 | 0.040909 [0.034262, 0.048115] | 0.876622 [0.857713, 0.895109] | 0.123378 | 348,604 | 24.897 ms | 41.73/s | eligible_comparator |

## Main comparison

C2 uses 86.31% fewer parameters than G1 (656,168 versus 4,791,688). Its five-fold WER is 0.040050, compared with 0.001074 for the accuracy reference.

C2 and C4 have statistically unresolved WER at 95%, but C4's measured batch-1 latency was 5.50× C2's latency, while C2 provided 20.52× higher batch-16 throughput on the frozen CPU benchmark.

## Aggregate error profile

| Model | Substitutions | Deletions | Insertions | Dominant error |
|---|---:|---:|---:|---|
| G1 | 1 | 27 | 2 | deletion |
| C1 | 81 | 1649 | 148 | deletion |
| C2 | 66 | 721 | 332 | deletion |
| C3 | 595 | 11801 | 28 | deletion |
| C4 | 104 | 932 | 107 | deletion |

Blank aggregate counts mean that the upstream summary did not retain
that count; no value is inferred or fabricated.

## Thesis-safe interpretation

- G1 is the recognition-quality reference, not a lightweight model.
- C2 is the selected lightweight quality–efficiency compromise.
- C4 remains a competitive quality comparator, but its measured runtime cost is high.
- C3 demonstrates that a low parameter count does not guarantee adequate recognition.
- These findings apply to the frozen mixed-subject five-fold protocol and measured hardware.

## Methodological boundary

No model was retrained and no test prediction was regenerated in Stage 35.
The stage only consolidates previously frozen evidence.
