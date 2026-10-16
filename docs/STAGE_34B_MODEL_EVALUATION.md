# Stage 34B — Five-Model Recognition Evaluation

## Scope

This report recomputes recognition results from 37,000 paired outer-fold
predictions: 7,400 samples for each of five models. No training is run.

## Primary ranking

| Rank | Model | Parameters | Pooled WER | Exact sentence accuracy | SER | Deletion rate | Blank-output rate |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | G1 | 4,791,688 | 0.001074 | 0.996486 | 0.003514 | 0.000966 | 0.000000 |
| 2 | C2 | 656,168 | 0.040050 | 0.864730 | 0.135270 | 0.025805 | 0.006351 |
| 3 | C4 | 348,604 | 0.040909 | 0.876622 | 0.123378 | 0.033357 | 0.000270 |
| 4 | C1 | 392,488 | 0.067215 | 0.799595 | 0.200405 | 0.059019 | 0.002838 |
| 5 | C3 | 237,352 | 0.444667 | 0.136486 | 0.863514 | 0.422369 | 0.040405 |

## Interpretation boundary

Pooled WER is the primary recognition metric. Exact sentence accuracy
is secondary because a sentence is counted wrong after any token error.
Parameter count describes compactness but does not by itself establish
runtime efficiency. Statistical comparisons follow in Stage 34C and
runtime/memory benchmarking follows in Stage 34D.

## Additional outputs

- Per-fold error metrics and stability inputs
- Per-subject performance
- Performance by reference sequence length
- Substitution, deletion and insertion rates
- Blank-output and output-length diagnostics
