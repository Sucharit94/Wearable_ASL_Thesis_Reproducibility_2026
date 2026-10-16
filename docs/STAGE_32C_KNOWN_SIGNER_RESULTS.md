# Stage 32C — Known-Signer Five-Fold Results

## Protocol

- 5 models × 5 outer folds = 25 evaluations
- 1,480 outer-test samples per fold
- Float32 execution on NVIDIA Tesla T4
- Automatic mixed precision disabled
- Every outer test evaluated exactly once

## Five-fold summary

| Rank | Model | Pooled WER | Mean accuracy | Accuracy SD | Mean SER |
|---:|---|---:|---:|---:|---:|
| 1 | G1 | 0.001074 | 0.996486 | 0.002761 | 0.003514 |
| 2 | C2 | 0.040050 | 0.864730 | 0.010026 | 0.135270 |
| 3 | C4 | 0.040909 | 0.876622 | 0.019001 | 0.123378 |
| 4 | C1 | 0.067215 | 0.799595 | 0.018953 | 0.200405 |
| 5 | C3 | 0.444667 | 0.136486 | 0.002984 | 0.863514 |

## Interpretation

G1 achieved the lowest known-signer WER, but this does not by itself establish the best overall model.

Final model judgement must also include model size, inference time, memory, cross-fold variability and unseen-signer performance.

C3 produced deletion-heavy recognition errors and substantially underperformed the recurrent and Tiny-Swin models.

## Audit status

**STAGE 32C RESULTS AUDIT PASSED**
