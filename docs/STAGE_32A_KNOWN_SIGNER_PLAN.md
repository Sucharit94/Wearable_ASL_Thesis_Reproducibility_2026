# Stage 32A - Known-signer plan freeze

Stage 32 will execute 25 outer evaluations: G1, C1, C2, C3 and C4 on
each of the five Stage-13 known-signer outer folds. Execution is
fold-major, so every model completes Fold 1 before Fold 2 starts.

## Leakage boundary

- Stage-31 frozen tag: `stage31-freeze-v1`
- Stage-31 commit: `62fc7ca0f14611bf0fbc305f1960af4cff441978`
- Stage-13 assignment SHA-256: `7b9943145ffd9658fe20fedc37e4bd4515496a7cd59b893f95e99f95ea6c95e9`
- Inner split unit: `stage12_split_group_id`
- Inner stratification: `subject + sentence_id`
- Inner validation seed: `20260826 + outer_fold`
- Outer test controls training: no

## Frozen inner assignments

| Outer fold | Inner train | Inner validation | Outer test | Duplicate crossings |
|---:|---:|---:|---:|---:|
| 1 | 5328 | 592 | 1480 | 0 |
| 2 | 5328 | 592 | 1480 | 0 |
| 3 | 5328 | 592 | 1480 | 0 |
| 4 | 5328 | 592 | 1480 | 0 |
| 5 | 5328 | 592 | 1480 | 0 |

The assignment file has 37,000 rows: all 7,400 samples
represented once for each outer fold. For each fold, the Stage-13 test
partition is preserved exactly.

## Nested execution

Each of the 25 evaluations has three registry rows: the parent outer
evaluation, inner selection, and fresh full-data refit. This produces 75
planned registry rows. After epoch E* is selected using inner-validation
WER, the model is reset, normalization is recomputed on the full outer
training partition, and the model is refit for exactly E* epochs. The
outer test partition is evaluated once.

## Stage boundary

This substage creates no score and no checkpoint. The resumable runner
must pass CPU dry-run tests and be committed and tagged
`stage32-known-runner-v1` before the first GPU run or outer-test evaluation.
