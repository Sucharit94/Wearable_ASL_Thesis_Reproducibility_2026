# Stage 33A - LOSO unseen-signer plan freeze

Stage 33 evaluates generalization to a signer excluded from all training and
model-selection data. It plans 15 outer evaluations: five models in each of
three held-out-subject directions.

## Frozen inner assignments

| Direction | Held out | Inner train | Inner validation | Outer test |
|---|---|---:|---:|---:|
| loso_subject1 | subject1 | 1260 | 140 | 6000 |
| loso_subject2 | subject2 | 5760 | 640 | 1000 |
| loso_subject3 | subject3 | 6300 | 700 | 400 |

The split unit is the exact-duplicate `split_group_id`. Inner validation is
stratified by training subject and sentence, using seed `20260826 + loso_index`.
The held-out subject is never used for normalization, selection, scheduling,
early stopping, or refit.

## Required reporting views

| Direction | All | Template supported | Token supported | Token unsupported |
|---|---:|---:|---:|---:|
| loso_subject1 | 6000 | 1340 | 1520 | 4480 |
| loso_subject2 | 1000 | 1000 | 1000 | 0 |
| loso_subject3 | 400 | 400 | 400 | 0 |

Support views are filters over the same predictions. They do not create new
models or permit test information to affect training.

## Evaluation judgment

Primary comparison uses pooled WER across all held-out samples. Also report
SER, exact sentence accuracy, edit counts, five-fold/three-direction variation,
efficiency measures, and support-stratified performance. Accuracy alone is not
the model-selection criterion.

## Stage boundary

Stage 33A starts no training and computes no outer score. A resumable GPU runner
must pass its dry-run verifier and be tagged `stage33-loso-runner-v1` before any
held-out-signer scoring begins.
