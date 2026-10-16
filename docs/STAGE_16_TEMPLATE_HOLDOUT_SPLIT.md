# Stage 16 — Vocabulary-Supported Template Holdout

## Purpose

Create one frozen complete-template holdout experiment.

The test templates are absent from recognition training while every
target gloss token remains supported by the remaining training
templates.

## Feasibility basis

Stage 16A found:

- 299 template-equivalence groups,
- 103 individually holdout-safe groups,
- 196 individually unsafe groups.

The final test set is selected only from the safe candidate pool.

## Test size

Eligible safe groups:

103

Selected test groups:

21

The test set therefore contains approximately 20% of the eligible
closed-vocabulary template groups.

It is not 20% of all 299 template-equivalence groups.

## Subject-support quotas

S1-only test groups:

15

S1+S2 test groups:

4

S1+S3 test groups:

2

This guarantees template-holdout test recordings for all three
subjects.

## Joint vocabulary-support rule

The entire proposed test set is removed simultaneously before support
is checked.

All target gloss tokens occurring in the held-out test groups must
remain present in at least one remaining training template.

Global gloss vocabulary:

455

Gloss tokens still represented in training:

455

Unsupported held-out target gloss tokens:

0

## Template leakage

Training/test template-group overlap:

0

Complete held-out template-equivalence groups therefore do not occur
in recognition training.

## Duplicate safety

Stage-12 exact numeric split-group role violations:

0

## Duplicate-label equivalence

Sentence IDs 39 and 83 remain one indivisible equivalence group.

Selected for test:

False

Both IDs therefore always remain on the same side of the split.

## Scientific interpretation

This experiment evaluates generalisation to unseen complete
sentence/templates under supported gloss vocabulary.

It does not test unseen gloss tokens.

It does not represent unseen-signer evaluation because signer identity
is not held out.

## Language-stage leakage rule

When this template-holdout split is used later for retrieval or
generative English correction, the same held-out template groups must
also be absent from retrieval references and language-stage training
examples.

## Frozen assignment

Selection seed:

20260830

Successful deterministic attempt:

0

Assignment SHA-256:

`b36096c0cb14cf137eb2a5bdbf206a5aad462326cdc490dbe5810705ac5e3376`
