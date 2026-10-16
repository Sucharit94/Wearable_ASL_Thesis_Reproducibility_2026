# Stage 13 — Known-Signer Five-Fold Split

## Purpose

Create the frozen outer five-fold partition for known-signer
repetition generalisation.

## Dataset

Recordings: 7,400

Atomic split groups: 7,397

Subject-template strata: 368

## Stratification

The split is stratified by:

`subject + sentence_id`

The atomic assignment unit is:

`stage12_split_group_id`

Therefore exact duplicate recordings identified in Stage 12 cannot be
separated between outer train and outer test.

## Outer-fold interpretation

For fold `k`:

- test = recordings with `outer_fold == k`
- outer training pool = recordings with `outer_fold != k`

No validation partition is created here.

Inner model-selection partitions are created later and may use only the
outer training pool.

## Scientific meaning

This is a known-signer experiment.

The same signer identities and supported sentence/template identities
are intentionally represented across folds.

The experiment measures recognition of held-out repetitions.

It must not be described as unseen-signer or unseen-template
generalisation.

## Determinism

Frozen split seed:

`20260829`

Assignment SHA-256:

`7b9943145ffd9658fe20fedc37e4bd4515496a7cd59b893f95e99f95ea6c95e9`

## Leakage safeguards

Exact numeric duplicate groups split across folds:

0

Subject-template strata missing a fold:

0

## Deferred decisions

- inner validation/model selection
- preprocessing statistics
- CTC feasibility
- model training

These are handled in later stages.
