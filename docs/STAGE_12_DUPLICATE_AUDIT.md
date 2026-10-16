# Stage 12 — Duplicate Audit

## Purpose

Detect exact duplicate recordings before any train/test folds are
created.

## Manifest

Recordings: 7,400

Stage 4 manifest hash verified: yes

## Byte duplicates

Exact byte-level duplicate groups:

3

Recordings inside byte-level duplicate groups:

6

## Numeric duplicates

Exact parsed-numeric duplicate groups:

3

Recordings inside numeric duplicate groups:

6

Maximum numeric duplicate group size:

2

Numeric duplicate groups that are not entirely byte-identical:

0

## Cross-boundary risks

Cross-subject numeric duplicate groups:

0

Cross-sentence-ID numeric duplicate groups:

0

Same-target but different-sentence-ID numeric duplicate groups:

0

Gloss-target conflict groups:

0

English-target conflict groups:

0

## Canonical numeric equality

Each CSV is parsed as a `T x 38` float64 matrix.

The numeric duplicate hash contains:

- matrix row count,
- matrix column count,
- canonical little-endian float64 values,
- negative zero normalized to positive zero.

No tolerance or approximate similarity is used.

Therefore Stage 12 detects exact equality after numerical parsing, not
near-duplicate motion sequences.

## Split rule

For known-signer folds, every exact numeric duplicate group is
indivisible.

All members of a duplicate group must be assigned to the same
partition.

Nonduplicate recordings receive their own singleton split group.

## LOSO rule

If a numeric duplicate group crosses subject boundaries, Stage 12 does
not silently resolve it.

LOSO handling must preserve the signer definition and explicitly deal
with any such leakage risk.

## Scope

No train/test split is created in Stage 12.

No raw recording is edited or removed.
