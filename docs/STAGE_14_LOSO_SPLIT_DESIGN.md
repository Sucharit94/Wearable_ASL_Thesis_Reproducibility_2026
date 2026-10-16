# Stage 14 — LOSO Split Design

## Purpose

Freeze the three signer-held-out train/test partitions.

Stage 14 defines signer membership only.

Support-aware evaluation subsets are defined later in Stage 15.

## LOSO runs

### LOSO Subject 1

Test subject:

`subject1`

Training subjects:

`subject2 + subject3`

Test recordings:

6,000

Training recordings:

1,400

### LOSO Subject 2

Test subject:

`subject2`

Training subjects:

`subject1 + subject3`

Test recordings:

1,000

Training recordings:

6,400

### LOSO Subject 3

Test subject:

`subject3`

Training subjects:

`subject1 + subject2`

Test recordings:

400

Training recordings:

7,000

## Duplicate safety

Stage 12 found zero cross-subject exact numeric duplicate groups.

Therefore signer boundaries do not force an exact numeric duplicate
across LOSO train and test.

## Interpretation

Every LOSO test partition contains a signer completely absent from
training.

However, the three runs do not have equivalent template or vocabulary
support.

That distinction is formalised in Stage 15.

## Preprocessing rule

No normalization, filtering statistics or model-selection decisions are
calculated here.

Future LOSO preprocessing must be fit only on the corresponding
training subjects.

## Frozen assignment

Assignment SHA-256:

`28fef677789f652c0939fb4550a78ef928dfdbc58615ebea773c5ff8a9c86de9`
