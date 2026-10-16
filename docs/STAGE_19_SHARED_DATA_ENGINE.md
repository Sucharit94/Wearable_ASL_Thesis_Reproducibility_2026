# Stage 19 — Shared Dataset and Collate Engine

## Purpose

Connect the frozen cache, labels, splits and train-only preprocessing
contract through one shared framework-independent data engine.

## Cache

Recordings:

7,400

Total valid time steps:

1,829,320

Feature width:

38

The Stage-17 binary is opened read-only through NumPy memmap.

## Targets

Sentence targets:

300

CTC output schema:

456 classes

CTC blank ID:

0

The blank ID is not permitted inside a reference gloss target.

## Dataset output

Each dataset item returns:

- sample ID,
- subject,
- sentence ID,
- repetition ID,
- variable-length `T x 38` input,
- original valid input length,
- gloss token IDs,
- target length,
- raw gloss text,
- English reference text.

## Normalization

Normalization statistics are supplied explicitly to a dataset.

Stage 19 does not fit a global normalization vector.

The integration smoke test fitted statistics only from known-signer
Fold-1 training recordings and then applied those fixed values to
held-out examples.

Those diagnostic values are not a reusable global normalization
artifact.

## Collation

A batch is padded to:

`B x T_max x 38`

Input padding value:

0.0

Original valid lengths are preserved separately.

A boolean valid mask is generated.

Reference targets use temporary padding value:

-1

CTC blank 0 is never overloaded as target padding.

## Split loaders

The same data engine can consume:

- five known-signer outer folds,
- three LOSO partitions,
- the frozen template-holdout partition.

## Deferred

Stage 19 does not calculate model output lengths.

CTC input feasibility is Stage 20.

CTC loss and decoding are Stage 21.

Length propagation through model architectures is Stage 22.
