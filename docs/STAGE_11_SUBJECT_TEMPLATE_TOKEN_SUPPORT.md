# Stage 11 — Subject / Template / Token Support Audit

## Purpose

Measure descriptive support before creating any train/test split.

## Global source

Recordings: 7,400

Sentence/template IDs: 300

Gloss vocabulary: 455

## Subject coverage

### subject1

Recordings: 6,000

Supported templates: 300

Supported gloss tokens: 455

### subject2

Recordings: 1,000

Supported templates: 48

Supported gloss tokens: 99

### subject3

Recordings: 400

Supported templates: 20

Supported gloss tokens: 54


## Cross-subject support

Templates supported by all three subjects:

1

Gloss tokens supported by all three subjects:

22

## Duplicate-label rule

Sentence IDs 39 and 83 carry the same gloss and English targets.

They remain distinct dataset IDs, but they must remain in the same
partition for experiments claiming unseen complete-template
generalisation.

## LOSO interpretation

The LOSO preview measures support only.

It does not create the final LOSO partitions.

Stage 15 will formally define:

- LOSO-All,
- LOSO-Template-Supported,
- LOSO-Token-Supported.

## Scope boundary

No folds or train/test assignments are generated in Stage 11.
