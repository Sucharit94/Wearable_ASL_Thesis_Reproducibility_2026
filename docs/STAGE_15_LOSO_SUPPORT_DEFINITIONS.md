# Stage 15 — LOSO Support Definitions

## Purpose

Freeze support-aware evaluation definitions for the three LOSO runs.

Support is determined only from the training partition of the
corresponding held-out-signer run.

## LOSO-All

Includes every recording from the held-out subject.

This is the complete unseen-signer evaluation.

It may also contain templates or gloss tokens absent from training.

## LOSO-Template-Supported

Includes held-out recordings whose exact sentence ID occurs in the
training partition.

This evaluates unseen-signer recognition while keeping the complete
sentence/template supported by training.

## LOSO-Token-Supported

Includes held-out recordings for which every target gloss token occurs
somewhere in the training partition.

The exact complete sentence does not need to have occurred in training.

Therefore this set can contain novel combinations of individually
supported gloss tokens.

## Nesting

`LOSO-Template-Supported`

is a subset of

`LOSO-Token-Supported`

which is a subset of

`LOSO-All`.

## Token-supported novel-template diagnostic

A held-out recording may have every gloss token supported while its
complete sentence/template is absent from training.

These cases are recorded separately because they test unseen signer
plus unseen template composition without introducing unseen gloss
tokens.

## Token-unsupported diagnostic

A held-out recording is token-unsupported if at least one target gloss
token never occurs in that LOSO training partition.

These recordings remain part of LOSO-All.

They are not deleted.

## Model-training rule

Support subsets do not create separate training jobs.

For each held-out signer, one recognizer is trained using the complete
LOSO training partition.

Its held-out predictions are then evaluated on:

- LOSO-All,
- LOSO-Template-Supported,
- LOSO-Token-Supported.

Thus support-aware reporting changes the evaluation denominator, not
the trained recognizer.

## Global vocabulary

The Stage-10 global CTC vocabulary remains the common output schema.

However, the existence of a token in that global vocabulary does not
mean the token has training support in a particular LOSO run.

## Reporting rule

The three held-out-signer directions must be reported separately.

In particular, LOSO Subject 1 contains substantial template and token
support shift, whereas LOSO Subjects 2 and 3 have different support
conditions.

No unsupported sample is silently removed from LOSO-All.
