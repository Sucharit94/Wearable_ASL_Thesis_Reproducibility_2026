# Stage 23 — Recognition Metrics

## Headline recognition metric

Pooled word error rate is calculated as:

`sum(S + D + I) / sum(reference gloss tokens)`

The unweighted average of per-sample WER values is retained only as a
diagnostic and is not the headline result.

## Sentence metrics

Sentence error rate:

`sentences containing at least one error / number of sentences`

Exact sentence accuracy:

`1 - sentence error rate`

## Alignment

Token-level Levenshtein alignment uses the frozen tie-breaking order:

1. exact match;
2. substitution;
3. deletion;
4. insertion.

This makes substitution, deletion and insertion reporting reproducible.

## Perfect frozen-target control

Sentence templates:

300

Reference tokens:

1147

Total errors:

0

Pooled WER:

0.0

Exact sentence accuracy:

1.0

## Synthetic controlled-error test

Substitutions:

1

Deletions:

1

Insertions:

1

Pooled WER:

0.2727272727272727

SER:

0.75

Exact sentence accuracy:

0.25

## Dependency protection

Stage-21 padded outputs excluded:

True

Stage-22 same final length supplied to loss and decoder:

True
