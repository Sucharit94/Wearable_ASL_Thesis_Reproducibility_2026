# Stage 16A — Template-Holdout Feasibility Audit

## Purpose

Determine which complete sentence/template groups are structurally
eligible for a vocabulary-supported template-holdout experiment.

No train/test split is created in Stage 16A.

## Label-equivalence groups

Sentence IDs:

300

Effective template-equivalence groups:

299

Sentence IDs 39 and 83 form one indivisible group because their gloss
and English targets are identical.

## Vocabulary

Global gloss tokens:

455

Tokens occurring in only one template-equivalence group:

278

## Individual holdout feasibility

Individually safe groups:

103

Individually unsafe groups:

196

A group is individually safe only when every one of its target gloss
tokens occurs in at least one other template-equivalence group.

## Important limitation

Individual safety is not sufficient for selecting multiple test
templates simultaneously.

If two candidate groups are the only remaining support for one
another's token, holding both out together would remove that token from
training.

Therefore Stage 16B must verify vocabulary support after removing the
entire proposed held-out group set.

## Scientific target

The final template-holdout experiment will test unseen complete
sentence/templates while retaining training support for every target
gloss token.

It is not an open-vocabulary experiment.
