# Stage 10 — Labels and Fresh CTC Vocabulary

## Source label files

Recognition/gloss target:

`label_no_order.txt`

English target:

`label_order.txt`

Both source hashes still match the immutable Stage 3 contract.

## Sentence mapping

The dataset uses recording sentence IDs `1..300`.

Both label files contain 300 non-empty physical lines.

The frozen operational convention is:

`sentence_id n -> physical label line n`

This is recorded as a positional dataset convention. No separate
author-provided ID-to-label mapping file is present in the project
source.

## Gloss tokenization

Gloss strings are split on whitespace.

Token spelling is otherwise preserved.

Underscores remain inside a token.

Example:

`whats_up`

is one token.

## CTC vocabulary

CTC blank token:

`<blank>`

CTC blank ID:

`0`

Nonblank gloss tokens:

455

Total CTC output classes:

456

No UNK, BOS, EOS or PAD token is added to the CTC vocabulary.

## Evaluation note

The vocabulary defines the global dataset label schema.

It does not imply that every token has training support in every
LOSO or template-holdout split.

Split-specific token support must be calculated from the corresponding
training partition later.

## Generated files

`artifacts/labels/stage10_sentence_targets.csv`

`artifacts/labels/stage10_gloss_vocabulary.json`

`artifacts/labels/stage10_token_to_id.json`

`artifacts/labels/stage10_id_to_token.json`

`artifacts/labels/stage10_label_vocabulary_summary.json`

`artifacts/labels/stage10_sentence_targets.sha256`

`artifacts/labels/stage10_gloss_vocabulary.sha256`

## Duplicate label-equivalence group

The author-provided label files contain one exact duplicate target pair:

- sentence ID 39: `my book where` -> `where is my book`
- sentence ID 83: `my book where` -> `where is my book`

The two sentence IDs remain distinct dataset identities and their
recordings are not merged.

For any experiment claiming unseen complete sentence/template
generalisation, IDs 39 and 83 must be assigned together so that one
identical target cannot appear in training while the other appears in
test.
