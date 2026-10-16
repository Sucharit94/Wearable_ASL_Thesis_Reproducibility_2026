# Stage 5 — Dataset Integrity Audit

## Purpose

Verify that every raw recording can be interpreted as a numerical
variable-length array with exactly 38 feature columns.

## Source

The audit consumes the frozen Stage 4 raw manifest.

Manifest SHA-256:

`34ee6ef501fa038b8db79bd67a0bb8ea24edb6ed438e6d3dec32a5af697403f3`

## Results

Recordings expected: 7,400

Recordings audited: 7,400

Recordings passed: 7,400

Recordings failed: 0

Parse failures: 0

File-size mismatches: 0

Empty recordings: 0

Wrong-width recordings: 0

Recordings containing NaN or Inf: 0

Expected feature width: 38

Total valid timesteps: 1,829,320

Total valid numeric values: 69,514,160

Globally constant features: 0

Zero-based constant-feature indices:

`[]`

## Important scope boundary

This stage does not interpret sequence-length distributions.

Detailed length analysis belongs to Stage 6.

This stage does not assign anatomical meaning to the 38 columns.

Feature semantics belong to Stage 7.

This stage does not normalize, filter, augment, split, cache or train
the dataset.
