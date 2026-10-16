# Stage 9 — Swin Topology Decision

## Frozen primary representation

The primary C4 input is the raw source sequence:

`T x 38`

The first axis is time.

The second axis contains the 38 features in their original CSV order.

## Why this representation is used

Stage 7 established the semantic inventory of the 38 rotation
coordinates but did not establish the exact semantic identity of
physical CSV columns 0-37.

Therefore the project preserves source order and makes no anatomical
adjacency claim.

## Safe interpretation

The C4 input is a:

`temporal-feature map`

Local and shifted Swin windows may model local interactions across time
and source-ordered feature positions.

Feature proximity in this representation does not imply verified
physical or anatomical proximity.

## Forbidden primary representations

The project will not:

- resize recordings to 224 x 224,
- stretch recordings to square images,
- convert the feature map to RGB,
- anatomically reorder columns without direct evidence,
- globally pool away time before CTC,
- interpolate merely for image-like appearance.

## Padding

Padding may later be required for batching or Swin patch/window
divisibility.

Such padding is artificial, must not alter the true sequence length,
and must be handled using explicit lengths or masks.

Exact padding rules are deferred to later implementation stages.

## CTC requirement

C4 must retain an ordered temporal output suitable for CTC.

A conventional image-classification Swin that globally pools the entire
time axis is not acceptable for the primary recognizer.

## Parameters deliberately not chosen in Stage 9

The following remain open:

- patch size,
- window size,
- shift size,
- embedding dimension,
- number of attention heads,
- number of Swin blocks,
- dropout.

These will be determined during C4 implementation and the frozen
training-protocol stages rather than being invented during the topology
audit.

## Circular encoding

The Stage 8 sin/cos representation remains a declared preprocessing
ablation.

It does not replace the primary `T x 38` C4 topology.
