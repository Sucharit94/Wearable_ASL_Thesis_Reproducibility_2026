# Stage 21 — CTC Loss and Length-Safe Decoder

## CTC schema

Output classes:

456

Blank ID:

0

Temporary target padding:

-1

## Reference targets

All 300 Stage-10 sentence templates were packed into one flat CTC
target stream and reconstructed using their frozen target lengths.

Round-trip failures:

0

Blank IDs in packed targets:

0

Padding IDs in packed targets:

0

## Decoder order

The frozen greedy decoder performs:

1. slice each sample to its valid model output length;
2. argmax over output classes;
3. collapse consecutive repeated predictions;
4. remove CTC blank ID 0.

Padding is therefore never decoded.

## CTC loss

Model logits are represented as:

`B x T x C`

The PyTorch loss wrapper applies log-softmax and converts this to:

`T x B x C`

Reference targets are packed using their true target lengths.

Before loss calculation, every sample is checked against the exact CTC
minimum alignment rule.

`zero_infinity=False` is intentional. Invalid CTC examples must be
detected rather than silently converted to zero loss.

## Local runtime

PyTorch available locally:

False

This does not affect the frozen scientific contract. PyTorch is loaded
lazily by the loss wrapper because GPU model execution occurs in the
Colab environment.

## Stage-20 dependency

Selected Stage-20 stride:

18

Selected-rule infeasible recordings:

0
