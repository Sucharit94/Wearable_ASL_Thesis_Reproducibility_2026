# Stage 22 — Length-Safe Sequence Processing

## Core rule

Every sample retains an exact valid temporal length from input through
the final CTC logits.

Padded tensor time is never substituted for valid sample length.

## Direct models

C1, C2 and C3 begin from the raw valid sequence length `T`.

Operations that preserve time retain the same valid length.

Any later convolution or downsampling operation must apply its exact
mathematical output-length formula.

## G1 and C4

Stage 20 froze:

- window length: 36
- stride: 18

The full-window temporal length is:

`1 + floor((T - W) / S)`

All 7,400 Stage-22 recomputations match the Stage-20 frozen audit.

## Conv1D

Exact formula:

`floor((L + 2P - D*(K-1) - 1) / S) + 1`

No code may substitute an approximate expression such as `T/2` or
`T/4`.

## BiLSTM

BiLSTM itself does not alter valid temporal length.

Variable-length recurrent processing must use packed sequences or an
equivalent method that prevents padded positions from influencing the
recurrent representation.

## Transformer

The Transformer receives an exact padding mask generated from the
propagated valid lengths.

## Tiny-Swin

The temporal-feature Swin path begins from the Stage-20 full-window
sequence.

No global temporal pooling is permitted before CTC.

Every temporal patch/merge operation must propagate exact lengths.

## CTC boundary

The final propagated output length is shared by:

- the Stage-21 pre-loss feasibility guard,
- CTCLoss,
- greedy CTC decoding.

Therefore loss and decoding cannot disagree about which temporal
positions are valid.
