# Stage 25 — Shared Model Interface and Configuration Schema

## Model catalog

Five frozen recognisers:

c1, c2, c3, c4, g1

## Shared output contract

All models return raw `B x T x 456` logits and exact valid output lengths.

CTC blank ID:

0

Target padding:

-1

## Input adapters

Direct-sequence models:

c1, c2, c3

Windowed 2D models:

c4, g1

Window length:

36

Window stride:

18

## Architecture identities

G1:

window_conv2d_two_bilstm_ctc

C1:

conv1d_bilstm_ctc

C2:

conv1d_bilstm_self_attention_ctc

C3:

conv1d_transformer_encoder_ctc

C4:

tiny_swin_bilstm_ctc

C4 retains its BiLSTM backend:

True

## Frozen dependencies

Stage-22 window-length safety:

True

Stage-24 registry passed:

True

Stage-24 identity matching required:

True
