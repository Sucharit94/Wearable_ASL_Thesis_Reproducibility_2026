# Stage 29 — C3 Compact Conv1D-Transformer-CTC

## Identity

C3 is the compact Transformer comparison. It processes direct `T x 38`
sequences through two stride-2 Conv1D blocks, fixed sinusoidal positions,
two pre-normalized Transformer layers and a 456-class CTC head.

## Architecture resolution

The roadmap fixes channels 38 to 64 to 96, kernel 5/stride 2, d-model 96,
two layers, four heads, FFN 192, Transformer dropout 0.10, positional
representation and a padding mask.

Before training, Stage 29 resolves unspecified details as padding 2,
length-masked BatchNorm, ReLU, sinusoidal positions to length 1000,
GELU, pre-normalization and final LayerNorm(96). Final LayerNorm
contributes 192 parameters and reconciles the exact 237,352 total.

## Verification

- Trainable parameters: 237352
- Parameter check passed: True
- Positional encoding passed: True
- Exact padding mask passed: True
- Transformer padding difference: 0.0
- Output shape: [3, 10, 456]
- Output lengths: [8, 9, 10]
- Padded logits zero: True
- Raw-padding difference: 0.0
- CTC loss finite: True
- Missing gradients: 0
- Non-finite gradients: 0
- Backward passed: True

## Training defaults

- Optimizer: AdamW
- Learning rate: 5e-05
- Weight decay: 0.0001
- Epoch cap: 80
- Early-stopping patience: 8
- Physical batch size: 32
- Gradient clipping norm: 5.0
- Scheduler: ReduceLROnPlateau
