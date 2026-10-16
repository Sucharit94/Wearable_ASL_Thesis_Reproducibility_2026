# Stage 28 — C2 Conv1D-BiLSTM Self-Attention CTC

## Identity

C2 is the recurrent self-attention comparison. It uses the exact C1
Conv1D and packed-BiLSTM backbone followed by one four-head
self-attention block with true padding masking, a residual connection
and post-normalization.

## Parameter-count resolution

The frozen 256-dimensional four-head architecture contains exactly
656168 trainable parameters. This matches the
expanded roadmap's 656,168 count.

It does not satisfy the older descriptive wording of under 0.5M.
Architecture identity takes precedence, and this discrepancy is recorded
before training or inspection of experimental results.

## Verification

- Parameter check passed: True
- Output shape: [3, 10, 456]
- Output lengths: [8, 9, 10]
- Stage-22 formula matched: True
- Padded logits zero: True
- Padding invariance difference: 0.0
- Attention mask passed: True
- Masked-key nonzero weights: 0
- Residual test difference: 0.0
- CTC loss finite: True
- Missing gradients: 0
- Non-finite gradients: 0
- Backward test passed: True

## Training defaults

- Optimizer: AdamW
- Learning rate: 5e-05
- Weight decay: 0.0001
- Epoch cap: 80
- Early-stopping patience: 8
- Physical batch size: 16
- Gradient clipping norm: 5.0
- Scheduler: ReduceLROnPlateau
