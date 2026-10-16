# Stage 30 — C4 Tiny-Swin-BiLSTM-CTC

## Scientific identity

C4 is the task-specific Tiny-Swin window-attention comparison. It is
not claimed as an exact reproduction of standard image Swin Transformer.
The frozen Stage-25 catalog is authoritative: the primary C4 family is
`tiny_swin_bilstm_ctc`, so C4 retains a packed BiLSTM after Tiny-Swin.

## Input adapter

- Raw input: `B x T x 38`
- Complete sensor windows: `N x 1 x 36 x 38`
- Temporal stride: 18
- Window count: `1 + floor((T - 36) / 18)`
- Feature-axis permutation: none
- 224 x 224 image resizing: none
- Padded windows entering Tiny-Swin: false

## Tiny-Swin encoder

- Patch projection: Conv2D 1→32, kernel/stride 2 x 2
- Patch grid: 18 x 19
- Stage depths: [2, 2]
- Heads: [2, 4]
- Window: 8 x 5
- Shift: 4 x 2 in the second block of each stage
- Patch merge: 2 x 2, 32→64
- MLP ratio: 4
- Activation: GELU
- Relative-position bias: true
- One masked-mean 64-D embedding per valid sensor window

## Sequence backend and CTC

- Packed one-layer BiLSTM
- Hidden size: 96 per direction
- Output width: 192
- LayerNorm(192), dropout 0.15
- Raw CTC logits: 456 classes
- Blank ID: 0

## Verification

- Trainable parameters: 348604
- Under 1.5M budget: True
- Shape trace passed: True
- Shifted block executed in stage 1: True
- Shifted block executed in stage 2: True
- Spatial-padding invariance difference: 0.0
- Patch-merge padding difference: 0.0
- Stage-22 window formula passed: True
- Feature ordering preserved: True
- Output shape: [3, 5, 456]
- Output lengths: [3, 4, 5]
- Padded logits zero: True
- Raw-padding invariance difference: 0.0
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
- Physical batch size: 16
- Gradient clipping norm: 5.0
- Scheduler: ReduceLROnPlateau
- Primary augmentation: none
