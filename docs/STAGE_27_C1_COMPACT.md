# Stage 27 — C1 Compact Conv1D-BiLSTM-CTC

## Scientific identity

C1 is the lightweight recurrent baseline. It consumes a direct
variable-length `T x 38` feature sequence and produces raw
`B x T_out x 456` CTC logits.

## Architecture

Two Conv1D blocks use channels 38 to 64 to 96, kernel 5, stride 2,
padding 2, length-masked BatchNorm1D, ReLU and dropout 0.10.

One packed BiLSTM uses 128 hidden units per direction. LayerNorm(256)
and dropout 0.15 precede the 456-class classifier. LayerNorm supplies
the 512 parameters required to reconcile the frozen parameter total.

## Verification

- Trainable parameters: 392488
- Expected parameters: 392488
- Parameter check passed: True
- Output shape: [3, 10, 456]
- Output lengths: [8, 9, 10]
- Stage-22 length formula matched: True
- Padded logits zero: True
- Padding maximum absolute difference: 0.0
- CTC loss finite: True
- Missing gradients: 0
- Non-finite gradients: 0
- Backward test passed: True

## Frozen training defaults

- Optimizer: AdamW
- Learning rate: 5e-05
- Weight decay: 0.0001
- Epoch cap: 80
- Early-stopping patience: 8
- Physical batch size: 32
- Gradient clipping norm: 5.0
- Scheduler: ReduceLROnPlateau

## Runtime

- PyTorch: 2.14.0+cpu
- CUDA available locally: False
- Verification device: cpu
