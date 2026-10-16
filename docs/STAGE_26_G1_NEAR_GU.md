# Stage 26 — G1 Dimension-Consistent Near-Gu Model

## Scientific identity

G1 is a dimension-consistent near-Gu implementation following the reported
Conv2D, two-BiLSTM and CTC structure. It is not claimed as an exact paper
reproduction.

The source architecture contains a dimensional ambiguity. This implementation
explicitly uses 256 hidden units per direction, producing 512 output features
from each BiLSTM layer.

The input contains 38 wearable motion-capture-derived joint-rotation features.

## Window adapter

- Window size: 36 x 38
- Temporal stride: 18
- Complete valid windows only
- Padded windows entering Conv2D: False

## Conv2D shape trace

- Input: [2, 1, 36, 38]
- After pool 1: [2, 32, 18, 19]
- After pool 2: [2, 64, 6, 6]
- After pool 3: [2, 128, 3, 3]
- Flattened embedding: [2, 1152]

## Recurrent and CTC output

- Two packed bidirectional LSTM layers
- Hidden size: 256 per direction
- Recurrent output size: 512
- Raw CTC classes: 456
- Blank ID: 0

## Parameters

- Conv2D encoder: 93120
- BiLSTM 1: 2887680
- BiLSTM 2: 1576960
- Classifier: 233928
- Total: 4791688

## Safety verification

- Padding-invariant valid logits: True
- Padded output logits zero: True
- Finite CTC loss: True
- Missing gradients: 0
- Non-finite gradients: 0
- Backward test passed: True

## Runtime

- PyTorch: 2.14.0+cpu
- Verification device: CPU
- Local CUDA available: False
