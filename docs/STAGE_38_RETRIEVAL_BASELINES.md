# Stage 38 — Immutable Predictions and Retrieval Baselines

Stage 38 reuses frozen outer-test predictions. It does not train or rescore a recognizer and does not call a generative language model.

## Results

| Recognizer | System | English WER | Exact accuracy | Content F1 | Unsupported content |
|---|---|---:|---:|---:|---:|
| G1 | L0_raw_gloss | 0.546866 | 0.097027 | 0.844096 | 0.000000 |
| G1 | L1_training_only_retrieval | 0.000058 | 0.999865 | 0.999865 | 0.000000 |
| C1 | L0_raw_gloss | 0.569206 | 0.086757 | 0.802545 | 0.000000 |
| C1 | L1_training_only_retrieval | 0.018373 | 0.975270 | 0.979037 | 0.000000 |
| C2 | L0_raw_gloss | 0.562047 | 0.082432 | 0.817902 | 0.000000 |
| C2 | L1_training_only_retrieval | 0.004715 | 0.987973 | 0.989700 | 0.000000 |
| C3 | L0_raw_gloss | 0.701380 | 0.034054 | 0.476531 | 0.000000 |
| C3 | L1_training_only_retrieval | 0.300431 | 0.640811 | 0.684331 | 0.000000 |
| C4 | L0_raw_gloss | 0.559546 | 0.092973 | 0.822163 | 0.000000 |
| C4 | L1_training_only_retrieval | 0.008453 | 0.988784 | 0.990140 | 0.000000 |

## Interpretation boundary

L1 is a deterministic closed-set retrieval baseline. Because every fold's training partition contains all 300 sentence templates, strong L1 results demonstrate template-based correction, not unrestricted generation to unseen sentence structures.

Unsupported-content rate is zero by construction for L0 and L1: L0 repeats the source and L1 returns a verified training-catalog sentence. It becomes a discriminative safety metric for constrained generation in Stage 39.

All five recognizers receive identical L0/L1 processing. C2 remains the primary lightweight recommendation and G1 remains the recognition-quality reference. No conclusion is based on accuracy alone.
