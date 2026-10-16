# Stage 36 — Fold-Level Error Analysis and Thesis Figures

## Purpose

This stage diagnoses variation and failure modes in the 25 frozen
mixed-subject five-fold evaluations. It does not train or rescore a model.

## Fold stability

| Model | Mean WER | SD | Range | Best fold | Worst fold | Fold wins |
|---|---:|---:|---:|---:|---:|---:|
| G1 | 0.001074 | 0.000868 | 0.001969 | 2 | 1 | 5 |
| C1 | 0.067215 | 0.007039 | 0.017538 | 2 | 5 | 0 |
| C2 | 0.040050 | 0.003225 | 0.008769 | 5 | 3 | 0 |
| C3 | 0.444667 | 0.006660 | 0.016822 | 4 | 2 | 0 |
| C4 | 0.040909 | 0.007042 | 0.015569 | 5 | 2 | 0 |

## Error composition

| Model | Substitutions | Deletions | Insertions | Dominant type |
|---|---:|---:|---:|---|
| G1 | 1 | 27 | 2 | deletion |
| C1 | 81 | 1649 | 148 | deletion |
| C2 | 66 | 721 | 332 | deletion |
| C3 | 595 | 11801 | 28 | deletion |
| C4 | 104 | 932 | 107 | deletion |

## C2 and C4

C2 obtained lower fold WER in 3 folds; C4 obtained lower fold WER in 2 folds. The paired Stage 34 inference remains authoritative: the overall WER difference is not resolved at 95% confidence.

## Thesis interpretation

- G1 is the stable recognition-quality reference, not the lightweight recommendation.
- C2 is the selected lightweight quality–efficiency compromise.
- C2 and C4 exchange fold-level advantages; this supports cautious statistical wording.
- Deletion-heavy outputs indicate CTC under-generation rather than simple class confusion.
- C3's low parameter count does not compensate for its recognition failure.

## Figures

- stage36_fold_wer.svg: cross-fold WER variation.
- stage36_quality_efficiency.svg: WER versus trainable parameters.

Both SVG files are vector graphics suitable for the thesis document.
