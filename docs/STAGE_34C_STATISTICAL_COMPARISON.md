# Stage 34C — Paired Statistical Model Comparison

## Method

The analysis uses 10,000 paired bootstrap replicates. The resampling
unit is the frozen subject-by-sentence cluster (368 clusters), so all
repetitions and predictions from all five models remain paired.
Pairwise WER differences use percentile confidence intervals. Exact
sentence correctness uses two-sided exact McNemar tests. Holm correction
controls family-wise error separately across the 10 WER comparisons and
the 10 McNemar comparisons.

## Model confidence intervals

| Model | WER | WER 95% CI | Exact accuracy | Accuracy 95% CI |
|---|---:|---:|---:|---:|
| G1 | 0.001074 | [0.000608, 0.001658] | 0.996486 | [0.994744, 0.997962] |
| C1 | 0.067215 | [0.057684, 0.077463] | 0.799595 | [0.774057, 0.824457] |
| C2 | 0.040050 | [0.033484, 0.046848] | 0.864730 | [0.842954, 0.885984] |
| C3 | 0.444667 | [0.427566, 0.462208] | 0.136486 | [0.111921, 0.161653] |
| C4 | 0.040909 | [0.034262, 0.048115] | 0.876622 | [0.857713, 0.895109] |

## Pairwise conclusions

| A | B | ΔWER A−B | 95% CI | Holm p | WER conclusion | Δ accuracy | McNemar Holm p |
|---|---|---:|---:|---:|---|---:|---:|
| G1 | C1 | -0.066142 | [-0.076342, -0.056669] | 0.0019998 | a_lower_wer | 0.196892 | 2.22507e-307 |
| G1 | C2 | -0.038976 | [-0.045706, -0.032531] | 0.0019998 | a_lower_wer | 0.131757 | 1.0804e-278 |
| G1 | C3 | -0.443593 | [-0.461162, -0.426604] | 0.0019998 | a_lower_wer | 0.860000 | 2.22507e-307 |
| G1 | C4 | -0.039835 | [-0.046925, -0.033250] | 0.0019998 | a_lower_wer | 0.119865 | 1.26874e-252 |
| C1 | C2 | 0.027165 | [0.018810, 0.035940] | 0.0019998 | b_lower_wer | -0.065135 | 1.40092e-33 |
| C1 | C3 | -0.377452 | [-0.392698, -0.362382] | 0.0019998 | a_lower_wer | 0.663108 | 2.22507e-307 |
| C1 | C4 | 0.026306 | [0.019052, 0.033888] | 0.0019998 | b_lower_wer | -0.077027 | 1.77417e-50 |
| C2 | C3 | -0.404617 | [-0.420717, -0.388743] | 0.0019998 | a_lower_wer | 0.728243 | 2.22507e-307 |
| C2 | C4 | -0.000859 | [-0.007594, 0.005857] | 0.811119 | not_resolved | -0.011892 | 0.0178597 |
| C3 | C4 | 0.403758 | [0.389124, 0.418993] | 0.0019998 | b_lower_wer | -0.740135 | 2.22507e-307 |

## Interpretation

A negative ΔWER means model A has lower WER. Statistical significance
does not measure practical importance; absolute differences, relative
WER reductions, model size and runtime evidence must also be considered.
