# Stage 43 All Five Model Statistical Validation

## Hypotheses

For the omnibus WER analysis:

H0: mu_G1 = mu_C1 = mu_C2 = mu_C3 = mu_C4.

H1: at least one model mean differs.

For each paired comparison A versus B:

H0: mu_d = 0, where d is the paired cluster-WER difference A minus B.

H1: mu_d is not equal to 0.

For exact sentence correctness, Cochran's Q tests whether all five paired
correctness probabilities are equal. Pairwise McNemar tests compare the
two discordant cells for each model pair.

## Formulae

WER = (S + D + I) / N

F = MS_model / MS_error

t = d_bar / (s_d / sqrt(n))

alpha = 0.05; reject H0 only when the relevant adjusted p-value is below alpha.

## Omnibus Results

Repeated-measures ANOVA: F(2.118, 777.174) = 1381.425, Greenhouse-Geisser corrected p <0.001, epsilon = 0.529.

Friedman robustness check: chi-square(4) = 1033.157, p <0.001, Kendall's W = 0.702.

Cochran's Q for exact correctness: Q(4) = 16940.293, p <0.001.

The global null hypothesis was rejected for both WER and exact sentence
correctness. At least one of the five recognizers therefore differed.

## C2 and C4

The pooled-WER difference C2 minus C4 was -0.000859. The 95% cluster-bootstrap CI was [-0.007594, 0.005857] and the Holm-adjusted p-value was 0.811119.

The supplementary paired t-test on equal-weight subject-sentence cluster WER also did not distinguish C2 and C4: t(367) = 1.367, Holm-adjusted p = 0.172332.

Pooled WER weights tokens, whereas mean cluster WER gives every subject-sentence
cluster equal weight. A small direction change between these estimands is possible
and does not alter the conclusion that the C2-C4 difference was unresolved.

## Thesis Interpretation

The omnibus tests establish that the five recognizers did not perform equally.
Corrected pairwise analyses identify the model pairs responsible for that global
difference. A small p-value does not mean that a model is good; the direction and
magnitude of the WER or accuracy difference determine which model performed better.
C2 was not declared statistically superior to C4. Selection between them therefore
also considered fold stability, exact accuracy, parameters and downstream language
quality.
