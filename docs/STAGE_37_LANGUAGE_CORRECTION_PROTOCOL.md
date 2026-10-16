# Stage 37 — Language-Correction Protocol Freeze

## Purpose

Stage 37 freezes the evaluation design for converting recognized ASL gloss into grammatical English. It performs no recognizer training, language-model inference or test scoring.

## Frozen model roles

- C2 is the primary lightweight recognizer pipeline.
- G1 is retained only as the recognition-quality reference.
- C1, C3 and C4 remain full comparative arms for every language system.
- Oracle gloss is a language-component ceiling and must never be mixed with predicted-gloss results.

## Systems

- L0: raw normalized recognized gloss (no correction).
- L1: deterministic nearest-template retrieval using fold-training data only.
- L2: constrained generation using the same training-only example bank.
- L3: a template-holdout challenge protocol, not a fourth correction model.

## Leakage firewall

For model selection, L1/L2 may use inner-train examples and inner validation only for tuning. After all settings are frozen, the final bank may combine inner train and validation; outer-test samples and references remain inaccessible until scoring.

## Five-fold plan

| Fold | Selection bank | Validation | Final bank | Outer test |
|---:|---:|---:|---:|---:|
| 1 | 5328 | 592 | 5920 | 1480 |
| 2 | 5328 | 592 | 5920 | 1480 |
| 3 | 5328 | 592 | 5920 | 1480 |
| 4 | 5328 | 592 | 5920 | 1480 |
| 5 | 5328 | 592 | 5920 | 1480 |

## Judgement rule

A correction system is useful only if it improves English WER and exact sentence accuracy without unacceptable unsupported-content or correction-harm rates. Fluency or accuracy alone is insufficient.

## Next execution stages

- Stage 38: assemble immutable predictions for all five models and execute L0/L1.
- Stage 39: freeze the generator identity, execute L2 once and record every response.
- Stage 40: paired comparison, confidence intervals, error analysis and thesis tables.
