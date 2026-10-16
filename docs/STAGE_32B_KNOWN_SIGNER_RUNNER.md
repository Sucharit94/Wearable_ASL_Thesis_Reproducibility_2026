# Stage 32B - Known-signer resumable runner

## Status

The Stage-32 known-signer execution runner is frozen before any outer-test
score. The static dry run passed without starting training, creating a
checkpoint, constructing an outer-test dataset, or computing a score.

## Frozen execution order

- 25 outer evaluations: five folds by five models
- fold-major model order: G1, C1, C2, C3, C4
- 75 registry identities: parent, inner selection, and full refit
- one command executes at most one parent outer evaluation

## Selection and refit

G1 completes all 20 inner-selection epochs. Compact models have an 80-epoch
cap, cannot stop before epoch 15, and use patience 8. Epoch E* is chosen by
lower validation WER, higher exact-sentence accuracy, lower validation CTC
loss, then earlier epoch. The model, optimizer, and RNG are reset before a
fresh refit on all non-test samples for exactly E* epochs.

The outer-test dataset is constructed only after the refit checkpoint has
reached E*. It is evaluated once, decoded only to valid output lengths, and
cannot affect training or epoch selection.

## Runtime gate

The real runner requires an NVIDIA T4, a clean checkout whose HEAD is the
annotated `stage32-known-runner-v1` tag, the frozen Stage-17 cache hash, and an external
durable artifact root. A verified OOM is the only allowed reason to halve a
physical batch. Gradient accumulation preserves the configured effective
batch, and the attempt restarts from the seed.

All five models run in float32 with automatic mixed precision disabled. This
policy was frozen before outer scoring after the T4 FP16 G1 preflight produced
non-finite gradients while the equivalent float32 preflight passed.

## Resume contract

Every epoch writes an atomic trusted checkpoint containing model, optimizer,
scheduler, AMP scaler, RNG, and training state. Completed outer predictions
are protected by a one-evaluation marker and hashes. Parent runs satisfy the
Stage-24 completion-artifact contract.

## Dry run

- report SHA-256: `4eeeb596cd7b25a4633c949747ec5d9532a3e983e8d5c6e13f1ab6f7811246d5`
- training started: `False`
- scores computed: `False`
- checkpoints created: `False`
- final status: `PASSED`
