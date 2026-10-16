# Stage 24 — Experiment Registry and Resume Safety

## Experiment identity

Every run is bound to an immutable identity containing model, phase, split,
manifest hash, vocabulary hash, configuration hash, Git commit, seed,
normalization, augmentation and required hardware.

Configuration hash stable across JSON key order:

True

Changed configuration rejected:

True

## Checkpoint contract

Checkpoint schema version:

1

Verified epoch:

7

Required RNG states:

- Python
- NumPy
- PyTorch CPU
- PyTorch CUDA when execution occurs on the reported T4

The checkpoint contains model, optimizer, optional scheduler/AMP scaler,
epoch-selection information and RNG states.

## Resume decisions

New run:

START

Interrupted run with matching checkpoint:

RESUME

Completed run:

SKIP_COMPLETE

## Completion safety

Required run artifacts:

13

Missing-artifact guard passed:

True

Completion marker written:

True

## Frozen dependency chain

Stage 21 padding-safe decoder:

True

Stage 22 length-safe processing:

True

Stage 23 perfect-control metric errors:

0
