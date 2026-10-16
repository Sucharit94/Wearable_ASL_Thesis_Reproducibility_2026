# Stage 33B - Resumable LOSO runner

This runner executes 15 unseen-signer evaluations in LOSO-major order. Each
invocation completes at most one model-direction pair and writes only to an
external durable artifact root.

The held-out subject is excluded from normalization, epoch selection, early
stopping, scheduling and refit. One outer prediction file is scored as LOSO-all,
template-supported, token-supported and token-unsupported views. The support
views never train separate models.

All models use float32 without AMP on NVIDIA T4. Epoch checkpoints are atomic
and resumable. Checkpoint loading explicitly uses trusted project mode, and
parent CSV finalization uses a union schema for selection and refit columns.

Runner tag required before scoring: `stage33-loso-runner-v1`

Dry-run SHA-256: `67008559f95aaf73f33860d39fbf1c88435cf27eb96ad36e1016ff25feef04f4`
Contract SHA-256: `102217b9e9986d8b5c0814c01a523112c7969fbc1d21b71e607df862ddb2effe`
