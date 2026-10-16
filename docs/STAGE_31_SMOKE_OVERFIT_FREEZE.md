# Stage 31 — Smoke, Tiny Overfit and Freeze

## Status

Stage 31 passed: **True**

This stage is a technical integration gate. It produces no outer-test
score, saves no model checkpoint, and its trained weights must never
be reused.

## Common fixture

- Source: known-signer Fold-1 training partition only
- Selection: 20 shortest recordings, ties by sample ID
- Same ordered sample IDs for all five models: yes
- Normalization: fit on the entire Fold-1 training partition only
- Saved reusable normalization statistics: no
- Fixture SHA-256: `f6922573a815f29336e454207058cb8c89c3da09945bfe6691c5b526f26d48ce`

## Results

| Model | Smoke | Initial loss | Best loss | Relative decrease | Epochs | Microbatch |
|---|---:|---:|---:|---:|---:|---:|
| G1 | True | 21.698499 | 17.786584 | 18.028501% | 2 | 4 |
| C1 | True | 120.121483 | 109.383926 | 8.938914% | 2 | 4 |
| C2 | True | 131.210052 | 77.082391 | 41.252679% | 2 | 4 |
| C3 | True | 128.715424 | 112.755748 | 12.399194% | 2 | 4 |
| C4 | True | 21.649311 | 15.742580 | 27.283688% | 2 | 4 |

## Frozen rule

Every model must complete a finite forward/backward pass with no
missing or non-finite gradients and must reduce deterministic
evaluation loss by at least 0.0100%. Batch
fallback is allowed only after a verified out-of-memory error; the
model is then rebuilt from the frozen seed.

## Configuration and code boundary

Eleven JSON-compatible YAML files are hashed in
`artifacts/config/stage31_config_hashes.json`. After verification,
commit this stage and create annotated tag `stage31-freeze-v1`.
No Stage-32 outer run may start before that tag points to the
Stage-31 commit and the worktree is clean.
