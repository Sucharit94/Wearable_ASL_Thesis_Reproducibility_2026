# Stage 39A — Constrained-Generation Input Freeze

Stage 39A freezes a reference-free, resumable L2 generation queue. It does not call a generator.

## Safety boundary

- Inputs contain recognized gloss and training-only candidate examples.
- Outer-test English references and sample identifiers are excluded from prompts.
- API keys must never be saved in the repository.
- The backend, model identifier and revision must be recorded before Stage 39B.
- Completed responses are immutable and execution resumes by request ID.

## Queue

Unique generation requests: 11720

| Model | Fold | Unique requests | Outer samples | Deduplication |
|---|---:|---:|---:|---:|
| G1 | 1 | 309 | 1480 | 4.79× |
| C1 | 1 | 503 | 1480 | 2.94× |
| C2 | 1 | 437 | 1480 | 3.39× |
| C3 | 1 | 669 | 1480 | 2.21× |
| C4 | 1 | 464 | 1480 | 3.19× |
| G1 | 2 | 301 | 1480 | 4.92× |
| C1 | 2 | 468 | 1480 | 3.16× |
| C2 | 2 | 442 | 1480 | 3.35× |
| C3 | 2 | 624 | 1480 | 2.37× |
| C4 | 2 | 471 | 1480 | 3.14× |
| G1 | 3 | 306 | 1480 | 4.84× |
| C1 | 3 | 504 | 1480 | 2.94× |
| C2 | 3 | 449 | 1480 | 3.30× |
| C3 | 3 | 660 | 1480 | 2.24× |
| C4 | 3 | 457 | 1480 | 3.24× |
| G1 | 4 | 302 | 1480 | 4.90× |
| C1 | 4 | 501 | 1480 | 2.95× |
| C2 | 4 | 434 | 1480 | 3.41× |
| C3 | 4 | 695 | 1480 | 2.13× |
| C4 | 4 | 425 | 1480 | 3.48× |
| G1 | 5 | 301 | 1480 | 4.92× |
| C1 | 5 | 511 | 1480 | 2.90× |
| C2 | 5 | 426 | 1480 | 3.47× |
| C3 | 5 | 642 | 1480 | 2.31× |
| C4 | 5 | 419 | 1480 | 3.53× |

## Interpretation

Stage 39B is not authorized until one reproducible generator backend is selected. L2 must be compared with L0 and L1 using paired outputs; fluency alone is not evidence of improvement.
