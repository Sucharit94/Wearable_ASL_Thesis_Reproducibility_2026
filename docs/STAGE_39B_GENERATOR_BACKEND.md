# Stage 39B — Frozen Constrained Generator

The L2 backend is `google/flan-t5-base`, pinned to revision
`7bcac572ce56db69c1ea7c8af255c5d7c9672fc2` under Apache-2.0.

Generation is greedy, temperature-zero, float32 inference on CUDA.
No fine-tuning, API key or outer-test English reference is used.
Responses are appended to durable storage and resume by request ID.
The generator-reported constraint fields are retained but not trusted;
Stage 40 independently calculates content support and correction harm.
