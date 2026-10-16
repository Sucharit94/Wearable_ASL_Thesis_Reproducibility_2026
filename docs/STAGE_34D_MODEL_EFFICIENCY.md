# Stage 34D — Model Efficiency Benchmark

## Protocol

All five untrained architecture instances are measured on the same CPU
using float32, one PyTorch intra-op thread, evaluation mode and inference
mode. Each model runs in a fresh process. Inputs have shape B×108×38.
Latency is the median after two warm-up iterations. Peak RSS is an
isolated-process measurement and includes model and runtime allocations.
The benchmark does not train a model and is not a GPU-speed claim.

## Runtime

- Python: 3.11.9
- PyTorch: 2.14.0+cpu
- Platform: Windows-10-10.0.26200-SP0
- Processor: Intel64 Family 6 Model 126 Stepping 5, GenuineIntel

## Results

| Model | Parameters | Reduction vs G1 | State dict MiB | Batch-1 ms | Batch-16 samples/s | Peak RSS increment MiB |
|---|---:|---:|---:|---:|---:|---:|
| G1 | 4,791,688 | 0.00% | 18.292 | 12.817 | 119.10 | 69.06 |
| C1 | 392,488 | 91.81% | 1.506 | 2.183 | 1705.92 | 20.70 |
| C2 | 656,168 | 86.31% | 2.514 | 4.525 | 856.19 | 25.80 |
| C3 | 237,352 | 95.05% | 1.287 | 2.090 | 1726.05 | 16.85 |
| C4 | 348,604 | 92.72% | 1.406 | 24.897 | 41.73 | 99.76 |

## Interpretation boundary

Parameter count and serialized size are architecture properties.
Latency, throughput and RSS are hardware/software dependent and must
be reported with this runtime identity. They should be used for
within-machine comparison, not as universal deployment numbers.
