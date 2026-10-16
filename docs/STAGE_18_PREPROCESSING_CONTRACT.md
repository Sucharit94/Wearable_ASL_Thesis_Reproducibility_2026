# Stage 18 — Preprocessing Contract

## Primary representation

The primary recognition input remains the supplied 38-dimensional
rotation-coordinate time series.

No temporal angle unwrapping is performed.

## Primary normalization

Each feature is standardized using:

`(x - training_mean) / training_std`

The mean and standard deviation are fitted only from the exact current
training partition.

Statistics are calculated across all valid training time steps.
Therefore the normalization is frame-weighted.

Population standard deviation (`ddof = 0`) is used.

Statistics are accumulated in float64 and normalized sequences are
returned as float32.

## Leakage rule

Outer-test recordings never contribute to outer-fold normalization.

Held-out LOSO subjects never contribute to LOSO normalization.

Held-out template groups never contribute to template-holdout
normalization.

During inner model selection, inner-validation recordings do not
contribute to fitted normalization statistics.

After model-selection decisions are frozen, final refitting may fit
new statistics using the complete outer-training pool while still
excluding the final test partition.

## Primary disabled operations

No filtering, smoothing, angle unwrapping, resampling, truncation,
augmentation or per-sample standardization is used in the primary
architecture comparison.

## Padding

Normalization occurs before batching and padding.

Future padding uses zero in normalized space and must always be paired
with valid sequence lengths or masks.

## Model adapters

C1, C2 and C3 receive the normalized variable-length `T x 38`
sequence directly.

G1 and C4 receive the same normalized sequence before their respective
model-specific temporal/window adapters.

The G1/C4 temporal stride is not selected in Stage 18.

Stage 20 freezes the shared CTC-feasible temporal rule.

## Circular ablation

The Stage-8 circular representation remains optional rather than
primary.

It maps each 38-dimensional degree-scaled rotation vector to an
interleaved 76-dimensional representation:

`sin(f0), cos(f0), ..., sin(f37), cos(f37)`

If evaluated later, its own normalization statistics must again be
fitted only from the corresponding training partition.

## Numeric statistics

Stage 18 freezes the preprocessing algorithm.

It deliberately does not freeze one global vector of means and
standard deviations.

Global-dataset normalization is prohibited because it would allow
held-out data to affect preprocessing.

## Test execution

The project root is exposed to pytest through `pytest.ini`.

The preferred test command is:

`python -m pytest`

## Frozen hashes

Preprocessing config:

`f1ba3d5e37210a135f4fd618c21aec61136096e5514b0606ce0bdc269b8b5731`

Implementation:

`58c91ed2b51874ae65c1b490c453497a9029fcd00a44dcdbacf41210096817f0`

Unit tests:

`957f51557aac3f7086043687c3bb714e0c1398894e21c4308547af953208e397`

Pytest configuration:

`0e6875a9a85c10c54f81fe5fdb2f2f7c84692c3450bf04100327043569343aab`
