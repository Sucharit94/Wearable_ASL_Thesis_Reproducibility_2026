# Stage 17 — Raw Binary Cache

## Purpose

Convert the immutable author CSV recordings into one contiguous binary
execution cache for faster local and Colab loading.

The original CSV files remain the scientific source of truth.

## Source

Recordings:

7,400

Feature width:

38

Total time steps:

1,829,320

All manifest raw-file SHA-256 values verified:

7,400

## Binary representation

File:

`cache/raw/stage17_raw_f32.bin`

Representation:

little-endian float32

Layout:

C-contiguous row-major

Logical shape:

`(1829320, 38)`

Size:

278,056,640 bytes

SHA-256:

`9d160032bd6879877b1e374ef5cc8fce8ae74bbdfe14b1bc5096790045c35e2a`

## Index

`artifacts/cache/stage17_raw_cache_index.csv`

Each recording stores its contiguous cache row range:

`cache_row_start : cache_row_end_exclusive`

The original variable sequence length is therefore preserved exactly.

## Float32 conversion

The author CSV values are parsed first as float64 and converted to
float32 for the execution cache.

Maximum observed absolute cast error:

4.999999987376214e-07

The float32 cache is not treated as a replacement for the immutable
CSV source.

## Full verification

Verified cache slices:

7,400

Failed cache slices:

0

Every cached slice was compared exactly against a fresh float32 cast of
its original CSV recording.

## No preprocessing

Stage 17 performs no:

- normalization,
- filtering,
- angle unwrapping,
- resampling,
- truncation,
- padding,
- augmentation,
- windowing,
- label processing,
- split-specific transformation.

Those decisions belong to later stages.
