# Stage 6 — Sequence-Length Audit

## Purpose

Measure the observed variable sequence length `T` for all fresh raw
recordings.

The sensor tensor remains:

`X in R^(T x 38)`

where `T` varies between recordings.

## Dataset

Recordings analysed: 7,400

Feature width: 38

## Overall observed sequence-length statistics

Minimum: 81

Maximum: 831

Mean: 247.205

Median: 241.000

Sample standard deviation: 69.617

P01: 111.000

P05: 151.000

P10: 171.000

P25: 201.000

P50: 241.000

P75: 291.000

P90: 341.000

P95: 381.000

P99: 431.000

## Shortest recording

Sample ID: `gu_s01_sent165_rep012`

Subject: `subject1`

Length: 81

## Longest recording

Sample ID: `gu_s02_sent029_rep080`

Subject: `subject2`

Length: 831

## Interpretation boundary

The observed raw-data distribution is the project source of truth.

No recording is truncated, stretched or resampled merely to agree with
a range reported in the publication.

Detailed CTC feasibility is not evaluated in this stage because target
gloss sequences have not yet been frozen.

## Outputs

`artifacts/audits/stage6_sequence_length_summary.json`

`artifacts/audits/stage6_subject_length_summary.csv`

`artifacts/audits/stage6_shortest_recordings.csv`

`artifacts/audits/stage6_longest_recordings.csv`

`artifacts/audits/stage6_length_threshold_counts.csv`

`artifacts/figures/stage6_length_histogram.png`

`artifacts/figures/stage6_length_boxplot_by_subject.png`
