# Stage 4 — Raw Recording Manifest

## Status

All project raw CSV recordings were independently enumerated and
SHA-256 hashed.

## Source

CSV root:

`data/raw/gu_author_dataset/csv`

No external-master path is used by the manifest builder.

## Recording identity

Each recording stores:

- stable sample ID
- subject
- sentence ID
- repetition ID
- original filename
- project-relative source path
- byte size
- SHA-256

Stable sample-ID format:

`gu_sNN_sentNNN_repNNN`

## Results

Total recordings hashed: 7,400

Unique sample IDs: 7,400

Unique relative paths: 7,400

Total CSV bytes: 700,937,794

### Observed subject counts

- subject1: 6,000 recordings
- subject2: 1,000 recordings
- subject3: 400 recordings

## Integrity guards

`label_no_order.txt`: unchanged from Stage 3

`label_order.txt`: unchanged from Stage 3

Representative Stage 3 CSV: unchanged

## Dataset recording fingerprint

`b765b500b3899e167cf98e4e431cb27bb06bff03faba592242fcfc661bc84aba`

## Manifest file SHA-256

`34ee6ef501fa038b8db79bd67a0bb8ea24edb6ed438e6d3dec32a5af697403f3`

## Outputs

`artifacts/manifests/raw_manifest.csv`

`artifacts/manifests/raw_manifest_summary.json`

`artifacts/manifests/raw_manifest.sha256`

`artifacts/manifests/recording_set_fingerprint.sha256`

## Important boundary

Stage 4 does not parse sensor-array values, infer feature semantics,
detect duplicate content, create data splits, normalize features,
or create model inputs.

Those operations belong to later stages.
