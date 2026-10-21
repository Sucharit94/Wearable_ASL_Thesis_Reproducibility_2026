# Selected Fold-1 pretrained checkpoints

This directory records the identities, configurations, reported Fold-1 results
and SHA-256 values for one trained checkpoint per recognizer: G1, C1, C2, C3
and C4. The binary `best.pt` files are not stored in the Git source tree.

## Distribution status

The checkpoint supplement has been prepared and validated by the researcher.
At the time these source instructions were added, it had not yet been uploaded
as a GitHub Release asset. Release v1.0.1 contains frozen analysis evidence and
does not contain trained checkpoints. The researcher must upload
`fold01_pretrained_checkpoints.zip` to a new release before advertising it as
a publicly downloadable checkpoint package. Consult the release page for the
actual published assets; do not infer availability from this directory.

The expected supplement contains a top-level `pretrained_checkpoints/`
directory. Extract it into the repository root. The resulting files include:

- `pretrained_checkpoints/checkpoint_manifest.csv`;
- `pretrained_checkpoints/checkpoints/fold01/<model_id>/identity.json`;
- `pretrained_checkpoints/checkpoints/fold01/<model_id>/best.pt`.

## Verify a locally available checkpoint supplement

After installing Python 3.11, NumPy and a suitable CPU PyTorch build, run:

```powershell
python scripts/verify_pretrained_models.py --project .
```

The verifier checks file sizes and hashes, run identities, strict weight
loading, model parameter counts and a CPU forward pass on a synthetic zero
input of shape `(1, 72, 38)`. It does not read participant recordings, measure
recognition accuracy or reproduce the original training. Its elapsed times
include loading and are not the thesis inference-latency measurements.

These checkpoints are a demonstration subset: five models from Fold 1 rather
than all 25 model/fold combinations used for the pooled thesis results. The
complete five-fold frozen predictions remain in the v1.0.1 release assets.

## Raw-input inference limitation

The prepared supplement does not contain the matching fitted normalization
vectors. The original runner stored these in the successful attempt's
`full_refit/normalization.json`, separately from the parent `best.pt` export.
A real-recording prediction additionally needs those matching statistics,
authorized input data, the exact feature order and a preprocessing/decoding
entry point. Do not replace the fitted vectors with test-sample statistics.

## Build the supplement from the researcher's local files

```powershell
python scripts/package_fold1_checkpoints.py --project path\to\compact_project --output path\to\fold01_pretrained_checkpoints.zip
```

The packager checks the manifest hashes, reads only an explicit file list and
excludes Git history, other folds, last.pt and temporary .best.pt.* files. It
does not delete originals or upload anything. Its output is created exclusively
and cannot overwrite an existing archive. Re-run the pretrained verifier after
extracting the resulting archive into a clean public-source checkout.
