# Reproducibility guide

## What is in Git

Git contains implementation code, configurations, tests, frozen split and
protocol contracts, compact result tables, statistical outputs, and reports.
Raw participant data, checkpoints, the Stage 17 binary cache, and large frozen
prediction/response files remain outside Git.

## Required external inputs

1. Obtain the Gu, Oku and Todoh dataset under the terms supplied by its authors.
2. Place its 7,400 CSV recordings and two label files as described in
   `DATA_ACCESS.md`.
3. Download these GitHub Release assets when rebuilding final evidence:
   - `stage32c_results_without_checkpoints.zip`
   - `stage39b_v2_results.zip`

Verify every release asset against `docs/release_asset_manifest.csv` before use.

## Environment

Use Python 3.11 for local work. Install `requirements-local.txt`, then install a
PyTorch build suitable for the machine. The recorded Colab T4 environment is in
`docs/environment/environment_colab_t4.json`; the full snapshot is in
`docs/environment/pip_freeze_colab.txt`.

## Rebuild Stage 43 statistics

```powershell
python scripts/build_stage43.py `
    path\to\stage32c_results_without_checkpoints.zip `
    --output artifacts\analysis\stage43

python scripts/verify_stage43.py artifacts\analysis\stage43
python -m unittest -v tests.test_stage43
```

This reads frozen out-of-fold predictions. It does not train a recognizer or
regenerate predictions.

## Rebuild Stages 40 and 41

```powershell
New-Item -ItemType Directory -Force imports | Out-Null
Copy-Item path\to\stage39b_v2_results.zip imports\stage39b_v2_results.zip

$env:PYTHONPATH = "."
python scripts/build_stage40_end_to_end_evaluation.py
python scripts/verify_stage40_end_to_end_evaluation.py
python scripts/build_stage41_final_thesis_evidence.py
python scripts/verify_stage41_final_thesis_evidence.py
```

Stage 40 hashes text after canonical newline normalization. Therefore its
contracts verify whether Git checks out text with LF or CRLF line endings.

## Full training reproduction

The training runners are in `src/training/` and the frozen experiment settings
are in `configs/experiments/`. Re-running training requires the raw data,
preprocessing cache generation, appropriate PyTorch hardware, and considerably
more compute than evidence verification. Checkpoints are deliberately excluded
because the Stage 32 frozen prediction archive is sufficient for the reported
recognition metrics and Stage 43 statistical analysis.

## Integrity rules

- Do not mix files from an earlier experimental implementation with this
  fresh-start evidence track.
- Do not alter raw participant files in place.
- Use the supplied verifiers after any transfer or line-ending conversion.
- Treat exact sentence accuracy as a secondary metric; the primary recognition
  metric is pooled gloss WER.
