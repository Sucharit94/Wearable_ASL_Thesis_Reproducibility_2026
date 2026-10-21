# Reproducibility guide

## Reproducibility levels

| Level | What can be reproduced | Additional requirements |
|---|---|---|
| Evidence verification | Integrity of the published contracts, tables, metrics and statistical outputs | Repository only |
| Analysis reconstruction | Five-model statistics and final language evaluation from frozen predictions and responses | Two GitHub Release assets |
| Saved-prediction rescoring | Recomputed edit counts, exact correctness and token mapping for all recognition predictions | v1.0.1 Stage 32 prediction ZIP |
| Pretrained model execution check | Checkpoint identity, strict loading and forward execution on synthetic input | Locally available Fold-1 checkpoint supplement; see its distribution status |
| Full retraining | Preprocessing, model selection, refitting and outer evaluation | Authorized raw dataset, compatible GPU environment, substantial compute and adaptation of original Git provenance gates |

The first two levels are the intended public review route. They do not require
participant recordings or model checkpoints.

The selected Fold-1 checkpoint manifest and verification/packaging scripts are
additional examination tools. See `pretrained_checkpoints/README.md` for actual
distribution status: the v1.0.1 assets do not contain checkpoint binaries.
The prepared supplement supports synthetic-input execution, and its matching
normalization vectors have not been included. It is not a complete real-input
prediction package or a replacement for the five-fold frozen results.

## External evidence assets

Download these files from the
[latest GitHub Release](https://github.com/Sucharit94/Wearable_ASL_Thesis_Reproducibility_2026/releases/latest):

- `stage32c_results_without_checkpoints.zip`
- `stage39b_v2_results.zip`
- `release_asset_manifest.csv`

Verify their sizes and SHA-256 values against both the Release digest and
`docs/release_asset_manifest.csv` before use.

## Environment

Use Python 3.11 for local analysis. Install `requirements-local.txt`, then
install a PyTorch build suitable for the machine if running model tests or
training. The recorded Colab T4 environment is in
`docs/environment/environment_colab_t4.json`; the complete package snapshot is
in `docs/environment/pip_freeze_colab.txt`.

The complete snapshot records packages present in the runtime, including
packages that were not imported or used by this project.

## Verify the evidence already stored in Git

```powershell
$env:PYTHONPATH = "."
python scripts/verify_stage34_model_evaluation.py
python scripts/verify_stage34_statistical_comparison.py
python scripts/verify_stage35_evidence_freeze.py
python scripts/verify_stage36_error_analysis.py
python scripts/verify_stage40_end_to_end_evaluation.py
python scripts/verify_stage41_final_thesis_evidence.py
python scripts/verify_stage43.py artifacts/analysis/stage43
```

## Rebuild Stage 43 statistics

```powershell
python scripts/build_stage43.py `
    path\to\stage32c_results_without_checkpoints.zip `
    --output artifacts\analysis\stage43_rebuilt

python scripts/verify_stage43.py artifacts\analysis\stage43_rebuilt
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

Stage 40 hashes text after canonical newline normalization. Its contracts
therefore verify after Git checks out text with LF or CRLF line endings.

## Requirements and limitation for full retraining

Full retraining requires the author-provided Gu, Oku and Todoh dataset under its
original terms, generation of the Stage 17 binary cache, a compatible NVIDIA
GPU environment, and considerably more compute than evidence verification.

The experimental training runners preserve strict checks against annotated Git
tags from the original execution history. The public examination repository
was repackaged into a new Git history, so those provenance gates must be adapted
and revalidated before starting a new full training run from this public clone.
This limitation does not affect verification or reconstruction from the frozen
published evidence.

## Integrity rules

- Do not alter raw participant files in place.
- Do not mix artifacts from an earlier implementation with this evidence track.
- Use the supplied verifiers after any transfer or line-ending conversion.
- Treat exact sentence accuracy as a secondary metric; pooled gloss WER is the
  primary recognition metric.
