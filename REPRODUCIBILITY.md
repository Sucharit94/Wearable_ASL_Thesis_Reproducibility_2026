# Reproducibility guide

## Reproducibility levels

| Level | What can be reproduced | Additional requirements |
|---|---|---|
| Evidence verification | Integrity of the published contracts, tables, metrics, and statistical outputs | Repository only |
| Analysis reconstruction | Five-model statistics and final language evaluation from frozen predictions and responses | Two frozen-output GitHub Release assets |
| Pretrained checkpoint verification | Checkpoint hashes, model identities, strict weight loading, and synthetic CPU inference for five Fold-1 models | Current `main` branch, checkpoint Release asset, PyTorch, and NumPy |
| Full retraining | Preprocessing, model selection, refitting, and outer evaluation | Authorized raw dataset, compatible GPU environment, substantial compute, and adaptation of the original Git provenance gates |

The first two levels are the intended public evidence-review route and do not require participant recordings or model checkpoints. Pretrained checkpoint verification is an additional optional check and does not reproduce recognition accuracy.

## External evidence assets

Download these files from [GitHub Release v1.0.1](https://github.com/Sucharit94/Wearable_ASL_Thesis_Reproducibility_2026/releases/tag/v1.0.1):

- `stage32c_results_without_checkpoints.zip`
- `stage39b_v2_results.zip`
- `fold01_pretrained_checkpoints.zip`
- `release_asset_manifest.csv`

Verify their sizes and SHA-256 values against both the GitHub Release digest and `docs/release_asset_manifest.csv` before use.

## Environment

Use Python 3.11 for local analysis. Install `requirements-local.txt`, and then install a PyTorch build suitable for the machine when running model tests, checkpoint verification, or training.

The recorded Colab T4 environment is stored in `docs/environment/environment_colab_t4.json`. The complete package snapshot is stored in `docs/environment/pip_freeze_colab.txt`.

The complete snapshot records packages present in the runtime, including packages that were not imported or used by this project.

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

These commands verify the published evidence already stored in the repository. They do not retrain a recognizer.

## Verify the optional pretrained checkpoints

Download `fold01_pretrained_checkpoints.zip` from GitHub Release v1.0.1 and extract it into the root of the current `main` branch.

Windows PowerShell:

```powershell
Expand-Archive .\fold01_pretrained_checkpoints.zip -DestinationPath . -Force
python .\scripts\verify_pretrained_models.py --project .
```

Linux or macOS:

```bash
unzip fold01_pretrained_checkpoints.zip
python scripts/verify_pretrained_models.py --project .
```

The verifier checks the SHA-256 hash, experiment identity, strict model-state loading, and one synthetic CPU-inference pass for each Fold-1 model.

The checkpoint verifier does not read raw recordings, regenerate the published predictions, measure recognition accuracy, or retrain the models.

Configuration files document the experimental preprocessing settings. However, repeating inference on raw recordings would additionally require the checkpoint-matched fitted normalization state and a tested command that converts raw recordings into model inputs and decoded predictions. These items are not included in the current public package. Therefore, the repository does not claim reproducible raw-recording inference from the supplied checkpoints.

## Rebuild Stage 43 statistics

```powershell
python scripts/build_stage43.py `
    path\to\stage32c_results_without_checkpoints.zip `
    --output artifacts\analysis\stage43_rebuilt

python scripts/verify_stage43.py artifacts\analysis\stage43_rebuilt
python -m unittest -v tests.test_stage43
```

This procedure reads the frozen out-of-fold predictions. It does not train a recognizer or regenerate predictions.

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

Stage 40 hashes text after canonical newline normalization. Its contracts therefore verify after Git checks out text with LF or CRLF line endings.

## Requirements and limitations for full retraining

Full retraining requires the author-provided Gu, Oku and Todoh dataset under its original terms, generation of the Stage 17 binary cache, a compatible NVIDIA GPU environment, and considerably more compute than evidence verification.

The experimental training runners preserve strict checks against annotated Git tags from the original execution history. The public examination repository was repackaged into a new Git history, so those provenance gates must be adapted and revalidated before starting a new full training run from this public clone.

These limitations do not affect verification or reconstruction from the frozen published evidence.

## Integrity rules

- Do not alter raw participant files in place.
- Do not mix artifacts from an earlier implementation with this evidence track.
- Use the supplied verifiers after any transfer or line-ending conversion.
- Treat exact sentence accuracy as a secondary metric; pooled gloss WER is the primary recognition metric.
