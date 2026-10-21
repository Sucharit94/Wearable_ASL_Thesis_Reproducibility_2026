# Independent verification quick start

This guide allows an examiner to verify the published evidence without access
to participant recordings or model checkpoints.

## 1. Download the repository

Either clone it:

```powershell
git clone https://github.com/Sucharit94/Wearable_ASL_Thesis_Reproducibility_2026.git
cd Wearable_ASL_Thesis_Reproducibility_2026
```

or select **Code > Download ZIP** on GitHub and extract it.

## 2. Create the Python environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-local.txt
```

Linux or macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-local.txt
```

PyTorch is needed for the complete model test suite and training, but the main
evidence-verification commands below do not start training.

## 3. Verify the published evidence

Windows PowerShell:

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

Linux or macOS:

```bash
export PYTHONPATH=.
python scripts/verify_stage34_model_evaluation.py
python scripts/verify_stage34_statistical_comparison.py
python scripts/verify_stage35_evidence_freeze.py
python scripts/verify_stage36_error_analysis.py
python scripts/verify_stage40_end_to_end_evaluation.py
python scripts/verify_stage41_final_thesis_evidence.py
python scripts/verify_stage43.py artifacts/analysis/stage43
```

Every verifier should finish with `FINAL STATUS: ... PASSED`.

## 4. Download and check the external evidence

Open the
[latest GitHub Release](https://github.com/Sucharit94/Wearable_ASL_Thesis_Reproducibility_2026/releases/latest)
and download:

- `stage32c_results_without_checkpoints.zip`
- `stage39b_v2_results.zip`
- `release_asset_manifest.csv`

Expected SHA-256 values:

```text
stage32c_results_without_checkpoints.zip
262d6a91f03db1f07ee3a8ad2509117dcdfda5773b53f99a06ef2a7e2c12eac7

stage39b_v2_results.zip
c79a495439b7c84a22e50770c54fc49880317c007a7dd1cb9fca3d06c7b7e15c
```

Windows verification example:

```powershell
Get-FileHash .\stage32c_results_without_checkpoints.zip -Algorithm SHA256
Get-FileHash .\stage39b_v2_results.zip -Algorithm SHA256
```

Linux or macOS verification example:

```bash
sha256sum stage32c_results_without_checkpoints.zip
sha256sum stage39b_v2_results.zip
```

## 5. Rebuild the five-model statistics

```powershell
python scripts/build_stage43.py `
    .\stage32c_results_without_checkpoints.zip `
    --output artifacts\analysis\stage43_rebuilt

python scripts/verify_stage43.py artifacts\analysis\stage43_rebuilt
python -m unittest -v tests.test_stage43
```

## 6. Rebuild the final language evaluation

```powershell
New-Item -ItemType Directory -Force imports | Out-Null
Copy-Item .\stage39b_v2_results.zip imports\stage39b_v2_results.zip

$env:PYTHONPATH = "."
python scripts/build_stage40_end_to_end_evaluation.py
python scripts/verify_stage40_end_to_end_evaluation.py
python scripts/build_stage41_final_thesis_evidence.py
python scripts/verify_stage41_final_thesis_evidence.py
```

## Interpretation

### Rescore the saved recognition predictions

The supplied Stage 43 builder consumes saved evaluation records. For an
additional check that recalculates edit counts from reference/predicted token
IDs and checks the vocabulary mapping, run:

```powershell
python scripts/verify_saved_predictions.py --project . --results-zip .\stage32c_results_without_checkpoints.zip
```

Expected coverage is 37,000 predictions, 7,400 per recognizer. This still uses
saved predictions and does not run a network on raw sensor inputs.

### Optional pretrained checkpoint check

`pretrained_checkpoints/README.md` documents the manifest, distribution status
and extraction paths for the selected Fold-1 supplement. v1.0.1 itself does not
include checkpoint binaries. If the supplement is available locally and
PyTorch is installed, run:

```powershell
python scripts/verify_pretrained_models.py --project .
```

Expected result: five models pass their hash, identity, strict weight-loading
and CPU forward-pass checks. The input is synthetic zeros. A real-recording
demonstration also requires the matching full-refit normalization values,
authorized input data and preprocessing/decoding tooling.

The verification and reconstruction procedures above reproduce the reported
tables, metrics and statistical conclusions from frozen out-of-fold
predictions and generation responses. They do not retrain the neural networks.
Full retraining additionally requires authorized dataset access, compatible GPU
hardware, cache generation and adaptation of the original Git provenance gates.
