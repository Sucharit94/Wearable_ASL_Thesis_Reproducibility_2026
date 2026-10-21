# Independent verification quick start

This guide allows an examiner to verify the published evidence without access to participant recordings. Evidence verification and analysis reconstruction do not require model checkpoints. An optional Fold-1 checkpoint package is provided separately for checkpoint-integrity, strict-loading, and synthetic CPU-inference checks.

## 1. Download the repository

Either clone the current `main` branch:

```powershell
git clone https://github.com/Sucharit94/Wearable_ASL_Thesis_Reproducibility_2026.git
cd Wearable_ASL_Thesis_Reproducibility_2026
```

Or select **Code > Download ZIP** on GitHub and extract it.

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

PyTorch is required for the complete model test suite, pretrained-checkpoint verification, and training. Install a PyTorch build suitable for the local CPU or CUDA environment. The main evidence-verification commands below do not start model training.

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

## 4. Download and check the external assets

Open the [GitHub Release v1.0.1](https://github.com/Sucharit94/Wearable_ASL_Thesis_Reproducibility_2026/releases/tag/v1.0.1) and download:

- `stage32c_results_without_checkpoints.zip`
- `stage39b_v2_results.zip`
- `fold01_pretrained_checkpoints.zip`
- `release_asset_manifest.csv`

Expected SHA-256 values:

```text
stage32c_results_without_checkpoints.zip
262d6a91f03db1f07ee3a8ad2509117dcdfda5773b53f99a06ef2a7e2c12eac7

stage39b_v2_results.zip
c79a495439b7c84a22e50770c54fc49880317c007a7dd1cb9fca3d06c7b7e15c

fold01_pretrained_checkpoints.zip
c26ab51821caa2d54405da65ad4b11d231e7bdd66a40d3f78fadd4caedd90f7d
```

Windows PowerShell:

```powershell
Get-FileHash .\stage32c_results_without_checkpoints.zip -Algorithm SHA256
Get-FileHash .\stage39b_v2_results.zip -Algorithm SHA256
Get-FileHash .\fold01_pretrained_checkpoints.zip -Algorithm SHA256
```

Linux or macOS:

```bash
sha256sum stage32c_results_without_checkpoints.zip
sha256sum stage39b_v2_results.zip
sha256sum fold01_pretrained_checkpoints.zip
```

The calculated values must match the expected SHA-256 values above and the values recorded in `release_asset_manifest.csv`.

## 5. Optional: Verify the Fold-1 pretrained checkpoints

Use the current `main` branch because `scripts/verify_pretrained_models.py` is provided there.

Extract the checkpoint package into the repository root.

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

The command verifies the SHA-256 hash, model identity, strict weight loading, and one synthetic CPU-inference pass for each of the five Fold-1 models.

It does not process participant recordings, retrain the models, or reproduce recognition accuracy.

## 6. Rebuild the five-model statistics

```powershell
python scripts/build_stage43.py `
    .\stage32c_results_without_checkpoints.zip `
    --output artifacts\analysis\stage43_rebuilt

python scripts/verify_stage43.py artifacts\analysis\stage43_rebuilt
python -m unittest -v tests.test_stage43
```

## 7. Rebuild the final language evaluation

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

The verification and reconstruction procedures reproduce the reported tables, metrics, and statistical conclusions from frozen out-of-fold predictions and generation responses. They do not retrain the neural networks.

The optional checkpoint procedure verifies checkpoint integrity, model identity, strict loading, and synthetic CPU execution. It does not independently reproduce the reported recognition accuracy.

Full retraining additionally requires authorized dataset access, compatible GPU hardware, cache generation, and adaptation of the original Git provenance gates.
