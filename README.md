# Wearable ASL Recognition and Gloss-to-English Evaluation

This repository contains the code, frozen configurations, tests, and compact evidence for a thesis study of sentence-level American Sign Language (ASL) recognition from wearable motion-capture-derived joint rotations. The evaluated pipeline is:

`38-feature sequence -> CTC gloss recognition -> English rendering`

The study uses 7,400 recordings from three subjects and 300 sentence IDs from the author-provided Gu, Oku and Todoh dataset. The raw dataset and large execution caches are not included in the Git repository. Five Fold-1 pretrained checkpoints are distributed separately through the GitHub release for checkpoint-integrity, strict-loading, and synthetic CPU-inference verification.

## Evaluated recognizers

| ID | Architecture | Parameters | Five-fold pooled WER | Exact sentence accuracy |
|---|---|---:|---:|---:|
| G1 | Near-Gu Conv2D–2xBiLSTM–CTC | 4,791,688 | 0.001074 | 0.996486 |
| C1 | Compact Conv1D–BiLSTM–CTC | 392,488 | 0.067215 | 0.799595 |
| C2 | Compact Conv1D–BiLSTM–self-attention–CTC | 656,168 | 0.040050 | 0.864730 |
| C3 | Compact Conv1D–Transformer–CTC | 237,352 | 0.444667 | 0.136486 |
| C4 | Custom Tiny-Swin–BiLSTM–CTC | 348,604 | 0.040909 | 0.876622 |

G1 is retained as the recognition-quality reference. C2 is the recommended lightweight recognizer because the C2–C4 WER difference was not resolved after Holm correction and C2 offered the stronger overall quality, efficiency, stability, and downstream-language trade-off. See `docs/STAGE_43_ALL_MODEL_STATISTICAL_VALIDATION.md` for the five-model tests.

## Language systems

- L0: normalized recognized gloss.
- L1: training-only retrieval over the closed 300-sentence inventory.
- L2: constrained FLAN-T5 correction.

The recommended lightweight closed-template pipeline is C2 + L1 (English WER 0.004715, exact accuracy 0.987973, content F1 0.9958). The best constrained generation result is G1 + L2 (English WER 0.085451). These results do not support open-vocabulary or unseen-template claims.

## Start here

- Follow `QUICKSTART.md` for the shortest independent verification procedure.
- Read `REPRODUCIBILITY.md` for evidence rebuilding and full-retraining scope.
- Download the frozen external evidence and optional Fold-1 checkpoint package from the [latest GitHub Release](https://github.com/Sucharit94/Wearable_ASL_Thesis_Reproducibility_2026/releases/latest).
- Read `DATA_ACCESS.md` before obtaining or placing the raw dataset.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Data, models, training, decoding, and evaluation modules |
| `scripts/` | Audits, builders, runners, and integrity verifiers |
| `tests/` | Unit and protocol tests |
| `configs/` | Frozen data, model, preprocessing, and experiment settings |
| `artifacts/` | Small manifests, tables, contracts, and figures |
| `docs/` | Stage reports and reproducibility guidance |
| `notebooks/` | Stage 39B V2 Colab generation notebook |

## Installation

Python 3.11 is the local target. Create an isolated environment and install the recorded analysis dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-local.txt
```

Install a PyTorch build appropriate for the local CPU/CUDA environment before running the complete test suite or training. Stage 39B generation has its own pinned file, `requirements-stage39b.txt`.

## Verify the published evidence

From the repository root:

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

These commands verify the frozen published evidence without retraining a model. The full test suite can be run with `python -m pytest` after PyTorch is installed.

## Verify the optional pretrained checkpoints

Download `fold01_pretrained_checkpoints.zip` from the [latest GitHub Release](https://github.com/Sucharit94/Wearable_ASL_Thesis_Reproducibility_2026/releases/latest) and extract it into the project root. The extracted package should create the `pretrained_checkpoints/` directory.

After installing PyTorch and NumPy, run:

```powershell
python .\scripts\verify_pretrained_models.py --project .
```

The verifier checks the SHA-256 hash, model identity, strict weight loading, and one synthetic CPU-inference pass for each Fold-1 model. It does not process the research dataset, reproduce the reported recognition accuracy, or retrain any model.

Dataset-level inference with these pretrained checkpoints would additionally require the matching fitted normalization and preprocessing state and a tested raw-data inference procedure. These materials are not currently distributed; therefore, the repository does not claim reproducible raw-recording inference from the supplied checkpoints.

## Scope, access and reuse

- The frozen Stage 32 predictions and Stage 39B V2 responses are distributed as GitHub Release assets instead of normal Git files.
- The optional Fold-1 pretrained checkpoints are distributed as a separate GitHub Release asset.
- The raw participant recordings are not included in the Git repository. See `DATA_ACCESS.md`.
- `docs/environment/pip_freeze_colab.txt` is an archival snapshot of every package present in the Colab runtime; inclusion in that list does not mean a package was imported or used by this project.
- The custom `LICENSE` permits downloading, inspection, local execution, and local adaptation for non-commercial academic examination and reproducibility verification. It is not an open-source licence.
- Citation metadata is provided in `CITATION.cff`.

The Git repository does not contain raw participant recordings, checkpoint binaries, credentials, virtual environments, or the author's local filesystem paths. The optional Fold-1 checkpoint package is distributed separately as a GitHub Release asset.
