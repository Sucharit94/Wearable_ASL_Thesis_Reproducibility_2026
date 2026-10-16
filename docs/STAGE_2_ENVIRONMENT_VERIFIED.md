# Stage 2 Environment Verification

## Local environment

- Python: 3.11.9
- Virtual environment: .venv
- Local PyTorch: intentionally not installed at Stage 2
- Local package import test: PASS
- Environment record: docs/environment_local.json

## Google Colab GPU environment

Verified on Google Colab:

- Python: 3.13.15
- PyTorch: 2.11.0+cu128
- CUDA runtime reported by PyTorch: 12.8
- CUDA available: true
- GPU: Tesla T4
- Reported GPU memory: 14.56 GB
- CUDA tensor computation test: PASS

Persistent Colab artifacts:

/content/drive/MyDrive/gu_asl_fresh_start_2026/stage2_environment/environment_colab_t4.json

/content/drive/MyDrive/gu_asl_fresh_start_2026/stage2_environment/pip_freeze_colab.txt

## Environment policy

Local development uses Python 3.11.

GPU training uses the recorded Google Colab environment.

Project source code should remain compatible with Python 3.11+.

PyTorch/CUDA supplied by Colab must not be blindly replaced during a
reported run. The actual environment must be recorded for every final
experiment session.
