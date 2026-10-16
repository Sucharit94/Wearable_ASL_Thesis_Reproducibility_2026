import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version

PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "PyYAML",
    "tqdm",
    "matplotlib",
    "jiwer",
    "pytest",
]

def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None

report = {
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "python_version": sys.version,
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "processor": platform.processor(),
    "packages": {
        name: package_version(name)
        for name in PACKAGES
    },
}

try:
    import torch

    report["torch"] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ],
    }

except Exception as exc:
    report["torch"] = {
        "installed": False,
        "reason": str(exc),
    }

out = Path("docs") / "environment_local.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print("=" * 70)
print("STAGE 2A — LOCAL ENVIRONMENT REPORT")
print("=" * 70)
print("Python:", sys.version.split()[0])
print("Executable:", sys.executable)
print("Platform:", platform.platform())

if report["torch"].get("installed"):
    print("PyTorch:", report["torch"]["version"])
    print("CUDA available:", report["torch"]["cuda_available"])
else:
    print("PyTorch: not installed locally — acceptable for Stage 2A")

print("Saved:", out)
print("=" * 70)
