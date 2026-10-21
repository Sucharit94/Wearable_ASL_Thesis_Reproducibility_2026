#!/usr/bin/env python3
"""Verify the compact ASL pretrained checkpoint package without training.

The verifier performs four checks for each Fold-1 model:
1. file size and SHA-256 match checkpoint_manifest.csv;
2. the trusted project checkpoint envelope matches identity.json;
3. all checkpoint weights load strictly into the intended architecture;
4. one CPU inference pass returns a valid shared model output.

No dataset, GPU, network access, training, or output-file creation is required.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path


MODEL_ORDER = ("g1", "c1", "c2", "c3", "c4")
SYNTHETIC_SEQUENCE_LENGTH = 72
EXPECTED_FEATURE_WIDTH = 38


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(package_root: Path) -> dict[str, dict[str, str]]:
    manifest_path = package_root / "checkpoint_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Checkpoint manifest not found: {manifest_path}")

    rows: dict[str, dict[str, str]] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            model_id = row["Model"].strip().lower()
            if model_id in rows:
                raise ValueError(f"Duplicate manifest entry for {model_id}.")
            rows[model_id] = row

    if set(rows) != set(MODEL_ORDER):
        raise ValueError(
            "Manifest models differ from the required set: "
            + ", ".join(MODEL_ORDER)
        )
    return rows


def verify_one_model(
    *,
    project_root: Path,
    package_root: Path,
    model_id: str,
    manifest_row: dict[str, str],
    torch,
    build_model,
    validate_model_parameter_count,
    validate_model_output,
    ExperimentIdentity,
    load_torch_checkpoint,
) -> dict[str, object]:
    model_directory = package_root / "checkpoints" / "fold01" / model_id
    checkpoint_path = package_root / Path(manifest_row["File"])
    identity_path = model_directory / "identity.json"

    if checkpoint_path != model_directory / "best.pt":
        raise ValueError(
            f"{model_id.upper()} manifest points outside its expected best.pt path."
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    if not identity_path.is_file():
        raise FileNotFoundError(f"Missing identity file: {identity_path}")

    expected_size = int(manifest_row["SizeBytes"])
    observed_size = checkpoint_path.stat().st_size
    if observed_size != expected_size:
        raise ValueError(
            f"Checkpoint size mismatch: expected {expected_size}, "
            f"observed {observed_size}."
        )

    expected_hash = manifest_row["SHA256"].strip().lower()
    observed_hash = sha256_file(checkpoint_path)
    if observed_hash != expected_hash:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch: expected {expected_hash}, "
            f"observed {observed_hash}."
        )

    identity_payload = json.loads(identity_path.read_text(encoding="utf-8-sig"))
    identity = ExperimentIdentity.from_dict(identity_payload)
    if identity.model_id.lower() != model_id:
        raise ValueError(
            f"Identity model is {identity.model_id!r}, expected {model_id!r}."
        )

    started = time.perf_counter()
    envelope = load_torch_checkpoint(
        checkpoint_path,
        identity,
        map_location="cpu",
    )

    model = build_model(model_id).cpu()
    parameter_count = validate_model_parameter_count(model_id, model)
    model.load_state_dict(envelope["model_state"], strict=True)
    model.eval()

    torch.manual_seed(20261021)
    synthetic_input = torch.zeros(
        (1, SYNTHETIC_SEQUENCE_LENGTH, EXPECTED_FEATURE_WIDTH),
        dtype=torch.float32,
    )
    raw_lengths = torch.tensor(
        [SYNTHETIC_SEQUENCE_LENGTH],
        dtype=torch.int64,
    )

    with torch.inference_mode():
        output = model(synthetic_input, raw_lengths)

    valid_lengths = validate_model_output(output)
    elapsed_seconds = time.perf_counter() - started
    output_shape = tuple(int(value) for value in output.logits_btc.shape)

    del output
    del synthetic_input
    del raw_lengths
    del model
    del envelope
    gc.collect()

    return {
        "model": model_id.upper(),
        "parameters": int(parameter_count),
        "checkpoint_mib": observed_size / (1024 * 1024),
        "output_shape": output_shape,
        "valid_lengths": tuple(int(value) for value in valid_lengths),
        "elapsed_seconds": elapsed_seconds,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify all five Fold-1 ASL pretrained checkpoints on CPU "
            "without retraining or reading the dataset."
        )
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path.cwd(),
        help="Project root (default: current working directory).",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    project_root = arguments.project.expanduser().resolve()
    package_root = project_root / "pretrained_checkpoints"

    if not (project_root / "src").is_dir():
        print(f"ERROR: project src folder not found: {project_root / 'src'}")
        return 2
    if not package_root.is_dir():
        print(f"ERROR: pretrained checkpoint package not found: {package_root}")
        return 2

    sys.path.insert(0, str(project_root))

    try:
        import numpy
        import torch

        from src.models.contracts import validate_model_output
        from src.models.factory import (
            build_model,
            validate_model_parameter_count,
        )
        from src.training.checkpoint import load_torch_checkpoint
        from src.training.registry import ExperimentIdentity
    except Exception as exc:
        print(f"ERROR: required project environment could not be imported: {exc}")
        return 2

    thread_count = max(1, min(4, os.cpu_count() or 1))
    torch.set_num_threads(thread_count)

    print("ASL PRETRAINED MODEL VERIFICATION")
    print(f"Project: {project_root}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"NumPy: {numpy.__version__}")
    print(f"Device: CPU ({thread_count} threads)")
    print("Dataset required: no")
    print("Training performed: no")
    print("")

    try:
        manifest = load_manifest(package_root)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    passed = 0
    failures: list[tuple[str, str]] = []

    for model_id in MODEL_ORDER:
        print(f"Checking {model_id.upper()}...", flush=True)
        try:
            result = verify_one_model(
                project_root=project_root,
                package_root=package_root,
                model_id=model_id,
                manifest_row=manifest[model_id],
                torch=torch,
                build_model=build_model,
                validate_model_parameter_count=validate_model_parameter_count,
                validate_model_output=validate_model_output,
                ExperimentIdentity=ExperimentIdentity,
                load_torch_checkpoint=load_torch_checkpoint,
            )
        except Exception as exc:
            failures.append((model_id.upper(), str(exc)))
            print(f"  FAIL: {exc}")
            continue

        passed += 1
        print(
            "  PASS: hash + identity + strict weights + CPU inference | "
            f"params={result['parameters']:,} | "
            f"checkpoint={result['checkpoint_mib']:.2f} MiB | "
            f"output={result['output_shape']} | "
            f"valid_lengths={result['valid_lengths']} | "
            f"time={result['elapsed_seconds']:.2f}s"
        )

    print("")
    if failures:
        print(f"VERIFICATION FAILED: {passed}/5 models passed.")
        for model_id, message in failures:
            print(f"- {model_id}: {message}")
        return 1

    print("VERIFICATION PASSED: 5/5 models passed all checks.")
    print("The packaged checkpoints can be loaded and used for inference on CPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

