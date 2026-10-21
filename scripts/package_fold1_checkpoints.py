#!/usr/bin/env python3
"""Create a release-asset ZIP from the verified Fold-1 files only.

The source is read-only. This excludes .git, partial .best.pt.* files, caches,
other folds, and last.pt. The output is created exclusively and cannot overwrite
an existing file. No files are uploaded by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import zipfile

MODELS = {"g1", "c1", "c2", "c3", "c4"}
METADATA = (
    "identity.json", "config.yaml", "code_commit.txt", "completion.json",
    "data_manifest_sha256.txt", "environment.json", "events.jsonl",
    "label_vocab_sha256.txt", "last_checkpoint.json", "metrics.json",
    "outer_evaluation_once.json", "outer_result.json", "predictions.jsonl",
    "split_sha256.txt", "train_log.csv",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    package = args.project.resolve() / "pretrained_checkpoints"
    manifest = package / "checkpoint_manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 5 or {row["Model"].lower() for row in rows} != MODELS:
        p.error("Manifest must describe exactly the five required models.")
    files = [manifest]
    if (package / "README.md").is_file():
        files.append(package / "README.md")
    for row in rows:
        model = row["Model"].lower()
        directory = package / "checkpoints" / "fold01" / model
        checkpoint = directory / "best.pt"
        if row["Fold"] != "fold01" or Path(row["File"]).as_posix() != f"checkpoints/fold01/{model}/best.pt":
            p.error(f"Unexpected checkpoint path for {model}.")
        if checkpoint.stat().st_size != int(row["SizeBytes"]) or sha256(checkpoint) != row["SHA256"].lower():
            p.error(f"Checkpoint size/hash mismatch: {model}.")
        if not (directory / "identity.json").is_file():
            p.error(f"Missing identity.json for {model}.")
        files.append(checkpoint)
        files.extend(directory / name for name in METADATA if (directory / name).is_file())
    # Fail before writing if the destination already exists.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for file in files:
                archive.write(file, Path("pretrained_checkpoints") / file.relative_to(package))
    print(f"Created {args.output.resolve()}")
    print(f"Files: {len(files)}; size: {args.output.stat().st_size / 2**20:.2f} MiB")
    print(f"SHA256: {sha256(args.output)}")
    print("Only existing best.pt hashes and identities were checked; no raw-sensor accuracy test was performed.")
    print("Normalization values are not supplied by this package. Do not describe it as a complete raw-input demo.")


if __name__ == "__main__":
    main()
