#!/usr/bin/env python3
"""Rescore every saved v1.0.1 recognition prediction from its token IDs.

This checks edit counts, exact correctness and ID-to-token mapping. It does not
read raw recordings, run recognition networks or regenerate predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

EXPECTED_ARCHIVE_SHA256 = "262d6a91f03db1f07ee3a8ad2509117dcdfda5773b53f99a06ef2a7e2c12eac7"
MODELS = {"g1", "c1", "c2", "c3", "c4"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=Path.cwd())
    p.add_argument("--results-zip", type=Path, required=True)
    args = p.parse_args()
    root = args.project.resolve()
    if hashlib.sha256(args.results_zip.read_bytes()).hexdigest() != EXPECTED_ARCHIVE_SHA256:
        p.error("Archive hash differs from the inspected v1.0.1 prediction asset.")
    sys.path.insert(0, str(root))
    from src.evaluation.metrics import score_sequence
    vocabulary = json.loads((root / "artifacts/labels/stage10_id_to_token.json").read_text(encoding="utf-8-sig"))
    totals = {}
    seen = {model: set() for model in MODELS}
    checked = 0
    with zipfile.ZipFile(args.results_zip) as archive:
        names = [name for name in archive.namelist() if name.endswith("/predictions.jsonl")]
        if len(names) != 25:
            raise ValueError(f"Expected 25 prediction files; found {len(names)}.")
        for name in names:
            model = name.split("/")[-2]
            if model not in MODELS:
                raise ValueError(f"Unexpected model: {model}")
            total = totals.setdefault(model, {"samples": 0, "edits": 0, "tokens": 0, "correct": 0})
            with archive.open(name) as handle:
                for line in handle:
                    row = json.loads(line)
                    sample_id = row["sample_id"]
                    if sample_id in seen[model]:
                        raise ValueError(f"Duplicate prediction: {model}/{sample_id}")
                    seen[model].add(sample_id)
                    score = score_sequence(row["reference_token_ids"], row["predicted_token_ids"])
                    for key in ("substitutions", "deletions", "insertions", "edit_distance", "exact_match"):
                        if score[key] != row[key]:
                            raise ValueError(f"Rescoring mismatch: {model}/{sample_id}/{key}")
                    for ids, words in (("reference_token_ids", "reference_tokens"), ("predicted_token_ids", "predicted_tokens")):
                        if [vocabulary[str(token)] for token in row[ids]] != row[words]:
                            raise ValueError(f"Token mapping mismatch: {model}/{sample_id}/{words}")
                    total["samples"] += 1
                    total["edits"] += score["edit_distance"]
                    total["tokens"] += score["reference_length"]
                    total["correct"] += int(score["exact_match"])
                    checked += 1
    if set(totals) != MODELS or any(len(ids) != 7400 for ids in seen.values()):
        raise ValueError("Incomplete model/sample coverage.")
    if any(ids != seen["g1"] for ids in seen.values()):
        raise ValueError("Model sample sets are not paired.")
    print("RESCORING SAVED PREDICTIONS: no new model inference")
    for model in ("g1", "c1", "c2", "c3", "c4"):
        t = totals[model]
        print(f"{model.upper()}: samples={t['samples']}; pooled WER={t['edits']/t['tokens']:.6f}; exact accuracy={t['correct']/t['samples']:.6f}")
    print(f"PASS: {checked:,} predictions rescored; all edit counts, exact matches and token mappings agree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
