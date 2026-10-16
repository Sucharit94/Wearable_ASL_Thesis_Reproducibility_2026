from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


OUTPUT_FILES = {
    "summary": "stage43_model_summary.csv",
    "omnibus": "stage43_omnibus_tests.csv",
    "ttests": "stage43_pairwise_ttests.csv",
    "mcnemar": "stage43_pairwise_mcnemar.csv",
    "bootstrap": "stage43_pairwise_cluster_bootstrap.csv",
    "thesis": "STAGE_43_THESIS_STATISTICAL_TEXT.md",
}


TEXT_HASH_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".txt", ".yaml", ".yml"}


def sha256(path: Path) -> str:
    """Hash text canonically so Git line-ending conversion cannot break evidence."""
    digest = hashlib.sha256()
    if path.suffix.lower() in TEXT_HASH_SUFFIXES:
        data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest.update(data)
        return digest.hexdigest()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()

    contract_path = root / "stage43_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    recorded_contract_hash = (root / "stage43_contract.sha256").read_text(
        encoding="utf-8"
    ).split()[0]
    checks = {
        "contract hash": sha256(contract_path) == recorded_contract_hash,
        "five models": contract["models"] == ["G1", "C1", "C2", "C3", "C4"],
        "paired samples": contract["samples"] == 7400,
        "clusters": contract["subject_sentence_clusters"] == 368,
        "bootstrap replicates": contract["bootstrap_replicates"] == 10000,
        "training not started": contract["training_started"] is False,
        "predictions not regenerated": contract["predictions_regenerated"] is False,
    }
    for key, filename in OUTPUT_FILES.items():
        checks[f"hash {filename}"] = (
            sha256(root / filename) == contract["output_hashes"][key]
        )

    omnibus = read_csv(root / OUTPUT_FILES["omnibus"])
    ttests = read_csv(root / OUTPUT_FILES["ttests"])
    bootstrap = read_csv(root / OUTPUT_FILES["bootstrap"])
    checks["three omnibus tests"] = len(omnibus) == 3
    checks["ten paired t tests"] = len(ttests) == 10
    checks["ten paired bootstraps"] = len(bootstrap) == 10
    c2c4_t = next(row for row in ttests if row["model_a"] == "C2" and row["model_b"] == "C4")
    c2c4_b = next(
        row for row in bootstrap if row["model_a"] == "C2" and row["model_b"] == "C4"
    )
    checks["C2-C4 t unresolved"] = c2c4_t["significant_0_05"] == "False"
    checks["C2-C4 bootstrap unresolved"] = c2c4_b["significant_0_05"] == "False"

    failed = [name for name, passed in checks.items() if not passed]
    print("=" * 78)
    print("STAGE 43 - ALL FIVE MODEL STATISTICAL VALIDATION VERIFIER")
    print("=" * 78)
    print(f"Checks: {len(checks)}")
    print(f"Failed: {len(failed)}")
    if failed:
        for name in failed:
            print(f"FAILED: {name}")
        raise SystemExit(1)
    print("Global F/Friedman/Cochran tests: verified")
    print("All ten paired comparisons: verified")
    print("C2-C4 WER ordering: not statistically resolved")
    print("FINAL STATUS: STAGE 43 PASSED")


if __name__ == "__main__":
    main()
