from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.stage41_thesis_evidence import MODEL_ORDER, SYSTEM_ORDER, read_csv_rows, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "artifacts/analysis/stage41"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    print("=" * 78)
    print("STAGE 41 — FINAL THESIS EVIDENCE VERIFIER")
    print("=" * 78)
    contract_path = OUTPUT / "stage41_contract.json"
    hash_path = OUTPUT / "stage41_contract.sha256"
    require(contract_path.is_file() and hash_path.is_file(), "Stage 41 contract or hash is missing")
    contract = json.loads(contract_path.read_text(encoding="utf-8-sig"))
    recorded = hash_path.read_text(encoding="utf-8-sig").strip().split()[0]
    require(sha256_file(contract_path) == recorded, "Stage 41 contract hash mismatch")
    for relative, expected in contract["inputs"].items():
        path = PROJECT_ROOT / relative
        require(path.is_file(), f"Frozen input is missing: {relative}")
        require(sha256_file(path) == expected, f"Frozen input changed: {relative}")
    for relative, expected in contract["outputs"].items():
        path = PROJECT_ROOT / relative
        require(path.is_file(), f"Generated output is missing: {relative}")
        require(sha256_file(path) == expected, f"Generated output changed: {relative}")

    recognition = read_csv_rows(OUTPUT / "stage41_recognition_model_table.csv")
    language = read_csv_rows(OUTPUT / "stage41_language_system_table.csv")
    selections = read_csv_rows(OUTPUT / "stage41_final_selection.csv")
    claims = read_csv_rows(OUTPUT / "stage41_claim_register.csv")
    require(len(recognition) == 5 and {row["model_id"] for row in recognition} == set(MODEL_ORDER), "Recognition coverage failed")
    require(len(language) == 15, "Language coverage failed")
    require({(row["model_id"], row["system_id"]) for row in language} == {(m, s) for m in MODEL_ORDER for s in SYSTEM_ORDER}, "Model/system matrix failed")
    require(any(row["selected_model"] == "C2" and row["selected_language_system"] == "L1" for row in selections), "C2+L1 final selection is missing")
    require(any(row["status"] == "not_executed" and "L3" in row["thesis_safe_claim"] for row in claims), "L3 boundary is missing")
    require(all(row["prohibited_overclaim"].strip() for row in claims), "Every claim needs an overclaim boundary")
    require(contract["decisions"]["recognition_quality_reference"] == "g1", "G1 role changed")
    require(contract["decisions"]["recommended_lightweight_recognizer"] == "c2", "C2 role changed")
    boundaries = contract["claim_boundaries"]
    require(boundaries == {"accuracy_used_alone": False, "l3_executed": False, "open_vocabulary_claim_allowed": False, "unseen_template_claim_allowed": False}, "Claim boundaries changed")
    require(not any(contract["execution"].values()), "Stage 41 unexpectedly executed an experiment")

    print("\nINTEGRITY")
    print("-" * 78)
    print("Frozen input hashes match: True")
    print("Generated output hashes match: True")
    print("Contract hash match: True")
    print("\nCOVERAGE")
    print("-" * 78)
    print("Recognition models: 5")
    print("Language systems: 15 model-system combinations")
    print(f"Final decisions: {len(selections)}")
    print(f"Controlled claims: {len(claims)}")
    print("\nFINAL DECISION")
    print("-" * 78)
    print("Quality reference: G1")
    print("Lightweight recognizer: C2")
    print("Lightweight end-to-end system: C2 + L1")
    print("Accuracy used alone: False")
    print("L3 executed: False")
    print("\nFINAL STATUS: STAGE 41 PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()
