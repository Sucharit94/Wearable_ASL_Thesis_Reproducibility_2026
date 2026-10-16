"""Verify Stage 39A without invoking a generator."""

from pathlib import Path

from scripts.build_stage39_generation_freeze import verify_stage38
from src.evaluation.stage37_language_protocol import portable_text_sha256, read_csv, read_json
from src.evaluation.stage39_generation import STAGE39_ROOT, read_jsonl, validate_reference_firewall


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    output_root = PROJECT_ROOT / STAGE39_ROOT
    contract_path = output_root / "stage39a_contract.json"
    hash_path = output_root / "stage39a_contract.sha256"
    require(contract_path.is_file() and hash_path.is_file(), "Stage 39A contract is missing.")
    require(hash_path.read_text(encoding="utf-8-sig").strip() == portable_text_sha256(contract_path), "Stage 39A contract hash mismatch.")
    contract = read_json(contract_path)
    require(contract.get("stage") == 39 and contract.get("substage") == "A", "Stage 39A identity failed.")
    require(contract.get("status") == "passed", "Stage 39A did not pass.")
    require(contract.get("training_started") is False, "Training was started.")
    require(contract.get("generator_called") is False, "A generator was called.")
    require(contract.get("outer_references_exposed") is False, "Outer references were exposed.")
    require(contract["protocol"]["backend_frozen"] is False, "Unexpected backend state.")
    require(contract["protocol"]["stage39b_authorized"] is False, "Stage 39B was prematurely authorized.")
    require(verify_stage38(PROJECT_ROOT) == contract["input_stage38_contract_portable_sha256"], "Stage 38 changed.")

    for relative, expected in contract["implementation_portable_sha256"].items():
        path = PROJECT_ROOT / relative
        require(path.is_file() and portable_text_sha256(path) == expected, f"Implementation hash mismatch: {relative}")
    for relative, expected in contract["generated_portable_sha256"].items():
        path = PROJECT_ROOT / relative
        require(path.is_file() and portable_text_sha256(path) == expected, f"Output hash mismatch: {relative}")

    queue = read_jsonl(output_root / "stage39a_generation_queue.jsonl")
    mapping = read_csv(output_root / "stage39a_sample_request_map.csv")
    statistics = read_csv(output_root / "stage39a_queue_statistics.csv")
    validate_reference_firewall(queue)
    require(len(mapping) == 37000, "Sample/request mapping must contain 37000 rows.")
    require(len(statistics) == 25, "Queue statistics must contain 25 rows.")
    require(len(queue) == contract["validation"]["unique_generation_requests"], "Queue count changed.")
    queue_ids = {row["request_id"] for row in queue}
    require(len(queue_ids) == len(queue), "Duplicate request IDs detected.")
    require({row["request_id"] for row in mapping} <= queue_ids, "Mapping refers to missing requests.")

    print("=" * 78)
    print("STAGE 39A — GENERATION INPUT FREEZE VERIFIER")
    print("=" * 78)
    print()
    print("INTEGRITY")
    print("-" * 78)
    print("Stage 38 unchanged: True")
    print("Stage 39A contract hash match: True")
    print("Implementation and output hashes match: True")
    print(f"Unique generation requests: {len(queue)}")
    print("Mapped outer samples: 37000")
    print("Models: G1, C1, C2, C3 and C4")
    print()
    print("SAFETY")
    print("-" * 78)
    print("Reference firewall passed: True")
    print("Generator called: False")
    print("Backend frozen: False")
    print("Stage 39B authorized: False")
    print("Training started: False")
    print()
    print("FINAL STATUS: STAGE 39A PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()
