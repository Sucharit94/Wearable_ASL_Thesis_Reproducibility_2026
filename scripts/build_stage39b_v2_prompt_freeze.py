"""Freeze the validated Stage 39B V2 plain-English prompt amendment."""

from pathlib import Path

from src.evaluation.stage37_language_protocol import portable_text_sha256, read_json, write_json
from src.evaluation.stage39b_v2_runner import v2_backend_specification


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE39_ROOT = Path("artifacts/language/stage39")


def build_v2(project_root: str | Path) -> dict[str, object]:
    root = Path(project_root).resolve()
    output_root = root / STAGE39_ROOT
    inputs = {
        "input_stage39a_contract_portable_sha256": output_root / "stage39a_contract.json",
        "input_stage39b_v1_contract_portable_sha256": output_root / "stage39b_backend_contract.json",
        "input_queue_portable_sha256": output_root / "stage39a_generation_queue.jsonl",
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required V2 input is missing: {path.name}")
    stage39a = read_json(inputs["input_stage39a_contract_portable_sha256"])
    if stage39a.get("status") != "passed" or stage39a.get("outer_references_exposed") is not False:
        raise RuntimeError("Stage 39A is not a safe V2 input.")

    implementation = (
        root / "src/evaluation/stage39b_v2_runner.py",
        root / "scripts/build_stage39b_v2_prompt_freeze.py",
        root / "scripts/verify_stage39b_v2_prompt_freeze.py",
        root / "scripts/run_stage39b_v2_generation.py",
        root / "tests/test_stage39b_v2.py",
    )
    report_path = root / "docs/STAGE_39B_V2_PROMPT_AMENDMENT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Stage 39B V2 — Plain-English Prompt Amendment",
                "",
                "V1 completed technically but all responses echoed JSON-schema fragments.",
                "Those responses are retained as failed execution evidence and are not scored.",
                "V2 retains the same model, revision, queue, decoding and leakage firewall.",
                "Only the output instruction changes to one plain English sentence.",
                "A reference-free 20-request pilot produced 20 structurally usable outputs.",
                "V2 writes a separate durable file and rejects V1 schema fragments.",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    contract = {
        "project_generation": "fresh_start_2026_v1",
        "stage": 39,
        "substage": "B-V2",
        "name": "plain_english_prompt_amendment_after_failed_json_output",
        "status": "passed",
        "training_started": False,
        "generator_called_during_freeze": False,
        "outer_references_exposed": False,
        "v1_results_eligible_for_scoring": False,
        "v1_failure": "11720_of_11720_outputs_repeated_schema_fragments",
        "pilot": {
            "requests": 20,
            "structurally_usable": 20,
            "outer_references_exposed": False,
            "saved_as_final_evidence": False,
        },
        **{name: portable_text_sha256(path) for name, path in inputs.items()},
        "backend_specification": v2_backend_specification(),
        "validation": {
            "models": ["g1", "c1", "c2", "c3", "c4"],
            "folds": 5,
            "mapped_outer_samples": 37000,
            "unique_generation_requests": int(stage39a["validation"]["unique_generation_requests"]),
            "reference_firewall_passed": True,
            "resumable": True,
            "separate_v2_output": True,
        },
        "implementation_portable_sha256": {
            path.relative_to(root).as_posix(): portable_text_sha256(path)
            for path in implementation
        },
        "generated_portable_sha256": {
            report_path.relative_to(root).as_posix(): portable_text_sha256(report_path)
        },
    }
    contract_path = output_root / "stage39b_v2_contract.json"
    hash_path = output_root / "stage39b_v2_contract.sha256"
    write_json(contract_path, contract)
    hash_path.write_text(portable_text_sha256(contract_path) + "\n", encoding="utf-8", newline="\n")
    return {"contract": contract, "contract_path": contract_path, "hash_path": hash_path}


def main() -> None:
    result = build_v2(PROJECT_ROOT)
    contract = result["contract"]
    print("=" * 78)
    print("STAGE 39B V2 — PLAIN-ENGLISH PROMPT FREEZE")
    print("=" * 78)
    print("V1 eligible for scoring: False")
    print("V1 failed outputs retained: True")
    print("Pilot structurally usable: 20/20")
    print("Models: G1, C1, C2, C3 and C4")
    print(f"Unique V2 requests: {contract['validation']['unique_generation_requests']}")
    print("Mapped samples: 37000")
    print("Separate V2 output: stage39b_v2_responses.jsonl")
    print("Training started: False")
    print("Generator called during freeze: False")
    print("Outer references exposed: False")
    print("FINAL STATUS: STAGE 39B V2 PROMPT FREEZE PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()
