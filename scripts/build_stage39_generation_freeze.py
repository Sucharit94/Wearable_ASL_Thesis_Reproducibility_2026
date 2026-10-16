"""Build the Stage 39A constrained-generation queue freeze."""

from pathlib import Path

from src.evaluation.stage37_language_protocol import (
    portable_text_sha256,
    read_csv,
    read_json,
    write_csv,
    write_json,
)
from src.evaluation.stage39_generation import (
    STAGE39_ROOT,
    build_generation_queue,
    generator_execution_template,
    queue_statistics,
    render_report,
    validate_reference_firewall,
    write_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def verify_stage38(root: Path) -> str:
    contract_path = root / "artifacts/language/stage38/stage38_contract.json"
    hash_path = root / "artifacts/language/stage38/stage38_contract.sha256"
    if not contract_path.is_file() or not hash_path.is_file():
        raise FileNotFoundError("Stage 38 must pass before Stage 39A.")
    observed = portable_text_sha256(contract_path)
    if hash_path.read_text(encoding="utf-8-sig").strip() != observed:
        raise RuntimeError("Stage 38 contract hash mismatch.")
    contract = read_json(contract_path)
    if contract.get("stage") != 38 or contract.get("status") != "passed":
        raise RuntimeError("Stage 38 is not frozen as passed.")
    if contract.get("training_started") is not False:
        raise RuntimeError("Stage 38 unexpectedly claims training.")
    if contract["protocol"]["primary_lightweight"] != "c2":
        raise RuntimeError("Stage 38 primary model changed.")
    for relative, expected in contract.get("generated_portable_sha256", {}).items():
        path = root / relative
        if not path.is_file() or portable_text_sha256(path) != expected:
            raise RuntimeError(f"Stage 38 output hash mismatch: {relative}")
    return observed


def build_stage39a(project_root: str | Path) -> dict[str, object]:
    root = Path(project_root).resolve()
    stage38_hash = verify_stage38(root)
    stage38_outputs_path = (
        root / "artifacts/language/stage38/stage38_l0_l1_outputs.csv"
    )
    target_path = root / "artifacts/labels/stage10_sentence_targets.csv"
    assignment_path = root / "artifacts/splits/stage32_known_inner_assignment.csv"
    stage38_rows = read_csv(stage38_outputs_path)
    targets = read_csv(target_path)
    assignments = read_csv(assignment_path)
    queue, mappings = build_generation_queue(stage38_rows, targets, assignments)
    validate_reference_firewall(queue)
    statistics = queue_statistics(queue, mappings)

    output_root = root / STAGE39_ROOT
    paths = {
        "queue": output_root / "stage39a_generation_queue.jsonl",
        "mapping": output_root / "stage39a_sample_request_map.csv",
        "statistics": output_root / "stage39a_queue_statistics.csv",
        "execution_template": output_root / "stage39a_generator_execution_template.json",
        "report": root / "docs/STAGE_39A_GENERATION_INPUT_FREEZE.md",
    }
    write_jsonl(paths["queue"], queue)
    write_csv(paths["mapping"], mappings)
    write_csv(paths["statistics"], statistics)
    write_json(paths["execution_template"], generator_execution_template())
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text(
        render_report(statistics, len(queue)), encoding="utf-8", newline="\n"
    )

    implementation = (
        root / "src/evaluation/stage39_generation.py",
        root / "scripts/build_stage39_generation_freeze.py",
        root / "scripts/verify_stage39_generation_freeze.py",
        root / "tests/test_stage39_generation_freeze.py",
    )
    contract = {
        "project_generation": "fresh_start_2026_v1",
        "stage": 39,
        "substage": "A",
        "name": "constrained_generation_input_and_execution_freeze",
        "status": "passed",
        "training_started": False,
        "generator_called": False,
        "outer_references_exposed": False,
        "input_stage38_contract_portable_sha256": stage38_hash,
        "protocol": {
            "systems": ["L2_constrained_generator"],
            "models": ["g1", "c1", "c2", "c3", "c4"],
            "folds": 5,
            "top_k_training_examples": 3,
            "temperature": 0.0,
            "resume_key": "request_id",
            "completed_responses_immutable": True,
            "backend_frozen": False,
            "stage39b_authorized": False,
        },
        "validation": {
            "stage38_l1_rows": 37000,
            "sample_request_rows": len(mappings),
            "unique_generation_requests": len(queue),
            "queue_statistics_rows": len(statistics),
            "reference_firewall_passed": True,
        },
        "implementation_portable_sha256": {
            path.relative_to(root).as_posix(): portable_text_sha256(path)
            for path in implementation
        },
        "generated_portable_sha256": {
            path.relative_to(root).as_posix(): portable_text_sha256(path)
            for path in paths.values()
        },
    }
    contract_path = output_root / "stage39a_contract.json"
    hash_path = output_root / "stage39a_contract.sha256"
    write_json(contract_path, contract)
    hash_path.write_text(
        portable_text_sha256(contract_path) + "\n", encoding="utf-8", newline="\n"
    )
    return {
        "contract": contract,
        "contract_path": contract_path,
        "hash_path": hash_path,
        "paths": paths,
        "statistics": statistics,
    }


def main() -> None:
    result = build_stage39a(PROJECT_ROOT)
    contract = result["contract"]
    print("=" * 78)
    print("STAGE 39A — CONSTRAINED-GENERATION INPUT FREEZE")
    print("=" * 78)
    print()
    print("QUEUE")
    print("-" * 78)
    print("Mapped outer samples across all five models: 37000")
    print(
        "Unique generation requests: "
        f"{contract['validation']['unique_generation_requests']}"
    )
    print("Training-only examples per request: 3")
    print("Temperature: 0.0")
    print()
    print("REFERENCE FIREWALL")
    print("-" * 78)
    print("Outer English references in queue: False")
    print("Sample IDs in prompts: False")
    print("Training-only candidates: True")
    print("API key stored: False")
    print()
    print("EXECUTION STATE")
    print("-" * 78)
    print("Generator called: False")
    print("Backend frozen: False")
    print("Stage 39B authorized: False")
    print()
    print("OUTPUTS")
    print("-" * 78)
    for path in result["paths"].values():
        print(path)
    print(result["contract_path"])
    print(result["hash_path"])
    print()
    print("FINAL STATUS: STAGE 39A PASSED")
    print("SELECT ONE GENERATOR BACKEND BEFORE STAGE 39B")
    print("=" * 78)


if __name__ == "__main__":
    main()
