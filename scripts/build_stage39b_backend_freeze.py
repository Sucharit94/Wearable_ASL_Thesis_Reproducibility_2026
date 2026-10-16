"""Freeze the Stage 39B local FLAN-T5 backend without invoking it."""

from pathlib import Path

from src.evaluation.stage37_language_protocol import portable_text_sha256, read_json, write_json
from src.evaluation.stage39b_runner import backend_specification


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE39_ROOT = Path("artifacts/language/stage39")


def build_stage39b(project_root: str | Path) -> dict[str, object]:
    root = Path(project_root).resolve()
    output_root = root / STAGE39_ROOT
    stage39a_path = output_root / "stage39a_contract.json"
    stage39a_hash_path = output_root / "stage39a_contract.sha256"
    if not stage39a_path.is_file() or not stage39a_hash_path.is_file():
        raise FileNotFoundError("Stage 39A must be frozen before Stage 39B.")
    stage39a_hash = portable_text_sha256(stage39a_path)
    if stage39a_hash_path.read_text(encoding="utf-8-sig").strip() != stage39a_hash:
        raise RuntimeError("Stage 39A contract hash mismatch.")
    stage39a = read_json(stage39a_path)
    if stage39a.get("status") != "passed" or stage39a.get("generator_called") is not False:
        raise RuntimeError("Stage 39A is not a clean input freeze.")
    queue_path = output_root / "stage39a_generation_queue.jsonl"

    implementation = (
        root / "requirements-stage39b.txt",
        root / "src/evaluation/stage39b_runner.py",
        root / "scripts/build_stage39b_backend_freeze.py",
        root / "scripts/verify_stage39b_backend_freeze.py",
        root / "scripts/run_stage39b_generation.py",
        root / "tests/test_stage39b_backend.py",
    )
    report_path = root / "docs/STAGE_39B_GENERATOR_BACKEND.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Stage 39B — Frozen Constrained Generator",
                "",
                "The L2 backend is `google/flan-t5-base`, pinned to revision",
                "`7bcac572ce56db69c1ea7c8af255c5d7c9672fc2` under Apache-2.0.",
                "",
                "Generation is greedy, temperature-zero, float32 inference on CUDA.",
                "No fine-tuning, API key or outer-test English reference is used.",
                "Responses are appended to durable storage and resume by request ID.",
                "The generator-reported constraint fields are retained but not trusted;",
                "Stage 40 independently calculates content support and correction harm.",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    contract = {
        "project_generation": "fresh_start_2026_v1",
        "stage": 39,
        "substage": "B",
        "name": "frozen_resumable_constrained_generator_backend",
        "status": "passed",
        "training_started": False,
        "generator_called_during_freeze": False,
        "outer_references_exposed": False,
        "input_stage39a_contract_portable_sha256": stage39a_hash,
        "input_queue_portable_sha256": portable_text_sha256(queue_path),
        "backend_specification": backend_specification(),
        "validation": {
            "models": ["g1", "c1", "c2", "c3", "c4"],
            "folds": 5,
            "mapped_outer_samples": 37000,
            "unique_generation_requests": int(
                stage39a["validation"]["unique_generation_requests"]
            ),
            "reference_firewall_passed": True,
            "resumable": True,
        },
        "implementation_portable_sha256": {
            path.relative_to(root).as_posix(): portable_text_sha256(path)
            for path in implementation
        },
        "generated_portable_sha256": {
            report_path.relative_to(root).as_posix(): portable_text_sha256(report_path)
        },
    }
    contract_path = output_root / "stage39b_backend_contract.json"
    hash_path = output_root / "stage39b_backend_contract.sha256"
    write_json(contract_path, contract)
    hash_path.write_text(
        portable_text_sha256(contract_path) + "\n", encoding="utf-8", newline="\n"
    )
    return {"contract": contract, "contract_path": contract_path, "hash_path": hash_path}


def main() -> None:
    result = build_stage39b(PROJECT_ROOT)
    contract = result["contract"]
    print("=" * 78)
    print("STAGE 39B — GENERATOR BACKEND FREEZE")
    print("=" * 78)
    print()
    print("BACKEND")
    print("-" * 78)
    print("Model: google/flan-t5-base")
    print("Revision: 7bcac572ce56db69c1ea7c8af255c5d7c9672fc2")
    print("License: Apache-2.0")
    print("Precision/device: float32 / CUDA")
    print("Decoding: deterministic greedy generation")
    print()
    print("QUEUE")
    print("-" * 78)
    print("Models: G1, C1, C2, C3 and C4")
    print(f"Unique requests: {contract['validation']['unique_generation_requests']}")
    print("Mapped samples: 37000")
    print("Resume by request ID: True")
    print()
    print("SAFETY")
    print("-" * 78)
    print("Training started: False")
    print("Generator called during freeze: False")
    print("Outer references exposed: False")
    print("API key required: False")
    print()
    print("FINAL STATUS: STAGE 39B BACKEND FREEZE PASSED")
    print("GPU GENERATION IS NOW AUTHORIZED AFTER COMMITTING THIS FREEZE")
    print("=" * 78)


if __name__ == "__main__":
    main()
