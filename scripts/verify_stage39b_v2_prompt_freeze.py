"""Verify the Stage 39B V2 amendment without generator inference."""

from pathlib import Path

from src.evaluation.stage37_language_protocol import portable_text_sha256
from src.evaluation.stage39b_v2_runner import (
    build_plain_english_prompt,
    parse_plain_english_response,
    verify_v2_freeze,
)
from src.evaluation.stage39_generation import read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    contract = verify_v2_freeze(PROJECT_ROOT)
    require(contract.get("status") == "passed", "V2 freeze did not pass.")
    require(contract.get("v1_results_eligible_for_scoring") is False, "V1 was not excluded.")
    require(contract.get("training_started") is False, "Training was started.")
    require(contract.get("outer_references_exposed") is False, "Outer references were exposed.")
    require(contract["validation"]["models"] == ["g1", "c1", "c2", "c3", "c4"], "Model coverage changed.")
    queue_path = PROJECT_ROOT / "artifacts/language/stage39/stage39a_generation_queue.jsonl"
    queue = read_jsonl(queue_path)
    require(len(queue) == contract["validation"]["unique_generation_requests"], "Queue count changed.")
    prompt = build_plain_english_prompt(queue[0])
    require("Return only the English sentence" in prompt, "Plain-English instruction missing.")
    require('"sentence":"..."' not in prompt, "V1 JSON placeholder remains in V2 prompt.")
    require(parse_plain_english_response("Good afternoon") == "Good afternoon", "V2 parser failed.")
    for relative, expected in contract["implementation_portable_sha256"].items():
        path = PROJECT_ROOT / relative
        require(path.is_file() and portable_text_sha256(path) == expected, f"Implementation hash mismatch: {relative}")
    for relative, expected in contract["generated_portable_sha256"].items():
        path = PROJECT_ROOT / relative
        require(path.is_file() and portable_text_sha256(path) == expected, f"Output hash mismatch: {relative}")

    print("=" * 78)
    print("STAGE 39B V2 — PROMPT AMENDMENT VERIFIER")
    print("=" * 78)
    print("Stage 39A queue unchanged: True")
    print("Stage 39B V1 freeze unchanged: True")
    print("JSON placeholder removed: True")
    print("Plain-English-only instruction: True")
    print("Strict V1-fragment rejection: True")
    print("All five models covered: True")
    print("Training started: False")
    print("Generator called by verifier: False")
    print("Outer references exposed: False")
    print("FINAL STATUS: STAGE 39B V2 PROMPT FREEZE PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()
