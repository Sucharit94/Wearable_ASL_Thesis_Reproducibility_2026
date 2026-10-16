"""Verify Stage 39B backend freeze without importing Transformers."""

from pathlib import Path

from src.evaluation.stage37_language_protocol import portable_text_sha256
from src.evaluation.stage39b_runner import MODEL_IDENTIFIER, MODEL_REVISION, verify_stage39b_freeze


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    contract = verify_stage39b_freeze(PROJECT_ROOT)
    require(contract.get("stage") == 39 and contract.get("substage") == "B", "Stage identity failed.")
    require(contract.get("status") == "passed", "Stage 39B freeze did not pass.")
    require(contract.get("training_started") is False, "Training was started.")
    require(contract.get("generator_called_during_freeze") is False, "Generator was called during freeze.")
    require(contract.get("outer_references_exposed") is False, "Outer references were exposed.")
    require(contract["validation"]["models"] == ["g1", "c1", "c2", "c3", "c4"], "Model coverage changed.")
    for relative, expected in contract["implementation_portable_sha256"].items():
        path = PROJECT_ROOT / relative
        require(path.is_file() and portable_text_sha256(path) == expected, f"Implementation hash mismatch: {relative}")
    for relative, expected in contract["generated_portable_sha256"].items():
        path = PROJECT_ROOT / relative
        require(path.is_file() and portable_text_sha256(path) == expected, f"Generated hash mismatch: {relative}")

    print("=" * 78)
    print("STAGE 39B — GENERATOR BACKEND VERIFIER")
    print("=" * 78)
    print()
    print("Backend frozen: True")
    print(f"Model: {MODEL_IDENTIFIER}")
    print(f"Revision: {MODEL_REVISION}")
    print("All five recognizers covered: True")
    print("Reference firewall retained: True")
    print("Resumable durable execution: True")
    print("Training started: False")
    print("Generator called by verifier: False")
    print()
    print("FINAL STATUS: STAGE 39B BACKEND FREEZE PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()
