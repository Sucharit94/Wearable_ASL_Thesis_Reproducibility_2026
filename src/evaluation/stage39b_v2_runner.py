"""Stage 39B V2 resumable plain-English FLAN-T5 runner.

V2 replaces the unsuccessful JSON-output instruction used by V1.  It consumes
the same immutable, reference-free Stage 39A request payload and constructs a
plain-English prompt from the recorded source gloss and training-only examples.
"""

from __future__ import annotations

import os
import random
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluation.stage37_language_protocol import portable_text_sha256, read_json
from src.evaluation.stage39_generation import canonical_json, read_jsonl


MODEL_IDENTIFIER = "google/flan-t5-base"
MODEL_REVISION = "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"
MODEL_LICENSE = "Apache-2.0"
BACKEND = "huggingface_transformers_local"
PROMPT_VERSION = 2
SEED = 20260826
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 48
DEFAULT_BATCH_SIZE = 16
BAD_OUTPUT_MARKERS = (
    "added_content_tokens",
    "constraint_passed",
    'sentence":"',
    "...",
)


def build_plain_english_prompt(request: Mapping[str, Any]) -> str:
    example_lines: list[str] = []
    candidates = request.get("training_candidates")
    if not isinstance(candidates, list) or len(candidates) != 3:
        raise RuntimeError("Every V2 request requires three training-only examples.")
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise RuntimeError("Invalid training-only example.")
        example_lines.extend(
            [
                f"ASL gloss: {candidate['gloss']}",
                f"English: {candidate['english']}",
                "",
            ]
        )
    examples = "\n".join(example_lines)
    return (
        "Rewrite the recognized ASL gloss as one natural English sentence.\n"
        "Preserve the meaning and do not add unsupported people, places, "
        "objects, numbers or actions.\n"
        "Return only the English sentence. Do not return JSON, labels or "
        "explanations.\n\n"
        "Training-only examples:\n"
        f"{examples}"
        "Now rewrite this input:\n"
        f"ASL gloss: {request['source_gloss']}\n"
        "English:"
    )


def parse_plain_english_response(raw_response: str) -> str:
    sentence = str(raw_response).strip()
    sentence = re.sub(r"^English\s*:\s*", "", sentence, flags=re.IGNORECASE)
    sentence = sentence.strip().strip('"').strip()
    if not sentence:
        raise ValueError("Generator returned an empty V2 sentence.")
    lowered = sentence.lower()
    if any(marker.lower() in lowered for marker in BAD_OUTPUT_MARKERS):
        raise ValueError("Generator repeated a forbidden V1 schema fragment.")
    if "\n" in sentence or "\r" in sentence:
        raise ValueError("Generator returned more than one output line.")
    if len(sentence.split()) > MAX_NEW_TOKENS:
        raise ValueError("Generated sentence exceeds the frozen output limit.")
    return sentence


def v2_backend_specification() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "stage": "39B-V2",
        "status": "FROZEN_AND_AUTHORIZED",
        "reason_for_amendment": "V1_generator_echoed_json_schema_placeholder",
        "backend": BACKEND,
        "model_identifier": MODEL_IDENTIFIER,
        "model_revision": MODEL_REVISION,
        "license": MODEL_LICENSE,
        "prompt_version": PROMPT_VERSION,
        "output_contract": "one_plain_english_sentence",
        "precision": "float32",
        "device": "cuda",
        "training": False,
        "decoding": {
            "strategy": "greedy",
            "do_sample": False,
            "temperature": 0.0,
            "num_beams": 1,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "responses_per_request": 1,
            "seed": SEED,
        },
        "execution": {
            "queue": "artifacts/language/stage39/stage39a_generation_queue.jsonl",
            "response_file": "stage39b_v2_responses.jsonl",
            "resume_key": "request_id",
            "completed_responses_immutable": True,
            "reject_v1_schema_fragments": True,
            "outer_reference_available_to_generator": False,
            "api_key_required": False,
        },
    }


def verify_v2_freeze(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    stage39_root = root / "artifacts/language/stage39"
    contract_path = stage39_root / "stage39b_v2_contract.json"
    hash_path = stage39_root / "stage39b_v2_contract.sha256"
    if not contract_path.is_file() or not hash_path.is_file():
        raise FileNotFoundError("Frozen Stage 39B V2 contract is missing.")
    observed = portable_text_sha256(contract_path)
    recorded = hash_path.read_text(encoding="utf-8-sig").strip().split()[0]
    if recorded != observed:
        raise RuntimeError("Stage 39B V2 contract hash mismatch.")
    contract = read_json(contract_path)
    if contract.get("backend_specification") != v2_backend_specification():
        raise RuntimeError("Stage 39B V2 backend specification changed.")
    for field, relative in (
        ("input_stage39a_contract_portable_sha256", "stage39a_contract.json"),
        ("input_stage39b_v1_contract_portable_sha256", "stage39b_backend_contract.json"),
        ("input_queue_portable_sha256", "stage39a_generation_queue.jsonl"),
    ):
        path = stage39_root / relative
        if portable_text_sha256(path) != contract.get(field):
            raise RuntimeError(f"Frozen V2 input changed: {relative}")
    return contract


def read_completed_v2(path: str | Path) -> dict[str, dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(source):
        identifier = str(row.get("request_id", ""))
        if (
            not identifier
            or row.get("status") != "COMPLETE"
            or row.get("prompt_version") != PROMPT_VERSION
        ):
            raise RuntimeError("Saved Stage 39B V2 response is invalid.")
        parse_plain_english_response(str(row.get("sentence", "")))
        if identifier in result:
            raise RuntimeError(f"Duplicate V2 response: {identifier}")
        result[identifier] = row
    return result


def append_v2_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def pending_v2(
    queue: Sequence[Mapping[str, Any]], completed: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    queue_ids = [str(row["request_id"]) for row in queue]
    if len(queue_ids) != len(set(queue_ids)):
        raise RuntimeError("Stage 39A queue has duplicate request IDs.")
    if set(completed) - set(queue_ids):
        raise RuntimeError("V2 output contains a request outside the frozen queue.")
    return [dict(row) for row in queue if str(row["request_id"]) not in completed]


def make_v2_record(
    request: Mapping[str, Any], raw_response: str, sentence: str
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "status": "COMPLETE",
        "prompt_version": PROMPT_VERSION,
        "request_id": str(request["request_id"]),
        "model_id": str(request["model_id"]),
        "outer_fold": int(request["outer_fold"]),
        "source_gloss": str(request["source_gloss"]),
        "raw_response": str(raw_response),
        "sentence": sentence,
        "backend": BACKEND,
        "model_identifier": MODEL_IDENTIFIER,
        "model_revision": MODEL_REVISION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_v2_generation(
    project_root: str | Path,
    output_root: str | Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_requests: int | None = None,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    root = Path(project_root).resolve()
    contract = verify_v2_freeze(root)
    queue_path = root / contract["backend_specification"]["execution"]["queue"]
    queue = read_jsonl(queue_path)
    expected = int(contract["validation"]["unique_generation_requests"])
    if len(queue) != expected:
        raise RuntimeError("Frozen queue count changed.")
    output = Path(output_root).resolve() / "stage39b_v2_responses.jsonl"
    completed = read_completed_v2(output)
    pending = pending_v2(queue, completed)
    if max_requests is not None:
        if max_requests <= 0:
            raise ValueError("max_requests must be positive.")
        pending = pending[:max_requests]
    if not pending:
        return {
            "status": "COMPLETE" if len(completed) == expected else "PAUSED",
            "completed": len(completed),
            "total": expected,
            "response_path": str(output),
        }

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 39B V2 requires CUDA.")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDENTIFIER, revision=MODEL_REVISION)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_IDENTIFIER,
        revision=MODEL_REVISION,
        dtype=torch.float32,
        use_safetensors=True,
    ).to("cuda")
    model.eval()

    processed = len(completed)
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        prompts = [build_plain_english_prompt(row) for row in batch]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        ).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        raw = tokenizer.batch_decode(generated, skip_special_tokens=True)
        rows = [
            make_v2_record(request, response, parse_plain_english_response(response))
            for request, response in zip(batch, raw, strict=True)
        ]
        append_v2_rows(output, rows)
        processed += len(rows)
        print(f"Completed {processed}/{expected} V2 requests", flush=True)

    final = read_completed_v2(output)
    return {
        "status": "COMPLETE" if len(final) == expected else "PAUSED",
        "completed": len(final),
        "total": expected,
        "response_path": str(output),
    }
