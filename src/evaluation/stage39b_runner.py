"""Frozen, resumable Stage 39B constrained-generation runner.

The runner consumes only the reference-free Stage 39A queue.  Model imports are
lazy so the local freeze/verifier never downloads or invokes the generator.
"""

from __future__ import annotations

import json
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
SEED = 20260826
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 48
DEFAULT_BATCH_SIZE = 16


def backend_specification() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": "39B",
        "status": "FROZEN_AND_AUTHORIZED",
        "backend": BACKEND,
        "model_identifier": MODEL_IDENTIFIER,
        "model_revision": MODEL_REVISION,
        "license": MODEL_LICENSE,
        "task": "constrained_gloss_to_english_text2text_generation",
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
        "dependencies": {
            "transformers": "4.57.1",
            "sentencepiece": "0.2.1",
        },
        "execution": {
            "queue": "artifacts/language/stage39/stage39a_generation_queue.jsonl",
            "resume_key": "request_id",
            "completed_responses_immutable": True,
            "raw_response_saved": True,
            "parsed_response_saved": True,
            "outer_reference_available_to_generator": False,
            "api_key_required": False,
            "output_root_is_external_and_durable": True,
        },
    }


def verify_stage39b_freeze(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    stage39_root = root / "artifacts/language/stage39"
    contract_path = stage39_root / "stage39b_backend_contract.json"
    hash_path = stage39_root / "stage39b_backend_contract.sha256"
    if not contract_path.is_file() or not hash_path.is_file():
        raise FileNotFoundError("Frozen Stage 39B backend contract is missing.")
    observed = portable_text_sha256(contract_path)
    recorded = hash_path.read_text(encoding="utf-8-sig").strip().split()[0]
    if observed != recorded:
        raise RuntimeError("Stage 39B backend contract hash mismatch.")
    contract = read_json(contract_path)
    if contract.get("backend_specification") != backend_specification():
        raise RuntimeError("Stage 39B backend specification changed.")
    stage39a_path = stage39_root / "stage39a_contract.json"
    if portable_text_sha256(stage39a_path) != contract.get(
        "input_stage39a_contract_portable_sha256"
    ):
        raise RuntimeError("Stage 39A changed after the backend freeze.")
    queue_path = stage39_root / "stage39a_generation_queue.jsonl"
    if portable_text_sha256(queue_path) != contract.get(
        "input_queue_portable_sha256"
    ):
        raise RuntimeError("Stage 39A generation queue changed.")
    return contract


def parse_generator_response(raw_response: str) -> dict[str, Any]:
    """Parse requested JSON, while retaining a deterministic raw-text fallback."""
    raw = str(raw_response).strip()
    unfenced = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    unfenced = re.sub(r"\s*```$", "", unfenced).strip()
    candidates = [unfenced]
    left, right = unfenced.find("{"), unfenced.rfind("}")
    if left >= 0 and right > left:
        candidates.append(unfenced[left : right + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping) and isinstance(value.get("sentence"), str):
            sentence = value["sentence"].strip()
            if sentence:
                added = value.get("added_content_tokens", [])
                if not isinstance(added, list) or not all(
                    isinstance(token, str) for token in added
                ):
                    added = []
                passed = value.get("constraint_passed")
                return {
                    "sentence": sentence,
                    "reported_added_content_tokens": added,
                    "reported_constraint_passed": (
                        passed if isinstance(passed, bool) else None
                    ),
                    "parse_mode": "json",
                }
    fallback = re.sub(r"^(sentence|english)\s*:\s*", "", unfenced, flags=re.I)
    fallback = fallback.strip().strip('"').strip()
    if not fallback:
        raise ValueError("Generator returned no usable sentence.")
    return {
        "sentence": fallback,
        "reported_added_content_tokens": [],
        "reported_constraint_passed": None,
        "parse_mode": "raw_text_fallback",
    }


def read_completed_responses(path: str | Path) -> dict[str, dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(source):
        identifier = str(row.get("request_id", ""))
        if not identifier or row.get("status") != "COMPLETE":
            raise RuntimeError("Saved Stage 39B response is incomplete or invalid.")
        if identifier in result:
            raise RuntimeError(f"Duplicate completed response: {identifier}")
        result[identifier] = row
    return result


def append_response_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def pending_requests(
    queue: Sequence[Mapping[str, Any]], completed: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    identifiers = [str(row["request_id"]) for row in queue]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Stage 39A queue contains duplicate request IDs.")
    unknown = set(completed) - set(identifiers)
    if unknown:
        raise RuntimeError("Saved responses contain request IDs outside the frozen queue.")
    return [dict(row) for row in queue if str(row["request_id"]) not in completed]


def make_response_record(
    request: Mapping[str, Any], raw_response: str, parsed: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "COMPLETE",
        "request_id": str(request["request_id"]),
        "model_id": str(request["model_id"]),
        "outer_fold": int(request["outer_fold"]),
        "source_gloss": str(request["source_gloss"]),
        "raw_response": str(raw_response),
        "sentence": str(parsed["sentence"]),
        "parse_mode": str(parsed["parse_mode"]),
        "reported_added_content_tokens": list(
            parsed["reported_added_content_tokens"]
        ),
        "reported_constraint_passed": parsed["reported_constraint_passed"],
        "backend": BACKEND,
        "model_identifier": MODEL_IDENTIFIER,
        "model_revision": MODEL_REVISION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_generation(
    project_root: str | Path,
    output_root: str | Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_requests: int | None = None,
) -> dict[str, Any]:
    """Run or resume greedy FLAN-T5 generation on the frozen queue."""
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    root = Path(project_root).resolve()
    contract = verify_stage39b_freeze(root)
    queue_path = root / contract["backend_specification"]["execution"]["queue"]
    queue = read_jsonl(queue_path)
    expected = int(contract["validation"]["unique_generation_requests"])
    if len(queue) != expected:
        raise RuntimeError("Frozen Stage 39A queue count changed.")

    durable_root = Path(output_root).resolve()
    response_path = durable_root / "stage39b_responses.jsonl"
    completed = read_completed_responses(response_path)
    pending = pending_requests(queue, completed)
    if max_requests is not None:
        if max_requests <= 0:
            raise ValueError("max_requests must be positive when supplied.")
        pending = pending[:max_requests]
    if not pending:
        return {
            "status": "COMPLETE" if len(completed) == expected else "PAUSED",
            "completed": len(completed),
            "total": expected,
            "response_path": str(response_path),
        }

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 39B requires a CUDA GPU.")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_IDENTIFIER, revision=MODEL_REVISION
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_IDENTIFIER,
        revision=MODEL_REVISION,
        torch_dtype=torch.float32,
        use_safetensors=True,
    ).to("cuda")
    model.eval()

    processed = len(completed)
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        prompts = [str(row["prompt"]) for row in batch]
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
        raw_responses = tokenizer.batch_decode(generated, skip_special_tokens=True)
        rows = [
            make_response_record(request, raw, parse_generator_response(raw))
            for request, raw in zip(batch, raw_responses, strict=True)
        ]
        append_response_rows(response_path, rows)
        processed += len(rows)
        print(f"Completed {processed}/{expected} generation requests", flush=True)

    final_completed = read_completed_responses(response_path)
    status = "COMPLETE" if len(final_completed) == expected else "PAUSED"
    return {
        "status": status,
        "completed": len(final_completed),
        "total": expected,
        "response_path": str(response_path),
    }
