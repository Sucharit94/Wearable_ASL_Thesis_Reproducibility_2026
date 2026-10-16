"""Stage 39A constrained-generation queue and prompt-freeze helpers."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.evaluation.stage37_language_protocol import word_edit_distance


STAGE39_ROOT = Path("artifacts/language/stage39")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def request_id(value: Mapping[str, Any]) -> str:
    return "l2_" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:24]


def target_bank(
    target_rows: Sequence[Mapping[str, str]],
    assignment_rows: Sequence[Mapping[str, str]],
    outer_fold: int,
) -> list[dict[str, Any]]:
    allowed_ids = {
        int(row["sentence_id"])
        for row in assignment_rows
        if int(row["stage32_outer_fold"]) == outer_fold
        and row["partition"] != "outer_test"
    }
    targets = {
        int(row["sentence_id"]): {
            "sentence_id": int(row["sentence_id"]),
            "gloss_tokens": tuple(json.loads(row["gloss_tokens_json"])),
            "gloss": row["gloss_raw"],
            "english": row["english_raw"],
        }
        for row in target_rows
    }
    if not allowed_ids <= set(targets):
        raise RuntimeError("Training bank contains unknown sentence IDs.")
    result = [targets[sentence_id] for sentence_id in sorted(allowed_ids)]
    if len(result) != 300:
        raise RuntimeError(
            f"Fold {outer_fold} must expose exactly 300 training templates."
        )
    return result


def nearest_candidates(
    predicted_tokens: Sequence[str],
    bank: Sequence[Mapping[str, Any]],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    if top_k <= 0 or top_k > len(bank):
        raise ValueError("Invalid top-k value.")
    query = tuple(str(token).lower() for token in predicted_tokens)

    def ranking(row: Mapping[str, Any]) -> tuple[Any, ...]:
        tokens = tuple(str(token).lower() for token in row["gloss_tokens"])
        distance = word_edit_distance(query, tokens)
        normalized = distance / max(len(query), len(tokens), 1)
        return normalized, distance, int(row["sentence_id"])

    selected = sorted(bank, key=ranking)[:top_k]
    return [
        {
            "rank": rank,
            "sentence_id": int(row["sentence_id"]),
            "gloss": str(row["gloss"]),
            "english": str(row["english"]),
            "normalized_gloss_distance": ranking(row)[0],
        }
        for rank, row in enumerate(selected, 1)
    ]


def build_prompt(source_gloss: str, candidates: Sequence[Mapping[str, Any]]) -> str:
    examples = "\n".join(
        f"{row['rank']}. gloss: {row['gloss']}\n   English: {row['english']}"
        for row in candidates
    )
    return (
        "Convert the recognized ASL gloss into one grammatical English sentence.\n"
        "Preserve the supported meaning. Do not invent a person, place, number, "
        "object or action. Prefer a supplied training example only when its meaning "
        "matches the recognized gloss.\n\n"
        f"Recognized gloss: {source_gloss}\n\n"
        f"Training-only examples:\n{examples}\n\n"
        "Return only JSON with this schema:\n"
        '{"sentence":"...","added_content_tokens":[],"constraint_passed":true}'
    )


def build_generation_queue(
    stage38_rows: Sequence[Mapping[str, str]],
    target_rows: Sequence[Mapping[str, str]],
    assignment_rows: Sequence[Mapping[str, str]],
    top_k: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    l1_rows = [
        row
        for row in stage38_rows
        if row["system_id"] == "L1_training_only_retrieval"
    ]
    if len(l1_rows) != 37000:
        raise RuntimeError(f"Expected 37000 L1 rows, found {len(l1_rows)}.")
    banks = {
        fold: target_bank(target_rows, assignment_rows, fold)
        for fold in range(1, 6)
    }
    queue_by_key: dict[str, dict[str, Any]] = {}
    mappings: list[dict[str, Any]] = []
    for row in l1_rows:
        fold = int(row["outer_fold"])
        model_id = row["model_id"].lower()
        if model_id not in {"g1", "c1", "c2", "c3", "c4"}:
            raise RuntimeError(f"Unexpected Stage 38 model: {model_id}")
        source_gloss = row["source_predicted_gloss"].strip()
        predicted_tokens = tuple(source_gloss.split()) if source_gloss else tuple()
        candidates = nearest_candidates(predicted_tokens, banks[fold], top_k=top_k)
        request_payload = {
            "schema_version": 1,
            "model_id": model_id,
            "outer_fold": fold,
            "source_gloss": source_gloss,
            "training_candidates": candidates,
        }
        identifier = request_id(request_payload)
        if identifier not in queue_by_key:
            queue_by_key[identifier] = {
                "request_id": identifier,
                **request_payload,
                "prompt": build_prompt(source_gloss, candidates),
                "expected_output_schema": {
                    "sentence": "string",
                    "added_content_tokens": "array_of_strings",
                    "constraint_passed": "boolean",
                },
            }
        mappings.append(
            {
                "sample_id": row["sample_id"],
                "model_id": model_id,
                "outer_fold": fold,
                "request_id": identifier,
            }
        )
    if len({(row["sample_id"], row["model_id"]) for row in mappings}) != 37000:
        raise RuntimeError("Stage 39 sample/request mapping is not one-to-one.")
    queue = sorted(
        queue_by_key.values(),
        key=lambda row: (int(row["outer_fold"]), row["model_id"], row["request_id"]),
    )
    return queue, mappings


def validate_reference_firewall(queue: Sequence[Mapping[str, Any]]) -> None:
    forbidden_keys = {
        "reference_english",
        "ground_truth_english",
        "outer_reference",
        "true_english",
        "label_order_text",
        "sample_id",
        "sentence_id",
    }

    def walk(value: Any, path: str = "") -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key).lower() in forbidden_keys:
                    raise RuntimeError(f"Reference-firewall violation at {path}/{key}")
                walk(child, f"{path}/{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}/{index}")

    for row in queue:
        # Training candidate IDs are allowed provenance; top-level true IDs are not.
        sanitized = dict(row)
        candidates = sanitized.pop("training_candidates", [])
        walk(sanitized)
        for candidate in candidates:
            allowed = {
                "rank",
                "sentence_id",
                "gloss",
                "english",
                "normalized_gloss_distance",
            }
            if set(candidate) != allowed:
                raise RuntimeError("Unexpected training-candidate field.")


def generator_execution_template() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": "39B",
        "status": "BLOCKED_UNTIL_BACKEND_FROZEN",
        "backend": None,
        "model_identifier": None,
        "model_revision": None,
        "license_or_terms_recorded": False,
        "decoding": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 48,
            "responses_per_request": 1,
            "seed": 20260826,
        },
        "execution": {
            "resume_by_request_id": True,
            "overwrite_completed_response": False,
            "raw_response_saved": True,
            "parsed_response_saved": True,
            "api_key_saved_in_repository": False,
            "reference_english_available_to_generator": False,
        },
    }


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("JSONL row must be an object.")
            rows.append(value)
    return rows


def queue_statistics(
    queue: Sequence[Mapping[str, Any]], mappings: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    queue_counts: dict[tuple[str, int], int] = defaultdict(int)
    sample_counts: dict[tuple[str, int], int] = defaultdict(int)
    for row in queue:
        queue_counts[(str(row["model_id"]), int(row["outer_fold"]))] += 1
    for row in mappings:
        sample_counts[(str(row["model_id"]), int(row["outer_fold"]))] += 1
    return [
        {
            "model_id": model_id,
            "outer_fold": fold,
            "unique_generation_requests": queue_counts[(model_id, fold)],
            "mapped_outer_samples": sample_counts[(model_id, fold)],
            "deduplication_ratio": sample_counts[(model_id, fold)]
            / queue_counts[(model_id, fold)],
        }
        for fold in range(1, 6)
        for model_id in ("g1", "c1", "c2", "c3", "c4")
    ]


def render_report(statistics: Sequence[Mapping[str, Any]], queue_size: int) -> str:
    lines = [
        "# Stage 39A — Constrained-Generation Input Freeze",
        "",
        "Stage 39A freezes a reference-free, resumable L2 generation queue. It does not call a generator.",
        "",
        "## Safety boundary",
        "",
        "- Inputs contain recognized gloss and training-only candidate examples.",
        "- Outer-test English references and sample identifiers are excluded from prompts.",
        "- API keys must never be saved in the repository.",
        "- The backend, model identifier and revision must be recorded before Stage 39B.",
        "- Completed responses are immutable and execution resumes by request ID.",
        "",
        "## Queue",
        "",
        f"Unique generation requests: {queue_size}",
        "",
        "| Model | Fold | Unique requests | Outer samples | Deduplication |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in statistics:
        lines.append(
            f"| {str(row['model_id']).upper()} | {row['outer_fold']} | "
            f"{row['unique_generation_requests']} | {row['mapped_outer_samples']} | "
            f"{float(row['deduplication_ratio']):.2f}× |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Stage 39B is not authorized until one reproducible generator backend is selected. L2 must be compared with L0 and L1 using paired outputs; fluency alone is not evidence of improvement.",
            "",
        ]
    )
    return "\n".join(lines)
