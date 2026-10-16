"""Stage 38 immutable prediction assembly and deterministic retrieval baselines."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation.stage37_language_protocol import deterministic_retrieval


STAGE38_ROOT = Path("artifacts/language/stage38")
LANGUAGE_MODELS = ("g1", "c1", "c2", "c3", "c4")
WORD_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
FUNCTION_WORDS = frozenset(
    {
        "a", "an", "the", "am", "are", "is", "was", "were", "be", "been",
        "being", "do", "does", "did", "to", "of", "for", "in", "on", "at",
        "and", "or", "but", "if", "then", "than", "that", "this", "these",
        "those", "i", "me", "my", "mine", "you", "your", "yours", "he", "him",
        "his", "she", "her", "hers", "it", "its", "we", "us", "our", "ours",
        "they", "them", "their", "theirs", "what", "when", "where", "who",
        "why", "how", "with", "from", "as", "by", "about", "can", "could",
        "will", "would", "shall", "should", "may", "might", "must", "not",
    }
)


@dataclass(frozen=True)
class PredictionSource:
    model_id: str
    outer_fold: int
    location: str
    data: bytes
    sha256: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tokenize_text(value: str) -> tuple[str, ...]:
    return tuple(WORD_PATTERN.findall(str(value).lower().replace("_", " ")))


def parse_jsonl(data: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(data.decode("utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise TypeError(f"Prediction line {line_number} is not an object.")
        rows.append(value)
    return rows


def _prediction_suffix(model_id: str, outer_fold: int) -> re.Pattern[str]:
    return re.compile(
        rf"(?:^|/)fold0?{outer_fold}/{re.escape(model_id)}/predictions\.jsonl$",
        flags=re.IGNORECASE,
    )


def locate_prediction_source(
    project_root: str | Path, model_id: str, outer_fold: int
) -> PredictionSource:
    root = Path(project_root).resolve()
    pattern = _prediction_suffix(model_id, outer_fold)
    found: list[tuple[str, bytes]] = []

    search_roots = [root / "artifacts", root / "imports"]
    for search_root in search_roots:
        if not search_root.is_dir():
            continue
        for path in search_root.rglob("predictions.jsonl"):
            relative = path.relative_to(root).as_posix()
            if pattern.search(relative):
                found.append((relative, path.read_bytes()))

    imports = root / "imports"
    archives = sorted(imports.rglob("*.zip")) if imports.is_dir() else []
    for archive in archives:
        try:
            with zipfile.ZipFile(archive) as handle:
                for name in handle.namelist():
                    normalized = name.replace("\\", "/").lstrip("./")
                    if pattern.search(normalized):
                        found.append(
                            (
                                f"{archive.relative_to(root).as_posix()}::{normalized}",
                                handle.read(name),
                            )
                        )
        except zipfile.BadZipFile:
            continue

    if not found:
        raise FileNotFoundError(
            f"Missing {model_id.upper()} Fold {outer_fold} predictions. "
            "Place stage32c_results_without_checkpoints.zip inside the project imports folder."
        )
    by_hash: dict[str, list[tuple[str, bytes]]] = defaultdict(list)
    for location, data in found:
        by_hash[sha256_bytes(data)].append((location, data))
    if len(by_hash) != 1:
        locations = sorted(location for location, _ in found)
        raise RuntimeError(
            f"Conflicting {model_id.upper()} Fold {outer_fold} prediction copies: {locations}"
        )
    digest = next(iter(by_hash))
    location, data = sorted(by_hash[digest], key=lambda item: item[0])[0]
    return PredictionSource(model_id, outer_fold, location, data, digest)


def prediction_sources(project_root: str | Path) -> list[PredictionSource]:
    return [
        locate_prediction_source(project_root, model_id, fold)
        for fold in range(1, 6)
        for model_id in LANGUAGE_MODELS
    ]


def _target_index(target_rows: Sequence[Mapping[str, str]]) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in target_rows:
        sentence_id = int(row["sentence_id"])
        indexed[sentence_id] = {
            "sentence_id": sentence_id,
            "gloss_tokens": tuple(json.loads(row["gloss_tokens_json"])),
            "reference_gloss": row["gloss_raw"],
            "reference_english": row["english_raw"],
        }
    return indexed


def _outer_assignment_index(
    assignment_rows: Sequence[Mapping[str, str]], outer_fold: int
) -> tuple[dict[str, Mapping[str, str]], set[int]]:
    outer: dict[str, Mapping[str, str]] = {}
    training_sentence_ids: set[int] = set()
    for row in assignment_rows:
        if int(row["stage32_outer_fold"]) != outer_fold:
            continue
        if row["partition"] == "outer_test":
            sample_id = row["sample_id"]
            if sample_id in outer:
                raise RuntimeError(f"Duplicate outer assignment: {sample_id}")
            outer[sample_id] = row
        else:
            training_sentence_ids.add(int(row["sentence_id"]))
    return outer, training_sentence_ids


def validate_predictions(
    source: PredictionSource,
    predictions: Sequence[Mapping[str, Any]],
    outer_assignments: Mapping[str, Mapping[str, str]],
    targets: Mapping[int, Mapping[str, Any]],
    expected_count: int = 1480,
) -> None:
    if len(predictions) != expected_count:
        raise RuntimeError(
            f"{source.model_id.upper()} Fold {source.outer_fold}: expected "
            f"{expected_count} predictions, found {len(predictions)}."
        )
    sample_ids = [str(row["sample_id"]) for row in predictions]
    if len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError("Duplicate prediction sample IDs detected.")
    if set(sample_ids) != set(outer_assignments):
        raise RuntimeError(
            f"{source.model_id.upper()} Fold {source.outer_fold} prediction/test IDs differ."
        )
    for row in predictions:
        sample_id = str(row["sample_id"])
        assignment = outer_assignments[sample_id]
        sentence_id = int(row["sentence_id"])
        if sentence_id != int(assignment["sentence_id"]):
            raise RuntimeError(f"Sentence ID mismatch: {sample_id}")
        if sentence_id not in targets:
            raise RuntimeError(f"Unknown sentence ID: {sentence_id}")
        predicted_tokens = row.get("predicted_tokens")
        reference_tokens = row.get("reference_tokens")
        if not isinstance(predicted_tokens, list) or not isinstance(reference_tokens, list):
            raise RuntimeError(f"Invalid token fields: {sample_id}")
        expected_reference = tuple(str(x) for x in targets[sentence_id]["gloss_tokens"])
        if tuple(str(x) for x in reference_tokens) != expected_reference:
            raise RuntimeError(f"Reference gloss changed: {sample_id}")


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> tuple[int, int, int]:
    # Each cell stores (cost, substitutions, deletions, insertions).
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, ref_token in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, hyp_token in enumerate(hypothesis, 1):
            if ref_token == hyp_token:
                candidates = [previous[j - 1]]
            else:
                cost, sub, delete, insert = previous[j - 1]
                candidates = [(cost + 1, sub + 1, delete, insert)]
            cost, sub, delete, insert = previous[j]
            candidates.append((cost + 1, sub, delete + 1, insert))
            cost, sub, delete, insert = current[j - 1]
            candidates.append((cost + 1, sub, delete, insert + 1))
            current.append(min(candidates))
        previous = current
    _, substitutions, deletions, insertions = previous[-1]
    return substitutions, deletions, insertions


def multiset_f1(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    ref = Counter(reference)
    hyp = Counter(hypothesis)
    overlap = sum((ref & hyp).values())
    if not ref and not hyp:
        return 1.0
    precision = overlap / sum(hyp.values()) if hyp else 0.0
    recall = overlap / sum(ref.values()) if ref else 0.0
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def content_tokens(tokens: Sequence[str]) -> tuple[str, ...]:
    return tuple(token for token in tokens if token not in FUNCTION_WORDS)


def execute_retrieval_baselines(
    assignment_rows: Sequence[Mapping[str, str]],
    target_rows: Sequence[Mapping[str, str]],
    sources: Sequence[PredictionSource],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    targets = _target_index(target_rows)
    output_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    seen_by_model: dict[str, set[str]] = defaultdict(set)
    for source in sources:
        predictions = parse_jsonl(source.data)
        outer, training_sentence_ids = _outer_assignment_index(
            assignment_rows, source.outer_fold
        )
        validate_predictions(source, predictions, outer, targets)
        bank = [targets[sentence_id] for sentence_id in sorted(training_sentence_ids)]
        if len(bank) != 300:
            raise RuntimeError(
                f"Fold {source.outer_fold} final retrieval bank must contain 300 templates."
            )
        cache: dict[tuple[str, ...], Mapping[str, Any]] = {}
        for prediction in predictions:
            sample_id = str(prediction["sample_id"])
            if sample_id in seen_by_model[source.model_id]:
                raise RuntimeError(f"Sample repeated across folds for {source.model_id}: {sample_id}")
            seen_by_model[source.model_id].add(sample_id)
            sentence_id = int(prediction["sentence_id"])
            target = targets[sentence_id]
            predicted = tuple(str(value) for value in prediction["predicted_tokens"])
            if predicted not in cache:
                cache[predicted] = deterministic_retrieval(predicted, bank)
            retrieved = cache[predicted]
            reference_english_tokens = tokenize_text(target["reference_english"])
            l0_tokens = tuple(
                token
                for gloss_token in predicted
                for token in tokenize_text(gloss_token)
            )
            l1_tokens = tokenize_text(str(retrieved["reference_english"]))
            for system_id, output_tokens, output_text in (
                ("L0_raw_gloss", l0_tokens, " ".join(l0_tokens)),
                ("L1_training_only_retrieval", l1_tokens, str(retrieved["reference_english"])),
            ):
                substitutions, deletions, insertions = edit_counts(
                    reference_english_tokens, output_tokens
                )
                output_content = content_tokens(output_tokens)
                # L0 repeats the source and L1 returns a verified training-catalog
                # sentence, so unsupported generation is impossible by construction.
                # This safety metric becomes discriminative for L2 in Stage 39.
                unsupported = 0
                output_rows.append(
                    {
                        "sample_id": sample_id,
                        "subject": prediction["subject"],
                        "sentence_id": sentence_id,
                        "outer_fold": source.outer_fold,
                        "model_id": source.model_id,
                        "system_id": system_id,
                        "source_predicted_gloss": " ".join(predicted),
                        "reference_gloss": target["reference_gloss"],
                        "reference_english": target["reference_english"],
                        "output_text": output_text,
                        "retrieved_sentence_id": (
                            "" if system_id == "L0_raw_gloss" else int(retrieved["sentence_id"])
                        ),
                        "substitutions": substitutions,
                        "deletions": deletions,
                        "insertions": insertions,
                        "edit_distance": substitutions + deletions + insertions,
                        "reference_words": len(reference_english_tokens),
                        "exact_match": output_tokens == reference_english_tokens,
                        "content_token_f1": multiset_f1(
                            content_tokens(reference_english_tokens), output_content
                        ),
                        "unsupported_content_tokens": unsupported,
                        "output_content_tokens": len(output_content),
                    }
                )
        manifest_rows.append(
            {
                "model_id": source.model_id,
                "outer_fold": source.outer_fold,
                "predictions": len(predictions),
                "prediction_sha256": source.sha256,
                "source_location": source.location,
                "outer_sample_ids_match": True,
                "training_templates_in_retrieval_bank": len(bank),
                "outer_samples_in_retrieval_bank": 0,
            }
        )
    for model_id in LANGUAGE_MODELS:
        if len(seen_by_model[model_id]) != 7400:
            raise RuntimeError(f"{model_id.upper()} does not contain 7400 unique predictions.")
    summary_rows = summarize_outputs(output_rows)
    return output_rows, summary_rows, manifest_rows


def summarize_outputs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model_id"]), str(row["system_id"]))].append(row)
    summaries: list[dict[str, Any]] = []
    for model_id in LANGUAGE_MODELS:
        by_system = {
            system: grouped[(model_id, system)]
            for system in ("L0_raw_gloss", "L1_training_only_retrieval")
        }
        l0_by_sample = {row["sample_id"]: row for row in by_system["L0_raw_gloss"]}
        for system_id, items in by_system.items():
            reference_words = sum(int(row["reference_words"]) for row in items)
            edits = sum(int(row["edit_distance"]) for row in items)
            exact = sum(str(row["exact_match"]).lower() == "true" if isinstance(row["exact_match"], str) else bool(row["exact_match"]) for row in items)
            output_content = sum(int(row["output_content_tokens"]) for row in items)
            unsupported = sum(int(row["unsupported_content_tokens"]) for row in items)
            gain = harm = gain_base = harm_base = 0
            if system_id == "L1_training_only_retrieval":
                for row in items:
                    l0_exact = bool(l0_by_sample[row["sample_id"]]["exact_match"])
                    l1_exact = bool(row["exact_match"])
                    gain_base += int(not l0_exact)
                    harm_base += int(l0_exact)
                    gain += int((not l0_exact) and l1_exact)
                    harm += int(l0_exact and (not l1_exact))
            summaries.append(
                {
                    "model_id": model_id,
                    "system_id": system_id,
                    "samples": len(items),
                    "english_word_error_rate": edits / reference_words,
                    "exact_sentence_accuracy": exact / len(items),
                    "mean_content_token_f1": sum(float(row["content_token_f1"]) for row in items) / len(items),
                    "unsupported_content_rate": unsupported / output_content if output_content else 0.0,
                    "correction_gain_rate": gain / gain_base if gain_base else "",
                    "correction_harm_rate": harm / harm_base if harm_base else "",
                    "substitutions": sum(int(row["substitutions"]) for row in items),
                    "deletions": sum(int(row["deletions"]) for row in items),
                    "insertions": sum(int(row["insertions"]) for row in items),
                }
            )
    return summaries


def render_report(summary_rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Stage 38 — Immutable Predictions and Retrieval Baselines",
        "",
            "Stage 38 reuses frozen outer-test predictions. It does not train or rescore a recognizer and does not call a generative language model.",
        "",
        "## Results",
        "",
        "| Recognizer | System | English WER | Exact accuracy | Content F1 | Unsupported content |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {str(row['model_id']).upper()} | {row['system_id']} | "
            f"{float(row['english_word_error_rate']):.6f} | "
            f"{float(row['exact_sentence_accuracy']):.6f} | "
            f"{float(row['mean_content_token_f1']):.6f} | "
            f"{float(row['unsupported_content_rate']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "L1 is a deterministic closed-set retrieval baseline. Because every fold's training partition contains all 300 sentence templates, strong L1 results demonstrate template-based correction, not unrestricted generation to unseen sentence structures.",
            "",
            "Unsupported-content rate is zero by construction for L0 and L1: L0 repeats the source and L1 returns a verified training-catalog sentence. It becomes a discriminative safety metric for constrained generation in Stage 39.",
            "",
            "All five recognizers receive identical L0/L1 processing. C2 remains the primary lightweight recommendation and G1 remains the recognition-quality reference. No conclusion is based on accuracy alone.",
            "",
        ]
    )
    return "\n".join(lines)
