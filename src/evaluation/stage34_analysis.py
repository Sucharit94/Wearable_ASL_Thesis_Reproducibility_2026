"""Deterministic Stage 34B analysis of frozen five-fold predictions."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


MODEL_ORDER = ("g1", "c1", "c2", "c3", "c4")
EXPECTED_PAIRS = {(model, fold) for fold in range(1, 6) for model in MODEL_ORDER}
EXPECTED_ARCHIVE_SHA256 = "262d6a91f03db1f07ee3a8ad2509117dcdfda5773b53f99a06ef2a7e2c12eac7"
EXPECTED_SAMPLES_PER_FOLD = 1480
EXPECTED_SAMPLES_PER_MODEL = 7400
EXPECTED_REFERENCE_TOKENS_PER_MODEL = 27940
EXPECTED_PARAMETERS = {
    "g1": 4_791_688,
    "c1": 392_488,
    "c2": 656_168,
    "c3": 237_352,
    "c4": 348_604,
}


_TEXT_HASH_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".py", ".txt", ".yaml", ".yml"}


def sha256_file(path: str | Path) -> str:
    """Hash text canonically so Git line-ending conversion is harmless."""
    target = Path(path)
    digest = hashlib.sha256()
    if target.suffix.lower() in _TEXT_HASH_SUFFIXES:
        data = target.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest.update(data)
        return digest.hexdigest()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}.")
            rows.append(value)
    return rows


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def find_unique(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {filename}; found {len(matches)}.")
    return matches[0]


def prediction_identity(path: str | Path) -> tuple[str, int]:
    normalized = Path(path).as_posix().lower()
    match = re.search(
        r"/fold0?([1-5])/(g1|c1|c2|c3|c4)/predictions\.jsonl$",
        normalized,
    )
    if match is None:
        raise ValueError(f"Cannot identify model/fold from path: {path}")
    return match.group(2), int(match.group(1))


def reference_length_bin(length: int) -> str:
    if length <= 0:
        raise ValueError("Reference length must be positive.")
    return str(length) if length <= 4 else "5+"


def aggregate_records(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not records:
        raise ValueError("Cannot aggregate an empty prediction collection.")

    samples = len(records)
    reference_tokens = sum(len(row["reference_token_ids"]) for row in records)
    hypothesis_tokens = sum(len(row["predicted_token_ids"]) for row in records)
    substitutions = sum(int(row["substitutions"]) for row in records)
    deletions = sum(int(row["deletions"]) for row in records)
    insertions = sum(int(row["insertions"]) for row in records)
    total_edits = substitutions + deletions + insertions
    sentence_errors = sum(not bool(row["exact_match"]) for row in records)
    blank_outputs = sum(len(row["predicted_token_ids"]) == 0 for row in records)
    matches = reference_tokens - substitutions - deletions
    if reference_tokens <= 0:
        raise ValueError("Reference token total must be positive.")

    macro_sample_wer = statistics.fmean(
        int(row["edit_distance"]) / len(row["reference_token_ids"])
        for row in records
    )
    return {
        "samples": samples,
        "reference_tokens": reference_tokens,
        "hypothesis_tokens": hypothesis_tokens,
        "matches": matches,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "total_edits": total_edits,
        "pooled_wer": total_edits / reference_tokens,
        "macro_sample_wer": macro_sample_wer,
        "sentence_errors": sentence_errors,
        "ser": sentence_errors / samples,
        "exact_sentence_accuracy": 1.0 - (sentence_errors / samples),
        "substitution_rate": substitutions / reference_tokens,
        "deletion_rate": deletions / reference_tokens,
        "insertion_rate": insertions / reference_tokens,
        "blank_outputs": blank_outputs,
        "blank_output_rate": blank_outputs / samples,
        "hypothesis_reference_length_ratio": hypothesis_tokens / reference_tokens,
    }


def _validate_prediction(row: Mapping[str, object], source: Path) -> None:
    required = {
        "sample_id",
        "subject",
        "sentence_id",
        "reference_token_ids",
        "predicted_token_ids",
        "substitutions",
        "deletions",
        "insertions",
        "edit_distance",
        "exact_match",
    }
    missing = required.difference(row)
    if missing:
        raise ValueError(f"{source} is missing fields: {sorted(missing)}")
    reference = row["reference_token_ids"]
    hypothesis = row["predicted_token_ids"]
    if not isinstance(reference, list) or not reference:
        raise ValueError(f"Invalid reference token list in {source}.")
    if not isinstance(hypothesis, list):
        raise ValueError(f"Invalid predicted token list in {source}.")
    components = (
        int(row["substitutions"])
        + int(row["deletions"])
        + int(row["insertions"])
    )
    if components != int(row["edit_distance"]):
        raise ValueError(f"Edit components disagree in {source}.")
    if bool(row["exact_match"]) != (int(row["edit_distance"]) == 0):
        raise ValueError(f"exact_match disagrees in {source}.")


def load_predictions(evidence_root: str | Path) -> tuple[list[dict], dict[str, str]]:
    root = Path(evidence_root)
    files = sorted(root.rglob("predictions.jsonl"))
    if len(files) != 25:
        raise RuntimeError(f"Expected 25 prediction files; found {len(files)}.")

    all_rows: list[dict] = []
    hashes: dict[str, str] = {}
    found_pairs: set[tuple[str, int]] = set()
    paired_references: dict[tuple[int, str], tuple[object, ...]] = {}
    sample_fold: dict[str, int] = {}
    per_model_samples: dict[str, set[str]] = defaultdict(set)

    for path in files:
        model_id, fold = prediction_identity(path)
        pair = (model_id, fold)
        if pair in found_pairs:
            raise RuntimeError(f"Duplicate prediction file for {model_id}, fold {fold}.")
        found_pairs.add(pair)
        rows = read_jsonl(path)
        if len(rows) != EXPECTED_SAMPLES_PER_FOLD:
            raise RuntimeError(
                f"{model_id} fold {fold} has {len(rows)} predictions, "
                f"expected {EXPECTED_SAMPLES_PER_FOLD}."
            )
        identifiers = [str(row.get("sample_id")) for row in rows]
        if len(set(identifiers)) != len(identifiers):
            raise RuntimeError(f"Duplicate sample IDs in {path}.")

        hashes[path.relative_to(root).as_posix()] = sha256_file(path)
        for original in rows:
            _validate_prediction(original, path)
            sample_id = str(original["sample_id"])
            reference_signature = (
                tuple(int(value) for value in original["reference_token_ids"]),
                str(original["subject"]),
                int(original["sentence_id"]),
            )
            pair_key = (fold, sample_id)
            previous = paired_references.setdefault(pair_key, reference_signature)
            if previous != reference_signature:
                raise RuntimeError(
                    f"Models disagree on the reference for fold {fold}, {sample_id}."
                )
            previous_fold = sample_fold.setdefault(sample_id, fold)
            if previous_fold != fold:
                raise RuntimeError(f"Sample {sample_id} occurs in multiple outer folds.")
            if sample_id in per_model_samples[model_id]:
                raise RuntimeError(f"Sample {sample_id} repeats for model {model_id}.")
            per_model_samples[model_id].add(sample_id)
            enriched = dict(original)
            enriched["model_id"] = model_id
            enriched["outer_fold"] = fold
            all_rows.append(enriched)

    if found_pairs != EXPECTED_PAIRS:
        missing = sorted(EXPECTED_PAIRS - found_pairs)
        unexpected = sorted(found_pairs - EXPECTED_PAIRS)
        raise RuntimeError(f"Prediction matrix mismatch; missing={missing}, extra={unexpected}.")
    for model_id in MODEL_ORDER:
        count = len(per_model_samples[model_id])
        if count != EXPECTED_SAMPLES_PER_MODEL:
            raise RuntimeError(f"{model_id} has {count} unique samples, expected 7400.")
    if len(paired_references) != EXPECTED_SAMPLES_PER_MODEL:
        raise RuntimeError("Paired sample/reference count is not 7400.")

    return all_rows, hashes


def group_rows(
    records: Iterable[dict],
    keys: Sequence[str],
) -> dict[tuple[object, ...], list[dict]]:
    groups: dict[tuple[object, ...], list[dict]] = defaultdict(list)
    for row in records:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups


def _close(actual: object, expected: object, tolerance: float = 1e-12) -> bool:
    return abs(float(actual) - float(expected)) <= tolerance


def validate_exported_metrics(
    fold_rows: Sequence[Mapping[str, object]],
    model_rows: Sequence[Mapping[str, object]],
    exported_fold_rows: Sequence[Mapping[str, str]],
    exported_model_rows: Sequence[Mapping[str, str]],
) -> None:
    exported_folds = {
        (row["model_id"].lower(), int(row["outer_fold"])): row
        for row in exported_fold_rows
    }
    if set(exported_folds) != EXPECTED_PAIRS:
        raise RuntimeError("Exported fold-results matrix is not complete.")
    numeric_fold_fields = (
        "samples",
        "reference_tokens",
        "hypothesis_tokens",
        "matches",
        "substitutions",
        "deletions",
        "insertions",
        "total_edits",
        "pooled_wer",
        "macro_sample_wer",
        "ser",
        "exact_sentence_accuracy",
    )
    for computed in fold_rows:
        key = (str(computed["model_id"]), int(computed["outer_fold"]))
        exported = exported_folds[key]
        for field in numeric_fold_fields:
            if not _close(computed[field], exported[field]):
                raise RuntimeError(f"Fold metric mismatch for {key}, {field}.")

    exported_models = {
        row["model_id"].lower(): row
        for row in exported_model_rows
    }
    if set(exported_models) != set(MODEL_ORDER):
        raise RuntimeError("Exported model-summary matrix is not complete.")
    for computed in model_rows:
        model_id = str(computed["model_id"])
        exported = exported_models[model_id]
        checks = {
            "samples": "total_samples",
            "reference_tokens": "total_reference_tokens",
            "total_edits": "total_edits",
            "pooled_wer": "cross_fold_pooled_wer",
            "mean_fold_wer": "mean_fold_wer",
            "sd_fold_wer": "sd_fold_wer",
            "exact_sentence_accuracy": "mean_exact_sentence_accuracy",
            "sd_exact_sentence_accuracy": "sd_exact_sentence_accuracy",
            "ser": "mean_ser",
            "sd_ser": "sd_ser",
        }
        for computed_field, exported_field in checks.items():
            if not _close(computed[computed_field], exported[exported_field]):
                raise RuntimeError(
                    f"Model summary mismatch for {model_id}, {computed_field}."
                )


def _ordered_metric_row(prefix: Mapping[str, object], metrics: Mapping[str, object]) -> dict:
    return {**prefix, **metrics}


def calculate_tables(records: Sequence[dict]) -> dict[str, list[dict]]:
    fold_groups = group_rows(records, ("model_id", "outer_fold"))
    fold_rows: list[dict] = []
    for fold in range(1, 6):
        for model_id in MODEL_ORDER:
            metrics = aggregate_records(fold_groups[(model_id, fold)])
            fold_rows.append(
                _ordered_metric_row(
                    {"model_id": model_id, "outer_fold": fold},
                    metrics,
                )
            )

    model_groups = group_rows(records, ("model_id",))
    model_rows: list[dict] = []
    for model_id in MODEL_ORDER:
        metrics = aggregate_records(model_groups[(model_id,)])
        if int(metrics["samples"]) != EXPECTED_SAMPLES_PER_MODEL:
            raise RuntimeError(f"Unexpected sample total for {model_id}.")
        if int(metrics["reference_tokens"]) != EXPECTED_REFERENCE_TOKENS_PER_MODEL:
            raise RuntimeError(f"Unexpected reference-token total for {model_id}.")
        model_folds = [row for row in fold_rows if row["model_id"] == model_id]
        metrics["mean_fold_wer"] = statistics.fmean(
            float(row["pooled_wer"]) for row in model_folds
        )
        metrics["sd_fold_wer"] = statistics.stdev(
            float(row["pooled_wer"]) for row in model_folds
        )
        metrics["sd_exact_sentence_accuracy"] = statistics.stdev(
            float(row["exact_sentence_accuracy"]) for row in model_folds
        )
        metrics["sd_ser"] = statistics.stdev(float(row["ser"]) for row in model_folds)
        model_rows.append(
            _ordered_metric_row(
                {
                    "model_id": model_id,
                    "parameter_count": EXPECTED_PARAMETERS[model_id],
                },
                metrics,
            )
        )

    ranking = sorted(model_rows, key=lambda row: (float(row["pooled_wer"]), MODEL_ORDER.index(str(row["model_id"]))))
    for rank, row in enumerate(ranking, start=1):
        row["wer_rank"] = rank
    model_rows = sorted(model_rows, key=lambda row: MODEL_ORDER.index(str(row["model_id"])))

    subject_groups = group_rows(records, ("model_id", "subject"))
    subjects = sorted({str(row["subject"]) for row in records})
    subject_rows: list[dict] = []
    for model_id in MODEL_ORDER:
        for subject in subjects:
            subject_rows.append(
                _ordered_metric_row(
                    {"model_id": model_id, "subject": subject},
                    aggregate_records(subject_groups[(model_id, subject)]),
                )
            )

    length_records: list[dict] = []
    for row in records:
        enriched = dict(row)
        enriched["reference_length_bin"] = reference_length_bin(
            len(row["reference_token_ids"])
        )
        length_records.append(enriched)
    length_groups = group_rows(length_records, ("model_id", "reference_length_bin"))
    bin_order = ("1", "2", "3", "4", "5+")
    length_rows: list[dict] = []
    for model_id in MODEL_ORDER:
        for length_bin in bin_order:
            group = length_groups.get((model_id, length_bin), [])
            if not group:
                continue
            length_rows.append(
                _ordered_metric_row(
                    {"model_id": model_id, "reference_length_bin": length_bin},
                    aggregate_records(group),
                )
            )

    return {
        "fold": fold_rows,
        "model": model_rows,
        "subject": subject_rows,
        "length": length_rows,
    }


def markdown_report(model_rows: Sequence[Mapping[str, object]]) -> str:
    ranking = sorted(model_rows, key=lambda row: int(row["wer_rank"]))
    lines = [
        "# Stage 34B — Five-Model Recognition Evaluation",
        "",
        "## Scope",
        "",
        "This report recomputes recognition results from 37,000 paired outer-fold",
        "predictions: 7,400 samples for each of five models. No training is run.",
        "",
        "## Primary ranking",
        "",
        "| Rank | Model | Parameters | Pooled WER | Exact sentence accuracy | SER | Deletion rate | Blank-output rate |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranking:
        lines.append(
            "| {rank} | {model} | {parameters:,} | {wer:.6f} | {accuracy:.6f} | "
            "{ser:.6f} | {deletion:.6f} | {blank:.6f} |".format(
                rank=int(row["wer_rank"]),
                model=str(row["model_id"]).upper(),
                parameters=int(row["parameter_count"]),
                wer=float(row["pooled_wer"]),
                accuracy=float(row["exact_sentence_accuracy"]),
                ser=float(row["ser"]),
                deletion=float(row["deletion_rate"]),
                blank=float(row["blank_output_rate"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Pooled WER is the primary recognition metric. Exact sentence accuracy",
            "is secondary because a sentence is counted wrong after any token error.",
            "Parameter count describes compactness but does not by itself establish",
            "runtime efficiency. Statistical comparisons follow in Stage 34C and",
            "runtime/memory benchmarking follows in Stage 34D.",
            "",
            "## Additional outputs",
            "",
            "- Per-fold error metrics and stability inputs",
            "- Per-subject performance",
            "- Performance by reference sequence length",
            "- Substitution, deletion and insertion rates",
            "- Blank-output and output-length diagnostics",
            "",
        ]
    )
    return "\n".join(lines)


def build_stage34b(project_root: str | Path) -> dict:
    root = Path(project_root).resolve()
    import_root = root / "imports" / "stage32_results"
    archive = import_root / "stage32c_results_without_checkpoints.zip"
    evidence_root = import_root / "extracted"
    if not archive.is_file():
        raise FileNotFoundError(archive)
    archive_hash = sha256_file(archive)
    if archive_hash != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("The Stage-32 evidence archive hash changed.")
    if not evidence_root.is_dir():
        raise FileNotFoundError(evidence_root)

    fold_source = find_unique(evidence_root, "stage32c_fold_results.csv")
    model_source = find_unique(evidence_root, "stage32c_model_summary.csv")
    records, prediction_hashes = load_predictions(evidence_root)
    tables = calculate_tables(records)
    validate_exported_metrics(
        tables["fold"],
        tables["model"],
        read_csv(fold_source),
        read_csv(model_source),
    )

    output_root = root / "artifacts" / "analysis" / "stage34"
    output_root.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "model_metrics": output_root / "stage34b_model_metrics.csv",
        "fold_metrics": output_root / "stage34b_fold_metrics.csv",
        "subject_metrics": output_root / "stage34b_subject_metrics.csv",
        "length_metrics": output_root / "stage34b_length_metrics.csv",
    }
    write_csv(output_paths["model_metrics"], tables["model"])
    write_csv(output_paths["fold_metrics"], tables["fold"])
    write_csv(output_paths["subject_metrics"], tables["subject"])
    write_csv(output_paths["length_metrics"], tables["length"])

    document_path = root / "docs" / "STAGE_34B_MODEL_EVALUATION.md"
    document_path.parent.mkdir(parents=True, exist_ok=True)
    with document_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(markdown_report(tables["model"]))

    generated_hashes = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in (*output_paths.values(), document_path)
    }
    ranking = [
        str(row["model_id"])
        for row in sorted(tables["model"], key=lambda row: int(row["wer_rank"]))
    ]
    contract = {
        "project_generation": "fresh_start_2026_v1",
        "stage": 34,
        "substage": "B",
        "status": "passed",
        "training_started": False,
        "input": {
            "archive_sha256": archive_hash,
            "prediction_files": 25,
            "prediction_rows": len(records),
            "paired_samples": 7400,
            "models": list(MODEL_ORDER),
            "folds": 5,
            "prediction_sha256": prediction_hashes,
            "fold_summary_sha256": sha256_file(fold_source),
            "model_summary_sha256": sha256_file(model_source),
        },
        "validation": {
            "paired_references_match": True,
            "exported_fold_metrics_match": True,
            "exported_model_metrics_match": True,
            "wer_ranking": ranking,
            "primary_metric": "cross_fold_pooled_wer",
        },
        "generated_sha256": generated_hashes,
    }
    contract_path = output_root / "stage34b_contract.json"
    contract_hash_path = output_root / "stage34b_contract.sha256"
    write_json(contract_path, contract)
    with contract_hash_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(sha256_file(contract_path) + "\n")

    return {
        "contract": contract,
        "contract_path": contract_path,
        "contract_hash_path": contract_hash_path,
        "document_path": document_path,
        "tables": tables,
    }
