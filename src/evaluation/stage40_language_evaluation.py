from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


MODEL_ORDER = ("g1", "c1", "c2", "c3", "c4")
SYSTEM_ORDER = ("L0", "L1", "L2")
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_SEED = 20260826
CONFIDENCE_LEVEL = 0.95

# A deliberately small, frozen function-word list. Pronouns are retained because
# they carry meaning in the sentence targets used by this project.
CONTENT_STOPWORDS = frozenset(
    {
        "a", "an", "the", "am", "is", "are", "was", "were", "be", "been",
        "being", "do", "does", "did", "of", "to", "for", "at", "in", "on",
        "with", "from", "by", "and", "or", "but", "as", "than", "that",
    }
)

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


@dataclass(frozen=True, order=True)
class SampleKey:
    model_id: str
    outer_fold: int
    sample_id: str


_TEXT_HASH_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".txt", ".yaml", ".yml"}


def sha256_file(path: str | Path) -> str:
    """Hash text canonically so Git line-ending conversion cannot break evidence."""
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


def canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def write_json(path: str | Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_json_bytes(payload))


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def write_csv_rows(
    path: str | Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError("fieldnames are required when rows are empty")
        fieldnames = list(rows[0])
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    return value


def _normalized_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def resolve_column(
    fieldnames: Sequence[str],
    aliases: Sequence[str],
    *,
    required: bool = True,
) -> str | None:
    lookup = {_normalized_header(name): name for name in fieldnames}
    for alias in aliases:
        match = lookup.get(_normalized_header(alias))
        if match is not None:
            return match
    if required:
        raise RuntimeError(
            "Required CSV column was not found. Expected one of "
            f"{list(aliases)}; available columns are {list(fieldnames)}."
        )
    return None


def semantic_column(
    fieldnames: Sequence[str],
    *,
    include_any: Sequence[str],
    include_all: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> str | None:
    for field in fieldnames:
        token = _normalized_header(field)
        if any(value in token for value in exclude):
            continue
        if include_all and not all(value in token for value in include_all):
            continue
        if any(value in token for value in include_any):
            return field
    return None


def normalize_model_id(value: object) -> str:
    model = str(value).strip().lower()
    if model not in MODEL_ORDER:
        raise RuntimeError(f"Unexpected model_id: {value!r}")
    return model


def normalize_system_id(value: object) -> str:
    token = _normalized_header(str(value))
    if token.startswith("l0") or token in {"raw", "raw_gloss", "uncorrected"}:
        return "L0"
    if token.startswith("l1") or "retrieval" in token:
        return "L1"
    if token.startswith("l2") or "generat" in token:
        return "L2"
    raise RuntimeError(f"Unexpected language system: {value!r}")


def normalized_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("’", "'").replace("‘", "'").replace("_", " ")
    return " ".join(_WORD_RE.findall(text.lower()))


def word_tokens(value: object) -> tuple[str, ...]:
    text = normalized_text(value)
    return tuple(text.split()) if text else ()


def content_tokens(value: object) -> tuple[str, ...]:
    return tuple(token for token in word_tokens(value) if token not in CONTENT_STOPWORDS)


def levenshtein_counts(
    reference: Sequence[str], hypothesis: Sequence[str]
) -> dict[str, int]:
    """Return deterministic match/substitution/deletion/insertion counts."""
    n, m = len(reference), len(hypothesis)
    distance = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        distance[i][0] = i
    for j in range(m + 1):
        distance[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                distance[i][j] = distance[i - 1][j - 1]
            else:
                distance[i][j] = 1 + min(
                    distance[i - 1][j - 1],
                    distance[i - 1][j],
                    distance[i][j - 1],
                )

    counts = {"matches": 0, "substitutions": 0, "deletions": 0, "insertions": 0}
    i, j = n, m
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and reference[i - 1] == hypothesis[j - 1]
            and distance[i][j] == distance[i - 1][j - 1]
        ):
            counts["matches"] += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and distance[i][j] == distance[i - 1][j - 1] + 1:
            counts["substitutions"] += 1
            i -= 1
            j -= 1
        elif i > 0 and distance[i][j] == distance[i - 1][j] + 1:
            counts["deletions"] += 1
            i -= 1
        elif j > 0 and distance[i][j] == distance[i][j - 1] + 1:
            counts["insertions"] += 1
            j -= 1
        else:
            raise RuntimeError("Levenshtein backtracking reached an invalid state")
    counts["total_edits"] = (
        counts["substitutions"] + counts["deletions"] + counts["insertions"]
    )
    return counts


def overlap_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> tuple[int, int, int]:
    ref_counter = Counter(reference)
    hyp_counter = Counter(hypothesis)
    true_positive = sum((ref_counter & hyp_counter).values())
    false_positive = sum(hyp_counter.values()) - true_positive
    false_negative = sum(ref_counter.values()) - true_positive
    return true_positive, false_positive, false_negative


def unsupported_count(source_gloss: object, hypothesis: object) -> tuple[int, int]:
    supported = set(content_tokens(source_gloss))
    predicted = content_tokens(hypothesis)
    return sum(token not in supported for token in predicted), len(predicted)


def score_sentence(reference: object, hypothesis: object, source_gloss: object) -> dict[str, object]:
    reference_words = word_tokens(reference)
    hypothesis_words = word_tokens(hypothesis)
    if not reference_words:
        raise RuntimeError("English reference is empty after normalization")
    counts = levenshtein_counts(reference_words, hypothesis_words)
    content_tp, content_fp, content_fn = overlap_counts(
        content_tokens(reference), content_tokens(hypothesis)
    )
    unsupported, predicted_content = unsupported_count(source_gloss, hypothesis)
    return {
        **counts,
        "reference_words": len(reference_words),
        "hypothesis_words": len(hypothesis_words),
        "exact_correct": int(reference_words == hypothesis_words),
        "sample_wer": counts["total_edits"] / len(reference_words),
        "content_true_positive": content_tp,
        "content_false_positive": content_fp,
        "content_false_negative": content_fn,
        "unsupported_content_tokens": unsupported,
        "predicted_content_tokens": predicted_content,
    }


def _value(row: Mapping[str, str], column: str | None, default: str = "") -> str:
    if column is None:
        return default
    return str(row.get(column, default) or default)


def load_stage38_outputs(path: str | Path) -> list[dict[str, object]]:
    rows = read_csv_rows(path)
    if not rows:
        raise RuntimeError("Stage 38 output is empty")
    fields = list(rows[0])
    model_col = resolve_column(fields, ("model_id", "recognizer", "model"))
    fold_col = resolve_column(fields, ("outer_fold", "fold"))
    sample_col = resolve_column(fields, ("sample_id", "sample"))
    system_col = resolve_column(
        fields, ("system_id", "language_system", "system", "correction_system"), required=False
    )
    reference_col = resolve_column(
        fields,
        (
            "reference_english", "english_reference", "target_english", "reference",
            "english_raw", "reference_text", "target_sentence", "english_target",
        ),
        required=False,
    )
    if reference_col is None:
        reference_col = semantic_column(
            fields,
            include_any=("reference", "target", "ground_truth"),
            exclude=("gloss", "token", "word_count"),
        )
    if reference_col is None:
        raise RuntimeError(f"No English-reference column was found in Stage 38: {fields}")

    gloss_col = resolve_column(
        fields,
        (
            "source_gloss", "predicted_gloss", "input_gloss", "gloss", "normalized_gloss",
            "recognized_gloss", "gloss_hypothesis", "prediction_gloss",
        ),
        required=False,
    )
    if gloss_col is None:
        gloss_col = semantic_column(
            fields,
            include_any=("gloss",),
            exclude=("reference", "target", "token_count", "request"),
        )
    if gloss_col is None:
        raise RuntimeError(f"No source-gloss column was found in Stage 38: {fields}")
    subject_col = resolve_column(fields, ("subject", "subject_id", "signer_id"), required=False)
    sentence_col = resolve_column(
        fields, ("sentence_id", "target_sentence_id", "sentence_key", "target_id"), required=False
    )

    output_aliases = (
        "output_english", "predicted_english", "prediction_english", "english_output",
        "system_output", "hypothesis_english", "hypothesis", "output_text", "sentence",
        "english_sentence", "predicted_sentence", "corrected_english", "candidate_english",
    )
    long_output_col = resolve_column(fields, output_aliases, required=False)
    if long_output_col is None:
        long_output_col = semantic_column(
            fields,
            include_any=("output", "hypothesis", "prediction", "predicted", "corrected"),
            exclude=("gloss", "reference", "target", "metric", "score"),
        )
    l0_col = resolve_column(
        fields,
        ("l0_output_english", "l0_english", "l0_output", "raw_gloss_english", "uncorrected_gloss"),
        required=False,
    )
    l1_col = resolve_column(
        fields,
        ("l1_output_english", "l1_english", "l1_output", "retrieval_english", "retrieved_english"),
        required=False,
    )

    if system_col is not None and long_output_col is None:
        raise RuntimeError("Stage 38 uses long format but no output-English column was found")
    if system_col is None and (l0_col is None or l1_col is None):
        raise RuntimeError("Stage 38 must contain either a system column or L0/L1 output columns")

    normalized: list[dict[str, object]] = []
    for row in rows:
        base = {
            "model_id": normalize_model_id(row[model_col]),
            "outer_fold": int(row[fold_col]),
            "sample_id": row[sample_col].strip(),
            "subject": _value(row, subject_col),
            "sentence_id": _value(row, sentence_col),
            "source_gloss": row[gloss_col],
            "reference_english": row[reference_col],
        }
        if not base["sample_id"]:
            raise RuntimeError("Stage 38 contains a blank sample_id")
        if system_col is not None:
            system_id = normalize_system_id(row[system_col])
            if system_id not in {"L0", "L1"}:
                raise RuntimeError("Stage 38 may contain only L0 and L1 before Stage 40")
            normalized.append({**base, "system_id": system_id, "output_english": row[long_output_col]})
        else:
            normalized.append({**base, "system_id": "L0", "output_english": row[l0_col]})
            normalized.append({**base, "system_id": "L1", "output_english": row[l1_col]})
    return normalized


def load_sample_metadata(path: str | Path) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(path)
    if not rows:
        return {}
    fields = list(rows[0])
    sample_col = resolve_column(fields, ("sample_id", "sample"))
    subject_col = resolve_column(fields, ("subject", "subject_id", "signer_id"))
    sentence_col = resolve_column(fields, ("sentence_id", "target_sentence_id", "target_id"))
    return {
        row[sample_col].strip(): {
            "subject": row[subject_col].strip(),
            "sentence_id": row[sentence_col].strip(),
        }
        for row in rows
    }


def load_request_map(path: str | Path) -> list[dict[str, object]]:
    rows = read_csv_rows(path)
    if not rows:
        raise RuntimeError("Stage 39A sample/request map is empty")
    fields = list(rows[0])
    request_col = resolve_column(fields, ("request_id", "l2_request_id", "generation_request_id"))
    model_col = resolve_column(fields, ("model_id", "recognizer", "model"))
    fold_col = resolve_column(fields, ("outer_fold", "fold"))
    sample_col = resolve_column(fields, ("sample_id", "sample"))
    return [
        {
            "request_id": row[request_col].strip(),
            "model_id": normalize_model_id(row[model_col]),
            "outer_fold": int(row[fold_col]),
            "sample_id": row[sample_col].strip(),
        }
        for row in rows
    ]


def _zip_member(names: Sequence[str], basename: str) -> str:
    matches = [name for name in names if Path(name).name == basename]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {basename} in V2 ZIP; found {len(matches)}")
    return matches[0]


def load_v2_result_archive(path: str | Path) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    archive_path = Path(path)
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        response_member = _zip_member(names, "stage39b_v2_responses.jsonl")
        hash_member = _zip_member(names, "stage39b_v2_responses.sha256")
        manifest_member = _zip_member(names, "stage39b_v2_execution_manifest.json")
        response_bytes = archive.read(response_member)
        recorded_hash = archive.read(hash_member).decode("utf-8-sig").strip().split()[0].lower()
        actual_hash = hashlib.sha256(response_bytes).hexdigest()
        if recorded_hash != actual_hash:
            raise RuntimeError("Stage 39B V2 response hash does not match the archived hash")
        manifest = json.loads(archive.read(manifest_member).decode("utf-8-sig"))
    if str(manifest.get("response_sha256", "")).lower() != actual_hash:
        raise RuntimeError("Stage 39B V2 manifest response hash does not match")
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError("Stage 39B V2 execution manifest is not COMPLETE")
    if manifest.get("outer_references_exposed") is not False:
        raise RuntimeError("Stage 39B V2 reference firewall is not confirmed")
    if manifest.get("training_started") is not False:
        raise RuntimeError("Stage 39B V2 unexpectedly reports training")

    responses: dict[str, dict[str, object]] = {}
    for line_number, raw_line in enumerate(response_bytes.decode("utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        request_id = str(row.get("request_id", "")).strip()
        if not request_id or request_id in responses:
            raise RuntimeError(f"Invalid or duplicate request_id at V2 line {line_number}")
        if row.get("status") != "COMPLETE":
            raise RuntimeError(f"Incomplete V2 response at line {line_number}")
        sentence = str(row.get("sentence", "")).strip()
        if not sentence:
            raise RuntimeError(f"Empty V2 sentence at line {line_number}")
        responses[request_id] = row

    if int(manifest.get("unique_responses", -1)) != len(responses):
        raise RuntimeError("Stage 39B V2 manifest response count does not match the file")
    evidence = {
        "archive_sha256": sha256_file(archive_path),
        "response_sha256": actual_hash,
        "response_count": len(responses),
        "manifest": manifest,
    }
    return responses, evidence


def assemble_and_score(
    stage38_rows: Sequence[Mapping[str, object]],
    request_rows: Sequence[Mapping[str, object]],
    responses: Mapping[str, Mapping[str, object]],
    metadata: Mapping[str, Mapping[str, str]],
) -> list[dict[str, object]]:
    base: dict[tuple[SampleKey, str], dict[str, object]] = {}
    for row in stage38_rows:
        key = SampleKey(str(row["model_id"]), int(row["outer_fold"]), str(row["sample_id"]))
        system = str(row["system_id"])
        compound = (key, system)
        if compound in base:
            raise RuntimeError(f"Duplicate Stage 38 row: {compound}")
        enriched = dict(row)
        meta = metadata.get(key.sample_id, {})
        enriched["subject"] = str(enriched.get("subject") or meta.get("subject", ""))
        enriched["sentence_id"] = str(enriched.get("sentence_id") or meta.get("sentence_id", ""))
        if not enriched["subject"] or not enriched["sentence_id"]:
            raise RuntimeError(f"Missing subject/sentence metadata for {key.sample_id}")
        base[compound] = enriched

    request_by_key: dict[SampleKey, Mapping[str, object]] = {}
    for row in request_rows:
        key = SampleKey(str(row["model_id"]), int(row["outer_fold"]), str(row["sample_id"]))
        if key in request_by_key:
            raise RuntimeError(f"Duplicate Stage 39A sample mapping: {key}")
        request_by_key[key] = row

    expected_keys = {key for key, system in base if system == "L0"}
    if expected_keys != {key for key, system in base if system == "L1"}:
        raise RuntimeError("Stage 38 L0 and L1 sample keys do not match")
    if expected_keys != set(request_by_key):
        missing = len(expected_keys - set(request_by_key))
        extra = len(set(request_by_key) - expected_keys)
        raise RuntimeError(f"Stage 39A mapping mismatch: missing={missing}, extra={extra}")

    used_requests: set[str] = set()
    scored: list[dict[str, object]] = []
    for key in sorted(expected_keys):
        l0 = base[(key, "L0")]
        l1 = base[(key, "L1")]
        for field in ("source_gloss", "reference_english", "subject", "sentence_id"):
            if normalized_text(l0[field]) != normalized_text(l1[field]):
                raise RuntimeError(f"L0/L1 {field} mismatch for {key}")
        request_id = str(request_by_key[key]["request_id"])
        response = responses.get(request_id)
        if response is None:
            raise RuntimeError(f"Missing V2 response for {request_id}")
        if normalize_model_id(response.get("model_id")) != key.model_id:
            raise RuntimeError(f"V2 model mismatch for {request_id}")
        if int(response.get("outer_fold")) != key.outer_fold:
            raise RuntimeError(f"V2 fold mismatch for {request_id}")
        if normalized_text(response.get("source_gloss")) != normalized_text(l0["source_gloss"]):
            raise RuntimeError(f"V2 source-gloss mismatch for {request_id}")
        used_requests.add(request_id)

        system_rows = [
            l0,
            l1,
            {**l0, "system_id": "L2", "output_english": str(response["sentence"]), "request_id": request_id},
        ]
        preliminary = [
            {**row, **score_sentence(row["reference_english"], row["output_english"], row["source_gloss"])}
            for row in system_rows
        ]
        l0_edits = int(preliminary[0]["total_edits"])
        for row in preliminary:
            system = str(row["system_id"])
            row["correction_harm"] = None if system == "L0" else int(int(row["total_edits"]) > l0_edits)
            row["correction_improvement"] = (
                None if system == "L0" else int(int(row["total_edits"]) < l0_edits)
            )
            row["correction_unchanged"] = (
                None if system == "L0" else int(int(row["total_edits"]) == l0_edits)
            )
            row["cluster_id"] = f"{row['subject']}|{row['sentence_id']}"
            scored.append(row)

    if used_requests != set(responses):
        raise RuntimeError(
            f"V2 response coverage mismatch: used={len(used_requests)}, available={len(responses)}"
        )
    return scored


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def aggregate_records(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise RuntimeError("Cannot aggregate an empty record set")
    sums = {
        name: sum(int(row[name]) for row in rows)
        for name in (
            "matches", "substitutions", "deletions", "insertions", "total_edits",
            "reference_words", "hypothesis_words", "exact_correct",
            "content_true_positive", "content_false_positive", "content_false_negative",
            "unsupported_content_tokens", "predicted_content_tokens",
        )
    }
    tp = sums["content_true_positive"]
    fp = sums["content_false_positive"]
    fn = sums["content_false_negative"]
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    harm_values = [int(row["correction_harm"]) for row in rows if row["correction_harm"] is not None]
    improvement_values = [
        int(row["correction_improvement"])
        for row in rows
        if row["correction_improvement"] is not None
    ]
    unchanged_values = [
        int(row["correction_unchanged"])
        for row in rows
        if row["correction_unchanged"] is not None
    ]
    return {
        "samples": len(rows),
        **sums,
        "pooled_wer": _safe_ratio(sums["total_edits"], sums["reference_words"]),
        "macro_sample_wer": sum(float(row["sample_wer"]) for row in rows) / len(rows),
        "exact_sentence_accuracy": _safe_ratio(sums["exact_correct"], len(rows)),
        "content_precision": precision,
        "content_recall": recall,
        "content_f1": f1,
        "unsupported_content_rate": _safe_ratio(
            sums["unsupported_content_tokens"], sums["predicted_content_tokens"]
        ),
        "correction_harm_rate": (
            sum(harm_values) / len(harm_values) if harm_values else None
        ),
        "correction_improvement_rate": (
            sum(improvement_values) / len(improvement_values) if improvement_values else None
        ),
        "correction_unchanged_rate": (
            sum(unchanged_values) / len(unchanged_values) if unchanged_values else None
        ),
    }


def _cluster_matrix(rows: Sequence[Mapping[str, object]]) -> np.ndarray:
    fields = (
        "total_edits", "reference_words", "exact_correct", "content_true_positive",
        "content_false_positive", "content_false_negative", "unsupported_content_tokens",
        "predicted_content_tokens", "correction_harm", "correction_improvement",
    )
    grouped: dict[str, list[float]] = defaultdict(lambda: [0.0] * (len(fields) + 1))
    for row in rows:
        values = grouped[str(row["cluster_id"])]
        for index, field in enumerate(fields):
            value = row[field]
            if value is not None:
                values[index] += float(value)
        values[-1] += 1.0
    return np.asarray([grouped[key] for key in sorted(grouped)], dtype=np.float64)


def bootstrap_metric_intervals(
    rows: Sequence[Mapping[str, object]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    matrix = _cluster_matrix(rows)
    cluster_count = matrix.shape[0]
    rng = np.random.default_rng(seed)
    outputs: dict[str, list[np.ndarray]] = defaultdict(list)
    remaining = replicates
    while remaining:
        size = min(500, remaining)
        indices = rng.integers(0, cluster_count, size=(size, cluster_count), dtype=np.int32)
        sampled = matrix[indices].sum(axis=1)
        edits, refs, correct = sampled[:, 0], sampled[:, 1], sampled[:, 2]
        tp, fp, fn = sampled[:, 3], sampled[:, 4], sampled[:, 5]
        unsupported, predicted_content = sampled[:, 6], sampled[:, 7]
        harm, improvement, samples = sampled[:, 8], sampled[:, 9], sampled[:, 10]
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) != 0,
        )
        outputs["pooled_wer"].append(edits / refs)
        outputs["exact_sentence_accuracy"].append(correct / samples)
        outputs["content_f1"].append(f1)
        outputs["unsupported_content_rate"].append(
            np.divide(
                unsupported,
                predicted_content,
                out=np.zeros_like(unsupported),
                where=predicted_content != 0,
            )
        )
        if any(row["correction_harm"] is not None for row in rows):
            outputs["correction_harm_rate"].append(harm / samples)
            outputs["correction_improvement_rate"].append(improvement / samples)
        remaining -= size
    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    return {
        metric: (
            float(np.quantile(np.concatenate(values), alpha)),
            float(np.quantile(np.concatenate(values), 1.0 - alpha)),
        )
        for metric, values in outputs.items()
    }


def exact_mcnemar_p(a_correct: Sequence[int], b_correct: Sequence[int]) -> tuple[int, int, float]:
    if len(a_correct) != len(b_correct):
        raise ValueError("McNemar inputs must be paired")
    a_only = sum(a == 1 and b == 0 for a, b in zip(a_correct, b_correct))
    b_only = sum(a == 0 and b == 1 for a, b in zip(a_correct, b_correct))
    discordant = a_only + b_only
    if discordant == 0:
        return a_only, b_only, 1.0
    lower = min(a_only, b_only)
    logs = [
        math.lgamma(discordant + 1)
        - math.lgamma(k + 1)
        - math.lgamma(discordant - k + 1)
        - discordant * math.log(2.0)
        for k in range(lower + 1)
    ]
    maximum = max(logs)
    tail = math.exp(maximum) * sum(math.exp(value - maximum) for value in logs)
    return a_only, b_only, min(1.0, 2.0 * tail)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _records_by_sample(rows: Sequence[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    output: dict[str, Mapping[str, object]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in output:
            raise RuntimeError(f"Duplicate sample in comparison: {sample_id}")
        output[sample_id] = row
    return output


def paired_wer_test(
    a_rows: Sequence[Mapping[str, object]],
    b_rows: Sequence[Mapping[str, object]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    a = _records_by_sample(a_rows)
    b = _records_by_sample(b_rows)
    if set(a) != set(b):
        raise RuntimeError("Pairwise comparison sample sets do not match")
    sample_ids = sorted(a)
    clusters = sorted({str(a[sample_id]["cluster_id"]) for sample_id in sample_ids})
    cluster_index = {cluster: index for index, cluster in enumerate(clusters)}
    values = np.zeros((len(clusters), 4), dtype=np.float64)
    a_correct: list[int] = []
    b_correct: list[int] = []
    for sample_id in sample_ids:
        ar, br = a[sample_id], b[sample_id]
        if str(ar["cluster_id"]) != str(br["cluster_id"]):
            raise RuntimeError("Pairwise cluster IDs do not match")
        index = cluster_index[str(ar["cluster_id"])]
        values[index, 0] += float(ar["total_edits"])
        values[index, 1] += float(ar["reference_words"])
        values[index, 2] += float(br["total_edits"])
        values[index, 3] += float(br["reference_words"])
        a_correct.append(int(ar["exact_correct"]))
        b_correct.append(int(br["exact_correct"]))
    observed = values[:, 0].sum() / values[:, 1].sum() - values[:, 2].sum() / values[:, 3].sum()
    rng = np.random.default_rng(seed)
    deltas: list[np.ndarray] = []
    remaining = replicates
    while remaining:
        size = min(500, remaining)
        indices = rng.integers(0, len(clusters), size=(size, len(clusters)), dtype=np.int32)
        sampled = values[indices].sum(axis=1)
        deltas.append(sampled[:, 0] / sampled[:, 1] - sampled[:, 2] / sampled[:, 3])
        remaining -= size
    delta_values = np.concatenate(deltas)
    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    lower = float(np.quantile(delta_values, alpha))
    upper = float(np.quantile(delta_values, 1.0 - alpha))
    left = (float(np.count_nonzero(delta_values <= 0)) + 1.0) / (replicates + 1.0)
    right = (float(np.count_nonzero(delta_values >= 0)) + 1.0) / (replicates + 1.0)
    bootstrap_p = min(1.0, 2.0 * min(left, right))
    a_only, b_only, mcnemar_p = exact_mcnemar_p(a_correct, b_correct)
    conclusion = "not_resolved"
    if upper < 0:
        conclusion = "a_lower_wer"
    elif lower > 0:
        conclusion = "b_lower_wer"
    return {
        "paired_samples": len(sample_ids),
        "clusters": len(clusters),
        "wer_delta_a_minus_b": observed,
        "wer_delta_ci_lower": lower,
        "wer_delta_ci_upper": upper,
        "wer_bootstrap_p_raw": bootstrap_p,
        "mcnemar_a_only_correct": a_only,
        "mcnemar_b_only_correct": b_only,
        "mcnemar_p_raw": mcnemar_p,
        "wer_conclusion": conclusion,
    }


def build_summaries_and_tests(
    scored_rows: Sequence[Mapping[str, object]],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in scored_rows:
        grouped[(str(row["model_id"]), str(row["system_id"]))].append(row)

    summary_rows: list[dict[str, object]] = []
    for model_index, model in enumerate(MODEL_ORDER):
        for system_index, system in enumerate(SYSTEM_ORDER):
            rows = grouped[(model, system)]
            aggregate = aggregate_records(rows)
            intervals = bootstrap_metric_intervals(
                rows,
                replicates=bootstrap_replicates,
                seed=seed + model_index * 101 + system_index * 17,
            )
            summary = {"model_id": model, "system_id": system, **aggregate}
            for metric, (lower, upper) in intervals.items():
                summary[f"{metric}_ci_lower"] = lower
                summary[f"{metric}_ci_upper"] = upper
            summary_rows.append(summary)

    system_tests: list[dict[str, object]] = []
    comparison_index = 0
    for model in MODEL_ORDER:
        for system_a, system_b in itertools.combinations(SYSTEM_ORDER, 2):
            result = paired_wer_test(
                grouped[(model, system_a)],
                grouped[(model, system_b)],
                replicates=bootstrap_replicates,
                seed=seed + 10_000 + comparison_index,
            )
            system_tests.append(
                {"model_id": model, "system_a": system_a, "system_b": system_b, **result}
            )
            comparison_index += 1
    _apply_holm_columns(system_tests)

    l2_tests: list[dict[str, object]] = []
    for comparison_index, (model_a, model_b) in enumerate(itertools.combinations(MODEL_ORDER, 2)):
        result = paired_wer_test(
            grouped[(model_a, "L2")],
            grouped[(model_b, "L2")],
            replicates=bootstrap_replicates,
            seed=seed + 20_000 + comparison_index,
        )
        l2_tests.append({"model_a": model_a, "model_b": model_b, **result})
    _apply_holm_columns(l2_tests)
    return summary_rows, system_tests, l2_tests


def _apply_holm_columns(rows: list[dict[str, object]]) -> None:
    wer_adjusted = holm_adjust([float(row["wer_bootstrap_p_raw"]) for row in rows])
    mcnemar_adjusted = holm_adjust([float(row["mcnemar_p_raw"]) for row in rows])
    for row, wer_p, mcnemar_p in zip(rows, wer_adjusted, mcnemar_adjusted):
        row["wer_bootstrap_p_holm"] = wer_p
        row["mcnemar_p_holm"] = mcnemar_p


def build_l2_ranking(summary_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    l2_rows = [dict(row) for row in summary_rows if row["system_id"] == "L2"]
    l2_rows.sort(
        key=lambda row: (
            float(row["pooled_wer"]),
            -float(row["content_f1"]),
            -float(row["exact_sentence_accuracy"]),
        )
    )
    ranking: list[dict[str, object]] = []
    for rank, row in enumerate(l2_rows, start=1):
        model = str(row["model_id"])
        role = "recognition_quality_reference" if model == "g1" else "comparator"
        if model == "c2":
            role = "primary_lightweight_recognizer"
        ranking.append(
            {
                "l2_quality_rank": rank,
                "model_id": model,
                "frozen_recognizer_role": role,
                "pooled_wer": row["pooled_wer"],
                "pooled_wer_ci_lower": row["pooled_wer_ci_lower"],
                "pooled_wer_ci_upper": row["pooled_wer_ci_upper"],
                "exact_sentence_accuracy": row["exact_sentence_accuracy"],
                "content_precision": row["content_precision"],
                "content_recall": row["content_recall"],
                "content_f1": row["content_f1"],
                "unsupported_content_rate": row["unsupported_content_rate"],
                "correction_harm_rate": row["correction_harm_rate"],
                "selection_scope": "L2_end_to_end_language_quality_only",
            }
        )
    return ranking


def compact_l2_rows(scored_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    fields = (
        "model_id", "outer_fold", "sample_id", "subject", "sentence_id", "cluster_id",
        "request_id", "source_gloss", "reference_english", "output_english", "matches",
        "substitutions", "deletions", "insertions", "total_edits", "reference_words",
        "hypothesis_words", "sample_wer", "exact_correct", "content_true_positive",
        "content_false_positive", "content_false_negative", "unsupported_content_tokens",
        "predicted_content_tokens", "correction_harm", "correction_improvement",
        "correction_unchanged",
    )
    return [
        {field: row.get(field) for field in fields}
        for row in scored_rows
        if row["system_id"] == "L2"
    ]


def relative_name(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def build_markdown_report(
    summary_rows: Sequence[Mapping[str, object]],
    ranking_rows: Sequence[Mapping[str, object]],
    system_tests: Sequence[Mapping[str, object]],
    l2_tests: Sequence[Mapping[str, object]],
    *,
    bootstrap_replicates: int,
) -> str:
    lines = [
        "# Stage 40: End-to-end language evaluation",
        "",
        "Stage 40 evaluates the same 7,400 paired outer-fold samples for each of the five recognizers. "
        "L0 is raw normalized gloss, L1 is deterministic training-only retrieval, and L2 is the frozen "
        "FLAN-T5 correction output. No recognizer training or language generation is performed here.",
        "",
        "## All-model results",
        "",
        "| Model | System | English WER | Exact accuracy | Content precision | Content recall | Content F1 | Unsupported content | Harm vs L0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        harm = "n.a." if row["correction_harm_rate"] is None else f"{float(row['correction_harm_rate']):.4%}"
        lines.append(
            f"| {str(row['model_id']).upper()} | {row['system_id']} | {float(row['pooled_wer']):.6f} | "
            f"{float(row['exact_sentence_accuracy']):.4%} | {float(row['content_precision']):.4f} | "
            f"{float(row['content_recall']):.4f} | {float(row['content_f1']):.4f} | "
            f"{float(row['unsupported_content_rate']):.4%} | {harm} |"
        )
    lines.extend(
        [
            "",
            "## L2 end-to-end quality ranking",
            "",
            "The rank uses pooled English WER as the primary criterion, content F1 as the first "
            "tie-breaker, and exact sentence accuracy as the second. It is a language-quality rank; "
            "the Stage 34 efficiency evidence remains necessary for a lightweight deployment decision.",
            "",
            "| Rank | Model | WER | Exact accuracy | Content F1 | Unsupported content | Harm vs L0 | Frozen role |",
            "|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in ranking_rows:
        lines.append(
            f"| {row['l2_quality_rank']} | {str(row['model_id']).upper()} | {float(row['pooled_wer']):.6f} | "
            f"{float(row['exact_sentence_accuracy']):.4%} | {float(row['content_f1']):.4f} | "
            f"{float(row['unsupported_content_rate']):.4%} | {float(row['correction_harm_rate']):.4%} | "
            f"{str(row['frozen_recognizer_role']).replace('_', ' ')} |"
        )
    unresolved_system = sum(row["wer_conclusion"] == "not_resolved" for row in system_tests)
    unresolved_l2 = sum(row["wer_conclusion"] == "not_resolved" for row in l2_tests)
    lines.extend(
        [
            "",
            "## Statistical method",
            "",
            f"- {bootstrap_replicates:,} paired cluster-bootstrap replicates with subject-sentence clusters.",
            "- Exact two-sided McNemar tests for paired sentence correctness.",
            "- Holm correction applied separately to the 15 within-model system comparisons and the 10 L2 model comparisons.",
            f"- WER differences unresolved at 95%: {unresolved_system} within-model system comparisons and {unresolved_l2} L2 model comparisons.",
            "",
            "## Metric definitions",
            "",
            "- English WER is total word substitutions, deletions, and insertions divided by reference words.",
            "- Exact accuracy requires equality after frozen lowercase, punctuation-stripping, underscore-splitting normalization.",
            "- Content precision, recall, and F1 use multiset token overlap after removing the frozen function-word list.",
            "- Unsupported-content rate is the proportion of predicted content tokens absent from the source-gloss content-token set.",
            "- Correction-harm rate is the proportion of samples whose system word-edit count is worse than L0 for the same sample.",
            "",
            "## Claim boundary",
            "",
            "The L2 ranking measures end-to-end English output quality under the known-signer five-fold protocol. "
            "It does not replace the recognition-only comparison or the efficiency measurements, and it does not establish template-holdout performance.",
            "",
        ]
    )
    return "\n".join(lines)
