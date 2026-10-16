"""Stage 37 language-correction protocol and leakage-firewall helpers.

This stage does not call a language model and does not alter recognition
predictions.  It freezes the systems, evaluation arms, metrics, fold-local
retrieval rules and claim boundaries that later language experiments must use.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


STAGE37_ROOT = Path("artifacts/language/stage37")
MODEL_ORDER = ("g1", "c1", "c2", "c3", "c4")


def portable_text_sha256(path: str | Path) -> str:
    raw = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def require_portable_recorded_hash(data_path: Path, hash_path: Path) -> str:
    if not data_path.is_file() or not hash_path.is_file():
        raise FileNotFoundError(f"Frozen input is missing: {data_path}")
    expected = hash_path.read_text(encoding="utf-8-sig").strip().split()[0].lower()
    raw = data_path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    candidates = {
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(lf).hexdigest(),
        hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest(),
    }
    if expected not in candidates:
        raise RuntimeError(f"Frozen hash mismatch: {data_path.name}")
    return expected


def verify_prior_contract(root: Path, stage: int, relative_root: Path) -> str:
    contract_path = root / relative_root / f"stage{stage}_contract.json"
    hash_path = root / relative_root / f"stage{stage}_contract.sha256"
    if not contract_path.is_file() or not hash_path.is_file():
        raise FileNotFoundError(f"Stage {stage} contract is required.")
    observed = portable_text_sha256(contract_path)
    recorded = hash_path.read_text(encoding="utf-8-sig").strip().split()[0]
    if recorded != observed:
        raise RuntimeError(f"Stage {stage} contract hash mismatch.")
    contract = read_json(contract_path)
    if int(contract.get("stage", -1)) != stage or contract.get("status") != "passed":
        raise RuntimeError(f"Stage {stage} is not frozen as passed.")
    if contract.get("training_started") is not False:
        raise RuntimeError(f"Stage {stage} unexpectedly claims training.")
    return observed


def validate_language_config(config: Mapping[str, Any]) -> None:
    protocol = config.get("protocol", {})
    if protocol.get("recognition_test_predictions_are_immutable_inputs") is not True:
        raise RuntimeError("Recognition predictions are not frozen as immutable.")
    if protocol.get("retrieval_and_prompt_library") != "training_partition_only":
        raise RuntimeError("Training-only retrieval firewall is missing.")
    if set(protocol.get("inputs", [])) != {"predicted_gloss", "oracle_gloss"}:
        raise RuntimeError("Predicted and oracle gloss arms must both be declared.")
    required = {
        "L0_raw_gloss",
        "L1_training_only_retrieval",
        "L2_constrained_generator",
        "L3_template_holdout",
    }
    if set(protocol.get("systems", [])) != required:
        raise RuntimeError("Language system set differs from the frozen design.")


def validate_targets(rows: Sequence[Mapping[str, str]], expected_count: int | None = None) -> None:
    if expected_count is not None and len(rows) != expected_count:
        raise RuntimeError(f"Expected {expected_count} sentence targets, found {len(rows)}.")
    ids: set[int] = set()
    for row in rows:
        sentence_id = int(row["sentence_id"])
        if sentence_id in ids:
            raise RuntimeError(f"Duplicate sentence target: {sentence_id}")
        ids.add(sentence_id)
        gloss = str(row["gloss_raw"]).strip()
        english = str(row["english_raw"]).strip()
        tokens = json.loads(row["gloss_tokens_json"])
        if not gloss or not english or not isinstance(tokens, list) or not tokens:
            raise RuntimeError(f"Invalid sentence target: {sentence_id}")
        if len(tokens) != int(row["gloss_token_count"]):
            raise RuntimeError(f"Gloss-token count mismatch: {sentence_id}")


def build_fold_plan(
    assignment_rows: Sequence[Mapping[str, str]], expected_folds: int = 5
) -> list[dict[str, Any]]:
    allowed = {"inner_train", "inner_validation", "outer_test"}
    by_fold: dict[int, list[Mapping[str, str]]] = defaultdict(list)
    sample_folds: dict[str, set[int]] = defaultdict(set)
    sample_test_count: Counter[str] = Counter()
    for row in assignment_rows:
        fold = int(row["stage32_outer_fold"])
        partition = row["partition"]
        sample_id = row["sample_id"]
        if partition not in allowed or not (1 <= fold <= expected_folds):
            raise RuntimeError("Invalid fold assignment row.")
        if fold in sample_folds[sample_id]:
            raise RuntimeError(f"Duplicate sample/fold row: {sample_id}/{fold}")
        sample_folds[sample_id].add(fold)
        sample_test_count[sample_id] += int(partition == "outer_test")
        by_fold[fold].append(row)
    if set(by_fold) != set(range(1, expected_folds + 1)):
        raise RuntimeError("The five-fold assignment is incomplete.")
    for sample_id in sample_folds:
        if len(sample_folds[sample_id]) != expected_folds or sample_test_count[sample_id] != 1:
            raise RuntimeError(f"Sample does not satisfy five-fold isolation: {sample_id}")

    result: list[dict[str, Any]] = []
    for fold in range(1, expected_folds + 1):
        rows = by_fold[fold]
        counts = Counter(row["partition"] for row in rows)
        template_sets = {
            partition: {
                int(row["sentence_id"])
                for row in rows
                if row["partition"] == partition
            }
            for partition in allowed
        }
        train_ids = {
            row["sample_id"] for row in rows if row["partition"] != "outer_test"
        }
        test_ids = {
            row["sample_id"] for row in rows if row["partition"] == "outer_test"
        }
        if train_ids & test_ids:
            raise RuntimeError(f"Fold {fold} has train/test sample leakage.")
        result.append(
            {
                "outer_fold": fold,
                "selection_retrieval_bank_samples": counts["inner_train"],
                "selection_validation_samples": counts["inner_validation"],
                "final_retrieval_bank_samples": counts["inner_train"]
                + counts["inner_validation"],
                "outer_evaluation_samples": counts["outer_test"],
                "selection_bank_templates": len(template_sets["inner_train"]),
                "final_bank_templates": len(
                    template_sets["inner_train"] | template_sets["inner_validation"]
                ),
                "outer_templates": len(template_sets["outer_test"]),
                "outer_samples_in_retrieval_bank": 0,
                "outer_references_visible_during_correction": False,
            }
        )
    return result


def word_edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        current = [i]
        for j, b in enumerate(right, start=1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b))
            )
        previous = current
    return previous[-1]


def deterministic_retrieval(
    query: Sequence[str], candidates: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    if not candidates:
        raise ValueError("Retrieval bank cannot be empty.")
    query_tokens = tuple(str(token).lower() for token in query)

    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        tokens = tuple(str(token).lower() for token in row["gloss_tokens"])
        distance = word_edit_distance(query_tokens, tokens)
        normalizer = max(len(query_tokens), len(tokens), 1)
        return (distance / normalizer, distance, int(row["sentence_id"]))

    return min(candidates, key=key)


def system_rows() -> list[dict[str, Any]]:
    return [
        {
            "system_id": "L0_raw_gloss",
            "role": "no_correction_baseline",
            "input": "predicted_gloss",
            "output": "normalized_gloss_text",
            "learned_component": "none",
            "outer_reference_access": False,
            "purpose": "measure recognizer output before language correction",
        },
        {
            "system_id": "L1_training_only_retrieval",
            "role": "deterministic_correction_baseline",
            "input": "predicted_gloss",
            "output": "english_from_nearest_training_gloss_template",
            "learned_component": "none",
            "outer_reference_access": False,
            "purpose": "test correction without a generative model",
        },
        {
            "system_id": "L2_constrained_generator",
            "role": "generative_correction_system",
            "input": "predicted_gloss_plus_training_only_examples",
            "output": "one_constrained_english_sentence",
            "learned_component": "frozen_external_language_model",
            "outer_reference_access": False,
            "purpose": "improve grammar while controlling unsupported content",
        },
        {
            "system_id": "L3_template_holdout",
            "role": "challenge_evaluation_protocol_not_separate_model",
            "input": "L1_or_L2_output_on_unseen_templates",
            "output": "same_system_output",
            "learned_component": "none_additional",
            "outer_reference_access": False,
            "purpose": "measure generalization beyond memorized sentence templates",
        },
    ]


def arm_rows() -> list[dict[str, Any]]:
    return [
        {
            "arm_id": "c2_predicted_gloss_primary",
            "gloss_source": "c2_frozen_outer_prediction",
            "role": "primary_lightweight_pipeline",
            "included_systems": "L0;L1;L2",
            "publishable_comparison": True,
        },
        {
            "arm_id": "g1_predicted_gloss_reference",
            "gloss_source": "g1_frozen_outer_prediction",
            "role": "recognition_quality_reference",
            "included_systems": "L0;L1;L2",
            "publishable_comparison": True,
        },
        {
            "arm_id": "c1_predicted_gloss_comparison",
            "gloss_source": "c1_frozen_outer_prediction",
            "role": "compact_recurrent_comparison",
            "included_systems": "L0;L1;L2",
            "publishable_comparison": True,
        },
        {
            "arm_id": "c3_predicted_gloss_comparison",
            "gloss_source": "c3_frozen_outer_prediction",
            "role": "compact_transformer_comparison",
            "included_systems": "L0;L1;L2",
            "publishable_comparison": True,
        },
        {
            "arm_id": "c4_predicted_gloss_comparison",
            "gloss_source": "c4_frozen_outer_prediction",
            "role": "tiny_swin_comparison",
            "included_systems": "L0;L1;L2",
            "publishable_comparison": True,
        },
        {
            "arm_id": "oracle_gloss_ceiling",
            "gloss_source": "ground_truth_gloss",
            "role": "language_component_ceiling_diagnostic",
            "included_systems": "L1;L2",
            "publishable_comparison": False,
        },
    ]


def metric_rows() -> list[dict[str, Any]]:
    return [
        {"metric": "english_word_error_rate", "direction": "lower", "primary": True, "reason": "word-level content errors"},
        {"metric": "exact_sentence_accuracy", "direction": "higher", "primary": True, "reason": "fully correct sentences"},
        {"metric": "corpus_bleu_4", "direction": "higher", "primary": False, "reason": "n-gram generation quality"},
        {"metric": "chrf_plus_plus", "direction": "higher", "primary": False, "reason": "character and word overlap robust to morphology"},
        {"metric": "content_token_f1", "direction": "higher", "primary": True, "reason": "preservation of reference content"},
        {"metric": "unsupported_content_rate", "direction": "lower", "primary": True, "reason": "hallucination safety proxy"},
        {"metric": "correction_gain_rate", "direction": "higher", "primary": False, "reason": "fraction of L0 errors corrected"},
        {"metric": "correction_harm_rate", "direction": "lower", "primary": True, "reason": "fraction of correct L0 outputs made wrong"},
    ]


def leakage_rows() -> list[dict[str, Any]]:
    return [
        {"check_id": "F1", "rule": "outer sample absent from retrieval and prompt library", "required": True},
        {"check_id": "F2", "rule": "outer English reference unavailable until scoring", "required": True},
        {"check_id": "F3", "rule": "recognition predictions remain immutable", "required": True},
        {"check_id": "F4", "rule": "generator identity and decoding parameters frozen before scoring", "required": True},
        {"check_id": "F5", "rule": "predicted-gloss and oracle-gloss results reported separately", "required": True},
        {"check_id": "F6", "rule": "L2 prompt examples drawn from the same fold training bank as L1", "required": True},
        {"check_id": "F7", "rule": "no candidate English list constructed from outer test labels", "required": True},
    ]


def prompt_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "asl_gloss_to_constrained_english",
        "input_fields": ["sample_id", "outer_fold", "source_gloss", "training_only_examples"],
        "instruction": (
            "Convert the ASL gloss tokens into one grammatical English sentence. "
            "Preserve supported meaning and do not invent people, places, numbers, "
            "objects or actions absent from the gloss or training-only examples."
        ),
        "output_schema": {
            "sentence": "string",
            "added_content_tokens": "array_of_strings",
            "constraint_passed": "boolean",
        },
        "decoding": {
            "temperature": 0.0,
            "n": 1,
            "retry_on_schema_failure": 1,
            "model_identifier": "must_be_recorded_and_frozen_before_stage39_execution",
        },
        "outer_reference_in_prompt": False,
        "human_post_editing": False,
    }


def render_report(folds: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Stage 37 — Language-Correction Protocol Freeze",
        "",
        "## Purpose",
        "",
        "Stage 37 freezes the evaluation design for converting recognized ASL gloss into grammatical English. It performs no recognizer training, language-model inference or test scoring.",
        "",
        "## Frozen model roles",
        "",
        "- C2 is the primary lightweight recognizer pipeline.",
        "- G1 is retained only as the recognition-quality reference.",
        "- C1, C3 and C4 remain full comparative arms for every language system.",
        "- Oracle gloss is a language-component ceiling and must never be mixed with predicted-gloss results.",
        "",
        "## Systems",
        "",
        "- L0: raw normalized recognized gloss (no correction).",
        "- L1: deterministic nearest-template retrieval using fold-training data only.",
        "- L2: constrained generation using the same training-only example bank.",
        "- L3: a template-holdout challenge protocol, not a fourth correction model.",
        "",
        "## Leakage firewall",
        "",
        "For model selection, L1/L2 may use inner-train examples and inner validation only for tuning. After all settings are frozen, the final bank may combine inner train and validation; outer-test samples and references remain inaccessible until scoring.",
        "",
        "## Five-fold plan",
        "",
        "| Fold | Selection bank | Validation | Final bank | Outer test |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in folds:
        lines.append(
            f"| {row['outer_fold']} | {row['selection_retrieval_bank_samples']} | "
            f"{row['selection_validation_samples']} | {row['final_retrieval_bank_samples']} | "
            f"{row['outer_evaluation_samples']} |"
        )
    lines.extend(
        [
            "",
            "## Judgement rule",
            "",
            "A correction system is useful only if it improves English WER and exact sentence accuracy without unacceptable unsupported-content or correction-harm rates. Fluency or accuracy alone is insufficient.",
            "",
            "## Next execution stages",
            "",
            "- Stage 38: assemble immutable predictions for all five models and execute L0/L1.",
            "- Stage 39: freeze the generator identity, execute L2 once and record every response.",
            "- Stage 40: paired comparison, confidence intervals, error analysis and thesis tables.",
            "",
        ]
    )
    return "\n".join(lines)
