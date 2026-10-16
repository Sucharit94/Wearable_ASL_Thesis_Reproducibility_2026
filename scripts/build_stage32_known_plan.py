from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.training.known_signer_plan import (
    MODEL_IDS,
    OUTER_FOLDS,
    PROJECT_SEED,
    build_outer_fold_assignment,
    build_registry_plan,
    build_run_matrix,
    normalize_stage13_rows,
    validate_registry_plan,
    validate_run_matrix,
    validate_stage13_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STAGE31_TAG = "stage31-freeze-v1"
STAGE32_RUNNER_TAG = "stage32-known-runner-v1"

STAGE13_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "splits"
    / "stage13_known_signer_5fold.csv"
)
STAGE13_HASH_PATH = STAGE13_PATH.with_suffix(".sha256")

STAGE17_INDEX_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "cache"
    / "stage17_raw_cache_index.csv"
)
STAGE17_INDEX_HASH_PATH = STAGE17_INDEX_PATH.with_suffix(".sha256")

LABEL_VOCAB_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "labels"
    / "stage10_gloss_vocabulary.json"
)
LABEL_VOCAB_HASH_PATH = LABEL_VOCAB_PATH.with_suffix(".sha256")

CONFIG_MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "config"
    / "stage31_config_hashes.json"
)
CONFIG_MANIFEST_HASH_PATH = CONFIG_MANIFEST_PATH.with_suffix(".sha256")

STAGE31_CONTRACT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "training"
    / "stage31_smoke_freeze_contract.json"
)
STAGE31_CONTRACT_HASH_PATH = STAGE31_CONTRACT_PATH.with_suffix(".sha256")

DATA_CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"
KNOWN_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "known_5fold.yaml"
)

ASSIGNMENT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "splits"
    / "stage32_known_inner_assignment.csv"
)
ASSIGNMENT_HASH_PATH = ASSIGNMENT_PATH.with_suffix(".sha256")
SUMMARY_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "splits"
    / "stage32_known_inner_summary.csv"
)
RUN_PLAN_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "experiments"
    / "stage32_known_run_plan.json"
)
RUN_PLAN_HASH_PATH = RUN_PLAN_PATH.with_suffix(".sha256")
REGISTRY_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "registry"
    / "run_registry.csv"
)
REGISTRY_HASH_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "registry"
    / "stage32_initial_run_registry.sha256"
)
CONTRACT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "experiments"
    / "stage32a_known_plan_contract.json"
)
CONTRACT_HASH_PATH = CONTRACT_PATH.with_suffix(".sha256")
DOC_PATH = PROJECT_ROOT / "docs" / "STAGE_32A_KNOWN_SIGNER_PLAN.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_hash(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").strip().split()[0].lower()


def write_hash(path: Path, digest: str, target_name: str) -> None:
    path.write_text(
        f"{digest}  {target_name}\n",
        encoding="utf-8",
        newline="\n",
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_hash_match(path: Path, hash_path: Path) -> str:
    require(path.is_file(), f"Missing frozen file: {path}.")
    require(hash_path.is_file(), f"Missing frozen hash: {hash_path}.")
    actual = sha256_file(path)
    recorded = read_hash(hash_path)
    require(recorded == actual, f"Frozen hash mismatch: {path}.")
    return actual


def git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


def manifest_hashes(manifest: dict) -> dict[str, str]:
    hashes = {}
    for entry in manifest["configs"]:
        relative_path = str(entry["path"]).replace("\\", "/")
        path = PROJECT_ROOT / relative_path
        actual = sha256_file(path)
        require(actual == entry["sha256"], f"Config changed: {relative_path}.")
        hashes[relative_path] = actual
    return hashes


def main() -> None:
    print("=" * 78)
    print("STAGE 32A - KNOWN-SIGNER SPLIT AND RUN-ORDER FREEZE")
    print("=" * 78)

    tag_type = git("cat-file", "-t", f"refs/tags/{STAGE31_TAG}")
    require(tag_type == "tag", f"{STAGE31_TAG} must be an annotated tag.")
    stage31_commit = git("rev-list", "-n", "1", STAGE31_TAG)
    head_commit = git("rev-parse", "HEAD")
    ancestor_check = subprocess.run(
        ["git", "merge-base", "--is-ancestor", stage31_commit, head_commit],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode
    require(ancestor_check == 0, "Stage-31 tag is not an ancestor of HEAD.")

    manifest_hash = require_hash_match(
        CONFIG_MANIFEST_PATH,
        CONFIG_MANIFEST_HASH_PATH,
    )
    stage31_contract_hash = require_hash_match(
        STAGE31_CONTRACT_PATH,
        STAGE31_CONTRACT_HASH_PATH,
    )
    stage13_hash = require_hash_match(STAGE13_PATH, STAGE13_HASH_PATH)
    data_manifest_hash = require_hash_match(
        STAGE17_INDEX_PATH,
        STAGE17_INDEX_HASH_PATH,
    )
    label_vocab_hash = require_hash_match(
        LABEL_VOCAB_PATH,
        LABEL_VOCAB_HASH_PATH,
    )

    stage31_contract = read_json(STAGE31_CONTRACT_PATH)
    require(stage31_contract["stage31_passed"] is True, "Stage 31 did not pass.")

    config_manifest = read_json(CONFIG_MANIFEST_PATH)
    require(config_manifest["stage"] == 31, "Wrong config-manifest stage.")
    require(config_manifest["config_count"] == 11, "Config count changed.")
    config_hashes = manifest_hashes(config_manifest)

    model_config_hashes = {
        model_id: config_hashes[f"configs/models/{model_id}.yaml"]
        for model_id in MODEL_IDS
    }
    data_config_hash = config_hashes["configs/data.yaml"]
    known_config_hash = config_hashes["configs/experiments/known_5fold.yaml"]
    require(data_config_hash == sha256_file(DATA_CONFIG_PATH), "Data config changed.")
    require(known_config_hash == sha256_file(KNOWN_CONFIG_PATH), "Known config changed.")

    with STAGE13_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
        source_rows = list(csv.DictReader(handle))

    stage13_summary = validate_stage13_rows(source_rows)
    require(stage13_summary["recordings"] == 7400, "Recording count changed.")
    require(stage13_summary["split_groups"] == 7397, "Atomic group count changed.")
    require(
        stage13_summary["subject_sentence_strata"] == 368,
        "Subject-sentence stratum count changed.",
    )
    normalized_rows = normalize_stage13_rows(source_rows)

    all_assignments = []
    fold_summaries = []
    for outer_fold in OUTER_FOLDS:
        assignments, fold_summary = build_outer_fold_assignment(
            normalized_rows,
            outer_fold,
            validation_fraction=0.10,
            seed=PROJECT_SEED,
        )
        all_assignments.extend(assignments)
        fold_summaries.append(fold_summary)

    require(len(all_assignments) == 37000, "Expected 37,000 assignment rows.")

    ASSIGNMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    assignment_fields = (
        "sample_id",
        "subject",
        "sentence_id",
        "stage12_split_group_id",
        "stage13_outer_fold",
        "stage32_outer_fold",
        "partition",
    )
    with ASSIGNMENT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=assignment_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_assignments)

    assignment_hash = sha256_file(ASSIGNMENT_PATH)
    write_hash(ASSIGNMENT_HASH_PATH, assignment_hash, ASSIGNMENT_PATH.name)

    summary_fields = (
        "outer_fold",
        "recordings",
        "inner_train_recordings",
        "inner_validation_recordings",
        "outer_test_recordings",
        "outer_training_pool_recordings",
        "target_validation_recordings",
        "observed_validation_fraction",
        "absolute_target_difference",
        "maximum_atomic_group_size",
        "duplicate_groups_crossing_partitions",
        "outer_test_assignment_mismatches",
        "subjects_missing_inner_partition",
        "strata_missing_inner_partition",
    )
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {key: summary[key] for key in summary_fields}
            for summary in fold_summaries
        )

    evaluations = build_run_matrix(
        model_config_hashes=model_config_hashes,
        data_config_sha256=data_config_hash,
        experiment_config_sha256=known_config_hash,
        stage13_split_sha256=stage13_hash,
        stage32_inner_split_sha256=assignment_hash,
        data_manifest_sha256=data_manifest_hash,
        label_vocab_sha256=label_vocab_hash,
    )
    matrix_summary = validate_run_matrix(evaluations)
    registry_rows = build_registry_plan(evaluations)
    registry_summary = validate_registry_plan(registry_rows)

    registry_fields = (
        "registry_order",
        "evaluation_run_order",
        "registry_id",
        "experiment_id",
        "parent_experiment_id",
        "project_generation",
        "model_id",
        "outer_fold",
        "split_id",
        "phase",
        "data_manifest_sha256",
        "label_vocab_sha256",
        "config_sha256",
        "code_commit",
        "seed",
        "normalization",
        "augmentation",
        "ctc_decoder",
        "hardware_required",
        "status",
        "artifact_path",
    )
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=registry_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(registry_rows)
    registry_hash = sha256_file(REGISTRY_PATH)
    write_hash(REGISTRY_HASH_PATH, registry_hash, REGISTRY_PATH.name)

    run_plan = {
        "project_generation": "fresh_start_2026_v1",
        "stage": 32,
        "substage": "A_plan_freeze",
        "protocol": {
            "name": "known_signer_5fold_nested_selection_refit",
            "outer_folds": 5,
            "model_order": list(MODEL_IDS),
            "matrix_order": "fold_major",
            "inner_validation_fraction": 0.10,
            "inner_split_seed_formula": "20260826 + outer_fold",
            "inner_split_unit": "stage12_split_group_id",
            "inner_stratification": ["subject", "sentence_id"],
            "selection_metric": "inner_validation_wer_minimum",
            "selection_tie_breaks": [
                "exact_sentence_accuracy_maximum",
                "validation_loss_minimum",
                "earliest_epoch",
            ],
            "reset_before_refit": True,
            "refit_epochs": "selected_epoch_E_star",
            "outer_test_evaluations_per_model_fold": 1,
            "test_controls_training": False,
            "normalization": "train_partition_feature_zscore",
            "augmentation": "none",
            "hardware_required": "NVIDIA_Tesla_T4",
        },
        "freeze_boundary": {
            "stage31_tag": STAGE31_TAG,
            "stage31_commit": stage31_commit,
            "stage31_tag_is_annotated": True,
            "stage31_tag_is_ancestor": True,
            "required_runner_tag_before_scoring": STAGE32_RUNNER_TAG,
            "runner_tag_created_in_substage_a": False,
        },
        "evaluations": evaluations,
        "registry_rows": registry_rows,
    }
    write_json(RUN_PLAN_PATH, run_plan)
    run_plan_hash = sha256_file(RUN_PLAN_PATH)
    write_hash(RUN_PLAN_HASH_PATH, run_plan_hash, RUN_PLAN_PATH.name)

    contract = {
        "project_generation": "fresh_start_2026_v1",
        "stage": 32,
        "substage": "A_plan_freeze",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage31": {
            "tag": STAGE31_TAG,
            "commit": stage31_commit,
            "head_when_built": head_commit,
            "config_manifest_sha256": manifest_hash,
            "smoke_contract_sha256": stage31_contract_hash,
            "passed": True,
        },
        "stage13": {
            **stage13_summary,
            "assignment_sha256": stage13_hash,
            "expected_assignment_sha256": (
                "7b9943145ffd9658fe20fedc37e4bd451"
                "5496a7cd59b893f95e99f95ea6c95e9"
            ),
        },
        "inner_splits": {
            "assignment_rows": len(all_assignments),
            "assignment_sha256": assignment_hash,
            "folds": fold_summaries,
            "all_outer_test_counts_1480": all(
                item["outer_test_recordings"] == 1480
                for item in fold_summaries
            ),
            "all_duplicate_crossings_zero": all(
                item["duplicate_groups_crossing_partitions"] == 0
                for item in fold_summaries
            ),
            "all_outer_assignment_mismatches_zero": all(
                item["outer_test_assignment_mismatches"] == 0
                for item in fold_summaries
            ),
            "all_strata_present_in_train_and_validation": all(
                item["strata_missing_inner_partition"] == 0
                for item in fold_summaries
            ),
        },
        "matrix": matrix_summary,
        "registry": registry_summary,
        "identity_sources": {
            "data_manifest_path": str(STAGE17_INDEX_PATH.relative_to(PROJECT_ROOT)),
            "data_manifest_sha256": data_manifest_hash,
            "label_vocab_path": str(LABEL_VOCAB_PATH.relative_to(PROJECT_ROOT)),
            "label_vocab_sha256": label_vocab_hash,
            "stage13_split_sha256": stage13_hash,
            "stage32_inner_split_sha256": assignment_hash,
        },
        "leakage_barrier": {
            "outer_scores_present": False,
            "training_started": False,
            "checkpoints_present": False,
            "outer_test_controls_training": False,
        },
        "outputs": {
            "assignment": str(ASSIGNMENT_PATH.relative_to(PROJECT_ROOT)),
            "summary": str(SUMMARY_PATH.relative_to(PROJECT_ROOT)),
            "run_plan": str(RUN_PLAN_PATH.relative_to(PROJECT_ROOT)),
            "run_registry": str(REGISTRY_PATH.relative_to(PROJECT_ROOT)),
        },
        "hashes": {
            "assignment": assignment_hash,
            "run_plan": run_plan_hash,
            "initial_run_registry": registry_hash,
        },
        "stage32a_passed": True,
    }
    require(
        contract["stage13"]["assignment_sha256"]
        == contract["stage13"]["expected_assignment_sha256"],
        "Stage-13 assignment differs from the frozen scientific artifact.",
    )
    require(contract["inner_splits"]["all_outer_test_counts_1480"], "Fold size changed.")
    require(contract["inner_splits"]["all_duplicate_crossings_zero"], "Leakage detected.")
    require(
        contract["inner_splits"]["all_strata_present_in_train_and_validation"],
        "Inner stratification coverage failed.",
    )

    write_json(CONTRACT_PATH, contract)
    contract_hash = sha256_file(CONTRACT_PATH)
    write_hash(CONTRACT_HASH_PATH, contract_hash, CONTRACT_PATH.name)

    fold_lines = "\n".join(
        "| {outer_fold} | {inner_train_recordings} | "
        "{inner_validation_recordings} | {outer_test_recordings} | "
        "{duplicate_groups_crossing_partitions} |".format(**item)
        for item in fold_summaries
    )
    markdown = f"""# Stage 32A - Known-signer plan freeze

Stage 32 will execute 25 outer evaluations: G1, C1, C2, C3 and C4 on
each of the five Stage-13 known-signer outer folds. Execution is
fold-major, so every model completes Fold 1 before Fold 2 starts.

## Leakage boundary

- Stage-31 frozen tag: `{STAGE31_TAG}`
- Stage-31 commit: `{stage31_commit}`
- Stage-13 assignment SHA-256: `{stage13_hash}`
- Inner split unit: `stage12_split_group_id`
- Inner stratification: `subject + sentence_id`
- Inner validation seed: `20260826 + outer_fold`
- Outer test controls training: no

## Frozen inner assignments

| Outer fold | Inner train | Inner validation | Outer test | Duplicate crossings |
|---:|---:|---:|---:|---:|
{fold_lines}

The assignment file has {len(all_assignments):,} rows: all 7,400 samples
represented once for each outer fold. For each fold, the Stage-13 test
partition is preserved exactly.

## Nested execution

Each of the 25 evaluations has three registry rows: the parent outer
evaluation, inner selection, and fresh full-data refit. This produces 75
planned registry rows. After epoch E* is selected using inner-validation
WER, the model is reset, normalization is recomputed on the full outer
training partition, and the model is refit for exactly E* epochs. The
outer test partition is evaluated once.

## Stage boundary

This substage creates no score and no checkpoint. The resumable runner
must pass CPU dry-run tests and be committed and tagged
`{STAGE32_RUNNER_TAG}` before the first GPU run or outer-test evaluation.
"""
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(markdown, encoding="utf-8", newline="\n")

    print()
    print("FREEZE BOUNDARY")
    print("-" * 78)
    print("Stage-31 tag:", STAGE31_TAG)
    print("Annotated tag:", True)
    print("Tag commit:", stage31_commit)
    print("Tag is ancestor:", True)
    print("Required runner tag before scoring:", STAGE32_RUNNER_TAG)

    print()
    print("STAGE-13 SOURCE")
    print("-" * 78)
    print("Recordings:", stage13_summary["recordings"])
    print("Atomic groups:", stage13_summary["split_groups"])
    print("Subject-sentence strata:", stage13_summary["subject_sentence_strata"])
    print("Assignment hash match:", True)

    print()
    print("INNER SPLIT SUMMARY")
    print("-" * 78)
    for item in fold_summaries:
        print(
            f"Fold {item['outer_fold']}: "
            f"train={item['inner_train_recordings']} "
            f"validation={item['inner_validation_recordings']} "
            f"test={item['outer_test_recordings']} "
            f"crossings={item['duplicate_groups_crossing_partitions']}"
        )

    print()
    print("RUN MATRIX")
    print("-" * 78)
    print("Outer evaluations:", matrix_summary["runs"])
    print("Registry rows:", registry_summary["registry_rows"])
    print("Order:", "fold-major")
    print("First evaluation:", evaluations[0]["experiment_id"])
    print("Last evaluation:", evaluations[-1]["experiment_id"])

    print()
    print("LEAKAGE BARRIER")
    print("-" * 78)
    print("Training started:", False)
    print("Outer scores present:", False)
    print("Checkpoints present:", False)
    print("Outer test controls training:", False)

    print()
    print("OUTPUTS")
    print("-" * 78)
    for path in (
        ASSIGNMENT_PATH,
        ASSIGNMENT_HASH_PATH,
        SUMMARY_PATH,
        RUN_PLAN_PATH,
        RUN_PLAN_HASH_PATH,
        REGISTRY_PATH,
        REGISTRY_HASH_PATH,
        CONTRACT_PATH,
        CONTRACT_HASH_PATH,
        DOC_PATH,
    ):
        print("Saved:", path.relative_to(PROJECT_ROOT))

    print()
    print("FINAL STATUS: STAGE 32A PLAN BUILD PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()
