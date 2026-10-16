from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from src.training.known_signer_plan import (
    OUTER_FOLDS,
    normalize_stage13_rows,
    validate_outer_fold_assignment,
    validate_registry_plan,
    validate_run_matrix,
    validate_stage13_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STAGE31_TAG = "stage31-freeze-v1"
STAGE32_RUNNER_TAG = "stage32-known-runner-v1"
EXPECTED_STAGE13_HASH = (
    "7b9943145ffd9658fe20fedc37e4bd451"
    "5496a7cd59b893f95e99f95ea6c95e9"
)

STAGE13_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "splits"
    / "stage13_known_signer_5fold.csv"
)
STAGE13_HASH_PATH = STAGE13_PATH.with_suffix(".sha256")
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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def main() -> None:
    print("=" * 78)
    print("STAGE 32A — KNOWN-SIGNER PLAN VERIFIER")
    print("=" * 78)

    required_paths = (
        PROJECT_ROOT / "src" / "training" / "known_signer_plan.py",
        PROJECT_ROOT / "scripts" / "build_stage32_known_plan.py",
        PROJECT_ROOT / "scripts" / "verify_stage32_known_plan.py",
        PROJECT_ROOT / "tests" / "test_known_signer_plan.py",
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
    )
    for path in required_paths:
        require(path.is_file(), f"Missing Stage-32A file: {path}.")

    tag_type = git("cat-file", "-t", f"refs/tags/{STAGE31_TAG}").stdout.strip()
    require(tag_type == "tag", "Stage-31 freeze tag is not annotated.")
    stage31_commit = git("rev-list", "-n", "1", STAGE31_TAG).stdout.strip()
    head_commit = git("rev-parse", "HEAD").stdout.strip()
    ancestor = git(
        "merge-base",
        "--is-ancestor",
        stage31_commit,
        head_commit,
        check=False,
    ).returncode == 0
    require(ancestor, "Stage-31 tag is not an ancestor of HEAD.")

    actual_stage13_hash = sha256_file(STAGE13_PATH)
    require(read_hash(STAGE13_HASH_PATH) == actual_stage13_hash, "Stage-13 hash mismatch.")
    require(actual_stage13_hash == EXPECTED_STAGE13_HASH, "Stage-13 artifact changed.")

    actual_assignment_hash = sha256_file(ASSIGNMENT_PATH)
    actual_plan_hash = sha256_file(RUN_PLAN_PATH)
    actual_registry_hash = sha256_file(REGISTRY_PATH)
    actual_contract_hash = sha256_file(CONTRACT_PATH)
    require(
        read_hash(ASSIGNMENT_HASH_PATH) == actual_assignment_hash,
        "Stage-32 inner-assignment hash mismatch.",
    )
    require(read_hash(RUN_PLAN_HASH_PATH) == actual_plan_hash, "Run-plan hash mismatch.")
    require(
        read_hash(REGISTRY_HASH_PATH) == actual_registry_hash,
        "Initial run-registry hash mismatch.",
    )
    require(
        read_hash(CONTRACT_HASH_PATH) == actual_contract_hash,
        "Stage-32A contract hash mismatch.",
    )

    with STAGE13_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
        stage13_source = list(csv.DictReader(handle))
    stage13_summary = validate_stage13_rows(stage13_source)
    normalized_stage13 = normalize_stage13_rows(stage13_source)
    expected_sample_ids = {row["sample_id"] for row in normalized_stage13}

    with ASSIGNMENT_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
        assignments = list(csv.DictReader(handle))
    require(len(assignments) == 37000, "Assignment row count must be 37,000.")

    per_sample = Counter(row["sample_id"] for row in assignments)
    require(
        set(per_sample) == expected_sample_ids and set(per_sample.values()) == {5},
        "Every sample must occur once in every Stage-32 fold plan.",
    )
    outer_test_frequency = Counter(
        row["sample_id"]
        for row in assignments
        if row["partition"] == "outer_test"
    )
    require(
        set(outer_test_frequency) == expected_sample_ids
        and set(outer_test_frequency.values()) == {1},
        "Every sample must be outer test exactly once.",
    )

    contract = read_json(CONTRACT_PATH)
    require(contract["stage"] == 32, "Contract stage changed.")
    require(contract["substage"] == "A_plan_freeze", "Contract substage changed.")
    require(contract["stage32a_passed"] is True, "Stage 32A is not passed.")
    require(contract["stage31"]["tag"] == STAGE31_TAG, "Stage-31 tag changed.")
    require(contract["stage31"]["commit"] == stage31_commit, "Stage-31 commit changed.")
    require(
        contract["inner_splits"]["assignment_sha256"] == actual_assignment_hash,
        "Contract assignment hash mismatch.",
    )
    require(contract["hashes"]["run_plan"] == actual_plan_hash, "Contract plan hash mismatch.")

    observed_fold_summaries = []
    for outer_fold in OUTER_FOLDS:
        fold_rows = [
            row
            for row in assignments
            if int(row["stage32_outer_fold"]) == outer_fold
        ]
        frozen_summary = next(
            item
            for item in contract["inner_splits"]["folds"]
            if int(item["outer_fold"]) == outer_fold
        )
        summary = validate_outer_fold_assignment(
            fold_rows,
            expected_sample_ids=expected_sample_ids,
            target_difference_limit=int(frozen_summary["maximum_atomic_group_size"]),
            target_validation_recordings=int(
                frozen_summary["target_validation_recordings"]
            ),
        )
        require(summary["outer_test_recordings"] == 1480, "Outer-test size changed.")
        observed_fold_summaries.append(summary)

    with SUMMARY_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
        summary_rows = list(csv.DictReader(handle))
    require(len(summary_rows) == 5, "Inner summary must have five rows.")

    run_plan = read_json(RUN_PLAN_PATH)
    require(run_plan["stage"] == 32, "Run-plan stage changed.")
    require(run_plan["substage"] == "A_plan_freeze", "Run-plan substage changed.")
    require(
        run_plan["freeze_boundary"]["required_runner_tag_before_scoring"]
        == STAGE32_RUNNER_TAG,
        "Required runner tag changed.",
    )
    matrix_summary = validate_run_matrix(run_plan["evaluations"])
    registry_summary = validate_registry_plan(run_plan["registry_rows"])
    with REGISTRY_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
        registry_csv_rows = list(csv.DictReader(handle))
    validate_registry_plan(registry_csv_rows)
    require(
        [row["experiment_id"] for row in registry_csv_rows]
        == [row["experiment_id"] for row in run_plan["registry_rows"]],
        "Run-registry CSV differs from the frozen JSON plan.",
    )
    require(
        contract["hashes"]["initial_run_registry"] == actual_registry_hash,
        "Contract registry hash mismatch.",
    )

    forbidden_names = {
        "best.pt",
        "last.pt",
        "predictions.jsonl",
        "metrics.json",
        "COMPLETE",
    }
    premature_artifacts = []
    for evaluation in run_plan["evaluations"]:
        run_directory = PROJECT_ROOT / evaluation["run_directory"]
        if not run_directory.exists():
            continue
        for path in run_directory.rglob("*"):
            if path.is_file() and path.name in forbidden_names:
                premature_artifacts.append(str(path.relative_to(PROJECT_ROOT)))
    require(not premature_artifacts, f"Premature scored artifacts: {premature_artifacts}.")

    leakage = contract["leakage_barrier"]
    require(leakage["training_started"] is False, "Training started during planning.")
    require(leakage["outer_scores_present"] is False, "Outer scores leaked.")
    require(leakage["checkpoints_present"] is False, "Checkpoints exist during planning.")
    require(
        leakage["outer_test_controls_training"] is False,
        "Outer test controls training.",
    )

    print()
    print("FREEZE BOUNDARY")
    print("-" * 78)
    print("Stage-31 tag:", STAGE31_TAG)
    print("Annotated tag:", True)
    print("Tag commit:", stage31_commit)
    print("Tag is ancestor of HEAD:", ancestor)

    print()
    print("SOURCE AND INNER SPLITS")
    print("-" * 78)
    print("Stage-13 recordings:", stage13_summary["recordings"])
    print("Stage-13 atomic groups:", stage13_summary["split_groups"])
    print("Stage-13 strata:", stage13_summary["subject_sentence_strata"])
    print("Stage-32 assignment rows:", len(assignments))
    print("Each sample appears in five plans:", True)
    print("Each sample is outer test once:", True)
    for summary in observed_fold_summaries:
        print(
            f"Fold {summary['outer_fold']}: "
            f"train={summary['inner_train_recordings']} "
            f"validation={summary['inner_validation_recordings']} "
            f"test={summary['outer_test_recordings']}"
        )
    print("Duplicate crossings:", 0)
    print("Outer assignment mismatches:", 0)

    print()
    print("RUN AND REGISTRY PLAN")
    print("-" * 78)
    print("Outer evaluations:", matrix_summary["runs"])
    print("Registry rows:", registry_summary["registry_rows"])
    print("Outer / inner / refit rows: 25 / 25 / 25")
    print("Fold-major order:", matrix_summary["fold_major_order"])
    print("All statuses PLANNED:", registry_summary["all_status_planned"])

    print()
    print("LEAKAGE BARRIER")
    print("-" * 78)
    print("Training started:", leakage["training_started"])
    print("Outer scores present:", leakage["outer_scores_present"])
    print("Checkpoints present:", leakage["checkpoints_present"])
    print("Premature artifacts:", len(premature_artifacts))
    print("Required runner tag before scoring:", STAGE32_RUNNER_TAG)

    print()
    print("HASHES")
    print("-" * 78)
    print("Stage-13 hash match:", True)
    print("Inner assignment hash match:", True)
    print("Run-plan hash match:", True)
    print("Initial run-registry hash match:", True)
    print("Contract hash match:", True)

    print()
    print("FINAL STATUS: STAGE 32A PLAN PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()
