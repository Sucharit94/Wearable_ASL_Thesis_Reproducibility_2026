from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.training.loso_plan import (
    HELD_OUT_SUBJECTS,
    LOSO_IDS,
    MODEL_IDS,
    PROJECT_SEED,
    REQUIRED_RUNNER_TAG,
    build_loso_assignment,
    build_registry_plan,
    build_run_matrix,
    validate_registry_plan,
    validate_run_matrix,
    validate_stage14_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE32_RESULTS_TAG = "stage32-known-results-v1"
STAGE32_RESULTS_COMMIT = "116116866d898ebe3e5fe2a15867191df3c01cf1"

STAGE14 = PROJECT_ROOT / "artifacts/splits/stage14_loso_assignment.csv"
STAGE15 = PROJECT_ROOT / "artifacts/splits/stage15_loso_support_assignment.csv"
STAGE17 = PROJECT_ROOT / "artifacts/cache/stage17_raw_cache_index.csv"
VOCAB = PROJECT_ROOT / "artifacts/labels/stage10_gloss_vocabulary.json"
CONFIG_MANIFEST = PROJECT_ROOT / "artifacts/config/stage31_config_hashes.json"
STAGE32_RESULTS = PROJECT_ROOT / "artifacts/results/stage32c_results_contract.json"
STAGE32_REGISTRY = PROJECT_ROOT / "artifacts/registry/stage32c_completed_run_registry.csv"
LOSO_CONFIG = PROJECT_ROOT / "configs/experiments/loso.yaml"

ASSIGNMENT = PROJECT_ROOT / "artifacts/splits/stage33_loso_inner_assignment.csv"
SUMMARY = PROJECT_ROOT / "artifacts/splits/stage33_loso_inner_summary.csv"
RUN_PLAN = PROJECT_ROOT / "artifacts/experiments/stage33_loso_run_plan.json"
REGISTRY = PROJECT_ROOT / "artifacts/registry/stage33_loso_run_registry.csv"
CONTRACT = PROJECT_ROOT / "artifacts/experiments/stage33a_loso_plan_contract.json"
DOC = PROJECT_ROOT / "docs/STAGE_33A_LOSO_PLAN.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_hash(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").strip().split()[0].lower()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_hash(path: Path) -> str:
    hash_path = path.with_suffix(".sha256")
    require(path.is_file(), f"Missing frozen artifact: {path}.")
    require(hash_path.is_file(), f"Missing frozen hash: {hash_path}.")
    recorded = read_hash(hash_path)
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    normalized = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    require(
        recorded in {actual, normalized},
        f"Frozen hash mismatch: {path.name}.",
    )
    return recorded


def write_hash(path: Path, digest: str) -> None:
    path.with_suffix(".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8", newline="\n"
    )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def git_context() -> dict:
    if not (PROJECT_ROOT / ".git").exists():
        return {
            "repository_available": False,
            "tag": STAGE32_RESULTS_TAG,
            "tag_is_annotated": None,
            "tag_commit": STAGE32_RESULTS_COMMIT,
            "tag_is_ancestor": None,
            "head_when_built": STAGE32_RESULTS_COMMIT,
        }

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()

    tag_type = git("cat-file", "-t", f"refs/tags/{STAGE32_RESULTS_TAG}")
    tag_commit = git("rev-list", "-n", "1", STAGE32_RESULTS_TAG)
    head = git("rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", tag_commit, head],
        cwd=PROJECT_ROOT,
    ).returncode == 0
    require(tag_type == "tag", "Stage-32 results tag must be annotated.")
    require(tag_commit == STAGE32_RESULTS_COMMIT, "Stage-32 results tag moved.")
    require(ancestor, "Stage-32 results tag is not an ancestor of HEAD.")
    return {
        "repository_available": True,
        "tag": STAGE32_RESULTS_TAG,
        "tag_is_annotated": True,
        "tag_commit": tag_commit,
        "tag_is_ancestor": True,
        "head_when_built": head,
    }


def config_hashes() -> dict[str, str]:
    require_hash(CONFIG_MANIFEST)
    manifest = json.loads(CONFIG_MANIFEST.read_text(encoding="utf-8-sig"))
    hashes = {}
    for entry in manifest["configs"]:
        relative = str(entry["path"]).replace("\\", "/")
        raw = (PROJECT_ROOT / relative).read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        normalized = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
        require(
            entry["sha256"] in {actual, normalized},
            f"Frozen config changed: {relative}.",
        )
        hashes[relative] = entry["sha256"]
    return hashes


def support_summary(stage14_rows: list[dict], stage15_rows: list[dict]) -> list[dict]:
    stage14_test = {
        (row["loso_id"], row["sample_id"])
        for row in stage14_rows if row["role"] == "test"
    }
    stage15_keys = {(row["loso_id"], row["sample_id"]) for row in stage15_rows}
    require(len(stage15_keys) == len(stage15_rows), "Duplicate Stage-15 support row.")
    require(stage15_keys == stage14_test, "Stage-15 support rows differ from LOSO tests.")
    output = []
    for loso_id in LOSO_IDS:
        rows = [row for row in stage15_rows if row["loso_id"] == loso_id]
        counts = Counter(row["support_class"] for row in rows)
        output.append(
            {
                "loso_id": loso_id,
                "held_out_subject": HELD_OUT_SUBJECTS[loso_id],
                "loso_all": len(rows),
                "template_supported": counts["template_supported"],
                "token_supported_novel_template": counts[
                    "token_supported_novel_template"
                ],
                "token_unsupported": counts["token_unsupported"],
                "token_supported_total": (
                    counts["template_supported"]
                    + counts["token_supported_novel_template"]
                ),
            }
        )
    return output


def main() -> None:
    print("=" * 78)
    print("STAGE 33A - LOSO INNER SPLIT AND 15-RUN PLAN FREEZE")
    print("=" * 78)

    frozen_git = git_context()
    hashes = config_hashes()
    stage14_hash = require_hash(STAGE14)
    stage15_hash = require_hash(STAGE15)
    stage17_hash = require_hash(STAGE17)
    vocab_hash = require_hash(VOCAB)
    stage32_results_hash = require_hash(STAGE32_RESULTS)
    stage32_registry_hash = require_hash(STAGE32_REGISTRY)

    stage32 = json.loads(STAGE32_RESULTS.read_text(encoding="utf-8-sig"))
    require(stage32["audit"]["completed_outer_evaluations"] == 25, "Stage 32C incomplete.")
    require(stage32["audit"]["all_registry_rows_complete"] is True, "Stage 32C registry incomplete.")
    require(stage32["audit"]["missing_or_invalid_artifacts"] == 0, "Stage 32C evidence invalid.")

    stage14_rows = read_csv(STAGE14)
    stage15_rows = read_csv(STAGE15)
    source = validate_stage14_rows(stage14_rows)
    support = support_summary(stage14_rows, stage15_rows)

    assignments = []
    summaries = []
    for loso_id in LOSO_IDS:
        direction, summary = build_loso_assignment(
            stage14_rows, loso_id, validation_fraction=0.10, seed=PROJECT_SEED
        )
        assignments.extend(direction)
        summaries.append(summary)
    require(len(assignments) == 22200, "Expected 22,200 Stage-33 assignment rows.")

    ASSIGNMENT.parent.mkdir(parents=True, exist_ok=True)
    assignment_fields = (
        "loso_id", "loso_index", "held_out_subject", "sample_id", "subject",
        "sentence_id", "split_group_id", "stage14_role", "partition",
    )
    with ASSIGNMENT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=assignment_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(assignments)
    assignment_hash = sha256_file(ASSIGNMENT)
    write_hash(ASSIGNMENT, assignment_hash)

    summary_fields = tuple(summaries[0])
    with SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)

    model_hashes = {
        model: hashes[f"configs/models/{model}.yaml"] for model in MODEL_IDS
    }
    evaluations = build_run_matrix(
        model_config_hashes=model_hashes,
        data_config_sha256=hashes["configs/data.yaml"],
        experiment_config_sha256=hashes["configs/experiments/loso.yaml"],
        stage14_split_sha256=stage14_hash,
        stage15_support_sha256=stage15_hash,
        stage33_inner_split_sha256=assignment_hash,
        data_manifest_sha256=stage17_hash,
        label_vocab_sha256=vocab_hash,
    )
    matrix = validate_run_matrix(evaluations)
    registry_rows = build_registry_plan(evaluations)
    registry_summary = validate_registry_plan(registry_rows)

    registry_fields = tuple(registry_rows[0])
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=registry_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(registry_rows)
    registry_hash = sha256_file(REGISTRY)
    write_hash(REGISTRY, registry_hash)

    run_plan = {
        "project_generation": "fresh_start_2026_v1",
        "stage": 33,
        "substage": "A_plan_freeze",
        "protocol": {
            "name": "three_subject_loso_nested_selection_refit",
            "direction_order": list(LOSO_IDS),
            "model_order": list(MODEL_IDS),
            "matrix_order": "loso_major",
            "inner_source": "training_subjects_only",
            "inner_validation_fraction": 0.10,
            "inner_split_seed_formula": "20260826 + loso_index",
            "inner_split_unit": "split_group_id",
            "inner_stratification": ["subject", "sentence_id"],
            "selection_metric": "inner_validation_wer_minimum",
            "reset_before_refit": True,
            "refit_epochs": "selected_epoch_E_star",
            "outer_test_evaluations_per_model_direction": 1,
            "outer_test_controls_training": False,
            "report_slices": [
                "loso_all", "template_supported", "token_supported", "support_statistics"
            ],
            "support_slices_are_evaluation_only": True,
            "hardware_required": "NVIDIA_Tesla_T4",
        },
        "freeze_boundary": {
            **frozen_git,
            "required_runner_tag_before_scoring": REQUIRED_RUNNER_TAG,
            "runner_tag_created_in_substage_a": False,
        },
        "support_summary": support,
        "evaluations": evaluations,
        "registry_rows": registry_rows,
    }
    write_json(RUN_PLAN, run_plan)
    run_plan_hash = sha256_file(RUN_PLAN)
    write_hash(RUN_PLAN, run_plan_hash)

    contract = {
        "project_generation": "fresh_start_2026_v1",
        "stage": 33,
        "substage": "A_plan_freeze",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_boundary": frozen_git,
        "stage32c": {
            "completed_outer_evaluations": 25,
            "completed_registry_rows": 75,
            "missing_or_invalid_artifacts": 0,
            "results_contract_sha256": stage32_results_hash,
            "completed_registry_sha256": stage32_registry_hash,
        },
        "stage14": {**source, "assignment_sha256": stage14_hash},
        "stage15": {"support_assignment_sha256": stage15_hash, "directions": support},
        "inner_splits": {
            "assignment_rows": len(assignments),
            "assignment_sha256": assignment_hash,
            "directions": summaries,
            "all_duplicate_crossings_zero": all(
                row["duplicate_groups_crossing_partitions"] == 0 for row in summaries
            ),
            "all_held_out_training_leaks_zero": all(
                row["held_out_subject_training_leaks"] == 0 for row in summaries
            ),
        },
        "matrix": matrix,
        "registry": registry_summary,
        "leakage_barrier": {
            "training_started": False,
            "outer_scores_present": False,
            "checkpoints_present": False,
            "outer_test_controls_training": False,
            "support_slices_train_separate_models": False,
        },
        "hashes": {
            "assignment": assignment_hash,
            "run_plan": run_plan_hash,
            "initial_loso_registry": registry_hash,
        },
        "stage33a_passed": True,
    }
    write_json(CONTRACT, contract)
    contract_hash = sha256_file(CONTRACT)
    write_hash(CONTRACT, contract_hash)

    rows_md = "\n".join(
        f"| {x['loso_id']} | {x['held_out_subject']} | {x['inner_train_recordings']} | "
        f"{x['inner_validation_recordings']} | {x['outer_test_recordings']} |"
        for x in summaries
    )
    support_md = "\n".join(
        f"| {x['loso_id']} | {x['loso_all']} | {x['template_supported']} | "
        f"{x['token_supported_total']} | {x['token_unsupported']} |"
        for x in support
    )
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(
        f"""# Stage 33A - LOSO unseen-signer plan freeze

Stage 33 evaluates generalization to a signer excluded from all training and
model-selection data. It plans 15 outer evaluations: five models in each of
three held-out-subject directions.

## Frozen inner assignments

| Direction | Held out | Inner train | Inner validation | Outer test |
|---|---|---:|---:|---:|
{rows_md}

The split unit is the exact-duplicate `split_group_id`. Inner validation is
stratified by training subject and sentence, using seed `20260826 + loso_index`.
The held-out subject is never used for normalization, selection, scheduling,
early stopping, or refit.

## Required reporting views

| Direction | All | Template supported | Token supported | Token unsupported |
|---|---:|---:|---:|---:|
{support_md}

Support views are filters over the same predictions. They do not create new
models or permit test information to affect training.

## Evaluation judgment

Primary comparison uses pooled WER across all held-out samples. Also report
SER, exact sentence accuracy, edit counts, five-fold/three-direction variation,
efficiency measures, and support-stratified performance. Accuracy alone is not
the model-selection criterion.

## Stage boundary

Stage 33A starts no training and computes no outer score. A resumable GPU runner
must pass its dry-run verifier and be tagged `{REQUIRED_RUNNER_TAG}` before any
held-out-signer scoring begins.
""",
        encoding="utf-8",
        newline="\n",
    )

    print("\nFROZEN DEPENDENCIES\n" + "-" * 78)
    print("Stage-32 completed evaluations: 25")
    print("Stage-32 completed registry rows: 75")
    print("Stage-14 assignment hash match: True")
    print("Stage-15 support hash match: True")
    print("Required future runner tag:", REQUIRED_RUNNER_TAG)
    print("\nINNER SPLIT SUMMARY\n" + "-" * 78)
    for row in summaries:
        print(
            f"{row['loso_id']} ({row['held_out_subject']}): "
            f"train={row['inner_train_recordings']} "
            f"validation={row['inner_validation_recordings']} "
            f"test={row['outer_test_recordings']} crossings=0 leaks=0"
        )
    print("\nSUPPORT-AWARE REPORTING\n" + "-" * 78)
    for row in support:
        print(
            f"{row['loso_id']}: all={row['loso_all']} "
            f"template={row['template_supported']} "
            f"token={row['token_supported_total']} "
            f"unsupported={row['token_unsupported']}"
        )
    print("\nRUN MATRIX\n" + "-" * 78)
    print("Outer evaluations:", matrix["runs"])
    print("Registry rows:", registry_summary["registry_rows"])
    print("Order: LOSO-major")
    print("First:", evaluations[0]["experiment_id"])
    print("Last:", evaluations[-1]["experiment_id"])
    print("\nLEAKAGE BARRIER\n" + "-" * 78)
    print("Training started: False")
    print("Outer scores present: False")
    print("Held-out subject controls training: False")
    print("Support slices train separate models: False")
    print("\nFINAL STATUS: STAGE 33A PLAN BUILD PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()
