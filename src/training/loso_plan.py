from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence


PROJECT_SEED = 20260826
LOSO_IDS = ("loso_subject1", "loso_subject2", "loso_subject3")
HELD_OUT_SUBJECTS = {
    "loso_subject1": "subject1",
    "loso_subject2": "subject2",
    "loso_subject3": "subject3",
}
MODEL_IDS = ("g1", "c1", "c2", "c3", "c4")
REQUIRED_RUNNER_TAG = "stage33-loso-runner-v1"


def stable_sha256(*values: object) -> str:
    text = "|".join(str(value) for value in values)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_stage14_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict]:
    if not rows:
        raise ValueError("Stage-14 LOSO assignment cannot be empty.")

    required = {
        "loso_id",
        "held_out_subject",
        "sample_id",
        "subject",
        "sentence_id",
        "split_group_id",
        "role",
    }
    if not required.issubset(rows[0]):
        raise ValueError("Stage-14 LOSO columns are incomplete.")

    normalized = []
    for source in rows:
        row = {key: str(source[key]).strip() for key in required}
        if any(not value for value in row.values()):
            raise ValueError("Stage-14 LOSO identity fields cannot be empty.")
        if row["role"] not in {"train", "test"}:
            raise ValueError("Stage-14 role must be train or test.")
        normalized.append(row)
    return normalized


def validate_stage14_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_recordings: int | None = 7400,
) -> dict:
    normalized = normalize_stage14_rows(rows)
    by_loso: dict[str, list[dict]] = defaultdict(list)
    for row in normalized:
        by_loso[row["loso_id"]].append(row)

    if set(by_loso) != set(LOSO_IDS):
        raise ValueError("Stage-14 LOSO IDs changed.")

    all_samples = {row["sample_id"] for row in normalized}
    if expected_recordings is not None and len(all_samples) != expected_recordings:
        raise ValueError(f"Expected {expected_recordings} unique recordings.")
    if len(normalized) != len(all_samples) * len(LOSO_IDS):
        raise ValueError("Every sample must appear in every LOSO direction.")

    expected_samples = set(all_samples)
    summaries = []
    test_appearances = Counter()
    total_crossings = 0

    for loso_id in LOSO_IDS:
        direction = by_loso[loso_id]
        held_out = HELD_OUT_SUBJECTS[loso_id]
        if {row["sample_id"] for row in direction} != expected_samples:
            raise ValueError("A LOSO direction does not cover all samples.")
        if {row["held_out_subject"] for row in direction} != {held_out}:
            raise ValueError("Held-out subject does not match LOSO ID.")

        mismatches = sum(
            (row["role"] == "test") != (row["subject"] == held_out)
            for row in direction
        )
        if mismatches:
            raise ValueError("Stage-14 role leaks the held-out subject.")

        groups: dict[str, set[str]] = defaultdict(set)
        for row in direction:
            groups[row["split_group_id"]].add(row["role"])
            if row["role"] == "test":
                test_appearances[row["sample_id"]] += 1
        crossings = sum(len(roles) != 1 for roles in groups.values())
        if crossings:
            raise ValueError("An atomic group crosses Stage-14 roles.")
        total_crossings += crossings

        counts = Counter(row["role"] for row in direction)
        summaries.append(
            {
                "loso_id": loso_id,
                "held_out_subject": held_out,
                "recordings": len(direction),
                "training_recordings": counts["train"],
                "outer_test_recordings": counts["test"],
                "training_subjects": ",".join(
                    sorted({row["subject"] for row in direction if row["role"] == "train"})
                ),
                "atomic_groups": len(groups),
                "duplicate_groups_crossing_roles": 0,
            }
        )

    if set(test_appearances.values()) != {1}:
        raise ValueError("Every sample must be outer test exactly once.")

    return {
        "assignment_rows": len(normalized),
        "unique_recordings": len(all_samples),
        "loso_directions": len(LOSO_IDS),
        "each_sample_outer_test_once": True,
        "duplicate_groups_crossing_roles": total_crossings,
        "directions": summaries,
    }


def _validation_groups(
    training_rows: Sequence[dict],
    *,
    loso_id: str,
    validation_fraction: float,
    seed: int,
) -> set[str]:
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5.")

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in training_rows:
        groups[row["split_group_id"]].append(row)

    groups_by_stratum: dict[tuple[str, str], list[str]] = defaultdict(list)
    for group_id, group_rows in groups.items():
        strata = {(row["subject"], row["sentence_id"]) for row in group_rows}
        if len(strata) != 1:
            raise ValueError("An atomic group crosses a subject-sentence stratum.")
        groups_by_stratum[next(iter(strata))].append(group_id)

    selected = set()
    for stratum in sorted(groups_by_stratum):
        group_ids = groups_by_stratum[stratum]
        if len(group_ids) < 2:
            raise ValueError("A training stratum has fewer than two atomic groups.")
        target_rows = max(
            1,
            int(round(
                sum(len(groups[group_id]) for group_id in group_ids)
                * validation_fraction
            )),
        )
        ordered = sorted(
            group_ids,
            key=lambda group_id: (
                len(groups[group_id]),
                stable_sha256("stage33_loso_inner_v1", seed, loso_id, group_id),
            ),
        )
        chosen_rows = 0
        for group_id in ordered:
            if chosen_rows >= target_rows:
                break
            selected.add(group_id)
            chosen_rows += len(groups[group_id])
        if chosen_rows == 0 or chosen_rows >= sum(len(groups[g]) for g in group_ids):
            raise RuntimeError("Could not create an inner validation stratum.")
    return selected


def build_loso_assignment(
    rows: Sequence[Mapping[str, object]],
    loso_id: str,
    *,
    validation_fraction: float = 0.10,
    seed: int = PROJECT_SEED,
) -> tuple[list[dict], dict]:
    if loso_id not in LOSO_IDS:
        raise ValueError("Unknown LOSO direction.")
    normalized = normalize_stage14_rows(rows)
    direction = [row for row in normalized if row["loso_id"] == loso_id]
    training = [row for row in direction if row["role"] == "train"]
    outer_test = [row for row in direction if row["role"] == "test"]
    if not training or not outer_test:
        raise ValueError("LOSO direction contains an empty role.")

    loso_index = LOSO_IDS.index(loso_id) + 1
    selected = _validation_groups(
        training,
        loso_id=loso_id,
        validation_fraction=validation_fraction,
        seed=seed + loso_index,
    )

    assignments = []
    for row in direction:
        if row["role"] == "test":
            partition = "outer_test"
        elif row["split_group_id"] in selected:
            partition = "inner_validation"
        else:
            partition = "inner_train"
        assignments.append(
            {
                "loso_id": loso_id,
                "loso_index": loso_index,
                "held_out_subject": row["held_out_subject"],
                "sample_id": row["sample_id"],
                "subject": row["subject"],
                "sentence_id": row["sentence_id"],
                "split_group_id": row["split_group_id"],
                "stage14_role": row["role"],
                "partition": partition,
            }
        )
    return assignments, validate_loso_assignment(assignments)


def validate_loso_assignment(
    assignments: Sequence[Mapping[str, object]],
) -> dict:
    if not assignments:
        raise ValueError("LOSO assignment cannot be empty.")
    loso_ids = {str(row["loso_id"]) for row in assignments}
    if len(loso_ids) != 1:
        raise ValueError("Assignment mixes LOSO directions.")
    loso_id = next(iter(loso_ids))
    held_out = HELD_OUT_SUBJECTS[loso_id]
    if len({str(row["sample_id"]) for row in assignments}) != len(assignments):
        raise ValueError("A sample appears twice in one LOSO assignment.")

    allowed = {"inner_train", "inner_validation", "outer_test"}
    if {str(row["partition"]) for row in assignments} != allowed:
        raise ValueError("A LOSO assignment is missing a partition.")

    groups: dict[str, set[str]] = defaultdict(set)
    strata: dict[tuple[str, str], set[str]] = defaultdict(set)
    held_out_leaks = 0
    outer_mismatches = 0
    for row in assignments:
        partition = str(row["partition"])
        groups[str(row["split_group_id"])].add(partition)
        if str(row["stage14_role"]) == "train":
            strata[(str(row["subject"]), str(row["sentence_id"]))].add(partition)
        if partition != "outer_test" and str(row["subject"]) == held_out:
            held_out_leaks += 1
        outer_mismatches += (
            (partition == "outer_test") != (str(row["stage14_role"]) == "test")
        )
    crossings = sum(len(parts) != 1 for parts in groups.values())
    missing_strata = sum(
        not {"inner_train", "inner_validation"}.issubset(parts)
        for parts in strata.values()
    )
    if crossings or held_out_leaks or outer_mismatches or missing_strata:
        raise ValueError("LOSO leakage or inner-stratification validation failed.")

    counts = Counter(str(row["partition"]) for row in assignments)
    pool = counts["inner_train"] + counts["inner_validation"]
    return {
        "loso_id": loso_id,
        "loso_index": LOSO_IDS.index(loso_id) + 1,
        "held_out_subject": held_out,
        "recordings": len(assignments),
        "inner_train_recordings": counts["inner_train"],
        "inner_validation_recordings": counts["inner_validation"],
        "outer_test_recordings": counts["outer_test"],
        "training_pool_recordings": pool,
        "observed_validation_fraction": counts["inner_validation"] / pool,
        "duplicate_groups_crossing_partitions": 0,
        "outer_test_assignment_mismatches": 0,
        "held_out_subject_training_leaks": 0,
        "strata_missing_inner_partition": 0,
    }


def build_run_matrix(
    *,
    model_config_hashes: Mapping[str, str],
    data_config_sha256: str,
    experiment_config_sha256: str,
    stage14_split_sha256: str,
    stage15_support_sha256: str,
    stage33_inner_split_sha256: str,
    data_manifest_sha256: str,
    label_vocab_sha256: str,
    seed: int = PROJECT_SEED,
) -> list[dict]:
    if set(model_config_hashes) != set(MODEL_IDS):
        raise ValueError("Model config hashes must cover exactly G1 and C1-C4.")
    runs = []
    for loso_index, loso_id in enumerate(LOSO_IDS, start=1):
        held_out = HELD_OUT_SUBJECTS[loso_id]
        for model_id in MODEL_IDS:
            run_order = len(runs) + 1
            experiment_id = f"{model_id}_{loso_id}_seed{seed}_v1"
            payload = {
                "model": model_config_hashes[model_id],
                "data": data_config_sha256,
                "experiment": experiment_config_sha256,
                "stage14": stage14_split_sha256,
                "stage15": stage15_support_sha256,
                "stage33": stage33_inner_split_sha256,
                "model_id": model_id,
                "loso_id": loso_id,
                "seed": seed,
            }
            runs.append(
                {
                    "run_order": run_order,
                    "experiment_id": experiment_id,
                    "model_id": model_id,
                    "loso_index": loso_index,
                    "loso_id": loso_id,
                    "held_out_subject": held_out,
                    "split_id": f"{loso_id}_v1",
                    "seed": seed,
                    "inner_seed": seed + loso_index,
                    "phase": "outer_evaluation",
                    "run_directory": f"artifacts/runs/loso/{loso_id}/{model_id}",
                    "data_manifest_sha256": data_manifest_sha256,
                    "label_vocab_sha256": label_vocab_sha256,
                    "config_sha256": canonical_json_sha256(payload),
                    "required_code_tag": REQUIRED_RUNNER_TAG,
                    "status": "PLANNED",
                    "outer_score_present": False,
                    "report_slices": [
                        "loso_all",
                        "template_supported",
                        "token_supported",
                        "support_statistics",
                    ],
                }
            )
    validate_run_matrix(runs)
    return runs


def validate_run_matrix(runs: Sequence[Mapping[str, object]]) -> dict:
    if len(runs) != 15:
        raise ValueError("LOSO run matrix must contain 15 evaluations.")
    expected = [(loso_id, model) for loso_id in LOSO_IDS for model in MODEL_IDS]
    observed = [(str(run["loso_id"]), str(run["model_id"])) for run in runs]
    if observed != expected:
        raise ValueError("Run matrix is not in frozen LOSO-major order.")
    if [int(run["run_order"]) for run in runs] != list(range(1, 16)):
        raise ValueError("Run order must be 1 through 15.")
    if len({str(run["experiment_id"]) for run in runs}) != 15:
        raise ValueError("Experiment IDs are not unique.")
    if any(str(run["status"]) != "PLANNED" for run in runs):
        raise ValueError("A Stage-33A run is not PLANNED.")
    if any(bool(run["outer_score_present"]) for run in runs):
        raise ValueError("Stage 33A cannot contain LOSO scores.")
    return {
        "runs": 15,
        "loso_directions": 3,
        "models_per_direction": 5,
        "loso_major_order": True,
        "unique_experiment_ids": 15,
        "outer_scores_present": 0,
    }


def build_registry_plan(evaluations: Sequence[Mapping[str, object]]) -> list[dict]:
    validate_run_matrix(evaluations)
    rows = []
    phases = ("outer_evaluation", "inner_selection", "full_refit")
    for evaluation in evaluations:
        parent = str(evaluation["experiment_id"])
        base = str(evaluation["run_directory"])
        for phase in phases:
            registry_id = parent if phase == "outer_evaluation" else f"{parent}__{phase}"
            artifact_path = base if phase == "outer_evaluation" else f"{base}/{phase}"
            rows.append(
                {
                    "registry_order": len(rows) + 1,
                    "evaluation_run_order": int(evaluation["run_order"]),
                    "registry_id": registry_id,
                    "experiment_id": registry_id,
                    "parent_experiment_id": parent,
                    "project_generation": "fresh_start_2026_v1",
                    "model_id": str(evaluation["model_id"]),
                    "loso_index": int(evaluation["loso_index"]),
                    "loso_id": str(evaluation["loso_id"]),
                    "held_out_subject": str(evaluation["held_out_subject"]),
                    "split_id": str(evaluation["split_id"]),
                    "phase": phase,
                    "data_manifest_sha256": str(evaluation["data_manifest_sha256"]),
                    "label_vocab_sha256": str(evaluation["label_vocab_sha256"]),
                    "config_sha256": str(evaluation["config_sha256"]),
                    "code_commit": "PENDING_STAGE33_RUNNER_TAG",
                    "seed": int(evaluation["seed"]),
                    "normalization": "train_partition_feature_zscore",
                    "augmentation": "none",
                    "ctc_decoder": "length_safe_greedy_ctc",
                    "hardware_required": "NVIDIA_Tesla_T4",
                    "status": "PLANNED",
                    "artifact_path": artifact_path,
                }
            )
    validate_registry_plan(rows)
    return rows


def validate_registry_plan(rows: Sequence[Mapping[str, object]]) -> dict:
    if len(rows) != 45:
        raise ValueError("Stage-33 registry must contain 45 rows.")
    if [int(row["registry_order"]) for row in rows] != list(range(1, 46)):
        raise ValueError("Registry order must be 1 through 45.")
    grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["evaluation_run_order"])].append(row)
    expected_phases = ("outer_evaluation", "inner_selection", "full_refit")
    if set(grouped) != set(range(1, 16)):
        raise ValueError("Registry does not cover all evaluations.")
    for group in grouped.values():
        if tuple(str(row["phase"]) for row in group) != expected_phases:
            raise ValueError("Registry phase order changed.")
    if len({str(row["registry_id"]) for row in rows}) != 45:
        raise ValueError("Registry IDs are not unique.")
    if any(str(row["status"]) != "PLANNED" for row in rows):
        raise ValueError("A registry row is not PLANNED.")
    if any(str(row["code_commit"]) != "PENDING_STAGE33_RUNNER_TAG" for row in rows):
        raise ValueError("A registry row prematurely claims runner code.")
    return {
        "registry_rows": 45,
        "outer_evaluation_rows": 15,
        "inner_selection_rows": 15,
        "full_refit_rows": 15,
        "unique_registry_ids": 45,
        "all_status_planned": True,
    }
