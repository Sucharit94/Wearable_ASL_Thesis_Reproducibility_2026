from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence


PROJECT_SEED = 20260826
OUTER_FOLDS = (1, 2, 3, 4, 5)
MODEL_IDS = ("g1", "c1", "c2", "c3", "c4")

STANDARD_COLUMNS = (
    "sample_id",
    "subject",
    "sentence_id",
    "stage12_split_group_id",
    "outer_fold",
)

COLUMN_ALIASES = {
    "sample_id": ("sample_id",),
    "subject": (
        "subject",
        "subject_id",
        "subject_key",
        "signer",
        "signer_id",
    ),
    "sentence_id": (
        "sentence_id",
        "sentence",
        "template_id",
    ),
    "stage12_split_group_id": (
        "stage12_split_group_id",
        "split_group_id",
        "duplicate_group_id",
        "atomic_group_id",
    ),
    "outer_fold": ("outer_fold",),
}


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


def _resolve_columns(fieldnames: Sequence[str]) -> dict[str, str]:
    observed = set(fieldnames)
    resolved = {}

    for standard_name, aliases in COLUMN_ALIASES.items():
        matches = [alias for alias in aliases if alias in observed]

        if len(matches) != 1:
            raise ValueError(
                f"Could not resolve exactly one {standard_name} column."
            )

        resolved[standard_name] = matches[0]

    return resolved


def normalize_stage13_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict]:
    if not rows:
        raise ValueError("Stage-13 assignment cannot be empty.")

    columns = _resolve_columns(tuple(rows[0]))
    normalized = []

    for source in rows:
        sample_id = str(source[columns["sample_id"]]).strip()
        subject = str(source[columns["subject"]]).strip()
        sentence_id = str(source[columns["sentence_id"]]).strip()
        group_id = str(
            source[columns["stage12_split_group_id"]]
        ).strip()

        try:
            outer_fold = int(source[columns["outer_fold"]])
        except (TypeError, ValueError) as exc:
            raise ValueError("outer_fold must be an integer.") from exc

        if not sample_id or not subject or not sentence_id or not group_id:
            raise ValueError("Stage-13 identity columns cannot be empty.")

        normalized.append(
            {
                "sample_id": sample_id,
                "subject": subject,
                "sentence_id": sentence_id,
                "stage12_split_group_id": group_id,
                "outer_fold": outer_fold,
            }
        )

    return normalized


def validate_stage13_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_recordings: int | None = 7400,
) -> dict:
    normalized = normalize_stage13_rows(rows)

    if (
        expected_recordings is not None
        and len(normalized) != int(expected_recordings)
    ):
        raise ValueError(
            f"Expected {expected_recordings} Stage-13 recordings."
        )

    sample_ids = [row["sample_id"] for row in normalized]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Stage-13 sample IDs are not unique.")

    observed_folds = {row["outer_fold"] for row in normalized}
    if observed_folds != set(OUTER_FOLDS):
        raise ValueError("Stage-13 outer folds must be exactly 1 through 5.")

    group_folds: dict[str, set[int]] = defaultdict(set)
    for row in normalized:
        group_folds[row["stage12_split_group_id"]].add(row["outer_fold"])

    split_groups = sorted(
        group_id
        for group_id, folds in group_folds.items()
        if len(folds) != 1
    )

    if split_groups:
        raise ValueError(
            "Exact-duplicate groups cross Stage-13 folds: "
            + ", ".join(split_groups[:10])
            + "."
        )

    return {
        "recordings": len(normalized),
        "subjects": len({row["subject"] for row in normalized}),
        "sentences": len({row["sentence_id"] for row in normalized}),
        "subject_sentence_strata": len(
            {
                (row["subject"], row["sentence_id"])
                for row in normalized
            }
        ),
        "split_groups": len(group_folds),
        "duplicate_groups_crossing_outer_folds": 0,
    }


def _select_validation_groups(
    pool_rows: Sequence[dict],
    *,
    outer_fold: int,
    validation_fraction: float,
    seed: int,
) -> tuple[set[str], dict]:
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5.")

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in pool_rows:
        groups[row["stage12_split_group_id"]].append(row)

    if len(groups) < 2:
        raise ValueError("Outer-training pool has too few atomic groups.")

    target_total = int(round(len(pool_rows) * validation_fraction))
    target_total = max(1, min(target_total, len(pool_rows) - 1))

    total_subject = Counter(row["subject"] for row in pool_rows)
    total_stratum = Counter(
        (row["subject"], row["sentence_id"])
        for row in pool_rows
    )
    target_subject = {
        key: value * validation_fraction
        for key, value in total_subject.items()
    }
    target_stratum = {
        key: value * validation_fraction
        for key, value in total_stratum.items()
    }

    candidates = []
    for group_id, group_rows in groups.items():
        subject_counts = Counter(row["subject"] for row in group_rows)
        stratum_counts = Counter(
            (row["subject"], row["sentence_id"])
            for row in group_rows
        )
        candidates.append(
            {
                "group_id": group_id,
                "size": len(group_rows),
                "subject_counts": subject_counts,
                "stratum_counts": stratum_counts,
                "stable_hash": stable_sha256(
                    "stage32_known_inner_v1",
                    seed,
                    outer_fold,
                    group_id,
                ),
            }
        )

    candidates_by_id = {
        candidate["group_id"]: candidate
        for candidate in candidates
    }
    groups_by_stratum: dict[tuple[str, str], set[str]] = defaultdict(set)
    for candidate in candidates:
        for stratum in candidate["stratum_counts"]:
            groups_by_stratum[stratum].add(candidate["group_id"])

    too_small_strata = sorted(
        stratum
        for stratum, group_ids in groups_by_stratum.items()
        if len(group_ids) < 2
    )
    if too_small_strata:
        raise ValueError(
            "An outer-training stratum has fewer than two atomic groups."
        )

    selected: set[str] = set()
    selected_total = 0
    selected_subject: Counter = Counter()
    selected_stratum: Counter = Counter()

    # Give every subject-sentence stratum at least one validation group.
    # This is the exact inner analogue of Stage 13's stratification rule.
    for stratum in sorted(groups_by_stratum):
        eligible = [
            candidates_by_id[group_id]
            for group_id in groups_by_stratum[stratum]
        ]
        chosen = min(
            eligible,
            key=lambda candidate: (
                0 if candidate["group_id"] in selected else 1,
                candidate["size"],
                candidate["stable_hash"],
            ),
        )

        if chosen["group_id"] not in selected:
            selected.add(chosen["group_id"])
            selected_total += chosen["size"]
            selected_subject.update(chosen["subject_counts"])
            selected_stratum.update(chosen["stratum_counts"])

    remaining = [
        candidate
        for candidate in candidates
        if candidate["group_id"] not in selected
    ]

    while remaining and selected_total < target_total:
        current_distance = abs(selected_total - target_total)
        scored = []

        for candidate in remaining:
            new_total = selected_total + candidate["size"]
            global_cost = (
                abs(new_total - target_total) / max(1, target_total)
            )
            overshoot_cost = (
                max(0, new_total - target_total) / max(1, target_total)
            )

            subject_delta = 0.0
            for key, count in candidate["subject_counts"].items():
                before = abs(
                    selected_subject.get(key, 0) - target_subject[key]
                )
                after = abs(
                    selected_subject.get(key, 0)
                    + count
                    - target_subject[key]
                )
                subject_delta += (after - before) / max(
                    1.0,
                    target_subject[key],
                )

            stratum_delta = 0.0
            for key, count in candidate["stratum_counts"].items():
                before = abs(
                    selected_stratum.get(key, 0) - target_stratum[key]
                )
                after = abs(
                    selected_stratum.get(key, 0)
                    + count
                    - target_stratum[key]
                )
                stratum_delta += (after - before) / max(
                    1.0,
                    target_stratum[key],
                )

            score = (
                10.0 * global_cost
                + 25.0 * overshoot_cost
                + 2.0 * subject_delta
                + 0.10 * stratum_delta
            )
            scored.append((score, candidate["stable_hash"], candidate))

        _, _, chosen = min(scored, key=lambda item: (item[0], item[1]))
        new_distance = abs(selected_total + chosen["size"] - target_total)

        if selected_total >= target_total and new_distance > current_distance:
            break

        selected.add(chosen["group_id"])
        selected_total += chosen["size"]
        selected_subject.update(chosen["subject_counts"])
        selected_stratum.update(chosen["stratum_counts"])
        remaining.remove(chosen)

    if not selected or selected_total >= len(pool_rows):
        raise RuntimeError("Could not create a valid inner-validation set.")

    maximum_group_size = max(candidate["size"] for candidate in candidates)

    return selected, {
        "target_validation_recordings": target_total,
        "observed_validation_recordings": selected_total,
        "absolute_target_difference": abs(selected_total - target_total),
        "maximum_atomic_group_size": maximum_group_size,
        "validation_fraction": selected_total / len(pool_rows),
    }


def build_outer_fold_assignment(
    rows: Sequence[Mapping[str, object]],
    outer_fold: int,
    *,
    validation_fraction: float = 0.10,
    seed: int = PROJECT_SEED,
) -> tuple[list[dict], dict]:
    outer_fold = int(outer_fold)
    if outer_fold not in OUTER_FOLDS:
        raise ValueError("outer_fold must be 1 through 5.")

    normalized = normalize_stage13_rows(rows)
    outer_test = [row for row in normalized if row["outer_fold"] == outer_fold]
    outer_pool = [row for row in normalized if row["outer_fold"] != outer_fold]

    if not outer_test or not outer_pool:
        raise ValueError("Outer fold contains an empty partition.")

    validation_groups, selection = _select_validation_groups(
        outer_pool,
        outer_fold=outer_fold,
        validation_fraction=validation_fraction,
        seed=seed + outer_fold,
    )

    assignments = []
    for row in normalized:
        if row["outer_fold"] == outer_fold:
            partition = "outer_test"
        elif row["stage12_split_group_id"] in validation_groups:
            partition = "inner_validation"
        else:
            partition = "inner_train"

        assignments.append(
            {
                "sample_id": row["sample_id"],
                "subject": row["subject"],
                "sentence_id": row["sentence_id"],
                "stage12_split_group_id": row["stage12_split_group_id"],
                "stage13_outer_fold": row["outer_fold"],
                "stage32_outer_fold": outer_fold,
                "partition": partition,
            }
        )

    summary = validate_outer_fold_assignment(
        assignments,
        expected_sample_ids={row["sample_id"] for row in normalized},
        target_difference_limit=selection["maximum_atomic_group_size"],
        target_validation_recordings=selection[
            "target_validation_recordings"
        ],
    )
    summary.update(selection)
    return assignments, summary


def validate_outer_fold_assignment(
    assignments: Sequence[Mapping[str, object]],
    *,
    expected_sample_ids: set[str] | None = None,
    target_difference_limit: int | None = None,
    target_validation_recordings: int | None = None,
) -> dict:
    if not assignments:
        raise ValueError("Fold assignment cannot be empty.")

    fold_values = {int(row["stage32_outer_fold"]) for row in assignments}
    if len(fold_values) != 1:
        raise ValueError("Assignment mixes Stage-32 outer folds.")
    outer_fold = next(iter(fold_values))

    sample_ids = [str(row["sample_id"]) for row in assignments]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("A sample appears more than once in one fold.")
    if expected_sample_ids is not None and set(sample_ids) != expected_sample_ids:
        raise ValueError("Fold assignment does not cover exact Stage-13 samples.")

    allowed = {"inner_train", "inner_validation", "outer_test"}
    partitions = {str(row["partition"]) for row in assignments}
    if partitions != allowed:
        raise ValueError("Fold must contain all three exact partitions.")

    group_partitions: dict[str, set[str]] = defaultdict(set)
    for row in assignments:
        group_partitions[str(row["stage12_split_group_id"])].add(
            str(row["partition"])
        )

    crossings = sum(len(values) != 1 for values in group_partitions.values())
    if crossings:
        raise ValueError("Duplicate groups cross Stage-32 partitions.")

    outer_mismatch = sum(
        (
            str(row["partition"]) == "outer_test"
        )
        != (
            int(row["stage13_outer_fold"]) == outer_fold
        )
        for row in assignments
    )
    if outer_mismatch:
        raise ValueError("Stage-32 outer test differs from Stage 13.")

    counts = Counter(str(row["partition"]) for row in assignments)
    pool_count = counts["inner_train"] + counts["inner_validation"]

    if target_validation_recordings is None:
        target_validation_recordings = int(round(pool_count * 0.10))
    difference = abs(counts["inner_validation"] - target_validation_recordings)

    if (
        target_difference_limit is not None
        and difference > int(target_difference_limit)
    ):
        raise ValueError("Inner-validation size exceeds atomic tolerance.")

    subject_partitions = defaultdict(set)
    stratum_partitions = defaultdict(set)
    for row in assignments:
        subject_partitions[str(row["subject"])].add(str(row["partition"]))
        stratum_partitions[
            (
                str(row["subject"]),
                str(row["sentence_id"]),
            )
        ].add(str(row["partition"]))

    subjects_missing_inner_partition = sum(
        not {"inner_train", "inner_validation"}.issubset(values)
        for values in subject_partitions.values()
    )
    if subjects_missing_inner_partition:
        raise ValueError("A subject is missing from an inner partition.")

    strata_missing_inner_partition = sum(
        not {"inner_train", "inner_validation"}.issubset(values)
        for values in stratum_partitions.values()
    )
    if strata_missing_inner_partition:
        raise ValueError("A subject-sentence stratum is missing an inner partition.")

    return {
        "outer_fold": outer_fold,
        "recordings": len(assignments),
        "inner_train_recordings": counts["inner_train"],
        "inner_validation_recordings": counts["inner_validation"],
        "outer_test_recordings": counts["outer_test"],
        "outer_training_pool_recordings": pool_count,
        "observed_validation_fraction": (
            counts["inner_validation"] / pool_count
        ),
        "duplicate_groups_crossing_partitions": 0,
        "outer_test_assignment_mismatches": 0,
        "subjects_missing_inner_partition": 0,
        "strata_missing_inner_partition": 0,
    }


def build_run_matrix(
    *,
    model_config_hashes: Mapping[str, str],
    data_config_sha256: str,
    experiment_config_sha256: str,
    stage13_split_sha256: str,
    stage32_inner_split_sha256: str,
    data_manifest_sha256: str,
    label_vocab_sha256: str,
    seed: int = PROJECT_SEED,
) -> list[dict]:
    if set(model_config_hashes) != set(MODEL_IDS):
        raise ValueError("Model config hashes must cover exactly G1 and C1-C4.")

    runs = []
    run_order = 0

    for outer_fold in OUTER_FOLDS:
        for model_id in MODEL_IDS:
            run_order += 1
            split_id = f"known_fold{outer_fold:02d}_v1"
            experiment_id = (
                f"{model_id}_known_fold{outer_fold:02d}_"
                f"seed{seed}_v1"
            )
            run_config_payload = {
                "data_config_sha256": data_config_sha256,
                "experiment_config_sha256": experiment_config_sha256,
                "model_config_sha256": model_config_hashes[model_id],
                "stage13_split_sha256": stage13_split_sha256,
                "stage32_inner_split_sha256": stage32_inner_split_sha256,
                "model_id": model_id,
                "outer_fold": outer_fold,
                "seed": seed,
            }
            runs.append(
                {
                    "run_order": run_order,
                    "experiment_id": experiment_id,
                    "model_id": model_id,
                    "outer_fold": outer_fold,
                    "split_id": split_id,
                    "seed": seed,
                    "phase": "outer_evaluation",
                    "run_directory": (
                        f"artifacts/runs/known_signer/"
                        f"fold{outer_fold:02d}/{model_id}"
                    ),
                    "data_manifest_sha256": data_manifest_sha256,
                    "label_vocab_sha256": label_vocab_sha256,
                    "config_sha256": canonical_json_sha256(
                        run_config_payload
                    ),
                    "required_code_tag": "stage32-known-runner-v1",
                    "status": "PLANNED",
                    "outer_score_present": False,
                }
            )

    return runs


def validate_run_matrix(runs: Sequence[Mapping[str, object]]) -> dict:
    if len(runs) != 25:
        raise ValueError("Known-signer run matrix must contain 25 runs.")

    expected_pairs = [
        (fold, model_id)
        for fold in OUTER_FOLDS
        for model_id in MODEL_IDS
    ]
    observed_pairs = [
        (int(run["outer_fold"]), str(run["model_id"]))
        for run in runs
    ]

    if observed_pairs != expected_pairs:
        raise ValueError("Run matrix is not in frozen fold-major order.")

    if [int(run["run_order"]) for run in runs] != list(range(1, 26)):
        raise ValueError("Run-order values must be exactly 1 through 25.")

    experiment_ids = [str(run["experiment_id"]) for run in runs]
    directories = [str(run["run_directory"]) for run in runs]

    if len(set(experiment_ids)) != 25 or len(set(directories)) != 25:
        raise ValueError("Run identities or directories are not unique.")

    if any(str(run["status"]) != "PLANNED" for run in runs):
        raise ValueError("A Stage-32A run is not PLANNED.")

    if any(bool(run["outer_score_present"]) for run in runs):
        raise ValueError("Stage 32A must not contain outer scores.")

    return {
        "runs": 25,
        "outer_folds": 5,
        "models_per_fold": 5,
        "fold_major_order": True,
        "unique_experiment_ids": 25,
        "outer_scores_present": 0,
    }


def build_registry_plan(
    evaluations: Sequence[Mapping[str, object]],
) -> list[dict]:
    validate_run_matrix(evaluations)

    registry_rows = []
    registry_order = 0

    for evaluation in evaluations:
        experiment_id = str(evaluation["experiment_id"])
        run_directory = str(evaluation["run_directory"])

        phases = (
            (
                "outer_evaluation",
                experiment_id,
                run_directory,
            ),
            (
                "inner_selection",
                f"{experiment_id}__inner_selection",
                f"{run_directory}/inner_selection",
            ),
            (
                "full_refit",
                f"{experiment_id}__full_refit",
                f"{run_directory}/full_refit",
            ),
        )

        for phase, registry_id, artifact_path in phases:
            registry_order += 1
            registry_rows.append(
                {
                    "registry_order": registry_order,
                    "evaluation_run_order": int(evaluation["run_order"]),
                    "registry_id": registry_id,
                    "parent_experiment_id": experiment_id,
                    "experiment_id": registry_id,
                    "project_generation": "fresh_start_2026_v1",
                    "model_id": str(evaluation["model_id"]),
                    "outer_fold": int(evaluation["outer_fold"]),
                    "split_id": str(evaluation["split_id"]),
                    "data_manifest_sha256": str(
                        evaluation["data_manifest_sha256"]
                    ),
                    "label_vocab_sha256": str(
                        evaluation["label_vocab_sha256"]
                    ),
                    "config_sha256": str(evaluation["config_sha256"]),
                    "code_commit": "PENDING_STAGE32_RUNNER_TAG",
                    "seed": int(evaluation["seed"]),
                    "normalization": "train_partition_feature_zscore",
                    "augmentation": "none",
                    "ctc_decoder": "length_safe_greedy_ctc",
                    "hardware_required": "NVIDIA_Tesla_T4",
                    "phase": phase,
                    "status": "PLANNED",
                    "artifact_path": artifact_path,
                }
            )

    return registry_rows


def validate_registry_plan(
    registry_rows: Sequence[Mapping[str, object]],
) -> dict:
    if len(registry_rows) != 75:
        raise ValueError("Stage-32 registry plan must contain 75 rows.")

    if [int(row["registry_order"]) for row in registry_rows] != list(
        range(1, 76)
    ):
        raise ValueError("Registry order must be exactly 1 through 75.")

    expected_phases = (
        "outer_evaluation",
        "inner_selection",
        "full_refit",
    )

    grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in registry_rows:
        grouped[int(row["evaluation_run_order"])].append(row)

    if set(grouped) != set(range(1, 26)):
        raise ValueError("Registry rows do not cover all 25 evaluations.")

    for evaluation_order in range(1, 26):
        rows = grouped[evaluation_order]
        if tuple(str(row["phase"]) for row in rows) != expected_phases:
            raise ValueError("Registry phase order changed.")
        if len({str(row["parent_experiment_id"]) for row in rows}) != 1:
            raise ValueError("Registry phase parent identities differ.")

    registry_ids = [str(row["registry_id"]) for row in registry_rows]
    if len(set(registry_ids)) != 75:
        raise ValueError("Registry IDs are not unique.")

    if any(str(row["status"]) != "PLANNED" for row in registry_rows):
        raise ValueError("A Stage-32A registry row is not PLANNED.")

    if any(
        str(row["code_commit"]) != "PENDING_STAGE32_RUNNER_TAG"
        for row in registry_rows
    ):
        raise ValueError("A Stage-32A registry row already claims runner code.")

    if any(
        str(row["ctc_decoder"]) != "length_safe_greedy_ctc"
        for row in registry_rows
    ):
        raise ValueError("Registry decoder contract changed.")

    return {
        "registry_rows": 75,
        "outer_evaluation_rows": 25,
        "inner_selection_rows": 25,
        "full_refit_rows": 25,
        "unique_registry_ids": 75,
        "all_status_planned": True,
    }
