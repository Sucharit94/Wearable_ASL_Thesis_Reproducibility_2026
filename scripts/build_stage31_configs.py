from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CATALOG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "models"
    / "stage25_model_catalog.json"
)

CONFIG_HASH_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "config"
    / "stage31_config_hashes.json"
)

CONFIG_HASH_SHA_PATH = (
    CONFIG_HASH_PATH.with_suffix(".sha256")
)

PROJECT_GENERATION = "fresh_start_2026_v1"
PROJECT_SEED = 20260826

MODEL_IDS = (
    "g1",
    "c1",
    "c2",
    "c3",
    "c4",
)

MODEL_FILES = {
    "g1": (
        "src/models/g1_near_gu.py",
        "artifacts/models/stage26_g1_model_contract.json",
        "artifacts/models/stage26_g1_model.sha256",
        "G1NearGu",
    ),
    "c1": (
        "src/models/c1_compact.py",
        "artifacts/models/stage27_c1_model_contract.json",
        "artifacts/models/stage27_c1_model.sha256",
        "C1Compact",
    ),
    "c2": (
        "src/models/c2_self_attention.py",
        "artifacts/models/stage28_c2_model_contract.json",
        "artifacts/models/stage28_c2_model.sha256",
        "C2SelfAttention",
    ),
    "c3": (
        "src/models/c3_transformer.py",
        "artifacts/models/stage29_c3_model_contract.json",
        "artifacts/models/stage29_c3_model.sha256",
        "C3Transformer",
    ),
    "c4": (
        "src/models/c4_tiny_swin.py",
        "artifacts/models/stage30_c4_model_contract.json",
        "artifacts/models/stage30_c4_model.sha256",
        "C4TinySwin",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(
        Path(path).read_text(
            encoding="utf-8-sig"
        )
    )


def write_json_yaml(
    path: Path,
    payload: dict,
) -> None:
    """
    JSON is a valid subset of YAML 1.2.
    This avoids adding PyYAML as a project dependency.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def required_sha256(
    relative_path: str,
) -> str:
    path = (
        PROJECT_ROOT
        / relative_path
    )

    if not path.is_file():
        raise FileNotFoundError(
            "Missing frozen dependency: "
            f"{relative_path}."
        )

    return sha256_file(path)


def common_recognition_policy() -> dict:
    return {
        "model_ids": list(MODEL_IDS),
        "seed": PROJECT_SEED,
        "normalization": (
            "feature_zscore_fit_on_current_"
            "training_partition_only"
        ),
        "augmentation": "none",
        "ctc_decoder": (
            "greedy_valid_length_truncated_v1"
        ),
        "test_controls_training": False,
        "selection": {
            "inner_validation_only": True,
            "metric": "word_error_rate",
            "tie_break_order": [
                "lower_validation_wer",
                "higher_validation_sentence_accuracy",
                "lower_validation_ctc_loss",
                "earlier_epoch",
            ],
            "reset_before_refit": True,
        },
        "refit": {
            "training_pool": (
                "all_non_test_training_samples"
            ),
            "epochs": "selected_epoch_count",
            "validation_used": False,
            "g1_schedule": (
                "constant_catalog_learning_rate"
            ),
            "compact_schedule": (
                "deterministic_linear_"
                "warmup_cosine_v1"
            ),
            "compact_warmup_epochs_formula": (
                "min(5,max(1,ceil("
                "0.10*selected_epochs)))"
            ),
            "compact_minimum_learning_rate_ratio": (
                0.01
            ),
        },
        "memory_fallback": {
            "allowed_cause": (
                "verified_out_of_memory_only"
            ),
            "physical_batch_candidates": (
                "halve_until_one"
            ),
            "preserve_effective_batch_with_"
            "gradient_accumulation": True,
            "restart_run_from_seed_after_change": True,
            "must_log_change": True,
            "other_hyperparameter_fallback_allowed": (
                False
            ),
        },
    }


def main() -> None:
    catalog = load_json(
        CATALOG_PATH
    )

    catalog_models = {
        row["model_id"]: row
        for row in catalog["models"]
    }

    if tuple(catalog_models) != MODEL_IDS:
        raise RuntimeError(
            "Stage-25 catalog model order changed."
        )

    data_payload = {
        "schema_version": 1,
        "project_generation": (
            PROJECT_GENERATION
        ),
        "status": (
            "frozen_before_outer_evaluation"
        ),
        "dataset": {
            "recordings": 7400,
            "sentence_templates": 300,
            "subjects": 3,
            "feature_width": 38,
            "input_layout": "T_x_38",
            "raw_cache": (
                "cache/raw/stage17_raw_f32.bin"
            ),
            "raw_cache_index": (
                "artifacts/cache/"
                "stage17_raw_cache_index.csv"
            ),
            "sentence_targets": (
                "artifacts/labels/"
                "stage10_sentence_targets.csv"
            ),
            "token_to_id": (
                "artifacts/labels/"
                "stage10_token_to_id.json"
            ),
        },
        "splits": {
            "known_signer_5fold": (
                "artifacts/splits/"
                "stage13_known_signer_5fold.csv"
            ),
            "loso": (
                "artifacts/splits/"
                "stage14_loso_assignment.csv"
            ),
            "loso_support": (
                "artifacts/splits/"
                "stage15_loso_support_assignment.csv"
            ),
            "template_holdout": (
                "artifacts/splits/"
                "stage16_template_holdout_assignment.csv"
            ),
        },
        "split_hashes": {
            "known_signer_5fold": (
                required_sha256(
                    "artifacts/splits/"
                    "stage13_known_signer_5fold.csv"
                )
            ),
            "loso": required_sha256(
                "artifacts/splits/"
                "stage14_loso_assignment.csv"
            ),
            "loso_support": required_sha256(
                "artifacts/splits/"
                "stage15_loso_support_assignment.csv"
            ),
            "template_holdout": required_sha256(
                "artifacts/splits/"
                "stage16_template_holdout_assignment.csv"
            ),
        },
        "preprocessing": {
            "normalization": (
                "training_partition_feature_"
                "zscore_per_feature"
            ),
            "normalization_recomputed_for_every_"
            "training_partition": True,
            "input_padding_value": 0.0,
            "target_padding_value": -1,
            "ctc_blank_id": 0,
            "valid_lengths_required": True,
            "augmentation": "none",
        },
        "leakage_guards": {
            "exact_duplicate_groups_must_not_"
            "cross_partitions": True,
            "test_labels_must_not_control_training": (
                True
            ),
            "normalization_must_not_observe_"
            "validation_or_test": True,
        },
    }

    config_paths: list[Path] = []

    data_path = (
        PROJECT_ROOT
        / "configs"
        / "data.yaml"
    )

    write_json_yaml(
        data_path,
        data_payload,
    )

    config_paths.append(
        data_path
    )

    for model_id in MODEL_IDS:
        (
            source_relative,
            contract_relative,
            recorded_hash_relative,
            constructor,
        ) = MODEL_FILES[model_id]

        source_path = (
            PROJECT_ROOT
            / source_relative
        )

        contract_path = (
            PROJECT_ROOT
            / contract_relative
        )

        recorded_hash_path = (
            PROJECT_ROOT
            / recorded_hash_relative
        )

        actual_model_hash = (
            sha256_file(source_path)
        )

        recorded_model_hash = (
            recorded_hash_path
            .read_text(
                encoding="utf-8-sig"
            )
            .strip()
            .split()[0]
            .lower()
        )

        if (
            actual_model_hash
            != recorded_model_hash
        ):
            raise RuntimeError(
                f"{model_id} frozen model "
                "hash mismatch."
            )

        implementation_contract = (
            load_json(
                contract_path
            )
        )

        catalog_entry = (
            catalog_models[model_id]
        )

        parameter_counts = (
            implementation_contract.get(
                "parameter_counts",
                implementation_contract.get(
                    "parameters"
                ),
            )
        )

        if parameter_counts is None:
            raise RuntimeError(
                f"{model_id} implementation "
                "contract has no parameter section."
            )

        model_payload = {
            "schema_version": 1,
            "project_generation": (
                PROJECT_GENERATION
            ),
            "status": (
                "frozen_before_outer_evaluation"
            ),
            "model_id": model_id,
            "constructor": constructor,
            "architecture_family": (
                catalog_entry[
                    "architecture_family"
                ]
            ),
            "scientific_role": (
                catalog_entry[
                    "scientific_role"
                ]
            ),
            "input_adapter": (
                catalog_entry[
                    "input_adapter"
                ]
            ),
            "source": {
                "path": source_relative,
                "sha256": actual_model_hash,
            },
            "implementation_contract": {
                "path": contract_relative,
                "sha256": sha256_file(
                    contract_path
                ),
                "stage": int(
                    implementation_contract[
                        "stage"
                    ]
                ),
                "parameter_counts": (
                    parameter_counts
                ),
            },
            "interface": {
                "input": (
                    "B_x_T_x_38_padded_float"
                ),
                "input_lengths": (
                    "B_exact_raw_valid_lengths"
                ),
                "output_logits": (
                    "B_x_Tout_x_456_raw_logits"
                ),
                "output_lengths": (
                    "B_exact_valid_output_lengths"
                ),
                "blank_id": 0,
                "target_padding_value": -1,
            },
            "training": {
                "optimizer": (
                    catalog_entry[
                        "optimizer"
                    ]
                ),
                "learning_rate": (
                    catalog_entry[
                        "learning_rate"
                    ]
                ),
                "weight_decay": (
                    catalog_entry[
                        "weight_decay"
                    ]
                ),
                "epoch_cap": (
                    catalog_entry[
                        "epoch_cap"
                    ]
                ),
                "early_stopping_patience": (
                    catalog_entry[
                        "early_stopping_patience"
                    ]
                ),
                "physical_batch_size": (
                    catalog_entry[
                        "physical_batch_size"
                    ]
                ),
                "gradient_clip_norm": (
                    catalog_entry[
                        "gradient_clip_norm"
                    ]
                ),
                "selection_scheduler": (
                    catalog_entry[
                        "scheduler"
                    ]
                ),
                "fixed_epoch_reference": (
                    catalog_entry[
                        "fixed_epoch_reference"
                    ]
                ),
            },
        }

        model_path = (
            PROJECT_ROOT
            / "configs"
            / "models"
            / f"{model_id}.yaml"
        )

        write_json_yaml(
            model_path,
            model_payload,
        )

        config_paths.append(
            model_path
        )

    common = (
        common_recognition_policy()
    )

    experiment_payloads = {
        "known_5fold": {
            "protocol_id": (
                "known_signer_5fold_v1"
            ),
            "split_path": (
                "artifacts/splits/"
                "stage13_known_signer_5fold.csv"
            ),
            "outer_folds": 5,
            "inner_validation_fraction": 0.10,
            "inner_grouping": (
                "exact_duplicate_sha256_group"
            ),
            "inner_stratification": (
                "subject_plus_sentence_as_"
                "distribution_permits"
            ),
            "inner_seed_formula": (
                "20260826 + outer_fold"
            ),
            "matrix_order": (
                "finish_all_five_models_on_fold_"
                "before_next_fold"
            ),
            "outer_test_evaluations_per_model": 5,
            "shared_split_for_all_models": True,
        },
        "loso": {
            "protocol_id": (
                "three_subject_loso_v1"
            ),
            "split_path": (
                "artifacts/splits/"
                "stage14_loso_assignment.csv"
            ),
            "support_path": (
                "artifacts/splits/"
                "stage15_loso_support_assignment.csv"
            ),
            "folds": {
                "L1": {
                    "train": ["S2", "S3"],
                    "test": ["S1"],
                },
                "L2": {
                    "train": ["S1", "S3"],
                    "test": ["S2"],
                },
                "L3": {
                    "train": ["S1", "S2"],
                    "test": ["S3"],
                },
            },
            "inner_validation_fraction": 0.10,
            "inner_source": (
                "training_subjects_only"
            ),
            "inner_grouping": (
                "exact_duplicate_sha256_group"
            ),
            "inner_seed_formula": (
                "20260826 + loso_fold_index"
            ),
            "report_slices": [
                "loso_all",
                "template_supported",
                "token_supported",
                "support_statistics",
            ],
        },
        "template_holdout": {
            "protocol_id": (
                "support_safe_template_holdout_v1"
            ),
            "split_path": (
                "artifacts/splits/"
                "stage16_template_holdout_assignment.csv"
            ),
            "grouping_unit": "sentence_id",
            "all_recordings_of_test_sentence_"
            "are_test_only": True,
            "support_requirement": (
                "every_test_gloss_token_occurs_"
                "in_at_least_two_remaining_"
                "training_sentence_ids"
            ),
            "inner_validation_fraction": 0.10,
            "inner_source": (
                "training_templates_only"
            ),
            "inner_grouping": (
                "sentence_id_with_"
                "duplicate_protection"
            ),
            "inner_seed": PROJECT_SEED,
            "one_common_holdout_for_all_models": (
                True
            ),
        },
        "language": {
            "protocol_id": (
                "gloss_to_english_language_v1"
            ),
            "source_gloss": (
                "recognized_label_no_order_tokens"
            ),
            "reference_english": (
                "label_order_text"
            ),
            "systems": [
                "L0_raw_gloss",
                "L1_training_only_retrieval",
                "L2_constrained_generator",
                "L3_template_holdout",
            ],
            "inputs": [
                "predicted_gloss",
                "oracle_gloss",
            ],
            "retrieval_and_prompt_library": (
                "training_partition_only"
            ),
            "held_out_template_leakage_"
            "test_required": True,
            "recognition_test_predictions_are_"
            "immutable_inputs": True,
        },
        "stage31_smoke": {
            "protocol_id": (
                "stage31_common_smoke_overfit_v1"
            ),
            "scientific_use": (
                "technical_diagnostic_only"
            ),
            "reported_outer_score": False,
            "source_split": (
                "known_signer_fold01_training_only"
            ),
            "fixture_size": 20,
            "fixture_selection": (
                "shortest_input_length_"
                "then_sample_id"
            ),
            "same_sample_ids_for_all_models": True,
            "normalization_fit_on": (
                "entire_known_signer_fold01_"
                "training_partition"
            ),
            "smoke_batch_size": 2,
            "initial_overfit_microbatch": 4,
            "overfit_microbatch_candidates_"
            "on_verified_oom": [
                4,
                2,
                1,
            ],
            "minimum_epochs": 2,
            "maximum_epochs": 8,
            "minimum_relative_loss_decrease": (
                0.0001
            ),
            "sample_order": (
                "fixed_no_shuffle"
            ),
            "optimizer_lr_weight_decay": (
                "stage25_catalog_values"
            ),
            "gradient_clip_norm": (
                "stage25_catalog_value"
            ),
            "scheduler": (
                "none_for_technical_smoke"
            ),
            "checkpoint_saved": False,
            "trained_weights_reusable": False,
            "seed": PROJECT_SEED,
            "device": "cpu",
            "failure_policy": (
                "fail_on_nonfinite_loss_or_"
                "gradient_or_no_loss_decrease"
            ),
        },
    }

    for (
        name,
        protocol,
    ) in experiment_payloads.items():
        payload = {
            "schema_version": 1,
            "project_generation": (
                PROJECT_GENERATION
            ),
            "status": (
                "frozen_before_outer_evaluation"
            ),
            "protocol": protocol,
            "shared_recognition_policy": (
                common
            ),
            "code_freeze": {
                "required_git_tag": (
                    "stage31-freeze-v1"
                ),
                "tag_must_point_to_"
                "stage31_commit": True,
                "working_tree_must_be_clean_"
                "before_outer_run": True,
            },
        }

        path = (
            PROJECT_ROOT
            / "configs"
            / "experiments"
            / f"{name}.yaml"
        )

        write_json_yaml(
            path,
            payload,
        )

        config_paths.append(
            path
        )

    entries = []

    for path in sorted(
        config_paths
    ):
        relative = (
            path.relative_to(
                PROJECT_ROOT
            ).as_posix()
        )

        entries.append(
            {
                "path": relative,
                "sha256": sha256_file(
                    path
                ),
            }
        )

    manifest = {
        "schema_version": 1,
        "project_generation": (
            PROJECT_GENERATION
        ),
        "stage": 31,
        "format": (
            "json_compatible_yaml_1_2"
        ),
        "config_count": len(entries),
        "configs": entries,
    }

    write_json_yaml(
        CONFIG_HASH_PATH,
        manifest,
    )

    CONFIG_HASH_SHA_PATH.write_text(
        sha256_file(
            CONFIG_HASH_PATH
        )
        + "  "
        + CONFIG_HASH_PATH.name
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print("=" * 78)
    print(
        "STAGE 31 — PROSPECTIVE "
        "CONFIGURATION FREEZE"
    )
    print("=" * 78)

    print(
        "Config files:",
        len(entries),
    )

    for entry in entries:
        print(
            entry["sha256"],
            entry["path"],
        )

    print(
        "Saved:",
        CONFIG_HASH_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved:",
        CONFIG_HASH_SHA_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "FINAL STATUS: "
        "STAGE 31 CONFIG BUILD PASSED"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
