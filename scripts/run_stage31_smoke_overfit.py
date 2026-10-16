from __future__ import annotations

import csv
import gc
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import (
    ASLSequenceDataset,
    RawCacheStore,
    TargetCatalog,
    fit_normalization_for_sample_ids,
)
from src.data.splits import (
    load_known_signer_outer_fold,
)
from src.models.contracts import (
    load_model_catalog,
)
from src.models.factory import (
    EXPECTED_TRAINABLE_PARAMETERS,
    MODEL_IDS,
    build_model,
    validate_model_parameter_count,
)
from src.training.smoke_overfit import (
    is_out_of_memory_error,
    one_batch_smoke,
    set_deterministic_seed,
    tiny_overfit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_CACHE_PATH = (
    PROJECT_ROOT
    / "cache"
    / "raw"
    / "stage17_raw_f32.bin"
)

RAW_INDEX_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "cache"
    / "stage17_raw_cache_index.csv"
)

TARGET_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "labels"
    / "stage10_sentence_targets.csv"
)

KNOWN_SPLIT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "splits"
    / "stage13_known_signer_5fold.csv"
)

CATALOG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "models"
    / "stage25_model_catalog.json"
)

SMOKE_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "stage31_smoke.yaml"
)

CONFIG_HASH_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "config"
    / "stage31_config_hashes.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "training"
)

FIXTURE_PATH = (
    OUTPUT_DIR
    / "stage31_tiny_overfit_samples.json"
)

FIXTURE_HASH_PATH = (
    FIXTURE_PATH.with_suffix(".sha256")
)

RESULTS_PATH = (
    OUTPUT_DIR
    / "stage31_smoke_overfit_results.json"
)

LOG_PATH = (
    OUTPUT_DIR
    / "stage31_smoke_overfit_log.csv"
)

ENVIRONMENT_PATH = (
    OUTPUT_DIR
    / "stage31_environment.json"
)

CONTRACT_PATH = (
    OUTPUT_DIR
    / "stage31_smoke_freeze_contract.json"
)

CONTRACT_HASH_PATH = (
    CONTRACT_PATH.with_suffix(".sha256")
)

DOC_PATH = (
    PROJECT_ROOT
    / "docs"
    / "STAGE_31_SMOKE_OVERFIT_FREEZE.md"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def canonical_sha256(
    payload: object,
) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(
        serialized
    ).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(
        Path(path).read_text(
            encoding="utf-8-sig"
        )
    )


def write_json(
    path: Path,
    payload: object,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def verify_config_manifest() -> dict:
    manifest = read_json(
        CONFIG_HASH_PATH
    )

    if manifest.get("stage") != 31:
        raise RuntimeError(
            "Stage-31 config manifest "
            "has the wrong stage."
        )

    if manifest.get(
        "config_count"
    ) != 11:
        raise RuntimeError(
            "Expected exactly 11 frozen "
            "Stage-31 configs."
        )

    for entry in manifest["configs"]:
        path = (
            PROJECT_ROOT
            / entry["path"]
        )

        if not path.is_file():
            raise RuntimeError(
                "Missing frozen config: "
                f"{entry['path']}."
            )

        if (
            sha256_file(path)
            != entry["sha256"]
        ):
            raise RuntimeError(
                "Frozen config hash mismatch: "
                f"{entry['path']}."
            )

    return manifest


def normalization_fingerprint(
    stats: object,
) -> str:
    payload = {
        "mean": np.asarray(
            getattr(
                stats,
                "mean",
            ),
            dtype=np.float64,
        ).tolist(),
        "std": np.asarray(
            getattr(
                stats,
                "std",
            ),
            dtype=np.float64,
        ).tolist(),
        "frame_count": int(
            getattr(
                stats,
                "frame_count",
            )
        ),
        "recording_count": int(
            getattr(
                stats,
                "recording_count",
            )
        ),
        "feature_width": int(
            getattr(
                stats,
                "feature_width",
            )
        ),
        "sample_ids_sha256": getattr(
            stats,
            "sample_ids_sha256",
        ),
    }

    return canonical_sha256(
        payload
    )


def select_fixture(
    store: RawCacheStore,
    targets: TargetCatalog,
    train_ids: list[str],
) -> tuple[
    list[str],
    list[int],
]:
    probe_dataset = ASLSequenceDataset(
        store,
        targets,
        train_ids,
        normalization_stats=None,
    )

    candidates = []

    for (
        index,
        expected_sample_id,
    ) in enumerate(
        train_ids
    ):
        item = probe_dataset[
            index
        ]

        observed_sample_id = str(
            item["sample_id"]
        )

        if (
            observed_sample_id
            != expected_sample_id
        ):
            raise RuntimeError(
                "Stage-19 dataset changed "
                "sample ordering."
            )

        input_length = int(
            item["input_length"]
        )

        if input_length < 36:
            raise RuntimeError(
                "A Fold-1 training sample "
                "is shorter than 36."
            )

        candidates.append(
            (
                input_length,
                observed_sample_id,
            )
        )

    selected = sorted(
        candidates,
        key=lambda row: (
            row[0],
            row[1],
        ),
    )[:20]

    if len(selected) != 20:
        raise RuntimeError(
            "Could not select the "
            "20-sample fixture."
        )

    return (
        [
            sample_id
            for _, sample_id in selected
        ],
        [
            input_length
            for input_length, _ in selected
        ],
    )


def clear_after_attempt() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    print("=" * 78)
    print(
        "STAGE 31 — FIVE-MODEL SMOKE, "
        "TINY OVERFIT AND FREEZE"
    )
    print("=" * 78)

    config_manifest = (
        verify_config_manifest()
    )

    smoke_config = read_json(
        SMOKE_CONFIG_PATH
    )["protocol"]

    project_seed = int(
        smoke_config["seed"]
    )

    fixture_size = int(
        smoke_config[
            "fixture_size"
        ]
    )

    smoke_batch_size = int(
        smoke_config[
            "smoke_batch_size"
        ]
    )

    microbatch_candidates = [
        int(value)
        for value in smoke_config[
            "overfit_microbatch_candidates_"
            "on_verified_oom"
        ]
    ]

    minimum_epochs = int(
        smoke_config[
            "minimum_epochs"
        ]
    )

    maximum_epochs = int(
        smoke_config[
            "maximum_epochs"
        ]
    )

    minimum_relative_decrease = float(
        smoke_config[
            "minimum_relative_loss_decrease"
        ]
    )

    if fixture_size != 20:
        raise RuntimeError(
            "Stage-31 fixture size "
            "must remain 20."
        )

    if microbatch_candidates != [
        4,
        2,
        1,
    ]:
        raise RuntimeError(
            "Stage-31 OOM fallback "
            "order changed."
        )

    if smoke_config["device"] != "cpu":
        raise RuntimeError(
            "Stage-31 verification device "
            "must remain CPU."
        )

    device = torch.device(
        "cpu"
    )

    catalog = load_model_catalog(
        CATALOG_PATH
    )

    store = RawCacheStore(
        RAW_CACHE_PATH,
        RAW_INDEX_PATH,
    )

    targets = TargetCatalog(
        TARGET_PATH
    )

    (
        fold1_train,
        fold1_test,
    ) = load_known_signer_outer_fold(
        KNOWN_SPLIT_PATH,
        1,
    )

    train_ids = [
        str(value)
        for value in fold1_train
    ]

    test_ids = [
        str(value)
        for value in fold1_test
    ]

    if (
        not train_ids
        or not test_ids
    ):
        raise RuntimeError(
            "Known-signer Fold 1 is empty."
        )

    if (
        set(train_ids)
        & set(test_ids)
    ):
        raise RuntimeError(
            "Known-signer Fold 1 "
            "train/test overlap."
        )

    print()
    print("COMMON DATA FIXTURE")
    print("-" * 78)

    print(
        "Fold-1 training recordings:",
        len(train_ids),
    )

    print(
        "Fold-1 test recordings "
        "(IDs only; not evaluated):",
        len(test_ids),
    )

    (
        selected_ids,
        selected_lengths,
    ) = select_fixture(
        store,
        targets,
        train_ids,
    )

    normalization_stats = (
        fit_normalization_for_sample_ids(
            store,
            train_ids,
        )
    )

    normalization_hash = (
        normalization_fingerprint(
            normalization_stats
        )
    )

    tiny_dataset = ASLSequenceDataset(
        store,
        targets,
        selected_ids,
        normalization_stats=(
            normalization_stats
        ),
    )

    items = [
        tiny_dataset[index]
        for index in range(
            len(tiny_dataset)
        )
    ]

    observed_ids = [
        str(
            item["sample_id"]
        )
        for item in items
    ]

    observed_lengths = [
        int(
            item["input_length"]
        )
        for item in items
    ]

    if observed_ids != selected_ids:
        raise RuntimeError(
            "Tiny fixture sample "
            "order changed."
        )

    if (
        observed_lengths
        != selected_lengths
    ):
        raise RuntimeError(
            "Tiny fixture input "
            "lengths changed."
        )

    fixture = {
        "schema_version": 1,
        "project_generation": (
            "fresh_start_2026_v1"
        ),
        "stage": 31,
        "fixture_id": (
            "known_fold01_train_"
            "shortest20_v1"
        ),
        "scientific_use": (
            "technical_diagnostic_only"
        ),
        "source_split": (
            "known_signer_fold01_"
            "training_only"
        ),
        "selection_rule": (
            "input_length_ascending_"
            "then_sample_id"
        ),
        "sample_count": len(
            selected_ids
        ),
        "sample_ids": selected_ids,
        "input_lengths": (
            selected_lengths
        ),
        "sample_ids_sha256": (
            canonical_sha256(
                selected_ids
            )
        ),
        "all_samples_in_fold01_train": (
            set(
                selected_ids
            ).issubset(
                train_ids
            )
        ),
        "samples_in_fold01_test": len(
            set(selected_ids)
            & set(test_ids)
        ),
        "normalization": {
            "fit_partition": (
                "entire_known_signer_"
                "fold01_training_partition"
            ),
            "fit_recordings": int(
                normalization_stats
                .recording_count
            ),
            "fit_frames": int(
                normalization_stats
                .frame_count
            ),
            "feature_width": int(
                normalization_stats
                .feature_width
            ),
            "fingerprint_only_sha256": (
                normalization_hash
            ),
            "statistics_saved_for_reuse": (
                False
            ),
        },
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json(
        FIXTURE_PATH,
        fixture,
    )

    FIXTURE_HASH_PATH.write_text(
        sha256_file(
            FIXTURE_PATH
        )
        + "  "
        + FIXTURE_PATH.name
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(
        "Fixture:",
        fixture["fixture_id"],
    )

    print(
        "Samples:",
        len(selected_ids),
    )

    print(
        "Input length range:",
        min(selected_lengths),
        "to",
        max(selected_lengths),
    )

    print(
        "Normalization fit recordings:",
        normalization_stats
        .recording_count,
    )

    print(
        "Same ordered fixture "
        "for all models:",
        True,
    )

    model_results = {}
    log_rows = []

    print()
    print(
        "MODEL SMOKE AND TINY OVERFIT"
    )
    print("-" * 78)

    for model_id in MODEL_IDS:
        specification = (
            catalog[model_id]
        )

        fallback_events = []
        completed = None

        start_time = (
            time.perf_counter()
        )

        for (
            microbatch_size
        ) in microbatch_candidates:
            smoke_model = None
            overfit_model = None

            try:
                set_deterministic_seed(
                    project_seed
                )

                smoke_model = (
                    build_model(
                        model_id
                    )
                )

                parameter_count = (
                    validate_model_parameter_count(
                        model_id,
                        smoke_model,
                    )
                )

                current_smoke_size = min(
                    smoke_batch_size,
                    microbatch_size,
                )

                smoke = one_batch_smoke(
                    smoke_model,
                    items[
                        :current_smoke_size
                    ],
                    device=device,
                )

                del smoke_model
                smoke_model = None

                clear_after_attempt()

                # The smoke model and its
                # BatchNorm state/gradients are
                # deliberately not reused.
                set_deterministic_seed(
                    project_seed
                )

                overfit_model = (
                    build_model(
                        model_id
                    )
                )

                validate_model_parameter_count(
                    model_id,
                    overfit_model,
                )

                overfit = tiny_overfit(
                    overfit_model,
                    items,
                    specification,
                    batch_size=(
                        microbatch_size
                    ),
                    minimum_epochs=(
                        minimum_epochs
                    ),
                    maximum_epochs=(
                        maximum_epochs
                    ),
                    minimum_relative_decrease=(
                        minimum_relative_decrease
                    ),
                    device=device,
                )

                if not overfit["passed"]:
                    raise RuntimeError(
                        f"{model_id} did not "
                        "meet the frozen "
                        "loss-decrease rule."
                    )

                completed = {
                    "model_id": model_id,
                    "fixture_id": (
                        fixture[
                            "fixture_id"
                        ]
                    ),
                    "sample_ids_sha256": (
                        fixture[
                            "sample_ids_sha256"
                        ]
                    ),
                    "parameter_count": (
                        parameter_count
                    ),
                    "expected_parameter_count": (
                        EXPECTED_TRAINABLE_PARAMETERS[
                            model_id
                        ]
                    ),
                    "optimizer": (
                        specification
                        .optimizer
                    ),
                    "learning_rate": (
                        specification
                        .learning_rate
                    ),
                    "weight_decay": (
                        specification
                        .weight_decay
                    ),
                    "gradient_clip_norm": (
                        specification
                        .gradient_clip_norm
                    ),
                    "seed": project_seed,
                    "device": str(
                        device
                    ),
                    "microbatch_size": (
                        microbatch_size
                    ),
                    "oom_fallback_used": bool(
                        fallback_events
                    ),
                    "oom_fallback_events": (
                        fallback_events
                    ),
                    "one_batch_smoke": (
                        smoke
                    ),
                    "tiny_overfit": (
                        overfit
                    ),
                    "checkpoint_saved": False,
                    "trained_weights_reusable": (
                        False
                    ),
                    "elapsed_seconds": (
                        time.perf_counter()
                        - start_time
                    ),
                    "passed": bool(
                        smoke["passed"]
                        and overfit["passed"]
                    ),
                }

                break

            except (
                RuntimeError,
                MemoryError,
            ) as error:
                verified_oom = (
                    isinstance(
                        error,
                        MemoryError,
                    )
                    or is_out_of_memory_error(
                        error
                    )
                )

                if not verified_oom:
                    raise

                fallback_events.append(
                    {
                        "failed_microbatch_size": (
                            microbatch_size
                        ),
                        "reason": (
                            "verified_out_of_memory"
                        ),
                        "message": (
                            str(error)[:500]
                        ),
                    }
                )

                if (
                    microbatch_size
                    == microbatch_candidates[
                        -1
                    ]
                ):
                    raise RuntimeError(
                        f"{model_id} failed "
                        "even at microbatch "
                        "size 1."
                    ) from error

            finally:
                if smoke_model is not None:
                    del smoke_model

                if overfit_model is not None:
                    del overfit_model

                clear_after_attempt()

        if completed is None:
            raise RuntimeError(
                f"{model_id} did not "
                "complete Stage 31."
            )

        model_results[
            model_id
        ] = completed

        for row in completed[
            "tiny_overfit"
        ]["history"]:
            log_rows.append(
                {
                    "model_id": (
                        model_id
                    ),
                    "fixture_id": (
                        fixture[
                            "fixture_id"
                        ]
                    ),
                    "microbatch_size": (
                        completed[
                            "microbatch_size"
                        ]
                    ),
                    "epoch": row[
                        "epoch"
                    ],
                    "training_loss": (
                        row[
                            "training_loss"
                        ]
                    ),
                    "evaluation_loss": (
                        row[
                            "evaluation_loss"
                        ]
                    ),
                    "relative_decrease_"
                    "from_initial": (
                        row[
                            "relative_decrease_"
                            "from_initial"
                        ]
                    ),
                    "maximum_preclip_"
                    "gradient_norm": (
                        row[
                            "maximum_preclip_"
                            "gradient_norm"
                        ]
                    ),
                }
            )

        print(
            model_id.upper(),
            "smoke=PASS",
            (
                "initial="
                f"{completed['tiny_overfit']['initial_evaluation_loss']:.6f}"
            ),
            (
                "best="
                f"{completed['tiny_overfit']['best_evaluation_loss']:.6f}"
            ),
            (
                "relative_decrease="
                f"{completed['tiny_overfit']['relative_decrease']:.6%}"
            ),
            (
                "epochs="
                f"{completed['tiny_overfit']['epochs_completed']}"
            ),
            (
                "microbatch="
                f"{completed['microbatch_size']}"
            ),
        )

    all_passed = (
        list(
            model_results
        ) == list(
            MODEL_IDS
        )
        and all(
            result["passed"]
            for result in (
                model_results.values()
            )
        )
        and all(
            result[
                "sample_ids_sha256"
            ]
            == fixture[
                "sample_ids_sha256"
            ]
            for result in (
                model_results.values()
            )
        )
    )

    results = {
        "schema_version": 1,
        "project_generation": (
            "fresh_start_2026_v1"
        ),
        "stage": 31,
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "scientific_use": (
            "technical_diagnostic_only_"
            "not_reportable"
        ),
        "fixture_path": (
            FIXTURE_PATH.relative_to(
                PROJECT_ROOT
            ).as_posix()
        ),
        "fixture_sha256": (
            sha256_file(
                FIXTURE_PATH
            )
        ),
        "model_order": list(
            MODEL_IDS
        ),
        "models": model_results,
        "all_models_passed": (
            all_passed
        ),
        "outer_scores_produced": False,
        "checkpoints_produced": False,
    }

    write_json(
        RESULTS_PATH,
        results,
    )

    with LOG_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fieldnames = [
            "model_id",
            "fixture_id",
            "microbatch_size",
            "epoch",
            "training_loss",
            "evaluation_loss",
            (
                "relative_decrease_"
                "from_initial"
            ),
            (
                "maximum_preclip_"
                "gradient_norm"
            ),
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            log_rows
        )

    environment = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pytorch": torch.__version__,
        "cuda_available": (
            torch.cuda.is_available()
        ),
        "verification_device": str(
            device
        ),
        "torch_threads": (
            torch.get_num_threads()
        ),
    }

    write_json(
        ENVIRONMENT_PATH,
        environment,
    )

    contract = {
        "schema_version": 1,
        "project_generation": (
            "fresh_start_2026_v1"
        ),
        "stage": 31,
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "identity": {
            "name": (
                "five_model_smoke_"
                "tiny_overfit_and_freeze"
            ),
            "model_ids": list(
                MODEL_IDS
            ),
            "seed": project_seed,
            "scientific_use": (
                "technical_diagnostic_only"
            ),
        },
        "config_freeze": {
            "config_count": (
                config_manifest[
                    "config_count"
                ]
            ),
            "manifest_path": (
                CONFIG_HASH_PATH
                .relative_to(
                    PROJECT_ROOT
                )
                .as_posix()
            ),
            "manifest_sha256": (
                sha256_file(
                    CONFIG_HASH_PATH
                )
            ),
            "required_git_tag": (
                "stage31-freeze-v1"
            ),
            "tag_state_at_build": (
                "pending_post_commit"
            ),
        },
        "fixture": {
            "fixture_id": (
                fixture[
                    "fixture_id"
                ]
            ),
            "sample_count": (
                fixture[
                    "sample_count"
                ]
            ),
            "sample_ids_sha256": (
                fixture[
                    "sample_ids_sha256"
                ]
            ),
            "fixture_sha256": (
                sha256_file(
                    FIXTURE_PATH
                )
            ),
            "same_ordered_samples_for_"
            "all_models": True,
            "all_samples_in_fold01_train": (
                fixture[
                    "all_samples_in_fold01_train"
                ]
            ),
            "samples_in_fold01_test": (
                fixture[
                    "samples_in_fold01_test"
                ]
            ),
            "normalization_fit_recordings": (
                fixture[
                    "normalization"
                ][
                    "fit_recordings"
                ]
            ),
            "normalization_statistics_saved_"
            "for_reuse": False,
        },
        "smoke": {
            "models_attempted": len(
                model_results
            ),
            "models_passed": sum(
                result[
                    "one_batch_smoke"
                ][
                    "passed"
                ]
                for result in (
                    model_results.values()
                )
            ),
            "finite_loss_all_models": all(
                result[
                    "one_batch_smoke"
                ][
                    "loss_finite"
                ]
                for result in (
                    model_results.values()
                )
            ),
            "missing_gradients_total": sum(
                result[
                    "one_batch_smoke"
                ][
                    "missing_gradients"
                ]
                for result in (
                    model_results.values()
                )
            ),
            "nonfinite_gradients_total": sum(
                result[
                    "one_batch_smoke"
                ][
                    "nonfinite_gradients"
                ]
                for result in (
                    model_results.values()
                )
            ),
        },
        "tiny_overfit": {
            "models_passed": sum(
                result[
                    "tiny_overfit"
                ][
                    "passed"
                ]
                for result in (
                    model_results.values()
                )
            ),
            "loss_decreased_all_models": all(
                result[
                    "tiny_overfit"
                ][
                    "best_evaluation_loss"
                ]
                < result[
                    "tiny_overfit"
                ][
                    "initial_evaluation_loss"
                ]
                for result in (
                    model_results.values()
                )
            ),
            "minimum_relative_decrease": (
                minimum_relative_decrease
            ),
            "oom_fallback_models": [
                model_id
                for (
                    model_id,
                    result,
                ) in model_results.items()
                if result[
                    "oom_fallback_used"
                ]
            ],
            "checkpoint_saved": False,
            "trained_weights_reusable": False,
        },
        "outputs": {
            "results_sha256": (
                sha256_file(
                    RESULTS_PATH
                )
            ),
            "log_sha256": (
                sha256_file(
                    LOG_PATH
                )
            ),
            "environment_sha256": (
                sha256_file(
                    ENVIRONMENT_PATH
                )
            ),
        },
        "outer_evaluation": {
            "scores_produced": False,
            "allowed_only_after_commit_and_tag": (
                True
            ),
        },
        "stage31_passed": all_passed,
    }

    write_json(
        CONTRACT_PATH,
        contract,
    )

    CONTRACT_HASH_PATH.write_text(
        sha256_file(
            CONTRACT_PATH
        )
        + "  "
        + CONTRACT_PATH.name
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    model_lines = []

    for (
        model_id,
        result,
    ) in model_results.items():
        overfit = result[
            "tiny_overfit"
        ]

        model_lines.append(
            "| "
            + model_id.upper()
            + " | "
            + str(
                result[
                    "one_batch_smoke"
                ][
                    "passed"
                ]
            )
            + " | "
            + (
                f"{overfit['initial_evaluation_loss']:.6f}"
            )
            + " | "
            + (
                f"{overfit['best_evaluation_loss']:.6f}"
            )
            + " | "
            + (
                f"{overfit['relative_decrease']:.6%}"
            )
            + " | "
            + str(
                overfit[
                    "epochs_completed"
                ]
            )
            + " | "
            + str(
                result[
                    "microbatch_size"
                ]
            )
            + " |"
        )

    markdown = f"""# Stage 31 — Smoke, Tiny Overfit and Freeze

## Status

Stage 31 passed: **{all_passed}**

This stage is a technical integration gate. It produces no outer-test
score, saves no model checkpoint, and its trained weights must never
be reused.

## Common fixture

- Source: known-signer Fold-1 training partition only
- Selection: 20 shortest recordings, ties by sample ID
- Same ordered sample IDs for all five models: yes
- Normalization: fit on the entire Fold-1 training partition only
- Saved reusable normalization statistics: no
- Fixture SHA-256: `{sha256_file(FIXTURE_PATH)}`

## Results

| Model | Smoke | Initial loss | Best loss | Relative decrease | Epochs | Microbatch |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(model_lines)}

## Frozen rule

Every model must complete a finite forward/backward pass with no
missing or non-finite gradients and must reduce deterministic
evaluation loss by at least {minimum_relative_decrease:.4%}. Batch
fallback is allowed only after a verified out-of-memory error; the
model is then rebuilt from the frozen seed.

## Configuration and code boundary

Eleven JSON-compatible YAML files are hashed in
`artifacts/config/stage31_config_hashes.json`. After verification,
commit this stage and create annotated tag `stage31-freeze-v1`.
No Stage-32 outer run may start before that tag points to the
Stage-31 commit and the worktree is clean.
"""

    DOC_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    DOC_PATH.write_text(
        markdown,
        encoding="utf-8",
        newline="\n",
    )

    print()
    print("INTEGRATION SUMMARY")
    print("-" * 78)

    print(
        "Models attempted / passed:",
        len(model_results),
        "/",
        sum(
            result["passed"]
            for result in (
                model_results.values()
            )
        ),
    )

    print(
        "Missing gradients total:",
        contract["smoke"][
            "missing_gradients_total"
        ],
    )

    print(
        "Non-finite gradients total:",
        contract["smoke"][
            "nonfinite_gradients_total"
        ],
    )

    print(
        "Loss decreased all models:",
        contract["tiny_overfit"][
            "loss_decreased_all_models"
        ],
    )

    print(
        "OOM fallback models:",
        contract["tiny_overfit"][
            "oom_fallback_models"
        ],
    )

    print(
        "Outer scores produced:",
        False,
    )

    print(
        "Checkpoints produced:",
        False,
    )

    print()
    print("OUTPUTS")
    print("-" * 78)

    for path in (
        FIXTURE_PATH,
        FIXTURE_HASH_PATH,
        RESULTS_PATH,
        LOG_PATH,
        ENVIRONMENT_PATH,
        CONTRACT_PATH,
        CONTRACT_HASH_PATH,
        DOC_PATH,
    ):
        print(
            "Saved:",
            path.relative_to(
                PROJECT_ROOT
            ),
        )

    print()

    if not all_passed:
        raise RuntimeError(
            "Stage 31 integration checks failed."
        )

    print(
        "FINAL STATUS: "
        "STAGE 31 BUILD PASSED"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
