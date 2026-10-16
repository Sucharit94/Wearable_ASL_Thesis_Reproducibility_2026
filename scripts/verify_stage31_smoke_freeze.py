from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_IDS = (
    "g1",
    "c1",
    "c2",
    "c3",
    "c4",
)

EXPECTED_PARAMETERS = {
    "g1": 4_791_688,
    "c1": 392_488,
    "c2": 656_168,
    "c3": 237_352,
    "c4": 348_604,
}

CONFIG_MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "config"
    / "stage31_config_hashes.json"
)

CONFIG_MANIFEST_HASH_PATH = (
    CONFIG_MANIFEST_PATH.with_suffix(
        ".sha256"
    )
)

FIXTURE_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "training"
    / "stage31_tiny_overfit_samples.json"
)

FIXTURE_HASH_PATH = (
    FIXTURE_PATH.with_suffix(
        ".sha256"
    )
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "training"
    / "stage31_smoke_overfit_results.json"
)

LOG_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "training"
    / "stage31_smoke_overfit_log.csv"
)

ENVIRONMENT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "training"
    / "stage31_environment.json"
)

CONTRACT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "training"
    / "stage31_smoke_freeze_contract.json"
)

CONTRACT_HASH_PATH = (
    CONTRACT_PATH.with_suffix(
        ".sha256"
    )
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


def read_json(path: Path) -> dict:
    return json.loads(
        Path(path).read_text(
            encoding="utf-8-sig"
        )
    )


def read_recorded_hash(
    path: Path,
) -> str:
    return (
        Path(path)
        .read_text(
            encoding="utf-8-sig"
        )
        .strip()
        .split()[0]
        .lower()
    )


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(
            message
        )


def main() -> None:
    print("=" * 78)
    print(
        "STAGE 31 — SMOKE, TINY OVERFIT "
        "AND CONFIG FREEZE VERIFIER"
    )
    print("=" * 78)

    required_paths = (
        PROJECT_ROOT
        / "src"
        / "models"
        / "factory.py",

        PROJECT_ROOT
        / "src"
        / "training"
        / "smoke_overfit.py",

        PROJECT_ROOT
        / "scripts"
        / "build_stage31_configs.py",

        PROJECT_ROOT
        / "scripts"
        / "run_stage31_smoke_overfit.py",

        PROJECT_ROOT
        / "scripts"
        / "verify_stage31_smoke_freeze.py",

        PROJECT_ROOT
        / "tests"
        / "test_stage31_smoke_freeze.py",

        CONFIG_MANIFEST_PATH,
        CONFIG_MANIFEST_HASH_PATH,
        FIXTURE_PATH,
        FIXTURE_HASH_PATH,
        RESULTS_PATH,
        LOG_PATH,
        ENVIRONMENT_PATH,
        CONTRACT_PATH,
        CONTRACT_HASH_PATH,
        DOC_PATH,
    )

    for path in required_paths:
        require(
            path.is_file(),
            "Missing Stage-31 file: "
            f"{path}.",
        )

    manifest = read_json(
        CONFIG_MANIFEST_PATH
    )

    recorded_manifest_hash = (
        read_recorded_hash(
            CONFIG_MANIFEST_HASH_PATH
        )
    )

    actual_manifest_hash = (
        sha256_file(
            CONFIG_MANIFEST_PATH
        )
    )

    require(
        recorded_manifest_hash
        == actual_manifest_hash,
        "Config manifest hash mismatch.",
    )

    require(
        manifest["stage"] == 31,
        "Config manifest stage changed.",
    )

    require(
        manifest["config_count"] == 11,
        "Frozen config count changed.",
    )

    config_hash_failures = []

    for entry in manifest["configs"]:
        path = (
            PROJECT_ROOT
            / entry["path"]
        )

        if (
            not path.is_file()
            or sha256_file(path)
            != entry["sha256"]
        ):
            config_hash_failures.append(
                entry["path"]
            )

    require(
        not config_hash_failures,
        "Config hash failures: "
        f"{config_hash_failures}.",
    )

    data_config = read_json(
        PROJECT_ROOT
        / "configs"
        / "data.yaml"
    )

    known_config = read_json(
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "known_5fold.yaml"
    )

    smoke_config = read_json(
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "stage31_smoke.yaml"
    )

    require(
        data_config["dataset"][
            "recordings"
        ] == 7400,
        "Recording count changed.",
    )

    require(
        data_config["dataset"][
            "feature_width"
        ] == 38,
        "Feature width changed.",
    )

    require(
        data_config["preprocessing"][
            "ctc_blank_id"
        ] == 0,
        "Blank ID changed.",
    )

    require(
        data_config["preprocessing"][
            "target_padding_value"
        ] == -1,
        "Target padding changed.",
    )

    require(
        known_config["protocol"][
            "outer_folds"
        ] == 5,
        "Outer-fold count changed.",
    )

    require(
        known_config["protocol"][
            "inner_validation_fraction"
        ] == 0.10,
        "Inner fraction changed.",
    )

    require(
        known_config[
            "shared_recognition_policy"
        ][
            "test_controls_training"
        ] is False,
        "Test leakage policy changed.",
    )

    require(
        smoke_config["protocol"][
            "fixture_size"
        ] == 20,
        "Fixture size changed.",
    )

    require(
        smoke_config["protocol"][
            "checkpoint_saved"
        ] is False,
        "Smoke checkpoint policy changed.",
    )

    for model_id in MODEL_IDS:
        config = read_json(
            PROJECT_ROOT
            / "configs"
            / "models"
            / f"{model_id}.yaml"
        )

        require(
            config["model_id"]
            == model_id,
            f"{model_id} config "
            "identity changed.",
        )

        require(
            config["source"]["sha256"]
            == sha256_file(
                PROJECT_ROOT
                / config["source"]["path"]
            ),
            f"{model_id} source "
            "hash mismatch.",
        )

        require(
            config["interface"][
                "output_logits"
            ]
            == (
                "B_x_Tout_x_456_raw_logits"
            ),
            f"{model_id} interface changed.",
        )

    fixture = read_json(
        FIXTURE_PATH
    )

    require(
        read_recorded_hash(
            FIXTURE_HASH_PATH
        )
        == sha256_file(
            FIXTURE_PATH
        ),
        "Fixture hash mismatch.",
    )

    require(
        fixture["stage"] == 31,
        "Fixture stage changed.",
    )

    require(
        fixture["sample_count"] == 20,
        "Fixture sample count changed.",
    )

    require(
        len(
            fixture["sample_ids"]
        ) == 20,
        "Fixture sample IDs missing.",
    )

    require(
        len(
            set(
                fixture["sample_ids"]
            )
        ) == 20,
        "Fixture sample IDs "
        "are not unique.",
    )

    require(
        len(
            fixture["input_lengths"]
        ) == 20,
        "Fixture lengths missing.",
    )

    require(
        fixture["input_lengths"]
        == sorted(
            fixture["input_lengths"]
        ),
        "Fixture is not length ordered.",
    )

    require(
        min(
            fixture["input_lengths"]
        ) >= 36,
        "Fixture contains a "
        "too-short recording.",
    )

    require(
        fixture[
            "all_samples_in_fold01_train"
        ] is True,
        "Fixture escaped Fold-1 training.",
    )

    require(
        fixture[
            "samples_in_fold01_test"
        ] == 0,
        "Fixture contains Fold-1 test data.",
    )

    require(
        fixture["normalization"][
            "statistics_saved_for_reuse"
        ] is False,
        "Transient normalization "
        "became reusable.",
    )

    results = read_json(
        RESULTS_PATH
    )

    require(
        results["stage"] == 31,
        "Results stage changed.",
    )

    require(
        results["model_order"]
        == list(MODEL_IDS),
        "Model execution order changed.",
    )

    require(
        set(
            results["models"]
        ) == set(
            MODEL_IDS
        ),
        "Result model IDs changed.",
    )

    require(
        results["fixture_sha256"]
        == sha256_file(
            FIXTURE_PATH
        ),
        "Results fixture hash mismatch.",
    )

    require(
        results[
            "all_models_passed"
        ] is True,
        "Not every model passed.",
    )

    require(
        results[
            "outer_scores_produced"
        ] is False,
        "Stage 31 produced outer scores.",
    )

    require(
        results[
            "checkpoints_produced"
        ] is False,
        "Stage 31 produced checkpoints.",
    )

    print()
    print("CONFIG FREEZE")
    print("-" * 78)

    print(
        "Config files:",
        manifest["config_count"],
    )

    print(
        "Config hash failures:",
        len(config_hash_failures),
    )

    print(
        "Config manifest hash match:",
        recorded_manifest_hash
        == actual_manifest_hash,
    )

    print()
    print("COMMON FIXTURE")
    print("-" * 78)

    print(
        "Fixture ID:",
        fixture["fixture_id"],
    )

    print(
        "Samples:",
        fixture["sample_count"],
    )

    print(
        "Input length range:",
        min(
            fixture["input_lengths"]
        ),
        "to",
        max(
            fixture["input_lengths"]
        ),
    )

    print(
        "Fold-1 test samples used:",
        fixture[
            "samples_in_fold01_test"
        ],
    )

    print(
        "Reusable normalization saved:",
        fixture["normalization"][
            "statistics_saved_for_reuse"
        ],
    )

    print()
    print("FIVE-MODEL RESULTS")
    print("-" * 78)

    for model_id in MODEL_IDS:
        result = (
            results["models"][
                model_id
            ]
        )

        smoke = result[
            "one_batch_smoke"
        ]

        overfit = result[
            "tiny_overfit"
        ]

        require(
            result["parameter_count"]
            == EXPECTED_PARAMETERS[
                model_id
            ],
            f"{model_id} parameter "
            "count changed.",
        )

        require(
            result["sample_ids_sha256"]
            == fixture[
                "sample_ids_sha256"
            ],
            f"{model_id} used a "
            "different fixture.",
        )

        require(
            result[
                "microbatch_size"
            ] in (
                1,
                2,
                4,
            ),
            f"{model_id} used an "
            "invalid microbatch.",
        )

        require(
            smoke["passed"] is True,
            f"{model_id} smoke failed.",
        )

        require(
            smoke["loss_finite"] is True
            and math.isfinite(
                smoke["loss"]
            ),
            f"{model_id} smoke loss "
            "is not finite.",
        )

        require(
            smoke[
                "missing_gradients"
            ] == 0,
            f"{model_id} has "
            "missing gradients.",
        )

        require(
            smoke[
                "nonfinite_gradients"
            ] == 0,
            f"{model_id} has "
            "non-finite gradients.",
        )

        require(
            smoke[
                "gradient_global_norm"
            ] > 0.0,
            f"{model_id} has "
            "zero gradient norm.",
        )

        require(
            overfit["passed"] is True,
            f"{model_id} tiny-overfit failed.",
        )

        require(
            math.isfinite(
                overfit[
                    "initial_evaluation_loss"
                ]
            ),
            f"{model_id} initial "
            "loss invalid.",
        )

        require(
            math.isfinite(
                overfit[
                    "best_evaluation_loss"
                ]
            ),
            f"{model_id} best "
            "loss invalid.",
        )

        require(
            overfit[
                "best_evaluation_loss"
            ]
            < overfit[
                "initial_evaluation_loss"
            ],
            f"{model_id} loss "
            "did not decrease.",
        )

        require(
            overfit[
                "relative_decrease"
            ]
            >= overfit[
                "minimum_relative_"
                "decrease_required"
            ],
            f"{model_id} loss decrease "
            "below threshold.",
        )

        require(
            result[
                "checkpoint_saved"
            ] is False,
            f"{model_id} saved a "
            "smoke checkpoint.",
        )

        require(
            result[
                "trained_weights_reusable"
            ] is False,
            f"{model_id} marked smoke "
            "weights reusable.",
        )

        print(
            model_id.upper(),
            "smoke=PASS",
            (
                "initial="
                f"{overfit['initial_evaluation_loss']:.6f}"
            ),
            (
                "best="
                f"{overfit['best_evaluation_loss']:.6f}"
            ),
            (
                "decrease="
                f"{overfit['relative_decrease']:.6%}"
            ),
            (
                "epochs="
                f"{overfit['epochs_completed']}"
            ),
            (
                "microbatch="
                f"{result['microbatch_size']}"
            ),
        )

    contract = read_json(
        CONTRACT_PATH
    )

    require(
        read_recorded_hash(
            CONTRACT_HASH_PATH
        )
        == sha256_file(
            CONTRACT_PATH
        ),
        "Stage-31 contract hash mismatch.",
    )

    require(
        contract["stage"] == 31,
        "Contract stage changed.",
    )

    require(
        contract["config_freeze"][
            "manifest_sha256"
        ] == actual_manifest_hash,
        "Contract config hash mismatch.",
    )

    require(
        contract["fixture"][
            "fixture_sha256"
        ] == sha256_file(
            FIXTURE_PATH
        ),
        "Contract fixture hash mismatch.",
    )

    require(
        contract["outputs"][
            "results_sha256"
        ] == sha256_file(
            RESULTS_PATH
        ),
        "Results hash mismatch.",
    )

    require(
        contract["outputs"][
            "log_sha256"
        ] == sha256_file(
            LOG_PATH
        ),
        "Log hash mismatch.",
    )

    require(
        contract["outputs"][
            "environment_sha256"
        ] == sha256_file(
            ENVIRONMENT_PATH
        ),
        "Environment hash mismatch.",
    )

    require(
        contract["smoke"][
            "models_passed"
        ] == 5,
        "Smoke pass count changed.",
    )

    require(
        contract["smoke"][
            "missing_gradients_total"
        ] == 0,
        "Missing gradients recorded.",
    )

    require(
        contract["smoke"][
            "nonfinite_gradients_total"
        ] == 0,
        "Non-finite gradients recorded.",
    )

    require(
        contract["tiny_overfit"][
            "models_passed"
        ] == 5,
        "Tiny-overfit pass count changed.",
    )

    require(
        contract["tiny_overfit"][
            "loss_decreased_all_models"
        ] is True,
        "Loss-decrease aggregate failed.",
    )

    require(
        contract["outer_evaluation"][
            "scores_produced"
        ] is False,
        "Outer scores were produced early.",
    )

    require(
        contract[
            "stage31_passed"
        ] is True,
        "Stage-31 contract is not passed.",
    )

    unexpected_checkpoints = list(
        CONTRACT_PATH.parent.glob(
            "stage31*.pt"
        )
    )

    require(
        not unexpected_checkpoints,
        "Unexpected Stage-31 "
        "checkpoint exists.",
    )

    print()
    print("FREEZE BOUNDARY")
    print("-" * 78)

    print(
        "Required Git tag:",
        contract["config_freeze"][
            "required_git_tag"
        ],
    )

    print(
        "Outer scores produced:",
        contract["outer_evaluation"][
            "scores_produced"
        ],
    )

    print(
        "Checkpoints produced:",
        contract["tiny_overfit"][
            "checkpoint_saved"
        ],
    )

    print(
        "Unexpected checkpoint files:",
        len(
            unexpected_checkpoints
        ),
    )

    print()
    print(
        "FINAL STATUS: "
        "STAGE 31 PASSED"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
