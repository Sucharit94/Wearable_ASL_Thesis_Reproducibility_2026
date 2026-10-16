from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.data.dataset import ASLSequenceDataset, RawCacheStore, TargetCatalog
from src.evaluation.metrics import score_batch
from src.training.known_signer_runner import (
    _append_parent_start_or_resume,
    _preflight_memory,
    _rows_for_evaluation,
    atomic_write_csv,
    atomic_write_jsonl,
    build_experiment_identity,
    canonical_json_sha256,
    canonical_sample_ids_sha256,
    ensure_external_artifact_root,
    evaluate_model,
    finalize_parent_run,
    gradient_accumulation_steps,
    is_out_of_memory_error,
    normalization_from_payload,
    physical_batch_candidates,
    read_csv_rows,
    read_recorded_sha256,
    run_full_refit,
    run_inner_selection,
    select_best_epoch,
    sha256_file,
    validate_runner_configuration as validate_stage32_runner_configuration,
    verify_t4_runtime,
)
from src.training.loso_plan import (
    HELD_OUT_SUBJECTS,
    LOSO_IDS,
    MODEL_IDS,
    REQUIRED_RUNNER_TAG,
)
from src.training.registry import (
    atomic_write_json,
    read_json,
    validate_status_transition,
)


PROJECT_GENERATION = "fresh_start_2026_v1"
STAGE33A_COMMIT = "010256a"
EXPECTED_ASSIGNMENT_ROWS = 22200
EXPECTED_COUNTS = {
    "loso_subject1": {
        "inner_train": 1260,
        "inner_validation": 140,
        "outer_test": 6000,
    },
    "loso_subject2": {
        "inner_train": 5760,
        "inner_validation": 640,
        "outer_test": 1000,
    },
    "loso_subject3": {
        "inner_train": 6300,
        "inner_validation": 700,
        "outer_test": 400,
    },
}
REGISTRY_PHASES = ("outer_evaluation", "inner_selection", "full_refit")
REGISTRY_FIELDS = (
    "registry_order",
    "evaluation_run_order",
    "registry_id",
    "experiment_id",
    "parent_experiment_id",
    "project_generation",
    "model_id",
    "loso_index",
    "loso_id",
    "held_out_subject",
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


@dataclass(frozen=True)
class LosoPartitions:
    loso_id: str
    loso_index: int
    held_out_subject: str
    inner_train: tuple[str, ...]
    inner_validation: tuple[str, ...]
    outer_test: tuple[str, ...]

    @property
    def outer_training_pool(self) -> tuple[str, ...]:
        return tuple(sorted((*self.inner_train, *self.inner_validation)))

    def to_summary(self) -> dict:
        return {
            "loso_id": self.loso_id,
            "loso_index": self.loso_index,
            "held_out_subject": self.held_out_subject,
            "inner_train": len(self.inner_train),
            "inner_validation": len(self.inner_validation),
            "outer_training_pool": len(self.outer_training_pool),
            "outer_test": len(self.outer_test),
            "held_out_training_leaks": 0,
            "all_unique": len(set((*self.outer_training_pool, *self.outer_test))) == 7400,
        }


def portable_sha256(path: str | Path) -> tuple[str, str, str]:
    raw = Path(path).read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    return (
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(lf).hexdigest(),
        hashlib.sha256(crlf).hexdigest(),
    )

def require_portable_recorded_hash(
    data_path: str | Path,
    hash_path: str | Path,
) -> str:
    recorded = read_recorded_sha256(hash_path)
    if recorded not in set(portable_sha256(data_path)):
        raise RuntimeError(f"Frozen hash mismatch for {Path(data_path).name}.")
    return recorded


def load_loso_partitions(
    assignment_path: str | Path,
    loso_id: str,
) -> LosoPartitions:
    if loso_id not in LOSO_IDS:
        raise ValueError("Unknown LOSO direction.")
    rows = read_csv_rows(assignment_path)
    if len(rows) != EXPECTED_ASSIGNMENT_ROWS:
        raise RuntimeError("Stage-33 assignment must contain 22,200 rows.")
    selected = [row for row in rows if row["loso_id"] == loso_id]
    if len(selected) != 7400:
        raise RuntimeError("Each LOSO direction must contain 7,400 rows.")

    held_out = HELD_OUT_SUBJECTS[loso_id]
    expected = EXPECTED_COUNTS[loso_id]
    by_partition = {name: [] for name in expected}
    groups: dict[str, set[str]] = {}
    for row in selected:
        partition = row["partition"]
        if partition not in by_partition:
            raise RuntimeError("Unknown Stage-33 partition.")
        by_partition[partition].append(row["sample_id"])
        groups.setdefault(row["split_group_id"], set()).add(partition)
        if (partition == "outer_test") != (row["stage14_role"] == "test"):
            raise RuntimeError("Stage-14/Stage-33 outer assignment mismatch.")
        if partition != "outer_test" and row["subject"] == held_out:
            raise RuntimeError("Held-out subject leaked into training.")
    for name, count in expected.items():
        if len(by_partition[name]) != count:
            raise RuntimeError(f"{loso_id} {name} count changed.")
    if len({sid for values in by_partition.values() for sid in values}) != 7400:
        raise RuntimeError("LOSO direction has missing or duplicate samples.")
    if any(len(parts) != 1 for parts in groups.values()):
        raise RuntimeError("An exact-duplicate group crosses LOSO partitions.")
    return LosoPartitions(
        loso_id=loso_id,
        loso_index=LOSO_IDS.index(loso_id) + 1,
        held_out_subject=held_out,
        inner_train=tuple(sorted(by_partition["inner_train"])),
        inner_validation=tuple(sorted(by_partition["inner_validation"])),
        outer_test=tuple(sorted(by_partition["outer_test"])),
    )


def validate_runner_configuration(payload: Mapping[str, object]) -> dict:
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported Stage-33 runner schema.")
    if payload.get("project_generation") != PROJECT_GENERATION:
        raise ValueError("Wrong project generation.")
    if int(payload.get("stage", -1)) != 33 or payload.get("substage") != "B":
        raise ValueError("Runner configuration must describe Stage 33B.")
    if payload["freeze"]["required_annotated_git_tag"] != REQUIRED_RUNNER_TAG:
        raise ValueError("Stage-33 runner tag changed.")
    precision = payload["precision"]
    if precision["mode"] != "float32" or precision["automatic_mixed_precision"] is not False:
        raise ValueError("Stage-33 execution must use float32 without AMP.")
    if tuple(precision["applies_to_models"]) != MODEL_IDS:
        raise ValueError("Precision policy must cover all five models.")
    selection = payload["selection"]
    if int(selection["compact_early_stopping_minimum_epoch"]) != 15:
        raise ValueError("Compact minimum selection epoch changed.")
    if selection["scheduler_monitor"] != "validation_ctc_loss":
        raise ValueError("Selection scheduler monitor changed.")
    expected_slices = [
        "loso_all", "template_supported", "token_supported",
        "token_unsupported", "support_statistics",
    ]
    if payload["reporting"]["report_slices"] != expected_slices:
        raise ValueError("LOSO support-reporting views changed.")
    if payload["reporting"]["support_slices_train_separate_models"] is not False:
        raise ValueError("Support views cannot train separate models.")
    return dict(payload)


def _projection(row: Mapping[str, object]) -> dict:
    return {
        key: str(row[key])
        for key in REGISTRY_FIELDS
        if key not in {"code_commit", "status"}
    }


def initialize_or_validate_live_registry(
    template_path: str | Path,
    live_path: str | Path,
    *,
    code_commit: str,
) -> list[dict[str, str]]:
    template = read_csv_rows(template_path)
    if len(template) != 45:
        raise RuntimeError("Stage-33 registry template must contain 45 rows.")
    if any(tuple(row) != REGISTRY_FIELDS for row in template):
        raise RuntimeError("Stage-33 registry column order changed.")
    live_path = Path(live_path)
    if not live_path.exists():
        live = []
        for source in template:
            row = dict(source)
            row["code_commit"] = code_commit
            row["status"] = "PLANNED"
            live.append(row)
        atomic_write_csv(live_path, live, REGISTRY_FIELDS)
        return live
    live = read_csv_rows(live_path)
    if len(live) != 45:
        raise RuntimeError("Live Stage-33 registry row count changed.")
    for expected, observed in zip(template, live, strict=True):
        if _projection(expected) != _projection(observed):
            raise RuntimeError("Live LOSO registry identity changed.")
        if observed["code_commit"] != code_commit:
            raise RuntimeError("Live LOSO registry belongs to another commit.")
    return live


def set_live_registry_status(
    live_path: str | Path,
    registry_id: str,
    next_status: str,
) -> dict[str, str]:
    rows = read_csv_rows(live_path)
    matches = [row for row in rows if row["registry_id"] == registry_id]
    if len(matches) != 1:
        raise RuntimeError("Expected one live registry row.")
    row = matches[0]
    validate_status_transition(row["status"], next_status)
    row["status"] = next_status
    atomic_write_csv(live_path, rows, REGISTRY_FIELDS)
    return dict(row)


def next_evaluation_row(rows: Sequence[Mapping[str, object]]) -> dict | None:
    parents = sorted(
        (row for row in rows if row["phase"] == "outer_evaluation"),
        key=lambda row: int(row["evaluation_run_order"]),
    )
    if len(parents) != 15:
        raise RuntimeError("Expected 15 LOSO outer-evaluation rows.")
    for row in parents:
        status = str(row["status"])
        if status in {"FAILED_SAFE", "INVALID"}:
            raise RuntimeError(f"LOSO matrix blocked by {row['registry_id']}.")
        if status != "COMPLETE":
            return dict(row)
    return None


def verify_git_freeze(project_root: str | Path) -> dict:
    project_root = Path(project_root).resolve()
    def git(*args: str, check: bool = True):
        return subprocess.run(
            ["git", *args], cwd=project_root, text=True,
            capture_output=True, check=check,
        )
    if git("rev-parse", "--is-inside-work-tree").stdout.strip() != "true":
        raise RuntimeError("Stage-33 execution requires a Git worktree.")
    tag_type = git("cat-file", "-t", f"refs/tags/{REQUIRED_RUNNER_TAG}").stdout.strip()
    if tag_type != "tag":
        raise RuntimeError(f"{REQUIRED_RUNNER_TAG} must be annotated.")
    tag_commit = git("rev-list", "-n", "1", REQUIRED_RUNNER_TAG).stdout.strip().lower()
    head = git("rev-parse", "HEAD").stdout.strip().lower()
    if tag_commit != head:
        raise RuntimeError("HEAD must equal the Stage-33 runner tag.")
    if git("merge-base", "--is-ancestor", STAGE33A_COMMIT, "HEAD", check=False).returncode:
        raise RuntimeError("Stage-33A commit is not an ancestor of HEAD.")
    status = git("status", "--porcelain").stdout.strip()
    if status:
        raise RuntimeError("Git worktree must be clean before LOSO training.")
    return {
        "runner_tag": REQUIRED_RUNNER_TAG,
        "runner_tag_annotated": True,
        "tag_commit": tag_commit,
        "head_commit": head,
        "tag_commit_equals_head": True,
        "stage33a_commit_is_ancestor": True,
        "working_tree_clean": True,
    }


def _empty_metrics() -> dict:
    return {
        "samples": 0,
        "reference_tokens": 0,
        "hypothesis_tokens": 0,
        "matches": 0,
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
        "total_edits": 0,
        "pooled_wer": None,
        "one_minus_wer": None,
        "macro_sample_wer": None,
        "sentence_errors": 0,
        "ser": None,
        "exact_sentence_accuracy": None,
    }


def score_prediction_subset(predictions: Sequence[Mapping[str, object]]) -> dict:
    if not predictions:
        return _empty_metrics()
    return score_batch(
        [row["reference_token_ids"] for row in predictions],
        [row["predicted_token_ids"] for row in predictions],
    )["aggregate"]


def load_support_map(
    support_path: str | Path,
    partitions: LosoPartitions,
) -> dict[str, dict]:
    rows = [
        row for row in read_csv_rows(support_path)
        if row["loso_id"] == partitions.loso_id
    ]
    mapping = {row["sample_id"]: row for row in rows}
    if len(mapping) != len(rows) or set(mapping) != set(partitions.outer_test):
        raise RuntimeError("Stage-15 support map differs from the LOSO test partition.")
    return mapping


def evaluate_loso_outer_once(
    *,
    parent_directory: Path,
    refit_directory: Path,
    refit_identity,
    model_id: str,
    selected_epochs: int,
    store: RawCacheStore,
    targets: TargetCatalog,
    partitions: LosoPartitions,
    support_path: str | Path,
    id_to_token: Mapping[str, str],
    physical_batch_size: int,
    device,
    use_amp: bool,
) -> dict:
    from src.models.factory import build_model, validate_model_parameter_count
    from src.training.checkpoint import load_torch_checkpoint

    marker_path = parent_directory / "outer_evaluation_once.json"
    predictions_path = parent_directory / "predictions.jsonl"
    result_path = parent_directory / "outer_result.json"
    if marker_path.exists():
        marker = read_json(marker_path)
        if marker.get("status") != "COMPLETE":
            raise RuntimeError("Invalid LOSO outer-evaluation marker.")
        if sha256_file(predictions_path) != marker["predictions_sha256"]:
            raise RuntimeError("Persisted LOSO predictions changed.")
        if sha256_file(result_path) != marker["outer_result_sha256"]:
            raise RuntimeError("Persisted LOSO result changed.")
        return read_json(result_path)
    if predictions_path.exists() or result_path.exists():
        raise RuntimeError("Partial LOSO outer output exists; refusing repeat scoring.")

    normalization = normalization_from_payload(read_json(refit_directory / "normalization.json"))
    if normalization.sample_ids_sha256 != canonical_sample_ids_sha256(
        partitions.outer_training_pool
    ):
        raise RuntimeError("LOSO refit normalization partition changed.")
    model = build_model(model_id).to(device)
    validate_model_parameter_count(model_id, model)
    checkpoint = load_torch_checkpoint(
        refit_directory / "last.pt", refit_identity, map_location="cpu"
    )
    if int(checkpoint["epoch"]) != int(selected_epochs):
        raise RuntimeError("LOSO refit did not reach selected epoch E*.")
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    dataset = ASLSequenceDataset(
        store, targets, partitions.outer_test, normalization_stats=normalization
    )
    evaluation = evaluate_model(
        model, dataset, batch_size=physical_batch_size, device=device,
        use_amp=use_amp, id_to_token=id_to_token,
    )
    expected_count = EXPECTED_COUNTS[partitions.loso_id]["outer_test"]
    if evaluation["aggregate"]["samples"] != expected_count:
        raise RuntimeError("LOSO prediction count changed.")

    support = load_support_map(support_path, partitions)
    annotated = []
    for prediction in evaluation["predictions"]:
        support_row = support[prediction["sample_id"]]
        annotated.append(
            {
                **prediction,
                "loso_id": partitions.loso_id,
                "held_out_subject": partitions.held_out_subject,
                "support_class": support_row["support_class"],
                "template_supported": support_row["template_supported"] == "True",
                "all_tokens_supported": support_row["all_tokens_supported"] == "True",
                "unsupported_token_count": int(support_row["unsupported_token_count"]),
            }
        )
    slices = {
        "loso_all": annotated,
        "template_supported": [p for p in annotated if p["template_supported"]],
        "token_supported": [p for p in annotated if p["all_tokens_supported"]],
        "token_unsupported": [p for p in annotated if not p["all_tokens_supported"]],
    }
    metrics = {name: score_prediction_subset(rows) for name, rows in slices.items()}
    support_counts = Counter(row["support_class"] for row in annotated)

    atomic_write_jsonl(predictions_path, annotated)
    result = {
        "status": "COMPLETE",
        "model_id": model_id,
        "loso_id": partitions.loso_id,
        "loso_index": partitions.loso_index,
        "held_out_subject": partitions.held_out_subject,
        "selected_epochs": selected_epochs,
        "outer_test_samples": expected_count,
        "outer_test_ctc_loss": evaluation["ctc_loss"],
        "outer_test_metrics": metrics["loso_all"],
        "support_metrics": metrics,
        "support_statistics": dict(sorted(support_counts.items())),
        "support_slices_train_separate_models": False,
        "outer_test_controlled_training": False,
        "decoder": "greedy_valid_length_truncated_v1",
    }
    atomic_write_json(result_path, result)
    marker = {
        "status": "COMPLETE",
        "outer_test_evaluation_count": 1,
        "outer_test_samples": expected_count,
        "predictions_sha256": sha256_file(predictions_path),
        "outer_result_sha256": sha256_file(result_path),
    }
    atomic_write_json(marker_path, marker)
    return result


def verify_frozen_inputs(project_root: str | Path) -> dict:
    root = Path(project_root).resolve()
    pairs = {
        "stage33_assignment": "artifacts/splits/stage33_loso_inner_assignment.csv",
        "stage33_run_plan": "artifacts/experiments/stage33_loso_run_plan.json",
        "stage33_initial_registry": "artifacts/registry/stage33_loso_run_registry.csv",
        "stage14_loso": "artifacts/splits/stage14_loso_assignment.csv",
        "stage15_support": "artifacts/splits/stage15_loso_support_assignment.csv",
        "stage17_cache_index": "artifacts/cache/stage17_raw_cache_index.csv",
        "stage10_vocabulary": "artifacts/labels/stage10_gloss_vocabulary.json",
        "stage32_results": "artifacts/results/stage32c_results_contract.json",
    }
    hashes = {
        name: require_portable_recorded_hash(root / relative, (root / relative).with_suffix(".sha256"))
        for name, relative in pairs.items()
    }
    plan = read_json(root / pairs["stage33_run_plan"])
    if len(plan["evaluations"]) != 15 or len(plan["registry_rows"]) != 45:
        raise RuntimeError("Stage-33 run plan dimensions changed.")
    if plan["freeze_boundary"]["required_runner_tag_before_scoring"] != REQUIRED_RUNNER_TAG:
        raise RuntimeError("Stage-33 required runner tag changed.")

    manifest_path = root / "artifacts/config/stage31_config_hashes.json"
    require_portable_recorded_hash(manifest_path, manifest_path.with_suffix(".sha256"))
    manifest = read_json(manifest_path)
    configs = {str(x["path"]).replace("\\", "/"): str(x["sha256"]) for x in manifest["configs"]}
    for relative, expected in configs.items():
        if expected not in set(portable_sha256(root / relative)):
            raise RuntimeError(f"Frozen config changed: {relative}.")
    for evaluation in plan["evaluations"]:
        model_id = evaluation["model_id"]
        payload = {
            "model": configs[f"configs/models/{model_id}.yaml"],
            "data": configs["configs/data.yaml"],
            "experiment": configs["configs/experiments/loso.yaml"],
            "stage14": hashes["stage14_loso"],
            "stage15": hashes["stage15_support"],
            "stage33": hashes["stage33_assignment"],
            "model_id": model_id,
            "loso_id": evaluation["loso_id"],
            "seed": int(evaluation["seed"]),
        }
        if canonical_json_sha256(payload) != evaluation["config_sha256"]:
            raise RuntimeError("A frozen LOSO run identity changed.")
    return hashes


def build_dry_run_report(project_root: str | Path) -> dict:
    root = Path(project_root).resolve()
    hashes = verify_frozen_inputs(root)
    config = validate_runner_configuration(
        read_json(root / "configs/experiments/stage33_runner.json")
    )
    plan = read_json(root / "artifacts/experiments/stage33_loso_run_plan.json")
    registry = read_csv_rows(root / "artifacts/registry/stage33_loso_run_registry.csv")
    partitions = [
        load_loso_partitions(
            root / "artifacts/splits/stage33_loso_inner_assignment.csv", loso_id
        ).to_summary()
        for loso_id in LOSO_IDS
    ]
    if any(row["status"] != "PLANNED" for row in registry):
        raise RuntimeError("Tracked Stage-33 registry must remain PLANNED.")
    expected_order = [(lid, model) for lid in LOSO_IDS for model in MODEL_IDS]
    observed = [(x["loso_id"], x["model_id"]) for x in plan["evaluations"]]
    if observed != expected_order:
        raise RuntimeError("Stage-33 run order changed.")
    mock = [
        {"epoch": 1, "validation_wer": .4, "validation_exact_sentence_accuracy": .2, "validation_ctc_loss": 2.0},
        {"epoch": 2, "validation_wer": .3, "validation_exact_sentence_accuracy": .25, "validation_ctc_loss": 1.9},
        {"epoch": 3, "validation_wer": .3, "validation_exact_sentence_accuracy": .30, "validation_ctc_loss": 2.1},
    ]
    if select_best_epoch(mock)["epoch"] != 3:
        raise RuntimeError("Selection tie-break failed.")
    premature = []
    runs_root = root / "artifacts/runs/loso"
    if runs_root.exists():
        premature = [str(path) for path in runs_root.rglob("*") if path.is_file()]
    if premature:
        raise RuntimeError("Premature LOSO artifacts exist in the repository.")
    return {
        "project_generation": PROJECT_GENERATION,
        "stage": 33,
        "substage": "B",
        "mode": "static_dry_run_no_training_no_scoring",
        "frozen_hashes": hashes,
        "runner_config_sha256": sha256_file(root / "configs/experiments/stage33_runner.json"),
        "matrix": {
            "outer_evaluations": 15,
            "registry_rows": 45,
            "validated_identities": 45,
            "loso_major": True,
            "first": plan["evaluations"][0]["experiment_id"],
            "last": plan["evaluations"][-1]["experiment_id"],
        },
        "partitions": partitions,
        "support_reporting": {
            "views": config["reporting"]["report_slices"],
            "same_predictions_for_all_views": True,
            "separate_models_trained": False,
        },
        "leakage_barrier": {
            "held_out_subject_used_for_training": False,
            "outer_test_constructed_during_dry_run": False,
            "training_started": False,
            "scores_computed": False,
            "checkpoints_created": False,
            "premature_repository_artifacts": 0,
        },
        "runtime_gate": {
            "required_tag": REQUIRED_RUNNER_TAG,
            "required_gpu_name_contains": "T4",
            "execution_precision": "float32",
            "automatic_mixed_precision": False,
            "external_artifact_root_required": True,
        },
        "passed": True,
    }


def run_one_loso_evaluation(
    *,
    project_root: str | Path,
    artifact_root: str | Path,
    raw_cache_path: str | Path,
    evaluation_row: Mapping[str, object],
    live_registry_path: str | Path,
    code_commit: str,
    environment: Mapping[str, object],
) -> dict:
    import torch

    root = Path(project_root).resolve()
    artifact_root = ensure_external_artifact_root(root, artifact_root)
    live_registry_path = Path(live_registry_path)
    phase_rows = _rows_for_evaluation(
        read_csv_rows(live_registry_path), str(evaluation_row["parent_experiment_id"])
    )
    model_id = str(evaluation_row["model_id"])
    loso_id = str(evaluation_row["loso_id"])
    model_config = read_json(root / f"configs/models/{model_id}.yaml")
    runner_config = validate_runner_configuration(
        read_json(root / "configs/experiments/stage33_runner.json")
    )
    # Core Stage-32 routines use the same validated selection/refit keys.
    stage32_config = read_json(root / "configs/experiments/stage32_runner.json")
    validate_stage32_runner_configuration(stage32_config)
    core_config = {**stage32_config, "precision": runner_config["precision"]}
    training_config = model_config["training"]
    combined_config = {
        "model": model_config,
        "runner": runner_config,
        "evaluation": dict(evaluation_row),
    }
    partitions = load_loso_partitions(
        root / "artifacts/splits/stage33_loso_inner_assignment.csv", loso_id
    )
    split_hash = require_portable_recorded_hash(
        root / "artifacts/splits/stage33_loso_inner_assignment.csv",
        root / "artifacts/splits/stage33_loso_inner_assignment.sha256",
    )
    store = RawCacheStore(raw_cache_path, root / "artifacts/cache/stage17_raw_cache_index.csv")
    targets = TargetCatalog(root / "artifacts/labels/stage10_sentence_targets.csv")
    id_to_token = read_json(root / "artifacts/labels/stage10_id_to_token.json")
    parent_directory = artifact_root / str(evaluation_row["artifact_path"])
    identities = {
        phase: build_experiment_identity(phase_rows[phase], code_commit)
        for phase in REGISTRY_PHASES
    }
    completion = parent_directory / "completion.json"
    if completion.exists():
        for phase in ("inner_selection", "full_refit", "outer_evaluation"):
            registry_id = phase_rows[phase]["registry_id"]
            current = {r["registry_id"]: r for r in read_csv_rows(live_registry_path)}[registry_id]["status"]
            if current != "COMPLETE":
                if current in {"PLANNED", "INTERRUPTED"}:
                    set_live_registry_status(live_registry_path, registry_id, "RUNNING")
                set_live_registry_status(live_registry_path, registry_id, "COMPLETE")
        return {"status": "SKIP_COMPLETE", "experiment_id": identities["outer_evaluation"].experiment_id}

    if phase_rows["outer_evaluation"]["status"] in {"PLANNED", "INTERRUPTED"}:
        set_live_registry_status(
            live_registry_path, phase_rows["outer_evaluation"]["registry_id"], "RUNNING"
        )
    configured_batch = int(training_config["physical_batch_size"])
    use_amp = False
    device = torch.device("cuda:0")
    last_error = None
    for attempt_index, physical_batch in enumerate(
        physical_batch_candidates(configured_batch), start=1
    ):
        attempt = parent_directory / "attempts" / f"attempt_{attempt_index:02d}_bs{physical_batch:02d}"
        state_path = attempt / "attempt_state.json"
        if state_path.exists() and read_json(state_path).get("status") == "FAILED_OOM":
            continue
        attempt.mkdir(parents=True, exist_ok=True)
        atomic_write_json(state_path, {
            "status": "RUNNING", "attempt": attempt_index,
            "configured_physical_batch_size": configured_batch,
            "physical_batch_size": physical_batch,
            "effective_batch_size": configured_batch,
            "gradient_accumulation_steps": gradient_accumulation_steps(configured_batch, physical_batch),
        })
        _append_parent_start_or_resume(
            parent_directory, identities["outer_evaluation"], physical_batch_size=physical_batch
        )
        for phase in ("inner_selection",):
            registry_id = phase_rows[phase]["registry_id"]
            current = {r["registry_id"]: r for r in read_csv_rows(live_registry_path)}[registry_id]["status"]
            if current in {"PLANNED", "INTERRUPTED"}:
                set_live_registry_status(live_registry_path, registry_id, "RUNNING")
        try:
            _preflight_memory(
                model_id=model_id, training_config=training_config, store=store,
                targets=targets, partitions=partitions,
                physical_batch_size=physical_batch, device=device,
                use_amp=use_amp, seed=int(evaluation_row["seed"]),
            )
            inner_directory = attempt / "inner_selection"
            refit_directory = attempt / "full_refit"
            selection = run_inner_selection(
                phase_directory=inner_directory,
                identity=identities["inner_selection"], model_id=model_id,
                training_config=training_config, runner_config=core_config,
                combined_config=combined_config, split_sha256=split_hash,
                store=store, targets=targets, partitions=partitions,
                id_to_token=id_to_token, physical_batch_size=physical_batch,
                device=device, use_amp=use_amp,
            )
            refit_id = phase_rows["full_refit"]["registry_id"]
            current = {r["registry_id"]: r for r in read_csv_rows(live_registry_path)}[refit_id]["status"]
            if current in {"PLANNED", "INTERRUPTED"}:
                set_live_registry_status(live_registry_path, refit_id, "RUNNING")
            refit = run_full_refit(
                phase_directory=refit_directory, identity=identities["full_refit"],
                model_id=model_id, selected_epochs=int(selection["selected_epoch"]),
                training_config=training_config, runner_config=core_config,
                combined_config=combined_config, split_sha256=split_hash,
                store=store, targets=targets, partitions=partitions,
                physical_batch_size=physical_batch, device=device, use_amp=use_amp,
            )
            outer = evaluate_loso_outer_once(
                parent_directory=parent_directory, refit_directory=refit_directory,
                refit_identity=identities["full_refit"], model_id=model_id,
                selected_epochs=int(selection["selected_epoch"]), store=store,
                targets=targets, partitions=partitions,
                support_path=root / "artifacts/splits/stage15_loso_support_assignment.csv",
                id_to_token=id_to_token, physical_batch_size=physical_batch,
                device=device, use_amp=use_amp,
            )
            finalize_parent_run(
                parent_directory=parent_directory,
                parent_identity=identities["outer_evaluation"],
                refit_identity=identities["full_refit"],
                inner_directory=inner_directory, refit_directory=refit_directory,
                combined_config=combined_config, split_sha256=split_hash,
                selection_result=selection, refit_result=refit,
                outer_result=outer, environment={
                    **environment, "execution_precision": "float32",
                    "automatic_mixed_precision": False,
                },
            )
            atomic_write_json(state_path, {
                "status": "SUCCEEDED", "attempt": attempt_index,
                "physical_batch_size": physical_batch,
            })
            for phase in ("inner_selection", "full_refit", "outer_evaluation"):
                registry_id = phase_rows[phase]["registry_id"]
                current = {r["registry_id"]: r for r in read_csv_rows(live_registry_path)}[registry_id]["status"]
                if current != "COMPLETE":
                    set_live_registry_status(live_registry_path, registry_id, "COMPLETE")
            return {
                "status": "COMPLETE",
                "experiment_id": identities["outer_evaluation"].experiment_id,
                "model_id": model_id,
                "loso_id": loso_id,
                "held_out_subject": partitions.held_out_subject,
                "selected_epoch": selection["selected_epoch"],
                "physical_batch_size": physical_batch,
                "outer_test_metrics": outer["outer_test_metrics"],
                "support_metrics": outer["support_metrics"],
            }
        except BaseException as exc:
            if not is_out_of_memory_error(exc):
                raise
            last_error = exc
            atomic_write_json(state_path, {
                "status": "FAILED_OOM", "attempt": attempt_index,
                "physical_batch_size": physical_batch,
                "verified_error": type(exc).__name__,
                "verified_error_message": str(exc),
                "restart_from_seed_required": True,
            })
            if physical_batch == 1:
                raise RuntimeError("Verified OOM persisted at batch size 1.") from exc
            torch.cuda.empty_cache()
    raise RuntimeError("No LOSO physical batch candidate completed.") from last_error


def run_next_evaluation(
    *,
    project_root: str | Path,
    artifact_root: str | Path,
    raw_cache_path: str | Path,
) -> dict:
    root = Path(project_root).resolve()
    artifact_root = ensure_external_artifact_root(root, artifact_root)
    verify_frozen_inputs(root)
    freeze = verify_git_freeze(root)
    environment = verify_t4_runtime()
    raw_cache_path = Path(raw_cache_path).resolve()
    expected_cache = read_recorded_sha256(root / "artifacts/cache/stage17_raw_cache.sha256")
    actual_cache = sha256_file(raw_cache_path)
    if actual_cache != expected_cache:
        raise RuntimeError("Stage-17 raw binary cache hash mismatch.")
    live_registry = artifact_root / "artifacts/registry/stage33_loso_run_registry.csv"
    rows = initialize_or_validate_live_registry(
        root / "artifacts/registry/stage33_loso_run_registry.csv",
        live_registry,
        code_commit=freeze["head_commit"],
    )
    evaluation = next_evaluation_row(rows)
    if evaluation is None:
        return {"status": "ALL_COMPLETE", "completed_outer_evaluations": 15}
    return run_one_loso_evaluation(
        project_root=root, artifact_root=artifact_root,
        raw_cache_path=raw_cache_path, evaluation_row=evaluation,
        live_registry_path=live_registry, code_commit=freeze["head_commit"],
        environment={**environment, "git": freeze, "stage17_raw_cache_sha256": actual_cache},
    )
