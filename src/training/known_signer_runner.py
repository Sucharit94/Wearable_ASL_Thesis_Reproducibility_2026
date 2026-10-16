from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.data.collate import collate_numpy
from src.data.dataset import (
    ASLSequenceDataset,
    RawCacheStore,
    TargetCatalog,
    fit_normalization_for_sample_ids,
)
from src.data.preprocessing import FeatureZScoreStats
from src.evaluation.ctc_decode import greedy_ctc_decode
from src.evaluation.metrics import score_batch, score_sequence
from src.training.checkpoint import (
    atomic_torch_save,
    build_checkpoint_envelope,
    load_torch_checkpoint,
    write_checkpoint_metadata,
)
from src.training.registry import (
    ExperimentIdentity,
    append_registry_event,
    atomic_write_json,
    read_json,
    read_registry_events,
    validate_status_transition,
    write_completion_marker,
    write_or_validate_identity,
)


PROJECT_GENERATION = "fresh_start_2026_v1"
RUNNER_SCHEMA_VERSION = 1
RUNNER_TAG = "stage32-known-runner-v1"
STAGE31_TAG = "stage31-freeze-v1"
STAGE31_COMMIT = "62fc7ca0f14611bf0fbc305f1960af4cff441978"
MODEL_ORDER = ("g1", "c1", "c2", "c3", "c4")
OUTER_FOLDS = (1, 2, 3, 4, 5)
EXPECTED_ASSIGNMENT_ROWS = 37000
EXPECTED_FOLD_COUNTS = {
    "inner_train": 5328,
    "inner_validation": 592,
    "outer_test": 1480,
}
REGISTRY_PHASES = (
    "outer_evaluation",
    "inner_selection",
    "full_refit",
)
REGISTRY_FIELDS = (
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_recorded_sha256(path: str | Path) -> str:
    value = (
        Path(path)
        .read_text(encoding="utf-8-sig")
        .strip()
        .split()[0]
        .lower()
    )

    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"Invalid SHA-256 record: {path}.")

    return value


def require_recorded_hash(
    data_path: str | Path,
    hash_path: str | Path,
) -> str:
    recorded = read_recorded_sha256(hash_path)
    actual = sha256_file(data_path)

    if recorded != actual:
        raise RuntimeError(
            f"Frozen hash mismatch for {Path(data_path).name}: "
            f"recorded={recorded}, actual={actual}."
        )

    return actual


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(temporary, path)


def atomic_write_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(temporary, path)


def atomic_write_jsonl(
    path: str | Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    text = "".join(
        json.dumps(
            dict(row),
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
        for row in rows
    )

    if not text:
        raise ValueError("A completed JSONL artifact cannot be empty.")

    atomic_write_text(path, text)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


@dataclass(frozen=True)
class KnownSignerPartitions:
    outer_fold: int
    inner_train: tuple[str, ...]
    inner_validation: tuple[str, ...]
    outer_test: tuple[str, ...]

    @property
    def outer_training_pool(self) -> tuple[str, ...]:
        return tuple(sorted((*self.inner_train, *self.inner_validation)))

    def to_summary(self) -> dict:
        return {
            "outer_fold": self.outer_fold,
            "inner_train": len(self.inner_train),
            "inner_validation": len(self.inner_validation),
            "outer_training_pool": len(self.outer_training_pool),
            "outer_test": len(self.outer_test),
            "all_unique": len(
                set((*self.outer_training_pool, *self.outer_test))
            )
            == 7400,
        }


def load_known_signer_partitions(
    assignment_path: str | Path,
    outer_fold: int,
) -> KnownSignerPartitions:
    outer_fold = int(outer_fold)
    if outer_fold not in OUTER_FOLDS:
        raise ValueError("outer_fold must be 1 through 5.")

    rows = read_csv_rows(assignment_path)
    if len(rows) != EXPECTED_ASSIGNMENT_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ASSIGNMENT_ROWS} Stage-32 assignment rows, "
            f"observed {len(rows)}."
        )

    selected = [
        row
        for row in rows
        if int(row["stage32_outer_fold"]) == outer_fold
    ]
    if len(selected) != 7400:
        raise RuntimeError("A Stage-32 fold plan must contain 7,400 rows.")

    by_partition: dict[str, list[str]] = {
        name: [] for name in EXPECTED_FOLD_COUNTS
    }
    group_partitions: dict[str, set[str]] = {}

    for row in selected:
        partition = row["partition"]
        if partition not in by_partition:
            raise RuntimeError(f"Unknown Stage-32 partition: {partition}.")

        sample_id = row["sample_id"]
        by_partition[partition].append(sample_id)
        group_partitions.setdefault(
            row["stage12_split_group_id"],
            set(),
        ).add(partition)

        expected_outer = int(row["stage13_outer_fold"]) == outer_fold
        if expected_outer != (partition == "outer_test"):
            raise RuntimeError(
                "Stage-13/Stage-32 outer assignment mismatch."
            )

    for name, expected in EXPECTED_FOLD_COUNTS.items():
        observed = len(by_partition[name])
        if observed != expected:
            raise RuntimeError(
                f"Fold {outer_fold} {name} count changed: "
                f"expected {expected}, observed {observed}."
            )

    all_ids = [
        sample_id
        for name in EXPECTED_FOLD_COUNTS
        for sample_id in by_partition[name]
    ]
    if len(set(all_ids)) != 7400:
        raise RuntimeError("Stage-32 fold plan has missing or duplicate sample IDs.")

    crossings = [
        group_id
        for group_id, partitions in group_partitions.items()
        if len(partitions) != 1
    ]
    if crossings:
        raise RuntimeError(
            "Exact-duplicate groups cross Stage-32 partitions: "
            + ", ".join(sorted(crossings)[:10])
            + "."
        )

    return KnownSignerPartitions(
        outer_fold=outer_fold,
        inner_train=tuple(sorted(by_partition["inner_train"])),
        inner_validation=tuple(sorted(by_partition["inner_validation"])),
        outer_test=tuple(sorted(by_partition["outer_test"])),
    )


def validate_runner_configuration(payload: Mapping[str, object]) -> dict:
    if int(payload.get("schema_version", -1)) != RUNNER_SCHEMA_VERSION:
        raise ValueError("Unsupported Stage-32 runner schema version.")
    if payload.get("project_generation") != PROJECT_GENERATION:
        raise ValueError("Wrong project generation in runner configuration.")
    if int(payload.get("stage", -1)) != 32 or payload.get("substage") != "B":
        raise ValueError("Runner configuration must describe Stage 32B.")

    freeze = payload["freeze"]
    if freeze["required_annotated_git_tag"] != RUNNER_TAG:
        raise ValueError("Stage-32 runner tag changed.")
    if freeze["stage31_commit"] != STAGE31_COMMIT:
        raise ValueError("Stage-31 commit changed.")

    precision = payload["precision"]
    if precision["mode"] != "float32":
        raise ValueError("Stage-32 execution precision must be float32.")
    if precision["automatic_mixed_precision"] is not False:
        raise ValueError("Stage-32 automatic mixed precision must be disabled.")
    if tuple(precision["applies_to_models"]) != MODEL_ORDER:
        raise ValueError("Stage-32 float32 policy must cover all five models.")

    selection = payload["selection"]
    if int(selection["compact_early_stopping_minimum_epoch"]) != 15:
        raise ValueError("Compact minimum selection epoch must be 15.")
    if float(selection["compact_plateau_factor"]) != 0.5:
        raise ValueError("Compact plateau factor must be 0.5.")
    if int(selection["compact_plateau_patience"]) != 2:
        raise ValueError("Compact plateau patience must be 2.")
    if selection["scheduler_monitor"] != "validation_ctc_loss":
        raise ValueError("Selection scheduler must monitor validation CTC loss.")

    expected_ties = [
        "lower_validation_wer",
        "higher_validation_exact_sentence_accuracy",
        "lower_validation_ctc_loss",
        "earlier_epoch",
    ]
    if selection["tie_break_order"] != expected_ties:
        raise ValueError("Stage-32 epoch-selection tie order changed.")

    return dict(payload)


def selection_key(record: Mapping[str, object]) -> tuple[float, float, float, int]:
    epoch = int(record["epoch"])
    wer = float(record["validation_wer"])
    accuracy = float(record["validation_exact_sentence_accuracy"])
    loss = float(record["validation_ctc_loss"])

    if epoch <= 0:
        raise ValueError("Selection epoch must be positive.")
    if not all(math.isfinite(value) for value in (wer, accuracy, loss)):
        raise ValueError("Selection record contains a non-finite metric.")
    if wer < 0.0 or not 0.0 <= accuracy <= 1.0 or loss < 0.0:
        raise ValueError("Selection record contains an invalid metric.")

    return (wer, -accuracy, loss, epoch)


def select_best_epoch(
    records: Sequence[Mapping[str, object]],
) -> dict:
    if not records:
        raise ValueError("Cannot select an epoch from an empty history.")

    epochs = [int(record["epoch"]) for record in records]
    if len(set(epochs)) != len(epochs):
        raise ValueError("Selection history contains duplicate epochs.")

    return dict(min(records, key=selection_key))


def compact_early_stopping_due(
    records: Sequence[Mapping[str, object]],
    *,
    current_epoch: int,
    minimum_epoch: int,
    patience: int,
) -> bool:
    current_epoch = int(current_epoch)
    minimum_epoch = int(minimum_epoch)
    patience = int(patience)

    if minimum_epoch <= 0 or patience <= 0:
        raise ValueError("Early-stopping bounds must be positive.")
    if current_epoch < minimum_epoch or len(records) < patience + 1:
        return False

    best = select_best_epoch(records)
    return current_epoch - int(best["epoch"]) >= patience


def compact_warmup_epochs(selected_epochs: int) -> int:
    selected_epochs = int(selected_epochs)
    if selected_epochs <= 0:
        raise ValueError("selected_epochs must be positive.")
    return min(5, max(1, math.ceil(0.10 * selected_epochs)))


def refit_learning_rate_multiplier(
    epoch: int,
    selected_epochs: int,
    *,
    minimum_ratio: float = 0.01,
) -> float:
    epoch = int(epoch)
    selected_epochs = int(selected_epochs)
    minimum_ratio = float(minimum_ratio)

    if not 1 <= epoch <= selected_epochs:
        raise ValueError("epoch is outside the refit epoch range.")
    if not 0.0 < minimum_ratio <= 1.0:
        raise ValueError("minimum_ratio must be in (0, 1].")

    warmup = compact_warmup_epochs(selected_epochs)
    if epoch <= warmup:
        return epoch / warmup
    if selected_epochs == warmup:
        return 1.0

    progress = (epoch - warmup) / (selected_epochs - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def physical_batch_candidates(configured_batch_size: int) -> tuple[int, ...]:
    configured_batch_size = int(configured_batch_size)
    if configured_batch_size <= 0:
        raise ValueError("configured_batch_size must be positive.")

    candidates = []
    value = configured_batch_size
    while True:
        candidates.append(value)
        if value == 1:
            break
        value = max(1, value // 2)

    return tuple(candidates)


def gradient_accumulation_steps(
    effective_batch_size: int,
    physical_batch_size: int,
) -> int:
    effective_batch_size = int(effective_batch_size)
    physical_batch_size = int(physical_batch_size)

    if physical_batch_size <= 0 or effective_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if physical_batch_size > effective_batch_size:
        raise ValueError("Physical batch cannot exceed effective batch.")
    if effective_batch_size % physical_batch_size != 0:
        raise ValueError("Physical batch must divide the effective batch.")

    return effective_batch_size // physical_batch_size


def deterministic_epoch_indices(
    sample_count: int,
    *,
    seed: int,
    epoch: int,
) -> tuple[int, ...]:
    sample_count = int(sample_count)
    seed = int(seed)
    epoch = int(epoch)
    if sample_count <= 0 or seed < 0 or epoch <= 0:
        raise ValueError("Invalid deterministic shuffle arguments.")

    generator = np.random.Generator(
        np.random.PCG64(seed + epoch)
    )
    return tuple(int(value) for value in generator.permutation(sample_count))


def ensure_external_artifact_root(
    project_root: str | Path,
    artifact_root: str | Path,
) -> Path:
    project = Path(project_root).resolve()
    artifact = Path(artifact_root).resolve()

    if artifact == project or project in artifact.parents:
        raise ValueError(
            "Stage-32 runtime artifacts must be outside the Git project root."
        )

    return artifact


def build_experiment_identity(
    registry_row: Mapping[str, object],
    code_commit: str,
) -> ExperimentIdentity:
    return ExperimentIdentity(
        experiment_id=str(registry_row["experiment_id"]),
        project_generation=str(registry_row["project_generation"]),
        model_id=str(registry_row["model_id"]),
        phase=str(registry_row["phase"]),
        split_id=str(registry_row["split_id"]),
        data_manifest_sha256=str(registry_row["data_manifest_sha256"]),
        label_vocab_sha256=str(registry_row["label_vocab_sha256"]),
        config_sha256=str(registry_row["config_sha256"]),
        code_commit=str(code_commit),
        seed=int(registry_row["seed"]),
        normalization=str(registry_row["normalization"]),
        augmentation=str(registry_row["augmentation"]),
        hardware_required=str(registry_row["hardware_required"]),
    )


def _immutable_registry_projection(row: Mapping[str, object]) -> dict:
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
    if len(template) != 75:
        raise RuntimeError("Stage-32 registry template must contain 75 rows.")

    for row in template:
        if tuple(row) != REGISTRY_FIELDS:
            raise RuntimeError("Stage-32 registry column order changed.")

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
    if len(live) != len(template):
        raise RuntimeError("Live Stage-32 registry row count changed.")

    for expected, observed in zip(template, live, strict=True):
        if _immutable_registry_projection(expected) != _immutable_registry_projection(observed):
            raise RuntimeError("Live registry identity differs from Stage-32A.")
        if observed["code_commit"] != code_commit:
            raise RuntimeError("Live registry was created by a different code commit.")

    return live


def set_live_registry_status(
    live_path: str | Path,
    registry_id: str,
    next_status: str,
) -> dict[str, str]:
    rows = read_csv_rows(live_path)
    matches = [row for row in rows if row["registry_id"] == registry_id]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one live registry row for {registry_id}.")

    row = matches[0]
    current = row["status"]
    validate_status_transition(current, next_status)
    row["status"] = next_status
    atomic_write_csv(live_path, rows, REGISTRY_FIELDS)
    return dict(row)


def next_evaluation_row(
    rows: Sequence[Mapping[str, object]],
) -> dict | None:
    parents = sorted(
        (
            row
            for row in rows
            if row["phase"] == "outer_evaluation"
        ),
        key=lambda row: int(row["evaluation_run_order"]),
    )

    if len(parents) != 25:
        raise RuntimeError("Expected 25 outer-evaluation registry rows.")

    for row in parents:
        status = str(row["status"])
        if status in {"FAILED_SAFE", "INVALID"}:
            raise RuntimeError(
                f"Run matrix is blocked by {row['registry_id']} status {status}."
            )
        if status != "COMPLETE":
            return dict(row)

    return None


def verify_git_freeze(
    project_root: str | Path,
    *,
    runner_tag: str = RUNNER_TAG,
    require_clean: bool = True,
) -> dict:
    project_root = Path(project_root).resolve()

    def git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=check,
        )

    inside = git("rev-parse", "--is-inside-work-tree").stdout.strip()
    if inside != "true":
        raise RuntimeError("Stage-32 execution requires a Git worktree.")

    tag_type = git("cat-file", "-t", f"refs/tags/{runner_tag}").stdout.strip()
    if tag_type != "tag":
        raise RuntimeError(f"{runner_tag} must be an annotated Git tag.")

    tag_commit = git("rev-list", "-n", "1", runner_tag).stdout.strip().lower()
    head = git("rev-parse", "HEAD").stdout.strip().lower()
    if tag_commit != head:
        raise RuntimeError(
            f"HEAD must equal {runner_tag} before Stage-32 training."
        )

    ancestor = git(
        "merge-base",
        "--is-ancestor",
        STAGE31_COMMIT,
        "HEAD",
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("Stage-31 freeze commit is not an ancestor of HEAD.")

    status = git("status", "--porcelain").stdout
    if require_clean and status.strip():
        raise RuntimeError("Git worktree must be clean before an outer run.")

    return {
        "runner_tag": runner_tag,
        "runner_tag_annotated": True,
        "tag_commit": tag_commit,
        "head_commit": head,
        "tag_commit_equals_head": True,
        "stage31_commit_is_ancestor": True,
        "working_tree_clean": not bool(status.strip()),
    }


def verify_t4_runtime() -> dict:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for Stage-32 execution.") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("Stage-32 scoring requires a CUDA GPU.")

    device_name = torch.cuda.get_device_name(0)
    if "T4" not in device_name.upper():
        raise RuntimeError(
            f"Stage-32 requires an NVIDIA T4; observed {device_name!r}."
        )

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": (
            torch.backends.cudnn.version()
            if hasattr(torch.backends, "cudnn")
            else None
        ),
        "cuda_available": True,
        "device_name": device_name,
        "device_capability": list(torch.cuda.get_device_capability(0)),
    }


def normalization_payload(stats: FeatureZScoreStats) -> dict:
    return {
        "mean": np.asarray(stats.mean, dtype=np.float64).tolist(),
        "std": np.asarray(stats.std, dtype=np.float64).tolist(),
        "frame_count": int(stats.frame_count),
        "recording_count": int(stats.recording_count),
        "feature_width": int(stats.feature_width),
        "sample_ids_sha256": stats.sample_ids_sha256,
    }


def normalization_from_payload(payload: Mapping[str, object]) -> FeatureZScoreStats:
    return FeatureZScoreStats(
        mean=np.asarray(payload["mean"], dtype=np.float64),
        std=np.asarray(payload["std"], dtype=np.float64),
        frame_count=int(payload["frame_count"]),
        recording_count=int(payload["recording_count"]),
        feature_width=int(payload["feature_width"]),
        sample_ids_sha256=str(payload["sample_ids_sha256"]),
    )


def fit_or_load_normalization(
    path: str | Path,
    store: RawCacheStore,
    sample_ids: Sequence[str],
) -> FeatureZScoreStats:
    path = Path(path)
    if path.exists():
        stats = normalization_from_payload(read_json(path))
        expected = canonical_sample_ids_sha256(sample_ids)
        if stats.sample_ids_sha256 != expected:
            raise RuntimeError("Normalization training-partition hash mismatch.")
        return stats

    stats = fit_normalization_for_sample_ids(store, sample_ids)
    atomic_write_json(path, normalization_payload(stats))
    return stats


def canonical_sample_ids_sha256(sample_ids: Sequence[str]) -> str:
    values = sorted(str(value) for value in sample_ids)
    if len(values) != len(set(values)):
        raise ValueError("Normalization sample IDs are not unique.")
    payload = "".join(f"{value}\n" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def capture_rng_state(torch_module) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch_module.get_rng_state(),
    }
    if torch_module.cuda.is_available():
        state["torch_cuda"] = torch_module.cuda.get_rng_state_all()
    return state


def restore_rng_state(torch_module, state: Mapping[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_module.set_rng_state(state["torch_cpu"])
    if torch_module.cuda.is_available() and "torch_cuda" in state:
        torch_module.cuda.set_rng_state_all(state["torch_cuda"])


def set_deterministic_runtime_seed(torch_module, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    if hasattr(torch_module.backends, "cudnn"):
        torch_module.backends.cudnn.benchmark = False
        torch_module.backends.cudnn.deterministic = True


def _make_grad_scaler(torch_module, enabled: bool):
    if hasattr(torch_module, "amp") and hasattr(torch_module.amp, "GradScaler"):
        return torch_module.amp.GradScaler("cuda", enabled=enabled)
    return torch_module.cuda.amp.GradScaler(enabled=enabled)


def _autocast(torch_module, enabled: bool):
    return torch_module.autocast(
        device_type="cuda",
        dtype=torch_module.float16,
        enabled=enabled,
    )


def _model_batch_loss(
    model,
    items: Sequence[dict],
    *,
    device,
    use_amp: bool,
):
    import torch

    from src.models.contracts import validate_model_output
    from src.training.ctc_loss import torch_ctc_loss_from_logits

    batch = collate_numpy(items)
    inputs = torch.as_tensor(
        batch["inputs"],
        dtype=torch.float32,
        device=device,
    )
    input_lengths = torch.as_tensor(
        batch["input_lengths"],
        dtype=torch.long,
        device="cpu",
    )
    targets_padded = torch.as_tensor(
        batch["targets_padded"],
        dtype=torch.long,
        device="cpu",
    )
    target_lengths = torch.as_tensor(
        batch["target_lengths"],
        dtype=torch.long,
        device="cpu",
    )

    with _autocast(torch, use_amp):
        output = model(inputs, input_lengths)

    validate_model_output(output)
    loss = torch_ctc_loss_from_logits(
        output.logits_btc.float(),
        targets_padded,
        output.output_lengths,
        target_lengths,
        reduction="mean",
    )
    return loss, output, batch


def train_epoch(
    model,
    dataset: ASLSequenceDataset,
    optimizer,
    scaler,
    *,
    epoch: int,
    seed: int,
    physical_batch_size: int,
    effective_batch_size: int,
    gradient_clip_norm: float,
    device,
    use_amp: bool,
) -> dict:
    import torch

    accumulation_steps = gradient_accumulation_steps(
        effective_batch_size,
        physical_batch_size,
    )
    order = deterministic_epoch_indices(
        len(dataset),
        seed=seed,
        epoch=epoch,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    weighted_loss = 0.0
    sample_count = 0
    group_samples = 0
    group_micro_batches = 0
    optimizer_steps = 0
    maximum_preclip_norm = 0.0

    for start in range(0, len(order), physical_batch_size):
        indices = order[start : start + physical_batch_size]
        items = [dataset[index] for index in indices]
        loss, _, _ = _model_batch_loss(
            model,
            items,
            device=device,
            use_amp=use_amp,
        )
        if not torch.isfinite(loss).item():
            raise RuntimeError("Training CTC loss is not finite.")

        count = len(items)
        scaler.scale(loss * count).backward()
        weighted_loss += float(loss.detach().cpu().item()) * count
        sample_count += count
        group_samples += count
        group_micro_batches += 1

        final_micro_batch = start + physical_batch_size >= len(order)
        if group_micro_batches == accumulation_steps or final_micro_batch:
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(group_samples)

            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(gradient_clip_norm),
                error_if_nonfinite=True,
            )
            maximum_preclip_norm = max(
                maximum_preclip_norm,
                float(torch.as_tensor(norm).detach().cpu().item()),
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            group_samples = 0
            group_micro_batches = 0

    return {
        "training_ctc_loss": weighted_loss / sample_count,
        "training_samples": sample_count,
        "optimizer_steps": optimizer_steps,
        "maximum_preclip_gradient_norm": maximum_preclip_norm,
        "physical_batch_size": physical_batch_size,
        "effective_batch_size": effective_batch_size,
        "gradient_accumulation_steps": accumulation_steps,
    }


def _token_text(token_ids: Sequence[int], id_to_token: Mapping[str, str]) -> list[str]:
    return [id_to_token[str(int(token_id))] for token_id in token_ids]


def evaluate_model(
    model,
    dataset: ASLSequenceDataset,
    *,
    batch_size: int,
    device,
    use_amp: bool,
    id_to_token: Mapping[str, str],
) -> dict:
    import torch

    if len(dataset) <= 0:
        raise ValueError("Evaluation dataset cannot be empty.")

    model.eval()
    references: list[tuple[int, ...]] = []
    hypotheses: list[tuple[int, ...]] = []
    predictions: list[dict] = []
    weighted_loss = 0.0
    evaluated = 0

    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            items = [
                dataset[index]
                for index in range(start, min(start + batch_size, len(dataset)))
            ]
            loss, output, _ = _model_batch_loss(
                model,
                items,
                device=device,
                use_amp=use_amp,
            )
            if not torch.isfinite(loss).item():
                raise RuntimeError("Evaluation CTC loss is not finite.")

            output_lengths = [
                int(value)
                for value in torch.as_tensor(output.output_lengths).cpu().tolist()
            ]
            decoded = greedy_ctc_decode(
                output.logits_btc.detach().float().cpu().numpy(),
                output_lengths,
            )

            for item, hypothesis, output_length in zip(
                items,
                decoded,
                output_lengths,
                strict=True,
            ):
                reference = tuple(int(value) for value in item["target_ids"])
                sample_score = score_sequence(reference, hypothesis)
                references.append(reference)
                hypotheses.append(hypothesis)
                predictions.append(
                    {
                        "sample_id": item["sample_id"],
                        "subject": item["subject"],
                        "sentence_id": int(item["sentence_id"]),
                        "reference_gloss": item["gloss_raw"],
                        "reference_token_ids": list(reference),
                        "reference_tokens": _token_text(reference, id_to_token),
                        "predicted_gloss": " ".join(
                            _token_text(hypothesis, id_to_token)
                        ),
                        "predicted_token_ids": list(hypothesis),
                        "predicted_tokens": _token_text(hypothesis, id_to_token),
                        "valid_output_length": output_length,
                        "substitutions": sample_score["substitutions"],
                        "deletions": sample_score["deletions"],
                        "insertions": sample_score["insertions"],
                        "edit_distance": sample_score["edit_distance"],
                        "exact_match": sample_score["exact_match"],
                    }
                )

            count = len(items)
            weighted_loss += float(loss.detach().cpu().item()) * count
            evaluated += count

    scored = score_batch(references, hypotheses)
    return {
        "ctc_loss": weighted_loss / evaluated,
        "aggregate": scored["aggregate"],
        "predictions": predictions,
    }


def _build_optimizer(model, training_config: Mapping[str, object]):
    import torch

    arguments = {
        "lr": float(training_config["learning_rate"]),
        "weight_decay": float(training_config["weight_decay"]),
    }
    if training_config["optimizer"] == "Adam":
        return torch.optim.Adam(model.parameters(), **arguments)
    if training_config["optimizer"] == "AdamW":
        return torch.optim.AdamW(model.parameters(), **arguments)
    raise ValueError(f"Unsupported optimizer: {training_config['optimizer']}.")


def _write_phase_static_artifacts(
    phase_directory: Path,
    identity: ExperimentIdentity,
    *,
    combined_config: Mapping[str, object],
    split_sha256: str,
) -> None:
    write_or_validate_identity(phase_directory, identity)
    atomic_write_text(
        phase_directory / "config.yaml",
        json.dumps(combined_config, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )
    atomic_write_text(
        phase_directory / "data_manifest_sha256.txt",
        identity.data_manifest_sha256 + "\n",
    )
    atomic_write_text(
        phase_directory / "label_vocab_sha256.txt",
        identity.label_vocab_sha256 + "\n",
    )
    atomic_write_text(
        phase_directory / "split_sha256.txt",
        split_sha256 + "\n",
    )
    atomic_write_text(
        phase_directory / "code_commit.txt",
        identity.code_commit + "\n",
    )


def _event_start_or_resume(
    events_path: Path,
    identity: ExperimentIdentity,
    *,
    details: Mapping[str, object],
) -> None:
    if events_path.exists() and events_path.stat().st_size > 0:
        event = "RESUME"
    else:
        event = "START"
    append_registry_event(
        events_path,
        identity,
        event=event,
        status="RUNNING",
        details=details,
    )


def _write_training_log(path: Path, history: Sequence[Mapping[str, object]]) -> None:
    if not history:
        raise ValueError("Training history cannot be empty.")
    fieldnames = tuple(history[0])
    atomic_write_csv(path, history, fieldnames)


def _save_checkpoint(
    path: Path,
    identity: ExperimentIdentity,
    *,
    epoch: int,
    selected_epochs: int | None,
    model,
    optimizer,
    scheduler,
    scaler,
    torch_module,
    training_state: Mapping[str, object],
    write_metadata: bool,
) -> dict:
    envelope = build_checkpoint_envelope(
        identity,
        epoch=epoch,
        selected_epochs=selected_epochs,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=(scheduler.state_dict() if scheduler is not None else None),
        amp_scaler_state=scaler.state_dict(),
        rng_state=capture_rng_state(torch_module),
    )
    envelope["training_state"] = dict(training_state)
    atomic_torch_save(envelope, path, identity)
    if write_metadata:
        write_checkpoint_metadata(
            path.parent / "last_checkpoint.json",
            envelope,
            identity,
        )
    return envelope


def _load_phase_resume(
    phase_directory: Path,
    identity: ExperimentIdentity,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    torch_module,
) -> tuple[int, dict]:
    checkpoint_path = phase_directory / "last.pt"
    metadata_path = phase_directory / "last_checkpoint.json"

    if checkpoint_path.exists() != metadata_path.exists():
        raise RuntimeError("Incomplete phase checkpoint pair.")
    if not checkpoint_path.exists():
        return 0, {}

    envelope = load_torch_checkpoint(
        checkpoint_path,
        identity,
        map_location="cpu",
    )
    model.load_state_dict(envelope["model_state"])
    optimizer.load_state_dict(envelope["optimizer_state"])
    if scheduler is not None and envelope["scheduler_state"] is not None:
        scheduler.load_state_dict(envelope["scheduler_state"])
    if envelope["amp_scaler_state"]:
        scaler.load_state_dict(envelope["amp_scaler_state"])
    restore_rng_state(torch_module, envelope["rng_state"])
    return int(envelope["epoch"]), dict(envelope.get("training_state", {}))


def run_inner_selection(
    *,
    phase_directory: Path,
    identity: ExperimentIdentity,
    model_id: str,
    training_config: Mapping[str, object],
    runner_config: Mapping[str, object],
    combined_config: Mapping[str, object],
    split_sha256: str,
    store: RawCacheStore,
    targets: TargetCatalog,
    partitions: KnownSignerPartitions,
    id_to_token: Mapping[str, str],
    physical_batch_size: int,
    device,
    use_amp: bool,
) -> dict:
    import torch

    from src.models.factory import build_model, validate_model_parameter_count

    phase_directory.mkdir(parents=True, exist_ok=True)
    state_path = phase_directory / "phase_state.json"
    if state_path.exists():
        state = read_json(state_path)
        if state.get("status") != "COMPLETE":
            raise RuntimeError("Invalid completed inner-selection state.")
        return state

    _write_phase_static_artifacts(
        phase_directory,
        identity,
        combined_config=combined_config,
        split_sha256=split_sha256,
    )
    normalization = fit_or_load_normalization(
        phase_directory / "normalization.json",
        store,
        partitions.inner_train,
    )
    training_dataset = ASLSequenceDataset(
        store,
        targets,
        partitions.inner_train,
        normalization_stats=normalization,
    )
    validation_dataset = ASLSequenceDataset(
        store,
        targets,
        partitions.inner_validation,
        normalization_stats=normalization,
    )

    set_deterministic_runtime_seed(torch, identity.seed)
    model = build_model(model_id).to(device)
    validate_model_parameter_count(model_id, model)
    optimizer = _build_optimizer(model, training_config)
    selection_policy = runner_config["selection"]
    scheduler = None
    if model_id != "g1":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(selection_policy["compact_plateau_factor"]),
            patience=int(selection_policy["compact_plateau_patience"]),
            threshold=float(selection_policy["compact_plateau_threshold"]),
            threshold_mode=str(selection_policy["compact_plateau_threshold_mode"]),
            min_lr=float(selection_policy["compact_plateau_min_lr"]),
        )
    scaler = _make_grad_scaler(torch, use_amp)

    start_epoch, training_state = _load_phase_resume(
        phase_directory,
        identity,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        torch_module=torch,
    )
    history = list(training_state.get("history", []))
    _event_start_or_resume(
        phase_directory / "events.jsonl",
        identity,
        details={
            "phase": "inner_selection",
            "resume_epoch": start_epoch,
            "outer_test_loaded": False,
        },
    )

    epoch_cap = int(training_config["epoch_cap"])
    minimum_epoch = int(selection_policy["compact_early_stopping_minimum_epoch"])
    patience = int(training_config["early_stopping_patience"])
    effective_batch_size = int(training_config["physical_batch_size"])

    for epoch in range(start_epoch + 1, epoch_cap + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        training_result = train_epoch(
            model,
            training_dataset,
            optimizer,
            scaler,
            epoch=epoch,
            seed=identity.seed,
            physical_batch_size=physical_batch_size,
            effective_batch_size=effective_batch_size,
            gradient_clip_norm=float(training_config["gradient_clip_norm"]),
            device=device,
            use_amp=use_amp,
        )
        validation = evaluate_model(
            model,
            validation_dataset,
            batch_size=physical_batch_size,
            device=device,
            use_amp=use_amp,
            id_to_token=id_to_token,
        )
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            **training_result,
            "validation_ctc_loss": validation["ctc_loss"],
            "validation_wer": validation["aggregate"]["pooled_wer"],
            "validation_exact_sentence_accuracy": validation["aggregate"][
                "exact_sentence_accuracy"
            ],
        }
        previous_best = select_best_epoch(history) if history else None
        history.append(record)
        current_best = select_best_epoch(history)
        is_new_best = previous_best is None or selection_key(record) < selection_key(previous_best)

        if scheduler is not None:
            scheduler.step(validation["ctc_loss"])

        state = {
            "history": history,
            "best_record": current_best,
            "physical_batch_size": physical_batch_size,
            "effective_batch_size": effective_batch_size,
            "outer_test_loaded": False,
        }
        _save_checkpoint(
            phase_directory / "last.pt",
            identity,
            epoch=epoch,
            selected_epochs=None,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            torch_module=torch,
            training_state=state,
            write_metadata=True,
        )
        if is_new_best:
            _save_checkpoint(
                phase_directory / "best.pt",
                identity,
                epoch=epoch,
                selected_epochs=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                torch_module=torch,
                training_state=state,
                write_metadata=False,
            )
            atomic_write_jsonl(
                phase_directory / "predictions.jsonl",
                validation["predictions"],
            )
            atomic_write_json(
                phase_directory / "metrics.json",
                {
                    "phase": "inner_selection",
                    "selected_so_far": current_best,
                    "validation": validation["aggregate"],
                    "validation_ctc_loss": validation["ctc_loss"],
                    "outer_test_loaded": False,
                },
            )

        _write_training_log(phase_directory / "train_log.csv", history)
        append_registry_event(
            phase_directory / "events.jsonl",
            identity,
            event="EPOCH_END",
            status="RUNNING",
            details={
                "epoch": epoch,
                "validation_wer": record["validation_wer"],
                "validation_exact_sentence_accuracy": record[
                    "validation_exact_sentence_accuracy"
                ],
                "validation_ctc_loss": record["validation_ctc_loss"],
                "is_new_best": is_new_best,
            },
        )
        append_registry_event(
            phase_directory / "events.jsonl",
            identity,
            event="CHECKPOINT_SYNC",
            status="RUNNING",
            details={"epoch": epoch},
        )

        if model_id != "g1" and compact_early_stopping_due(
            history,
            current_epoch=epoch,
            minimum_epoch=minimum_epoch,
            patience=patience,
        ):
            break

    best = select_best_epoch(history)
    result = {
        "status": "COMPLETE",
        "phase": "inner_selection",
        "model_id": model_id,
        "epochs_completed": int(history[-1]["epoch"]),
        "selected_epoch": int(best["epoch"]),
        "selected_validation_wer": float(best["validation_wer"]),
        "selected_validation_exact_sentence_accuracy": float(
            best["validation_exact_sentence_accuracy"]
        ),
        "selected_validation_ctc_loss": float(best["validation_ctc_loss"]),
        "g1_full_epoch_cap_completed": (
            model_id != "g1" or int(history[-1]["epoch"]) == epoch_cap
        ),
        "outer_test_loaded": False,
        "normalization_sample_ids_sha256": normalization.sample_ids_sha256,
        "completed_at_utc": utc_now(),
    }
    atomic_write_json(state_path, result)
    append_registry_event(
        phase_directory / "events.jsonl",
        identity,
        event="COMPLETE",
        status="COMPLETE",
        details={
            "selected_epoch": result["selected_epoch"],
            "outer_test_loaded": False,
        },
    )
    return result


def run_full_refit(
    *,
    phase_directory: Path,
    identity: ExperimentIdentity,
    model_id: str,
    selected_epochs: int,
    training_config: Mapping[str, object],
    runner_config: Mapping[str, object],
    combined_config: Mapping[str, object],
    split_sha256: str,
    store: RawCacheStore,
    targets: TargetCatalog,
    partitions: KnownSignerPartitions,
    physical_batch_size: int,
    device,
    use_amp: bool,
) -> dict:
    import torch

    from src.models.factory import build_model, validate_model_parameter_count

    phase_directory.mkdir(parents=True, exist_ok=True)
    state_path = phase_directory / "phase_state.json"
    if state_path.exists():
        state = read_json(state_path)
        if state.get("status") != "COMPLETE":
            raise RuntimeError("Invalid completed refit state.")
        if int(state["selected_epochs"]) != int(selected_epochs):
            raise RuntimeError("Refit selected-epoch count changed.")
        return state

    _write_phase_static_artifacts(
        phase_directory,
        identity,
        combined_config=combined_config,
        split_sha256=split_sha256,
    )
    normalization = fit_or_load_normalization(
        phase_directory / "normalization.json",
        store,
        partitions.outer_training_pool,
    )
    training_dataset = ASLSequenceDataset(
        store,
        targets,
        partitions.outer_training_pool,
        normalization_stats=normalization,
    )

    set_deterministic_runtime_seed(torch, identity.seed)
    model = build_model(model_id).to(device)
    validate_model_parameter_count(model_id, model)
    optimizer = _build_optimizer(model, training_config)
    scaler = _make_grad_scaler(torch, use_amp)
    start_epoch, training_state = _load_phase_resume(
        phase_directory,
        identity,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
        torch_module=torch,
    )
    history = list(training_state.get("history", []))
    _event_start_or_resume(
        phase_directory / "events.jsonl",
        identity,
        details={
            "phase": "full_refit",
            "resume_epoch": start_epoch,
            "selected_epochs": selected_epochs,
            "validation_loaded": False,
            "outer_test_loaded": False,
        },
    )

    base_learning_rate = float(training_config["learning_rate"])
    effective_batch_size = int(training_config["physical_batch_size"])

    for epoch in range(start_epoch + 1, selected_epochs + 1):
        if model_id == "g1":
            multiplier = 1.0
        else:
            multiplier = refit_learning_rate_multiplier(
                epoch,
                selected_epochs,
                minimum_ratio=float(
                    runner_config["refit"]["compact_minimum_learning_rate_ratio"]
                ),
            )
        learning_rate = base_learning_rate * multiplier
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        training_result = train_epoch(
            model,
            training_dataset,
            optimizer,
            scaler,
            epoch=epoch,
            seed=identity.seed,
            physical_batch_size=physical_batch_size,
            effective_batch_size=effective_batch_size,
            gradient_clip_norm=float(training_config["gradient_clip_norm"]),
            device=device,
            use_amp=use_amp,
        )
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "learning_rate_multiplier": multiplier,
            **training_result,
            "validation_used": False,
            "outer_test_loaded": False,
        }
        history.append(record)
        state = {
            "history": history,
            "selected_epochs": selected_epochs,
            "physical_batch_size": physical_batch_size,
            "effective_batch_size": effective_batch_size,
            "validation_used": False,
            "outer_test_loaded": False,
        }
        _save_checkpoint(
            phase_directory / "last.pt",
            identity,
            epoch=epoch,
            selected_epochs=selected_epochs,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=scaler,
            torch_module=torch,
            training_state=state,
            write_metadata=True,
        )
        _write_training_log(phase_directory / "train_log.csv", history)
        append_registry_event(
            phase_directory / "events.jsonl",
            identity,
            event="EPOCH_END",
            status="RUNNING",
            details={
                "epoch": epoch,
                "validation_used": False,
                "outer_test_loaded": False,
            },
        )
        append_registry_event(
            phase_directory / "events.jsonl",
            identity,
            event="CHECKPOINT_SYNC",
            status="RUNNING",
            details={"epoch": epoch},
        )

    shutil.copy2(phase_directory / "last.pt", phase_directory / "best.pt")
    atomic_write_json(
        phase_directory / "metrics.json",
        {
            "phase": "full_refit",
            "selected_epochs": selected_epochs,
            "epochs_completed": len(history),
            "validation_used": False,
            "outer_test_loaded": False,
        },
    )
    atomic_write_jsonl(
        phase_directory / "predictions.jsonl",
        [
            {
                "phase": "full_refit",
                "predictions_available": False,
                "reason": "outer_test_is_evaluated_once_by_parent_after_refit",
            }
        ],
    )
    result = {
        "status": "COMPLETE",
        "phase": "full_refit",
        "model_id": model_id,
        "selected_epochs": selected_epochs,
        "epochs_completed": len(history),
        "validation_used": False,
        "outer_test_loaded": False,
        "normalization_sample_ids_sha256": normalization.sample_ids_sha256,
        "completed_at_utc": utc_now(),
    }
    atomic_write_json(state_path, result)
    append_registry_event(
        phase_directory / "events.jsonl",
        identity,
        event="COMPLETE",
        status="COMPLETE",
        details={
            "selected_epochs": selected_epochs,
            "validation_used": False,
            "outer_test_loaded": False,
        },
    )
    return result


def evaluate_outer_test_once(
    *,
    parent_directory: Path,
    refit_directory: Path,
    refit_identity: ExperimentIdentity,
    model_id: str,
    selected_epochs: int,
    store: RawCacheStore,
    targets: TargetCatalog,
    partitions: KnownSignerPartitions,
    id_to_token: Mapping[str, str],
    physical_batch_size: int,
    device,
    use_amp: bool,
) -> dict:
    import torch

    from src.models.factory import build_model, validate_model_parameter_count

    marker_path = parent_directory / "outer_evaluation_once.json"
    predictions_path = parent_directory / "predictions.jsonl"
    result_path = parent_directory / "outer_result.json"
    if marker_path.exists():
        marker = read_json(marker_path)
        if marker.get("status") != "COMPLETE":
            raise RuntimeError("Invalid outer-evaluation marker.")
        if sha256_file(predictions_path) != marker["predictions_sha256"]:
            raise RuntimeError("Persisted outer predictions changed.")
        if sha256_file(result_path) != marker["outer_result_sha256"]:
            raise RuntimeError("Persisted outer result changed.")
        return read_json(result_path)

    if predictions_path.exists() or result_path.exists():
        raise RuntimeError(
            "Outer outputs exist without the one-evaluation marker. "
            "Refusing to evaluate the test partition again; inspect the "
            "partial atomic-finalization state first."
        )

    normalization = normalization_from_payload(
        read_json(refit_directory / "normalization.json")
    )
    expected_normalization_ids = canonical_sample_ids_sha256(
        partitions.outer_training_pool
    )
    if normalization.sample_ids_sha256 != expected_normalization_ids:
        raise RuntimeError("Outer-refit normalization partition changed.")

    model = build_model(model_id).to(device)
    validate_model_parameter_count(model_id, model)
    checkpoint = load_torch_checkpoint(
        refit_directory / "last.pt",
        refit_identity,
        map_location="cpu",
    )
    if int(checkpoint["epoch"]) != int(selected_epochs):
        raise RuntimeError("Refit checkpoint did not reach selected epoch E*.")
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    # This is intentionally the first construction of the outer-test dataset.
    test_dataset = ASLSequenceDataset(
        store,
        targets,
        partitions.outer_test,
        normalization_stats=normalization,
    )
    evaluation = evaluate_model(
        model,
        test_dataset,
        batch_size=physical_batch_size,
        device=device,
        use_amp=use_amp,
        id_to_token=id_to_token,
    )
    if evaluation["aggregate"]["samples"] != 1480:
        raise RuntimeError("Outer evaluation must produce exactly 1,480 predictions.")

    atomic_write_jsonl(predictions_path, evaluation["predictions"])
    result = {
        "status": "COMPLETE",
        "evaluated_at_utc": utc_now(),
        "model_id": model_id,
        "outer_fold": partitions.outer_fold,
        "selected_epochs": selected_epochs,
        "outer_test_samples": 1480,
        "outer_test_ctc_loss": evaluation["ctc_loss"],
        "outer_test_metrics": evaluation["aggregate"],
        "outer_test_controlled_training": False,
        "decoder": "greedy_valid_length_truncated_v1",
    }
    atomic_write_json(result_path, result)
    marker = {
        "status": "COMPLETE",
        "outer_test_evaluation_count": 1,
        "outer_test_samples": 1480,
        "predictions_sha256": sha256_file(predictions_path),
        "outer_result_sha256": sha256_file(result_path),
        "created_at_utc": utc_now(),
    }
    atomic_write_json(marker_path, marker)
    return result


def _parent_training_log(
    inner_log_path: Path,
    refit_log_path: Path,
) -> list[dict]:
    output = []
    for phase, path in (
        ("inner_selection", inner_log_path),
        ("full_refit", refit_log_path),
    ):
        for source in read_csv_rows(path):
            output.append({"phase": phase, **source})
    return output


def finalize_parent_run(
    *,
    parent_directory: Path,
    parent_identity: ExperimentIdentity,
    refit_identity: ExperimentIdentity,
    inner_directory: Path,
    refit_directory: Path,
    combined_config: Mapping[str, object],
    split_sha256: str,
    selection_result: Mapping[str, object],
    refit_result: Mapping[str, object],
    outer_result: Mapping[str, object],
    environment: Mapping[str, object],
) -> None:
    import torch

    _write_phase_static_artifacts(
        parent_directory,
        parent_identity,
        combined_config=combined_config,
        split_sha256=split_sha256,
    )
    atomic_write_json(parent_directory / "environment.json", dict(environment))

    combined_log = _parent_training_log(
        inner_directory / "train_log.csv",
        refit_directory / "train_log.csv",
    )
    combined_fields = tuple(
        dict.fromkeys(
            key
            for row in combined_log
            for key in row
        )
    )
    atomic_write_csv(
        parent_directory / "train_log.csv",
        combined_log,
        combined_fields,
    )
    atomic_write_json(
        parent_directory / "metrics.json",
        {
            "identity_fingerprint": parent_identity.fingerprint,
            "selection": dict(selection_result),
            "refit": dict(refit_result),
            "outer_evaluation": dict(outer_result),
        },
    )

    refit_checkpoint = load_torch_checkpoint(
        refit_directory / "last.pt",
        refit_identity,
        map_location="cpu",
    )
    parent_checkpoint = build_checkpoint_envelope(
        parent_identity,
        epoch=int(refit_checkpoint["epoch"]),
        selected_epochs=int(selection_result["selected_epoch"]),
        model_state=refit_checkpoint["model_state"],
        optimizer_state=refit_checkpoint["optimizer_state"],
        scheduler_state=refit_checkpoint["scheduler_state"],
        amp_scaler_state=refit_checkpoint["amp_scaler_state"],
        rng_state=refit_checkpoint["rng_state"],
    )
    parent_checkpoint["training_state"] = {
        "selection": dict(selection_result),
        "refit": dict(refit_result),
        "outer_result": dict(outer_result),
    }
    atomic_torch_save(
        parent_checkpoint,
        parent_directory / "last.pt",
        parent_identity,
    )
    atomic_torch_save(
        parent_checkpoint,
        parent_directory / "best.pt",
        parent_identity,
    )
    write_checkpoint_metadata(
        parent_directory / "last_checkpoint.json",
        parent_checkpoint,
        parent_identity,
    )

    events_path = parent_directory / "events.jsonl"
    existing_events = read_registry_events(events_path)
    final_status = existing_events[-1]["status"] if existing_events else None
    final_event = existing_events[-1]["event"] if existing_events else None

    if final_status != "COMPLETE":
        if final_event != "EVAL":
            append_registry_event(
                events_path,
                parent_identity,
                event="EVAL",
                status="RUNNING",
                details={
                    "outer_test_samples": outer_result["outer_test_samples"],
                    "outer_test_evaluation_count": 1,
                },
            )
        append_registry_event(
            events_path,
            parent_identity,
            event="COMPLETE",
            status="COMPLETE",
            details={
                "selected_epoch": selection_result["selected_epoch"],
                "outer_test_samples": outer_result["outer_test_samples"],
            },
        )
    write_completion_marker(parent_directory, parent_identity)


def _load_json_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _rows_for_evaluation(
    registry_rows: Sequence[Mapping[str, object]],
    parent_experiment_id: str,
) -> dict[str, dict]:
    matches = [
        dict(row)
        for row in registry_rows
        if row["parent_experiment_id"] == parent_experiment_id
    ]
    by_phase = {row["phase"]: row for row in matches}
    if set(by_phase) != set(REGISTRY_PHASES):
        raise RuntimeError("Evaluation registry phase set changed.")
    return by_phase


def _append_parent_start_or_resume(
    parent_directory: Path,
    identity: ExperimentIdentity,
    *,
    physical_batch_size: int,
) -> None:
    parent_directory.mkdir(parents=True, exist_ok=True)
    write_or_validate_identity(parent_directory, identity)
    _event_start_or_resume(
        parent_directory / "events.jsonl",
        identity,
        details={
            "physical_batch_size": physical_batch_size,
            "outer_test_loaded": False,
        },
    )


def _preflight_memory(
    *,
    model_id: str,
    training_config: Mapping[str, object],
    store: RawCacheStore,
    targets: TargetCatalog,
    partitions: KnownSignerPartitions,
    physical_batch_size: int,
    device,
    use_amp: bool,
    seed: int,
) -> None:
    import torch

    from src.models.factory import build_model

    set_deterministic_runtime_seed(torch, seed)
    normalization = fit_normalization_for_sample_ids(store, partitions.inner_train)
    dataset = ASLSequenceDataset(
        store,
        targets,
        partitions.inner_train,
        normalization_stats=normalization,
    )
    longest = sorted(
        range(len(dataset)),
        key=lambda index: store.get_entry(dataset.sample_ids[index]).timesteps,
        reverse=True,
    )[:physical_batch_size]
    items = [dataset[index] for index in longest]
    model = build_model(model_id).to(device)
    optimizer = _build_optimizer(model, training_config)
    scaler = _make_grad_scaler(torch, use_amp)
    optimizer.zero_grad(set_to_none=True)
    loss, _, _ = _model_batch_loss(
        model,
        items,
        device=device,
        use_amp=use_amp,
    )
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        float(training_config["gradient_clip_norm"]),
        error_if_nonfinite=True,
    )
    scaler.step(optimizer)
    scaler.update()
    del model, optimizer, scaler, dataset, normalization, items
    torch.cuda.empty_cache()


def is_out_of_memory_error(error: BaseException) -> bool:
    message = str(error).lower()
    markers = (
        "out of memory",
        "cannot allocate memory",
        "not enough memory",
        "defaultcpuallocator",
    )
    return any(marker in message for marker in markers)


def run_one_known_signer_evaluation(
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

    project_root = Path(project_root).resolve()
    artifact_root = ensure_external_artifact_root(project_root, artifact_root)
    live_registry_path = Path(live_registry_path)
    all_rows = read_csv_rows(live_registry_path)
    phase_rows = _rows_for_evaluation(
        all_rows,
        str(evaluation_row["parent_experiment_id"]),
    )

    model_id = str(evaluation_row["model_id"])
    outer_fold = int(evaluation_row["outer_fold"])
    model_config_path = project_root / "configs" / "models" / f"{model_id}.yaml"
    runner_config_path = (
        project_root / "configs" / "experiments" / "stage32_runner.json"
    )
    model_config = _load_json_config(model_config_path)
    runner_config = validate_runner_configuration(_load_json_config(runner_config_path))
    training_config = model_config["training"]
    combined_config = {
        "model": model_config,
        "runner": runner_config,
        "evaluation": dict(evaluation_row),
    }
    partitions = load_known_signer_partitions(
        project_root
        / "artifacts"
        / "splits"
        / "stage32_known_inner_assignment.csv",
        outer_fold,
    )
    split_sha256 = require_recorded_hash(
        project_root
        / "artifacts"
        / "splits"
        / "stage32_known_inner_assignment.csv",
        project_root
        / "artifacts"
        / "splits"
        / "stage32_known_inner_assignment.sha256",
    )
    store = RawCacheStore(
        raw_cache_path,
        project_root
        / "artifacts"
        / "cache"
        / "stage17_raw_cache_index.csv",
    )
    targets = TargetCatalog(
        project_root
        / "artifacts"
        / "labels"
        / "stage10_sentence_targets.csv"
    )
    id_to_token = _load_json_config(
        project_root / "artifacts" / "labels" / "stage10_id_to_token.json"
    )
    parent_directory = artifact_root / str(evaluation_row["artifact_path"])
    parent_identity = build_experiment_identity(
        phase_rows["outer_evaluation"],
        code_commit,
    )
    inner_identity = build_experiment_identity(
        phase_rows["inner_selection"],
        code_commit,
    )
    refit_identity = build_experiment_identity(
        phase_rows["full_refit"],
        code_commit,
    )

    completion_path = parent_directory / "completion.json"
    if completion_path.exists():
        for phase in REGISTRY_PHASES[1:] + REGISTRY_PHASES[:1]:
            registry_id = phase_rows[phase]["registry_id"]
            current = {
                row["registry_id"]: row
                for row in read_csv_rows(live_registry_path)
            }[registry_id]["status"]
            if current != "COMPLETE":
                if current in {"PLANNED", "INTERRUPTED"}:
                    set_live_registry_status(
                        live_registry_path,
                        registry_id,
                        "RUNNING",
                    )
                set_live_registry_status(
                    live_registry_path,
                    registry_id,
                    "COMPLETE",
                )
        return {
            "status": "SKIP_COMPLETE",
            "experiment_id": parent_identity.experiment_id,
        }

    configured_batch = int(training_config["physical_batch_size"])
    candidates = physical_batch_candidates(configured_batch)
    precision_config = runner_config["precision"]
    use_amp = bool(precision_config["automatic_mixed_precision"])
    device = torch.device("cuda:0")

    parent_status = phase_rows["outer_evaluation"]["status"]
    if parent_status in {"PLANNED", "INTERRUPTED"}:
        set_live_registry_status(
            live_registry_path,
            phase_rows["outer_evaluation"]["registry_id"],
            "RUNNING",
        )

    last_error = None
    for attempt_index, physical_batch in enumerate(candidates, start=1):
        attempt_directory = (
            parent_directory
            / "attempts"
            / f"attempt_{attempt_index:02d}_bs{physical_batch:02d}"
        )
        attempt_state_path = attempt_directory / "attempt_state.json"
        if attempt_state_path.exists():
            state = read_json(attempt_state_path)
            if state.get("status") == "FAILED_OOM":
                continue

        attempt_directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            attempt_state_path,
            {
                "status": "RUNNING",
                "attempt": attempt_index,
                "configured_physical_batch_size": configured_batch,
                "physical_batch_size": physical_batch,
                "effective_batch_size": configured_batch,
                "gradient_accumulation_steps": gradient_accumulation_steps(
                    configured_batch, physical_batch
                ),
                "started_at_utc": utc_now(),
            },
        )
        _append_parent_start_or_resume(
            parent_directory,
            parent_identity,
            physical_batch_size=physical_batch,
        )

        inner_registry_id = phase_rows["inner_selection"]["registry_id"]
        refit_registry_id = phase_rows["full_refit"]["registry_id"]
        current_live = {
            row["registry_id"]: row for row in read_csv_rows(live_registry_path)
        }
        if current_live[inner_registry_id]["status"] in {"PLANNED", "INTERRUPTED"}:
            set_live_registry_status(live_registry_path, inner_registry_id, "RUNNING")

        try:
            _preflight_memory(
                model_id=model_id,
                training_config=training_config,
                store=store,
                targets=targets,
                partitions=partitions,
                physical_batch_size=physical_batch,
                device=device,
                use_amp=use_amp,
                seed=int(evaluation_row["seed"]),
            )
            inner_directory = attempt_directory / "inner_selection"
            refit_directory = attempt_directory / "full_refit"
            selection_result = run_inner_selection(
                phase_directory=inner_directory,
                identity=inner_identity,
                model_id=model_id,
                training_config=training_config,
                runner_config=runner_config,
                combined_config=combined_config,
                split_sha256=split_sha256,
                store=store,
                targets=targets,
                partitions=partitions,
                id_to_token=id_to_token,
                physical_batch_size=physical_batch,
                device=device,
                use_amp=use_amp,
            )
            current_live = {
                row["registry_id"]: row for row in read_csv_rows(live_registry_path)
            }
            if current_live[refit_registry_id]["status"] in {"PLANNED", "INTERRUPTED"}:
                set_live_registry_status(live_registry_path, refit_registry_id, "RUNNING")

            refit_result = run_full_refit(
                phase_directory=refit_directory,
                identity=refit_identity,
                model_id=model_id,
                selected_epochs=int(selection_result["selected_epoch"]),
                training_config=training_config,
                runner_config=runner_config,
                combined_config=combined_config,
                split_sha256=split_sha256,
                store=store,
                targets=targets,
                partitions=partitions,
                physical_batch_size=physical_batch,
                device=device,
                use_amp=use_amp,
            )
            outer_result = evaluate_outer_test_once(
                parent_directory=parent_directory,
                refit_directory=refit_directory,
                refit_identity=refit_identity,
                model_id=model_id,
                selected_epochs=int(selection_result["selected_epoch"]),
                store=store,
                targets=targets,
                partitions=partitions,
                id_to_token=id_to_token,
                physical_batch_size=physical_batch,
                device=device,
                use_amp=use_amp,
            )
            finalize_parent_run(
                parent_directory=parent_directory,
                parent_identity=parent_identity,
                refit_identity=refit_identity,
                inner_directory=inner_directory,
                refit_directory=refit_directory,
                combined_config=combined_config,
                split_sha256=split_sha256,
                selection_result=selection_result,
                refit_result=refit_result,
                outer_result=outer_result,
                environment={
                    **environment,
                    "execution_precision": precision_config["mode"],
                    "automatic_mixed_precision": use_amp,
                },
            )
            atomic_write_json(
                attempt_state_path,
                {
                    "status": "SUCCEEDED",
                    "attempt": attempt_index,
                    "configured_physical_batch_size": configured_batch,
                    "physical_batch_size": physical_batch,
                    "effective_batch_size": configured_batch,
                    "gradient_accumulation_steps": gradient_accumulation_steps(
                        configured_batch, physical_batch
                    ),
                    "completed_at_utc": utc_now(),
                },
            )
            for registry_id in (
                inner_registry_id,
                refit_registry_id,
                phase_rows["outer_evaluation"]["registry_id"],
            ):
                live_by_id = {
                    row["registry_id"]: row
                    for row in read_csv_rows(live_registry_path)
                }
                if live_by_id[registry_id]["status"] != "COMPLETE":
                    set_live_registry_status(live_registry_path, registry_id, "COMPLETE")

            return {
                "status": "COMPLETE",
                "experiment_id": parent_identity.experiment_id,
                "model_id": model_id,
                "outer_fold": outer_fold,
                "selected_epoch": selection_result["selected_epoch"],
                "physical_batch_size": physical_batch,
                "outer_test_metrics": outer_result["outer_test_metrics"],
            }

        except BaseException as exc:
            if not is_out_of_memory_error(exc):
                raise

            last_error = exc
            atomic_write_json(
                attempt_state_path,
                {
                    "status": "FAILED_OOM",
                    "attempt": attempt_index,
                    "configured_physical_batch_size": configured_batch,
                    "physical_batch_size": physical_batch,
                    "effective_batch_size": configured_batch,
                    "verified_error": type(exc).__name__,
                    "verified_error_message": str(exc),
                    "restart_from_seed_required": True,
                    "failed_at_utc": utc_now(),
                },
            )
            if physical_batch == 1:
                raise RuntimeError(
                    "Verified OOM persisted at physical batch size 1."
                ) from exc
            torch.cuda.empty_cache()

    raise RuntimeError("No physical batch candidate completed.") from last_error


def verify_frozen_inputs(project_root: str | Path) -> dict:
    project_root = Path(project_root).resolve()
    hashes = {}
    pairs = {
        "stage32_assignment": (
            "artifacts/splits/stage32_known_inner_assignment.csv",
            "artifacts/splits/stage32_known_inner_assignment.sha256",
        ),
        "stage32_run_plan": (
            "artifacts/experiments/stage32_known_run_plan.json",
            "artifacts/experiments/stage32_known_run_plan.sha256",
        ),
        "stage32_initial_registry": (
            "artifacts/registry/run_registry.csv",
            "artifacts/registry/stage32_initial_run_registry.sha256",
        ),
        "stage17_cache_index": (
            "artifacts/cache/stage17_raw_cache_index.csv",
            "artifacts/cache/stage17_raw_cache_index.sha256",
        ),
        "stage31_config_manifest": (
            "artifacts/config/stage31_config_hashes.json",
            "artifacts/config/stage31_config_hashes.sha256",
        ),
        "stage31_smoke_contract": (
            "artifacts/training/stage31_smoke_freeze_contract.json",
            "artifacts/training/stage31_smoke_freeze_contract.sha256",
        ),
        "stage13_outer_split": (
            "artifacts/splits/stage13_known_signer_5fold.csv",
            "artifacts/splits/stage13_known_signer_5fold.sha256",
        ),
        "stage10_gloss_vocabulary": (
            "artifacts/labels/stage10_gloss_vocabulary.json",
            "artifacts/labels/stage10_gloss_vocabulary.sha256",
        ),
    }
    for name, (relative_data, relative_hash) in pairs.items():
        hashes[name] = require_recorded_hash(
            project_root / relative_data,
            project_root / relative_hash,
        )

    plan = read_json(
        project_root / "artifacts" / "experiments" / "stage32_known_run_plan.json"
    )
    if len(plan["evaluations"]) != 25 or len(plan["registry_rows"]) != 75:
        raise RuntimeError("Stage-32 run plan dimensions changed.")
    if plan["freeze_boundary"]["required_runner_tag_before_scoring"] != RUNNER_TAG:
        raise RuntimeError("Stage-32 run-plan tag requirement changed.")

    config_manifest = read_json(
        project_root / "artifacts" / "config" / "stage31_config_hashes.json"
    )
    if config_manifest.get("stage") != 31 or config_manifest.get("config_count") != 11:
        raise RuntimeError("Stage-31 config manifest changed.")
    config_hashes = {
        str(entry["path"]).replace("\\", "/"): str(entry["sha256"])
        for entry in config_manifest["configs"]
    }
    for relative, expected in config_hashes.items():
        if sha256_file(project_root / relative) != expected:
            raise RuntimeError(f"Frozen Stage-31 config changed: {relative}.")

    for evaluation in plan["evaluations"]:
        model_id = str(evaluation["model_id"])
        run_config_payload = {
            "data_config_sha256": config_hashes["configs/data.yaml"],
            "experiment_config_sha256": config_hashes[
                "configs/experiments/known_5fold.yaml"
            ],
            "model_config_sha256": config_hashes[f"configs/models/{model_id}.yaml"],
            "stage13_split_sha256": hashes["stage13_outer_split"],
            "stage32_inner_split_sha256": hashes["stage32_assignment"],
            "model_id": model_id,
            "outer_fold": int(evaluation["outer_fold"]),
            "seed": int(evaluation["seed"]),
        }
        if canonical_json_sha256(run_config_payload) != evaluation["config_sha256"]:
            raise RuntimeError(
                f"Frozen Stage-32 run identity changed for "
                f"{evaluation['experiment_id']}."
            )
        if evaluation["data_manifest_sha256"] != hashes["stage17_cache_index"]:
            raise RuntimeError("Stage-32 data-manifest identity changed.")
        if evaluation["label_vocab_sha256"] != hashes["stage10_gloss_vocabulary"]:
            raise RuntimeError("Stage-32 label-vocabulary identity changed.")

    return hashes


def build_dry_run_report(project_root: str | Path) -> dict:
    project_root = Path(project_root).resolve()
    frozen_hashes = verify_frozen_inputs(project_root)
    runner_config = validate_runner_configuration(
        read_json(project_root / "configs" / "experiments" / "stage32_runner.json")
    )
    plan = read_json(
        project_root / "artifacts" / "experiments" / "stage32_known_run_plan.json"
    )
    registry_rows = read_csv_rows(
        project_root / "artifacts" / "registry" / "run_registry.csv"
    )

    expected_order = [
        (fold, model_id)
        for fold in OUTER_FOLDS
        for model_id in MODEL_ORDER
    ]
    observed_order = [
        (int(row["outer_fold"]), row["model_id"])
        for row in plan["evaluations"]
    ]
    if observed_order != expected_order:
        raise RuntimeError("Stage-32 evaluation order is not fold-major.")

    phase_counts = {
        phase: sum(row["phase"] == phase for row in registry_rows)
        for phase in REGISTRY_PHASES
    }
    if any(count != 25 for count in phase_counts.values()):
        raise RuntimeError("Stage-32 registry phase counts changed.")
    if any(row["status"] != "PLANNED" for row in registry_rows):
        raise RuntimeError("Tracked Stage-32 registry must remain fully PLANNED.")

    partitions = [
        load_known_signer_partitions(
            project_root
            / "artifacts"
            / "splits"
            / "stage32_known_inner_assignment.csv",
            fold,
        ).to_summary()
        for fold in OUTER_FOLDS
    ]

    config_manifest = read_json(
        project_root / "artifacts" / "config" / "stage31_config_hashes.json"
    )
    config_hashes = {
        str(entry["path"]).replace("\\", "/"): str(entry["sha256"])
        for entry in config_manifest["configs"]
    }
    for relative, expected in config_hashes.items():
        if sha256_file(project_root / relative) != expected:
            raise RuntimeError(f"Frozen Stage-31 config changed: {relative}.")

    stage13_split_sha256 = require_recorded_hash(
        project_root / "artifacts" / "splits" / "stage13_known_signer_5fold.csv",
        project_root / "artifacts" / "splits" / "stage13_known_signer_5fold.sha256",
    )
    model_hashes = {}
    for evaluation in plan["evaluations"]:
        model_id = evaluation["model_id"]
        relative_model_config = f"configs/models/{model_id}.yaml"
        model_config_hash = config_hashes[relative_model_config]
        run_config_payload = {
            "data_config_sha256": config_hashes["configs/data.yaml"],
            "experiment_config_sha256": config_hashes[
                "configs/experiments/known_5fold.yaml"
            ],
            "model_config_sha256": model_config_hash,
            "stage13_split_sha256": stage13_split_sha256,
            "stage32_inner_split_sha256": frozen_hashes["stage32_assignment"],
            "model_id": model_id,
            "outer_fold": int(evaluation["outer_fold"]),
            "seed": int(evaluation["seed"]),
        }
        if canonical_json_sha256(run_config_payload) != evaluation["config_sha256"]:
            raise RuntimeError(f"Frozen {model_id} run-identity hash mismatch.")
        model_hashes[model_id] = model_config_hash

    mock_history = [
        {
            "epoch": 1,
            "validation_wer": 0.4,
            "validation_exact_sentence_accuracy": 0.2,
            "validation_ctc_loss": 2.0,
        },
        {
            "epoch": 2,
            "validation_wer": 0.3,
            "validation_exact_sentence_accuracy": 0.25,
            "validation_ctc_loss": 1.9,
        },
        {
            "epoch": 3,
            "validation_wer": 0.3,
            "validation_exact_sentence_accuracy": 0.30,
            "validation_ctc_loss": 2.1,
        },
    ]
    selected_mock = select_best_epoch(mock_history)
    if selected_mock["epoch"] != 3:
        raise RuntimeError("Stage-32 selection tie-break dry run failed.")

    premature_paths = []
    runs_root = project_root / "artifacts" / "runs"
    if runs_root.exists():
        for pattern in ("*.pt", "predictions.jsonl", "metrics.json"):
            premature_paths.extend(str(path) for path in runs_root.rglob(pattern))
    if premature_paths:
        raise RuntimeError("Premature Stage-32 scoring artifacts exist in the repository.")

    synthetic_commit = STAGE31_COMMIT
    identity_count = 0
    for row in registry_rows:
        build_experiment_identity(row, synthetic_commit)
        identity_count += 1

    report = {
        "project_generation": PROJECT_GENERATION,
        "stage": 32,
        "substage": "B",
        "created_at_utc": utc_now(),
        "mode": "static_dry_run_no_training_no_scoring",
        "frozen_hashes": frozen_hashes,
        "runner_config_sha256": sha256_file(
            project_root / "configs" / "experiments" / "stage32_runner.json"
        ),
        "model_config_hashes": dict(sorted(model_hashes.items())),
        "matrix": {
            "outer_evaluations": len(plan["evaluations"]),
            "registry_rows": len(registry_rows),
            "validated_identities": identity_count,
            "phase_counts": phase_counts,
            "fold_major": True,
            "first": plan["evaluations"][0]["experiment_id"],
            "last": plan["evaluations"][-1]["experiment_id"],
        },
        "partitions": partitions,
        "selection": {
            "tie_break_mock_selected_epoch": selected_mock["epoch"],
            "g1_full_epoch_cap": True,
            "compact_minimum_epoch": runner_config["selection"][
                "compact_early_stopping_minimum_epoch"
            ],
            "scheduler_monitor": "validation_ctc_loss",
        },
        "refit": {
            "fresh_reset_required": True,
            "training_pool": "all_non_test_outer_training_samples",
            "validation_used": False,
            "exact_selected_epochs": True,
        },
        "leakage_barrier": {
            "outer_test_constructed_during_dry_run": False,
            "training_started": False,
            "scores_computed": False,
            "checkpoints_created": False,
            "premature_repository_artifacts": len(premature_paths),
        },
        "runtime_gate": {
            "required_tag": RUNNER_TAG,
            "tag_must_be_annotated": True,
            "tag_commit_must_equal_head": True,
            "clean_worktree_required": True,
            "external_artifact_root_required": True,
            "required_gpu_name_contains": "T4",
            "execution_precision": runner_config["precision"]["mode"],
            "automatic_mixed_precision": runner_config["precision"][
                "automatic_mixed_precision"
            ],
        },
        "passed": True,
    }
    return report


def run_next_evaluation(
    *,
    project_root: str | Path,
    artifact_root: str | Path,
    raw_cache_path: str | Path,
) -> dict:
    project_root = Path(project_root).resolve()
    artifact_root = ensure_external_artifact_root(project_root, artifact_root)
    verify_frozen_inputs(project_root)
    freeze = verify_git_freeze(project_root)
    environment = verify_t4_runtime()

    raw_cache_path = Path(raw_cache_path).resolve()
    expected_cache_hash = read_recorded_sha256(
        project_root / "artifacts" / "cache" / "stage17_raw_cache.sha256"
    )
    actual_cache_hash = sha256_file(raw_cache_path)
    if actual_cache_hash != expected_cache_hash:
        raise RuntimeError("Stage-17 raw binary cache hash mismatch.")

    live_registry_path = (
        artifact_root / "artifacts" / "registry" / "run_registry.csv"
    )
    rows = initialize_or_validate_live_registry(
        project_root / "artifacts" / "registry" / "run_registry.csv",
        live_registry_path,
        code_commit=freeze["head_commit"],
    )
    evaluation = next_evaluation_row(rows)
    if evaluation is None:
        return {
            "status": "ALL_COMPLETE",
            "completed_outer_evaluations": 25,
        }

    return run_one_known_signer_evaluation(
        project_root=project_root,
        artifact_root=artifact_root,
        raw_cache_path=raw_cache_path,
        evaluation_row=evaluation,
        live_registry_path=live_registry_path,
        code_commit=freeze["head_commit"],
        environment={
            **environment,
            "git": freeze,
            "stage17_raw_cache_sha256": actual_cache_hash,
        },
    )
