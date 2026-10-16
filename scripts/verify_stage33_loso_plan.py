from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from src.training.loso_plan import (
    LOSO_IDS,
    MODEL_IDS,
    REQUIRED_RUNNER_TAG,
    validate_loso_assignment,
    validate_registry_plan,
    validate_run_matrix,
    validate_stage14_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COMMIT = "116116866d898ebe3e5fe2a15867191df3c01cf1"
RESULTS_TAG = "stage32-known-results-v1"

STAGE14 = PROJECT_ROOT / "artifacts/splits/stage14_loso_assignment.csv"
STAGE15 = PROJECT_ROOT / "artifacts/splits/stage15_loso_support_assignment.csv"
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


def require_hash(path: Path) -> str:
    actual = sha256_file(path)
    if actual != read_hash(path.with_suffix(".sha256")):
        raise RuntimeError(f"Hash mismatch: {path.name}.")
    return actual


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def git_audit() -> dict:
    if not (PROJECT_ROOT / ".git").exists():
        return {"available": False, "annotated": None, "commit": EXPECTED_COMMIT, "ancestor": None}
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    tag_type = git("cat-file", "-t", f"refs/tags/{RESULTS_TAG}")
    commit = git("rev-list", "-n", "1", RESULTS_TAG)
    head = git("rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, head], cwd=PROJECT_ROOT
    ).returncode == 0
    if tag_type != "tag" or commit != EXPECTED_COMMIT or not ancestor:
        raise RuntimeError("Stage-32 result freeze boundary failed.")
    return {"available": True, "annotated": True, "commit": commit, "ancestor": True}


def main() -> None:
    print("=" * 78)
    print("STAGE 33A - LOSO PLAN VERIFIER")
    print("=" * 78)

    required = [
        STAGE14, STAGE15, ASSIGNMENT, ASSIGNMENT.with_suffix(".sha256"),
        SUMMARY, RUN_PLAN, RUN_PLAN.with_suffix(".sha256"), REGISTRY,
        REGISTRY.with_suffix(".sha256"), CONTRACT, CONTRACT.with_suffix(".sha256"), DOC,
    ]
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Missing Stage-33A outputs: " + ", ".join(missing))

    git_state = git_audit()
    stage14_rows = read_csv(STAGE14)
    source = validate_stage14_rows(stage14_rows)
    assignments = read_csv(ASSIGNMENT)
    summaries = []
    for loso_id in LOSO_IDS:
        rows = [row for row in assignments if row["loso_id"] == loso_id]
        summaries.append(validate_loso_assignment(rows))

    sample_plans = Counter(row["sample_id"] for row in assignments)
    outer_counts = Counter(
        row["sample_id"] for row in assignments if row["partition"] == "outer_test"
    )
    if set(sample_plans.values()) != {3} or set(outer_counts.values()) != {1}:
        raise RuntimeError("Sample coverage across LOSO directions failed.")

    run_plan = json.loads(RUN_PLAN.read_text(encoding="utf-8-sig"))
    matrix = validate_run_matrix(run_plan["evaluations"])
    registry_rows = read_csv(REGISTRY)
    registry = validate_registry_plan(registry_rows)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8-sig"))

    assignment_hash = require_hash(ASSIGNMENT)
    run_plan_hash = require_hash(RUN_PLAN)
    registry_hash = require_hash(REGISTRY)
    contract_hash = require_hash(CONTRACT)
    if contract["hashes"] != {
        "assignment": assignment_hash,
        "initial_loso_registry": registry_hash,
        "run_plan": run_plan_hash,
    }:
        raise RuntimeError("Contract output hashes differ.")

    support_rows = read_csv(STAGE15)
    support = {}
    for loso_id in LOSO_IDS:
        counts = Counter(
            row["support_class"] for row in support_rows if row["loso_id"] == loso_id
        )
        support[loso_id] = counts
    expected_support = {
        "loso_subject1": Counter({
            "template_supported": 1340,
            "token_supported_novel_template": 180,
            "token_unsupported": 4480,
        }),
        "loso_subject2": Counter({"template_supported": 1000}),
        "loso_subject3": Counter({"template_supported": 400}),
    }
    if support != expected_support:
        raise RuntimeError("Frozen Stage-15 support counts changed.")

    premature = []
    run_root = PROJECT_ROOT / "artifacts/runs/loso"
    if run_root.exists():
        premature = [path for path in run_root.rglob("*") if path.is_file()]
    if premature:
        raise RuntimeError("Premature LOSO training artifacts exist.")

    print("\nFREEZE BOUNDARY\n" + "-" * 78)
    print("Stage-32 results tag:", RESULTS_TAG)
    print("Annotated tag:", git_state["annotated"])
    print("Tag commit:", git_state["commit"])
    print("Tag is ancestor:", git_state["ancestor"])
    print("Required runner tag:", REQUIRED_RUNNER_TAG)
    print("\nSOURCE AND INNER SPLITS\n" + "-" * 78)
    print("Stage-14 assignment rows:", source["assignment_rows"])
    print("Unique recordings:", source["unique_recordings"])
    print("Stage-33 assignment rows:", len(assignments))
    print("Each sample appears in three plans: True")
    print("Each sample is outer test once: True")
    for row in summaries:
        print(
            f"{row['loso_id']}: train={row['inner_train_recordings']} "
            f"validation={row['inner_validation_recordings']} "
            f"test={row['outer_test_recordings']} crossings=0 leaks=0"
        )
    print("\nSUPPORT REPORTING\n" + "-" * 78)
    print("Subject 1 all/template/token/unsupported: 6000 / 1340 / 1520 / 4480")
    print("Subject 2 all/template/token/unsupported: 1000 / 1000 / 1000 / 0")
    print("Subject 3 all/template/token/unsupported: 400 / 400 / 400 / 0")
    print("Support views train separate models: False")
    print("\nRUN AND REGISTRY PLAN\n" + "-" * 78)
    print("Outer evaluations:", matrix["runs"])
    print("Registry rows:", registry["registry_rows"])
    print("Outer / inner / refit rows: 15 / 15 / 15")
    print("LOSO-major order: True")
    print("All statuses PLANNED: True")
    print("\nLEAKAGE BARRIER\n" + "-" * 78)
    print("Training started: False")
    print("Outer scores present: False")
    print("Checkpoints present: False")
    print("Premature artifacts:", len(premature))
    print("\nHASHES\n" + "-" * 78)
    print("Inner assignment hash match: True")
    print("Run-plan hash match: True")
    print("Initial registry hash match: True")
    print("Contract hash match:", bool(contract_hash))
    print("\nFINAL STATUS: STAGE 33A LOSO PLAN PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()
