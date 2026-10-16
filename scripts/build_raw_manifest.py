from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_CONTRACT = (
    PROJECT_ROOT
    / "artifacts"
    / "source"
    / "SOURCE_CONTRACT.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "manifests"
)

MANIFEST_CSV = OUTPUT_DIR / "raw_manifest.csv"
SUMMARY_JSON = OUTPUT_DIR / "raw_manifest_summary.json"
MANIFEST_HASH_FILE = OUTPUT_DIR / "raw_manifest.sha256"
RECORDING_FINGERPRINT_FILE = (
    OUTPUT_DIR / "recording_set_fingerprint.sha256"
)
SUMMARY_MD = (
    PROJECT_ROOT
    / "docs"
    / "STAGE_4_RAW_MANIFEST.md"
)


SUBJECT_PATTERN = re.compile(
    r"^subject(?P<subject_number>\d+)$",
    flags=re.IGNORECASE,
)

FILENAME_PATTERN = re.compile(
    r"^(?P<sentence_id>\d+)_(?P<repetition_id>\d+)\.csv$",
    flags=re.IGNORECASE,
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def load_contract() -> dict:
    if not SOURCE_CONTRACT.exists():
        raise FileNotFoundError(
            f"Missing source contract: {SOURCE_CONTRACT}"
        )

    return json.loads(
        SOURCE_CONTRACT.read_text(encoding="utf-8-sig")
    )


def fail(message: str) -> None:
    print()
    print("=" * 78)
    print("STAGE 4 FAILED")
    print(message)
    print("=" * 78)
    sys.exit(1)


def main() -> None:
    print("=" * 78)
    print("STAGE 4 — RAW RECORDING MANIFEST + FULL SHA-256")
    print("=" * 78)

    contract = load_contract()

    raw_relative = Path(
        contract["project_raw_source"]["relative_path"]
    )

    csv_relative = Path(
        contract["project_raw_source"]["csv_root"]
    )

    raw_root = PROJECT_ROOT / raw_relative
    csv_root = PROJECT_ROOT / csv_relative

    expected_csv_count = int(
        contract["project_raw_source"]["csv_files"]
    )

    if not raw_root.exists():
        fail(f"Raw source does not exist: {raw_root}")

    if not csv_root.exists():
        fail(f"CSV root does not exist: {csv_root}")

    # ---------------------------------------------------------
    # Guard the two label files before touching CSV metadata.
    # ---------------------------------------------------------

    recognition_label = raw_root / "label_no_order.txt"
    english_label = raw_root / "label_order.txt"

    if not recognition_label.exists():
        fail("label_no_order.txt is missing.")

    if not english_label.exists():
        fail("label_order.txt is missing.")

    expected_gloss_hash = (
        contract["labels"]["recognition"]["sha256"].lower()
    )

    expected_english_hash = (
        contract["labels"]["english"]["sha256"].lower()
    )

    observed_gloss_hash = sha256_file(recognition_label)
    observed_english_hash = sha256_file(english_label)

    if observed_gloss_hash != expected_gloss_hash:
        fail(
            "label_no_order.txt SHA-256 no longer matches "
            "the Stage 3 source contract."
        )

    if observed_english_hash != expected_english_hash:
        fail(
            "label_order.txt SHA-256 no longer matches "
            "the Stage 3 source contract."
        )

    print("Source label integrity: PASS")
    print()

    # ---------------------------------------------------------
    # Discover all CSV recordings.
    # ---------------------------------------------------------

    csv_files = sorted(
        csv_root.rglob("*.csv"),
        key=lambda p: p.relative_to(csv_root)
        .as_posix()
        .lower(),
    )

    print(f"CSV files discovered: {len(csv_files):,}")
    print(f"CSV files expected:   {expected_csv_count:,}")

    if len(csv_files) != expected_csv_count:
        fail(
            f"Expected {expected_csv_count:,} CSV files but "
            f"found {len(csv_files):,}."
        )

    rows: list[dict] = []
    errors: list[str] = []

    print()
    print("Hashing all recordings...")
    print()

    for index, path in enumerate(csv_files, start=1):
        relative_to_csv = path.relative_to(csv_root)

        # Expected structure:
        #
        # csv/
        #   subject1/
        #       1_1.csv
        #
        if len(relative_to_csv.parts) != 2:
            errors.append(
                "Unexpected CSV nesting: "
                + relative_to_csv.as_posix()
            )
            continue

        subject_folder = relative_to_csv.parts[0]
        filename = relative_to_csv.parts[1]

        subject_match = SUBJECT_PATTERN.fullmatch(
            subject_folder
        )

        filename_match = FILENAME_PATTERN.fullmatch(
            filename
        )

        if subject_match is None:
            errors.append(
                f"Invalid subject folder: {subject_folder}"
            )
            continue

        if filename_match is None:
            errors.append(
                f"Invalid recording filename: {filename}"
            )
            continue

        subject_number = int(
            subject_match.group("subject_number")
        )

        sentence_id = int(
            filename_match.group("sentence_id")
        )

        repetition_id = int(
            filename_match.group("repetition_id")
        )

        subject = f"subject{subject_number}"

        # Stable identity is semantic rather than absolute-path based.
        sample_id = (
            f"gu_s{subject_number:02d}"
            f"_sent{sentence_id:03d}"
            f"_rep{repetition_id:03d}"
        )

        file_size = path.stat().st_size
        file_hash = sha256_file(path)

        project_relative = (
            path.relative_to(PROJECT_ROOT).as_posix()
        )

        rows.append(
            {
                "sample_id": sample_id,
                "subject": subject,
                "subject_number": subject_number,
                "sentence_id": sentence_id,
                "repetition_id": repetition_id,
                "filename": filename,
                "relative_path": project_relative,
                "bytes": file_size,
                "sha256": file_hash,
            }
        )

        if (
            index == 1
            or index % 250 == 0
            or index == len(csv_files)
        ):
            print(
                f"Hashed {index:>4,} / "
                f"{len(csv_files):,} recordings"
            )

    if errors:
        print()
        print("Identity parsing errors:")

        for error in errors[:50]:
            print(" -", error)

        if len(errors) > 50:
            print(
                f"... plus {len(errors) - 50} more errors."
            )

        fail(
            "One or more recordings could not be assigned "
            "an unambiguous identity."
        )

    if len(rows) != expected_csv_count:
        fail(
            "Manifest row count does not equal expected "
            "recording count."
        )

    # ---------------------------------------------------------
    # Identity uniqueness checks.
    # ---------------------------------------------------------

    sample_ids = [
        row["sample_id"]
        for row in rows
    ]

    relative_paths = [
        row["relative_path"]
        for row in rows
    ]

    if len(set(sample_ids)) != len(sample_ids):
        duplicates = [
            key
            for key, count in Counter(sample_ids).items()
            if count > 1
        ]

        fail(
            "Duplicate stable sample IDs detected: "
            + ", ".join(duplicates[:20])
        )

    if len(set(relative_paths)) != len(relative_paths):
        fail("Duplicate relative paths detected.")

    # ---------------------------------------------------------
    # Stage 3 representative recording integrity guard.
    # ---------------------------------------------------------

    representative_rel = (
        contract["representative_csv"]["relative_path"]
    ).replace("\\", "/")

    representative_expected_hash = (
        contract["representative_csv"]["sha256"].lower()
    )

    representative_manifest_path = (
        raw_relative.as_posix().rstrip("/")
        + "/"
        + representative_rel
    )

    representative_rows = [
        row
        for row in rows
        if row["relative_path"]
        == representative_manifest_path
    ]

    if len(representative_rows) != 1:
        fail(
            "Stage 3 representative CSV was not found "
            "exactly once in the Stage 4 manifest."
        )

    representative_observed_hash = (
        representative_rows[0]["sha256"]
    )

    if (
        representative_observed_hash
        != representative_expected_hash
    ):
        fail(
            "Representative CSV hash does not match "
            "Stage 3."
        )

    # ---------------------------------------------------------
    # Write deterministic manifest.
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "sample_id",
        "subject",
        "subject_number",
        "sentence_id",
        "repetition_id",
        "filename",
        "relative_path",
        "bytes",
        "sha256",
    ]

    with MANIFEST_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    # Hash the exact manifest file.
    manifest_file_hash = sha256_file(
        MANIFEST_CSV
    )

    MANIFEST_HASH_FILE.write_text(
        manifest_file_hash + "\n",
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Create a platform-independent recording-set fingerprint.
    #
    # This intentionally excludes absolute paths and timestamps.
    # ---------------------------------------------------------

    recording_digest = hashlib.sha256()

    for row in rows:
        canonical_line = (
            f"{row['relative_path']}\t"
            f"{row['bytes']}\t"
            f"{row['sha256']}\n"
        )

        recording_digest.update(
            canonical_line.encode("utf-8")
        )

    recording_fingerprint = (
        recording_digest.hexdigest()
    )

    RECORDING_FINGERPRINT_FILE.write_text(
        recording_fingerprint + "\n",
        encoding="utf-8",
    )

    subject_counts = Counter(
        row["subject"]
        for row in rows
    )

    total_csv_bytes = sum(
        int(row["bytes"])
        for row in rows
    )

    summary = {
        "project_generation": contract[
            "project_generation"
        ],
        "stage": 4,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "hash_algorithm": "SHA-256",
        "source_csv_root": (
            contract["project_raw_source"]["csv_root"]
        ),
        "csv_files_expected": expected_csv_count,
        "csv_files_hashed": len(rows),
        "unique_sample_ids": len(set(sample_ids)),
        "unique_relative_paths": len(
            set(relative_paths)
        ),
        "total_csv_bytes": total_csv_bytes,
        "subject_counts": dict(
            sorted(subject_counts.items())
        ),
        "label_integrity_guard": {
            "label_no_order_match": True,
            "label_order_match": True,
        },
        "representative_csv_guard": {
            "relative_path": representative_rel,
            "sha256_match": True,
        },
        "identity_contract": {
            "subject_folder_pattern": "subject<number>",
            "filename_pattern": "<sentence_id>_<repetition_id>.csv",
            "sample_id_pattern": (
                "gu_sNN_sentNNN_repNNN"
            ),
        },
        "raw_manifest_file": (
            "artifacts/manifests/raw_manifest.csv"
        ),
        "raw_manifest_sha256": manifest_file_hash,
        "recording_set_fingerprint_sha256": (
            recording_fingerprint
        ),
        "raw_files_modified": False,
        "next_stage": {
            "stage": 5,
            "action": (
                "Parse every CSV and perform the fresh "
                "dataset integrity audit."
            ),
        },
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Human-readable Stage 4 record.
    # ---------------------------------------------------------

    subject_lines = "\n".join(
        f"- {subject}: {count:,} recordings"
        for subject, count in sorted(
            subject_counts.items()
        )
    )

    markdown = f"""# Stage 4 — Raw Recording Manifest

## Status

All project raw CSV recordings were independently enumerated and
SHA-256 hashed.

## Source

CSV root:

`{contract["project_raw_source"]["csv_root"]}`

No external-master path is used by the manifest builder.

## Recording identity

Each recording stores:

- stable sample ID
- subject
- sentence ID
- repetition ID
- original filename
- project-relative source path
- byte size
- SHA-256

Stable sample-ID format:

`gu_sNN_sentNNN_repNNN`

## Results

Total recordings hashed: {len(rows):,}

Unique sample IDs: {len(set(sample_ids)):,}

Unique relative paths: {len(set(relative_paths)):,}

Total CSV bytes: {total_csv_bytes:,}

### Observed subject counts

{subject_lines}

## Integrity guards

`label_no_order.txt`: unchanged from Stage 3

`label_order.txt`: unchanged from Stage 3

Representative Stage 3 CSV: unchanged

## Dataset recording fingerprint

`{recording_fingerprint}`

## Manifest file SHA-256

`{manifest_file_hash}`

## Outputs

`artifacts/manifests/raw_manifest.csv`

`artifacts/manifests/raw_manifest_summary.json`

`artifacts/manifests/raw_manifest.sha256`

`artifacts/manifests/recording_set_fingerprint.sha256`

## Important boundary

Stage 4 does not parse sensor-array values, infer feature semantics,
detect duplicate content, create data splits, normalize features,
or create model inputs.

Those operations belong to later stages.
"""

    SUMMARY_MD.write_text(
        markdown,
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("STAGE 4 MANIFEST BUILD COMPLETE")
    print("=" * 78)
    print(f"Manifest rows:       {len(rows):,}")
    print(
        f"Unique sample IDs:   "
        f"{len(set(sample_ids)):,}"
    )
    print(
        f"Unique paths:        "
        f"{len(set(relative_paths)):,}"
    )
    print(
        f"Total CSV bytes:     "
        f"{total_csv_bytes:,}"
    )

    print()
    print("Observed subjects:")

    for subject, count in sorted(
        subject_counts.items()
    ):
        print(
            f"  {subject}: {count:,}"
        )

    print()
    print(
        "Manifest SHA-256:    "
        + manifest_file_hash
    )

    print(
        "Recording fingerprint:"
    )
    print(recording_fingerprint)

    print()
    print(
        "Saved:",
        MANIFEST_CSV.relative_to(PROJECT_ROOT),
    )
    print(
        "Saved:",
        SUMMARY_JSON.relative_to(PROJECT_ROOT),
    )
    print(
        "Saved:",
        MANIFEST_HASH_FILE.relative_to(
            PROJECT_ROOT
        ),
    )
    print(
        "Saved:",
        RECORDING_FINGERPRINT_FILE.relative_to(
            PROJECT_ROOT
        ),
    )
    print(
        "Saved:",
        SUMMARY_MD.relative_to(PROJECT_ROOT),
    )

    print()
    print("FINAL STATUS: STAGE 4 BUILD PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()
