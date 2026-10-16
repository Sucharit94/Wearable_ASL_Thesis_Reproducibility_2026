from google.colab import drive

drive.mount("/content/drive")

import hashlib
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


DRIVE_TRANSFER = Path("/content/drive/MyDrive/gu_asl_stage39_transfer")
DRIVE_BUNDLE = DRIVE_TRANSFER / "gu_asl_stage39b_v2_generator.bundle"
DRIVE_HASH_FILE = DRIVE_TRANSFER / "gu_asl_stage39b_v2_generator.bundle.sha256"
EXPECTED_BUNDLE_HASH = "0c9ce5fb7e42c3e6ae77358f31df94057580b639c0fc741ed3d0b765fadf2b4d"
EXPECTED_COMMIT = "c66f7a1685160ceb90da59956e9691aa48a6c92f"
FROZEN_TAG = "stage39b-v2-generator-v1"
LOCAL_BUNDLE = Path("/content/gu_asl_stage39b_v2_generator.bundle")
PROJECT_ROOT = Path("/content/gu_asl_stage39b_v2_workspace")
OUTPUT_ROOT = Path("/content/drive/MyDrive/gu_asl_stage39_runs")
RESPONSE_PATH = OUTPUT_ROOT / "stage39b_v2_responses.jsonl"
EXPECTED_RESPONSES = 11720


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def run_checked(command, cwd=None):
    result = subprocess.run(command, cwd=cwd, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {result.returncode}: {command}"
        )
    return result


print("=" * 78)
print("STAGE 39B V2 â€” COLAB RESTORATION AND CORRECTED L2 GENERATION")
print("=" * 78)

print("\nDRIVE INPUTS")
print("-" * 78)
print("V2 bundle exists:", DRIVE_BUNDLE.is_file())
print("V2 hash file exists:", DRIVE_HASH_FILE.is_file())

if not DRIVE_BUNDLE.is_file() or not DRIVE_HASH_FILE.is_file():
    raise FileNotFoundError(
        "Upload the V2 bundle and its SHA-256 file to gu_asl_stage39_transfer."
    )

recorded_hash = DRIVE_HASH_FILE.read_text(encoding="utf-8-sig").strip().split()[0].lower()
actual_hash = sha256_file(DRIVE_BUNDLE)

print("\nBUNDLE VERIFICATION")
print("-" * 78)
print("Expected:", EXPECTED_BUNDLE_HASH)
print("Recorded:", recorded_hash)
print("Actual:", actual_hash)
print("Recorded hash match:", recorded_hash == EXPECTED_BUNDLE_HASH)
print("Actual hash match:", actual_hash == EXPECTED_BUNDLE_HASH)

if recorded_hash != EXPECTED_BUNDLE_HASH or actual_hash != EXPECTED_BUNDLE_HASH:
    raise RuntimeError("The uploaded Stage 39B V2 bundle failed hash verification.")

os.chdir("/content")

if PROJECT_ROOT.exists():
    shutil.rmtree(PROJECT_ROOT)
if LOCAL_BUNDLE.exists():
    LOCAL_BUNDLE.unlink()

print("\nCopying the V2 bundle to local Colab storage...")
shutil.copy2(DRIVE_BUNDLE, LOCAL_BUNDLE)

if sha256_file(LOCAL_BUNDLE) != EXPECTED_BUNDLE_HASH:
    raise RuntimeError("The local V2 bundle copy failed hash verification.")

print("Restoring the frozen V2 project...")
run_checked(["git", "clone", str(LOCAL_BUNDLE), str(PROJECT_ROOT)], cwd="/content")
run_checked(["git", "checkout", FROZEN_TAG], cwd=PROJECT_ROOT)

head_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
).strip()
tag_commit = subprocess.check_output(
    ["git", "rev-list", "-n", "1", FROZEN_TAG], cwd=PROJECT_ROOT, text=True
).strip()
working_tree = subprocess.check_output(
    ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
).strip()

print("\nFROZEN PROJECT")
print("-" * 78)
print("HEAD:", head_commit)
print("Tag commit:", tag_commit)
print("Correct commit:", head_commit == EXPECTED_COMMIT)
print("Tag equals HEAD:", tag_commit == head_commit)
print("Working tree clean:", working_tree == "")

if head_commit != EXPECTED_COMMIT or tag_commit != head_commit or working_tree:
    raise RuntimeError("The restored Stage 39B V2 project failed verification.")

print("\nGPU")
print("-" * 78)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("Select a T4 GPU runtime before running Stage 39B V2.")
print("GPU:", torch.cuda.get_device_name(0))

print("\nRUNNING V2 FREEZE VERIFIER")
print("-" * 78)
run_checked(
    [sys.executable, "-m", "scripts.verify_stage39b_v2_prompt_freeze"],
    cwd=PROJECT_ROOT,
)

print("\nINSTALLING FROZEN LANGUAGE DEPENDENCIES")
print("-" * 78)
run_checked(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--disable-pip-version-check",
        "-r",
        str(PROJECT_ROOT / "requirements-stage39b.txt"),
    ],
    cwd=PROJECT_ROOT,
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
completed_before = 0
if RESPONSE_PATH.is_file():
    with RESPONSE_PATH.open("r", encoding="utf-8-sig") as handle:
        completed_before = sum(1 for line in handle if line.strip())

print("\nDURABLE V2 OUTPUT")
print("-" * 78)
print("Output file:", RESPONSE_PATH)
print("V2 responses already saved:", completed_before)
print("Expected V2 responses:", EXPECTED_RESPONSES)
print("Failed V1 file will be modified:", False)
print("Resume enabled:", True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("\n" + "=" * 78)
print("STAGE 39B V2 â€” STARTING OR RESUMING CORRECTED GENERATION")
print("=" * 78)
print("Generator: google/flan-t5-base")
print("Prompt: validated plain-English V2")
print("Models: G1, C1, C2, C3 and C4")
print("Unique requests: 11720")
print("Batch size: 16")
print("Started UTC:", datetime.now(timezone.utc).isoformat())
print("The existing failed V1 output remains untouched.")
print("If Colab restarts, rerun this same complete cell to resume.")
print("=" * 78 + "\n")

generation_result = subprocess.run(
    [
        sys.executable,
        "-m",
        "scripts.run_stage39b_v2_generation",
        "--output-root",
        str(OUTPUT_ROOT),
        "--batch-size",
        "16",
    ],
    cwd=PROJECT_ROOT,
    text=True,
)

print("\n" + "=" * 78)
print("V2 PROCESS RETURN CODE:", generation_result.returncode)
print("Finished UTC:", datetime.now(timezone.utc).isoformat())
print("=" * 78)

if generation_result.returncode != 0:
    raise RuntimeError(
        "Stage 39B V2 stopped with an error. Keep all Drive files and send the traceback."
    )

completed_after = 0
if RESPONSE_PATH.is_file():
    with RESPONSE_PATH.open("r", encoding="utf-8-sig") as handle:
        completed_after = sum(1 for line in handle if line.strip())

print("\nFINAL SAVED STATE")
print("-" * 78)
print("Completed V2 responses:", completed_after)
print("Expected V2 responses:", EXPECTED_RESPONSES)
print("V2 response file:", RESPONSE_PATH)
print("Failed V1 response file retained:", True)

if completed_after != EXPECTED_RESPONSES:
    raise RuntimeError("V2 is incomplete. Rerun this same cell to resume.")

print("\n" + "=" * 78)
print("FINAL STATUS: STAGE 39B V2 GENERATION COMPLETED")
print("All corrected L2 responses are saved in Google Drive.")
print("No recognizer was retrained and no outer reference was exposed.")
print("=" * 78)