"""Reproducible CPU efficiency measurements for Stage 34D."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import io
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import torch

from src.models.factory import (
    EXPECTED_TRAINABLE_PARAMETERS,
    MODEL_IDS,
    build_model,
    count_trainable_parameters,
)


MODEL_ORDER = tuple(MODEL_IDS)
BENCHMARK_SEQUENCE_LENGTH = 108
BENCHMARK_FEATURE_WIDTH = 38
BENCHMARK_BATCH_SIZES = (1, 16)
BENCHMARK_WARMUP_ITERATIONS = 2
BENCHMARK_TIMED_ITERATIONS = {1: 10, 16: 5}
BENCHMARK_SEED = 20260912
WORKER_PREFIX = "STAGE34D_WORKER_JSON="


def byte_count(value: object) -> int:
    """Return tensor bytes contained in a possibly nested value."""
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(byte_count(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(byte_count(item) for item in value)
    if hasattr(value, "logits_btc") and hasattr(value, "output_lengths"):
        return byte_count(value.logits_btc) + byte_count(value.output_lengths)
    return 0


def serialized_state_dict_bytes(model: torch.nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return int(buffer.tell())


def parameter_and_buffer_bytes(model: torch.nn.Module) -> tuple[int, int]:
    parameter_bytes = sum(
        int(parameter.numel() * parameter.element_size())
        for parameter in model.parameters()
    )
    buffer_bytes = sum(
        int(buffer.numel() * buffer.element_size())
        for buffer in model.buffers()
    )
    return parameter_bytes, buffer_bytes


def _linux_memory_bytes(field: str) -> int | None:
    status = Path("/proc/self/status")
    if not status.is_file():
        return None
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith(field + ":"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1]) * 1024
    return None


def _windows_memory_bytes(peak: bool) -> int | None:
    if os.name != "nt":
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel32.GetCurrentProcess()
    succeeded = psapi.GetProcessMemoryInfo(
        handle,
        ctypes.byref(counters),
        counters.cb,
    )
    if not succeeded:
        return None
    return int(counters.PeakWorkingSetSize if peak else counters.WorkingSetSize)


def current_rss_bytes() -> int | None:
    return _windows_memory_bytes(False) or _linux_memory_bytes("VmRSS")


def peak_rss_bytes() -> int | None:
    return _windows_memory_bytes(True) or _linux_memory_bytes("VmHWM")


def _median_and_iqr(milliseconds: Iterable[float]) -> tuple[float, float, float]:
    values = sorted(float(value) for value in milliseconds)
    if not values:
        raise ValueError("At least one timing is required.")
    median = statistics.median(values)
    if len(values) == 1:
        return median, median, median
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return median, quartiles[0], quartiles[2]


def benchmark_model(
    model_id: str,
    *,
    warmup_iterations: int = BENCHMARK_WARMUP_ITERATIONS,
    timed_iterations: dict[int, int] | None = None,
) -> dict[str, object]:
    """Benchmark one untrained architecture in an isolated CPU worker."""
    normalized = str(model_id).strip().lower()
    if normalized not in MODEL_ORDER:
        raise ValueError(f"Unknown model: {model_id}")
    if warmup_iterations < 0:
        raise ValueError("Warmup iterations cannot be negative.")
    repetitions = dict(BENCHMARK_TIMED_ITERATIONS if timed_iterations is None else timed_iterations)
    if set(repetitions) != set(BENCHMARK_BATCH_SIZES):
        raise ValueError("Timed iterations must cover batch sizes 1 and 16.")
    if any(int(value) <= 0 for value in repetitions.values()):
        raise ValueError("Every timed iteration count must be positive.")

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(BENCHMARK_SEED)
    baseline_rss = current_rss_bytes()
    model = build_model(normalized).cpu().eval()
    observed_parameters = count_trainable_parameters(model)
    expected_parameters = int(EXPECTED_TRAINABLE_PARAMETERS[normalized])
    if observed_parameters != expected_parameters:
        raise RuntimeError(
            f"{normalized} parameter mismatch: {observed_parameters} != {expected_parameters}."
        )

    parameter_bytes, buffer_bytes = parameter_and_buffer_bytes(model)
    state_dict_bytes = serialized_state_dict_bytes(model)
    timings: dict[int, dict[str, float | int]] = {}
    largest_output = 0
    output_lengths: dict[int, list[int]] = {}

    with torch.inference_mode():
        for batch_size in BENCHMARK_BATCH_SIZES:
            generator = torch.Generator(device="cpu").manual_seed(
                BENCHMARK_SEED + batch_size
            )
            inputs = torch.randn(
                batch_size,
                BENCHMARK_SEQUENCE_LENGTH,
                BENCHMARK_FEATURE_WIDTH,
                generator=generator,
                dtype=torch.float32,
            )
            lengths = torch.full(
                (batch_size,),
                BENCHMARK_SEQUENCE_LENGTH,
                dtype=torch.long,
            )
            for _ in range(warmup_iterations):
                output = model(inputs, lengths)
                largest_output = max(largest_output, byte_count(output))

            milliseconds: list[float] = []
            last_output = None
            for _ in range(int(repetitions[batch_size])):
                started = time.perf_counter_ns()
                last_output = model(inputs, lengths)
                elapsed = time.perf_counter_ns() - started
                milliseconds.append(elapsed / 1_000_000.0)
                largest_output = max(largest_output, byte_count(last_output))

            if last_output is None:
                raise RuntimeError("Benchmark produced no output.")
            median_ms, q1_ms, q3_ms = _median_and_iqr(milliseconds)
            output_lengths[batch_size] = [
                int(value)
                for value in last_output.output_lengths.detach().cpu().tolist()
            ]
            timings[batch_size] = {
                "batch_size": batch_size,
                "timed_iterations": int(repetitions[batch_size]),
                "latency_median_ms": median_ms,
                "latency_q1_ms": q1_ms,
                "latency_q3_ms": q3_ms,
                "throughput_samples_per_second": batch_size / (median_ms / 1000.0),
            }

    observed_peak_rss = peak_rss_bytes()
    rss_increment = None
    if baseline_rss is not None and observed_peak_rss is not None:
        rss_increment = max(0, int(observed_peak_rss) - int(baseline_rss))

    return {
        "model_id": normalized,
        "trainable_parameters": observed_parameters,
        "parameter_bytes": parameter_bytes,
        "buffer_bytes": buffer_bytes,
        "state_dict_serialized_bytes": state_dict_bytes,
        "largest_observed_output_bytes": largest_output,
        "baseline_rss_bytes": baseline_rss,
        "peak_process_rss_bytes": observed_peak_rss,
        "incremental_peak_rss_bytes": rss_increment,
        "batch1": timings[1],
        "batch16": timings[16],
        "output_lengths": output_lengths,
        "model_in_eval_mode": model.training is False,
        "gradients_created": any(parameter.grad is not None for parameter in model.parameters()),
        "training_started": False,
        "device": "cpu",
        "dtype": "float32",
    }


def run_isolated_worker(project_root: str | Path, model_id: str) -> dict[str, object]:
    root = Path(project_root).resolve()
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.build_stage34_efficiency",
            "--worker",
            str(model_id),
        ],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Stage 34D worker failed for {model_id}.\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    payload_lines = [
        line[len(WORKER_PREFIX) :]
        for line in completed.stdout.splitlines()
        if line.startswith(WORKER_PREFIX)
    ]
    if len(payload_lines) != 1:
        raise RuntimeError(f"Stage 34D worker returned invalid output for {model_id}.")
    return json.loads(payload_lines[0])


def runtime_identity() -> dict[str, object]:
    return {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "not_reported",
        "logical_cpu_count": os.cpu_count(),
        "torch_intraop_threads": 1,
        "torch_interop_threads": 1,
    }
