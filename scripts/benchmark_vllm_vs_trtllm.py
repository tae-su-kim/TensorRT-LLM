# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run an initial 1x H100 throughput comparison between TRT-LLM and vLLM.

The script implements the benchmark plan for:
  - Model: ``Qwen/Qwen3-30B-A3B-Thinking-2507-FP8``
  - Workload: ``1024`` input tokens, ``1024`` output tokens
  - Scope: single GPU, offline throughput, synthetic requests

It uses the repo's native ``python -m tensorrt_llm.commands.bench`` entrypoints
for TRT-LLM and the vLLM Python API for the vLLM branch. The vLLM run consumes
the same tokenized prompt set generated for TRT-LLM via ``prompt_token_ids`` so
the comparison stays on one shared synthetic workload artifact.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import yaml


DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"
DEFAULT_CONCURRENCY = [1, 2, 4, 8, 16, 32, 64, 128]
TRTLLM_BENCH_MODULE = "tensorrt_llm.commands.bench"
OOM_PATTERNS = (
    "out of memory",
    "cuda out of memory",
    "cublas_status_alloc_failed",
    "cuda_error_out_of_memory",
    "resourceexhaustederror",
)


@dataclass(frozen=True)
class RequestSample:
    """Tokenized benchmark request."""

    task_id: int
    input_ids: list[int]
    output_tokens: int


@dataclass(frozen=True)
class MetricSummary:
    """Normalized throughput metrics across benchmark stacks."""

    request_throughput_req_s: float
    output_throughput_tok_s: float
    total_throughput_tok_s: float
    total_latency_ms: float


@dataclass(frozen=True)
class SweepResult:
    """Single benchmark sweep result for one stack at one concurrency."""

    stack: str
    backend: str
    concurrency: int | None
    status: str
    reason: str | None = None
    request_throughput_req_s: float | None = None
    output_throughput_tok_s: float | None = None
    total_throughput_tok_s: float | None = None
    total_latency_ms: float | None = None
    peak_gpu_memory_gb: float | None = None
    report_json: str | None = None
    stdout_log: str | None = None
    stderr_log: str | None = None


def slugify_model_id(model: str) -> str:
    """Convert a model id into a filesystem-safe directory name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_")


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for a path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    """Write text to disk, creating parent directories when needed."""
    ensure_parent_dir(path)
    path.write_text(content)


def write_json(path: Path, payload: Any) -> None:
    """Write JSON to disk, creating parent directories when needed."""
    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_tokenized_dataset(path: Path) -> list[RequestSample]:
    """Load TRT-LLM synthetic requests from a JSONL dataset."""
    samples: list[RequestSample] = []
    with open(path) as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            input_ids = payload.get("input_ids")
            if not isinstance(input_ids, list) or not all(isinstance(x, int) for x in input_ids):
                raise ValueError(f"Invalid input_ids at line {index + 1} in {path}")
            output_tokens = payload.get("output_tokens")
            if not isinstance(output_tokens, int):
                raise ValueError(f"Invalid output_tokens at line {index + 1} in {path}")
            task_id = payload.get("task_id", index)
            if not isinstance(task_id, int):
                raise ValueError(f"Invalid task_id at line {index + 1} in {path}")
            samples.append(
                RequestSample(task_id=task_id, input_ids=input_ids, output_tokens=output_tokens)
            )
    if not samples:
        raise ValueError(f"Dataset {path} contains no requests")
    return samples


def validate_fixed_lengths(
    samples: list[RequestSample],
    expected_input_len: int,
    expected_output_len: int,
) -> None:
    """Validate that the shared synthetic workload matches the requested shape."""
    for sample in samples:
        if len(sample.input_ids) != expected_input_len:
            raise ValueError(
                f"Request {sample.task_id} has {len(sample.input_ids)} input tokens, "
                f"expected {expected_input_len}"
            )
        if sample.output_tokens != expected_output_len:
            raise ValueError(
                f"Request {sample.task_id} has {sample.output_tokens} output tokens, "
                f"expected {expected_output_len}"
            )


def build_vllm_prompts(samples: list[RequestSample]) -> list[dict[str, list[int]]]:
    """Convert TRT-LLM tokenized requests into vLLM token prompts."""
    return [{"prompt_token_ids": sample.input_ids} for sample in samples]


def build_pytorch_backend_config(concurrencies: list[int]) -> str:
    """Return a TRT-LLM PyTorch backend config tuned for the Qwen3 throughput sweep."""
    batch_sizes = sorted({value for value in concurrencies if value > 0})
    payload = {
        "enable_chunked_prefill": True,
        "cuda_graph_config": {
            "enable_padding": True,
            "batch_sizes": batch_sizes,
        },
        "torch_compile_config": {
            "enable_fullgraph": True,
            "enable_inductor": False,
            "enable_piecewise_cuda_graph": True,
            "enable_userbuffers": True,
            "max_num_streams": 3,
        },
        "kv_cache_config": {"free_gpu_memory_fraction": 0.8},
        "print_iter_log": True,
    }
    return yaml.safe_dump(
        payload,
        sort_keys=False,
        default_flow_style=False,
    )


def compute_metrics(
    num_requests: int,
    input_len: int,
    output_len: int,
    total_latency_s: float,
) -> MetricSummary:
    """Compute normalized throughput metrics from a wall-clock duration."""
    if total_latency_s <= 0:
        raise ValueError(f"total_latency_s must be > 0, got {total_latency_s}")
    total_latency_ms = total_latency_s * 1000.0
    return MetricSummary(
        request_throughput_req_s=num_requests / total_latency_s,
        output_throughput_tok_s=(num_requests * output_len) / total_latency_s,
        total_throughput_tok_s=(num_requests * (input_len + output_len)) / total_latency_s,
        total_latency_ms=total_latency_ms,
    )


def classify_failure_reason(stdout: str, stderr: str) -> str:
    """Classify benchmark failures into coarse, summary-friendly categories."""
    combined = f"{stdout}\n{stderr}".lower()
    if any(pattern in combined for pattern in OOM_PATTERNS):
        return "oom"
    if "unsupported" in combined or "not supported" in combined:
        return "unsupported"
    if "importerror" in combined or "modulenotfounderror" in combined:
        return "missing_dependency"
    return "runtime_failure"


def best_successful_results(results: list[SweepResult]) -> list[SweepResult]:
    """Return the best successful row per stack by output token throughput."""
    grouped: dict[str, list[SweepResult]] = {}
    for result in results:
        if result.status != "passed" or result.output_throughput_tok_s is None:
            continue
        grouped.setdefault(result.stack, []).append(result)
    best_results = []
    for entries in grouped.values():
        best_results.append(max(entries, key=lambda item: item.output_throughput_tok_s or 0.0))
    return sorted(best_results, key=lambda item: item.stack)


def relative_str(path: Path | None, root: Path) -> str | None:
    """Render a path relative to the output root when possible."""
    if path is None:
        return None
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def render_summary_markdown(
    *,
    model: str,
    num_requests: int,
    input_len: int,
    output_len: int,
    warmup_requests: int,
    concurrencies: list[int],
    dataset_path: Path,
    results: list[SweepResult],
) -> str:
    """Render a concise markdown summary for the benchmark sweep."""
    best_rows = best_successful_results(results)
    lines = [
        f"# Benchmark Summary: {model}",
        "",
        f"- Workload: {input_len} input / {output_len} output tokens",
        f"- Requests: {num_requests}",
        f"- Warmup requests excluded from measurement: {warmup_requests}",
        f"- Concurrency sweep: {', '.join(str(value) for value in concurrencies)}",
        f"- Shared dataset: `{dataset_path}`",
        f"- vLLM prompt mode: shared token ids via `prompt_token_ids`",
        "",
    ]

    if best_rows:
        lines.extend(
            [
                "## Best Successful Result",
                "",
                "| Stack | Backend | Concurrency | Output tok/s | Total tok/s | Req/s | Total latency (ms) |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in best_rows:
            lines.append(
                "| "
                f"{row.stack} | {row.backend} | {row.concurrency} | "
                f"{row.output_throughput_tok_s:.2f} | {row.total_throughput_tok_s:.2f} | "
                f"{row.request_throughput_req_s:.2f} | {row.total_latency_ms:.2f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Full Sweep",
            "",
            "| Stack | Backend | Concurrency | Status | Output tok/s | Total tok/s | Req/s | Reason |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in results:
        lines.append(
            "| "
            f"{row.stack} | {row.backend} | "
            f"{'-' if row.concurrency is None else row.concurrency} | "
            f"{row.status} | "
            f"{'-' if row.output_throughput_tok_s is None else f'{row.output_throughput_tok_s:.2f}'} | "
            f"{'-' if row.total_throughput_tok_s is None else f'{row.total_throughput_tok_s:.2f}'} | "
            f"{'-' if row.request_throughput_req_s is None else f'{row.request_throughput_req_s:.2f}'} | "
            f"{row.reason or '-'} |"
        )
    lines.append("")
    return "\n".join(lines)


def extract_trtllm_metrics(report_json: Path) -> MetricSummary:
    """Read TRT-LLM benchmark metrics from a report JSON file."""
    payload = json.loads(report_json.read_text())
    performance = payload["performance"]
    return MetricSummary(
        request_throughput_req_s=float(performance["request_throughput_req_s"]),
        output_throughput_tok_s=float(performance["system_output_throughput_tok_s"]),
        total_throughput_tok_s=float(performance["system_total_throughput_tok_s"]),
        total_latency_ms=float(performance["total_latency_ms"]),
    )


def query_gpu_info() -> dict[str, Any] | None:
    """Collect a minimal single-GPU descriptor via nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return None
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    first = lines[0].split(",")
    if len(first) != 2:
        return None
    name = first[0].strip()
    memory_mib = float(first[1].strip())
    return {"name": name, "memory.total_gb": memory_mib / 1024.0}


class BenchmarkRunner:
    """Coordinator for dataset preparation, backend runs, and summary emission."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        output_root = args.output_root
        if output_root is None:
            output_root = (
                Path("benchmark_artifacts")
                / f"{slugify_model_id(args.model)}_1x_h100_1024in_1024out"
            )
        workspace = args.workspace
        if workspace is None:
            workspace = Path("/tmp") / f"trtllm_benchmark_{slugify_model_id(args.model)}"

        self.output_root = output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.dataset_path = (
            args.dataset_path.resolve()
            if args.dataset_path is not None
            else self.output_root / "datasets" / "synthetic_1024_1024_token_ids.jsonl"
        )
        self.pytorch_config_path = self.output_root / "configs" / "trtllm_pytorch_qwen3.yaml"
        self.metadata_path = self.output_root / "run_metadata.json"
        self.summary_json_path = self.output_root / "summary.json"
        self.summary_md_path = self.output_root / "summary.md"
        self.samples: list[RequestSample] | None = None

    @property
    def model_source(self) -> str:
        """Return the local path when provided, otherwise the Hugging Face model id."""
        if self.args.model_path is not None:
            return str(self.args.model_path.resolve())
        return self.args.model

    @property
    def engine_dir(self) -> Path:
        """Expected engine directory produced by trtllm-bench build."""
        return self.workspace / self.args.model / "tp_1_pp_1"

    def run(self) -> int:
        """Execute the benchmark workflow."""
        self.write_metadata()
        self.prepare_dataset()
        self.write_pytorch_config()
        if self.args.prepare_only:
            return 0

        results: list[SweepResult] = []
        if not self.args.skip_pytorch_backend:
            results.extend(self.run_trtllm_backend(backend="pytorch"))
        if not self.args.skip_tensorrt_backend:
            results.extend(self.run_tensorrt_backend())
        if not self.args.skip_vllm:
            results.extend(self.run_vllm_backend())
        self.write_summary(results)
        return 0

    def write_metadata(self) -> None:
        """Persist benchmark plan metadata for reproducibility."""
        metadata = {
            "model": self.args.model,
            "model_path": None if self.args.model_path is None else str(self.args.model_path.resolve()),
            "revision": self.args.revision,
            "input_len": self.args.input_len,
            "output_len": self.args.output_len,
            "num_requests": self.args.num_requests,
            "warmup_requests": self.args.warmup_requests,
            "concurrency": self.args.concurrency,
            "seed": self.args.seed,
            "workspace": str(self.workspace),
            "dataset_path": str(self.dataset_path),
            "gpu": query_gpu_info(),
        }
        write_json(self.metadata_path, metadata)

    def base_trtllm_command(self) -> list[str]:
        """Build the common command prefix for the repo bench entrypoint."""
        command = [
            sys.executable,
            "-m",
            TRTLLM_BENCH_MODULE,
            "--model",
            self.args.model,
            "--workspace",
            str(self.workspace),
        ]
        if self.args.model_path is not None:
            command.extend(["--model_path", str(self.args.model_path.resolve())])
        if self.args.revision is not None:
            command.extend(["--revision", self.args.revision])
        return command

    def prepare_dataset(self) -> None:
        """Prepare or validate the shared synthetic token-id dataset."""
        if self.args.dataset_path is None:
            command = self.base_trtllm_command()
            command.extend(
                [
                    "prepare-dataset",
                    "--output",
                    str(self.dataset_path),
                    "--random-seed",
                    str(self.args.seed),
                ]
            )
            if self.args.trust_remote_code:
                command.append("--trust-remote-code")
            command.extend(
                [
                    "token-norm-dist",
                    "--num-requests",
                    str(self.args.num_requests),
                    "--input-mean",
                    str(self.args.input_len),
                    "--input-stdev",
                    "0",
                    "--output-mean",
                    str(self.args.output_len),
                    "--output-stdev",
                    "0",
                ]
            )
            dataset_logs = self.output_root / "logs" / "prepare_dataset"
            completed = self.run_subprocess(
                command,
                stdout_log=dataset_logs.with_suffix(".stdout.log"),
                stderr_log=dataset_logs.with_suffix(".stderr.log"),
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Failed to prepare the shared dataset. "
                    f"See {dataset_logs.with_suffix('.stderr.log')}."
                )

        self.samples = load_tokenized_dataset(self.dataset_path)
        validate_fixed_lengths(
            self.samples,
            expected_input_len=self.args.input_len,
            expected_output_len=self.args.output_len,
        )

    def write_pytorch_config(self) -> None:
        """Write the PyTorch backend config fragment used by TRT-LLM."""
        config = build_pytorch_backend_config(self.args.concurrency)
        write_text(self.pytorch_config_path, config)

    def run_trtllm_backend(self, *, backend: str) -> list[SweepResult]:
        """Run the TRT-LLM throughput sweep for a single backend."""
        stack = f"trtllm-{backend}"
        results: list[SweepResult] = []
        for concurrency in self.args.concurrency:
            result_dir = self.output_root / stack / f"concurrency_{concurrency}"
            report_json = result_dir / "report.json"
            stdout_log = result_dir / "stdout.log"
            stderr_log = result_dir / "stderr.log"

            command = self.base_trtllm_command()
            command.extend(
                [
                    "throughput",
                    "--dataset",
                    str(self.dataset_path),
                    "--backend",
                    backend,
                    "--num_requests",
                    str(self.args.num_requests),
                    "--warmup",
                    str(self.args.warmup_requests),
                    "--concurrency",
                    str(concurrency),
                    "--report_json",
                    str(report_json),
                    "--target_input_len",
                    str(self.args.input_len),
                    "--target_output_len",
                    str(self.args.output_len),
                ]
            )
            if backend == "pytorch":
                command.extend(["--config", str(self.pytorch_config_path)])
            else:
                command.extend(["--engine_dir", str(self.engine_dir)])

            completed = self.run_subprocess(
                command,
                stdout_log=stdout_log,
                stderr_log=stderr_log,
            )

            if completed.returncode != 0:
                results.append(
                    SweepResult(
                        stack=stack,
                        backend=backend,
                        concurrency=concurrency,
                        status="failed",
                        reason=classify_failure_reason(completed.stdout, completed.stderr),
                        report_json=relative_str(report_json, self.output_root),
                        stdout_log=relative_str(stdout_log, self.output_root),
                        stderr_log=relative_str(stderr_log, self.output_root),
                    )
                )
                break

            metrics = extract_trtllm_metrics(report_json)
            results.append(
                SweepResult(
                    stack=stack,
                    backend=backend,
                    concurrency=concurrency,
                    status="passed",
                    request_throughput_req_s=metrics.request_throughput_req_s,
                    output_throughput_tok_s=metrics.output_throughput_tok_s,
                    total_throughput_tok_s=metrics.total_throughput_tok_s,
                    total_latency_ms=metrics.total_latency_ms,
                    report_json=relative_str(report_json, self.output_root),
                    stdout_log=relative_str(stdout_log, self.output_root),
                    stderr_log=relative_str(stderr_log, self.output_root),
                )
            )
        return results

    def run_tensorrt_backend(self) -> list[SweepResult]:
        """Build the TensorRT engine and, if successful, run the sweep."""
        build_dir = self.output_root / "trtllm-tensorrt" / "build"
        build_stdout = build_dir / "stdout.log"
        build_stderr = build_dir / "stderr.log"
        build_command = self.base_trtllm_command()
        build_command.extend(
            [
                "build",
                "--dataset",
                str(self.dataset_path),
                "--quantization",
                "FP8",
            ]
        )
        if self.args.trust_remote_code:
            build_command.extend(["--trust_remote_code", "true"])
        completed = self.run_subprocess(
            build_command,
            stdout_log=build_stdout,
            stderr_log=build_stderr,
        )
        if completed.returncode != 0:
            return [
                SweepResult(
                    stack="trtllm-tensorrt",
                    backend="tensorrt",
                    concurrency=None,
                    status="blocked",
                    reason=classify_failure_reason(completed.stdout, completed.stderr),
                    stdout_log=relative_str(build_stdout, self.output_root),
                    stderr_log=relative_str(build_stderr, self.output_root),
                )
            ]
        return self.run_trtllm_backend(backend="tensorrt")

    def run_vllm_backend(self) -> list[SweepResult]:
        """Run the vLLM sweep against the shared tokenized dataset."""
        if self.args.vllm_python is not None:
            return self.run_vllm_backend_external(self.args.vllm_python.resolve())

        try:
            import torch
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            return [
                SweepResult(
                    stack="vllm",
                    backend="vllm",
                    concurrency=None,
                    status="blocked",
                    reason=f"missing_dependency: {exc}",
                )
            ]

        assert self.samples is not None
        prompts = build_vllm_prompts(self.samples[: self.args.num_requests])
        warmup_prompts = prompts[: min(self.args.warmup_requests, len(prompts))]
        results: list[SweepResult] = []

        for concurrency in self.args.concurrency:
            result_dir = self.output_root / "vllm" / f"concurrency_{concurrency}"
            report_json = result_dir / "report.json"
            stderr_log = result_dir / "stderr.log"
            result_dir.mkdir(parents=True, exist_ok=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            llm = None
            try:
                llm_kwargs = self.filter_supported_kwargs(
                    LLM.__init__,
                    {
                        "model": self.model_source,
                        "tokenizer": self.model_source,
                        "trust_remote_code": self.args.trust_remote_code,
                        "skip_tokenizer_init": True,
                        "tensor_parallel_size": 1,
                        "max_model_len": self.args.input_len + self.args.output_len,
                        "max_num_seqs": concurrency,
                        "gpu_memory_utilization": self.args.vllm_gpu_memory_utilization,
                    },
                )
                llm = LLM(**llm_kwargs)
                sampling_params = SamplingParams(
                    max_tokens=self.args.output_len,
                    ignore_eos=True,
                    temperature=0.0,
                )
                generate_kwargs = self.build_vllm_generate_kwargs(
                    llm.generate,
                    prompts=warmup_prompts,
                    sampling_params=sampling_params,
                )
                if warmup_prompts:
                    llm.generate(**generate_kwargs)

                timed_generate_kwargs = self.build_vllm_generate_kwargs(
                    llm.generate,
                    prompts=prompts,
                    sampling_params=sampling_params,
                )
                start = time.perf_counter()
                llm.generate(**timed_generate_kwargs)
                total_latency_s = time.perf_counter() - start
                metrics = compute_metrics(
                    num_requests=self.args.num_requests,
                    input_len=self.args.input_len,
                    output_len=self.args.output_len,
                    total_latency_s=total_latency_s,
                )
                peak_gpu_memory_gb = None
                if torch.cuda.is_available():
                    peak_gpu_memory_gb = torch.cuda.max_memory_allocated() / (1024.0**3)

                payload = {
                    "engine": {"model": self.model_source, "backend": "vllm"},
                    "machine": query_gpu_info(),
                    "request_info": {
                        "num_requests": self.args.num_requests,
                        "avg_input_length": self.args.input_len,
                        "avg_output_length": self.args.output_len,
                        "configured_max_num_seqs": concurrency,
                    },
                    "performance": asdict(metrics),
                    "peak_gpu_memory_gb": peak_gpu_memory_gb,
                }
                write_json(report_json, payload)
                results.append(
                    SweepResult(
                        stack="vllm",
                        backend="vllm",
                        concurrency=concurrency,
                        status="passed",
                        request_throughput_req_s=metrics.request_throughput_req_s,
                        output_throughput_tok_s=metrics.output_throughput_tok_s,
                        total_throughput_tok_s=metrics.total_throughput_tok_s,
                        total_latency_ms=metrics.total_latency_ms,
                        peak_gpu_memory_gb=peak_gpu_memory_gb,
                        report_json=relative_str(report_json, self.output_root),
                        stderr_log=relative_str(stderr_log, self.output_root),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                write_text(stderr_log, traceback.format_exc())
                results.append(
                    SweepResult(
                        stack="vllm",
                        backend="vllm",
                        concurrency=concurrency,
                        status="failed",
                        reason=f"{classify_failure_reason('', str(exc))}: {exc}",
                        report_json=relative_str(report_json, self.output_root),
                        stderr_log=relative_str(stderr_log, self.output_root),
                    )
                )
                break
            finally:
                if llm is not None:
                    del llm
        return results

    def run_vllm_backend_external(self, vllm_python: Path) -> list[SweepResult]:
        """Run the vLLM sweep through a separate Python interpreter."""
        results: list[SweepResult] = []
        for concurrency in self.args.concurrency:
            result_dir = self.output_root / "vllm" / f"concurrency_{concurrency}"
            report_json = result_dir / "report.json"
            stdout_log = result_dir / "stdout.log"
            stderr_log = result_dir / "stderr.log"
            command = [
                str(vllm_python),
                str(Path(__file__).resolve()),
                "--internal-vllm-worker",
                "--model-source",
                self.model_source,
                "--dataset-path",
                str(self.dataset_path),
                "--output-json",
                str(report_json),
                "--num-requests",
                str(self.args.num_requests),
                "--warmup-requests",
                str(self.args.warmup_requests),
                "--concurrency",
                str(concurrency),
                "--input-len",
                str(self.args.input_len),
                "--output-len",
                str(self.args.output_len),
                "--gpu-memory-utilization",
                str(self.args.vllm_gpu_memory_utilization),
            ]
            if self.args.trust_remote_code:
                command.append("--trust-remote-code")

            completed = self.run_subprocess(
                command,
                stdout_log=stdout_log,
                stderr_log=stderr_log,
            )
            if completed.returncode != 0:
                results.append(
                    SweepResult(
                        stack="vllm",
                        backend="vllm",
                        concurrency=concurrency,
                        status="failed",
                        reason=classify_failure_reason(completed.stdout, completed.stderr),
                        report_json=relative_str(report_json, self.output_root),
                        stdout_log=relative_str(stdout_log, self.output_root),
                        stderr_log=relative_str(stderr_log, self.output_root),
                    )
                )
                break

            payload = json.loads(report_json.read_text())
            performance = payload["performance"]
            results.append(
                SweepResult(
                    stack="vllm",
                    backend="vllm",
                    concurrency=concurrency,
                    status="passed",
                    request_throughput_req_s=float(performance["request_throughput_req_s"]),
                    output_throughput_tok_s=float(performance["output_throughput_tok_s"]),
                    total_throughput_tok_s=float(performance["total_throughput_tok_s"]),
                    total_latency_ms=float(performance["total_latency_ms"]),
                    peak_gpu_memory_gb=payload.get("peak_gpu_memory_gb"),
                    report_json=relative_str(report_json, self.output_root),
                    stdout_log=relative_str(stdout_log, self.output_root),
                    stderr_log=relative_str(stderr_log, self.output_root),
                )
            )
        return results

    @staticmethod
    def filter_supported_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        """Keep only keyword arguments accepted by a callable."""
        signature = inspect.signature(fn)
        if any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            return dict(kwargs)
        accepted = set(signature.parameters)
        return {key: value for key, value in kwargs.items() if key in accepted}

    @classmethod
    def build_vllm_generate_kwargs(
        cls,
        fn: Callable[..., Any],
        *,
        prompts: list[dict[str, list[int]]],
        sampling_params: Any,
    ) -> dict[str, Any]:
        """Build a version-tolerant kwarg set for ``LLM.generate``."""
        signature = inspect.signature(fn)
        accepted = set(signature.parameters)
        kwargs: dict[str, Any] = {}
        if "prompts" in accepted:
            kwargs["prompts"] = prompts
        elif "inputs" in accepted:
            kwargs["inputs"] = prompts
        else:
            raise TypeError("vLLM generate() does not expose a prompts/inputs argument")
        if "sampling_params" in accepted:
            kwargs["sampling_params"] = sampling_params
        if "use_tqdm" in accepted:
            kwargs["use_tqdm"] = False
        return cls.filter_supported_kwargs(fn, kwargs)

    @staticmethod
    def run_subprocess(
        command: list[str],
        *,
        stdout_log: Path,
        stderr_log: Path,
    ) -> subprocess.CompletedProcess[str]:
        """Run a subprocess, capturing stdout and stderr into log files."""
        ensure_parent_dir(stdout_log)
        ensure_parent_dir(stderr_log)
        env = BenchmarkRunner.build_subprocess_env(command)
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        stdout_log.write_text(completed.stdout)
        stderr_log.write_text(completed.stderr)
        return completed

    @staticmethod
    def build_subprocess_env(command: list[str]) -> dict[str, str]:
        """Return a subprocess environment aligned with the selected interpreter."""
        env = os.environ.copy()
        executable = Path(command[0])
        if not executable.is_absolute():
            return env

        bin_dir = str(executable.parent)
        path_entries = [entry for entry in env.get("PATH", "").split(os.pathsep) if entry]
        filtered_entries = [entry for entry in path_entries if entry != bin_dir]
        env["PATH"] = os.pathsep.join([bin_dir, *filtered_entries])
        return env

    def write_summary(self, results: list[SweepResult]) -> None:
        """Persist the summary JSON and markdown views."""
        payload = {
            "model": self.args.model,
            "model_path": None if self.args.model_path is None else str(self.args.model_path.resolve()),
            "num_requests": self.args.num_requests,
            "input_len": self.args.input_len,
            "output_len": self.args.output_len,
            "warmup_requests": self.args.warmup_requests,
            "concurrency": self.args.concurrency,
            "dataset_path": str(self.dataset_path),
            "results": [asdict(result) for result in results],
            "best_results": [asdict(result) for result in best_successful_results(results)],
        }
        write_json(self.summary_json_path, payload)
        write_text(
            self.summary_md_path,
            render_summary_markdown(
                model=self.args.model,
                num_requests=self.args.num_requests,
                input_len=self.args.input_len,
                output_len=self.args.output_len,
                warmup_requests=self.args.warmup_requests,
                concurrencies=self.args.concurrency,
                dataset_path=self.dataset_path,
                results=results,
            ),
        )


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the benchmark harness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id to benchmark.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional local model path. When set, it is used instead of downloading from HF.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional Hugging Face revision (branch, tag, or commit).",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=1024,
        help="Number of input tokens per request.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=1024,
        help="Number of generated tokens per request.",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=1000,
        help="Number of benchmark requests in the measurement set.",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=32,
        help="Number of warmup requests to exclude from measurement.",
    )
    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Concurrency sweep values.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=420,
        help="Deterministic seed for synthetic token generation.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Optional pre-generated TRT-LLM dataset JSONL to reuse.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory where logs, reports, and summaries are written.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Workspace directory for TRT-LLM intermediate files and engines.",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Value passed through to vLLM when that branch is enabled.",
    )
    parser.add_argument(
        "--vllm-python",
        type=Path,
        default=None,
        help="Optional external Python interpreter to use for the vLLM branch.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code for model loading paths that require it.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only generate the dataset and TRT-LLM config files.",
    )
    parser.add_argument(
        "--skip-pytorch-backend",
        action="store_true",
        help="Skip the TRT-LLM PyTorch backend sweep.",
    )
    parser.add_argument(
        "--skip-tensorrt-backend",
        action="store_true",
        help="Skip the TRT-LLM TensorRT backend build and sweep.",
    )
    parser.add_argument(
        "--skip-vllm",
        action="store_true",
        help="Skip the vLLM sweep.",
    )
    return parser


def build_vllm_worker_parser() -> argparse.ArgumentParser:
    """Create the hidden CLI parser for the external vLLM worker."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--internal-vllm-worker", action="store_true")
    parser.add_argument("--model-source", required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--num-requests", type=int, required=True)
    parser.add_argument("--warmup-requests", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def run_vllm_worker(args: argparse.Namespace) -> int:
    """Execute one vLLM benchmark point and write its JSON report."""
    import torch
    from vllm import LLM, SamplingParams

    samples = load_tokenized_dataset(args.dataset_path)
    validate_fixed_lengths(
        samples,
        expected_input_len=args.input_len,
        expected_output_len=args.output_len,
    )
    prompts = build_vllm_prompts(samples[: args.num_requests])
    warmup_prompts = prompts[: min(args.warmup_requests, len(prompts))]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    llm_kwargs = BenchmarkRunner.filter_supported_kwargs(
        LLM.__init__,
        {
            "model": args.model_source,
            "tokenizer": args.model_source,
            "trust_remote_code": args.trust_remote_code,
            "skip_tokenizer_init": True,
            "tensor_parallel_size": 1,
            "max_model_len": args.input_len + args.output_len,
            "max_num_seqs": args.concurrency,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
    )
    llm = LLM(**llm_kwargs)
    try:
        sampling_params = SamplingParams(
            max_tokens=args.output_len,
            ignore_eos=True,
            temperature=0.0,
        )
        warmup_kwargs = BenchmarkRunner.build_vllm_generate_kwargs(
            llm.generate,
            prompts=warmup_prompts,
            sampling_params=sampling_params,
        )
        if warmup_prompts:
            llm.generate(**warmup_kwargs)

        timed_kwargs = BenchmarkRunner.build_vllm_generate_kwargs(
            llm.generate,
            prompts=prompts,
            sampling_params=sampling_params,
        )
        start = time.perf_counter()
        llm.generate(**timed_kwargs)
        total_latency_s = time.perf_counter() - start
        metrics = compute_metrics(
            num_requests=args.num_requests,
            input_len=args.input_len,
            output_len=args.output_len,
            total_latency_s=total_latency_s,
        )
        peak_gpu_memory_gb = None
        if torch.cuda.is_available():
            peak_gpu_memory_gb = torch.cuda.max_memory_allocated() / (1024.0**3)

        payload = {
            "engine": {"model": args.model_source, "backend": "vllm"},
            "machine": query_gpu_info(),
            "request_info": {
                "num_requests": args.num_requests,
                "avg_input_length": args.input_len,
                "avg_output_length": args.output_len,
                "configured_max_num_seqs": args.concurrency,
            },
            "performance": asdict(metrics),
            "peak_gpu_memory_gb": peak_gpu_memory_gb,
        }
        write_json(args.output_json, payload)
        return 0
    finally:
        del llm


def main() -> int:
    """CLI entrypoint."""
    if "--internal-vllm-worker" in sys.argv:
        args = build_vllm_worker_parser().parse_args()
        return run_vllm_worker(args)
    args = build_arg_parser().parse_args()
    runner = BenchmarkRunner(args)
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
