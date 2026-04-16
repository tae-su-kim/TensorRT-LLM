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

"""Profile Qwen3.5 backend runs under Nsight Systems.

This script has two modes:

1. Orchestration mode (default): wraps a backend run with ``nsys profile``
   and exports summary reports via ``nsys stats``.
2. Inner-run mode (``--inner-run``): executes the backend benchmark body with
   aligned NVTX ranges so the exported summaries can be filtered to the timed
   section.

The default target is the current local Qwen3.5 decode-heavy comparison:
``128 req, bs=64, 1 input token, 1024 output tokens``.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import subprocess
import sys
import textwrap
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path(
    "/workspace/.hf_home/hub/models--Qwen--Qwen3.5-0.8B/snapshots/"
    "2fc06364715b967f1860aea9cf38778875588b17")
DEFAULT_ENGINE_DIR = Path("/tmp/qwen35_native_bench_engine_bs64_qwen_update_conv")
DEFAULT_DATASETS = {
    "decode_heavy": Path("/tmp/qwen35_bench_dataset_128x1_1024.jsonl"),
    "mixed": Path("/tmp/qwen35_bench_dataset_128x1024_1024.jsonl"),
    "prefill_heavy": Path("/tmp/qwen35_bench_dataset_128x1024_1.jsonl"),
}
DEFAULT_OUTPUT_DIR = Path("/tmp/qwen35_nsys_profiles")
DEFAULT_BATCH_SIZE = 64
DEFAULT_TRACE = "cuda,nvtx,osrt"
DEFAULT_REPORTS = "cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,nvtx_sum,osrt_sum"
DEFAULT_PYTORCH_PYTHON = "/venv/main/bin/python"
DEFAULT_TENSORRT_PYTHON = "/venv/main/bin/python"
DEFAULT_VLLM_PYTHON = "/venv/vllm/bin/python"
DEFAULT_TENSORRT_PACKAGE_ROOT = Path("/tmp/qwen35_native_pkg")
TIMED_RANGE_NAME = "timed_run"
WARMUP_RANGE_NAME = "warmup"


@dataclass
class RunArtifact:
    backend: str
    workload: str
    profile_base: str
    rep_path: str
    throughput_report: str
    stats_base: str
    stats_reports: list[str]


def load_dataset(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


@contextlib.contextmanager
def nvtx_range(name: str):
    import torch

    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def get_output_len(result) -> int | None:
    try:
        if hasattr(result, "outputs") and result.outputs:
            output = result.outputs[0]
            if hasattr(output, "token_ids"):
                return len(output.token_ids)
        if hasattr(result, "token_ids"):
            return len(result.token_ids)
    except Exception:
        return None
    return None


def run_pytorch_case(model_path: Path, dataset_path: Path, report_path: Path,
                     batch_size: int, label: str) -> None:
    import torch
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi.llm_args import CudaGraphConfig, KvCacheConfig

    records = load_dataset(dataset_path)
    assert len(records) == 128, len(records)
    output_tokens = records[0]["output_tokens"]
    prompts = [record["input_ids"] for record in records]
    sampling_params = SamplingParams(max_tokens=output_tokens,
                                     temperature=0.0,
                                     end_id=0,
                                     pad_id=0,
                                     ignore_eos=True,
                                     detokenize=False)

    warmup_batch = prompts[:batch_size]
    timed_batches = [
        prompts[batch_size * i:batch_size * (i + 1)] for i in range(2)
    ]

    llm = None
    try:
        with nvtx_range("setup"):
            llm = LLM(model=model_path,
                      backend="pytorch",
                      skip_tokenizer_init=True,
                      tensor_parallel_size=1,
                      enable_chunked_prefill=True,
                      max_batch_size=batch_size,
                      max_num_tokens=2048,
                      cuda_graph_config=CudaGraphConfig(
                          enable_padding=True, batch_sizes=[1, batch_size]),
                      torch_compile_config=None,
                      kv_cache_config=KvCacheConfig(
                          free_gpu_memory_fraction=0.8),
                      print_iter_log=True)

        with nvtx_range(WARMUP_RANGE_NAME):
            outputs = llm.generate(warmup_batch,
                                   sampling_params=sampling_params,
                                   use_tqdm=False)
            torch.cuda.synchronize()
            warmup_actual = sum(get_output_len(result) or 0
                                for result in outputs)

        batch_times = []
        actual_tokens = 0
        requested_tokens = 0
        with nvtx_range(TIMED_RANGE_NAME):
            for batch_idx, batch in enumerate(timed_batches, start=1):
                with nvtx_range(f"batch_{batch_idx}"):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    outputs = llm.generate(batch,
                                           sampling_params=sampling_params,
                                           use_tqdm=False)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    batch_times.append(elapsed)
                    batch_actual = sum(get_output_len(result) or 0
                                       for result in outputs)
                    actual_tokens += batch_actual
                    requested_tokens += len(batch) * output_tokens

        elapsed = sum(batch_times)
        report = {
            "label": label,
            "model": str(model_path),
            "backend": "pytorch",
            "dataset": str(dataset_path),
            "batch_size": batch_size,
            "num_requests": len(records),
            "warmup_batches": 1,
            "timed_batches": 2,
            "input_tokens": len(prompts[0]),
            "output_tokens": output_tokens,
            "warmup_actual_output_tokens": warmup_actual,
            "batch_times_s": batch_times,
            "elapsed_s": elapsed,
            "requested_output_tokens": requested_tokens,
            "actual_output_tokens": actual_tokens,
            "requested_output_toks_per_s": requested_tokens / elapsed,
            "actual_output_toks_per_s": actual_tokens / elapsed
            if actual_tokens else None,
        }
        report_path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    finally:
        if llm is not None:
            with contextlib.suppress(Exception):
                llm.shutdown()


def run_tensorrt_case(engine_dir: Path, dataset_path: Path, report_path: Path,
                      batch_size: int, label: str) -> None:
    import torch
    from tensorrt_llm.plugin.plugin import _load_plugin_lib
    from tensorrt_llm.runtime.model_runner_cpp import ModelRunnerCpp

    _load_plugin_lib()
    records = load_dataset(dataset_path)
    assert len(records) == 128, len(records)
    output_tokens = records[0]["output_tokens"]
    input_len = len(records[0]["input_ids"])
    batches = [
        records[:batch_size],
        records[batch_size:batch_size * 2],
    ]

    with nvtx_range("setup"):
        runner = ModelRunnerCpp.from_dir(str(engine_dir),
                                         rank=0,
                                         max_batch_size=batch_size,
                                         max_input_len=max(input_len, 1),
                                         max_output_len=output_tokens,
                                         enable_chunked_context=False)

    warmup_inputs = [
        torch.tensor(row["input_ids"], dtype=torch.int32, device="cuda")
        for row in batches[0]
    ]
    with nvtx_range(WARMUP_RANGE_NAME):
        runner.generate(warmup_inputs,
                        max_new_tokens=output_tokens,
                        end_id=0,
                        pad_id=0)
        torch.cuda.synchronize()

    elapsed = 0.0
    batch_times = []
    with nvtx_range(TIMED_RANGE_NAME):
        for batch_idx, batch in enumerate(batches, start=1):
            batch_inputs = [
                torch.tensor(row["input_ids"], dtype=torch.int32, device="cuda")
                for row in batch
            ]
            with nvtx_range(f"batch_{batch_idx}"):
                torch.cuda.synchronize()
                start = time.perf_counter()
                runner.generate(batch_inputs,
                                max_new_tokens=output_tokens,
                                end_id=0,
                                pad_id=0)
                torch.cuda.synchronize()
                batch_elapsed = time.perf_counter() - start
                batch_times.append(batch_elapsed)
                elapsed += batch_elapsed

    requested_generated = len(records) * output_tokens
    report = {
        "backend": "tensorrt_cpp_executor_native",
        "case": label,
        "model": "Qwen/Qwen3.5-0.8B",
        "engine_dir": str(engine_dir),
        "dataset": str(dataset_path),
        "num_requests": len(records),
        "batch_size": batch_size,
        "input_tokens_per_request": input_len,
        "output_tokens_per_request": output_tokens,
        "requested_generated_tokens_total": requested_generated,
        "elapsed_s": elapsed,
        "output_tokens_per_s_requested": requested_generated / elapsed,
        "batch_times_s": batch_times,
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def run_vllm_case(model_path: Path, dataset_path: Path, report_path: Path,
                  batch_size: int, label: str) -> None:
    import inspect

    import torch
    from vllm import LLM, SamplingParams

    records = load_dataset(dataset_path)
    assert len(records) == 128, len(records)
    output_tokens = int(records[0]["output_tokens"])
    prompts = [{"prompt_token_ids": record["input_ids"]} for record in records]
    sampling_params = SamplingParams(max_tokens=output_tokens,
                                     ignore_eos=True,
                                     temperature=0.0,
                                     top_k=1,
                                     detokenize=False)

    with nvtx_range("setup"):
        llm = LLM(model=str(model_path),
                  tokenizer=str(model_path),
                  trust_remote_code=True,
                  tensor_parallel_size=1,
                  max_model_len=2048,
                  max_num_seqs=batch_size,
                  gpu_memory_utilization=0.9)

    generate_sig = inspect.signature(llm.generate)

    def run_batch(batch_prompts: list[dict]) -> list:
        kwargs = {"sampling_params": sampling_params}
        if "prompts" in generate_sig.parameters:
            kwargs["prompts"] = batch_prompts
            if "use_tqdm" in generate_sig.parameters:
                kwargs["use_tqdm"] = False
        elif "inputs" in generate_sig.parameters:
            kwargs["inputs"] = batch_prompts
        else:
            raise RuntimeError(
                f"Unsupported vLLM generate signature: {generate_sig}")
        return llm.generate(**kwargs)

    try:
        with nvtx_range(WARMUP_RANGE_NAME):
            run_batch(prompts[:batch_size])
            torch.cuda.synchronize()

        batch_times = []
        actual_generated = 0
        with nvtx_range(TIMED_RANGE_NAME):
            for batch_idx in range(2):
                batch_prompts = prompts[batch_idx * batch_size:(batch_idx + 1)
                                        * batch_size]
                with nvtx_range(f"batch_{batch_idx + 1}"):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    outputs = run_batch(batch_prompts)
                    torch.cuda.synchronize()
                    batch_elapsed = time.perf_counter() - start
                    batch_times.append(batch_elapsed)
                    for output in outputs:
                        actual_generated += len(output.outputs[0].token_ids)

        elapsed = sum(batch_times)
        requested_generated = len(records) * output_tokens
        report = {
            "backend": "vllm",
            "case": label,
            "model": "Qwen/Qwen3.5-0.8B",
            "model_path": str(model_path),
            "dataset": str(dataset_path),
            "num_requests": len(records),
            "batch_size": batch_size,
            "input_tokens_per_request": len(records[0]["input_ids"]),
            "output_tokens_per_request": output_tokens,
            "requested_generated_tokens_total": requested_generated,
            "actual_generated_tokens_total": actual_generated,
            "elapsed_s": elapsed,
            "output_tokens_per_s_requested": requested_generated / elapsed,
            "output_tokens_per_s_actual": actual_generated / elapsed,
            "batch_times_s": batch_times,
        }
        report_path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    finally:
        del llm


def configure_import_path(args: argparse.Namespace) -> None:
    repo_root_str = str(REPO_ROOT.resolve())
    staged_root_str = str(args.tensorrt_package_root.resolve())

    rewritten_path = []
    for entry in sys.path:
        resolved = str(Path(entry or os.getcwd()).resolve())
        if args.backend == "tensorrt" and resolved.startswith(repo_root_str):
            continue
        if args.backend == "pytorch" and resolved.startswith(staged_root_str):
            continue
        rewritten_path.append(entry)

    if args.backend == "tensorrt":
        sys.path = [staged_root_str] + rewritten_path
    elif args.backend == "pytorch":
        sys.path = [repo_root_str] + rewritten_path
    else:
        sys.path = rewritten_path

    for module_name in list(sys.modules):
        if module_name == "tensorrt_llm" or module_name.startswith(
                "tensorrt_llm."):
            del sys.modules[module_name]
    importlib.invalidate_caches()


def build_inner_env(args: argparse.Namespace, backend: str) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("HWLOC_COMPONENTS", "-gl")
    env.setdefault("MPI4PY_RC_INITIALIZE", "0")
    torch_lib_dir = "/venv/main/lib/python3.12/site-packages/torch/lib"
    if backend != "vllm":
        package_root = str(args.tensorrt_package_root)
        lib_dir = Path(package_root) / "tensorrt_llm" / "libs"
        plugin_dir = Path(package_root) / "tensorrt_llm" / "plugins"
        ld_library_path = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(
            [str(lib_dir),
             str(plugin_dir), torch_lib_dir, ld_library_path]).rstrip(":")
    if backend == "pytorch":
        env.setdefault("TLLM_DISABLE_MPI", "1")
        env.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
        env.setdefault("RAY_DISABLE_DOCKER_CPU_WARNING", "1")
        env["PYTHONPATH"] = package_root
        env["PATH"] = f"/venv/trtllm/bin:{env.get('PATH', '')}"
    elif backend == "tensorrt":
        env.setdefault("TLLM_DISABLE_MPI", "1")
        env["TRT_LLM_MINIMAL_IMPORT"] = "1"
        env["PYTHONPATH"] = package_root
    else:
        env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        env["PATH"] = f"/venv/vllm/bin:{env.get('PATH', '')}"
        env["PYTHONPATH"] = env.get("PYTHONPATH", "")
    return env


def get_python_for_backend(args: argparse.Namespace, backend: str) -> str:
    if backend == "pytorch":
        return args.pytorch_python
    if backend == "tensorrt":
        return args.tensorrt_python
    return args.vllm_python


def write_sitecustomize(sitecustomize_path: Path) -> None:
    sitecustomize_path.write_text(
        textwrap.dedent("""\
import contextlib

with contextlib.suppress(ImportError):
    import ray.util.placement_group as _ray_pg
    if not hasattr(_ray_pg, "PlacementGroupSchedulingStrategy"):
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy as _PGSS
        _ray_pg.PlacementGroupSchedulingStrategy = _PGSS
"""))


def write_tensorrt_inner_helper(helper_path: Path, engine_dir: Path,
                                dataset_path: Path, report_path: Path,
                                batch_size: int, label: str,
                                package_root: Path) -> None:
    helper_code = textwrap.dedent(f"""\
import contextlib
import importlib.machinery
import json
import types
import sys
import time
from pathlib import Path

sys.path[:] = [{json.dumps(str(package_root.resolve()))}] + [
    path for path in sys.path
    if not str(Path(path or '.').resolve()).startswith({json.dumps(str(REPO_ROOT.resolve()))})
]


try:
    import soundfile  # noqa: F401
except ModuleNotFoundError:
    soundfile_stub = types.ModuleType("soundfile")
    soundfile_stub.__spec__ = importlib.machinery.ModuleSpec(
        "soundfile", loader=None)

    def _unsupported_soundfile(*args, **kwargs):
        raise RuntimeError("soundfile is not available in this profiling environment")

    soundfile_stub.read = _unsupported_soundfile
    soundfile_stub.write = _unsupported_soundfile
    soundfile_stub.SoundFile = object
    sys.modules["soundfile"] = soundfile_stub

import torch
import nvtx.colors as _nvtx_colors

_orig_color_to_hex = _nvtx_colors.color_to_hex


def _patched_color_to_hex(color):
    if color in {{"grey", "gray"}}:
        return 0x808080
    return _orig_color_to_hex(color)


_nvtx_colors.color_to_hex = _patched_color_to_hex

from tensorrt_llm.plugin.plugin import _load_plugin_lib
from tensorrt_llm.runtime.model_runner_cpp import ModelRunnerCpp


ENGINE_DIR = Path({json.dumps(str(engine_dir))})
DATASET_PATH = Path({json.dumps(str(dataset_path))})
REPORT_PATH = Path({json.dumps(str(report_path))})
BATCH_SIZE = {batch_size}
LABEL = {json.dumps(label)}
TIMED_RANGE_NAME = {json.dumps(TIMED_RANGE_NAME)}
WARMUP_RANGE_NAME = {json.dumps(WARMUP_RANGE_NAME)}


def load_dataset(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f]


@contextlib.contextmanager
def nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def main():
    _load_plugin_lib()
    records = load_dataset(DATASET_PATH)
    output_tokens = records[0]["output_tokens"]
    input_len = len(records[0]["input_ids"])
    batches = [records[:BATCH_SIZE], records[BATCH_SIZE:BATCH_SIZE * 2]]

    with nvtx_range("setup"):
        runner = ModelRunnerCpp.from_dir(
            str(ENGINE_DIR),
            rank=0,
            max_batch_size=BATCH_SIZE,
            max_input_len=max(input_len, 1),
            max_output_len=output_tokens,
            enable_chunked_context=False,
        )

    warmup_inputs = [
        torch.tensor(row["input_ids"], dtype=torch.int32, device="cuda")
        for row in batches[0]
    ]
    with nvtx_range(WARMUP_RANGE_NAME):
        runner.generate(warmup_inputs, max_new_tokens=output_tokens, end_id=0, pad_id=0)
        torch.cuda.synchronize()

    elapsed = 0.0
    batch_times = []
    with nvtx_range(TIMED_RANGE_NAME):
        for batch_idx, batch in enumerate(batches, start=1):
            batch_inputs = [
                torch.tensor(row["input_ids"], dtype=torch.int32, device="cuda")
                for row in batch
            ]
            with nvtx_range(f"batch_{{batch_idx}}"):
                torch.cuda.synchronize()
                start = time.perf_counter()
                runner.generate(batch_inputs, max_new_tokens=output_tokens, end_id=0, pad_id=0)
                torch.cuda.synchronize()
                batch_elapsed = time.perf_counter() - start
                batch_times.append(batch_elapsed)
                elapsed += batch_elapsed

    requested_generated = len(records) * output_tokens
    report = {{
        "backend": "tensorrt_cpp_executor_native",
        "case": LABEL,
        "model": "Qwen/Qwen3.5-0.8B",
        "engine_dir": str(ENGINE_DIR),
        "dataset": str(DATASET_PATH),
        "num_requests": len(records),
        "batch_size": BATCH_SIZE,
        "input_tokens_per_request": input_len,
        "output_tokens_per_request": output_tokens,
        "requested_generated_tokens_total": requested_generated,
        "elapsed_s": elapsed,
        "output_tokens_per_s_requested": requested_generated / elapsed,
        "batch_times_s": batch_times,
    }}
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
""")
    helper_path.write_text(helper_code)


def write_pytorch_inner_helper(helper_path: Path, model_path: Path,
                               dataset_path: Path, report_path: Path,
                               batch_size: int, label: str,
                               package_root: Path) -> None:
    helper_code = textwrap.dedent(f"""\
import contextlib
import importlib.machinery
import json
import types
import sys
import time
from pathlib import Path

sys.path[:] = [{json.dumps(str(package_root.resolve()))}] + [
    path for path in sys.path
    if not str(Path(path or '.').resolve()).startswith({json.dumps(str(REPO_ROOT.resolve()))})
]


try:
    import soundfile  # noqa: F401
except ModuleNotFoundError:
    soundfile_stub = types.ModuleType("soundfile")
    soundfile_stub.__spec__ = importlib.machinery.ModuleSpec(
        "soundfile", loader=None)

    def _unsupported_soundfile(*args, **kwargs):
        raise RuntimeError("soundfile is not available in this profiling environment")

    soundfile_stub.read = _unsupported_soundfile
    soundfile_stub.write = _unsupported_soundfile
    soundfile_stub.SoundFile = object
    sys.modules["soundfile"] = soundfile_stub

import torch
import nvtx.colors as _nvtx_colors

_orig_color_to_hex = _nvtx_colors.color_to_hex


def _patched_color_to_hex(color):
    if color in {{"grey", "gray"}}:
        return 0x808080
    return _orig_color_to_hex(color)


_nvtx_colors.color_to_hex = _patched_color_to_hex

try:
    import ray.util.placement_group as _ray_pg
    if not hasattr(_ray_pg, "PlacementGroupSchedulingStrategy"):
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy as _PGSS
        _ray_pg.PlacementGroupSchedulingStrategy = _PGSS
except ImportError:
    pass

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi.llm_args import CudaGraphConfig, KvCacheConfig


MODEL = Path({json.dumps(str(model_path))})
DATASET_PATH = Path({json.dumps(str(dataset_path))})
REPORT_PATH = Path({json.dumps(str(report_path))})
BATCH_SIZE = {batch_size}
LABEL = {json.dumps(label)}
TIMED_RANGE_NAME = {json.dumps(TIMED_RANGE_NAME)}
WARMUP_RANGE_NAME = {json.dumps(WARMUP_RANGE_NAME)}


def load_dataset(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f]


@contextlib.contextmanager
def nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def get_output_len(result):
    try:
        if hasattr(result, "outputs") and result.outputs:
            output = result.outputs[0]
            if hasattr(output, "token_ids"):
                return len(output.token_ids)
        if hasattr(result, "token_ids"):
            return len(result.token_ids)
    except Exception:
        return None
    return None


def main():
    llm = None
    try:
        records = load_dataset(DATASET_PATH)
        output_tokens = records[0]["output_tokens"]
        prompts = [record["input_ids"] for record in records]
        sampling_params = SamplingParams(
            max_tokens=output_tokens,
            temperature=0.0,
            end_id=0,
            pad_id=0,
            ignore_eos=True,
            detokenize=False,
        )
        warmup_batch = prompts[:BATCH_SIZE]
        timed_batches = [prompts[BATCH_SIZE * i:BATCH_SIZE * (i + 1)] for i in range(2)]

        with nvtx_range("setup"):
            llm = LLM(
                model=MODEL,
                backend="pytorch",
                skip_tokenizer_init=True,
                tensor_parallel_size=1,
                enable_chunked_prefill=True,
                max_batch_size=BATCH_SIZE,
                max_num_tokens=2048,
                cuda_graph_config=CudaGraphConfig(enable_padding=True, batch_sizes=[1, BATCH_SIZE]),
                torch_compile_config=None,
                kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.8),
                print_iter_log=True,
            )

        with nvtx_range(WARMUP_RANGE_NAME):
            outputs = llm.generate(warmup_batch, sampling_params=sampling_params, use_tqdm=False)
            torch.cuda.synchronize()
            warmup_actual = sum(get_output_len(result) or 0 for result in outputs)

        batch_times = []
        actual_tokens = 0
        requested_tokens = 0
        with nvtx_range(TIMED_RANGE_NAME):
            for batch_idx, batch in enumerate(timed_batches, start=1):
                with nvtx_range(f"batch_{{batch_idx}}"):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    outputs = llm.generate(batch, sampling_params=sampling_params, use_tqdm=False)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    batch_times.append(elapsed)
                    batch_actual = sum(get_output_len(result) or 0 for result in outputs)
                    actual_tokens += batch_actual
                    requested_tokens += len(batch) * output_tokens

        elapsed = sum(batch_times)
        report = {{
            "label": LABEL,
            "model": str(MODEL),
            "backend": "pytorch",
            "dataset": str(DATASET_PATH),
            "batch_size": BATCH_SIZE,
            "num_requests": len(records),
            "warmup_batches": 1,
            "timed_batches": 2,
            "input_tokens": len(prompts[0]),
            "output_tokens": output_tokens,
            "warmup_actual_output_tokens": warmup_actual,
            "batch_times_s": batch_times,
            "elapsed_s": elapsed,
            "requested_output_tokens": requested_tokens,
            "actual_output_tokens": actual_tokens,
            "requested_output_toks_per_s": requested_tokens / elapsed,
            "actual_output_toks_per_s": actual_tokens / elapsed if actual_tokens else None,
        }}
        REPORT_PATH.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    finally:
        if llm is not None:
            with contextlib.suppress(Exception):
                llm.shutdown()


if __name__ == "__main__":
    main()
""")
    helper_path.write_text(helper_code)


def write_vllm_inner_helper(helper_path: Path, model_path: Path,
                            dataset_path: Path, report_path: Path,
                            batch_size: int, label: str) -> None:
    helper_code = textwrap.dedent(f"""\
import contextlib
import inspect
import json
import time
from pathlib import Path

import torch
from vllm import LLM, SamplingParams


MODEL = Path({json.dumps(str(model_path))})
DATASET_PATH = Path({json.dumps(str(dataset_path))})
REPORT_PATH = Path({json.dumps(str(report_path))})
BATCH_SIZE = {batch_size}
LABEL = {json.dumps(label)}
TIMED_RANGE_NAME = {json.dumps(TIMED_RANGE_NAME)}
WARMUP_RANGE_NAME = {json.dumps(WARMUP_RANGE_NAME)}


def load_dataset(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f]


@contextlib.contextmanager
def nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def main():
    records = load_dataset(DATASET_PATH)
    output_tokens = int(records[0]["output_tokens"])
    prompts = [{{"prompt_token_ids": record["input_ids"]}} for record in records]
    sampling_params = SamplingParams(
        max_tokens=output_tokens,
        ignore_eos=True,
        temperature=0.0,
        top_k=1,
        detokenize=False,
    )

    with nvtx_range("setup"):
        llm = LLM(
            model=str(MODEL),
            tokenizer=str(MODEL),
            trust_remote_code=True,
            tensor_parallel_size=1,
            max_model_len=2048,
            max_num_seqs=BATCH_SIZE,
            gpu_memory_utilization=0.9,
        )

    try:
        generate_sig = inspect.signature(llm.generate)

        def run_batch(batch_prompts):
            kwargs = {{"sampling_params": sampling_params}}
            if "prompts" in generate_sig.parameters:
                kwargs["prompts"] = batch_prompts
                if "use_tqdm" in generate_sig.parameters:
                    kwargs["use_tqdm"] = False
            elif "inputs" in generate_sig.parameters:
                kwargs["inputs"] = batch_prompts
            else:
                raise RuntimeError(f"Unsupported vLLM generate signature: {{generate_sig}}")
            return llm.generate(**kwargs)

        with nvtx_range(WARMUP_RANGE_NAME):
            run_batch(prompts[:BATCH_SIZE])
            torch.cuda.synchronize()

        batch_times = []
        actual_generated = 0
        with nvtx_range(TIMED_RANGE_NAME):
            for batch_idx in range(2):
                batch_prompts = prompts[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE]
                with nvtx_range(f"batch_{{batch_idx + 1}}"):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    outputs = run_batch(batch_prompts)
                    torch.cuda.synchronize()
                    batch_elapsed = time.perf_counter() - start
                    batch_times.append(batch_elapsed)
                    for output in outputs:
                        actual_generated += len(output.outputs[0].token_ids)

        elapsed = sum(batch_times)
        requested_generated = len(records) * output_tokens
        report = {{
            "backend": "vllm",
            "case": LABEL,
            "model": "Qwen/Qwen3.5-0.8B",
            "model_path": str(MODEL),
            "dataset": str(DATASET_PATH),
            "num_requests": len(records),
            "batch_size": BATCH_SIZE,
            "input_tokens_per_request": len(records[0]["input_ids"]),
            "output_tokens_per_request": output_tokens,
            "requested_generated_tokens_total": requested_generated,
            "actual_generated_tokens_total": actual_generated,
            "elapsed_s": elapsed,
            "output_tokens_per_s_requested": requested_generated / elapsed,
            "output_tokens_per_s_actual": actual_generated / elapsed,
            "batch_times_s": batch_times,
        }}
        REPORT_PATH.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    finally:
        del llm


if __name__ == "__main__":
    main()
""")
    helper_path.write_text(helper_code)


def profile_one(args: argparse.Namespace, backend: str,
                workload: str) -> RunArtifact:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_sitecustomize(output_dir / "sitecustomize.py")

    base_name = f"{backend}_{workload}"
    profile_base = output_dir / base_name
    throughput_report = output_dir / f"{base_name}_throughput.json"
    stats_base = output_dir / f"{base_name}_stats"
    rep_path = Path(f"{profile_base}.nsys-rep")

    cmd = [
        "nsys",
        "profile",
        "--force-overwrite=true",
        f"--trace={args.trace}",
        "--sample=none",
        "--cpuctxsw=none",
        "-o",
        str(profile_base),
        get_python_for_backend(args, backend),
    ]
    helper_path = output_dir / f"_{base_name}_inner.py"
    if backend == "tensorrt":
        write_tensorrt_inner_helper(helper_path, args.engine_dir,
                                    args.dataset_map[workload],
                                    throughput_report, args.batch_size,
                                    workload, args.tensorrt_package_root)
    elif backend == "pytorch":
        write_pytorch_inner_helper(helper_path, args.model_path,
                                   args.dataset_map[workload],
                                   throughput_report, args.batch_size,
                                   workload, args.tensorrt_package_root)
    else:
        write_vllm_inner_helper(helper_path, args.model_path,
                                args.dataset_map[workload], throughput_report,
                                args.batch_size, workload)
    cmd.append(str(helper_path))

    env = build_inner_env(args, backend)
    env["PYTHONPATH"] = f"{output_dir}:{env['PYTHONPATH']}"
    subprocess.run(cmd, check=True, env=env, cwd=str(output_dir))

    stats_cmd = [
        "nsys",
        "stats",
        "--force-overwrite=true",
        "--report",
        args.reports,
        "--format",
        "csv",
        "--output",
        str(stats_base),
        "--filter-nvtx",
        f"{TIMED_RANGE_NAME}/0",
        str(rep_path),
    ]
    subprocess.run(stats_cmd, check=True, cwd=str(output_dir))

    return RunArtifact(backend=backend,
                       workload=workload,
                       profile_base=str(profile_base),
                       rep_path=str(rep_path),
                       throughput_report=str(throughput_report),
                       stats_base=str(stats_base),
                       stats_reports=args.reports.split(","))


def run_inner(args: argparse.Namespace) -> int:
    configure_import_path(args)
    dataset_path = args.dataset_path
    report_path = args.throughput_report
    label = args.workload
    if args.backend == "pytorch":
        run_pytorch_case(args.model_path, dataset_path, report_path,
                         args.batch_size, label)
    elif args.backend == "tensorrt":
        run_tensorrt_case(args.engine_dir, dataset_path, report_path,
                          args.batch_size, label)
    else:
        run_vllm_case(args.model_path, dataset_path, report_path,
                      args.batch_size, label)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=
        "Profile Qwen3.5 PyTorch, TensorRT, and vLLM backends with nsys.")
    parser.add_argument("--inner-run",
                        action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--backend",
                        choices=["pytorch", "tensorrt", "vllm"],
                        action="append",
                        help="Backend(s) to profile. Default profiles both.")
    parser.add_argument("--workload",
                        choices=list(DEFAULT_DATASETS.keys()),
                        action="append",
                        help="Workload(s) to profile. Default is decode-heavy.")
    parser.add_argument("--output-dir",
                        type=Path,
                        default=DEFAULT_OUTPUT_DIR,
                        help="Directory for .nsys-rep, stats, and throughput reports.")
    parser.add_argument("--model-path",
                        type=Path,
                        default=DEFAULT_MODEL,
                        help="Local HF model path for the PyTorch backend.")
    parser.add_argument("--engine-dir",
                        type=Path,
                        default=DEFAULT_ENGINE_DIR,
                        help="Engine directory for the TensorRT backend.")
    parser.add_argument("--dataset-path",
                        type=Path,
                        help=argparse.SUPPRESS)
    parser.add_argument("--throughput-report",
                        type=Path,
                        help=argparse.SUPPRESS)
    parser.add_argument("--batch-size",
                        type=int,
                        default=DEFAULT_BATCH_SIZE,
                        help="Batch size for warmup and timed runs.")
    parser.add_argument("--trace",
                        default=DEFAULT_TRACE,
                        help="Comma-separated nsys trace domains.")
    parser.add_argument("--reports",
                        default=DEFAULT_REPORTS,
                        help="Comma-separated nsys stats reports.")
    parser.add_argument("--pytorch-python",
                        default=DEFAULT_PYTORCH_PYTHON,
                        help="Python executable for the PyTorch backend profile.")
    parser.add_argument("--tensorrt-python",
                        default=DEFAULT_TENSORRT_PYTHON,
                        help="Python executable for the TensorRT backend profile.")
    parser.add_argument("--vllm-python",
                        default=DEFAULT_VLLM_PYTHON,
                        help="Python executable for the vLLM backend profile.")
    parser.add_argument(
        "--tensorrt-package-root",
        type=Path,
        default=DEFAULT_TENSORRT_PACKAGE_ROOT,
        help="Staged package root for the native TensorRT backend profile.")

    args = parser.parse_args()
    args.dataset_map = DEFAULT_DATASETS
    if args.inner_run:
        if isinstance(args.backend, list):
            args.backend = args.backend[0]
        if isinstance(args.workload, list):
            args.workload = args.workload[0]
        return args
    if args.backend is None:
        args.backend = ["pytorch", "tensorrt"]
    if args.workload is None:
        args.workload = ["decode_heavy"]
    return args


def main() -> int:
    args = parse_args()
    if args.inner_run:
        return run_inner(args)

    artifacts = []
    for backend in args.backend:
        for workload in args.workload:
            artifacts.append(asdict(profile_one(args, backend, workload)))

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"runs": artifacts}, indent=2))
    print(json.dumps({"runs": artifacts}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        traceback.print_exc()
        raise SystemExit(exc.returncode)
