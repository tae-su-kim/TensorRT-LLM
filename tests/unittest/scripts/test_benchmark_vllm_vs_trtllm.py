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
"""Unit tests for the cross-stack throughput benchmark harness."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_vllm_vs_trtllm as benchmark_script


def test_load_tokenized_dataset_and_vllm_prompts(tmp_path: Path) -> None:
    """The harness should parse TRT-LLM JSONL requests and convert them for vLLM."""
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                '{"task_id":0,"input_ids":[1,2,3,4],"output_tokens":6}',
                '{"task_id":1,"input_ids":[9,8,7,6],"output_tokens":6}',
            ]
        )
        + "\n"
    )

    samples = benchmark_script.load_tokenized_dataset(dataset_path)
    benchmark_script.validate_fixed_lengths(samples, expected_input_len=4, expected_output_len=6)
    prompts = benchmark_script.build_vllm_prompts(samples)

    assert [sample.task_id for sample in samples] == [0, 1]
    assert prompts == [
        {"prompt_token_ids": [1, 2, 3, 4]},
        {"prompt_token_ids": [9, 8, 7, 6]},
    ]


def test_validate_fixed_lengths_rejects_mismatch() -> None:
    """The shared workload validation should fail fast on the wrong prompt length."""
    samples = [benchmark_script.RequestSample(task_id=0, input_ids=[1, 2], output_tokens=4)]
    with pytest.raises(ValueError, match="input tokens"):
        benchmark_script.validate_fixed_lengths(
            samples,
            expected_input_len=4,
            expected_output_len=4,
        )


def test_build_pytorch_backend_config_contains_expected_fields() -> None:
    """The generated TRT-LLM config should expose the expected tuning knobs."""
    config_text = benchmark_script.build_pytorch_backend_config([8, 1, 4, 4])
    payload = yaml.safe_load(config_text)

    assert payload["enable_chunked_prefill"] is True
    assert payload["cuda_graph_config"]["enable_padding"] is True
    assert payload["cuda_graph_config"]["batch_sizes"] == [1, 4, 8]
    assert payload["torch_compile_config"]["enable_piecewise_cuda_graph"] is True
    assert payload["kv_cache_config"]["free_gpu_memory_fraction"] == pytest.approx(0.8)


def test_compute_metrics_matches_expected_rates() -> None:
    """Normalized throughput math should stay consistent across stacks."""
    metrics = benchmark_script.compute_metrics(
        num_requests=1000,
        input_len=1024,
        output_len=1024,
        total_latency_s=10.0,
    )

    assert metrics.request_throughput_req_s == pytest.approx(100.0)
    assert metrics.output_throughput_tok_s == pytest.approx(102400.0)
    assert metrics.total_throughput_tok_s == pytest.approx(204800.0)
    assert metrics.total_latency_ms == pytest.approx(10000.0)


def test_build_vllm_generate_kwargs_supports_prompts_and_inputs() -> None:
    """The vLLM call site should tolerate both common generate signatures."""

    def generate_with_prompts(prompts, sampling_params, use_tqdm=True):  # noqa: ARG001
        return None

    def generate_with_inputs(inputs, sampling_params):  # noqa: ARG001
        return None

    prompts_kwargs = benchmark_script.BenchmarkRunner.build_vllm_generate_kwargs(
        generate_with_prompts,
        prompts=[{"prompt_token_ids": [1, 2, 3]}],
        sampling_params="params",
    )
    inputs_kwargs = benchmark_script.BenchmarkRunner.build_vllm_generate_kwargs(
        generate_with_inputs,
        prompts=[{"prompt_token_ids": [1, 2, 3]}],
        sampling_params="params",
    )

    assert prompts_kwargs == {
        "prompts": [{"prompt_token_ids": [1, 2, 3]}],
        "sampling_params": "params",
        "use_tqdm": False,
    }
    assert inputs_kwargs == {
        "inputs": [{"prompt_token_ids": [1, 2, 3]}],
        "sampling_params": "params",
    }


def test_filter_supported_kwargs_keeps_var_keyword_arguments() -> None:
    """Callable signatures with ``**kwargs`` should keep caller-provided tuning args."""

    def init_with_kwargs(model, **kwargs):  # noqa: ARG001
        return None

    filtered = benchmark_script.BenchmarkRunner.filter_supported_kwargs(
        init_with_kwargs,
        {
            "model": "model-id",
            "max_model_len": 2048,
            "max_num_seqs": 8,
        },
    )

    assert filtered == {
        "model": "model-id",
        "max_model_len": 2048,
        "max_num_seqs": 8,
    }


def test_build_subprocess_env_prefers_absolute_executable_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Child processes should resolve helper binaries from the selected Python environment first."""
    monkeypatch.setenv("PATH", "/venv/main/bin:/usr/bin")

    env = benchmark_script.BenchmarkRunner.build_subprocess_env(
        ["/venv/trtllm/bin/python", "-m", "tensorrt_llm.commands.bench"]
    )

    assert env["PATH"].split(":")[:3] == ["/venv/trtllm/bin", "/venv/main/bin", "/usr/bin"]


@pytest.mark.parametrize(
    ("stdout", "stderr", "expected"),
    [
        ("", "CUDA out of memory", "oom"),
        ("", "ModuleNotFoundError: No module named 'vllm'", "missing_dependency"),
        ("", "This model is not supported yet", "unsupported"),
        ("", "some other failure", "runtime_failure"),
    ],
)
def test_classify_failure_reason(stdout: str, stderr: str, expected: str) -> None:
    """Failure classification should stay summary-friendly and deterministic."""
    assert benchmark_script.classify_failure_reason(stdout, stderr) == expected


def test_best_results_and_markdown_rendering() -> None:
    """The summary should pick the best successful row per stack."""
    results = [
        benchmark_script.SweepResult(
            stack="trtllm-pytorch",
            backend="pytorch",
            concurrency=1,
            status="passed",
            request_throughput_req_s=1.0,
            output_throughput_tok_s=100.0,
            total_throughput_tok_s=200.0,
            total_latency_ms=10.0,
        ),
        benchmark_script.SweepResult(
            stack="trtllm-pytorch",
            backend="pytorch",
            concurrency=2,
            status="passed",
            request_throughput_req_s=2.0,
            output_throughput_tok_s=150.0,
            total_throughput_tok_s=300.0,
            total_latency_ms=8.0,
        ),
        benchmark_script.SweepResult(
            stack="vllm",
            backend="vllm",
            concurrency=1,
            status="failed",
            reason="oom",
        ),
    ]

    best_rows = benchmark_script.best_successful_results(results)
    assert len(best_rows) == 1
    assert best_rows[0].concurrency == 2

    markdown = benchmark_script.render_summary_markdown(
        model="Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
        num_requests=1000,
        input_len=1024,
        output_len=1024,
        warmup_requests=32,
        concurrencies=[1, 2],
        dataset_path=Path("dataset.jsonl"),
        results=results,
    )
    assert "Best Successful Result" in markdown
    assert "trtllm-pytorch" in markdown
    assert "vLLM prompt mode: shared token ids via `prompt_token_ids`" in markdown
