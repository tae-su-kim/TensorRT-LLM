<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Work Status

Updated: 2026-04-16

## Executive Summary

- Goal:
  - bring up `Qwen/Qwen3.5-0.8B` on the legacy TensorRT backend native C++ executor path
  - benchmark it against `vLLM`
  - identify the highest-value remaining optimization targets
- What we have done:
  - enabled and iterated on the TensorRT recurrent path for Qwen3.5
  - added and benchmarked multiple recurrent-side optimizations:
    - fused `GatedDelta` plugin
    - decode-specialized recurrent path
    - FlashInfer prefill fast path
    - FlashInfer-style decode fast path
    - TensorRT-side `g` / `beta` computation cleanup
    - decode greedy / penalty-skip runtime fast path
  - updated the Qwen3.5 TensorRT config path so recurrent `state_dtype` follows model dtype for `bfloat16` /
    `float16`
  - fixed the local rebuild path by replacing stale repo-native libraries with a consistent rebuilt native set
- Best kept TensorRT baseline today:
  - engine: `/tmp/qwen35_native_bench_engine_bs64_qwen_update_conv`
  - mixed `1024 -> 1024`: `14793.09` tok/s
  - decode-heavy `1 -> 1024`: `16468.63` tok/s
  - prefill-heavy `1024 -> 1`: `0.403s`
- Best kept gap vs `vLLM`:
  - mixed: TensorRT is `14.5%` slower
  - decode-heavy: TensorRT is `7.0%` slower
  - prefill-heavy: TensorRT is `15.5%` lower latency
- Current repo-native rebuild status:
  - the rebuild path is working again
  - a fresh repo-native engine can now be rebuilt reproducibly from the current checkout
  - that fresh direct-state-write rebuild regressed:
    - mixed: `10619.18` tok/s
    - decode-heavy: `11698.87` tok/s
    - prefill-heavy: `0.411s`
  - it is therefore not a replacement for the current kept TensorRT baseline
- Practical conclusion:
  - the next priority is not more Qwen-only conv tuning
  - the next priority is to recover the missing high-throughput decode runtime path in a clean repo-native rebuild

## Reproducibility

### Reproducibility Scope

- There are two useful environments to reproduce:
  - the current kept TensorRT performance baseline for comparison against `vLLM`
  - the current repo-native rebuild path from source, which is reproducible but slower
- Use the kept baseline when you want to rerun the strongest local TensorRT-vs-vLLM comparison.
- Use the repo-native rebuild path when you want to validate source changes or rebuild the engine from the current
  checkout.

### Hardware And Software Assumptions

- GPU: `1x H100 80GB`
- Repo root: `/workspace/TensorRT-LLM`
- TensorRT / main Python env: `/venv/main`
- vLLM Python env: `/venv/vllm`
- HF snapshot used in all current controls:
  - `/workspace/.hf_home/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17`

### Local Artifact Paths

- Working kept TensorRT baseline package root:
  - `/tmp/qwen35_native_pkg`
- Working kept TensorRT baseline engine:
  - `/tmp/qwen35_native_bench_engine_bs64_qwen_update_conv`
- Lean repo-native build tree:
  - `/tmp/trtllm_qwen35_native_build4`
- Rebuilt repo-native staged package root:
  - `/tmp/qwen35_native_pkg_repo_directstate`
- Fresh repo-native rebuilt engine:
  - `/tmp/qwen35_native_bench_engine_bs64_directstate`
- Build config used for bs64 rebuild:
  - `/tmp/qwen35_build_config_bs64_qwen_update_conv.json`
- Checkpoint used for bf16-state rebuild:
  - `/tmp/qwen35_native_bench_ckpt_bf16_state`
- Synthetic datasets:
  - mixed `128 x 1024 -> 1024`: `/tmp/qwen35_bench_dataset_128x1024_1024.jsonl`
  - decode-heavy `128 x 1 -> 1024`: `/tmp/qwen35_bench_dataset_128x1_1024.jsonl`
  - prefill-heavy `128 x 1024 -> 1`: `/tmp/qwen35_bench_dataset_128x1024_1.jsonl`

### Required Environment Variables

- Always set:
  - `HWLOC_COMPONENTS=-gl`
- For local TensorRT rebuild / runtime bring-up:
  - `TLLM_DISABLE_MPI=1`
- For staged repo-native package import:
  - `TRT_LLM_MINIMAL_IMPORT=1`
  - `PYTHONPATH=/tmp/qwen35_native_pkg_repo_directstate`
  - `LD_LIBRARY_PATH=/tmp/qwen35_native_pkg_repo_directstate/tensorrt_llm/libs:/venv/main/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH}`

### Reproduce The Kept TensorRT Baseline

- This is the easiest way to rerun the strongest local TensorRT-vs-vLLM comparison.
- Use the current checked-in `nsys` harness:

```bash
cd /workspace/TensorRT-LLM
HWLOC_COMPONENTS=-gl TLLM_DISABLE_MPI=1 /venv/main/bin/python \
  scripts/profile_qwen35_nsys.py \
  --backend tensorrt \
  --backend vllm \
  --workload mixed \
  --workload decode_heavy \
  --workload prefill_heavy \
  --engine-dir /tmp/qwen35_native_bench_engine_bs64_qwen_update_conv \
  --tensorrt-package-root /tmp/qwen35_native_pkg \
  --output-dir /tmp/qwen35_nsys_profiles_compare_trt_vllm_repro_20260416
```

- Source-of-truth kept baseline reports on disk:
  - mixed TensorRT: `/tmp/qwen35_qwen_update_conv_mixed_inner_report.json`
  - decode-heavy TensorRT: `/tmp/qwen35_qwen_update_conv_decode_inner_report.json`
  - prefill-heavy TensorRT: `/tmp/qwen35_qwen_update_conv_prefill_inner_report.json`
  - mixed vLLM: `/tmp/qwen35_vllm_bs64_bench_report.json`
  - decode-heavy vLLM: `/tmp/qwen35_vllm_decode_heavy_report.json`
  - prefill-heavy vLLM: `/tmp/qwen35_vllm_prefill_heavy_report.json`

### Reproduce The Repo-Native Rebuild Path

#### 1. Rebuild Native Libraries

- Reuse the lean `sm90` build tree:

```bash
cmake --build /tmp/trtllm_qwen35_native_build4 \
  --target nvinfer_plugin_tensorrt_llm tensorrt_llm th_common bindings \
  -j 8
```

- The current build tree is already configured against the repo checkout at `/workspace/TensorRT-LLM/cpp`.
- If it must be recreated from scratch, use `/tmp/trtllm_qwen35_native_build4/CMakeCache.txt` as the source of truth.
- Important current cache characteristics:
  - `CMAKE_CUDA_ARCHITECTURES=90`
  - `BUILD_PYT=ON`
  - `BUILD_TESTS=OFF`
  - `BUILD_BENCHMARKS=OFF`
  - `BUILD_DEEP_EP=OFF`
  - `BUILD_DEEP_GEMM=OFF`
  - `BUILD_FLASH_MLA=ON`
  - `BUILD_CONTEXT_FMHA=ON`
  - `BUILD_DECODER_ATTENTION=ON`
  - `FAST_BUILD=ON`

#### 2. Create The Staged Repo-Native Package Overlay

- Build the staged runtime package by overlaying current repo Python sources on top of the known-good package root and
  replacing the stale native libraries with the freshly rebuilt ones:

```bash
rm -rf /tmp/qwen35_native_pkg_repo_directstate
cp -a /tmp/qwen35_native_pkg /tmp/qwen35_native_pkg_repo_directstate
rsync -a --exclude='*.so' --exclude='libs/' \
  /workspace/TensorRT-LLM/tensorrt_llm/ \
  /tmp/qwen35_native_pkg_repo_directstate/tensorrt_llm/
cp /tmp/trtllm_qwen35_native_build4/tensorrt_llm/nanobind/bindings.cpython-312-x86_64-linux-gnu.so \
  /tmp/qwen35_native_pkg_repo_directstate/tensorrt_llm/
cp /tmp/trtllm_qwen35_native_build4/tensorrt_llm/thop/libth_common.so \
  /tmp/qwen35_native_pkg_repo_directstate/tensorrt_llm/
cp /tmp/trtllm_qwen35_native_build4/tensorrt_llm/thop/libth_common.so \
  /tmp/qwen35_native_pkg_repo_directstate/tensorrt_llm/libs/libth_common.so
cp /tmp/trtllm_qwen35_native_build4/tensorrt_llm/libtensorrt_llm.so \
  /tmp/qwen35_native_pkg_repo_directstate/tensorrt_llm/libs/libtensorrt_llm.so
cp /tmp/trtllm_qwen35_native_build4/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so \
  /tmp/qwen35_native_pkg_repo_directstate/tensorrt_llm/libs/libnvinfer_plugin_tensorrt_llm.so
cp /tmp/trtllm_qwen35_native_build4/tensorrt_llm/runtime/utils/libpg_utils.so \
  /tmp/qwen35_native_pkg_repo_directstate/tensorrt_llm/libs/libpg_utils.so
```

#### 3. Smoke-Test The Staged Package

```bash
cd /tmp
PYTHONPATH=/tmp/qwen35_native_pkg_repo_directstate \
LD_LIBRARY_PATH=/tmp/qwen35_native_pkg_repo_directstate/tensorrt_llm/libs:/venv/main/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH} \
TRT_LLM_MINIMAL_IMPORT=1 TLLM_DISABLE_MPI=1 HWLOC_COMPONENTS=-gl \
/venv/main/bin/python -u - <<'PY'
import tensorrt_llm.bindings
from tensorrt_llm.plugin.plugin import _load_plugin_lib
from tensorrt_llm.builder import BuildConfig
_load_plugin_lib()
print("staged repo-native import ok")
PY
```

#### 4. Build The Fresh Repo-Native Engine

```bash
cd /tmp
rm -rf /tmp/qwen35_native_bench_engine_bs64_directstate
mkdir -p /tmp/qwen35_native_bench_engine_bs64_directstate
PYTHONPATH=/tmp/qwen35_native_pkg_repo_directstate \
LD_LIBRARY_PATH=/tmp/qwen35_native_pkg_repo_directstate/tensorrt_llm/libs:/venv/main/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH} \
HWLOC_COMPONENTS=-gl TLLM_DISABLE_MPI=1 TRT_LLM_MINIMAL_IMPORT=1 \
/venv/main/bin/python -u - <<'PY'
from pathlib import Path
import torch
from tensorrt_llm.builder import BuildConfig
from tensorrt_llm.commands.build import build_model
from tensorrt_llm.models import PretrainedConfig
from tensorrt_llm.plugin.plugin import _load_plugin_lib

torch.cuda.set_device(0)
_load_plugin_lib()

out_dir = Path("/tmp/qwen35_native_bench_engine_bs64_directstate")
ckpt_dir = Path("/tmp/qwen35_native_bench_ckpt_bf16_state")
cache_path = out_dir / "model.cache"

build_config = BuildConfig.from_json_file("/tmp/qwen35_build_config_bs64_qwen_update_conv.json")
build_config.input_timing_cache = str(cache_path) if cache_path.exists() else None
build_config.output_timing_cache = str(cache_path)

model_config = PretrainedConfig.from_json_file(ckpt_dir / "config.json")
engine = build_model(
    build_config,
    rank=0,
    ckpt_dir=str(ckpt_dir),
    model_config=model_config,
    logits_dtype=None,
    use_fused_mlp=True,
    lora_dir=None,
    lora_ckpt_source="hf",
    max_lora_rank=64,
    lora_target_modules=None,
    strip_plan=False,
    refit=False,
)
engine.save(str(out_dir))
print(sorted(p.name for p in out_dir.iterdir()))
PY
```

- Expected outputs:
  - `config.json`
  - `model.cache`
  - `rank0.engine`

#### 5. Benchmark The Fresh Repo-Native Engine

- Source-of-truth current report files:
  - mixed: `/tmp/qwen35_cpp_native_mixed_directstate_report.json`
  - decode-heavy: `/tmp/qwen35_cpp_native_decode_heavy_directstate_report.json`
  - prefill-heavy: `/tmp/qwen35_cpp_native_prefill_heavy_directstate_report.json`
- If you need `nsys` on the fresh repo-native engine, reuse the checked-in harness with the staged package root:

```bash
cd /workspace/TensorRT-LLM
HWLOC_COMPONENTS=-gl TLLM_DISABLE_MPI=1 /venv/main/bin/python \
  scripts/profile_qwen35_nsys.py \
  --backend tensorrt \
  --workload mixed \
  --workload decode_heavy \
  --workload prefill_heavy \
  --engine-dir /tmp/qwen35_native_bench_engine_bs64_directstate \
  --tensorrt-package-root /tmp/qwen35_native_pkg_repo_directstate \
  --output-dir /tmp/qwen35_nsys_profiles_trt_directstate_20260416
```

### Reproducibility Pitfalls

- If the builder cannot find `QwenCausalConv1d`, the repo-native plugin library is stale.
  - Rebuild `/tmp/trtllm_qwen35_native_build4` and recreate `/tmp/qwen35_native_pkg_repo_directstate`.
- If `bindings` import fails with an undefined `c10::MessageLogger` symbol, the staged package is mixing old and new
  native libraries.
  - Recreate the staged package overlay from scratch.
- If `build_model()` succeeds but the engine directory only contains `model.cache`, you forgot to call:
  - `engine.save(output_dir)`
- If `vLLM` is launched from stdin and crashes with `/tmp/<stdin>` under multiprocessing spawn, use a file-backed
  script instead of a heredoc.

## Qwen3.5 Findings

### Scope

- Model: `Qwen/Qwen3.5-0.8B`
- Goal: enable the legacy TensorRT backend native C++ executor path and profile the remaining throughput gap
  against `vLLM`

### Current Status

- Native C++ TensorRT execution for `Qwen3.5` is working.
- The kept high-watermark TensorRT baseline is still:
  - `/tmp/qwen35_native_bench_engine_bs64_qwen_update_conv`
- The fresh repo-native rebuild path is also working again:
  - build tree: `/tmp/trtllm_qwen35_native_build4`
  - staged package root: `/tmp/qwen35_native_pkg_repo_directstate`
  - rebuilt engine: `/tmp/qwen35_native_bench_engine_bs64_directstate`
- The strongest validated TensorRT improvements so far remain:
  - fused `GatedDelta`
  - FlashInfer prefill integration
  - decode runtime greedy / penalty-skip fast path
- The fresh repo-native direct-state-write conv rebuild did not beat either:
  - the kept TensorRT baseline
  - `vLLM`

### Environment Notes

- The OpenMPI / `hwloc_gl` workaround is still required locally:
  - `HWLOC_COMPONENTS=-gl`
- For the direct minimal-import smoke / profiling path, the TensorRT plugin library must be loaded before engine
  deserialization:
  - `from tensorrt_llm.plugin.plugin import _load_plugin_lib`
  - `_load_plugin_lib()`

### 2026-04-15 Rebaseline

- The earlier TensorRT-vs-vLLM comparison in this file was stale for the main Qwen3.5 path because the local
  `nsys` harness defaulted to the older engine `/tmp/qwen35_native_bench_engine_bs64_inkernel_qk`.
- The current kept TensorRT baseline is now:
  - engine: `/tmp/qwen35_native_bench_engine_bs64_qwen_update_conv`
  - harness default updated in `scripts/profile_qwen35_nsys.py`
- Fresh current-control reports on that engine:
  - mixed `1024 -> 1024`: `/tmp/qwen35_qwen_update_conv_mixed_inner_report.json`
  - decode-heavy `1 -> 1024`: `/tmp/qwen35_qwen_update_conv_decode_inner_report.json`
  - prefill-heavy `1024 -> 1`: `/tmp/qwen35_qwen_update_conv_prefill_inner_report.json`
- Matching current vLLM controls:
  - mixed: `/tmp/qwen35_vllm_bs64_bench_report.json`
  - decode-heavy: `/tmp/qwen35_vllm_decode_heavy_report.json`
  - prefill-heavy: `/tmp/qwen35_vllm_prefill_heavy_report.json`

### Current Gap Vs vLLM

All results below are single-H100 offline throughput at `128` requests and `batch size 64`.

#### Mixed workload: `1024 in / 1024 out`

- TensorRT native `qwen_update_conv`: `14793.09` output tok/s
  - elapsed: `8.860s`
- vLLM: `16936.33` output tok/s
  - elapsed: `7.739s`
- Gap:
  - `vLLM / TensorRT = 1.145x`
  - `vLLM` is about `14.5%` faster on this control

#### Decode-heavy workload: `1 in / 1024 out`

- TensorRT native `qwen_update_conv`: `16468.63` requested output tok/s
  - elapsed: `7.959s`
- vLLM: `17625.09` output tok/s
  - elapsed: `7.437s`
- Gap:
  - `vLLM / TensorRT = 1.070x`
  - `vLLM` is about `7.0%` faster on this control

#### Prefill-heavy workload: `1024 in / 1 out`

- TensorRT native `qwen_update_conv` elapsed: `0.403s`
- vLLM elapsed: `0.477s`
- Gap:
  - TensorRT latency is about `15.5%` lower than `vLLM` on this control

### Current Decode Targets

- The remaining performance gap is no longer prefill-dominated. Pure prefill is already competitive or better than
  `vLLM` on this workload.
- The highest-value remaining decode-side targets from the current `nsys` trace are:
  - GEMM kernels: `28.7%` of timed decode-heavy GPU kernel time
  - `qwenGatedDeltaDecodeKernel`: `27.2%`
  - `kernel_mha`: `9.3%`
  - fused GEMM tail: `7.5%`
  - `causal_conv1d_update_kernel`: `3.4%`
  - post-recurrent helper chain (`__myl_*` kernels): about `6-7%` combined
- Timed decode-heavy memcpy breakdown is still mostly `Device-to-Device`:
  - `81.7%` D2D
  - `9.8%` H2D
  - `7.9%` D2H

### Local Code Changes Landed

- `scripts/profile_qwen35_nsys.py`
  - default TensorRT engine updated to `/tmp/qwen35_native_bench_engine_bs64_qwen_update_conv`
  - default workload map now includes `prefill_heavy`
- `tensorrt_llm/models/qwen/config.py`
  - Qwen3.5 recurrent `state_dtype` now follows the model dtype for `bfloat16` / `float16` models instead of being
    hard-pinned to `float32`
- `tests/unittest/trt/model/test_qwen.py`
  - Qwen3.5 config expectation updated to reflect the new `bfloat16` default on the current HF config fixture
- Local native-kernel prototype also staged:
  - direct write of Qwen conv final state in `causalConv1d.cu`
  - removal of the plugin-side pre-copy in `qwenCausalConv1dPlugin`
  - this still needs a fresh native rebuild and benchmark before it should be treated as a validated improvement

### Historical Rebuild Blocker

- The earlier local rebuild blocker is now resolved.
- The actual root causes were:
  - stale repo-native libraries that did not register `QwenCausalConv1d`
  - an older staged package whose `libth_common.so` / `bindings` pair was built against a different `torch` ABI
- The resolved path is:
  - rebuild native libraries in `/tmp/trtllm_qwen35_native_build4`
  - recreate `/tmp/qwen35_native_pkg_repo_directstate`
  - rebuild the engine from that staged package
- The rebuilt repo-native path is reproducible, but the resulting direct-state-write engine is slower than the current
  kept TensorRT baseline.

### Closest Existing bf16-State Evidence

- The nearest previously built `bf16`-state engines already on disk are older than the current `qwen_update_conv`
  baseline:
  - mixed: `/tmp/qwen35_cpp_native_executor_bs64_bench_bf16_state_report.json`
    - `8909.39` requested output tok/s
  - decode-heavy: `/tmp/qwen35_cpp_native_decode_heavy_report_bf16_state.json`
    - `9750.26` requested output tok/s
- Those older `bf16`-state artifacts are slower than the current `qwen_update_conv` baseline, so they should not be
  used as the final answer for the current path.
- The practical conclusion is:
  - changing recurrent state dtype alone is not enough to beat `vLLM`
  - the real remaining upside is still in the current decode kernel path and its surrounding GEMM / helper tail
  - the staged direct-state-write conv change should be re-benchmarked once the local build path is working again

### End-to-End Throughput

All results below are single-GPU offline throughput at `128` requests and `batch size 64`.

#### Mixed workload: `1024 in / 1024 out`

- TensorRT native FlashInfer-prefill + internal gating path: `10916.98` output tok/s
  - report: `/tmp/qwen35_cpp_native_executor_bs64_bench_gating_cleanup_report.json`
- TensorRT native FlashInfer-prefill + FlashInfer-style decode path: `10480.47` output tok/s
  - report: `/tmp/qwen35_cpp_native_executor_bs64_bench_flashinfer_decode_report.json`
- Prior TensorRT native fused + decode-specialized path: `6978.41` output tok/s
  - report: `/tmp/qwen35_cpp_native_executor_bs64_bench_decodeopt_report.json`
- Prior fused-only TensorRT point: `6400.01` output tok/s
  - report: `/tmp/qwen35_cpp_native_executor_bs64_bench_fused_plugin_report.json`
- vLLM: `16936.33` output tok/s
  - report: `/tmp/qwen35_vllm_bs64_bench_report.json`
- Gap:
  - `vLLM / TensorRT = 1.62x`

#### Decode-heavy workload: `1 in / 1024 out`

- TensorRT native internal-gating decode path: `11825.17` requested output tok/s
  - report: `/tmp/qwen35_cpp_native_decode_heavy_report_gating_cleanup.json`
- TensorRT native FlashInfer-style decode path: `11436.75` requested output tok/s
  - report: `/tmp/qwen35_cpp_native_decode_heavy_report_flashinfer_decode.json`
- Prior TensorRT native fused + decode-specialized path: `10525.57` requested output tok/s
  - report: `/tmp/qwen35_cpp_native_decode_heavy_report_decodeopt.json`
- Prior fused-only TensorRT point: `9045.65` requested output tok/s
  - report: `/tmp/qwen35_cpp_native_decode_heavy_report.json`
- vLLM: `17625.09` output tok/s
  - report: `/tmp/qwen35_vllm_decode_heavy_report.json`
- Gap:
  - `vLLM / TensorRT = 1.54x`

#### Prefill-heavy workload: `1024 in / 1 out`

- Prior TensorRT native fused elapsed time: `5.77s`
  - report: `/tmp/qwen35_cpp_native_prefill_heavy_report.json`
- TensorRT native FlashInfer-prefill elapsed time: `0.485s`
  - report: `/tmp/qwen35_cpp_native_prefill_heavy_report_flashinfer_prefill.json`
- vLLM elapsed time: `0.48s`
  - report: `/tmp/qwen35_vllm_prefill_heavy_report.json`
- Normalized by input tokens:
  - prior TensorRT: `22707.13` input tok/s
  - FlashInfer-prefill TensorRT: `270187.55` input tok/s
  - vLLM: `274946.07` input tok/s
  - FlashInfer-prefill gap vs `vLLM`: `1.02x`

### Kernel-Level Findings

- Context-style recurrent block benchmark:
  - fused vs unfused: `2.47x` faster
  - report: `/tmp/qwen35_gated_delta_context_block_bench.json`
- The older shared-kernel decode block microbenchmark is now stale for the end-to-end decode path:
  - old fused-vs-unfused result: `0.92x`
  - report: `/tmp/qwen35_gated_delta_decode_block_bench.json`
  - the new decode-specialized plugin path improved end-to-end decode-heavy throughput, so the block benchmark
    should be rerun against the new engine/kernel dispatch before using it for further kernel-level conclusions

### Interpretation

- The fused plugin materially helped end-to-end throughput:
  - old native TensorRT bs64 mixed point: `5112.40` tok/s
  - current fused native TensorRT bs64 mixed point: `6400.01` tok/s
  - improvement: about `25.2%`
- The decode-specialized path provided another measurable uplift:
  - mixed `1024/1024`: `6400.01` -> `6978.41` tok/s
  - improvement: about `9.0%`
  - decode-heavy `1/1024`: `9045.65` -> `10525.57` requested tok/s
  - improvement: about `16.4%`
- The FlashInfer-style decode integration provided another measurable uplift on top of that:
  - mixed `1024/1024`: `6978.41` -> `10480.47` tok/s
  - improvement: about `50.2%`
  - decode-heavy `1/1024`: `10525.57` -> `11436.75` requested tok/s
  - improvement: about `8.7%`
- Moving `g/beta` computation from the TensorRT graph into the plugin provided another smaller but real uplift:
  - mixed `1024/1024`: `10480.47` -> `10916.98` tok/s
  - improvement: about `4.2%`
  - decode-heavy `1/1024`: `11436.75` -> `11825.17` requested tok/s
  - improvement: about `3.4%`
- Moving Q/K normalization and query scaling into the TensorRT recurrent kernels provided another smaller decode-side
  uplift:
  - mixed `1024/1024`: `10916.98` -> `10934.05` tok/s
  - improvement: about `0.16%`
  - decode-heavy `1/1024`: `11825.17` -> `12130.36` requested tok/s
  - improvement: about `2.58%`
  - engine/report artifacts:
    - engine: `/tmp/qwen35_native_bench_engine_bs64_inkernel_qk`
    - mixed report: `/tmp/qwen35_cpp_native_executor_bs64_bench_inkernel_qk_report.json`
    - decode-heavy report: `/tmp/qwen35_cpp_native_decode_heavy_report_inkernel_qk.json`
- A follow-up experiment to keep Q/K in grouped-head layout and remove TensorRT-graph `repeat_interleave`
  expansion was benchmarked and rejected:
  - mixed `1024/1024`: `10916.98` -> `10833.58` tok/s
  - decode-heavy `1/1024`: `11825.17` -> `11758.86` requested tok/s
  - engine/report artifacts:
    - engine: `/tmp/qwen35_native_bench_engine_bs64_grouped_qk`
    - mixed report: `/tmp/qwen35_cpp_native_executor_bs64_bench_grouped_qk_report.json`
    - decode-heavy report: `/tmp/qwen35_cpp_native_decode_heavy_report_grouped_qk.json`
  - conclusion: the current grouped-head no-expand path is slightly slower than the expanded-head baseline, so it
    should not be kept without additional kernel work.
- A follow-up mixed-batch plugin split experiment was also benchmarked and rejected:
  - approach: split contiguous `context` prefix and `generation` suffix inside the plugin, run FlashInfer prefill on
    the prefix, and run the decode fast path on the suffix
  - mixed `1024/1024`: `10916.98` -> `10735.45` tok/s
  - decode-heavy `1/1024`: `11825.17` -> `11929.48` requested tok/s
  - engine/report artifacts:
    - engine: `/tmp/qwen35_native_bench_engine_bs64_mixed_split`
    - mixed report: `/tmp/qwen35_cpp_native_executor_bs64_bench_mixed_split_report.json`
    - decode-heavy report: `/tmp/qwen35_cpp_native_decode_heavy_report_mixed_split.json`
  - conclusion: the split path helps pure decode slightly but hurts the full mixed workload, so it should not be kept
    in its current form.
- A follow-up decode-conv experiment to reuse the native `MambaConv1d` TensorRT plugin from the Qwen3.5 recurrent
  path was also benchmarked and rejected:
  - implementation: route Qwen3.5 decode-time conv through the existing native conv plugin using the current Qwen
    conv-state layout bridged by transposes and a zero-bias adapter
  - mixed `1024/1024`: `10934.05` -> `10564.64` tok/s
  - decode-heavy `1/1024`: `12130.36` -> `11583.64` requested tok/s
  - engine/report artifacts:
    - engine: `/tmp/qwen35_native_bench_engine_bs64_conv_plugin`
    - mixed report: `/tmp/qwen35_cpp_native_executor_bs64_bench_conv_plugin_report.json`
    - decode-heavy report: `/tmp/qwen35_cpp_native_decode_heavy_report_conv_plugin.json`
  - conclusion: direct reuse of the existing `MambaConv1d` plugin is slower for Qwen3.5 as wired today; the
    transpose/layout-bridge overhead appears to erase the expected decode benefit, so this path should not be kept
    without a Qwen-native conv kernel or a layout-compatible plugin contract.
- A narrower Qwen-native decode-conv fast path that mirrors the PyTorch backend structure was then benchmarked and
  kept:
  - implementation: add a dedicated decode-only `QwenCausalConv1d` plugin that calls
    `causal_conv1d_update_cuda` directly on the existing Qwen conv-state layout `[B, C, W-1]`, then feed its output
    into the existing GatedDelta decode plugin
  - mixed `1024/1024`: `10934.05` -> `11699.00` tok/s
  - decode-heavy `1/1024`: `12130.36` -> `12843.10` requested tok/s
  - engine/report artifacts:
    - engine: `/tmp/qwen35_native_bench_engine_bs64_qwen_update_conv`
    - mixed report: `/tmp/qwen35_cpp_native_executor_bs64_bench_qwen_update_conv_report.json`
    - decode-heavy report: `/tmp/qwen35_cpp_native_decode_heavy_report_qwen_update_conv.json`
  - conclusion: a Qwen-native `causal_conv1d_update` path is directionally correct and materially better than both
    the graph `conv2d` baseline and the earlier generic `MambaConv1d` reuse attempt.
- A follow-up grouped-head internalization experiment to remove TensorRT-graph `repeat_interleave(query/key)` by
  making the GatedDelta plugin expand grouped query/key heads internally was benchmarked and rejected:
  - implementation: keep Qwen query/key in grouped-head layout for the plugin path and replicate to value heads
    inside the native preprocess/decode kernels
  - mixed `1024/1024`: `11699.00` -> `3534.05` tok/s
  - decode-heavy `1/1024`: `12843.10` -> `9498.27` requested tok/s
  - engine/report artifacts:
    - engine: `/tmp/qwen35_native_bench_engine_bs64_grouped_heads_internal`
    - mixed report: `/tmp/qwen35_cpp_native_executor_bs64_bench_grouped_heads_internal_report.json`
    - decode-heavy report: `/tmp/qwen35_cpp_native_decode_heavy_report_grouped_heads_internal.json`
  - conclusion: removing graph-side grouped-head expansion this way is dramatically slower in the current TensorRT
    recurrent stack, so the value-head-expanded plugin contract should stay in place until the recurrent kernels are
    substantially redesigned.
- The FlashInfer-prefill path materially closed the context-side gap:
  - prefill-heavy elapsed: `5.772s` -> `0.485s`
  - improvement: about `11.90x`
  - prefill-heavy throughput: `22.7k` -> `270.2k` input tok/s
  - result is now within about `1.8%` of `vLLM` on this control workload
- With both prefill and pure-decode fast paths in place, the remaining mixed-workload gap is materially smaller and is
  still best interpreted as decode-side plus mixed-batch scheduling overhead.

### Main Bottlenecks

- Decode-side recurrent kernel quality is still not where it needs to be.
  - The FlashInfer-style decode path improved the real decode-heavy point, but there is still a `1.54x` gap against
    `vLLM`.
  - The next decode investigation should focus on the specialized update kernel itself, not the older shared-kernel
    microbenchmark.
- Prefill/context is no longer the dominant blocker on the pure-context control workload.
  - The FlashInfer-backed prefill path closed most of that gap for uniform pure-context batches.
  - Remaining context-side work is still needed for broader cases, especially packed-input / non-uniform scheduling.
- Mixed-batch behavior still has room to improve.
  - The mixed `1024/1024` point improved much more than the pure decode control once the engine was rebuilt with the
    shared K-last recurrent state layout, but it still trails `vLLM` by `1.62x`.
  - That points to remaining work in the decode fast path and in how mixed context/decode batches fall back or bridge
    between the two fast paths.
- The native C++ executor itself is no longer the primary blocker.
  - The work to admit and execute direct recurrent-state engines is functioning.
  - The remaining loss is now mostly in the decode model/kernel path.
- Compared with the TensorRT-LLM PyTorch backend on the same bs64 synthetic runs:
  - PyTorch mixed: `12523.17` tok/s vs TensorRT mixed: `11699.00` tok/s
  - PyTorch decode-heavy: `18501.23` tok/s vs TensorRT decode-heavy: `12843.10` tok/s
  - TensorRT is now within about `1.07x` on mixed, but still behind by about `1.44x` on decode-heavy.

### Why vLLM Is Ahead

- `vLLM` is using specialized kernels and scheduling for this model family:
  - `FlashInfer GDN prefill kernel`
  - `FlashAttention version 3`
  - cached compilation / CUDA graph capture
  - packed and chunked prefill scheduling
- TensorRT-LLM now has a competitive pure-context prefill path, but its decode recurrent path is still less mature.

### Notes On Report Interpretation

- For the TensorRT native control workloads, `ModelRunnerCpp.generate()` pads returned sequences to engine limits on
  this path.
- Because of that, for the `1 in / 1024 out` and `1024 in / 1 out` control runs:
  - use `requested_generated_tokens_total` and elapsed time for decode-heavy comparison
  - use elapsed time or normalized input-token throughput for prefill-heavy comparison
  - do not use `actual_generated_tokens_total` from the TensorRT control reports as the primary metric

### Next Steps

1. Re-profile and optimize the single-token FlashInfer-style decode path and its remaining fallback conditions.
2. Remove the remaining TensorRT-graph grouped-head expansion for the decode fast path without regressing mixed
   throughput.
3. Add better mixed-batch bridging only if it can outperform the current single-path baseline on the full
   `1024 in / 1024 out` workload.

## Objective

Benchmark `Qwen/Qwen3-30B-A3B-Thinking-2507-FP8` on `1x H100 80GB` for the `1024 tokens in / 1024 tokens out` workload and compare:

- TensorRT-LLM PyTorch backend
- TensorRT-LLM TensorRT backend
- vLLM

## Environment Findings

- TRT-LLM startup was initially blocked by OpenMPI / `hwloc_gl` probing X11 displays. The local workaround is:
  - `HWLOC_COMPONENTS=-gl`
- TRT-LLM PyTorch backend also needed the TRT-LLM venv ahead of the default shell `PATH` so JIT helpers resolve the correct package metadata:
  - `PATH=/venv/trtllm/bin:$PATH`

## Backend Findings

- PyTorch backend works in this environment after the two fixes above.
- The direct TensorRT build path from the Hugging Face FP8 release `Qwen/Qwen3-30B-A3B-Thinking-2507-FP8` failed in the MoE plugin path with:
  - `PLUGIN_V2_MixtureOfExperts_0: could not find any supported formats consistent with input/output data types`
- Building from a TensorRT-LLM FP8 checkpoint produced by `examples/quantization/quantize.py` works.
- Conclusion so far: for this model, the TensorRT backend wants a TensorRT-LLM checkpoint, not the raw Hugging Face FP8 release artifact.

## TensorRT FP8 Path

- Source model used for export: `Qwen/Qwen3-30B-A3B-Thinking-2507`
- Export command path:
  - `examples/quantization/quantize.py`
- Working exported checkpoint:
  - `benchmark_runs/qwen3_30b_a3b_thinking_2507_fp8_quantize_calib1_20260414/checkpoint`
- Important caveat:
  - the successful export used `calib_size=1` and `calib_max_seq_length=128`
  - this is enough to validate engine build and throughput plumbing
  - this is not yet a quality-valid final quantization recipe

## TensorRT Benchmark Progress

- First working TensorRT engine:
  - `benchmark_runs/qwen3_30b_a3b_thinking_2507_fp8_engine_build_from_tllm_ckpt_20260414/engine`
  - built with `max_batch_size=32`
- Corrected TensorRT engine:
  - `benchmark_runs/qwen3_30b_a3b_thinking_2507_fp8_engine_build_bs64_20260414/engine`
  - built with `max_batch_size=64`

## Throughput Results

All results below are for `1024/1024`, single GPU, offline throughput.

### Earlier 64-request points

- TRT-LLM PyTorch `c=16`: `1801.82` output tok/s
- TRT-LLM PyTorch `c=32`: `2759.55` output tok/s
- vLLM `c=16`: `2007.42` output tok/s
- vLLM `c=32`: `3120.69` output tok/s

### Matched TensorRT FP8 points from the corrected `max_batch_size=64` engine, using 128 requests

- TensorRT `c=32`: `3659.28` output tok/s
  - report: `benchmark_runs/qwen3_30b_a3b_thinking_2507_fp8_tensorrt_bs64_n128_c32_20260414/trtllm-tensorrt/concurrency_32/report.json`
- TensorRT `c=64`: `5970.99` output tok/s
  - report: `benchmark_runs/qwen3_30b_a3b_thinking_2507_fp8_tensorrt_bs64_n128_c64_20260414/trtllm-tensorrt/concurrency_64/report.json`

### Matched vLLM point on the same 128-request dataset

- vLLM `c=32`: `3158.90` output tok/s
  - report: `benchmark_runs/qwen3_30b_a3b_thinking_2507_vllm_n128_c32_20260414/report.json`

## Current Comparison

- On the matched 128-request run at `c=32`, TensorRT FP8 from the corrected `bs=64` engine is ahead of vLLM:
  - TensorRT FP8: `3659.28` output tok/s
  - vLLM: `3158.90` output tok/s
  - delta: about `15.8%`
- TensorRT FP8 at `c=64` is currently `5970.99` output tok/s.
- A matched vLLM `c=64` run on the 128-request dataset has not been completed yet.

## Open Items

- Run vLLM `c=64` on the 128-request dataset for a like-for-like comparison with the corrected TensorRT `bs=64` engine.
- Replace the minimal FP8 calibration recipe with a realistic calibration run before treating the TensorRT FP8 numbers as final quality-preserving results.
- If needed, rerun TensorRT `c=16` on the `bs=64` engine so all TensorRT points come from one consistent engine profile.

## Qwen3.5 Decode Fast Path

- Added a greedy / no-penalty fast path in the legacy TensorRT decoder for the pure decode case.
- Kept changes:
  - `cpp/tensorrt_llm/layers/penaltyLayer.cpp`
  - `cpp/tensorrt_llm/layers/samplingLayer.cpp`
  - `cpp/tensorrt_llm/layers/samplingLayer.h`
  - `cpp/tensorrt_llm/kernels/samplingTopKKernels.cu`
  - `cpp/tensorrt_llm/kernels/samplingTopKKernels.h`
- The fast path skips the generic penalty + softmax + TopK sampling stack when the active batch is effectively greedy:
  - neutral penalties
  - no output or cumulative log probs
  - no MinP
  - `topK=1`, `topP=1`
  - single-token decode (`maxTokensPerStep == 1`)

### Qwen3.5 Throughput After Greedy Fast Path

- Engine used:
  - `/tmp/qwen35_native_bench_engine_bs64_qwen_update_conv`
- Mixed `128 req, bs=64, 1024 in / 1024 out`:
  - `13960.37` tok/s
  - report: `/tmp/qwen35_cpp_native_executor_bs64_bench_greedy_fastpath_report.json`
- Decode-heavy `128 req, bs=64, 1 in / 1024 out`:
  - `15379.05` tok/s
  - report: `/tmp/qwen35_cpp_native_decode_heavy_greedy_fastpath.json`

### Delta Vs Previous Kept TensorRT Baseline

- Previous mixed baseline:
  - `12482.86` tok/s
  - report: `/tmp/qwen35_cpp_native_executor_bs64_bench_recurrent_rewrite_report.json`
- Previous decode-heavy baseline:
  - `13961.20` tok/s
  - report: `/tmp/qwen35_cpp_native_decode_heavy_recurrent_rewrite.json`
- Improvement:
  - mixed: about `+11.8%`
  - decode-heavy: about `+10.2%`

### Nsight Decode Trace

- New trace:
  - `/tmp/qwen35_nsys_profiles_trt_greedy_fastpath/tensorrt_decode_heavy.nsys-rep`
- Throughput under `nsys`:
  - `14333.07` tok/s
  - report: `/tmp/qwen35_nsys_profiles_trt_greedy_fastpath/tensorrt_decode_heavy_throughput.json`
- Previous TensorRT decode-heavy `nsys` throughput:
  - `12614.92` tok/s
  - report: `/tmp/qwen35_nsys_profiles_trt_recurrent_rewrite/tensorrt_decode_heavy_throughput.json`
- Trace comparison:
  - `addBiasSoftMax` dropped from `9.5%` of GPU kernel time to below the top buckets
  - `topKStage1` dropped from `4.1%` to below the top buckets
  - new `greedySampling` kernel is only `2.1%`
  - `batchApplyPenalty` remains at `7.6%`, which means the skip path is not yet engaging for all timed decode steps inside the native runtime

### Current Gap

- TensorRT decode-heavy under `nsys`: `14333.07` tok/s
- PyTorch decode-heavy under `nsys`: `17158.91` tok/s
- Remaining gap: about `1.20x`

### Next Likely Optimization

- The remaining decode gap is no longer dominated by TopK staging.
- Highest-value next targets:
  - understand why `batchApplyPenalty` is still present in the timed trace despite the new fast path
  - reduce remaining GEMM + helper-kernel overhead around the decode recurrent block
  - revisit decode attention only after the penalty-path leak is understood

## Qwen3.5 Penalty Skip Follow-Up

- Kept follow-up changes:
  - `cpp/include/tensorrt_llm/runtime/decodingInput.h`
  - `cpp/tensorrt_llm/runtime/decoderState.cpp`
  - `cpp/tensorrt_llm/runtime/gptDecoder.cpp`
  - `cpp/tensorrt_llm/batch_manager/createNewDecoderRequests.cpp`
  - `cpp/tensorrt_llm/layers/decodingParams.h`
  - `cpp/tensorrt_llm/layers/penaltyLayer.cpp`
  - `cpp/tensorrt_llm/layers/samplingLayer.cpp`
  - `cpp/tensorrt_llm/layers/samplingLayer.h`
- The follow-up makes the greedy fast path work on the batched `logitsVec` contract used by the native C++ runtime instead of only on dense `logits`.
- Important bring-up note:
  - after changing `DecodingInput` / `DecodingInputs` layout, the staged package also needed refreshed `libth_common.so` and `bindings.cpython-312-x86_64-linux-gnu.so`
  - otherwise decode-heavy crashed with an ABI mismatch in `DecodingInputs` teardown

### Qwen3.5 Throughput After Penalty Skip Wiring

- Mixed `128 req, bs=64, 1024 in / 1024 out`:
  - `14642.49` tok/s
  - report: `/tmp/qwen35_cpp_native_mixed_penalty_skip_report.json`
- Decode-heavy `128 req, bs=64, 1 in / 1024 out`:
  - `16425.51` tok/s
  - report: `/tmp/qwen35_cpp_native_decode_heavy_penalty_skip_report.json`

### Delta Vs Greedy Fast Path Baseline

- Previous mixed baseline:
  - `13960.37` tok/s
  - report: `/tmp/qwen35_cpp_native_executor_bs64_bench_greedy_fastpath_report.json`
- Previous decode-heavy baseline:
  - `15379.05` tok/s
  - report: `/tmp/qwen35_cpp_native_decode_heavy_greedy_fastpath.json`
- Improvement:
  - mixed: about `+4.9%`
  - decode-heavy: about `+6.8%`

### Updated Nsight Decode Trace

- New trace:
  - `/tmp/qwen35_nsys_profiles_trt_penalty_skip/tensorrt_decode_heavy.nsys-rep`
- Throughput under `nsys`:
  - `15493.14` tok/s
  - report: `/tmp/qwen35_nsys_profiles_trt_penalty_skip/tensorrt_decode_heavy_throughput.json`
- Key trace differences vs the previous greedy-fastpath trace:
  - `batchApplyPenalty` dropped out of the top GPU buckets
  - `qwenGatedDeltaDecodeKernel` is still large at `27.2%`
  - GEMM is now the top bucket at `28.7%`
  - `kernel_mha` is `9.3%`
  - `greedySampling` is `2.4%`
  - the remaining TensorRT helper stack is now mostly Qwen-side elementwise / reshape kernels around recurrent output:
    - `__myl_SlicReshSigmMul...`: `2.9%`
    - `__myl_CastSiluCastMulMeanAddSqrtDivMulMulMulCast...`: `2.0%`
    - `__myl_ReshAddCastMulMeanAddSqrtDivMulCastMul...`: `1.9%`
    - `__myl_SlicSiluSlicMul...`: `1.5%`

### Updated Decode-Heavy Gap

- TensorRT decode-heavy under `nsys`: `15493.14` tok/s
- PyTorch decode-heavy under `nsys`: `17158.91` tok/s
- Remaining gap: about `1.11x`

### Next Likely Optimization

- `batchApplyPenalty` is no longer the main blocker.
- Highest-value next targets:
  - fuse or replace the remaining Qwen post-recurrent helper chain (`__myl_*` RMSNorm / gating / reshape kernels)
  - reduce decode GEMM overhead around the Qwen recurrent block
  - only then revisit attention decode or another recurrent-kernel pass

## Qwen3.5 RMSNorm Plugin Experiment

- Tried a decode-side `QwenGatedRmsNorm` TensorRT plugin to replace the remaining post-recurrent helper chain.
- The first engine artifact built before the final plugin rebuild was invalid at load time:
  - deserialize assertion in `QwenGatedRmsNormPluginCreator::deserializePlugin`
  - root cause was stale serialized plugin payload versus the rebuilt native plugin library
- Rebuilt a clean engine from the current staged package:
  - engine: `/tmp/qwen35_native_bench_engine_bs64_qwen_rms_plugin_fresh`

### Results

- Mixed `128 req, bs=64, 1024 in / 1024 out`:
  - `10689.38` tok/s
  - report: `/tmp/qwen35_cpp_native_mixed_qwen_rms_plugin_report.json`
- That regressed the current stable baseline:
  - stable mixed baseline: `14642.49` tok/s
  - report: `/tmp/qwen35_cpp_native_mixed_penalty_skip_report.json`

### Stability

- Decode-heavy `1 in / 1024 out` was not stable on the clean rebuilt engine.
- It crashed in the decoder path during generation/teardown:
  - `DecodingInputs::~DecodingInputs()`
  - `DecodingLayer<float>::forwardAsync(...)`
  - `GptDecoderBatched::forwardAsync(...)`
- No valid decode-heavy throughput result was kept for this experiment.

### Outcome

- The `QwenGatedRmsNorm` plugin experiment was reverted from source.
- Current stable kept baseline remains:
  - mixed: `14642.49` tok/s
  - decode-heavy: `16425.51` tok/s

### Conclusion

- The remaining helper-kernel tail is not worth chasing through a new Qwen-specific RMSNorm plugin in the current legacy TensorRT path.
- The next viable optimization should move to shared decode infrastructure again:
  - decode attention path quality
  - shared GEMM/helper fragmentation
  - avoid more Qwen-only graph/plugin rewrites unless the shared-path options are exhausted

## 2026-04-16 Direct-State-Write Rebuild

- Rebuilt a fresh repo-native binary set from the current checkout using the lean `sm90` build tree:
  - build tree: `/tmp/trtllm_qwen35_native_build4`
  - rebuilt artifacts:
    - `libtensorrt_llm.so`
    - `libnvinfer_plugin_tensorrt_llm.so`
    - `libth_common.so`
    - `bindings.cpython-312-x86_64-linux-gnu.so`
- Staged those rebuilt binaries together with the current repo Python sources in:
  - `/tmp/qwen35_native_pkg_repo_directstate`
- Built a fresh bs64 engine from that staged package:
  - engine: `/tmp/qwen35_native_bench_engine_bs64_directstate`

### Rebuild Notes

- The earlier local rebuild blocker is resolved.
- Root causes were:
  - the repo checkout had stale native libraries that did not register `QwenCausalConv1d`
  - the older staged package carried a `libth_common.so` / `bindings` pair built against a different `torch` ABI
- The fresh build from `/tmp/trtllm_qwen35_native_build4` fixed both issues and produced a clean engine build and runtime load path.

### Results

- Fresh repo-native direct-state-write engine reports:
  - mixed `1024 -> 1024`: `/tmp/qwen35_cpp_native_mixed_directstate_report.json`
  - decode-heavy `1 -> 1024`: `/tmp/qwen35_cpp_native_decode_heavy_directstate_report.json`
  - prefill-heavy `1024 -> 1`: `/tmp/qwen35_cpp_native_prefill_heavy_directstate_report.json`

#### Mixed workload: `1024 in / 1024 out`

- Fresh repo-native direct-state-write engine:
  - `10619.18` output tok/s
  - elapsed: `12.343s`
- Current kept TensorRT baseline:
  - `14793.09` output tok/s
  - elapsed: `8.860s`
- vLLM control:
  - `16936.33` output tok/s
  - elapsed: `7.739s`
- Delta:
  - vs kept TensorRT baseline: `-28.2%`
  - vs `vLLM`: `-37.3%`

#### Decode-heavy workload: `1 in / 1024 out`

- Fresh repo-native direct-state-write engine:
  - `11698.87` output tok/s
  - elapsed: `11.204s`
- Current kept TensorRT baseline:
  - `16468.63` output tok/s
  - elapsed: `7.959s`
- vLLM control:
  - `17625.09` output tok/s
  - elapsed: `7.437s`
- Delta:
  - vs kept TensorRT baseline: `-29.0%`
  - vs `vLLM`: `-33.6%`

#### Prefill-heavy workload: `1024 in / 1 out`

- Fresh repo-native direct-state-write engine:
  - elapsed: `0.411s`
- Current kept TensorRT baseline:
  - elapsed: `0.403s`
- vLLM control:
  - elapsed: `0.477s`
- Delta:
  - vs kept TensorRT baseline latency: `+2.2%`
  - vs `vLLM` latency: `-13.7%`

### Outcome

- This implementation did not outperform `vLLM`.
- It also did not match the current kept TensorRT baseline.
- The direct-state-write conv change is therefore not a candidate replacement for the current kept path in its present repo-native form.

### Interpretation

- The fresh repo-native rebuild behaves much closer to the older pre-penalty-skip / pre-fastpath TensorRT points than to the kept `qwen_update_conv` control.
- Inference:
  - the stronger kept local TensorRT point still depends on native/runtime behavior that is not being reproduced by the current repo-native rebuild path
  - further Qwen-only conv-side optimization is lower priority than recovering the missing decode-side runtime fast path in a clean rebuildable form

### Next Targets

- Reconcile the repo-native decoder runtime against the kept high-throughput local package, with focus on:
  - greedy / no-penalty sampling fast path engagement
  - penalty-path bypass on the batched `logitsVec` native runtime path
  - decode-side GEMM / helper fragmentation after the runtime fast path is restored
