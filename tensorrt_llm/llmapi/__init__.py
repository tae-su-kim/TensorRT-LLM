# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

if os.getenv("TRT_LLM_MINIMAL_IMPORT", "0") == "1":
    __all__ = []
else:
    from .._torch.async_llm import AsyncLLM
    from ..disaggregated_params import DisaggregatedParams, DisaggScheduleStyle
    from ..executor import CompletionOutput, LoRARequest, RequestError
    from ..sampling_params import GuidedDecodingParams, SamplingParams
    from .build_cache import BuildCacheConfig
    from .llm import LLM, RequestOutput
    # yapf: disable
    from .llm_args import (AttentionDpConfig, AutoDecodingConfig, BatchingType,
                           CacheTransceiverConfig, CalibConfig,
                           CapacitySchedulerPolicy, ContextChunkingPolicy,
                           CudaGraphConfig, DeepSeekSparseAttentionConfig,
                           DraftTargetDecodingConfig, DynamicBatchConfig,
                           Eagle3DecodingConfig, EagleDecodingConfig,
                           ExtendedRuntimePerfKnobConfig, KvCacheConfig, LlmArgs,
                           LookaheadDecodingConfig, MedusaDecodingConfig, MoeConfig,
                           MTPDecodingConfig, NGramDecodingConfig,
                           PARDDecodingConfig, PrometheusMetricsConfig,
                           RocketSparseAttentionConfig, SADecodingConfig,
                           SAEnhancerConfig, SaveHiddenStatesDecodingConfig,
                           SchedulerConfig, SkipSoftmaxAttentionConfig,
                           TorchCompileConfig, TorchLlmArgs, TrtLlmArgs,
                           UserProvidedDecodingConfig)
    from .llm_utils import (BuildConfig, KvCacheRetentionConfig, QuantAlgo,
                            QuantConfig)
    from .mm_encoder import MultimodalEncoder
    from .mpi_session import MpiCommSession

    __all__ = [
        'LLM',
        'AsyncLLM',
        'MultimodalEncoder',
        'CompletionOutput',
        'RequestOutput',
        'GuidedDecodingParams',
        'SamplingParams',
        'DisaggregatedParams',
        'DisaggScheduleStyle',
        'KvCacheConfig',
        'KvCacheRetentionConfig',
        'CudaGraphConfig',
        'MoeConfig',
        'LookaheadDecodingConfig',
        'MedusaDecodingConfig',
        'EagleDecodingConfig',
        'Eagle3DecodingConfig',
        'MTPDecodingConfig',
        'SchedulerConfig',
        'CapacitySchedulerPolicy',
        'BuildConfig',
        'QuantConfig',
        'QuantAlgo',
        'CalibConfig',
        'BuildCacheConfig',
        'RequestError',
        'MpiCommSession',
        'ExtendedRuntimePerfKnobConfig',
        'BatchingType',
        'ContextChunkingPolicy',
        'DynamicBatchConfig',
        'CacheTransceiverConfig',
        'NGramDecodingConfig',
        'PARDDecodingConfig',
        'SADecodingConfig',
        'SAEnhancerConfig',
        'UserProvidedDecodingConfig',
        'TorchCompileConfig',
        'DraftTargetDecodingConfig',
        'LlmArgs',
        'TorchLlmArgs',
        'TrtLlmArgs',
        'AutoDecodingConfig',
        'AttentionDpConfig',
        'LoRARequest',
        'SaveHiddenStatesDecodingConfig',
        'RocketSparseAttentionConfig',
        'DeepSeekSparseAttentionConfig',
        'SkipSoftmaxAttentionConfig',
        'PrometheusMetricsConfig',
    ]
