/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

struct GatedDeltaParamsBase
{
    int batch;
    int maxSeqLen;
    int numHeads;
    int keyDim;
    int valueDim;

    void const* __restrict__ queryPtr;
    void const* __restrict__ keyPtr;
    void const* __restrict__ valuePtr;
    void const* __restrict__ gPtr;
    void const* __restrict__ betaPtr;
    void const* __restrict__ aLogPtr;
    void const* __restrict__ aPtr;
    void const* __restrict__ dtBiasPtr;
    void const* __restrict__ bPtr;
    void const* __restrict__ stateInPtr;
    void* __restrict__ stateOutPtr;
    void* __restrict__ outputPtr;
};

template <typename ValueType, typename StateType>
void invokeGatedDelta(GatedDeltaParamsBase const& params, int const* sampleLengths, cudaStream_t stream);

template <typename ValueType, typename StateType>
void invokeGatedDeltaDecode(GatedDeltaParamsBase const& params, cudaStream_t stream);

template <typename ValueType>
void invokeGatedDeltaPreprocess(GatedDeltaParamsBase const& params, ValueType* queryBuffer, ValueType* keyBuffer,
    float* gBuffer, float* betaBuffer, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
