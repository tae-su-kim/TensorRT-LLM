/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "flashInferGatedDeltaPrefill.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

#ifndef PLACEHOLDER_KERNELS
#include "flashinfer/flat/prefill/prefill_kernel_delta_rule_sm90.cuh"
#endif

#include <algorithm>
#include <cstddef>
#include <exception>
#include <type_traits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

constexpr size_t kWorkspaceAlignment = 256;
constexpr size_t kWorkspaceBytesPerSm = 128;
constexpr int kMaxSupportedSmCount = 256;

size_t alignSize(size_t size)
{
    return ((size + kWorkspaceAlignment - 1) / kWorkspaceAlignment) * kWorkspaceAlignment;
}

template <typename ValueType>
constexpr bool isFlashInferSupportedValueType()
{
    return std::is_same_v<ValueType, half>;
}

#ifdef ENABLE_BF16
template <>
constexpr bool isFlashInferSupportedValueType<__nv_bfloat16>()
{
    return true;
}
#endif

int getSmCount()
{
    int deviceId = 0;
    tensorrt_llm::common::check(cudaGetDevice(&deviceId), "cudaGetDevice(&deviceId)", __FILE__, __LINE__);

    int smCount = 0;
    tensorrt_llm::common::check(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, deviceId),
        "cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, deviceId)", __FILE__, __LINE__);
    return smCount;
}

int getDeviceMajor()
{
    int deviceId = 0;
    tensorrt_llm::common::check(cudaGetDevice(&deviceId), "cudaGetDevice(&deviceId)", __FILE__, __LINE__);

    int deviceMajor = 0;
    tensorrt_llm::common::check(cudaDeviceGetAttribute(&deviceMajor, cudaDevAttrComputeCapabilityMajor, deviceId),
        "cudaDeviceGetAttribute(&deviceMajor, cudaDevAttrComputeCapabilityMajor, deviceId)", __FILE__, __LINE__);
    return deviceMajor;
}

} // namespace

size_t getFlashInferGatedDeltaPrefillWorkspaceSize(int batchSize)
{
    auto const cuSeqlensBytes = alignSize(static_cast<size_t>(batchSize + 1) * sizeof(int64_t));
    return cuSeqlensBytes + kWorkspaceBytesPerSm * kMaxSupportedSmCount;
}

template <typename ValueType, typename StateType>
bool tryInvokeFlashInferGatedDeltaPrefill(
    GatedDeltaParamsBase const& params, int64_t const* cuSeqlensDevice, void* workspaceBuffer, cudaStream_t stream)
{
#ifdef PLACEHOLDER_KERNELS
    static_cast<void>(params);
    static_cast<void>(cuSeqlensDevice);
    static_cast<void>(workspaceBuffer);
    static_cast<void>(stream);
    return false;
#else
    if constexpr (!isFlashInferSupportedValueType<ValueType>())
    {
        return false;
    }
    else
    {
        if (params.keyDim != params.valueDim)
        {
            return false;
        }

        if (getDeviceMajor() != 9)
        {
            return false;
        }

        try
        {
            flat::launch_delta_rule_prefill_kernel_gbai<false, true, true, true, cutlass::arch::Sm90, ValueType,
                ValueType, StateType>(stream, static_cast<ValueType*>(params.outputPtr),
                static_cast<StateType*>(params.stateOutPtr), static_cast<ValueType const*>(params.queryPtr),
                static_cast<ValueType const*>(params.keyPtr), static_cast<ValueType const*>(params.valuePtr),
                static_cast<StateType const*>(params.stateInPtr), static_cast<float const*>(params.gPtr),
                static_cast<float const*>(params.betaPtr), cuSeqlensDevice, static_cast<uint8_t*>(workspaceBuffer),
                params.batch, params.numHeads, params.numHeads, params.numHeads, params.numHeads, params.keyDim,
                static_cast<int64_t>(params.batch) * params.maxSeqLen, 1.0F, getSmCount());
        }
        catch (std::exception const&)
        {
            return false;
        }

        return true;
    }
#endif
}

template bool tryInvokeFlashInferGatedDeltaPrefill<half, float>(
    GatedDeltaParamsBase const& params, int64_t const* cuSeqlensDevice, void* workspaceBuffer, cudaStream_t stream);
template bool tryInvokeFlashInferGatedDeltaPrefill<half, half>(
    GatedDeltaParamsBase const& params, int64_t const* cuSeqlensDevice, void* workspaceBuffer, cudaStream_t stream);

#ifdef ENABLE_BF16
template bool tryInvokeFlashInferGatedDeltaPrefill<half, __nv_bfloat16>(
    GatedDeltaParamsBase const& params, int64_t const* cuSeqlensDevice, void* workspaceBuffer, cudaStream_t stream);
template bool tryInvokeFlashInferGatedDeltaPrefill<__nv_bfloat16, float>(
    GatedDeltaParamsBase const& params, int64_t const* cuSeqlensDevice, void* workspaceBuffer, cudaStream_t stream);
template bool tryInvokeFlashInferGatedDeltaPrefill<__nv_bfloat16, half>(
    GatedDeltaParamsBase const& params, int64_t const* cuSeqlensDevice, void* workspaceBuffer, cudaStream_t stream);
template bool tryInvokeFlashInferGatedDeltaPrefill<__nv_bfloat16, __nv_bfloat16>(
    GatedDeltaParamsBase const& params, int64_t const* cuSeqlensDevice, void* workspaceBuffer, cudaStream_t stream);
#endif

template bool tryInvokeFlashInferGatedDeltaPrefill<float, float>(
    GatedDeltaParamsBase const& params, int64_t const* cuSeqlensDevice, void* workspaceBuffer, cudaStream_t stream);
template bool tryInvokeFlashInferGatedDeltaPrefill<float, half>(
    GatedDeltaParamsBase const& params, int64_t const* cuSeqlensDevice, void* workspaceBuffer, cudaStream_t stream);
#ifdef ENABLE_BF16
template bool tryInvokeFlashInferGatedDeltaPrefill<float, __nv_bfloat16>(
    GatedDeltaParamsBase const& params, int64_t const* cuSeqlensDevice, void* workspaceBuffer, cudaStream_t stream);
#endif

} // namespace kernels

TRTLLM_NAMESPACE_END
