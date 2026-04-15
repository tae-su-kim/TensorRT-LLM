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

#include "flashInferGatedDeltaDecode.h"

#include "tensorrt_llm/common/cudaUtils.h"

#include <algorithm>
#include <type_traits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

// Adapted from the FlashInfer decode implementation in:
// flashinfer/gdn_decode.py::gated_delta_rule_decode_pretranspose.
constexpr int kDecodeThreadsPerBlock = 128;
constexpr int kDecodeWarpsPerBlock = kDecodeThreadsPerBlock / 32;
constexpr int kFlashInferDecodeKeyDim = 128;
constexpr int kFlashInferDecodeRowsPerBlock = kDecodeWarpsPerBlock;
constexpr int kQwenDecodeRowsPerWarp = 4;
constexpr int kQwenDecodeRowsPerBlock = kDecodeWarpsPerBlock * kQwenDecodeRowsPerWarp;
constexpr int kQwenDecodeKeyChunksPerLane = kFlashInferDecodeKeyDim / 32;
constexpr float kSoftplusBeta = 1.0F;
constexpr float kSoftplusThreshold = 20.0F;
constexpr float kL2NormEps = 1.0e-6F;

int getDeviceMajor()
{
    int deviceId = 0;
    tensorrt_llm::common::check(cudaGetDevice(&deviceId), "cudaGetDevice(&deviceId)", __FILE__, __LINE__);

    int deviceMajor = 0;
    tensorrt_llm::common::check(cudaDeviceGetAttribute(&deviceMajor, cudaDevAttrComputeCapabilityMajor, deviceId),
        "cudaDeviceGetAttribute(&deviceMajor, cudaDevAttrComputeCapabilityMajor, deviceId)", __FILE__, __LINE__);
    return deviceMajor;
}

template <typename T>
__device__ float valueToFloat(T value);

template <>
__device__ float valueToFloat(float value)
{
    return value;
}

template <>
__device__ float valueToFloat(half value)
{
    return __half2float(value);
}

#ifdef ENABLE_BF16
template <>
__device__ float valueToFloat(__nv_bfloat16 value)
{
    return __bfloat162float(value);
}
#endif

template <typename T>
__device__ void storeValue(T* output, float value);

template <>
__device__ void storeValue(float* output, float value)
{
    *output = value;
}

template <>
__device__ void storeValue(half* output, float value)
{
    *output = __float2half(value);
}

#ifdef ENABLE_BF16
template <>
__device__ void storeValue(__nv_bfloat16* output, float value)
{
    *output = __float2bfloat16(value);
}
#endif

__device__ float softplus(float value)
{
    auto const betaX = kSoftplusBeta * value;
    if (betaX <= kSoftplusThreshold)
    {
        return log1pf(expf(betaX)) / kSoftplusBeta;
    }
    return value;
}

__device__ float sigmoid(float value)
{
    return 1.0F / (1.0F + expf(-value));
}

__device__ float warpReduceSum(float value)
{
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        value += __shfl_xor_sync(0xFFFFFFFF, value, offset);
    }
    return value;
}

__device__ float blockReduceSum(float value)
{
    __shared__ float partialSums[32];
    auto const lane = threadIdx.x % 32;
    auto const warp = threadIdx.x / 32;
    auto const numWarps = (blockDim.x + 31) / 32;

    value = warpReduceSum(value);
    if (lane == 0)
    {
        partialSums[warp] = value;
    }
    __syncthreads();

    if (warp == 0)
    {
        value = lane < numWarps ? partialSums[lane] : 0.0F;
        value = warpReduceSum(value);
        if (lane == 0)
        {
            partialSums[0] = value;
        }
    }
    __syncthreads();
    return partialSums[0];
}

template <typename ValueType, typename StateType>
__global__ void flashInferGatedDeltaDecodeKernel(GatedDeltaParamsBase params)
{
    auto const valuesPerTile = kFlashInferDecodeRowsPerBlock;
    auto const tilesPerHead = (params.valueDim + valuesPerTile - 1) / valuesPerTile;
    auto const sampleHeadTileIdx = blockIdx.x;
    auto const sampleHeadIdx = sampleHeadTileIdx / tilesPerHead;
    auto const tileIdx = sampleHeadTileIdx % tilesPerHead;
    auto const sample = sampleHeadIdx / params.numHeads;
    auto const head = sampleHeadIdx % params.numHeads;

    if (sample >= params.batch)
    {
        return;
    }

    auto const lane = threadIdx.x % 32;
    auto const warp = threadIdx.x / 32;
    auto const valueIdx = tileIdx * valuesPerTile + warp;

    extern __shared__ float sharedQk[];
    auto* sharedQuery = sharedQk;
    auto* sharedKey = sharedQk + params.keyDim;

    auto const* queryBase = static_cast<ValueType const*>(params.queryPtr);
    auto const* keyBase = static_cast<ValueType const*>(params.keyPtr);
    auto const* valueBase = static_cast<ValueType const*>(params.valuePtr);
    auto const* aLogBase = static_cast<float const*>(params.aLogPtr);
    auto const* aBase = static_cast<float const*>(params.aPtr);
    auto const* dtBiasBase = static_cast<float const*>(params.dtBiasPtr);
    auto const* bBase = static_cast<float const*>(params.bPtr);
    auto const* stateInBase = static_cast<StateType const*>(params.stateInPtr);
    auto* stateOutBase = static_cast<StateType*>(params.stateOutPtr);
    auto* outputBase = static_cast<ValueType*>(params.outputPtr);

    auto const qkStride = params.numHeads * params.keyDim;
    auto const valueStride = params.numHeads * params.valueDim;
    auto const qkOffset = sample * params.maxSeqLen * qkStride + head * params.keyDim;
    auto const valueOffset = sample * params.maxSeqLen * valueStride + head * params.valueDim;
    auto const scalarOffset = sample * params.maxSeqLen * params.numHeads + head;

    float querySqNorm = 0.0F;
    float keySqNorm = 0.0F;
    for (int keyIdx = threadIdx.x; keyIdx < params.keyDim; keyIdx += blockDim.x)
    {
        auto const rawQuery = valueToFloat(queryBase[qkOffset + keyIdx]);
        auto const rawKey = valueToFloat(keyBase[qkOffset + keyIdx]);
        sharedQuery[keyIdx] = rawQuery;
        sharedKey[keyIdx] = rawKey;
        querySqNorm += rawQuery * rawQuery;
        keySqNorm += rawKey * rawKey;
    }
    __syncthreads();

    auto const queryInvNorm = rsqrtf(blockReduceSum(querySqNorm) + kL2NormEps) * rsqrtf(static_cast<float>(params.keyDim));
    auto const keyInvNorm = rsqrtf(blockReduceSum(keySqNorm) + kL2NormEps);
    for (int keyIdx = threadIdx.x; keyIdx < params.keyDim; keyIdx += blockDim.x)
    {
        sharedQuery[keyIdx] *= queryInvNorm;
        sharedKey[keyIdx] *= keyInvNorm;
    }
    __syncthreads();

    if (valueIdx >= params.valueDim)
    {
        return;
    }

    auto const aLog = aLogBase[head];
    auto const a = aBase[scalarOffset];
    auto const dtBias = dtBiasBase[head];
    auto const b = bBase[scalarOffset];
    auto const decay = expf(-expf(aLog) * softplus(a + dtBias));
    auto const beta = sigmoid(b);

    auto const stateBaseOffset = ((sample * params.numHeads + head) * params.valueDim + valueIdx) * params.keyDim;
    auto const* stateInRow = stateInBase + stateBaseOffset;
    auto* stateOutRow = stateOutBase + stateBaseOffset;
    auto const* value = valueBase + valueOffset;
    auto* output = outputBase + valueOffset;

    float decayedState[4];
    float pred = 0.0F;
    for (int keyIdx = lane, slot = 0; keyIdx < params.keyDim; keyIdx += 32, ++slot)
    {
        auto const stateVal = valueToFloat(stateInRow[keyIdx]) * decay;
        decayedState[slot] = stateVal;
        pred += stateVal * sharedKey[keyIdx];
    }
    pred = warpReduceSum(pred);

    auto const valuePrime = (valueToFloat(value[valueIdx]) - pred) * beta;

    float out = 0.0F;
    for (int keyIdx = lane, slot = 0; keyIdx < params.keyDim; keyIdx += 32, ++slot)
    {
        auto const updatedState = decayedState[slot] + sharedKey[keyIdx] * valuePrime;
        storeValue(stateOutRow + keyIdx, updatedState);
        out += updatedState * sharedQuery[keyIdx];
    }
    out = warpReduceSum(out);

    if (lane == 0)
    {
        storeValue(output + valueIdx, out);
    }
}

template <typename ValueType, typename StateType>
__global__ void qwenGatedDeltaDecodeKernel(GatedDeltaParamsBase params)
{
    auto const tilesPerHead = (params.valueDim + kQwenDecodeRowsPerBlock - 1) / kQwenDecodeRowsPerBlock;
    auto const sampleHeadTileIdx = blockIdx.x;
    auto const sampleHeadIdx = sampleHeadTileIdx / tilesPerHead;
    auto const tileIdx = sampleHeadTileIdx % tilesPerHead;
    auto const sample = sampleHeadIdx / params.numHeads;
    auto const head = sampleHeadIdx % params.numHeads;

    if (sample >= params.batch)
    {
        return;
    }

    auto const lane = threadIdx.x % 32;
    auto const warp = threadIdx.x / 32;
    auto const localValueBase = tileIdx * kQwenDecodeRowsPerBlock + warp * kQwenDecodeRowsPerWarp;

    extern __shared__ float sharedQk[];
    auto* sharedQuery = sharedQk;
    auto* sharedKey = sharedQk + params.keyDim;

    auto const* queryBase = static_cast<ValueType const*>(params.queryPtr);
    auto const* keyBase = static_cast<ValueType const*>(params.keyPtr);
    auto const* valueBase = static_cast<ValueType const*>(params.valuePtr);
    auto const* aLogBase = static_cast<float const*>(params.aLogPtr);
    auto const* aBase = static_cast<float const*>(params.aPtr);
    auto const* dtBiasBase = static_cast<float const*>(params.dtBiasPtr);
    auto const* bBase = static_cast<float const*>(params.bPtr);
    auto const* stateInBase = static_cast<StateType const*>(params.stateInPtr);
    auto* stateOutBase = static_cast<StateType*>(params.stateOutPtr);
    auto* outputBase = static_cast<ValueType*>(params.outputPtr);

    auto const qkStride = params.numHeads * params.keyDim;
    auto const valueStride = params.numHeads * params.valueDim;
    auto const qkOffset = sample * params.maxSeqLen * qkStride + head * params.keyDim;
    auto const valueOffset = sample * params.maxSeqLen * valueStride + head * params.valueDim;
    auto const scalarOffset = sample * params.maxSeqLen * params.numHeads + head;

    float querySqNorm = 0.0F;
    float keySqNorm = 0.0F;
    for (int keyIdx = threadIdx.x; keyIdx < params.keyDim; keyIdx += blockDim.x)
    {
        auto const rawQuery = valueToFloat(queryBase[qkOffset + keyIdx]);
        auto const rawKey = valueToFloat(keyBase[qkOffset + keyIdx]);
        sharedQuery[keyIdx] = rawQuery;
        sharedKey[keyIdx] = rawKey;
        querySqNorm += rawQuery * rawQuery;
        keySqNorm += rawKey * rawKey;
    }
    __syncthreads();

    auto const queryInvNorm = rsqrtf(blockReduceSum(querySqNorm) + kL2NormEps) * rsqrtf(static_cast<float>(params.keyDim));
    auto const keyInvNorm = rsqrtf(blockReduceSum(keySqNorm) + kL2NormEps);
    for (int keyIdx = threadIdx.x; keyIdx < params.keyDim; keyIdx += blockDim.x)
    {
        sharedQuery[keyIdx] *= queryInvNorm;
        sharedKey[keyIdx] *= keyInvNorm;
    }
    __syncthreads();

    auto const aLog = aLogBase[head];
    auto const a = aBase[scalarOffset];
    auto const dtBias = dtBiasBase[head];
    auto const b = bBase[scalarOffset];
    auto const decay = expf(-expf(aLog) * softplus(a + dtBias));
    auto const beta = sigmoid(b);

    auto const stateHeadOffset = (sample * params.numHeads + head) * params.valueDim * params.keyDim;
    auto const* stateInHead = stateInBase + stateHeadOffset;
    auto* stateOutHead = stateOutBase + stateHeadOffset;
    auto const* value = valueBase + valueOffset;
    auto* output = outputBase + valueOffset;

    float decayedState[kQwenDecodeRowsPerWarp][kQwenDecodeKeyChunksPerLane];
    float pred[kQwenDecodeRowsPerWarp]{};
    float valuePrime[kQwenDecodeRowsPerWarp]{};
    float out[kQwenDecodeRowsPerWarp]{};

    for (int row = 0; row < kQwenDecodeRowsPerWarp; ++row)
    {
        auto const valueIdx = localValueBase + row;
        if (valueIdx >= params.valueDim)
        {
            continue;
        }

        auto const* stateInRow = stateInHead + valueIdx * params.keyDim;
        for (int chunk = 0; chunk < kQwenDecodeKeyChunksPerLane; ++chunk)
        {
            auto const keyIdx = lane + chunk * 32;
            auto const stateVal = valueToFloat(stateInRow[keyIdx]) * decay;
            decayedState[row][chunk] = stateVal;
            pred[row] += stateVal * sharedKey[keyIdx];
        }
    }

    for (int row = 0; row < kQwenDecodeRowsPerWarp; ++row)
    {
        pred[row] = warpReduceSum(pred[row]);
        auto const valueIdx = localValueBase + row;
        if (valueIdx < params.valueDim)
        {
            valuePrime[row] = (valueToFloat(value[valueIdx]) - pred[row]) * beta;
        }
    }

    for (int row = 0; row < kQwenDecodeRowsPerWarp; ++row)
    {
        auto const valueIdx = localValueBase + row;
        if (valueIdx >= params.valueDim)
        {
            continue;
        }

        auto* stateOutRow = stateOutHead + valueIdx * params.keyDim;
        for (int chunk = 0; chunk < kQwenDecodeKeyChunksPerLane; ++chunk)
        {
            auto const keyIdx = lane + chunk * 32;
            auto const updatedState = decayedState[row][chunk] + sharedKey[keyIdx] * valuePrime[row];
            storeValue(stateOutRow + keyIdx, updatedState);
            out[row] += updatedState * sharedQuery[keyIdx];
        }
    }

    for (int row = 0; row < kQwenDecodeRowsPerWarp; ++row)
    {
        auto const valueIdx = localValueBase + row;
        if (valueIdx >= params.valueDim)
        {
            continue;
        }

        out[row] = warpReduceSum(out[row]);
        if (lane == 0)
        {
            storeValue(output + valueIdx, out[row]);
        }
    }
}

template <typename ValueType>
constexpr bool isSupportedValueType()
{
    return std::is_same_v<ValueType, float> || std::is_same_v<ValueType, half>;
}

#ifdef ENABLE_BF16
template <>
constexpr bool isSupportedValueType<__nv_bfloat16>()
{
    return true;
}
#endif

template <typename StateType>
constexpr bool isSupportedStateType()
{
    return std::is_same_v<StateType, float> || std::is_same_v<StateType, half>;
}

#ifdef ENABLE_BF16
template <>
constexpr bool isSupportedStateType<__nv_bfloat16>()
{
    return true;
}
#endif

} // namespace

template <typename ValueType, typename StateType>
bool tryInvokeFlashInferGatedDeltaDecode(GatedDeltaParamsBase const& params, cudaStream_t stream)
{
    if constexpr (!isSupportedValueType<ValueType>() || !isSupportedStateType<StateType>())
    {
        return false;
    }

    if (params.maxSeqLen != 1 || params.keyDim != kFlashInferDecodeKeyDim || params.valueDim <= 0
        || getDeviceMajor() != 9)
    {
        return false;
    }

    if (params.valueDim == kFlashInferDecodeKeyDim)
    {
        auto const tilesPerHead = (params.valueDim + kQwenDecodeRowsPerBlock - 1) / kQwenDecodeRowsPerBlock;
        auto const gridSize = params.batch * params.numHeads * tilesPerHead;
        auto const sharedMemSize = static_cast<size_t>(2 * params.keyDim) * sizeof(float);
        qwenGatedDeltaDecodeKernel<ValueType, StateType>
            <<<gridSize, kDecodeThreadsPerBlock, sharedMemSize, stream>>>(params);
        return true;
    }

    auto const tilesPerHead = (params.valueDim + kFlashInferDecodeRowsPerBlock - 1) / kFlashInferDecodeRowsPerBlock;
    auto const gridSize = params.batch * params.numHeads * tilesPerHead;
    auto const sharedMemSize = static_cast<size_t>(2 * params.keyDim) * sizeof(float);
    flashInferGatedDeltaDecodeKernel<ValueType, StateType>
        <<<gridSize, kDecodeThreadsPerBlock, sharedMemSize, stream>>>(params);
    return true;
}

template bool tryInvokeFlashInferGatedDeltaDecode<float, float>(
    GatedDeltaParamsBase const& params, cudaStream_t stream);
template bool tryInvokeFlashInferGatedDeltaDecode<float, half>(
    GatedDeltaParamsBase const& params, cudaStream_t stream);
template bool tryInvokeFlashInferGatedDeltaDecode<half, float>(
    GatedDeltaParamsBase const& params, cudaStream_t stream);
template bool tryInvokeFlashInferGatedDeltaDecode<half, half>(
    GatedDeltaParamsBase const& params, cudaStream_t stream);
#ifdef ENABLE_BF16
template bool tryInvokeFlashInferGatedDeltaDecode<float, __nv_bfloat16>(
    GatedDeltaParamsBase const& params, cudaStream_t stream);
template bool tryInvokeFlashInferGatedDeltaDecode<half, __nv_bfloat16>(
    GatedDeltaParamsBase const& params, cudaStream_t stream);
template bool tryInvokeFlashInferGatedDeltaDecode<__nv_bfloat16, float>(
    GatedDeltaParamsBase const& params, cudaStream_t stream);
template bool tryInvokeFlashInferGatedDeltaDecode<__nv_bfloat16, half>(
    GatedDeltaParamsBase const& params, cudaStream_t stream);
template bool tryInvokeFlashInferGatedDeltaDecode<__nv_bfloat16, __nv_bfloat16>(
    GatedDeltaParamsBase const& params, cudaStream_t stream);
#endif

} // namespace kernels

TRTLLM_NAMESPACE_END
