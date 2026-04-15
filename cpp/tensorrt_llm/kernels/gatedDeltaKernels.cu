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

#include "gatedDeltaKernels.h"
#include "flashInferGatedDeltaPrefill/flashInferGatedDeltaPrefill.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

constexpr float kSoftplusBeta = 1.0F;
constexpr float kSoftplusThreshold = 20.0F;
constexpr float kL2NormEps = 1.0e-6F;

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

__device__ float softplus(float value);
__device__ float sigmoid(float value);
__device__ float warpReduceSum(float value);
__device__ float blockReduceSum(float value);
template <typename ValueType>
__device__ void loadAndNormalizeQueryKey(
    GatedDeltaParamsBase const& params, ValueType const* queryBase, ValueType const* keyBase, int64_t queryOffset,
    int64_t keyOffset, float* sharedQuery, float* sharedKey);

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

template <typename ValueType>
__device__ void loadAndNormalizeQueryKey(
    GatedDeltaParamsBase const& params, ValueType const* queryBase, ValueType const* keyBase, int64_t queryOffset,
    int64_t keyOffset, float* sharedQuery, float* sharedKey)
{
    float querySqNorm = 0.0F;
    float keySqNorm = 0.0F;
    for (int keyIdx = threadIdx.x; keyIdx < params.keyDim; keyIdx += blockDim.x)
    {
        auto const rawQuery = valueToFloat(queryBase[queryOffset + keyIdx]);
        auto const rawKey = valueToFloat(keyBase[keyOffset + keyIdx]);
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
}

template <typename ValueType>
__global__ void gatedDeltaPreprocessKernel(
    GatedDeltaParamsBase params, ValueType* queryBuffer, ValueType* keyBuffer, float* gBuffer, float* betaBuffer)
{
    auto const vectorIdx = static_cast<int64_t>(blockIdx.x);
    auto const totalVectors
        = static_cast<int64_t>(params.batch) * params.maxSeqLen * params.numHeads;
    if (vectorIdx >= totalVectors)
    {
        return;
    }

    extern __shared__ float sharedQk[];
    auto* sharedQuery = sharedQk;
    auto* sharedKey = sharedQk + params.keyDim;

    auto const head = vectorIdx % params.numHeads;
    auto const* queryBase = static_cast<ValueType const*>(params.queryPtr);
    auto const* keyBase = static_cast<ValueType const*>(params.keyPtr);
    auto const* aLogBase = static_cast<float const*>(params.aLogPtr);
    auto const* aBase = static_cast<float const*>(params.aPtr);
    auto const* dtBiasBase = static_cast<float const*>(params.dtBiasPtr);
    auto const* bBase = static_cast<float const*>(params.bPtr);

    auto const qkTokenOffset = vectorIdx * params.keyDim;
    loadAndNormalizeQueryKey(params, queryBase, keyBase, qkTokenOffset, qkTokenOffset, sharedQuery, sharedKey);
    for (int keyIdx = threadIdx.x; keyIdx < params.keyDim; keyIdx += blockDim.x)
    {
        storeValue(queryBuffer + qkTokenOffset + keyIdx, sharedQuery[keyIdx]);
        storeValue(keyBuffer + qkTokenOffset + keyIdx, sharedKey[keyIdx]);
    }

    auto const aLog = aLogBase[head];
    auto const a = aBase[vectorIdx];
    auto const dtBias = dtBiasBase[head];
    auto const b = bBase[vectorIdx];
    if (threadIdx.x == 0)
    {
        gBuffer[vectorIdx] = -expf(aLog) * softplus(a + dtBias);
        betaBuffer[vectorIdx] = sigmoid(b);
    }
}

template <typename ValueType, typename StateType>
__global__ void gatedDeltaKernel(GatedDeltaParamsBase params, int const* sampleLengths)
{
    auto const sampleHeadIdx = blockIdx.x;
    auto const sample = sampleHeadIdx / params.numHeads;
    auto const head = sampleHeadIdx % params.numHeads;
    if (sample >= params.batch)
    {
        return;
    }

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

    auto const stateStridePerHead = params.keyDim * params.valueDim;
    auto const stateOffset = (sample * params.numHeads + head) * stateStridePerHead;
    auto const* stateIn = stateInBase + stateOffset;
    auto* stateOut = stateOutBase + stateOffset;

    auto const tokensPerHeadQk = params.numHeads * params.keyDim;
    auto const tokensPerHeadValue = params.numHeads * params.valueDim;
    auto const tokensPerHeadScalar = params.numHeads;
    auto const sampleTokenOffsetQk = sample * params.maxSeqLen * tokensPerHeadQk;
    auto const sampleTokenOffsetValue = sample * params.maxSeqLen * tokensPerHeadValue;
    auto const sampleTokenOffsetScalar = sample * params.maxSeqLen * tokensPerHeadScalar;

    auto numTokens = sampleLengths[sample];
    if (numTokens < 0)
    {
        numTokens = 0;
    }
    if (numTokens > params.maxSeqLen)
    {
        numTokens = params.maxSeqLen;
    }

    for (int tokenIdx = 0; tokenIdx < params.maxSeqLen; ++tokenIdx)
    {
        auto const qkTokenOffset = sampleTokenOffsetQk + tokenIdx * tokensPerHeadQk + head * params.keyDim;
        loadAndNormalizeQueryKey(
            params, queryBase, keyBase, qkTokenOffset, qkTokenOffset, sharedQuery, sharedKey);

        auto* output = outputBase + sampleTokenOffsetValue + tokenIdx * tokensPerHeadValue + head * params.valueDim;
        if (tokenIdx >= numTokens)
        {
            for (int valueIdx = threadIdx.x; valueIdx < params.valueDim; valueIdx += blockDim.x)
            {
                storeValue(output + valueIdx, 0.0F);
            }
            __syncthreads();
            continue;
        }

        auto const scalarTokenOffset = sampleTokenOffsetScalar + tokenIdx * tokensPerHeadScalar + head;
        auto const aLog = aLogBase[head];
        auto const a = aBase[scalarTokenOffset];
        auto const dtBias = dtBiasBase[head];
        auto const b = bBase[scalarTokenOffset];
        auto const g = -expf(aLog) * softplus(a + dtBias);
        auto const decay = expf(g);
        auto const beta = sigmoid(b);
        auto const* value = valueBase + sampleTokenOffsetValue + tokenIdx * tokensPerHeadValue + head * params.valueDim;

        for (int valueIdx = threadIdx.x; valueIdx < params.valueDim; valueIdx += blockDim.x)
        {
            float dot = 0.0F;
            auto const* previousState = tokenIdx == 0 ? stateIn : stateOut;
            for (int keyIdx = 0; keyIdx < params.keyDim; ++keyIdx)
            {
                auto const stateIndex = valueIdx * params.keyDim + keyIdx;
                auto const decayedState = valueToFloat(previousState[stateIndex]) * decay;
                storeValue(stateOut + stateIndex, decayedState);
                dot += sharedKey[keyIdx] * decayedState;
            }

            auto const valuePrime = (valueToFloat(value[valueIdx]) - dot) * beta;

            float outputValue = 0.0F;
            for (int keyIdx = 0; keyIdx < params.keyDim; ++keyIdx)
            {
                auto const stateIndex = valueIdx * params.keyDim + keyIdx;
                auto const updatedState = valueToFloat(stateOut[stateIndex]) + sharedKey[keyIdx] * valuePrime;
                storeValue(stateOut + stateIndex, updatedState);
                outputValue += sharedQuery[keyIdx] * updatedState;
            }
            storeValue(output + valueIdx, outputValue);
        }
        __syncthreads();
    }
}

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

template <typename ValueType, typename StateType>
__global__ void gatedDeltaDecodeKernel(GatedDeltaParamsBase params)
{
    auto const sampleHeadIdx = blockIdx.x;
    auto const sample = sampleHeadIdx / params.numHeads;
    auto const head = sampleHeadIdx % params.numHeads;
    if (sample >= params.batch)
    {
        return;
    }

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

    auto const tokensPerHeadQk = params.numHeads * params.keyDim;
    auto const tokensPerHeadValue = params.numHeads * params.valueDim;
    auto const sampleTokenOffsetQk = sample * params.maxSeqLen * tokensPerHeadQk + head * params.keyDim;
    auto const sampleTokenOffsetValue = sample * params.maxSeqLen * tokensPerHeadValue + head * params.valueDim;
    auto const sampleTokenOffsetScalar = sample * params.maxSeqLen * params.numHeads + head;

    loadAndNormalizeQueryKey(
        params, queryBase, keyBase, sampleTokenOffsetQk, sampleTokenOffsetQk, sharedQuery, sharedKey);

    auto const aLog = aLogBase[head];
    auto const a = aBase[sampleTokenOffsetScalar];
    auto const dtBias = dtBiasBase[head];
    auto const b = bBase[sampleTokenOffsetScalar];
    auto const g = -expf(aLog) * softplus(a + dtBias);
    auto const decay = expf(g);
    auto const beta = sigmoid(b);

    auto const stateStridePerHead = params.keyDim * params.valueDim;
    auto const stateOffset = (sample * params.numHeads + head) * stateStridePerHead;
    auto const* stateIn = stateInBase + stateOffset;
    auto* stateOut = stateOutBase + stateOffset;
    auto const* value = valueBase + sampleTokenOffsetValue;
    auto* output = outputBase + sampleTokenOffsetValue;

    for (int valueIdx = threadIdx.x; valueIdx < params.valueDim; valueIdx += blockDim.x)
    {
        float dot = 0.0F;
        for (int keyIdx = 0; keyIdx < params.keyDim; ++keyIdx)
        {
            auto const stateIndex = valueIdx * params.keyDim + keyIdx;
            dot += sharedKey[keyIdx] * valueToFloat(stateIn[stateIndex]) * decay;
        }

        auto const valuePrime = (valueToFloat(value[valueIdx]) - dot) * beta;

        float outputValue = 0.0F;
        for (int keyIdx = 0; keyIdx < params.keyDim; ++keyIdx)
        {
            auto const stateIndex = valueIdx * params.keyDim + keyIdx;
            auto const updatedState = valueToFloat(stateIn[stateIndex]) * decay + sharedKey[keyIdx] * valuePrime;
            storeValue(stateOut + stateIndex, updatedState);
            outputValue += sharedQuery[keyIdx] * updatedState;
        }
        storeValue(output + valueIdx, outputValue);
    }
}

template <typename ValueType>
int selectBlockSize(int valueDim)
{
    if (valueDim <= 32)
    {
        return 32;
    }
    if (valueDim <= 64)
    {
        return 64;
    }
    if (valueDim <= 128)
    {
        return 128;
    }
    return 256;
}

template <typename ValueType, typename StateType>
void launchGatedDeltaKernel(GatedDeltaParamsBase const& params, int const* sampleLengths, cudaStream_t stream)
{
    auto const blockSize = selectBlockSize<ValueType>(params.valueDim);
    auto const gridSize = params.batch * params.numHeads;
    auto const sharedMemSize = static_cast<std::size_t>(2 * params.keyDim) * sizeof(float);
    gatedDeltaKernel<ValueType, StateType><<<gridSize, blockSize, sharedMemSize, stream>>>(params, sampleLengths);
}

template <typename ValueType, typename StateType>
void launchGatedDeltaDecodeKernel(GatedDeltaParamsBase const& params, cudaStream_t stream)
{
    auto const blockSize = selectBlockSize<ValueType>(params.valueDim);
    auto const gridSize = params.batch * params.numHeads;
    auto const sharedMemSize = static_cast<std::size_t>(2 * params.keyDim) * sizeof(float);
    gatedDeltaDecodeKernel<ValueType, StateType><<<gridSize, blockSize, sharedMemSize, stream>>>(params);
}

} // namespace

template <typename ValueType, typename StateType>
void invokeGatedDelta(GatedDeltaParamsBase const& params, int const* sampleLengths, cudaStream_t stream)
{
    launchGatedDeltaKernel<ValueType, StateType>(params, sampleLengths, stream);
}

template <typename ValueType, typename StateType>
void invokeGatedDeltaDecode(GatedDeltaParamsBase const& params, cudaStream_t stream)
{
    launchGatedDeltaDecodeKernel<ValueType, StateType>(params, stream);
}

template <typename ValueType>
void invokeGatedDeltaPreprocess(GatedDeltaParamsBase const& params, ValueType* queryBuffer, ValueType* keyBuffer,
    float* gBuffer, float* betaBuffer, cudaStream_t stream)
{
    auto const totalVectors = static_cast<int64_t>(params.batch) * params.maxSeqLen * params.numHeads;
    if (totalVectors == 0)
    {
        return;
    }

    auto constexpr blockSize = 128;
    auto const sharedMemSize = static_cast<std::size_t>(2 * params.keyDim) * sizeof(float);
    gatedDeltaPreprocessKernel<ValueType>
        <<<static_cast<int>(totalVectors), blockSize, sharedMemSize, stream>>>(
            params, queryBuffer, keyBuffer, gBuffer, betaBuffer);
}

template void invokeGatedDelta<float, float>(
    GatedDeltaParamsBase const& params, int const* sampleLengths, cudaStream_t stream);
template void invokeGatedDelta<float, half>(
    GatedDeltaParamsBase const& params, int const* sampleLengths, cudaStream_t stream);
template void invokeGatedDelta<half, float>(
    GatedDeltaParamsBase const& params, int const* sampleLengths, cudaStream_t stream);
template void invokeGatedDelta<half, half>(
    GatedDeltaParamsBase const& params, int const* sampleLengths, cudaStream_t stream);
template void invokeGatedDeltaDecode<float, float>(GatedDeltaParamsBase const& params, cudaStream_t stream);
template void invokeGatedDeltaDecode<float, half>(GatedDeltaParamsBase const& params, cudaStream_t stream);
template void invokeGatedDeltaDecode<half, float>(GatedDeltaParamsBase const& params, cudaStream_t stream);
template void invokeGatedDeltaDecode<half, half>(GatedDeltaParamsBase const& params, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeGatedDelta<float, __nv_bfloat16>(
    GatedDeltaParamsBase const& params, int const* sampleLengths, cudaStream_t stream);
template void invokeGatedDelta<half, __nv_bfloat16>(
    GatedDeltaParamsBase const& params, int const* sampleLengths, cudaStream_t stream);
template void invokeGatedDelta<__nv_bfloat16, float>(
    GatedDeltaParamsBase const& params, int const* sampleLengths, cudaStream_t stream);
template void invokeGatedDelta<__nv_bfloat16, half>(
    GatedDeltaParamsBase const& params, int const* sampleLengths, cudaStream_t stream);
template void invokeGatedDelta<__nv_bfloat16, __nv_bfloat16>(
    GatedDeltaParamsBase const& params, int const* sampleLengths, cudaStream_t stream);
template void invokeGatedDeltaDecode<float, __nv_bfloat16>(
    GatedDeltaParamsBase const& params, cudaStream_t stream);
template void invokeGatedDeltaDecode<half, __nv_bfloat16>(GatedDeltaParamsBase const& params, cudaStream_t stream);
template void invokeGatedDeltaDecode<__nv_bfloat16, float>(
    GatedDeltaParamsBase const& params, cudaStream_t stream);
template void invokeGatedDeltaDecode<__nv_bfloat16, half>(GatedDeltaParamsBase const& params, cudaStream_t stream);
template void invokeGatedDeltaDecode<__nv_bfloat16, __nv_bfloat16>(
    GatedDeltaParamsBase const& params, cudaStream_t stream);
#endif
template void invokeGatedDeltaPreprocess<float>(GatedDeltaParamsBase const& params, float* queryBuffer,
    float* keyBuffer, float* gBuffer, float* betaBuffer, cudaStream_t stream);
template void invokeGatedDeltaPreprocess<half>(GatedDeltaParamsBase const& params, half* queryBuffer,
    half* keyBuffer, float* gBuffer, float* betaBuffer, cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeGatedDeltaPreprocess<__nv_bfloat16>(GatedDeltaParamsBase const& params,
    __nv_bfloat16* queryBuffer, __nv_bfloat16* keyBuffer, float* gBuffer, float* betaBuffer, cudaStream_t stream);
#endif

} // namespace kernels

TRTLLM_NAMESPACE_END
