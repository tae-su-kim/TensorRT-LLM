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

#include "gatedDeltaPlugin.h"

#include "tensorrt_llm/kernels/flashInferGatedDeltaDecode/flashInferGatedDeltaDecode.h"
#include "tensorrt_llm/kernels/flashInferGatedDeltaPrefill/flashInferGatedDeltaPrefill.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/runtime/common.h"

#include <algorithm>
#include <vector>

using namespace nvinfer1;
using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;
using tensorrt_llm::plugins::GatedDeltaPlugin;
using tensorrt_llm::plugins::GatedDeltaPluginCreator;

namespace
{

char const* kGatedDeltaPluginVersion{"2"};
char const* kGatedDeltaPluginName{"GatedDelta"};
constexpr size_t kWorkspaceAlignment = 256;

size_t alignSize(size_t size)
{
    return ((size + kWorkspaceAlignment - 1) / kWorkspaceAlignment) * kWorkspaceAlignment;
}

bool isValueTypeSupported(nvinfer1::DataType type)
{
    return type == DataType::kFLOAT || type == DataType::kHALF || type == DataType::kBF16;
}

bool isStateTypeSupported(nvinfer1::DataType type)
{
    return type == DataType::kFLOAT || type == DataType::kHALF || type == DataType::kBF16;
}

size_t getTypeSize(nvinfer1::DataType type)
{
    switch (type)
    {
    case DataType::kFLOAT: return sizeof(float);
    case DataType::kHALF: return sizeof(half);
#ifdef ENABLE_BF16
    case DataType::kBF16: return sizeof(__nv_bfloat16);
#endif
    default: TLLM_THROW("Unsupported value dtype for GatedDelta workspace.");
    }
}

} // namespace

PluginFieldCollection GatedDeltaPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> GatedDeltaPluginCreator::mPluginAttributes;

GatedDeltaPlugin::GatedDeltaPlugin(void const* data, size_t length)
{
    TLLM_CHECK(length == 0);
}

nvinfer1::IPluginV2DynamicExt* GatedDeltaPlugin::clone() const noexcept
{
    auto* plugin = new GatedDeltaPlugin();
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs GatedDeltaPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    if (outputIndex == 0)
    {
        return inputs[getValueIdx()];
    }
    return inputs[getStateIdx()];
}

bool GatedDeltaPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    auto const queryType = inOut[getQueryIdx()].type;
    auto const stateType = inOut[getStateIdx()].type;

    if (pos == getHostRequestTypesIdx() || pos == getHostContextLengthsIdx())
    {
        return inOut[pos].type == nvinfer1::DataType::kINT32;
    }

    if (pos == getALogIdx() || pos == getAIdx() || pos == getDtBiasIdx() || pos == getBIdx())
    {
        return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    }

    if (pos == getQueryIdx())
    {
        return isValueTypeSupported(inOut[pos].type) && inOut[pos].format == TensorFormat::kLINEAR;
    }

    if (pos == getKeyIdx() || pos == getValueIdx() || pos == nbInputs)
    {
        return inOut[pos].type == queryType && inOut[pos].format == TensorFormat::kLINEAR;
    }

    if (pos == getStateIdx())
    {
        return isStateTypeSupported(inOut[pos].type) && inOut[pos].format == TensorFormat::kLINEAR;
    }

    if (pos == nbInputs + 1)
    {
        return inOut[pos].type == stateType && inOut[pos].format == TensorFormat::kLINEAR;
    }

    return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
}

void GatedDeltaPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t GatedDeltaPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    auto const batchSize = std::max(1, static_cast<int>(inputs[getHostRequestTypesIdx()].dims.d[0]));
    auto const genericWorkspace = static_cast<size_t>(batchSize) * sizeof(std::int32_t);
    auto const maxSeqLen = std::max(1, static_cast<int>(inputs[getQueryIdx()].dims.d[1]));
    auto const numHeads = std::max(1, static_cast<int>(inputs[getQueryIdx()].dims.d[2]));
    auto const keyDim = std::max(1, static_cast<int>(inputs[getQueryIdx()].dims.d[3]));
    auto const qkTensorBytes = alignSize(
        static_cast<size_t>(batchSize) * maxSeqLen * numHeads * keyDim * getTypeSize(inputs[getQueryIdx()].type));
    auto const scalarTensorBytes
        = alignSize(static_cast<size_t>(batchSize) * maxSeqLen * numHeads * sizeof(float));
    auto const flashinferWorkspace = qkTensorBytes * 2 + scalarTensorBytes * 2
        + tensorrt_llm::kernels::getFlashInferGatedDeltaPrefillWorkspaceSize(batchSize);
    return std::max(genericWorkspace, flashinferWorkspace);
}

template <typename ValueType, typename StateType>
int GatedDeltaPlugin::enqueueImpl(nvinfer1::PluginTensorDesc const* inputDesc, void const* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream)
{
    auto const batchSize = static_cast<int>(inputDesc[getHostRequestTypesIdx()].dims.d[0]);
    auto const maxSeqLen = static_cast<int>(inputDesc[getQueryIdx()].dims.d[1]);
    auto const numHeads = static_cast<int>(inputDesc[getQueryIdx()].dims.d[2]);
    auto const keyDim = static_cast<int>(inputDesc[getQueryIdx()].dims.d[3]);
    auto const valueDim = static_cast<int>(inputDesc[getValueIdx()].dims.d[3]);

    auto const* hostRequestTypes = static_cast<tensorrt_llm::runtime::RequestType const*>(inputs[getHostRequestTypesIdx()]);
    auto const* hostContextLengths = static_cast<int const*>(inputs[getHostContextLengthsIdx()]);

    bool hasContextRequests = false;
    bool hasGenerationRequests = false;
    for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        hasContextRequests = hasContextRequests || hostRequestTypes[batchIdx] == tensorrt_llm::runtime::RequestType::kCONTEXT;
        hasGenerationRequests
            = hasGenerationRequests || hostRequestTypes[batchIdx] == tensorrt_llm::runtime::RequestType::kGENERATION;
    }

    auto const isPureDecode = !hasContextRequests && maxSeqLen == 1;
    auto const isPureContext = hasContextRequests && !hasGenerationRequests;
    bool isUniformFullContext = isPureContext;
    if (isUniformFullContext)
    {
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            auto const contextLength
                = hostContextLengths != nullptr ? static_cast<int>(hostContextLengths[batchIdx]) : maxSeqLen;
            if (contextLength != maxSeqLen)
            {
                isUniformFullContext = false;
                break;
            }
        }
    }

    GatedDeltaParamsBase params{};
    params.batch = batchSize;
    params.maxSeqLen = maxSeqLen;
    params.numHeads = numHeads;
    params.keyDim = keyDim;
    params.valueDim = valueDim;
    params.queryPtr = inputs[getQueryIdx()];
    params.keyPtr = inputs[getKeyIdx()];
    params.valuePtr = inputs[getValueIdx()];
    params.gPtr = nullptr;
    params.betaPtr = nullptr;
    params.aLogPtr = inputs[getALogIdx()];
    params.aPtr = inputs[getAIdx()];
    params.dtBiasPtr = inputs[getDtBiasIdx()];
    params.bPtr = inputs[getBIdx()];
    params.stateInPtr = inputs[getStateIdx()];
    params.stateOutPtr = outputs[1];
    params.outputPtr = outputs[0];

    if (isUniformFullContext)
    {
        auto const cuSeqlensBytes = alignSize(static_cast<size_t>(batchSize + 1) * sizeof(int64_t));
        auto const qkTensorBytes = alignSize(
            static_cast<size_t>(batchSize) * maxSeqLen * numHeads * keyDim * sizeof(ValueType));
        auto const scalarTensorBytes
            = alignSize(static_cast<size_t>(batchSize) * maxSeqLen * numHeads * sizeof(float));
        auto* cuSeqlensDevice = static_cast<int64_t*>(workspace);
        auto* queryWorkspace
            = reinterpret_cast<ValueType*>(static_cast<char*>(workspace) + cuSeqlensBytes);
        auto* keyWorkspace
            = reinterpret_cast<ValueType*>(reinterpret_cast<char*>(queryWorkspace) + qkTensorBytes);
        auto* gWorkspace = reinterpret_cast<float*>(reinterpret_cast<char*>(keyWorkspace) + qkTensorBytes);
        auto* betaWorkspace
            = reinterpret_cast<float*>(reinterpret_cast<char*>(gWorkspace) + scalarTensorBytes);
        auto* flashinferWorkspace = static_cast<void*>(reinterpret_cast<char*>(betaWorkspace) + scalarTensorBytes);

        std::vector<int64_t> cuSeqlens(batchSize + 1);
        for (int batchIdx = 0; batchIdx <= batchSize; ++batchIdx)
        {
            cuSeqlens[batchIdx] = static_cast<int64_t>(batchIdx) * maxSeqLen;
        }

        check_cuda_error(cudaMemcpyAsync(
            cuSeqlensDevice, cuSeqlens.data(), cuSeqlens.size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        invokeGatedDeltaPreprocess<ValueType>(params, queryWorkspace, keyWorkspace, gWorkspace, betaWorkspace, stream);
        params.queryPtr = queryWorkspace;
        params.keyPtr = keyWorkspace;
        params.gPtr = gWorkspace;
        params.betaPtr = betaWorkspace;
        if (tryInvokeFlashInferGatedDeltaPrefill<ValueType, StateType>(
                params, cuSeqlensDevice, flashinferWorkspace, stream))
        {
            sync_check_cuda_error(stream);
            return 0;
        }
    }

    if (!isPureDecode)
    {
        auto const generationInputLength = hasContextRequests ? 1 : maxSeqLen;
        std::vector<std::int32_t> sampleLengths(batchSize);
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            if (hostRequestTypes[batchIdx] == tensorrt_llm::runtime::RequestType::kCONTEXT)
            {
                auto const contextLength
                    = hostContextLengths != nullptr ? static_cast<int>(hostContextLengths[batchIdx]) : maxSeqLen;
                sampleLengths[batchIdx] = std::clamp(contextLength, 0, maxSeqLen);
            }
            else
            {
                sampleLengths[batchIdx] = generationInputLength;
            }
        }

        check_cuda_error(cudaMemcpyAsync(workspace, sampleLengths.data(), sampleLengths.size() * sizeof(std::int32_t),
            cudaMemcpyHostToDevice, stream));
    }

    if (isPureDecode)
    {
        if (!tryInvokeFlashInferGatedDeltaDecode<ValueType, StateType>(params, stream))
        {
            invokeGatedDeltaDecode<ValueType, StateType>(params, stream);
        }
    }
    else
    {
        invokeGatedDelta<ValueType, StateType>(params, static_cast<int const*>(workspace), stream);
    }
    sync_check_cuda_error(stream);
    return 0;
}

int GatedDeltaPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    if (isBuilding())
    {
        return 0;
    }

    auto const stateType = inputDesc[getStateIdx()].type;
    auto const dispatchForStateType = [this, inputDesc, inputs, outputs, workspace,
                                          stream](auto stateTypeTag) -> int
    {
        using StateType = decltype(stateTypeTag);
        auto const valueType = inputDesc[getValueIdx()].type;
        if (valueType == DataType::kHALF)
        {
            return enqueueImpl<half, StateType>(inputDesc, inputs, outputs, workspace, stream);
        }
        if (valueType == DataType::kFLOAT)
        {
            return enqueueImpl<float, StateType>(inputDesc, inputs, outputs, workspace, stream);
        }
#ifdef ENABLE_BF16
        if (valueType == DataType::kBF16)
        {
            return enqueueImpl<__nv_bfloat16, StateType>(inputDesc, inputs, outputs, workspace, stream);
        }
#endif
        return 0;
    };

    if (stateType == DataType::kFLOAT)
    {
        return dispatchForStateType(float{});
    }
    if (stateType == DataType::kHALF)
    {
        return dispatchForStateType(half{});
    }
#ifdef ENABLE_BF16
    if (stateType == DataType::kBF16)
    {
        return dispatchForStateType(__nv_bfloat16{});
    }
#endif
    return 0;
}

nvinfer1::DataType GatedDeltaPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    if (index == 0)
    {
        return inputTypes[getValueIdx()];
    }
    return inputTypes[getStateIdx()];
}

char const* GatedDeltaPlugin::getPluginType() const noexcept
{
    return kGatedDeltaPluginName;
}

char const* GatedDeltaPlugin::getPluginVersion() const noexcept
{
    return kGatedDeltaPluginVersion;
}

int GatedDeltaPlugin::getNbOutputs() const noexcept
{
    return 2;
}

int GatedDeltaPlugin::initialize() noexcept
{
    return 0;
}

void GatedDeltaPlugin::terminate() noexcept {}

size_t GatedDeltaPlugin::getSerializationSize() const noexcept
{
    return 0;
}

void GatedDeltaPlugin::serialize(void* buffer) const noexcept {}

void GatedDeltaPlugin::destroy() noexcept
{
    delete this;
}

GatedDeltaPluginCreator::GatedDeltaPluginCreator()
{
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* GatedDeltaPluginCreator::getPluginName() const noexcept
{
    return kGatedDeltaPluginName;
}

char const* GatedDeltaPluginCreator::getPluginVersion() const noexcept
{
    return kGatedDeltaPluginVersion;
}

PluginFieldCollection const* GatedDeltaPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

nvinfer1::IPluginV2* GatedDeltaPluginCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept
{
    try
    {
        auto* obj = new GatedDeltaPlugin();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

nvinfer1::IPluginV2* GatedDeltaPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto* obj = new GatedDeltaPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
