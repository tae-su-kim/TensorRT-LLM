/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "qwenCausalConv1dPlugin.h"

#include "tensorrt_llm/common/assert.h"

#include <cstring>
#include <string>

using namespace nvinfer1;
using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels::causal_conv1d;
using tensorrt_llm::plugins::QwenCausalConv1dPlugin;
using tensorrt_llm::plugins::QwenCausalConv1dPluginCreator;

namespace
{

char const* kPluginVersion{"1"};
char const* kPluginName{"QwenCausalConv1d"};

bool isTypeSupported(nvinfer1::DataType type)
{
    return type == DataType::kFLOAT || type == DataType::kHALF || type == DataType::kBF16;
}

} // namespace

PluginFieldCollection QwenCausalConv1dPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> QwenCausalConv1dPluginCreator::mPluginAttributes;

QwenCausalConv1dPlugin::QwenCausalConv1dPlugin(int dim, int width, nvinfer1::DataType type)
    : mDim(dim)
    , mWidth(width)
    , mType(type)
{
    TLLM_CHECK_WITH_INFO(isTypeSupported(mType), "Only float, half, and bfloat16 are supported.");
    TLLM_CHECK_WITH_INFO(mWidth >= 2 && mWidth <= 4, "QwenCausalConv1d only supports width between 2 and 4.");
}

QwenCausalConv1dPlugin::QwenCausalConv1dPlugin(void const* data, size_t length)
{
    char const* d = reinterpret_cast<char const*>(data);
    char const* a = d;
    read(d, mDim);
    read(d, mWidth);
    read(d, mType);
    TLLM_CHECK(d == a + length);
    TLLM_CHECK_WITH_INFO(isTypeSupported(mType), "Only float, half, and bfloat16 are supported.");
    TLLM_CHECK_WITH_INFO(mWidth >= 2 && mWidth <= 4, "QwenCausalConv1d only supports width between 2 and 4.");
}

nvinfer1::IPluginV2DynamicExt* QwenCausalConv1dPlugin::clone() const noexcept
{
    auto* plugin = new QwenCausalConv1dPlugin(mDim, mWidth, mType);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs QwenCausalConv1dPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    if (outputIndex == 0)
    {
        return inputs[getInputTensorIdx()];
    }
    return inputs[getConvStateIdx()];
}

bool QwenCausalConv1dPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    return inOut[pos].type == mType && inOut[pos].format == TensorFormat::kLINEAR;
}

void QwenCausalConv1dPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t QwenCausalConv1dPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return 0;
}

void QwenCausalConv1dPlugin::setCommonParams(ConvParamsBase& params, int batchSize, int seqLen, void const* input,
    void* convState, void const* weight, void* output) const
{
    std::memset(&params, 0, sizeof(params));

    params.batch = batchSize;
    params.dim = mDim;
    params.seqlen = seqLen;
    params.width = mWidth;
    params.pad_slot_id = -1;
    params.silu_activation = true;

    params.x_ptr = const_cast<void*>(input);
    params.weight_ptr = const_cast<void*>(weight);
    params.bias_ptr = nullptr;
    params.out_ptr = output;

    params.x_batch_stride = seqLen * mDim;
    params.x_c_stride = 1;
    params.x_l_stride = mDim;

    params.weight_c_stride = mWidth;
    params.weight_width_stride = 1;

    params.out_batch_stride = seqLen * mDim;
    params.out_c_stride = 1;
    params.out_l_stride = mDim;

    params.conv_state_ptr = convState;
    params.conv_state_len = mWidth - 1;
    params.conv_state_batch_stride = mDim * (mWidth - 1);
    params.conv_state_c_stride = mWidth - 1;
    params.conv_state_l_stride = 1;
}

template <typename T>
int QwenCausalConv1dPlugin::enqueueImpl(nvinfer1::PluginTensorDesc const* inputDesc, void const* const* inputs,
    void* const* outputs, cudaStream_t stream)
{
    int const batchSize = static_cast<int>(inputDesc[getInputTensorIdx()].dims.d[0]);
    int const seqLen = static_cast<int>(inputDesc[getInputTensorIdx()].dims.d[1]);

    size_t const stateBytes = static_cast<size_t>(batchSize) * mDim * (mWidth - 1) * sizeof(T);
    if (inputs[getConvStateIdx()] != outputs[1])
    {
        check_cuda_error(cudaMemcpyAsync(
            outputs[1], inputs[getConvStateIdx()], stateBytes, cudaMemcpyDeviceToDevice, stream));
    }

    ConvParamsBase params{};
    setCommonParams(params, batchSize, seqLen, inputs[getInputTensorIdx()], outputs[1], inputs[getWeightIdx()],
        outputs[0]);
    causal_conv1d_update_cuda<T, T>(params, stream);

    sync_check_cuda_error(stream);
    return 0;
}

int QwenCausalConv1dPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    if (isBuilding())
    {
        return 0;
    }

    if (mType == DataType::kHALF)
    {
        return enqueueImpl<half>(inputDesc, inputs, outputs, stream);
    }
    if (mType == DataType::kFLOAT)
    {
        return enqueueImpl<float>(inputDesc, inputs, outputs, stream);
    }
#ifdef ENABLE_BF16
    if (mType == DataType::kBF16)
    {
        return enqueueImpl<__nv_bfloat16>(inputDesc, inputs, outputs, stream);
    }
#endif
    return 1;
}

nvinfer1::DataType QwenCausalConv1dPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    return inputTypes[getInputTensorIdx()];
}

char const* QwenCausalConv1dPlugin::getPluginType() const noexcept
{
    return kPluginName;
}

char const* QwenCausalConv1dPlugin::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

int QwenCausalConv1dPlugin::getNbOutputs() const noexcept
{
    return 2;
}

int QwenCausalConv1dPlugin::initialize() noexcept
{
    return 0;
}

void QwenCausalConv1dPlugin::terminate() noexcept {}

size_t QwenCausalConv1dPlugin::getSerializationSize() const noexcept
{
    return sizeof(mDim) + sizeof(mWidth) + sizeof(mType);
}

void QwenCausalConv1dPlugin::serialize(void* buffer) const noexcept
{
    char* d = static_cast<char*>(buffer);
    write(d, mDim);
    write(d, mWidth);
    write(d, mType);
}

void QwenCausalConv1dPlugin::destroy() noexcept
{
    delete this;
}

QwenCausalConv1dPluginCreator::QwenCausalConv1dPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("dim", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("width", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 0));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* QwenCausalConv1dPluginCreator::getPluginName() const noexcept
{
    return kPluginName;
}

char const* QwenCausalConv1dPluginCreator::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

nvinfer1::PluginFieldCollection const* QwenCausalConv1dPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

nvinfer1::IPluginV2* QwenCausalConv1dPluginCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept
{
    int dim = 0;
    int width = 0;
    int typeId = 0;

    for (int i = 0; i < fc->nbFields; ++i)
    {
        std::string const fieldName(fc->fields[i].name);
        if (fieldName == "dim")
        {
            dim = *static_cast<int const*>(fc->fields[i].data);
        }
        else if (fieldName == "width")
        {
            width = *static_cast<int const*>(fc->fields[i].data);
        }
        else if (fieldName == "type_id")
        {
            typeId = *static_cast<int const*>(fc->fields[i].data);
        }
    }

    try
    {
        auto* obj = new QwenCausalConv1dPlugin(dim, width, static_cast<nvinfer1::DataType>(typeId));
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

nvinfer1::IPluginV2* QwenCausalConv1dPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        auto* obj = new QwenCausalConv1dPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
