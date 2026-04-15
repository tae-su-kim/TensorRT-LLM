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

#ifndef TRT_GATED_DELTA_PLUGIN_H
#define TRT_GATED_DELTA_PLUGIN_H

#include "tensorrt_llm/kernels/gatedDeltaKernels.h"
#include "tensorrt_llm/plugins/common/plugin.h"

namespace tensorrt_llm::plugins
{

class GatedDeltaPlugin : public BasePlugin
{
public:
    GatedDeltaPlugin() = default;
    GatedDeltaPlugin(void const* data, size_t length);
    ~GatedDeltaPlugin() override = default;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept override;
    int enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    template <typename ValueType, typename StateType>
    int enqueueImpl(nvinfer1::PluginTensorDesc const* inputDesc, void const* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream);

    nvinfer1::DataType getOutputDataType(
        int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept override;

    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;

private:
    using IndexType = std::int32_t;

    static constexpr IndexType getQueryIdx()
    {
        return 0;
    }

    static constexpr IndexType getKeyIdx()
    {
        return 1;
    }

    static constexpr IndexType getValueIdx()
    {
        return 2;
    }

    static constexpr IndexType getStateIdx()
    {
        return 3;
    }

    static constexpr IndexType getALogIdx()
    {
        return 4;
    }

    static constexpr IndexType getAIdx()
    {
        return 5;
    }

    static constexpr IndexType getDtBiasIdx()
    {
        return 6;
    }

    static constexpr IndexType getBIdx()
    {
        return 7;
    }

    static constexpr IndexType getHostRequestTypesIdx()
    {
        return 8;
    }

    static constexpr IndexType getHostContextLengthsIdx()
    {
        return 9;
    }
};

class GatedDeltaPluginCreator : public BaseCreator
{
public:
    GatedDeltaPluginCreator();

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

} // namespace tensorrt_llm::plugins

#endif // TRT_GATED_DELTA_PLUGIN_H
