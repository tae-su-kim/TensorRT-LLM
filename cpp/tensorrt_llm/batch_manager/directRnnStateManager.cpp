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

#include "tensorrt_llm/batch_manager/directRnnStateManager.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"

#include <algorithm>
#include <string>

namespace tensorrt_llm::batch_manager::rnn_state_manager
{

namespace
{

std::vector<runtime::SizeType32> getLocalRecurrentLayerIds(
    runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig)
{
    using LayerType = runtime::ModelConfig::LayerType;
    using SizeType32 = runtime::SizeType32;

    auto const ppSize = worldConfig.getPipelineParallelism();
    auto const ppRank = worldConfig.getPipelineParallelRank();
    auto const localFirstLayer = modelConfig.getFirstLocalLayer(ppSize, ppRank);
    auto const numLocalLayers = modelConfig.getNbLayers(ppSize, ppRank);
    auto const& layerTypes = modelConfig.getLayerTypes();

    if (layerTypes.empty())
    {
        auto const numLocalRnnLayers = modelConfig.getNbRnnLayers(ppSize, ppRank);
        auto const totalRnnLayers = modelConfig.getNbRnnLayers();
        auto const baseLayersPerRank = totalRnnLayers / ppSize;
        auto const numRanksWithExtraLayer = totalRnnLayers % ppSize;
        auto const firstRnnLayer = ppRank * baseLayersPerRank + std::min(ppRank, numRanksWithExtraLayer);

        std::vector<SizeType32> layerIds(numLocalRnnLayers);
        for (SizeType32 localLayer = 0; localLayer < numLocalRnnLayers; ++localLayer)
        {
            layerIds[localLayer] = firstRnnLayer + localLayer;
        }
        return layerIds;
    }

    std::vector<SizeType32> layerIds;
    layerIds.reserve(modelConfig.getNbRnnLayers(ppSize, ppRank));
    for (SizeType32 localOffset = 0; localOffset < numLocalLayers; ++localOffset)
    {
        auto const globalLayer = localFirstLayer + localOffset;
        if (layerTypes[globalLayer] == LayerType::kRECURRENT)
        {
            layerIds.push_back(globalLayer);
        }
    }
    return layerIds;
}

runtime::ITensor::Shape getEngineStateShape(
    runtime::TllmRuntime const& tllmRuntime, std::string const& tensorName, runtime::SizeType32 batchSize)
{
    auto shape = tllmRuntime.getEngine().getTensorShape(tensorName.c_str());
    TLLM_CHECK_WITH_INFO(shape.nbDims > 0, "Tensor %s must have at least one dimension.", tensorName.c_str());
    TLLM_CHECK_WITH_INFO(
        shape.d[0] == -1 || shape.d[0] == batchSize, "Tensor %s must have a dynamic or matching batch dimension.",
        tensorName.c_str());
    shape.d[0] = batchSize;
    for (int dimIdx = 1; dimIdx < shape.nbDims; ++dimIdx)
    {
        TLLM_CHECK_WITH_INFO(shape.d[dimIdx] >= 0, "Tensor %s has unsupported dynamic dimension %d.", tensorName.c_str(),
            dimIdx);
    }
    return shape;
}

} // namespace

DirectRnnStateManager::DirectRnnStateManager(SizeType32 maxNumSequences, runtime::ModelConfig const& modelConfig,
    runtime::WorldConfig const& worldConfig, runtime::TllmRuntime const& tllmRuntime)
    : mGlobalLayerNumsPerPP{getLocalRecurrentLayerIds(modelConfig, worldConfig)}
    , mMaxNumSequences{maxNumSequences}
{
    TLLM_CHECK_WITH_INFO(
        modelConfig.usesDirectRecurrentState(), "DirectRnnStateManager requires a direct recurrent-state model.");

    auto const rnnConfig = modelConfig.getRnnConfig();
    TLLM_CHECK_WITH_INFO(rnnConfig.has_value(), "DirectRnnStateManager requires RNN model configuration.");

    if (mGlobalLayerNumsPerPP.empty())
    {
        return;
    }

    auto const& engine = tllmRuntime.getEngine();
    auto const firstLayer = mGlobalLayerNumsPerPP.front();
    auto const convTensorName = std::string("present_conv_state_") + std::to_string(firstLayer);
    auto const rnnTensorName = std::string("present_rnn_state_") + std::to_string(firstLayer);
    mConvStateDtype = engine.getTensorDataType(convTensorName.c_str());
    mRnnStateDtype = engine.getTensorDataType(rnnTensorName.c_str());

    auto const& bufferManager = tllmRuntime.getBufferManager();
    mConvStates.reserve(mGlobalLayerNumsPerPP.size());
    mRnnStates.reserve(mGlobalLayerNumsPerPP.size());
    for (auto const globalLayer : mGlobalLayerNumsPerPP)
    {
        auto const localConvTensorName = std::string("present_conv_state_") + std::to_string(globalLayer);
        auto const localRnnTensorName = std::string("present_rnn_state_") + std::to_string(globalLayer);
        auto const convStateDtype = engine.getTensorDataType(localConvTensorName.c_str());
        auto const rnnStateDtype = engine.getTensorDataType(localRnnTensorName.c_str());
        auto const convStateShape = getEngineStateShape(tllmRuntime, localConvTensorName, maxNumSequences);
        auto const rnnStateShape = getEngineStateShape(tllmRuntime, localRnnTensorName, maxNumSequences);
        auto convState = runtime::ITensor::SharedPtr(bufferManager.gpu(convStateShape, convStateDtype));
        auto rnnState = runtime::ITensor::SharedPtr(bufferManager.gpu(rnnStateShape, rnnStateDtype));
        bufferManager.setZero(*convState);
        bufferManager.setZero(*rnnState);
        mConvStates.push_back(convState);
        mRnnStates.push_back(rnnState);
    }
}

DirectRnnStateManager::SizeType32 DirectRnnStateManager::getGlobalLayerNum(SizeType32 localOffset) const
{
    TLLM_CHECK(localOffset < static_cast<SizeType32>(mGlobalLayerNumsPerPP.size()));
    return mGlobalLayerNumsPerPP[localOffset];
}

DirectRnnStateManager::TensorPtr DirectRnnStateManager::getConvStates(SizeType32 localOffset) const
{
    TLLM_CHECK(localOffset < static_cast<SizeType32>(mConvStates.size()));
    return mConvStates[localOffset];
}

DirectRnnStateManager::TensorPtr DirectRnnStateManager::getRnnStates(SizeType32 localOffset) const
{
    TLLM_CHECK(localOffset < static_cast<SizeType32>(mRnnStates.size()));
    return mRnnStates[localOffset];
}

} // namespace tensorrt_llm::batch_manager::rnn_state_manager
