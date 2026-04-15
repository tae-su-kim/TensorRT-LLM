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

#include "tensorrt_llm/batch_manager/directRnnStateBuffers.h"

#include "tensorrt_llm/batch_manager/directRnnStateManager.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"

#include <algorithm>
#include <string>

namespace tensorrt_llm::batch_manager
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

runtime::SizeType64 getRowSize(runtime::ITensor const& tensor)
{
    auto const& shape = tensor.getShape();
    TLLM_CHECK(shape.nbDims > 0);
    TLLM_CHECK(shape.d[0] > 0);
    return tensor.getSize() / static_cast<runtime::SizeType64>(shape.d[0]);
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

DirectRnnStateBuffers::DirectRnnStateBuffers(SizeType32 maxBatchSize, runtime::TllmRuntime const& tllmRuntime,
    runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig)
    : mGlobalLayerNumsPerPP{getLocalRecurrentLayerIds(modelConfig, worldConfig)}
{
    TLLM_CHECK_WITH_INFO(
        modelConfig.usesDirectRecurrentState(), "DirectRnnStateBuffers require a direct recurrent-state model.");

    auto const rnnConfig = modelConfig.getRnnConfig();
    TLLM_CHECK_WITH_INFO(rnnConfig.has_value(), "DirectRnnStateBuffers require RNN model configuration.");

    auto const& bufferManager = tllmRuntime.getBufferManager();
    auto const& engine = tllmRuntime.getEngine();
    auto const offsetsShape = tensorrt_llm::runtime::ITensor::makeShape({maxBatchSize});
    mCopySrcOffsetsHost = runtime::ITensor::SharedPtr(
        runtime::BufferManager::pinnedPool(offsetsShape, nvinfer1::DataType::kINT64));
    mCopySrcOffsetsDevice = runtime::ITensor::SharedPtr(bufferManager.gpu(offsetsShape, nvinfer1::DataType::kINT64));
    mCopyDstOffsetsHost = runtime::ITensor::SharedPtr(
        runtime::BufferManager::pinnedPool(offsetsShape, nvinfer1::DataType::kINT64));
    mCopyDstOffsetsDevice = runtime::ITensor::SharedPtr(bufferManager.gpu(offsetsShape, nvinfer1::DataType::kINT64));
    mCopySizesHost = runtime::ITensor::SharedPtr(
        runtime::BufferManager::pinnedPool(offsetsShape, nvinfer1::DataType::kINT64));
    mCopySizesDevice = runtime::ITensor::SharedPtr(bufferManager.gpu(offsetsShape, nvinfer1::DataType::kINT64));

    if (mGlobalLayerNumsPerPP.empty())
    {
        return;
    }

    mPastConvStates.reserve(mGlobalLayerNumsPerPP.size());
    mPresentConvStates.reserve(mGlobalLayerNumsPerPP.size());
    mRnnStates.reserve(mGlobalLayerNumsPerPP.size());
    for (auto const globalLayer : mGlobalLayerNumsPerPP)
    {
        auto const convTensorName = std::string("present_conv_state_") + std::to_string(globalLayer);
        auto const rnnTensorName = std::string("present_rnn_state_") + std::to_string(globalLayer);
        auto const convStateDtype = engine.getTensorDataType(convTensorName.c_str());
        auto const rnnStateDtype = engine.getTensorDataType(rnnTensorName.c_str());
        auto const convStateShape = getEngineStateShape(tllmRuntime, convTensorName, maxBatchSize);
        auto const rnnStateShape = getEngineStateShape(tllmRuntime, rnnTensorName, maxBatchSize);
        mPastConvStates.push_back(runtime::ITensor::SharedPtr(bufferManager.gpu(convStateShape, convStateDtype)));
        mPresentConvStates.push_back(runtime::ITensor::SharedPtr(bufferManager.gpu(convStateShape, convStateDtype)));
        mRnnStates.push_back(runtime::ITensor::SharedPtr(bufferManager.gpu(rnnStateShape, rnnStateDtype)));
    }
}

bool DirectRnnStateBuffers::tryEnableContiguousGenerationStateViews(SizeType32 numContextRequests, SizeType32 numSequences,
    runtime::ITensor const& seqSlotsHost, rnn_state_manager::DirectRnnStateManager const& directRnnStateManager)
{
    resetStateViews();
    if (numContextRequests != 0 || numSequences == 0)
    {
        return false;
    }

    auto const* seqSlots = runtime::bufferCast<SizeType32>(seqSlotsHost);
    auto const firstSlot = seqSlots[0];
    for (SizeType32 batchIdx = 0; batchIdx < numSequences; ++batchIdx)
    {
        if (seqSlots[batchIdx] != firstSlot + batchIdx)
        {
            return false;
        }
    }

    mPastConvStateViews.reserve(mGlobalLayerNumsPerPP.size());
    mRnnStateViews.reserve(mGlobalLayerNumsPerPP.size());
    for (SizeType32 localLayer = 0; localLayer < static_cast<SizeType32>(mGlobalLayerNumsPerPP.size()); ++localLayer)
    {
        mPastConvStateViews.push_back(TensorPtr(
            runtime::ITensor::slice(directRnnStateManager.getConvStates(localLayer), firstSlot, numSequences)));
        mRnnStateViews.push_back(TensorPtr(
            runtime::ITensor::slice(directRnnStateManager.getRnnStates(localLayer), firstSlot, numSequences)));
    }

    mUseContiguousGenerationStateViews = true;
    mContiguousGenerationStateOffset = firstSlot;
    return true;
}

void DirectRnnStateBuffers::resetStateViews()
{
    mPastConvStateViews.clear();
    mRnnStateViews.clear();
    mUseContiguousGenerationStateViews = false;
    mContiguousGenerationStateOffset = 0;
}

void DirectRnnStateBuffers::reshape(SizeType32 numSequences)
{
    resetStateViews();

    auto reshapeTensors = [numSequences](std::vector<TensorPtr>& tensors)
    {
        for (auto const& tensor : tensors)
        {
            auto shape = tensor->getShape();
            shape.d[0] = numSequences;
            tensor->reshape(shape);
        }
    };

    reshapeTensors(mPastConvStates);
    reshapeTensors(mPresentConvStates);
    reshapeTensors(mRnnStates);

    auto const offsetsShape = tensorrt_llm::runtime::ITensor::makeShape({numSequences});
    mCopySrcOffsetsHost->reshape(offsetsShape);
    mCopySrcOffsetsDevice->reshape(offsetsShape);
    mCopyDstOffsetsHost->reshape(offsetsShape);
    mCopyDstOffsetsDevice->reshape(offsetsShape);
    mCopySizesHost->reshape(offsetsShape);
    mCopySizesDevice->reshape(offsetsShape);
}

void DirectRnnStateBuffers::copyRows(runtime::ITensor const& srcTensor, runtime::ITensor& dstTensor,
    runtime::ITensor const& seqSlotsHost, SizeType32 batchStart, SizeType32 numRows, bool gatherFromStateStore,
    runtime::TllmRuntime const& tllmRuntime)
{
    if (numRows == 0)
    {
        return;
    }

    auto const rowSize = getRowSize(srcTensor);
    auto const* seqSlots = runtime::bufferCast<SizeType32>(seqSlotsHost);
    auto* srcOffsetsHost = runtime::bufferCast<SizeType64>(*mCopySrcOffsetsHost);
    auto* dstOffsetsHost = runtime::bufferCast<SizeType64>(*mCopyDstOffsetsHost);
    auto* copySizesHost = runtime::bufferCast<SizeType64>(*mCopySizesHost);

    for (SizeType32 batchOffset = 0; batchOffset < numRows; ++batchOffset)
    {
        auto const batchIndex = batchStart + batchOffset;
        auto const stateOffset = static_cast<SizeType64>(seqSlots[batchIndex]) * rowSize;
        auto const batchTensorOffset = static_cast<SizeType64>(batchIndex) * rowSize;
        srcOffsetsHost[batchOffset] = gatherFromStateStore ? stateOffset : batchTensorOffset;
        dstOffsetsHost[batchOffset] = gatherFromStateStore ? batchTensorOffset : stateOffset;
        copySizesHost[batchOffset] = rowSize;
    }

    auto const& bufferManager = tllmRuntime.getBufferManager();
    auto const copySrcOffsetsHost = runtime::ITensor::slice(mCopySrcOffsetsHost, 0, numRows);
    auto const copyDstOffsetsHost = runtime::ITensor::slice(mCopyDstOffsetsHost, 0, numRows);
    auto const copySizesHostTensor = runtime::ITensor::slice(mCopySizesHost, 0, numRows);
    auto const copySrcOffsetsDevice = runtime::ITensor::slice(mCopySrcOffsetsDevice, 0, numRows);
    auto const copyDstOffsetsDevice = runtime::ITensor::slice(mCopyDstOffsetsDevice, 0, numRows);
    auto const copySizesDevice = runtime::ITensor::slice(mCopySizesDevice, 0, numRows);
    bufferManager.copy(*copySrcOffsetsHost, *copySrcOffsetsDevice);
    bufferManager.copy(*copyDstOffsetsHost, *copyDstOffsetsDevice);
    bufferManager.copy(*copySizesHostTensor, *copySizesDevice);

    runtime::kernels::invokeCopyBatch(
        srcTensor, dstTensor, *copySrcOffsetsDevice, *copyDstOffsetsDevice, *copySizesDevice, rowSize,
        tllmRuntime.getStream());
}

void DirectRnnStateBuffers::setFromInputs(SizeType32 numContextRequests, SizeType32 numSequences,
    runtime::ITensor const& seqSlotsHost, rnn_state_manager::DirectRnnStateManager const& directRnnStateManager,
    runtime::TllmRuntime const& tllmRuntime)
{
    if (tryEnableContiguousGenerationStateViews(
            numContextRequests, numSequences, seqSlotsHost, directRnnStateManager))
    {
        return;
    }

    auto const& bufferManager = tllmRuntime.getBufferManager();
    for (auto const& tensor : mPastConvStates)
    {
        bufferManager.setZero(*tensor);
    }
    for (auto const& tensor : mRnnStates)
    {
        bufferManager.setZero(*tensor);
    }

    auto const numGenerationRows = numSequences - numContextRequests;
    if (numGenerationRows == 0)
    {
        return;
    }

    for (SizeType32 localLayer = 0; localLayer < static_cast<SizeType32>(mGlobalLayerNumsPerPP.size()); ++localLayer)
    {
        copyRows(*directRnnStateManager.getConvStates(localLayer), *mPastConvStates[localLayer], seqSlotsHost,
            numContextRequests, numGenerationRows, /*gatherFromStateStore=*/true, tllmRuntime);
        copyRows(*directRnnStateManager.getRnnStates(localLayer), *mRnnStates[localLayer], seqSlotsHost,
            numContextRequests, numGenerationRows, /*gatherFromStateStore=*/true, tllmRuntime);
    }
}

void DirectRnnStateBuffers::commitOutputs(SizeType32 numSequences, runtime::ITensor const& seqSlotsHost,
    rnn_state_manager::DirectRnnStateManager& directRnnStateManager, runtime::TllmRuntime const& tllmRuntime)
{
    if (mUseContiguousGenerationStateViews)
    {
        auto const& bufferManager = tllmRuntime.getBufferManager();
        for (SizeType32 localLayer = 0; localLayer < static_cast<SizeType32>(mGlobalLayerNumsPerPP.size()); ++localLayer)
        {
            auto directConvState = TensorPtr(runtime::ITensor::slice(
                directRnnStateManager.getConvStates(localLayer), mContiguousGenerationStateOffset, numSequences));
            bufferManager.copy(*mPresentConvStates[localLayer], *directConvState);
        }
        return;
    }

    for (SizeType32 localLayer = 0; localLayer < static_cast<SizeType32>(mGlobalLayerNumsPerPP.size()); ++localLayer)
    {
        copyRows(*mPresentConvStates[localLayer], *directRnnStateManager.getConvStates(localLayer), seqSlotsHost, 0,
            numSequences, /*gatherFromStateStore=*/false, tllmRuntime);
        copyRows(*mRnnStates[localLayer], *directRnnStateManager.getRnnStates(localLayer), seqSlotsHost, 0,
            numSequences, /*gatherFromStateStore=*/false, tllmRuntime);
    }
}

void DirectRnnStateBuffers::getBuffers(TensorMap& inputBuffers, TensorMap& outputBuffers) const
{
    for (std::size_t localLayer = 0; localLayer < mGlobalLayerNumsPerPP.size(); ++localLayer)
    {
        auto const layerId = mGlobalLayerNumsPerPP[localLayer];
        auto const layerSuffix = std::to_string(layerId);
        auto const& pastConvState = mUseContiguousGenerationStateViews ? mPastConvStateViews[localLayer]
                                                                       : mPastConvStates[localLayer];
        auto const& rnnState = mUseContiguousGenerationStateViews ? mRnnStateViews[localLayer] : mRnnStates[localLayer];
        inputBuffers.insert_or_assign("past_conv_state_" + layerSuffix, pastConvState);
        outputBuffers.insert_or_assign("present_conv_state_" + layerSuffix, mPresentConvStates[localLayer]);
        inputBuffers.insert_or_assign("past_rnn_state_" + layerSuffix, rnnState);
        outputBuffers.insert_or_assign("present_rnn_state_" + layerSuffix, rnnState);
    }
}

} // namespace tensorrt_llm::batch_manager
