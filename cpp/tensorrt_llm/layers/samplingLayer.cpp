/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/samplingTopKKernels.h"
#include "tensorrt_llm/layers/defaultDecodingParams.h"
#include "tensorrt_llm/layers/layerUtils.h"
#include "tensorrt_llm/layers/topKSamplingLayer.h"
#include "tensorrt_llm/layers/topPSamplingLayer.h"

#include "samplingLayer.h"
#include <algorithm>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::layers
{

namespace
{

bool isGreedySamplingConfig(SizeType32 topK, float topP)
{
    return topK == 1 && topP == 1.0f;
}

} // namespace

template <typename T>
SamplingLayer<T>::SamplingLayer(executor::DecodingMode const& mode, DecoderDomain const& decoderDomain,
    std::shared_ptr<BufferManager> bufferManager)
    : BaseLayer(decoderDomain, bufferManager)
    , mDecodingMode(mode)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK_WITH_INFO(!mDecodingMode.isBeamSearch(), "SamplingLayer does not support Beam search mode");
    TLLM_CHECK_WITH_INFO(mDecodingMode.isTopKorTopP(), "SamplingLayer requires TopK or TopP mode");
    if (mDecodingMode.isTopK())
    {
        mSamplingLayers.emplace_back(std::make_unique<TopKSamplingLayer<T>>(decoderDomain, mBufferManager));
    }

    if (mDecodingMode.isTopP())
    {
        mSamplingLayers.emplace_back(
            std::make_unique<TopPSamplingLayer<T>>(decoderDomain, mBufferManager, /* deterministic */ true));
    }

    allocateBuffer(decoderDomain.getBatchSize());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void SamplingLayer<T>::allocateBuffer(SizeType32 batchSize)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    size_t workspaceSize = 0;
    for (auto&& layer : mSamplingLayers)
    {
        workspaceSize = std::max(workspaceSize, layer->getWorkspaceSize());
    }
    mWorkspaceSize = workspaceSize;

    auto const batchSizeShape = ITensor::makeShape({batchSize});
    mSetupWorkspaceSize = DecodingLayerWorkspace::calculateRequiredWorkspaceSize(
        std::make_pair(batchSizeShape, TRTDataType<uint64_t>::value));
    mSkipDecodeDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<bool>::value);
    mCurandStatesDevice
        = mBufferManager->gpu(ITensor::makeShape({batchSize, sizeof(curandState_t)}), TRTDataType<int8_t>::value);

    // host buffers.
    mSkipDecodeHost = mBufferManager->pinnedPool(batchSizeShape, TRTDataType<bool>::value);
    mLogitsPtrsHost = mBufferManager->pinnedPool(batchSizeShape, TRTDataType<T*>::value);

    mRuntimeMinPHost = mBufferManager->pinnedPool(batchSizeShape, TRTDataType<float>::value);
    mRuntimeMinPDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<float>::value);
    mGreedySamplingHost = mBufferManager->cpu(batchSizeShape, TRTDataType<bool>::value);
    TLLM_CHECK(mSkipDecodeHost != nullptr);

    std::fill_n(bufferCast<bool>(*mGreedySamplingHost), batchSize, false);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void SamplingLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, TensorConstPtr batchSlots,
    std::shared_ptr<BaseSetupParams> const& baseSetupParams,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(SamplingLayer_setup);

    auto setupParams = std::dynamic_pointer_cast<SamplingSetupParams>(baseSetupParams);
    mNormalizeLogProbs = setupParams->normalizeLogProbs.value_or(false);

    workspace->initializeDeviceCurandStates(
        setupParams->randomSeed, batchSize, workspace->getDeviceBatchSlots(), mCurandStatesDevice);

    if (setupParams->outputLogProbs)
    {
        // FIXME: monotonically growing
        mOutputLogProbs = std::any_of(setupParams->outputLogProbs->begin(), setupParams->outputLogProbs->end(),
            [this](bool outputLogProbs) { return this->mOutputLogProbs | outputLogProbs; });
    }

    if (setupParams->cumLogProbs)
    {
        // FIXME: monotonically growing
        mCumLogProbs = std::any_of(setupParams->cumLogProbs->begin(), setupParams->cumLogProbs->end(),
            [this](bool cumLogProbs) { return this->mCumLogProbs | cumLogProbs; });
    }

    for (auto&& layer : mSamplingLayers)
    {
        layer->setup(batchSize, beamWidth, batchSlots, setupParams, workspace);
    }

    FillBuffers const fillBuffers{batchSize, mDecoderDomain.getBatchSize(), mBufferManager};
    bool const useMinP = mDecodingMode.isUseMinP() && setupParams->runtimeMinP.has_value();
    mUseMinP |= useMinP;
    if (mUseMinP)
    {
        fillBuffers(setupParams->runtimeMinP, DefaultDecodingParams::getMinP(), mRuntimeMinPHost, mRuntimeMinPDevice,
            batchSlots, std::pair<float, float>(-1e-6f, 1.0f), "min_p");
    }

    auto runtimeTopK = setupParams->runtimeTopK.value_or(std::vector{DefaultDecodingParams::getTopK()});
    auto runtimeTopP = setupParams->runtimeTopP.value_or(std::vector{DefaultDecodingParams::getTopP()});
    auto const paramsSize = expandMatchElements(batchSize, runtimeTopK, runtimeTopP);
    TLLM_CHECK_WITH_INFO(paramsSize != 0,
        "SamplingLayer got parameter with unexpected size, want 1 or batchSize(%d), got"
        "runtimeTopK.size() = %zu, runtimeTopP.size() = %zu",
        batchSize, runtimeTopK.size(), runtimeTopP.size());
    for (size_t i = 0; i < paramsSize; ++i)
    {
        auto& topK = runtimeTopK[i];
        auto& topP = runtimeTopP[i];
        clampTopK(topK);
        clampTopP(topP);
        regularizeTopKTopP(topK, topP);
    }

    auto* greedySamplingHost = bufferCast<bool>(*mGreedySamplingHost);
    auto const* batchSlotsHost = bufferCast<SizeType32>(*batchSlots);
    for (SizeType32 bi = 0; bi < batchSize; ++bi)
    {
        auto const batchSlot = batchSlotsHost[bi];
        auto const topK = paramsSize == 1 ? runtimeTopK.front() : runtimeTopK[bi];
        auto const topP = paramsSize == 1 ? runtimeTopP.front() : runtimeTopP[bi];
        greedySamplingHost[batchSlot] = isGreedySamplingConfig(topK, topP);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void SamplingLayer<T>::forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(SamplingLayer_forwardAsync);

    auto inputs = std::dynamic_pointer_cast<SamplingInputs>(baseInputs);

    auto const localDecoderDomain = getLocalDecoderDomain(inputs, mDecoderDomain);
    auto const batchSize
        = inputs->logits ? inputs->logits.value()->getDimension<0>() : static_cast<SizeType32>(inputs->logitsVec->size());

    auto const* endIds = bufferCast<TokenIdType>(*inputs->endIds);

    FinishedState const* finishedInput = (inputs->finished)
        ? reinterpret_cast<FinishedState const*>(bufferCast<FinishedState::UnderlyingType>(*inputs->finished.value()))
        : nullptr;

    auto const skipTopP = !mDecodingMode.isTopP();

    auto const* batchSlotsHostPtr = bufferCast<SizeType32>(*inputs->batchSlots);
    auto minPs = mUseMinP
            && !allOfBatchSlots(batchSlotsHostPtr, bufferCast<float>(*mRuntimeMinPHost),
                localDecoderDomain.getBatchSize(), DefaultDecodingParams::getMinP())
        ? mRuntimeMinPDevice
        : nullptr;
    bool const hasBadWords = inputs->banWordsInputs && inputs->banWordsInputs->maxBadWordsLen != 0;

    bool const canUseGreedySampling = mDecoderDomain.getMaxDecodingTokens() == 1
        && localDecoderDomain.getBeamWidth() == 1 && (inputs->logits.has_value() || inputs->logitsVec.has_value())
        && !mNormalizeLogProbs && !mOutputLogProbs && !mCumLogProbs && minPs == nullptr && !inputs->probsComputed
        && !hasBadWords
        && allOfBatchSlots(batchSlotsHostPtr, bufferCast<bool>(*mGreedySamplingHost),
            localDecoderDomain.getBatchSize(), true);

    inputs->curandStates = reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*mCurandStatesDevice));
    if (canUseGreedySampling)
    {
        TopKSamplingKernelParams<T> params;
        if (inputs->logits)
        {
            params.logProbs = bufferCastOrNull<T>(inputs->logits);
        }
        else
        {
            TLLM_CHECK(inputs->logitsVec.has_value());
            TLLM_CHECK_WITH_INFO(inputs->logitsVec->size() == static_cast<size_t>(batchSize),
                "Logits vector size (%zu) does not match batch size (%zu)", inputs->logitsVec->size(),
                static_cast<size_t>(batchSize));
            auto logitsPtrsHostSlice = ITensor::slice(mLogitsPtrsHost, 0, batchSize);
            auto logitsPtrsHostData = bufferCast<T const*>(*logitsPtrsHostSlice);
            for (SizeType32 bi = 0; bi < batchSize; ++bi)
            {
                logitsPtrsHostData[bi] = bufferCastOrNull<T>(inputs->logitsVec.value()[bi]);
            }
            auto [logitsPtrsDeviceSlice] = workspace->mirrorInWorkspace(logitsPtrsHostSlice.get());
            params.logProbsPtrs = reinterpret_cast<T const* const*>(bufferCast<T const*>(*logitsPtrsDeviceSlice));
        }
        params.outputIdsPtrs = bufferCastOrNull<TokenIdType*>(outputs->outputIdsPtr);
        params.outputIds = nullptr;
        params.workspace = workspace->getRawWorkspaceDevicePtr();
        params.endIds = endIds;
        params.sequenceLengths = bufferCastOrNull<SizeType32>(outputs->sequenceLength);
        params.batchSlots = workspace->getDeviceBatchSlotsPtr();
        params.finishedInput = finishedInput;
        params.finishedOutput = (outputs->finished)
            ? reinterpret_cast<FinishedState*>(bufferCastOrNull<FinishedState::UnderlyingType>(outputs->finished))
            : nullptr;
        params.skipDecode = bufferCastOrNull<bool>(mSkipDecodeDevice);
        params.batchSize = batchSize;
        params.maxBatchSize = mDecoderDomain.getBatchSize();
        params.vocabSizePadded = mDecoderDomain.getVocabSizePadded();
        params.maxTokensPerStep = 1;
        params.maxSeqLen = outputs->outputIds->getDimension<-1>();
        invokeBatchGreedySampling(params, getStream());
        inputs->probsComputed = false;
        TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
        return;
    }

    // Compute probabilities either for TopP or if cumLogProbs or outputLogProbs are specified
    bool const skipSoftMax = skipTopP && !mOutputLogProbs && !mCumLogProbs && minPs == nullptr;
    inputs->probsComputed = !skipSoftMax;
    if (!skipSoftMax)
    {
        auto runtimeLogitsPtr = bufferCast<T>(*workspace->getDeviceRuntimeLogits());
        auto logitsPtrsPtr = static_cast<T**>(nullptr);
        auto biasPtr = static_cast<T*>(nullptr);
        auto const* batchSlotsPtr = workspace->getDeviceBatchSlotsPtr();

        BiasSoftmaxParams<T> biasSoftmaxParams;
        biasSoftmaxParams.logits = runtimeLogitsPtr;
        biasSoftmaxParams.logitsPtrs = logitsPtrsPtr;
        biasSoftmaxParams.probs = runtimeLogitsPtr;
        biasSoftmaxParams.bias = biasPtr;
        biasSoftmaxParams.endIds = endIds;
        biasSoftmaxParams.finished = finishedInput;
        biasSoftmaxParams.batchSlots = batchSlotsPtr;
        biasSoftmaxParams.batchSize = batchSize;
        biasSoftmaxParams.maxBatchSize = mDecoderDomain.getBatchSize();
        biasSoftmaxParams.maxBeamWidth = 1;
        biasSoftmaxParams.vocabSize = mDecoderDomain.getVocabSize();
        biasSoftmaxParams.vocabSizePadded = mDecoderDomain.getVocabSizePadded();
        biasSoftmaxParams.skipSoftMax = skipSoftMax;
        biasSoftmaxParams.batchSlotsLogits = false;
        biasSoftmaxParams.minPs = bufferCastOrNull<float>(minPs);
        biasSoftmaxParams.checkParams();
        invokeAddBiasSoftMax(biasSoftmaxParams, getStream());
        sync_check_cuda_error(getStream());
    }

    for (auto&& layer : mSamplingLayers)
    {
        layer->forwardAsync(outputs, baseInputs, workspace);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
size_t SamplingLayer<T>::getWorkspaceSize() const noexcept
{
    return std::max(mWorkspaceSize, mSetupWorkspaceSize);
}

template class SamplingLayer<float>;
template class SamplingLayer<half>;

} // namespace tensorrt_llm::layers
