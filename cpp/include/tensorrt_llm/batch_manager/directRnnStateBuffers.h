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

#pragma once

#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <vector>

namespace tensorrt_llm::runtime
{
class TllmRuntime;
} // namespace tensorrt_llm::runtime

namespace tensorrt_llm::batch_manager::rnn_state_manager
{
class DirectRnnStateManager;
}

namespace tensorrt_llm::batch_manager
{

class DirectRnnStateBuffers
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using SizeType64 = tensorrt_llm::runtime::SizeType64;
    using TensorPtr = runtime::ITensor::SharedPtr;
    using TensorMap = runtime::StringPtrMap<runtime::ITensor>;

    DirectRnnStateBuffers(SizeType32 maxBatchSize, runtime::TllmRuntime const& tllmRuntime,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig);

    void reshape(SizeType32 numSequences);

    void setFromInputs(SizeType32 numContextRequests, SizeType32 numSequences, runtime::ITensor const& seqSlotsHost,
        rnn_state_manager::DirectRnnStateManager const& directRnnStateManager,
        runtime::TllmRuntime const& tllmRuntime);

    void commitOutputs(SizeType32 numSequences, runtime::ITensor const& seqSlotsHost,
        rnn_state_manager::DirectRnnStateManager& directRnnStateManager,
        runtime::TllmRuntime const& tllmRuntime);

    void getBuffers(TensorMap& inputBuffers, TensorMap& outputBuffers) const;

private:
    [[nodiscard]] bool tryEnableContiguousGenerationStateViews(SizeType32 numContextRequests, SizeType32 numSequences,
        runtime::ITensor const& seqSlotsHost, rnn_state_manager::DirectRnnStateManager const& directRnnStateManager);

    void resetStateViews();

    void copyRows(runtime::ITensor const& srcTensor, runtime::ITensor& dstTensor, runtime::ITensor const& seqSlotsHost,
        SizeType32 batchStart, SizeType32 numRows, bool gatherFromStateStore,
        runtime::TllmRuntime const& tllmRuntime);

    std::vector<SizeType32> mGlobalLayerNumsPerPP;
    std::vector<TensorPtr> mPastConvStates;
    std::vector<TensorPtr> mPresentConvStates;
    std::vector<TensorPtr> mRnnStates;
    TensorPtr mCopySrcOffsetsHost;
    TensorPtr mCopySrcOffsetsDevice;
    TensorPtr mCopyDstOffsetsHost;
    TensorPtr mCopyDstOffsetsDevice;
    TensorPtr mCopySizesHost;
    TensorPtr mCopySizesDevice;
    std::vector<TensorPtr> mPastConvStateViews;
    std::vector<TensorPtr> mRnnStateViews;
    bool mUseContiguousGenerationStateViews{false};
    SizeType32 mContiguousGenerationStateOffset{0};
};

} // namespace tensorrt_llm::batch_manager
