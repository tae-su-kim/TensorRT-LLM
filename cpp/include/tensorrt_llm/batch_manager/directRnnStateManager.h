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

class DirectRnnStateManager
{
public:
    using TensorPtr = runtime::ITensor::SharedPtr;
    using SizeType32 = tensorrt_llm::runtime::SizeType32;

    DirectRnnStateManager(SizeType32 maxNumSequences, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig, runtime::TllmRuntime const& tllmRuntime);

    [[nodiscard]] SizeType32 getNumLocalLayers() const noexcept
    {
        return static_cast<SizeType32>(mGlobalLayerNumsPerPP.size());
    }

    [[nodiscard]] SizeType32 getGlobalLayerNum(SizeType32 localOffset) const;

    [[nodiscard]] TensorPtr getConvStates(SizeType32 localOffset) const;

    [[nodiscard]] TensorPtr getRnnStates(SizeType32 localOffset) const;

    [[nodiscard]] SizeType32 getMaxNumSequences() const noexcept
    {
        return mMaxNumSequences;
    }

    [[nodiscard]] nvinfer1::DataType getConvStateDataType() const noexcept
    {
        return mConvStateDtype;
    }

    [[nodiscard]] nvinfer1::DataType getRnnStateDataType() const noexcept
    {
        return mRnnStateDtype;
    }

private:
    std::vector<SizeType32> mGlobalLayerNumsPerPP;
    std::vector<TensorPtr> mConvStates;
    std::vector<TensorPtr> mRnnStates;
    SizeType32 mMaxNumSequences{0};
    nvinfer1::DataType mConvStateDtype{nvinfer1::DataType::kFLOAT};
    nvinfer1::DataType mRnnStateDtype{nvinfer1::DataType::kFLOAT};
};

} // namespace tensorrt_llm::batch_manager::rnn_state_manager
