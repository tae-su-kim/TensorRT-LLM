# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.qwen3_5_weight_mapper import \
    Qwen3_5MoeHfWeightMapper
from tensorrt_llm.mapping import Mapping


class _DummyModel(torch.nn.Module):

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config

    @property
    def config(self):
        return self.model_config.pretrained_config


def _make_dense_qwen3_5_model_config() -> ModelConfig:
    pretrained_config = SimpleNamespace(
        architectures=["Qwen3_5ForCausalLM"],
        model_type="qwen3_5_text",
        num_hidden_layers=2,
        num_experts=0,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        torch_dtype=torch.bfloat16,
    )
    return ModelConfig(pretrained_config=pretrained_config,
                       mapping=Mapping(world_size=1, rank=0, tp_size=1))


def test_dense_qwen3_5_mapper_expands_dense_mlp_namespace():
    model_config = _make_dense_qwen3_5_model_config()
    mapper = Qwen3_5MoeHfWeightMapper()
    mapper.init_model_and_config(_DummyModel(model_config), model_config)

    weights = {
        "model.language_model.layers.0.mlp.gate_proj.weight":
        torch.randn(8, 8),
        "model.language_model.layers.0.mlp.up_proj.weight":
        torch.randn(8, 8),
        "model.language_model.layers.0.mlp.down_proj.weight":
        torch.randn(8, 8),
        "mtp.layers.0.mlp.gate_proj.weight":
        torch.randn(8, 8),
        "mtp.layers.0.mlp.up_proj.weight":
        torch.randn(8, 8),
        "mtp.layers.0.mlp.down_proj.weight":
        torch.randn(8, 8),
    }

    remapped = mapper.preprocess_weights(weights)

    assert "model.layers.0.mlp.mlp.gate_proj.weight" in remapped
    assert "model.layers.0.mlp.mlp.up_proj.weight" in remapped
    assert "model.layers.0.mlp.mlp.down_proj.weight" in remapped
    assert "model.layers.2.mlp.mlp.gate_proj.weight" in remapped
    assert "model.layers.2.mlp.mlp.up_proj.weight" in remapped
    assert "model.layers.2.mlp.mlp.down_proj.weight" in remapped

    assert "model.layers.0.mlp.gate_proj.weight" not in remapped
    assert "model.layers.2.mlp.gate_proj.weight" not in remapped
