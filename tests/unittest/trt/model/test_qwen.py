# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tensorrt_llm.llmapi.llm_args import _ModelFormatKind, get_model_format
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import MODEL_MAP
from tensorrt_llm.models.automodel import AutoConfig as TrtAutoConfig
from tensorrt_llm.models.qwen.config import QWenConfig
from tensorrt_llm.models.qwen.model import QWenForCausalLM


def _make_qwen3_5_hf_config():
    text_config = SimpleNamespace(
        model_type="qwen3_5_text",
        architectures=None,
        num_hidden_layers=4,
        num_attention_heads=8,
        hidden_size=1024,
        intermediate_size=2816,
        num_key_value_heads=4,
        head_dim=128,
        vocab_size=151936,
        max_position_embeddings=131072,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        tie_word_embeddings=False,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        attn_output_gate=True,
        torch_dtype=torch.bfloat16,
    )
    return SimpleNamespace(
        model_type="qwen3_5",
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config=text_config,
    )


def _make_qwen3_5_hf_config_dict():
    return {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_text",
            "architectures": None,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "hidden_size": 1024,
            "intermediate_size": 2816,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "vocab_size": 151936,
            "max_position_embeddings": 131072,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": False,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 6,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "attn_output_gate": True,
            "torch_dtype": "bfloat16",
        },
    }


@patch("transformers.AutoConfig.from_pretrained")
def test_qwen3_5_config_from_hf_normalizes_composite_config(mock_from_pretrained):
    mock_from_pretrained.return_value = _make_qwen3_5_hf_config()

    config = QWenConfig.from_hugging_face(
        "fake-qwen3.5",
        dtype="auto",
        mapping=Mapping(world_size=2, rank=0, tp_size=2),
        trust_remote_code=True,
    )

    assert config.architecture == "Qwen3_5ForConditionalGeneration"
    assert config.qwen_type == "qwen3_5"
    assert config.text_config_model_type == "qwen3_5_text"
    assert config.dtype == "bfloat16"
    assert config.attn_bias is False
    assert config.head_size == 128
    assert config.layer_types == [
        "recurrent",
        "recurrent",
        "recurrent",
        "attention",
    ]
    assert config.is_hybrid_linear_attention_model is True
    assert config.linear_num_key_heads == 2
    assert config.linear_num_value_heads == 6
    assert config.linear_key_head_dim == 128
    assert config.linear_value_head_dim == 128
    assert config.linear_conv_kernel_dim == 4
    assert config.attn_output_gate is True
    assert config.conv_kernel == 4
    assert config.rnn_hidden_size == 768
    assert config.rnn_head_size == 128
    assert config.rnn_conv_dim_size == 1280
    assert config.state_size == 128
    assert config.state_dtype == "bfloat16"


def test_qwen3_5_architecture_is_registered_in_model_map():
    assert MODEL_MAP["Qwen3_5ForConditionalGeneration"] is QWenForCausalLM


@patch("transformers.AutoConfig.from_pretrained")
def test_trt_auto_config_supports_qwen3_5(mock_from_pretrained):
    mock_from_pretrained.return_value = _make_qwen3_5_hf_config()

    config = TrtAutoConfig.from_hugging_face("fake-qwen3.5",
                                             trust_remote_code=True)

    assert isinstance(config, QWenConfig)
    assert config.qwen_type == "qwen3_5"
    assert config.architecture == "Qwen3_5ForConditionalGeneration"


@patch("transformers.AutoConfig.from_pretrained",
       side_effect=ValueError("unsupported qwen3_5"))
def test_trt_auto_config_supports_qwen3_5_from_raw_config_json(
        mock_from_pretrained, tmp_path):
    del mock_from_pretrained
    (tmp_path / "config.json").write_text(
        json.dumps(_make_qwen3_5_hf_config_dict()))

    config = TrtAutoConfig.from_hugging_face(tmp_path, trust_remote_code=True)

    assert isinstance(config, QWenConfig)
    assert config.qwen_type == "qwen3_5"
    assert config.architecture == "Qwen3_5ForConditionalGeneration"


@patch("transformers.AutoConfig.from_pretrained",
       side_effect=ValueError("unsupported qwen3_5"))
def test_get_model_format_supports_qwen3_5_from_raw_config_json(
        mock_from_pretrained, tmp_path):
    del mock_from_pretrained
    (tmp_path / "config.json").write_text(
        json.dumps(_make_qwen3_5_hf_config_dict()))

    assert get_model_format(tmp_path,
                            trust_remote_code=True) == _ModelFormatKind.HF


@patch("transformers.AutoConfig.from_pretrained")
def test_qwen3_5_default_plugin_config_disables_legacy_incompatible_features(
        mock_from_pretrained):
    mock_from_pretrained.return_value = _make_qwen3_5_hf_config()

    config = QWenConfig.from_hugging_face("fake-qwen3.5",
                                          trust_remote_code=True)
    model = QWenForCausalLM(config)
    plugin_config = model.default_plugin_config()

    assert plugin_config.remove_input_padding is False
    assert plugin_config.paged_state is False
    assert plugin_config.mamba_conv1d_plugin is None
