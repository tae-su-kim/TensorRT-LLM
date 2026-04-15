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

from typing import List, Optional, Tuple

from ...functional import LayerNormType
from ...layers import (Attention, AttentionMaskType, AttentionParams,
                       Embedding, GatedMLP, KeyValueCacheParams, RmsNorm)
from ...module import Module, ModuleList
from .config import QWenConfig
from .gated_delta import Qwen3_5GatedDeltaNet


def expand_qwen3_5_layer_types(config: QWenConfig) -> List[str]:
    if not config.layer_types:
        return ["attention"] * config.num_hidden_layers

    layer_types = list(config.layer_types)
    if len(layer_types) < config.num_hidden_layers:
        repeat_count = (config.num_hidden_layers + len(layer_types) -
                        1) // len(layer_types)
        layer_types = (layer_types * repeat_count)[:config.num_hidden_layers]
    else:
        layer_types = layer_types[:config.num_hidden_layers]
    return layer_types


def qwen3_5_attention_layer_indices(config: QWenConfig) -> List[int]:
    return [
        layer_idx
        for layer_idx, layer_type in enumerate(expand_qwen3_5_layer_types(config))
        if layer_type == "attention"
    ]


class QWen3_5DecoderLayer(Module):

    def __init__(self, config: QWenConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.layer_type = expand_qwen3_5_layer_types(config)[layer_idx]

        dtype = config.dtype
        self.input_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                       eps=config.norm_epsilon,
                                       dtype=dtype)
        self.post_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                      eps=config.norm_epsilon,
                                      dtype=dtype)

        if self.layer_type == "recurrent":
            self.linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_key_heads=config.linear_num_key_heads,
                num_value_heads=config.linear_num_value_heads,
                key_head_dim=config.linear_key_head_dim,
                value_head_dim=config.linear_value_head_dim,
                conv_kernel_size=config.linear_conv_kernel_dim,
                rms_norm_eps=config.norm_epsilon,
                dtype=dtype,
                tp_group=config.mapping.tp_group,
                tp_size=config.mapping.tp_size,
            )
        else:
            local_attn_layer_idx = qwen3_5_attention_layer_indices(
                config).index(layer_idx)
            self.attention = Attention(
                local_layer_idx=local_attn_layer_idx,
                hidden_size=config.hidden_size,
                attention_head_size=config.head_size,
                num_attention_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                max_seqlen_for_logn_scaling=config.seq_length,
                max_position_embeddings=config.max_position_embeddings,
                dtype=dtype,
                attention_mask_type=AttentionMaskType.causal,
                bias=config.attn_bias,
                position_embedding_type=config.position_embedding_type,
                rotary_embedding_base=config.rotary_base,
                rotary_embedding_scaling=config.rotary_scaling,
                tp_rank=config.mapping.tp_rank,
                tp_group=config.mapping.tp_group,
                tp_size=config.mapping.tp_size,
                cp_rank=config.mapping.cp_rank,
                cp_size=config.mapping.cp_size,
                cp_group=config.mapping.cp_group,
                quant_mode=config.quant_mode,
                use_logn_scaling=config.use_logn_attn,
                dense_bias=False,
                qk_layernorm=True,
                layernorm_type=LayerNormType.RmsNorm,
                attn_output_gate=config.attn_output_gate,
            )

        self.mlp = GatedMLP(hidden_size=config.hidden_size,
                            ffn_hidden_size=config.intermediate_size,
                            hidden_act=config.hidden_act,
                            dtype=dtype,
                            bias=config.mlp_bias,
                            tp_group=config.mapping.tp_group,
                            tp_size=config.mapping.tp_size,
                            quant_mode=config.quant_mode)

    def forward(
        self,
        hidden_states,
        use_cache: bool = False,
        attention_mask=None,
        kv_cache_params: Optional[KeyValueCacheParams] = None,
        attention_params: Optional[AttentionParams] = None,
        conv_state=None,
        rnn_state=None,
        lora_layer_params=None,
        spec_decoding_params=None,
        mrope_params=None,
    ) -> Tuple:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        present_kv, present_conv, present_rnn = None, None, None
        if self.layer_type == "recurrent":
            host_request_types = None
            host_context_lengths = None
            if attention_params is not None:
                host_request_types = attention_params.host_request_types
                host_context_lengths = attention_params.host_context_lengths
            hidden_states, present_conv, present_rnn = self.linear_attn(
                hidden_states,
                conv_state,
                rnn_state,
                host_request_types=host_request_types,
                host_context_lengths=host_context_lengths)
        else:
            hidden_states = self.attention(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                spec_decoding_params=spec_decoding_params,
                kv_cache_params=kv_cache_params,
                attention_params=attention_params,
                lora_layer_params=lora_layer_params,
                mrope_params=mrope_params,
            )
            if use_cache:
                hidden_states, present_kv = hidden_states

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states,
                                 lora_layer_params=lora_layer_params)
        hidden_states = residual + hidden_states

        return hidden_states, present_kv, present_conv, present_rnn


class QWen3_5Model(Module):

    def __init__(self, config: QWenConfig) -> None:
        super().__init__()
        if config.mapping.pp_size != 1:
            raise NotImplementedError(
                "Legacy TensorRT Qwen3.5 support currently requires pp_size == 1."
            )

        self.vocab_embedding = ModuleList([
        ])  # placate linting about conditional attribute assignment
        self.vocab_embedding = None
        self.ln_f = None

        if config.mapping.is_first_pp_rank():
            self.vocab_embedding = Embedding(config.vocab_size,
                                             config.hidden_size,
                                             dtype=config.dtype)

        self.layers = ModuleList([
            QWen3_5DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])

        if config.mapping.is_last_pp_rank():
            self.ln_f = RmsNorm(normalized_shape=config.hidden_size,
                                eps=config.norm_epsilon,
                                dtype=config.dtype)

    def forward(
        self,
        input_ids,
        use_cache: bool = False,
        attention_mask=None,
        kv_cache_params: Optional[KeyValueCacheParams] = None,
        attention_params: Optional[AttentionParams] = None,
        conv_states=None,
        rnn_states=None,
        hidden_states=None,
        prompt_embedding_table=None,
        prompt_tasks=None,
        prompt_vocab_size=None,
        lora_params=None,
        spec_decoding_params=None,
        mrope_params=None,
    ):
        ptuning_args = [
            prompt_embedding_table, prompt_tasks, prompt_vocab_size
        ] if prompt_embedding_table is not None else []
        hidden_states = self.vocab_embedding(input_ids, *ptuning_args)

        present_kvs, present_convs, present_rnns = [], [], []
        for layer_idx, layer in enumerate(self.layers):
            lora_layer_params = None
            if lora_params is not None and lora_params.lora_ranks is not None:
                lora_layer_params = lora_params.get_layer_params(layer_idx)

            hidden_states, present_kv, present_conv, present_rnn = layer(
                hidden_states,
                use_cache=use_cache,
                attention_mask=attention_mask,
                kv_cache_params=KeyValueCacheParams(
                    past_key_value=[kv_cache_params.past_key_value[layer_idx]],
                    host_past_key_value_lengths=kv_cache_params.
                    host_past_key_value_lengths,
                    host_max_attention_window_sizes=kv_cache_params.
                    host_max_attention_window_sizes,
                    host_sink_token_length=kv_cache_params.
                    host_sink_token_length,
                    kv_cache_block_offsets=kv_cache_params.
                    kv_cache_block_offsets,
                    host_kv_cache_block_offsets=kv_cache_params.
                    host_kv_cache_block_offsets,
                    host_kv_cache_pool_pointers=kv_cache_params.
                    host_kv_cache_pool_pointers,
                    host_kv_cache_pool_mapping=kv_cache_params.
                    host_kv_cache_pool_mapping,
                    cache_indirection=kv_cache_params.cache_indirection,
                ),
                attention_params=attention_params,
                conv_state=conv_states[layer_idx],
                rnn_state=rnn_states[layer_idx],
                lora_layer_params=lora_layer_params,
                spec_decoding_params=spec_decoding_params,
                mrope_params=mrope_params,
            )
            present_kvs.append(present_kv)
            present_convs.append(present_conv)
            present_rnns.append(present_rnn)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states, tuple(present_kvs), tuple(present_convs), tuple(
            present_rnns)
