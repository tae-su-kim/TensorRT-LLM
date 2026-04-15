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

from collections import OrderedDict
import copy
import os
from typing import Optional, Union

import tensorrt as trt
import torch
from tqdm import tqdm

from ..._common import default_net
from ..._utils import pad_vocab_size, str_dtype_to_trt
from ...functional import (LayerNormType, Tensor, gather_last_token_logits,
                           recv, send)
from ...layers import (MOE, Attention, AttentionMaskType, ColumnLinear,
                       AttentionParams, Embedding, GatedMLP,
                       KeyValueCacheParams, RmsNorm, SharedMoE)
from ...layers.moe import MOEWeightWrapper
from ...logger import logger
from ...lora_helper import (LoraConfig,
                            get_default_trtllm_modules_to_hf_modules, use_lora)
from ...mapping import Mapping
from ...module import Module
from ...quantization import QuantAlgo
from ..generation_mixin import GenerationMixin
from ..model_weights_loader import ModelWeightsLoader
from ..modeling_utils import (DecoderLayerList, DecoderModelForCausalLM,
                              QuantConfig, get_kv_cache_type_from_legacy)
from .config import QWenConfig
from .convert import (load_hf_qwen, load_weights_from_hf_gptq_model,
                      load_weights_from_hf_model)
from .hybrid import (QWen3_5Model, expand_qwen3_5_layer_types,
                     qwen3_5_attention_layer_indices)


def _raise_if_unsupported_legacy_qwen_backend(config: QWenConfig) -> None:
    if config.qwen_type != 'qwen3_5':
        return

    if config.mapping.tp_size != 1:
        raise NotImplementedError(
            "Legacy TensorRT Qwen3.5 support currently requires tp_size == 1."
        )
    if config.mapping.pp_size != 1:
        raise NotImplementedError(
            "Legacy TensorRT Qwen3.5 support currently requires pp_size == 1."
        )


class QWenDecoderLayer(Module):

    def __init__(self, config: QWenConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        dtype = config.dtype
        self.tp_group = config.mapping.tp_group
        self.tp_size = config.mapping.tp_size

        self.input_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                       eps=config.norm_epsilon,
                                       dtype=dtype)

        layers_range = config.mapping.pp_layers(config.num_hidden_layers)
        local_layer_idx = layer_idx - layers_range[0]
        # Qwen3: Enable qk_layernorm for Q/K normalization (similar to Gemma3)
        qk_layernorm = config.qwen_type in ('qwen3', 'qwen3_moe')

        self.attention = Attention(
            local_layer_idx=local_layer_idx,
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
            tp_group=self.tp_group,
            tp_size=self.tp_size,
            cp_rank=config.mapping.cp_rank,
            cp_size=config.mapping.cp_size,
            cp_group=config.mapping.cp_group,
            quant_mode=config.quant_mode,
            use_logn_scaling=config.use_logn_attn,
            dense_bias=False,
            # Qwen3: Add Q/K layer normalization
            qk_layernorm=qk_layernorm,
            layernorm_type=LayerNormType.RmsNorm
            if qk_layernorm else LayerNormType.LayerNorm)

        if config.moe.has_moe():
            mlp_kwargs = {'moe_config': config.moe, 'mapping': config.mapping}
            if config.qwen_type == 'qwen2_moe':
                # Qwen2 MoE uses SharedMoE with shared expert
                ClsMLP = SharedMoE
                mlp_kwargs['use_shared_gate'] = True
                mlp_kwargs['use_side_stream'] = True
                mlp_kwargs['moe_config'].shared_expert_intermediate_size = \
                    config.moe_shared_expert_intermediate_size
            elif config.qwen_type == 'qwen3_moe':
                # Qwen3 MoE uses standard MOE without shared expert
                ClsMLP = MOE
            else:
                ClsMLP = MOE
        else:
            ClsMLP = GatedMLP
            mlp_kwargs = {}

        # Qwen's real inter_size depends on qwen_type
        if self.config.qwen_type == 'qwen':
            intermediate_size = config.intermediate_size // 2
        elif self.config.qwen_type in ('qwen2_moe', 'qwen3_moe'):
            intermediate_size = config.moe_intermediate_size
        else:
            intermediate_size = config.intermediate_size

        self.mlp = ClsMLP(hidden_size=config.hidden_size,
                          ffn_hidden_size=intermediate_size,
                          hidden_act=config.hidden_act,
                          dtype=dtype,
                          bias=config.mlp_bias,
                          tp_group=self.tp_group,
                          tp_size=self.tp_size,
                          quant_mode=config.quant_mode,
                          **mlp_kwargs)
        self.post_layernorm = RmsNorm(normalized_shape=config.hidden_size,
                                      eps=config.norm_epsilon,
                                      dtype=dtype)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask=None,
        use_cache=False,
        spec_decoding_params=None,
        kv_cache_params=None,
        attention_params=None,
        lora_layer_params=None,
        mrope_params=None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_output = self.attention(
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
            attention_output, presents = attention_output

        hidden_states = residual + attention_output

        residual = hidden_states

        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states,
                                 lora_layer_params=lora_layer_params)

        hidden_states = residual + hidden_states
        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class QWenModel(Module):

    def __init__(self, config: QWenConfig) -> None:
        super().__init__()
        self.mapping = config.mapping
        if self.mapping.is_first_pp_rank():
            self.vocab_embedding = Embedding(config.vocab_size,
                                             config.hidden_size,
                                             dtype=config.dtype)

        self.layers = DecoderLayerList(QWenDecoderLayer, config)

        if self.mapping.is_last_pp_rank():
            self.ln_f = RmsNorm(normalized_shape=config.hidden_size,
                                eps=config.norm_epsilon,
                                dtype=config.dtype)

    def forward(self,
                input_ids: Tensor,
                position_ids=None,
                use_cache=False,
                spec_decoding_params=None,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None,
                mrope_params=None,
                hidden_states=None,
                prompt_embedding_table: Optional[Tensor] = None,
                prompt_tasks: Optional[Tensor] = None,
                prompt_vocab_size: Optional[Tensor] = None,
                lora_params=None):

        ptuning_args = [
            prompt_embedding_table, prompt_tasks, prompt_vocab_size
        ] if prompt_embedding_table is not None else []

        if self.mapping.is_first_pp_rank():
            hidden_states = self.vocab_embedding(input_ids, *ptuning_args)
        else:
            hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())

        hidden_states = self.layers.forward(
            hidden_states,
            use_cache=use_cache,
            spec_decoding_params=spec_decoding_params,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            lora_params=lora_params,
            mrope_params=mrope_params)

        if use_cache:
            hidden_states, presents = hidden_states

        if self.mapping.is_last_pp_rank():
            hidden_states = self.ln_f(hidden_states)
        else:
            hidden_states = send(hidden_states, self.mapping.next_pp_rank())

        if use_cache:
            return (hidden_states, tuple(presents))
        return hidden_states


class QWenForCausalLM(DecoderModelForCausalLM):
    config_class = QWenConfig

    def __init__(self, config: QWenConfig):
        transformer = QWen3_5Model(
            config) if config.qwen_type == 'qwen3_5' else QWenModel(config)
        vocab_size_padded = pad_vocab_size(config.vocab_size,
                                           config.mapping.tp_size)

        if config.mapping.is_last_pp_rank():
            if config.architecture == 'Qwen2ForSequenceClassification':
                lm_head = ColumnLinear(config.hidden_size,
                                       config.num_labels,
                                       bias=False,
                                       dtype=config.dtype,
                                       tp_group=config.mapping.tp_group,
                                       tp_size=config.mapping.tp_size,
                                       gather_output=True)
            else:
                lm_head = ColumnLinear(config.hidden_size,
                                       vocab_size_padded,
                                       bias=False,
                                       dtype=config.dtype,
                                       tp_group=config.mapping.tp_group,
                                       tp_size=config.mapping.tp_size,
                                       gather_output=True)
        else:
            lm_head = None
        self.quant_mode = config.quant_mode
        self.mapping = config.mapping
        if config.qwen_type == 'qwen':
            self.trtllm_modules_to_hf_modules = {
                "attn_qkv": "c_attn",
                "attn_dense": "attn.c_proj",
                "mlp_h_to_4h": "w2",
                "mlp_4h_to_h": "mlp.c_proj",
                "mlp_gate": "w1",
            }
        elif config.qwen_type in ('qwen2_moe', 'qwen3_moe'):
            self.trtllm_modules_to_hf_modules = copy.copy(
                get_default_trtllm_modules_to_hf_modules())
            # Common MoE expert mappings for both Qwen2 and Qwen3 MoE
            self.trtllm_modules_to_hf_modules.update({
                "moe_h_to_4h":
                "mlp.experts.gate_proj",
                "moe_4h_to_h":
                "mlp.experts.down_proj",
                "moe_gate":
                "mlp.experts.up_proj",
            })
            # Qwen2 MoE additionally has shared expert
            if config.qwen_type == 'qwen2_moe':
                self.trtllm_modules_to_hf_modules.update({
                    "mlp_h_to_4h":
                    "mlp.shared_expert.gate_proj",
                    "mlp_4h_to_h":
                    "mlp.shared_expert.down_proj",
                    "mlp_gate":
                    "mlp.shared_expert.up_proj",
                    "mlp_router":
                    "mlp.shared_expert_gate",
                })
        else:
            self.trtllm_modules_to_hf_modules = None
        self.gather_context_logits = False
        self.layer_types = expand_qwen3_5_layer_types(
            config) if config.qwen_type == 'qwen3_5' else []
        super().__init__(config, transformer, lm_head)

    def forward(self,
                input_ids: Tensor,
                position_ids=None,
                use_cache=False,
                last_token_ids=None,
                last_token_ids_for_logits=None,
                attention_mask=None,
                kv_cache_params=None,
                attention_params=None,
                conv_states=None,
                rnn_states=None,
                mrope_params=None,
                hidden_states=None,
                prompt_embedding_table: Optional[Tensor] = None,
                prompt_tasks: Optional[Tensor] = None,
                prompt_vocab_size: Optional[Tensor] = None,
                lora_params=None,
                spec_decoding_params=None):
        if self.config.qwen_type != 'qwen3_5':
            return super().forward(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=use_cache,
                last_token_ids=last_token_ids,
                attention_mask=attention_mask,
                kv_cache_params=kv_cache_params,
                attention_params=attention_params,
                mrope_params=mrope_params,
                hidden_states=hidden_states,
                prompt_embedding_table=prompt_embedding_table,
                prompt_tasks=prompt_tasks,
                prompt_vocab_size=prompt_vocab_size,
                lora_params=lora_params,
                spec_decoding_params=spec_decoding_params)

        attention_params = Attention.fill_attention_params(
            self, attention_params)
        hidden_states, present_kvs, present_convs, present_rnns = self.transformer(
            input_ids=input_ids,
            use_cache=use_cache,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            conv_states=conv_states,
            rnn_states=rnn_states,
            hidden_states=hidden_states,
            prompt_embedding_table=prompt_embedding_table,
            prompt_tasks=prompt_tasks,
            prompt_vocab_size=prompt_vocab_size,
            lora_params=lora_params,
            spec_decoding_params=spec_decoding_params,
            mrope_params=mrope_params)

        if not self.gather_context_logits:
            if last_token_ids_for_logits is None:
                last_token_ids_for_logits = last_token_ids
            hidden_states = gather_last_token_logits(
                hidden_states, last_token_ids_for_logits,
                default_net().plugin_config.remove_input_padding)

        lm_logits = self.lm_head(hidden_states)
        lm_logits.mark_output('logits', self.config.logits_dtype)

        if not default_net().plugin_config.paged_kv_cache:
            for i, present_kv in enumerate(present_kvs):
                if present_kv is not None:
                    present_kv.mark_output(f'present_key_value_{i}',
                                           self.config.kv_dtype)

        if not default_net().plugin_config.paged_state:
            for i, present_conv in enumerate(present_convs):
                if present_conv is not None:
                    present_conv.mark_output(f'present_conv_state_{i}',
                                             self.config.dtype)
            for i, present_rnn in enumerate(present_rnns):
                if present_rnn is not None:
                    present_rnn.mark_output(f'present_rnn_state_{i}',
                                            str_dtype_to_trt(
                                                self.config.state_dtype))

        return (lm_logits, present_kvs, present_convs, present_rnns)

    def prepare_recurrent_inputs(self, max_batch_size, num_profiles,
                                 mapping: Mapping):
        del mapping
        conv_states = []
        rnn_states = []
        batch_range = [GenerationMixin.default_range(max_batch_size)
                       ] * num_profiles
        conv_state_dim_range = OrderedDict([
            ('batch_size', batch_range),
            ('dim_size', [self.config.rnn_conv_dim_size] * num_profiles),
            ('kernel_size', [self.config.conv_kernel - 1] * num_profiles),
        ])
        rnn_state_dim_range = OrderedDict([
            ('batch_size', batch_range),
            ('head_size',
             [self.config.rnn_hidden_size // self.config.rnn_head_size] *
             num_profiles),
            ('value_size', [self.config.rnn_head_size] * num_profiles),
            ('state_size', [self.config.state_size] * num_profiles),
        ])

        for layer_idx in range(self.config.num_hidden_layers):
            if self.layer_types[layer_idx] == 'recurrent':
                conv_state = Tensor(
                    name=f'past_conv_state_{layer_idx}',
                    dtype=self.config.dtype,
                    shape=[-1, self.config.rnn_conv_dim_size,
                           self.config.conv_kernel - 1],
                    dim_range=conv_state_dim_range)
                rnn_state = Tensor(
                    name=f'past_rnn_state_{layer_idx}',
                    dtype=str_dtype_to_trt(self.config.state_dtype),
                    shape=[
                        -1,
                        self.config.rnn_hidden_size // self.config.rnn_head_size,
                        self.config.rnn_head_size,
                        self.config.state_size,
                    ],
                    dim_range=rnn_state_dim_range)
            else:
                conv_state = None
                rnn_state = None
            conv_states.append(conv_state)
            rnn_states.append(rnn_state)

        return {
            'conv_states': conv_states,
            'rnn_states': rnn_states,
        }

    def prepare_inputs(
            self,
            max_batch_size,
            max_input_len,
            max_seq_len,
            max_num_tokens,
            use_cache,
            max_beam_width: int = 1,
            opt_num_tokens: int = None,
            opt_batch_size: int = 0,
            prompt_embedding_table_size: int = 0,
            max_draft_len: int = 0,
            gather_context_logits: bool = False,
            lora_target_modules: list[str] = None,
            speculative_decoding_draft_tokens_external: bool = False):
        if self.config.qwen_type != 'qwen3_5':
            return super().prepare_inputs(
                max_batch_size=max_batch_size,
                max_input_len=max_input_len,
                max_seq_len=max_seq_len,
                max_num_tokens=max_num_tokens,
                use_cache=use_cache,
                max_beam_width=max_beam_width,
                opt_num_tokens=opt_num_tokens,
                opt_batch_size=opt_batch_size,
                prompt_embedding_table_size=prompt_embedding_table_size,
                max_draft_len=max_draft_len,
                gather_context_logits=gather_context_logits,
                lora_target_modules=lora_target_modules,
                speculative_decoding_draft_tokens_external=
                speculative_decoding_draft_tokens_external)

        del prompt_embedding_table_size
        del lora_target_modules
        assert speculative_decoding_draft_tokens_external == False, \
            "We don't support speculative decoding for the Qwen3.5 legacy TensorRT model."
        assert max_beam_width == 1, \
            "We don't support beam search for the Qwen3.5 legacy TensorRT model."

        remove_input_padding = default_net().plugin_config.remove_input_padding
        use_gpt_attention_plugin = default_net(
        ).plugin_config.gpt_attention_plugin
        use_gemm_plugin = default_net().plugin_config.gemm_plugin
        paged_kv_cache = default_net().plugin_config.paged_kv_cache
        tokens_per_block = default_net().plugin_config.tokens_per_block
        multiple_profiles = default_net().plugin_config.multiple_profiles
        use_mamba_conv1d_plugin = bool(
            default_net().plugin_config.mamba_conv1d_plugin)

        assert not remove_input_padding, \
            "Qwen3.5 legacy TensorRT support requires remove_input_padding == False."
        assert not default_net().plugin_config.paged_state, \
            "Qwen3.5 legacy TensorRT support requires paged_state == False."
        assert not use_mamba_conv1d_plugin, \
            "Qwen3.5 legacy TensorRT support requires mamba_conv1d_plugin == False."

        self.gather_context_logits = gather_context_logits
        mapping = self.config.mapping
        kv_cache_type = get_kv_cache_type_from_legacy(use_cache,
                                                      paged_kv_cache)

        enable_ctx_gen_opt_profiles = GenerationMixin.has_ctx_gen_opt_profiles(
            use_gpt_attention_plugin=use_gpt_attention_plugin,
            use_gemm_plugin=use_gemm_plugin,
            remove_input_padding=remove_input_padding,
            kv_cache_type=kv_cache_type)
        num_profiles, ranges = GenerationMixin.get_profiles_ranges(
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_input_len=max_input_len,
            max_num_tokens=max_num_tokens,
            max_draft_len=max_draft_len,
            opt_batch_size=opt_batch_size,
            opt_num_tokens=opt_num_tokens,
            enable_ctx_gen_opt_profiles=enable_ctx_gen_opt_profiles,
            multiple_profiles=multiple_profiles,
            kv_cache_type=kv_cache_type)

        input_ids = Tensor(name='input_ids',
                           dtype=trt.int32,
                           shape=[-1, -1],
                           dim_range=OrderedDict([
                               ('batch_size_beam_width', ranges['bb_range']),
                               ('input_len', ranges['inlen_range']),
                           ]))
        position_ids = Tensor(name='position_ids',
                              dtype=trt.int32,
                              shape=[-1, -1],
                              dim_range=OrderedDict([
                                  ('batch_size_beam_width', ranges['bb_range']),
                                  ('position_ids_inlen_range',
                                   ranges['position_ids_inlen_range']),
                              ]))

        attn_layer_idx = qwen3_5_attention_layer_indices(self.config)
        attention_inputs = self.prepare_attention_inputs(
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_input_len=max_input_len,
            max_seq_len=max_seq_len,
            num_kv_heads=self.config.num_key_value_heads,
            head_size=self.config.head_size,
            num_layers=self.config.num_hidden_layers,
            kv_dtype=str_dtype_to_trt(self.config.kv_dtype),
            kv_cache_type=kv_cache_type,
            num_profiles=num_profiles,
            enable_ctx_gen_opt_profiles=enable_ctx_gen_opt_profiles,
            remove_input_padding=remove_input_padding,
            use_gpt_attention_plugin=use_gpt_attention_plugin,
            tokens_per_block=tokens_per_block,
            mapping=mapping,
            attn_layer_idx=attn_layer_idx)

        host_context_lengths = attention_inputs['host_context_lengths']
        if host_context_lengths is None:
            host_context_lengths = Tensor(
                name='host_context_lengths',
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([('batch_size_beam_width',
                                        ranges['bb_range'])]),
            )

        recurrent_inputs = self.prepare_recurrent_inputs(
            max_batch_size=max_batch_size,
            num_profiles=num_profiles,
            mapping=mapping)

        last_token_ids = Tensor(
            name='last_token_ids',
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([
                ('batch_size_last_token_ids', ranges['bbd_range']),
            ]),
        )
        last_token_ids_for_logits = None
        if not gather_context_logits:
            last_token_ids_for_logits = last_token_ids

        return {
            'input_ids':
            input_ids,
            'position_ids':
            position_ids,
            'use_cache':
            True,
            'last_token_ids':
            last_token_ids,
            'last_token_ids_for_logits':
            last_token_ids_for_logits,
            'attention_mask':
            attention_inputs['attention_mask'],
            'kv_cache_params':
            KeyValueCacheParams(
                past_key_value=attention_inputs['past_key_value'],
                host_past_key_value_lengths=attention_inputs[
                    'host_past_key_value_lengths'],
                host_max_attention_window_sizes=attention_inputs[
                    'host_max_attention_window_sizes'],
                host_sink_token_length=attention_inputs[
                    'host_sink_token_length'],
                kv_cache_block_offsets=attention_inputs[
                    'kv_cache_block_offsets'],
                host_kv_cache_block_offsets=attention_inputs[
                    'host_kv_cache_block_offsets'],
                host_kv_cache_pool_pointers=attention_inputs[
                    'host_kv_cache_pool_pointers'],
                host_kv_cache_pool_mapping=attention_inputs[
                    'host_kv_cache_pool_mapping'],
                cache_indirection=attention_inputs['cache_indirection'],
            ),
            'attention_params':
            AttentionParams(
                sequence_length=attention_inputs['sequence_length'],
                context_lengths=attention_inputs['context_lengths'],
                host_context_lengths=host_context_lengths,
                max_context_length=max_input_len,
                host_request_types=attention_inputs['host_request_types'],
                host_runtime_perf_knobs=attention_inputs[
                    'host_runtime_perf_knobs'],
                host_context_progress=attention_inputs['host_context_progress'],
            ),
            'conv_states':
            recurrent_inputs['conv_states'],
            'rnn_states':
            recurrent_inputs['rnn_states'],
        }

    @classmethod
    def from_hugging_face(
            cls,
            hf_model_or_dir: Union[str, 'transformers.PreTrainedModel'],
            dtype: str = 'auto',
            mapping: Optional[Mapping] = None,
            quant_config: Optional[QuantConfig] = None,
            **kwargs):
        ''' Create a QWenForCausalLM object from give parameters
        '''
        import transformers

        load_model_on_cpu = kwargs.pop('load_model_on_cpu', False)
        use_autoawq = kwargs.pop('use_autoawq', False)

        assert hf_model_or_dir is not None
        use_preloading = isinstance(hf_model_or_dir,
                                    transformers.PreTrainedModel)
        if use_preloading:
            hf_model = hf_model_or_dir
            hf_config_or_dir = hf_model.config
        else:
            hf_model_dir = hf_model_or_dir
            hf_config_or_dir = hf_model_or_dir

        config = QWenConfig.from_hugging_face(hf_config_or_dir,
                                              dtype=dtype,
                                              mapping=mapping,
                                              quant_config=quant_config,
                                              **kwargs)
        _raise_if_unsupported_legacy_qwen_backend(config)

        if os.environ.get("TRTLLM_DISABLE_UNIFIED_CONVERTER") is None:
            arg_dict = {"use_autoawq": True} if use_autoawq else {}
            custom_dict = {}

            if config.qwen_type == "qwen":
                custom_dict = {
                    "transformer": "transformer",
                    "vocab_embedding": "wte",
                    "ln_f": "ln_f",
                    "layers": "h",
                    "attention": "attn",
                    "qkv": "c_attn",
                    "dense": "c_proj",
                    "gate": "w1",
                    "proj": "c_proj",
                    "fc": "w2",
                    "input_layernorm": "ln_1",
                    "post_layernorm": "ln_2",
                }
            elif config.qwen_type == "qwen2_moe":
                custom_dict = {
                    "mlp.shared_expert": "mlp.shared_expert",
                    "mlp.shared_expert_gate": "mlp.shared_expert_gate",
                    "fc": ["up_proj", "gate_proj"],
                }
            elif config.qwen_type == "qwen3_moe":
                custom_dict = {
                    "fc": ["up_proj", "gate_proj"],
                    "q_layernorm": "q_norm",
                    "k_layernorm": "k_norm",
                }
            elif config.qwen_type in {"qwen2", "qwen2_vl"
                                      } and config.tie_word_embeddings:
                custom_dict = {"lm_head": "model.embed_tokens"}
            elif config.architecture == "Qwen2ForSequenceClassification":
                custom_dict = {
                    "lm_head": "score",
                }
            elif config.qwen_type == "qwen2_llava_onevision":
                custom_dict = {
                    "transformer": "language_model.model",
                    "lm_head": "language_model.lm_head",
                }
            elif config.qwen_type == "qwen2_audio":
                custom_dict = {
                    "transformer": "language_model.model",
                    "lm_head": "language_model.lm_head",
                }
            elif config.qwen_type == "qwen3":
                custom_dict = {
                    "q_layernorm": "q_norm",
                    "k_layernorm": "k_norm",
                }
            elif config.qwen_type == "qwen3_5":
                custom_dict = {
                    "transformer": "model.language_model",
                    "q_layernorm": "q_norm",
                    "k_layernorm": "k_norm",
                }
            loader = ModelWeightsLoader(hf_model_dir, custom_dict)
            model = cls(config)

            if config.qwen_type == "qwen" and model.config.mapping.has_tp():

                def reshape_qkv(weights):
                    if weights is None:
                        return weights
                    mapping = model.config.mapping
                    unsqueeze = False
                    if isinstance(weights, torch.Tensor):
                        unsqueeze = True
                        weights = [weights]

                    for idx, w in enumerate(weights):
                        if quant_config.quant_algo == QuantAlgo.W4A16_GPTQ:
                            w = w.reshape(-1, 3, w.shape[-1] // 3)
                            w = w.chunk(mapping.tp_size, 2)[mapping.tp_rank]
                            if w.shape[0] == 1:
                                weights[idx] = w.reshape(-1)
                            else:
                                weights[idx] = w.reshape(w.shape[0], -1)
                        else:
                            w = w.reshape(3, w.shape[0] // 3, -1)
                            w = w.chunk(mapping.tp_size, 1)[mapping.tp_rank]
                            if w.shape[-1] == 1:
                                weights[idx] = w.reshape(-1)
                            else:
                                weights[idx] = w.reshape(-1, w.shape[-1])
                    if unsqueeze:
                        return weights[0]
                    else:
                        return weights

                loader.update_key_mapping(model)
                tllm_weights = {}
                for tllm_key, _ in tqdm(model.named_parameters()):
                    if "qkv" in tllm_key:
                        tllm_weights.update(
                            loader.load(tllm_key,
                                        reshape_qkv,
                                        skip_tp=True,
                                        custom_postprocess_kwargs=arg_dict))
                    else:
                        tllm_weights.update(
                            loader.load(tllm_key,
                                        custom_postprocess_kwargs=arg_dict))
                loader.fill(tllm_weights)
            elif config.qwen_type in ("qwen2_moe", "qwen3_moe"):
                for tllm_key, _ in model.named_parameters():
                    sub_module = model
                    for attr in tllm_key.split(".")[:-1]:
                        sub_module = getattr(sub_module, attr)
                    if "router" in tllm_key or isinstance(
                            sub_module, MOEWeightWrapper):
                        sub_module_dic = sub_module.tllm_to_externel_key_dict
                        sub_module_dic["mlp"] = "mlp"
                        if "fc" in sub_module_dic.keys():
                            sub_module_dic["fc"] = [
                                hf_keyword.replace("w1", "gate_proj")
                                for hf_keyword in sub_module_dic["fc"]
                            ]
                            sub_module_dic["fc"] = [
                                hf_keyword.replace("w3", "up_proj")
                                for hf_keyword in sub_module_dic["fc"]
                            ]
                        if "proj" in sub_module_dic.keys():
                            sub_module_dic["proj"] = [
                                hf_keyword.replace("w2", "down_proj")
                                for hf_keyword in sub_module_dic["proj"]
                            ]
                        sub_module.tllm_to_externel_key_dict = sub_module_dic

                def concat_gate_up_proj(weights):
                    return torch.cat(weights, dim=-2)

                loader.update_key_mapping(model)
                tllm_weights = {}
                for tllm_key, _ in tqdm(model.named_parameters()):
                    if tllm_key.endswith("shared_expert.fc.weight"):
                        tllm_weights.update(
                            loader.load(tllm_key,
                                        concat_gate_up_proj,
                                        custom_postprocess_kwargs=arg_dict))
                    else:
                        tllm_weights.update(
                            loader.load(tllm_key,
                                        custom_postprocess_kwargs=arg_dict))
                loader.fill(tllm_weights)
            else:
                # For Qwen1 w/o TP, Qwen1.5 and Qwen2 w/o MoE
                loader.generate_tllm_weights(model, arg_dict)
        else:
            if config.qwen_type == "qwen3_5":
                raise NotImplementedError(
                    "Qwen3.5 legacy TensorRT support requires the unified "
                    "ModelWeightsLoader path.")
            if not use_preloading:
                hf_model = load_hf_qwen(hf_model_dir, load_model_on_cpu)

            logger.debug(f"HuggingFace model: {hf_model}")

            model = QWenForCausalLM(config)
            logger.debug(f"TensorRT LLM model: {model}")

            if quant_config.quant_algo == QuantAlgo.W4A16_GPTQ:
                weights = load_weights_from_hf_gptq_model(hf_model, config)
            else:
                weights = load_weights_from_hf_model(hf_model, config)
            model.load(weights)
        return model

    def default_plugin_config(self, **kwargs):
        if self.config.qwen_type == 'qwen3_5':
            kwargs.setdefault('remove_input_padding', False)
            kwargs.setdefault('paged_state', False)
            kwargs.setdefault('gated_delta_plugin', True)
            kwargs.setdefault('mamba_conv1d_plugin', None)
        plugin_config = super().default_plugin_config(**kwargs)
        if self.quant_mode.is_int4_weight_only_per_group():
            plugin_config.weight_only_groupwise_quant_matmul_plugin = 'auto'
        return plugin_config

    @classmethod
    def quantize(
        cls,
        hf_model_dir: str,
        output_dir: str,
        dtype: str = 'auto',
        mapping: Optional[Mapping] = None,
        quant_config: Optional[QuantConfig] = None,
        *,
        calib_dataset='cnn_dailymail',
        calib_batches=512,
        calib_batch_size=1,
        calib_max_seq_length=512,
        random_seed=1234,
        tokenizer_max_seq_length=2048,
        **kwargs,
    ):
        if quant_config._requires_modelopt_quantization:
            # modelopt quantization flow
            super().quantize(hf_model_dir,
                             output_dir,
                             dtype=dtype,
                             mapping=mapping,
                             quant_config=quant_config,
                             calib_dataset=calib_dataset,
                             calib_batches=calib_batches,
                             calib_batch_size=calib_batch_size,
                             calib_max_seq_length=calib_max_seq_length,
                             random_seed=random_seed,
                             tokenizer_max_seq_length=tokenizer_max_seq_length)
        elif quant_config._requires_calibration:
            # non-modelopt quantization flow
            from . import convert

            config = QWenConfig.from_hugging_face(hf_model_dir,
                                                  dtype=dtype,
                                                  mapping=mapping,
                                                  quant_config=quant_config,
                                                  **kwargs)
            _raise_if_unsupported_legacy_qwen_backend(config)
            convert.quantize(hf_model_dir,
                             output_dir,
                             config=config,
                             calib_dataset=calib_dataset)
        else:
            raise ValueError(
                f"The quant_config ({quant_config}) does not require calibration, try {cls.__name__}.from_hugging_face instead."
            )

    def use_lora(self, lora_config: LoraConfig):
        use_lora(self, lora_config, self.trtllm_modules_to_hf_modules)
