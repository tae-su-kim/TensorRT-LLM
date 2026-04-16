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
from typing import Optional, Union
from types import SimpleNamespace

from ...layers import MoeConfig
from ...mapping import Mapping
from ..convert_utils import infer_dtype
from ..modeling_utils import PretrainedConfig, QuantConfig


def _normalize_qwen3_5_layer_type(layer_type: str) -> str:
    if layer_type == 'linear_attention':
        return 'recurrent'
    if layer_type == 'full_attention':
        return 'attention'
    return layer_type


def _normalize_qwen3_5_layer_types(layer_types: list[str]) -> list[str]:
    return [_normalize_qwen3_5_layer_type(layer_type) for layer_type in layer_types]


def _dict_to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(
            **{key: _dict_to_namespace(val)
               for key, val in value.items()})
    if isinstance(value, list):
        return [_dict_to_namespace(item) for item in value]
    return value


class QWenConfig(PretrainedConfig):

    def __init__(self,
                 *,
                 mlp_bias: bool = False,
                 attn_bias: bool = True,
                 rotary_base: float = 10000.0,
                 rotary_scaling: Optional[dict] = None,
                 disable_weight_only_quant_plugin: bool = False,
                 use_logn_attn: bool = False,
                 moe: Optional[Union[MoeConfig, dict]] = None,
                 num_labels: int = 1,
                 mlp_only_layers: Optional[list] = None,
                 decoder_sparse_step: int = 1,
                 layer_types: Optional[list] = None,
                 linear_num_key_heads: Optional[int] = None,
                 linear_num_value_heads: Optional[int] = None,
                 linear_key_head_dim: Optional[int] = None,
                 linear_value_head_dim: Optional[int] = None,
                 linear_conv_kernel_dim: Optional[int] = None,
                 attn_output_gate: bool = False,
                 text_config_model_type: Optional[str] = None,
                 **kwargs):
        self.mlp_bias = mlp_bias
        self.attn_bias = attn_bias
        self.rotary_base = rotary_base
        self.rotary_scaling = rotary_scaling
        self.disable_weight_only_quant_plugin = disable_weight_only_quant_plugin
        self.num_labels = num_labels
        self.use_logn_attn = use_logn_attn
        self.mlp_only_layers = mlp_only_layers or []
        self.decoder_sparse_step = decoder_sparse_step
        self.layer_types = layer_types or []
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.attn_output_gate = attn_output_gate
        self.text_config_model_type = text_config_model_type
        if moe is None:
            # Legacy MOE config fields
            moe = MoeConfig(num_experts=kwargs.pop('moe_num_experts', 0),
                            top_k=kwargs.pop('moe_top_k', 0),
                            normalization_mode=kwargs.pop(
                                'moe_normalization_mode',
                                MoeConfig.ExpertScaleNormalizationMode.NONE))
        elif isinstance(moe, dict):
            moe = MoeConfig.from_dict(moe)
        assert isinstance(moe, MoeConfig)
        self.moe = moe.validate()

        super().__init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        # Serialize the fields added in QWenConfig
        output['mlp_bias'] = self.mlp_bias
        output['attn_bias'] = self.attn_bias
        output['rotary_base'] = self.rotary_base
        output['rotary_scaling'] = self.rotary_scaling
        output[
            'disable_weight_only_quant_plugin'] = self.disable_weight_only_quant_plugin
        output['use_logn_attn'] = self.use_logn_attn
        output['mlp_only_layers'] = self.mlp_only_layers
        output['decoder_sparse_step'] = self.decoder_sparse_step
        output['layer_types'] = self.layer_types
        output['linear_num_key_heads'] = self.linear_num_key_heads
        output['linear_num_value_heads'] = self.linear_num_value_heads
        output['linear_key_head_dim'] = self.linear_key_head_dim
        output['linear_value_head_dim'] = self.linear_value_head_dim
        output['linear_conv_kernel_dim'] = self.linear_conv_kernel_dim
        output['attn_output_gate'] = self.attn_output_gate
        output['text_config_model_type'] = self.text_config_model_type
        output['moe'] = self.moe.to_dict()
        return output

    @property
    def is_hybrid_linear_attention_model(self) -> bool:
        return self.qwen_type == 'qwen3_5' or any(
            layer_type in ('linear_attention', 'recurrent')
            for layer_type in self.layer_types)

    @classmethod
    def from_hugging_face(cls,
                          hf_config_or_dir: Union[
                              str, 'transformers.PretrainedConfig'],
                          dtype: str = 'auto',
                          mapping: Optional[Mapping] = None,
                          quant_config: Optional[QuantConfig] = None,
                          **kwargs) -> "QWenConfig":
        import transformers
        trust_remote_code = kwargs.pop('trust_remote_code', True)

        if isinstance(hf_config_or_dir, transformers.PretrainedConfig):
            hf_config = hf_config_or_dir
        else:
            hf_config_dir = str(hf_config_or_dir)
            try:
                hf_config = transformers.AutoConfig.from_pretrained(
                    hf_config_dir, trust_remote_code=trust_remote_code)
            except Exception as exc:
                config_dict, _ = transformers.PretrainedConfig.get_config_dict(
                    hf_config_dir, trust_remote_code=trust_remote_code)
                if config_dict.get('model_type') != 'qwen3_5':
                    raise exc
                hf_config = _dict_to_namespace(config_dict)
        if hasattr(hf_config, 'llm_config'):
            hf_config = hf_config.llm_config

        parent_architecture = None
        if getattr(hf_config, 'architectures', None):
            parent_architecture = hf_config.architectures[0]

        qwen_type = hf_config.model_type
        # lmms llava onevision qwen
        if qwen_type == 'llava':
            qwen_type = 'qwen2'
        if hf_config.architectures and hf_config.architectures[
                0] == 'LlavaQwenForCausalLM':
            hf_config.architectures[0] = 'Qwen2ForCausalLM'
        # hf llava onevision qwen
        if qwen_type == 'llava_onevision':
            hf_config = hf_config.text_config
            qwen_type = f'{hf_config.model_type}_llava_onevision'
        # Qwen2-Audio
        if qwen_type == 'qwen2_audio':
            hf_config = hf_config.text_config
            hf_config.architectures = ['Qwen2ForCausalLM']
        # Qwen3.5 dense uses a multimodal top-level config with the text backbone
        # stored under text_config.
        if qwen_type == 'qwen3_5':
            if not hasattr(hf_config, 'text_config'):
                raise ValueError("Qwen3.5 config is missing text_config")
            hf_config = hf_config.text_config
            if not getattr(hf_config, 'architectures', None):
                hf_config.architectures = [parent_architecture
                                           or 'Qwen3_5ForConditionalGeneration']

        valid_types = ('qwen', 'qwen2', 'qwen2_moe', 'qwen2_llava_onevision',
                       'qwen2_vl', 'qwen2_audio', 'qwen3', 'qwen3_moe',
                       'qwen3_5')
        assert qwen_type in valid_types, f"Unsupported Qwen type: {qwen_type}, only {valid_types} are acceptable."
        num_key_value_heads = getattr(hf_config, "num_key_value_heads",
                                      hf_config.num_attention_heads)
        head_dim = getattr(
            hf_config, "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads)
        head_size = getattr(hf_config, "kv_channels", head_dim)
        hidden_act = getattr(hf_config, "hidden_act", "silu")
        if qwen_type in ("qwen2_moe", "qwen3_moe"):
            hidden_act = "swiglu"

        # Qwen3 models have no attention bias, while legacy models have bias
        if qwen_type in ('qwen3', 'qwen3_moe', 'qwen3_5'):
            attn_bias = False  # Qwen3 models have no attn bias
        else:
            attn_bias = True  # Legacy Qwen models have attn bias
        rotary_scaling = getattr(hf_config, "rope_scaling", None)
        seq_length = getattr(hf_config, "seq_length", 8192)
        use_logn_attn = getattr(hf_config, "use_logn_attn", False)
        disable_weight_only_quant_plugin = kwargs.pop(
            'disable_weight_only_quant_plugin', False)
        if qwen_type == "qwen":
            rms_norm_eps = hf_config.layer_norm_epsilon
            rotary_base = getattr(hf_config, "rotary_emb_base", 10000.0)
        else:
            rms_norm_eps = hf_config.rms_norm_eps
            rotary_base = getattr(hf_config, "rope_theta", 100000.0)

        num_labels = 1
        if hf_config.architectures[0] == "Qwen2ForSequenceClassification":
            num_labels = hf_config.num_labels

        moe_num_experts = getattr(hf_config, "num_experts", 0)
        moe_top_k = getattr(hf_config, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(hf_config, "moe_intermediate_size", 0)
        moe_shared_expert_intermediate_size = getattr(
            hf_config, "shared_expert_intermediate_size", 0)
        moe_normalization_mode = MoeConfig.ExpertScaleNormalizationMode.NONE

        # Add support for mlp_only_layers and decoder_sparse_step (Qwen3 MoE)
        mlp_only_layers = getattr(hf_config, "mlp_only_layers", [])
        decoder_sparse_step = getattr(hf_config, "decoder_sparse_step", 1)
        layer_types = getattr(hf_config, "layer_types", [])
        linear_num_key_heads = getattr(hf_config, "linear_num_key_heads", None)
        linear_num_value_heads = getattr(hf_config, "linear_num_value_heads",
                                         None)
        linear_key_head_dim = getattr(hf_config, "linear_key_head_dim", None)
        linear_value_head_dim = getattr(hf_config, "linear_value_head_dim",
                                        None)
        linear_conv_kernel_dim = getattr(hf_config, "linear_conv_kernel_dim",
                                         None)
        attn_output_gate = getattr(hf_config, "attn_output_gate", False)
        dtype = infer_dtype(dtype, getattr(hf_config, 'torch_dtype', None))
        qwen3_5_recurrent_kwargs = {}
        if qwen_type == 'qwen3_5':
            required_fields = {
                'linear_num_key_heads': linear_num_key_heads,
                'linear_num_value_heads': linear_num_value_heads,
                'linear_key_head_dim': linear_key_head_dim,
                'linear_value_head_dim': linear_value_head_dim,
                'linear_conv_kernel_dim': linear_conv_kernel_dim,
            }
            missing_fields = [
                field_name for field_name, value in required_fields.items()
                if value is None
            ]
            if missing_fields:
                raise ValueError(
                    "Qwen3.5 config is missing required linear attention "
                    f"fields: {missing_fields}")
            layer_types = _normalize_qwen3_5_layer_types(layer_types)
            state_dtype = dtype if dtype in ('bfloat16', 'float16') else 'float32'
            qwen3_5_recurrent_kwargs = {
                'conv_kernel':
                linear_conv_kernel_dim,
                'rnn_hidden_size':
                linear_num_value_heads * linear_value_head_dim,
                'rnn_head_size':
                linear_value_head_dim,
                'rnn_conv_dim_size':
                linear_num_key_heads * linear_key_head_dim * 2 +
                linear_num_value_heads * linear_value_head_dim,
                'state_size':
                linear_key_head_dim,
                'state_dtype':
                state_dtype,
            }

        moe_config = MoeConfig(num_experts=moe_num_experts,
                               top_k=moe_top_k,
                               normalization_mode=moe_normalization_mode)
        moe_config.validate()
        tie_word_embeddings = getattr(hf_config, 'tie_word_embeddings', False)

        if qwen_type == 'qwen2_vl':
            pe_type = 'mrope'
            rotary_embedding_percentage = getattr(hf_config, 'rotary_pct', 1.0)
            rotary_embedding_dim = getattr(
                hf_config, 'rotary_dim',
                int(hf_config.hidden_size / hf_config.num_attention_heads *
                    rotary_embedding_percentage))
            rotary_scaling['type'] = 'mrope'
        else:
            pe_type = 'rope_gpt_neox'
            rotary_embedding_dim = None

        return cls(
            architecture=hf_config.architectures[0],
            dtype=dtype,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_key_value_heads=num_key_value_heads,
            head_size=head_size,
            vocab_size=hf_config.vocab_size,
            position_embedding_type=pe_type,
            max_position_embeddings=hf_config.max_position_embeddings,
            rotary_embedding_dim=rotary_embedding_dim,
            hidden_act=hidden_act,
            norm_epsilon=rms_norm_eps,
            attn_bias=attn_bias,
            rotary_base=rotary_base,
            rotary_scaling=rotary_scaling,
            disable_weight_only_quant_plugin=disable_weight_only_quant_plugin,
            seq_length=seq_length,
            use_logn_attn=use_logn_attn,
            qwen_type=qwen_type,
            moe_intermediate_size=moe_intermediate_size,
            moe_shared_expert_intermediate_size=
            moe_shared_expert_intermediate_size,
            mlp_only_layers=mlp_only_layers,
            decoder_sparse_step=decoder_sparse_step,
            layer_types=layer_types,
            linear_num_key_heads=linear_num_key_heads,
            linear_num_value_heads=linear_num_value_heads,
            linear_key_head_dim=linear_key_head_dim,
            linear_value_head_dim=linear_value_head_dim,
            linear_conv_kernel_dim=linear_conv_kernel_dim,
            attn_output_gate=attn_output_gate,
            text_config_model_type=getattr(hf_config, 'model_type', None),
            moe=moe_config,
            mapping=mapping,
            quantization=quant_config,
            num_labels=num_labels,
            tie_word_embeddings=tie_word_embeddings,
            **qwen3_5_recurrent_kwargs,
            **kwargs)
