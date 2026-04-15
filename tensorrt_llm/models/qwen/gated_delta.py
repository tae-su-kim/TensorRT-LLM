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

from typing import Tuple

import tensorrt as trt

from ..._common import default_net, default_trtnet
from ...functional import (Tensor, _create_tensor, cast, concat, conv2d, einsum,
                           exp, gated_delta_plugin, mean, permute,
                           qwen_causal_conv1d,
                           repeat_interleave, shape, sigmoid,
                           silu, slice, softplus, split, sqrt, sum,
                           unsqueeze)
from ...layers.linear import ColumnLinear, RowLinear
from ...module import Module
from ...parameter import Parameter


def _l2_norm(input: Tensor, eps: float = 1e-6) -> Tensor:
    """Apply per-head L2 normalization on the last dimension."""
    input_fp32 = cast(input, 'float32')
    squared_norm = sum(input_fp32 * input_fp32, dim=-1, keepdim=True)
    return input_fp32 / sqrt(squared_norm + eps)


def _take_last_tokens(input: Tensor, num_tokens: int, dim: int) -> Tensor:
    """Take the final `num_tokens` values along a dynamic dimension."""
    if dim < 0:
        dim += input.ndim()

    starts = []
    sizes = []
    for axis in range(input.ndim()):
        if axis == dim:
            starts.append(shape(input, axis) - num_tokens)
            sizes.append(num_tokens)
        else:
            starts.append(0)
            sizes.append(shape(input, axis))

    return slice(input, concat(starts), concat(sizes))


def _gated_rms_norm(hidden_states: Tensor, gate: Tensor, weight: Tensor,
                    eps: float) -> Tensor:
    """Compute `rms_norm(hidden_states) * weight * silu(gate)` in float32."""
    input_dtype = hidden_states.dtype
    hidden_states_fp32 = cast(hidden_states, 'float32')
    gate_fp32 = cast(gate, 'float32')
    weight_fp32 = cast(weight, 'float32')

    variance = mean(hidden_states_fp32 * hidden_states_fp32,
                    dim=-1,
                    keepdim=True)
    normalized = hidden_states_fp32 / sqrt(variance + eps)
    normalized = normalized * weight_fp32
    normalized = normalized * silu(gate_fp32)
    return cast(normalized, input_dtype)


def _expand_qk_heads(query: Tensor, key: Tensor,
                     num_value_heads: int) -> Tuple[Tensor, Tensor]:
    """Expand grouped query/key heads to the value-head count."""
    num_key_heads = query.shape[2]
    if num_key_heads <= 0:
        raise ValueError("num_key_heads must be a positive static dimension")
    if num_value_heads % num_key_heads != 0:
        raise ValueError(
            "num_value_heads must be divisible by num_key_heads")

    repeat_factor = num_value_heads // num_key_heads
    if repeat_factor > 1:
        query = repeat_interleave(query, repeat_factor, dim=2)
        key = repeat_interleave(key, repeat_factor, dim=2)

    return query, key


def _normalize_and_expand_qk(query: Tensor, key: Tensor,
                             num_value_heads: int) -> Tuple[Tensor, Tensor]:
    """Apply L2 normalization and expand grouped-query heads."""
    query = _l2_norm(query)
    key = _l2_norm(key)
    return _expand_qk_heads(query, key, num_value_heads)


def _compute_gated_delta_g_beta(a: Tensor, b: Tensor, a_log: Tensor,
                                dt_bias: Tensor) -> Tuple[Tensor, Tensor]:
    """Compute the recurrent decay and update gates in float32."""
    decay = -1.0 * exp(cast(a_log, 'float32'))
    gated = softplus(cast(a, 'float32') + cast(dt_bias, 'float32'),
                     beta=1.0,
                     threshold=20.0)
    beta = sigmoid(cast(b, 'float32'))
    g = decay * gated

    return g, beta


def gated_delta_step(query: Tensor, key: Tensor, value: Tensor, g: Tensor,
                     beta: Tensor, state: Tensor,
                     scale: float) -> Tuple[Tensor, Tensor]:
    """Run a single-token gated delta-rule update."""
    decay = exp(unsqueeze(unsqueeze(g, -1), -1))
    state = state * decay

    v_prime = value - einsum('bhk,bhvk->bhv', [key, state])
    v_prime = v_prime * unsqueeze(beta, -1)
    state = state + einsum('bhk,bhv->bhvk', [key, v_prime])
    output = einsum('bhk,bhvk->bhv', [query * scale, state])
    return output, state


def gated_delta_rule(query: Tensor, key: Tensor, value: Tensor,
                     g: Tensor | None, beta: Tensor | None,
                     initial_state: Tensor,
                     scale: float,
                     a_log: Tensor | None = None,
                     a: Tensor | None = None,
                     dt_bias: Tensor | None = None,
                     b: Tensor | None = None,
                     host_request_types: Tensor | None = None,
                     host_context_lengths: Tensor | None = None
                     ) -> Tuple[Tensor, Tensor]:
    """Run the gated delta rule over a full sequence using a TensorRT loop.

    Args:
        query: [B, S, H, K] float32, GQA-expanded for the plugin path.
        key: [B, S, H, K] float32, GQA-expanded for the plugin path.
        value: [B, S, H, V] float32.
        g: [B, S, H] float32 log-space decay values, or `None` when the
            plugin fast path computes gating internally.
        beta: [B, S, H] float32 sigmoid-activated update scale, or `None`
            when the plugin fast path computes gating internally.
        initial_state: [B, H, V, K] float32.
        scale: query scaling factor.

    Returns:
        A tuple of:
          - outputs: [B, S, H, V] float32
          - final_state: [B, H, V, K] float32
    """
    if (default_net().plugin_config.gated_delta_plugin
            and host_request_types is not None
            and host_context_lengths is not None):
        if a_log is None or a is None or dt_bias is None or b is None:
            raise ValueError(
                "raw gating inputs are required when gated_delta_plugin is enabled"
            )
        return gated_delta_plugin(cast(query, value.dtype), cast(key,
                                                                 value.dtype),
                                  value,
                                  cast(initial_state, 'float32'),
                                  cast(a_log, 'float32'), cast(a, 'float32'),
                                  cast(dt_bias, 'float32'), cast(b, 'float32'),
                                  host_request_types, host_context_lengths)

    if g is None or beta is None:
        raise ValueError(
            "precomputed g and beta are required when gated_delta_plugin is disabled"
        )

    query = cast(query, 'float32')
    key = cast(key, 'float32')
    value = cast(value, 'float32')
    g = cast(g, 'float32')
    beta = cast(beta, 'float32')
    initial_state = cast(initial_state, 'float32')

    loop = default_trtnet().add_loop()
    trip_limit = shape(query, 1).trt_tensor
    loop.add_trip_limit(trip_limit, trt.TripLimit.COUNT)

    query_iter = loop.add_iterator(query.trt_tensor, 1)
    key_iter = loop.add_iterator(key.trt_tensor, 1)
    value_iter = loop.add_iterator(value.trt_tensor, 1)
    g_iter = loop.add_iterator(g.trt_tensor, 1)
    beta_iter = loop.add_iterator(beta.trt_tensor, 1)

    query_t = _create_tensor(query_iter.get_output(0), query_iter)
    key_t = _create_tensor(key_iter.get_output(0), key_iter)
    value_t = _create_tensor(value_iter.get_output(0), value_iter)
    g_t = _create_tensor(g_iter.get_output(0), g_iter)
    beta_t = _create_tensor(beta_iter.get_output(0), beta_iter)

    state_recurrence = loop.add_recurrence(initial_state.trt_tensor)
    state_t = _create_tensor(state_recurrence.get_output(0), state_recurrence)

    output_t, next_state = gated_delta_step(query_t, key_t, value_t, g_t,
                                            beta_t, state_t, scale)
    state_recurrence.set_input(1, next_state.trt_tensor)

    output_layer = loop.add_loop_output(output_t.trt_tensor,
                                        trt.LoopOutput.CONCATENATE, 1)
    output_layer.set_input(1, trip_limit)

    state_layer = loop.add_loop_output(state_recurrence.get_output(0),
                                       trt.LoopOutput.LAST_VALUE)
    return (_create_tensor(output_layer.get_output(0), output_layer),
            _create_tensor(state_layer.get_output(0), state_layer))


class Qwen3_5CausalConv1d(Module):
    """Stateful depthwise causal conv used by Qwen3.5 GatedDeltaNet."""

    def __init__(self, channels: int, kernel_size: int, dtype=None) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.weight = Parameter(shape=(channels, 1, kernel_size), dtype=dtype)

    def forward(self,
                input: Tensor,
                past_conv_state: Tensor,
                host_request_types: Tensor | None = None) -> Tuple[Tensor, Tensor]:
        """Run causal depthwise conv on `[B, S, C]` inputs.

        Args:
            input: [B, S, C]
            past_conv_state: [B, C, kernel_size - 1]
            host_request_types: [B] host request-type tensor that enables the
                runtime conv fast path when the recurrent plugin path is active.

        Returns:
            A tuple of:
              - convolved output: [B, S, C]
              - present conv state: [B, C, kernel_size - 1]
        """
        use_plugin = (default_net().plugin_config.gated_delta_plugin
                      and host_request_types is not None)
        if use_plugin:
            return qwen_causal_conv1d(input, past_conv_state, self.weight.value)

        input_t = input.permute([0, 2, 1])
        full_input = concat([past_conv_state, input_t], dim=2)
        present_conv_state = _take_last_tokens(full_input,
                                               self.kernel_size - 1,
                                               dim=2)

        full_input = full_input.view(
            concat([shape(full_input, 0),
                    shape(full_input, 1),
                    shape(full_input, 2), 1]))
        weight = self.weight.value.view([self.channels, 1, self.kernel_size, 1])
        output = conv2d(full_input, weight, groups=self.channels)
        output = output.view(
            concat([shape(output, 0),
                    shape(output, 1),
                    shape(output, 2)]))
        output = silu(output).permute([0, 2, 1])
        return output, present_conv_state


class Qwen3_5RmsNormGated(Module):
    """Qwen3.5 gated RMSNorm with a shared per-head hidden dimension weight."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, dtype=None):
        super().__init__()
        self.weight = Parameter(shape=(hidden_size, ), dtype=dtype)
        self.eps = eps

    def forward(self, hidden_states: Tensor, gate: Tensor) -> Tensor:
        return _gated_rms_norm(hidden_states, gate, self.weight.value, self.eps)


class Qwen3_5GatedDeltaNet(Module):
    """Legacy TensorRT implementation of the Qwen3.5 GatedDeltaNet block."""

    def __init__(self,
                 hidden_size: int,
                 num_key_heads: int,
                 num_value_heads: int,
                 key_head_dim: int,
                 value_head_dim: int,
                 conv_kernel_size: int,
                 rms_norm_eps: float = 1e-6,
                 dtype=None,
                 tp_group=None,
                 tp_size: int = 1) -> None:
        super().__init__()

        if num_key_heads % tp_size != 0:
            raise ValueError("num_key_heads must be divisible by tp_size")
        if num_value_heads % tp_size != 0:
            raise ValueError("num_value_heads must be divisible by tp_size")

        self.hidden_size = hidden_size
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        self.conv_kernel_size = conv_kernel_size
        self.rms_norm_eps = rms_norm_eps
        self.dtype = dtype
        self.tp_group = tp_group
        self.tp_size = tp_size

        self.num_key_heads_per_tp = num_key_heads // tp_size
        self.num_value_heads_per_tp = num_value_heads // tp_size
        if self.num_value_heads_per_tp % self.num_key_heads_per_tp != 0:
            raise ValueError(
                "Local num_value_heads must be divisible by local num_key_heads")

        self.key_dim = self.key_head_dim * self.num_key_heads
        self.value_dim = self.value_head_dim * self.num_value_heads
        self.key_dim_per_tp = self.key_head_dim * self.num_key_heads_per_tp
        self.value_dim_per_tp = self.value_head_dim * self.num_value_heads_per_tp
        self.conv_dim_per_tp = self.key_dim_per_tp * 2 + self.value_dim_per_tp
        self.scale = self.key_head_dim**-0.5

        self.in_proj_qkv = ColumnLinear(hidden_size,
                                        self.key_dim * 2 + self.value_dim,
                                        bias=False,
                                        dtype=dtype,
                                        tp_group=tp_group,
                                        tp_size=tp_size,
                                        gather_output=False)
        self.in_proj_z = ColumnLinear(hidden_size,
                                      self.value_dim,
                                      bias=False,
                                      dtype=dtype,
                                      tp_group=tp_group,
                                      tp_size=tp_size,
                                      gather_output=False)
        self.in_proj_b = ColumnLinear(hidden_size,
                                      self.num_value_heads,
                                      bias=False,
                                      dtype=dtype,
                                      tp_group=tp_group,
                                      tp_size=tp_size,
                                      gather_output=False)
        self.in_proj_a = ColumnLinear(hidden_size,
                                      self.num_value_heads,
                                      bias=False,
                                      dtype=dtype,
                                      tp_group=tp_group,
                                      tp_size=tp_size,
                                      gather_output=False)
        self.conv1d = Qwen3_5CausalConv1d(self.conv_dim_per_tp,
                                          conv_kernel_size,
                                          dtype=dtype)
        self.dt_bias = Parameter(shape=(self.num_value_heads_per_tp, ),
                                 dtype='float32')
        self.A_log = Parameter(shape=(self.num_value_heads_per_tp, ),
                               dtype='float32')
        self.norm = Qwen3_5RmsNormGated(self.value_head_dim,
                                        eps=rms_norm_eps,
                                        dtype=dtype)
        self.out_proj = RowLinear(self.value_dim,
                                  hidden_size,
                                  bias=False,
                                  dtype=dtype,
                                  tp_group=tp_group,
                                  tp_size=tp_size)

    def forward(self,
                hidden_states: Tensor,
                past_conv_state: Tensor,
                past_delta_state: Tensor,
                host_request_types: Tensor | None = None,
                host_context_lengths: Tensor | None = None
                ) -> Tuple[Tensor, Tensor, Tensor]:
        """Run the Qwen3.5 GatedDeltaNet block.

        Args:
            hidden_states: [B, S, hidden_size]
            past_conv_state: [B, conv_dim_per_tp, conv_kernel_size - 1]
            past_delta_state:
                [B, num_value_heads_per_tp, value_head_dim, key_head_dim]

        Returns:
            A tuple of:
              - output: [B, S, hidden_size]
              - present_conv_state: [B, conv_dim_per_tp, conv_kernel_size - 1]
              - present_delta_state:
                [B, num_value_heads_per_tp, value_head_dim, key_head_dim]
        """
        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        mixed_qkv, present_conv_state = self.conv1d(
            mixed_qkv,
            past_conv_state,
            host_request_types=host_request_types)
        query, key, value = split(
            mixed_qkv,
            [self.key_dim_per_tp, self.key_dim_per_tp, self.value_dim_per_tp],
            dim=-1)

        query = query.view(
            concat([shape(query, 0),
                    shape(query, 1), self.num_key_heads_per_tp,
                    self.key_head_dim]))
        key = key.view(
            concat([shape(key, 0),
                    shape(key, 1), self.num_key_heads_per_tp,
                    self.key_head_dim]))
        value = value.view(
            concat([shape(value, 0),
                    shape(value, 1), self.num_value_heads_per_tp,
                    self.value_head_dim]))
        z = z.view(
            concat([shape(z, 0),
                    shape(z, 1), self.num_value_heads_per_tp,
                    self.value_head_dim]))
        use_plugin = (default_net().plugin_config.gated_delta_plugin
                      and host_request_types is not None
                      and host_context_lengths is not None)
        if use_plugin:
            query, key = _expand_qk_heads(query, key,
                                          self.num_value_heads_per_tp)
            g = None
            beta = None
        else:
            query, key = _normalize_and_expand_qk(query, key,
                                                  self.num_value_heads_per_tp)
            g, beta = _compute_gated_delta_g_beta(a, b, self.A_log.value,
                                                  self.dt_bias.value)
        core_attn_out, present_delta_state = gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            past_delta_state,
            self.scale,
            a_log=self.A_log.value,
            a=a,
            dt_bias=self.dt_bias.value,
            b=b,
            host_request_types=host_request_types,
            host_context_lengths=host_context_lengths)

        core_attn_out = cast(core_attn_out, hidden_states.dtype)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.view(
            concat([shape(core_attn_out, 0),
                    shape(core_attn_out, 1), self.value_dim_per_tp]))
        output = self.out_proj(core_attn_out)
        return output, present_conv_state, present_delta_state
