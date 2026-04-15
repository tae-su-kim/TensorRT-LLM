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

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.models.qwen.gated_delta import Qwen3_5GatedDeltaNet
from utils.util import create_session, run_session


def _torch_l2_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


class _TorchQwen3_5RmsNormGated(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor,
                gate: torch.Tensor) -> torch.Tensor:
        hidden_states_fp32 = hidden_states.float()
        gate_fp32 = gate.float()
        variance = hidden_states_fp32.pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states_fp32 * torch.rsqrt(variance + self.eps)
        normalized = normalized * self.weight.float()
        normalized = normalized * F.silu(gate_fp32)
        return normalized.to(hidden_states.dtype)


class _TorchQwen3_5GatedDeltaNet(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_key_heads: int,
                 num_value_heads: int,
                 key_head_dim: int,
                 value_head_dim: int,
                 conv_kernel_size: int,
                 rms_norm_eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        self.conv_kernel_size = conv_kernel_size

        self.num_key_heads_per_tp = num_key_heads
        self.num_value_heads_per_tp = num_value_heads
        self.key_dim_per_tp = key_head_dim * num_key_heads
        self.value_dim_per_tp = value_head_dim * num_value_heads
        self.conv_dim_per_tp = self.key_dim_per_tp * 2 + self.value_dim_per_tp
        self.scale = key_head_dim**-0.5

        self.in_proj_qkv = nn.Linear(hidden_size,
                                     self.key_dim_per_tp * 2 +
                                     self.value_dim_per_tp,
                                     bias=False)
        self.in_proj_z = nn.Linear(hidden_size,
                                   self.value_dim_per_tp,
                                   bias=False)
        self.in_proj_b = nn.Linear(hidden_size,
                                   self.num_value_heads_per_tp,
                                   bias=False)
        self.in_proj_a = nn.Linear(hidden_size,
                                   self.num_value_heads_per_tp,
                                   bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim_per_tp,
                                self.conv_dim_per_tp,
                                kernel_size=conv_kernel_size,
                                groups=self.conv_dim_per_tp,
                                bias=False)
        self.dt_bias = nn.Parameter(torch.ones(self.num_value_heads_per_tp,
                                               dtype=torch.float32))
        self.A_log = nn.Parameter(torch.randn(self.num_value_heads_per_tp,
                                              dtype=torch.float32))
        self.norm = _TorchQwen3_5RmsNormGated(value_head_dim, eps=rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim_per_tp,
                                  hidden_size,
                                  bias=False)

    def forward(
            self, hidden_states: torch.Tensor, past_conv_state: torch.Tensor,
            past_delta_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        full_input = torch.cat(
            [past_conv_state, mixed_qkv.transpose(1, 2).contiguous()], dim=2)
        present_conv_state = full_input[:, :, -self.conv_kernel_size +
                                        1:].contiguous()
        mixed_qkv = F.conv1d(full_input,
                             self.conv1d.weight,
                             bias=None,
                             groups=self.conv_dim_per_tp)
        mixed_qkv = F.silu(mixed_qkv).transpose(1, 2).contiguous()

        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim_per_tp, self.key_dim_per_tp, self.value_dim_per_tp],
            dim=-1)

        query = query.view(batch_size, seq_len, self.num_key_heads_per_tp,
                           self.key_head_dim)
        key = key.view(batch_size, seq_len, self.num_key_heads_per_tp,
                       self.key_head_dim)
        value = value.view(batch_size, seq_len, self.num_value_heads_per_tp,
                           self.value_head_dim)
        z = z.view(batch_size, seq_len, self.num_value_heads_per_tp,
                   self.value_head_dim)
        query = _torch_l2_norm(query.float())
        key = _torch_l2_norm(key.float())
        repeat_factor = self.num_value_heads_per_tp // self.num_key_heads_per_tp
        if repeat_factor > 1:
            query = query.repeat_interleave(repeat_factor, dim=2)
            key = key.repeat_interleave(repeat_factor, dim=2)

        g = -self.A_log.float().exp() * F.softplus(
            a.float() + self.dt_bias.float())
        beta = b.float().sigmoid()

        state = past_delta_state.float()
        outputs = []
        for token_idx in range(seq_len):
            decay = torch.exp(g[:, token_idx][..., None, None])
            state = state * decay
            v_prime = value[:, token_idx].float() - torch.einsum(
                'bhk,bhvk->bhv', key[:, token_idx], state)
            v_prime = v_prime * beta[:, token_idx][..., None]
            state = state + torch.einsum('bhk,bhv->bhvk', key[:, token_idx],
                                         v_prime)
            outputs.append(
                torch.einsum('bhk,bhvk->bhv',
                             query[:, token_idx] * self.scale, state))

        core_attn_out = torch.stack(outputs, dim=1).to(hidden_states.dtype)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.view(batch_size, seq_len,
                                           self.value_dim_per_tp)
        output = self.out_proj(core_attn_out)
        return output, present_conv_state, state


@pytest.mark.parametrize(
    ("seq_len", "request_type", "state_dtype", "atol", "rtol"),
    [
        (4, 0, 'float32', 1e-5, 1e-5),
        (1, 1, 'float32', 1e-5, 1e-5),
        (1, 1, 'bfloat16', 5e-2, 5e-2),
    ],
)
def test_qwen3_5_gated_delta_runtime_matches_torch_reference(
        seq_len: int, request_type: int, state_dtype: str, atol: float,
        rtol: float):
    tensorrt_llm.logger.set_level('error')
    torch.manual_seed(1234)

    dtype = 'float32'
    device = 'cuda'
    batch_size = 2
    hidden_size = 16
    num_key_heads = 2
    num_value_heads = 4
    key_head_dim = 4
    value_head_dim = 2
    conv_kernel_size = 3

    torch_model = _TorchQwen3_5GatedDeltaNet(
        hidden_size=hidden_size,
        num_key_heads=num_key_heads,
        num_value_heads=num_value_heads,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        conv_kernel_size=conv_kernel_size,
    ).to(device=device, dtype=torch.float32).eval()

    hidden_states = torch.randn(batch_size,
                                seq_len,
                                hidden_size,
                                device=device,
                                dtype=torch.float32)
    past_conv_state = torch.randn(batch_size,
                                  torch_model.conv_dim_per_tp,
                                  conv_kernel_size - 1,
                                  device=device,
                                  dtype=torch.float32)
    past_delta_state = torch.randn(batch_size,
                                   num_value_heads,
                                   value_head_dim,
                                   key_head_dim,
                                   device=device,
                                   dtype=torch.float32)
    host_request_types = torch.full((batch_size, ),
                                    request_type,
                                    dtype=torch.int32)
    host_context_lengths = torch.full((batch_size, ),
                                      seq_len,
                                      dtype=torch.int32)

    with torch.no_grad():
        ref_output, ref_present_conv_state, ref_present_delta_state = torch_model(
            hidden_states, past_conv_state, past_delta_state)

    tllm_model = Qwen3_5GatedDeltaNet(hidden_size=hidden_size,
                                      num_key_heads=num_key_heads,
                                      num_value_heads=num_value_heads,
                                      key_head_dim=key_head_dim,
                                      value_head_dim=value_head_dim,
                                      conv_kernel_size=conv_kernel_size,
                                      dtype=dtype)
    tllm_model.update_parameters(torch_model)

    builder = tensorrt_llm.Builder()
    network = builder.create_network()
    network.plugin_config.gated_delta_plugin = True
    with tensorrt_llm.net_guard(network):
        network.set_named_parameters(tllm_model.named_parameters())

        hidden_states_tensor = Tensor(name='hidden_states',
                                      shape=hidden_states.shape,
                                      dtype=tensorrt_llm.str_dtype_to_trt(
                                          dtype))
        past_conv_state_tensor = Tensor(name='past_conv_state',
                                        shape=past_conv_state.shape,
                                        dtype=tensorrt_llm.str_dtype_to_trt(
                                            dtype))
        past_delta_state_tensor = Tensor(name='past_delta_state',
                                         shape=past_delta_state.shape,
                                         dtype=tensorrt_llm.str_dtype_to_trt(
                                             state_dtype))
        host_request_types_tensor = Tensor(
            name='host_request_types',
            shape=host_request_types.shape,
            dtype=tensorrt_llm.str_dtype_to_trt('int32'))
        host_context_lengths_tensor = Tensor(
            name='host_context_lengths',
            shape=host_context_lengths.shape,
            dtype=tensorrt_llm.str_dtype_to_trt('int32'))

        output, present_conv_state, present_delta_state = tllm_model(
            hidden_states_tensor,
            past_conv_state_tensor,
            past_delta_state_tensor,
            host_request_types_tensor,
            host_context_lengths_tensor)
        output.mark_output('output', dtype)
        present_conv_state.mark_output('present_conv_state', dtype)
        present_delta_state.mark_output('present_delta_state', state_dtype)

    session = create_session(builder, network, precision=dtype)
    outputs = run_session(session, {
        'hidden_states': hidden_states,
        'past_conv_state': past_conv_state,
        'past_delta_state': past_delta_state.to(str_dtype_to_torch(state_dtype)),
        'host_request_types': host_request_types,
        'host_context_lengths': host_context_lengths,
    })

    torch.testing.assert_close(outputs['output'],
                               ref_output,
                               atol=atol,
                               rtol=rtol)
    torch.testing.assert_close(outputs['present_conv_state'],
                               ref_present_conv_state,
                               atol=atol,
                               rtol=rtol)
    torch.testing.assert_close(outputs['present_delta_state'].float(),
                               ref_present_delta_state,
                               atol=atol,
                               rtol=rtol)
