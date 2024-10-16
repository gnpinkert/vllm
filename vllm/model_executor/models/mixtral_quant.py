# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Mixtral model."""
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import MixtralConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from vllm.model_executor.layers.fused_moe import DebugCudaEvent, MoeGpuBuffer

from moe_predict import Inference
from moe_predict.models import MixtralModelConfig


class MixtralMLP(nn.Module):

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.ffn_dim = intermediate_size
        self.hidden_dim = hidden_size

        self.w1 = ReplicatedLinear(self.hidden_dim,
                                   self.ffn_dim,
                                   bias=False,
                                   quant_config=quant_config)
        self.w2 = ReplicatedLinear(self.ffn_dim,
                                   self.hidden_dim,
                                   bias=False,
                                   quant_config=quant_config)
        self.w3 = ReplicatedLinear(self.hidden_dim,
                                   self.ffn_dim,
                                   bias=False,
                                   quant_config=quant_config)

        # TODO: Use vllm's SiluAndMul
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor, moe_gpu_buffer: MoeGpuBuffer,
                active_expert_idx: int) -> torch.Tensor:
        moe_gpu_buffer.set_weight_class("W1")
        moe_gpu_buffer.load_predicted_experts_stream.synchronize()
        w1_out, _ = self.w1(hidden_states, moe_gpu_buffer, active_expert_idx)
        w1_out = self.act_fn(w1_out)
        moe_gpu_buffer.set_weight_class("W3")
        w3_out, _ = self.w3(hidden_states, moe_gpu_buffer, active_expert_idx)
        current_hidden_states = w1_out * w3_out
        moe_gpu_buffer.set_weight_class("W2")
        current_hidden_states, _ = self.w2(current_hidden_states, moe_gpu_buffer, active_expert_idx)
        return current_hidden_states


def find_invalid_indices(predicted_expert_ids: torch.Tensor, actual_expert_ids: torch.Tensor) -> List[int] :
    mask = torch.isin(predicted_expert_ids, actual_expert_ids)
    invalid_indices = torch.nonzero(~mask, as_tuple=True)[0]

    return invalid_indices.tolist()

class MixtralMoE(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_total_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        if self.tp_size > self.num_total_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {self.num_total_experts}.")
        # Split experts equally between ranks
        self.expert_indicies = np.array_split(range(
            self.num_total_experts), self.tp_size)[self.rank].tolist()
        if not self.expert_indicies:
            raise ValueError(
                f"Rank {self.rank} has no experts assigned to it.")

        self.experts = nn.ModuleList([
            MixtralMLP(self.num_total_experts,
                       config.hidden_size,
                       config.intermediate_size,
                       quant_config=quant_config)
            if idx in self.expert_indicies else None
            for idx in range(self.num_total_experts)
        ])
        self.gate = ReplicatedLinear(config.hidden_size,
                                     self.num_total_experts,
                                     bias=False,
                                     quant_config=None)

    def forward(self, hidden_states: torch.Tensor, moe_gpu_buffer: MoeGpuBuffer,
                router_event: DebugCudaEvent, is_prefill: bool) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights,
                                                       self.top_k,
                                                       dim=-1)

        final_hidden_states = None
        unique_values = torch.unique(selected_experts)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        if is_prefill:
            first_index = unique_values[0]
            prefill_stream = torch.cuda.Stream()
            predicted_expert_list = [-1, -1]
            for expert_idx in self.expert_indicies:
                if expert_idx not in unique_values:
                    continue
                expert_mlp = self.experts[expert_idx]
                expert_mask = (selected_experts == expert_idx)
                expert_weights = (routing_weights * expert_mask).sum(dim=-1,
                                                                     keepdim=True)

                if expert_idx == first_index:
                    predicted_expert_list[0] = expert_idx
                    moe_gpu_buffer = self.load_experts(predicted_expert_list, stream=moe_gpu_buffer.load_predicted_experts_stream, moe_gpu_buffer=moe_gpu_buffer)

                with torch.cuda.stream(prefill_stream):
                    if expert_idx < 7:
                        next_id = expert_idx + 1
                        while next_id not in unique_values and next_id < 9:
                            next_id += 1
                        if next_id < 8:
                                prev_index = predicted_expert_list.index(expert_idx)
                                next_index = 1 - prev_index
                                predicted_expert_list[next_index] = next_id
                                predicted_expert_list[prev_index] = -1
                                moe_gpu_buffer = self.load_experts(predicted_expert_list, stream=prefill_stream, moe_gpu_buffer=moe_gpu_buffer)

                current_hidden_states = expert_mlp(hidden_states,
                                                   active_expert_idx=expert_idx,
                                                   moe_gpu_buffer=moe_gpu_buffer).mul_(expert_weights)
                router_event.mlp_w2_finished_event.record()
                if final_hidden_states is None:
                    final_hidden_states = current_hidden_states
                else:
                    final_hidden_states.add_(current_hidden_states)

            return tensor_model_parallel_all_reduce(final_hidden_states).view(
                num_tokens, hidden_dim)

        router_event.triggerTopkEvent(selected_experts)

        if router_event.is_first_layer:
            moe_gpu_buffer = self.load_experts(selected_experts[0].tolist(), stream=moe_gpu_buffer.load_predicted_experts_stream, moe_gpu_buffer=moe_gpu_buffer)
        else:
            invalid_indices = find_invalid_indices(torch.tensor(moe_gpu_buffer.expert_ids).to('cuda'), selected_experts[0].to('cuda'))
            replaced_ids = []
            for expert in selected_experts[0].tolist():
                if expert not in moe_gpu_buffer.expert_ids:
                    replaced_ids.append(expert)
            moe_gpu_buffer = self.load_experts(experts=replaced_ids, stream=moe_gpu_buffer.load_predicted_experts_stream, moe_gpu_buffer=moe_gpu_buffer, invalid_indices=invalid_indices)

        for expert_idx in self.expert_indicies:
            if expert_idx not in unique_values:
                continue

            expert_mlp = self.experts[expert_idx]
            expert_mask = (selected_experts == expert_idx)
            expert_weights = (routing_weights * expert_mask).sum(dim=-1,
                                                                 keepdim=True)
            current_hidden_states = expert_mlp(hidden_states,
                                               active_expert_idx=expert_idx,
                                               moe_gpu_buffer=moe_gpu_buffer).mul_(expert_weights)
            router_event.mlp_w2_finished_event.record()
            if final_hidden_states is None:
                final_hidden_states = current_hidden_states
            else:
                final_hidden_states.add_(current_hidden_states)

        return tensor_model_parallel_all_reduce(final_hidden_states).view(
            num_tokens, hidden_dim)

    def load_experts(self, experts: List, moe_gpu_buffer: MoeGpuBuffer, stream: torch.cuda.Stream, invalid_indices = None) -> MoeGpuBuffer:
        # apparently if you pin memory and then copy non blocking then you can interleave
        # data transfer operations which is why we are using nonblocking and then sychronizing
        with torch.cuda.stream(stream):
            for predicted_idx, predicted_val in enumerate(experts):
                if predicted_val == -1:
                    continue
                new_idx = predicted_idx
                if invalid_indices is not None:
                    new_idx = invalid_indices.pop(0)

                self.experts[predicted_val].w1.qweight.data.pin_memory()
                #1_data.pin_memory()

                w2_data = self.experts[predicted_val].w2.qweight.data
                w2_data.pin_memory()

                w3_data = self.experts[predicted_val].w3.qweight.data
                w3_data.pin_memory()

                moe_gpu_buffer.qweight_w1s[new_idx, :, :].copy_(self.experts[predicted_val].w1.qweight.data, non_blocking=True)
                moe_gpu_buffer.qweight_w2s[new_idx, :, :].copy_(w2_data, non_blocking=True)
                moe_gpu_buffer.qweight_w3s[new_idx, :, :].copy_(w3_data, non_blocking=True)

                moe_gpu_buffer.expert_ids[new_idx] = predicted_val

        return moe_gpu_buffer


class MixtralAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=int(self.rope_theta),
            is_neox_style=True,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class MixtralDecoderLayer(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 10000)
        self.self_attn = MixtralAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            cache_config=cache_config,
            quant_config=quant_config)
        self.block_sparse_moe = MixtralMoE(config=config,
                                           quant_config=quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        router_event: DebugCudaEvent,
        moe_gpu_buffer: MoeGpuBuffer,
        is_prefill: bool
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.block_sparse_moe(hidden_states, moe_gpu_buffer=moe_gpu_buffer,
                                              router_event=router_event, is_prefill=is_prefill)
        return hidden_states, residual

def normalize(tensor):
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True)
    return (tensor - mean) / (std + 1e-5)

class MixtralModel(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.predictor = Inference(model=MixtralModelConfig())
        self.moe_gpu_buffers = MoeGpuBuffer(num_experts=2, w1s_shape=(256, 28672), w2s_shape=(896, 8192),
                                            w3s_shape=(256, 28672))
        self.norm_previous = torch.zeros(62, dtype=torch.float32).to('cuda')
        self.moe_events = DebugCudaEvent(topk=2)
        self.mlp_stream = torch.cuda.Stream()
        self.mixtral_model_config = MixtralModelConfig()

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            MixtralDecoderLayer(config,
                                cache_config,
                                quant_config=quant_config)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        is_prefill = input_ids.size(0) > 1
        if is_prefill:
            for i in range(len(self.layers)):
                layer = self.layers[i]
                hidden_states, residual = layer(positions, hidden_states,
                                                kv_caches[i], attn_metadata,
                                                residual=residual,
                                                moe_gpu_buffer=self.moe_gpu_buffers,
                                                router_event=self.moe_events,
                                                is_prefill=is_prefill)

            hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states

        self.moe_events.is_first_layer = True
        previous_experts = torch.zeros((self.mixtral_model_config.num_layers - 1) * 2)
        for layer_id in range(len(self.layers)):
            layer = self.layers[layer_id]
            with torch.cuda.stream(self.mlp_stream):
                hidden_states, residual = layer(positions, hidden_states,
                                                kv_caches[layer_id], attn_metadata,
                                                residual=residual,
                                                moe_gpu_buffer=self.moe_gpu_buffers,
                                                router_event=self.moe_events,
                                                is_prefill=is_prefill)
                self.moe_events.is_first_layer = False

            if layer_id < len(self.layers) - 1:
                self.moe_events._topk_decided_event.wait()
                previous_experts[layer_id * 2] = self.moe_events.experts[0][0]
                previous_experts[layer_id * 2 + 1] = self.moe_events.experts[0][1]
                self.norm_previous = normalize(previous_experts).to('cuda')
                predicted_experts = self.predictor.predict_next_experts(self.moe_events.experts[0], layer_id, self.norm_previous)
                self.moe_events.mlp_w2_finished_event.wait()
                self.moe_gpu_buffers = self.layers[layer_id + 1].block_sparse_moe.load_experts(predicted_experts[0], stream=self.moe_gpu_buffers.load_predicted_experts_stream, moe_gpu_buffer=self.moe_gpu_buffers)
                self.moe_events.reset_events()

                # Set norm_previous to zero since the next iteration will be from a new prompt
        self.norm_previous.zero_()

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class MixtralForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: MixtralConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = MixtralModel(config, cache_config, quant_config)
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      quant_config=quant_config)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip experts that are not assigned to this worker.
                if ("block_sparse_moe.experts." in name
                        and name not in params_dict):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
