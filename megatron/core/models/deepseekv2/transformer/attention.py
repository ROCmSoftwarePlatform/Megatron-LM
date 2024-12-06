from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union
import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.utils import divide
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.attention import Attention, SelfAttention, SelfAttentionSubmodules

from megatron.core.models.deepseekv2.yarn_rotary_pos_embedding import DeepseekV2YarnRotaryEmbedding, \
    apply_rotary_pos_emb, yarn_get_mscale


class DeepSeekv2Attention(Attention, ABC):
    """Attention layer abstract  class modified for Deepseekv2.

    This layer only contains common modules required for the "self attn" and
    "cross attn" specializations.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
    ):
        super().__init__(config=config,submodules=submodules,layer_number=layer_number,attn_mask_type=attn_mask_type,attention_type=attention_type)
        
        self.num_heads = self.config.num_attention_heads
        self.q_head_dim = self.config.qk_nope_head_dim + self.config.qk_rope_head_dim
        kwargs = {
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
        }

        self.rotary_pos_emb = DeepseekV2YarnRotaryEmbedding(
            self.config.qk_rope_head_dim,
            max_position_embeddings=self.config.max_position_embeddings,
            scaling_factor=self.config.rotary_scaling_factor,
            base=self.config.rotary_base,
            **kwargs,
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        packed_seq_params=None,
        position_ids=None
    ):
        # hidden_states: [sq, b, h]
        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        # query: [1, 8, 96, 192], key:[1, 8, 96, 192], value:[1, 8, 96, 128]
        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states, position_ids)

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        # ==================================
        # core attention computation
        # ==================================

        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=self.attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=self.attn_mask_type,
                packed_seq_params=packed_seq_params,
            )

        if packed_seq_params is not None:
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.linear_proj(core_attn_out)

        return output, bias
    
class DeepSeekv2SelfAttention(SelfAttention, DeepSeekv2Attention):
    """Self-attention layer class modified for Deepseekv2

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """
    def __init__(self,config: TransformerConfig,
                 submodules: SelfAttentionSubmodules,
                 layer_number: int,
                 attn_mask_type=AttnMaskType.padding,):
        super().__init__(config=config,
                         submodules=submodules,
                         layer_number=layer_number,
                         attn_mask_type=attn_mask_type)

        if self.config.q_lora_rank is None:

            self.linear_q_proj = build_module(
                submodules.linear_q_proj,
                self.config.hidden_size,
                self.num_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
            )

        else:

            self.linear_q_a_proj = build_module(
                submodules.linear_q_a_proj,
                self.config.hidden_size,
                self.config.q_lora_rank,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
            )

            self.linear_q_b_proj = build_module(
                submodules.linear_q_b_proj,
                self.config.q_lora_rank//self.world_size,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=True,
            )

        self.linear_kv_a_proj_with_mqa = build_module(
            submodules.linear_kv_a_proj_with_mqa,
            self.config.hidden_size,
            (self.config.kv_lora_rank + self.config.qk_rope_head_dim)*self.world_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
        )

        self.linear_kv_b_proj = build_module(
            submodules.linear_kv_b_proj,
            self.config.kv_lora_rank,
            self.config.num_attention_heads * (self.q_head_dim - self.config.qk_rope_head_dim + self.config.v_head_dim),
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=True,
        )

        if self.config.q_lora_rank is not None:

            self.q_a_layernorm = build_module(
                submodules.q_a_layernorm,
                hidden_size=self.config.q_lora_rank//self.world_size,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )

        self.kv_a_layernorm = build_module(
            submodules.kv_a_layernorm,
            hidden_size=self.config.kv_lora_rank,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )