# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

import torch

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    BaseTransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.utils import make_viewless_tensor, StragglerDetector


stimer = StragglerDetector()


class GrokTransformerLayer(MegatronModule, BaseTransformerLayer):
    def __init__(
        self: "GrokTransformerLayer",
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
    ):
        super().__init__(config=config)

        if config.enable_cuda_graph and self.training:
            raise Exception("Cudagraphs not supported")

        self.submodules_config = submodules
        self.layer_number = layer_number + self._get_layer_offset()
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout

        # [Module 1: Pre SelfAttention RMSNorm]
        self.pre_attn_layernorm = build_module(
            TENorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # [Module 2: SelfAttention]
        self.self_attention = build_module(
            submodules.self_attention, config=self.config, layer_number=layer_number
        )

        # [Module 3: BiasDropoutFusion]
        self.self_attn_bda = build_module(submodules.self_attn_bda)

        # [Module 4: Post SelfAttention RMSNorm]
        self.post_attn_layernorm = build_module(
            TENorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # [Module 5: Pre MoE RMSNorm]
        self.pre_mlp_layernorm = build_module(
            TENorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # [Module 6: MLP block]
        self.mlp = build_module(submodules.mlp, config=self.config)
        if hasattr(self.mlp, 'set_layer_number'):
            self.mlp.set_layer_number(self.layer_number)

        # [Module 7: BiasDropoutFusion]
        self.mlp_bda = build_module(submodules.mlp_bda)

        self.bias_dropout_add_exec_handler = torch.enable_grad

        # [Module 8: Post MoE RMSNorm]
        self.post_mlp_layernorm = build_module(
            TENorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

    def _get_layer_offset(self: "GrokTransformerLayer") -> int:
        """Get the index number of this layer, given the level of pipelining."""
        pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()

        num_layers_per_pipeline_rank = (
            self.config.num_layers // self.config.pipeline_model_parallel_size
        )

        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            vp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
            vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

            total_num_layers = self.config.num_layers
            num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size
            total_virtual_chunks = total_num_layers // vp_size
            offset = vp_rank * total_virtual_chunks + \
                (pipeline_rank * num_layers_per_virtual_rank)

        else:
            # Each stage gets a contiguous set of layers.
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                if (
                    self.config.first_pipeline_num_layers is not None
                    or self.config.last_pipeline_num_layers is not None
                ):
                    # Calculate number of pipelines for distributing layers
                    middle_pipeline_stages = parallel_state.get_pipeline_model_parallel_world_size()
                    middle_pipeline_stages -= sum(
                        [
                            1 if x is not None else 0
                            for x in (
                                self.config.first_pipeline_num_layers,
                                self.config.last_pipeline_num_layers,
                            )
                        ]
                    )

                    # Calculate layers to distribute
                    first_pipeline_offset = (
                        0
                        if self.config.first_pipeline_num_layers is None
                        else self.config.first_pipeline_num_layers
                    )
                    last_pipeline_offset = (
                        0
                        if self.config.first_pipeline_num_layers is None
                        else self.config.last_pipeline_num_layers
                    )

                    middle_num_layers = (
                        self.config.num_layers - first_pipeline_offset - last_pipeline_offset
                    )

                    if middle_pipeline_stages > 0:
                        num_layers_per_pipeline_rank = middle_num_layers // middle_pipeline_stages
                    else:
                        num_layers_per_pipeline_rank = 0

                    middle_pipeline_rank = (
                        pipeline_rank
                        if self.config.first_pipeline_num_layers is None
                        else pipeline_rank - 1
                    )

                    if pipeline_rank == 0:
                        offset = 0
                    else:
                        offset = (
                            middle_pipeline_rank * num_layers_per_pipeline_rank
                        ) + first_pipeline_offset
                else:
                    offset = pipeline_rank * num_layers_per_pipeline_rank
            else:
                offset = 0

        return offset

    def forward(
        self: "GrokTransformerLayer",
        hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        inference_params=None,
        packed_seq_params=None,
    ):
        # Residual connection.
        residual = hidden_states

        # Attention layers
        pre_attn_layernorm_output = self.pre_attn_layernorm(hidden_states)
        attention_output_with_bias = self.self_attention(
            pre_attn_layernorm_output,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, torch.zeros_like(
                    residual), self.hidden_dropout
            )
        post_attn_layernorm_output = self.post_attn_layernorm(hidden_states)
        hidden_states = residual + post_attn_layernorm_output

        # Residual connection.
        residual = hidden_states

        # MLP layers
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, torch.zeros_like(
                    residual), self.hidden_dropout
            )
        post_mlp_layernorm_output = self.post_mlp_layernorm(hidden_states)
        hidden_states = residual + post_mlp_layernorm_output

        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )
        return output, context