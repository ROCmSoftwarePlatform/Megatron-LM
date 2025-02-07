# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

import os
import sys
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            os.path.pardir
        )
    )
)

import inspect
from contextlib import nullcontext

from megatron.core.enums import ModelType
from megatron.core.extensions.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import _get_mlp_module_spec
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.grok_transformer_layer import GrokTransformerLayer
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.training import (
    pretrain,
    print_rank_0,
    get_args,
)
from megatron.training.arguments import core_transformer_config_from_args

from pretrain_gpt import train_valid_test_datasets_provider, forward_step


def model_provider(pre_process=True, post_process=True) -> GPTModel:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        GPTModel: The returned model
    """
    args = get_args()

    print_rank_0('building Grok-1 model ...')
    config = core_transformer_config_from_args(args)

    build_model_context = nullcontext
    build_model_context_args = {}
    if args.fp8_param_gather:
        try:
            from transformer_engine.pytorch import fp8_model_init

            build_model_context = fp8_model_init
            build_model_context_args["enabled"] = True

            # Check if fp8_model_init supports preserve_high_precision_init_val
            if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                build_model_context_args["preserve_high_precision_init_val"] = True
        except:
            raise RuntimeError(
                "--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found.")

    mlp = _get_mlp_module_spec(
        use_te=True,
        num_experts=args.num_experts,
        moe_grouped_gemm=args.moe_grouped_gemm,
        fp8=args.fp8
    )
    with build_model_context(**build_model_context_args):
        model = GPTModel(
            config=config,
            transformer_layer_spec=ModuleSpec(
                module=GrokTransformerLayer,
                submodules=TransformerLayerSubmodules(
                    self_attention=ModuleSpec(
                        module=SelfAttention,
                        params={"attn_mask_type": AttnMaskType.causal},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=TELayerNormColumnParallelLinear,
                            core_attention=TEDotProductAttention,
                            linear_proj=TERowParallelLinear,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                ),
            ),
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling
        )

    return model


if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
    )
