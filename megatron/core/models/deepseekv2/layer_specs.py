# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules, DeepSeekv2SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules, TransformerLayer

from megatron.core.models.deepseekv2.transformer.attention import DeepSeekv2Attention, DeepSeekv2SelfAttention

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TEColumnParallelLinear,
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TENorm,
        TERowParallelGroupedLinear,
        TERowParallelLinear,
        TEDotProductAttentionMLA
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

from megatron.legacy.model.rms_norm import RMSNorm


def get_gpt_layer_with_transformer_engine_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    fp8: Optional[str] = None,
) -> ModuleSpec:
    """Use this spec to use lower-level Transformer Engine modules (required for fp8 training).


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        fp8 (str, optional): Flag to decide the linear layer spec for MoE. Defaults to None.

    Returns:
        ModuleSpec: Module specification with TE modules
    """
    mlp = _get_mlp_module_spec(
        use_te=True, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm, fp8=fp8
    )
    mlp_dense = _get_mlp_module_spec(
        use_te=False, num_experts=None, moe_grouped_gemm=moe_grouped_gemm
    )

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=DeepSeekv2SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=DeepSeekv2SelfAttentionSubmodules(
                    linear_q_proj=TEColumnParallelLinear,
                     linear_q_down_proj=TEColumnParallelLinear,
                     linear_q_up_proj=ColumnParallelLinear,
                     linear_kv_down_proj=TEColumnParallelLinear,
                     linear_kv_up_proj=ColumnParallelLinear,
                     linear_proj=TERowParallelLinear,
                     q_a_layernorm=TENorm if qk_layernorm else IdentityOp,
                     kv_a_layernorm=TENorm if qk_layernorm else IdentityOp,
                     core_attention=TEDotProductAttentionMLA,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=TENorm if num_experts else IdentityOp,
            mlp=mlp,
            mlp_dense=mlp_dense,
            mlp_bda=get_bias_dropout_add,
        ),
    )


def get_gpt_layer_local_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
) -> ModuleSpec:
    """Use this spec for an implementation using only modules in Megatron-Core.


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.

    Returns:
        ModuleSpec: Module specification with Megatron-Core modules
    """
    mlp = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    mlp_dense = _get_mlp_module_spec(
        use_te=False, num_experts=None, moe_grouped_gemm=moe_grouped_gemm
    )

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=DeepSeekv2SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=DeepSeekv2SelfAttentionSubmodules(
                    linear_q_proj=ColumnParallelLinear,
                    linear_q_down_proj=ColumnParallelLinear,
                    linear_q_up_proj=ColumnParallelLinear,
                    linear_kv_down_proj=ColumnParallelLinear,
                    linear_kv_up_proj=ColumnParallelLinear,
                    linear_proj=RowParallelLinear,
                    q_a_layernorm=RMSNorm if qk_layernorm else IdentityOp,
                    kv_a_layernorm=RMSNorm if qk_layernorm else IdentityOp,
                    core_attention=DotProductAttention,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=RMSNorm if num_experts else IdentityOp,
            input_layernorm=RMSNorm if num_experts else IdentityOp,
            mlp=mlp,
            mlp_dense=mlp_dense,
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
            },
        ),
    )


def _get_mlp_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    fp8: Optional[str] = None,
) -> ModuleSpec:
    """Helper function to get module spec for MLP/MoE"""
    if num_experts is None:
        # Dense MLP w/ or w/o TE modules.
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=TELayerNormColumnParallelLinear if use_te else ColumnParallelLinear,
                linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
            ),
        )
    else:
        # Mixture of experts with modules in megatron core.
        if use_te and moe_grouped_gemm:
            linear_fc1 = TEColumnParallelGroupedLinear
            linear_fc2 = TERowParallelGroupedLinear
        elif use_te and fp8:
            linear_fc1 = TEColumnParallelLinear
            linear_fc2 = TERowParallelLinear
        else:
            linear_fc1 = ColumnParallelLinear
            linear_fc2 = RowParallelLinear

        use_te_grouped_gemm = use_te and TEColumnParallelGroupedLinear is not None

        return ModuleSpec(
            module=MoELayer,
            submodules=MoESubmodules(
                experts=(
                    MLPSubmodules(linear_fc1=linear_fc1, linear_fc2=linear_fc2)
                    if not moe_grouped_gemm or use_te_grouped_gemm
                    else None
                ),
                shared_experts=ModuleSpec(
                    module=SharedExpertMLP,
                    params={"gate": False},
                    submodules=MLPSubmodules(
                        linear_fc1=TEColumnParallelLinear if use_te else ColumnParallelLinear,
                        linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
                    ),
                ),
            ),
        )