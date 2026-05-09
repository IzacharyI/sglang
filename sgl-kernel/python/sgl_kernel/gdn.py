from typing import Optional

import torch

LAYOUT_KV = 0
LAYOUT_VK = 1


def hip_fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    is_kda: bool = False,
) -> torch.Tensor:
    """ROCm HIP inline-asm GDN decode kernel.

    The state tensor must be in VK layout: [pool, num_v_heads, V, K]. This is
    the layout produced by ``hip_state_transpose_inplace_multi_layer`` when
    ``target_layout`` is ``LAYOUT_VK``.
    """
    del softplus_beta, softplus_threshold, is_kda

    B, T, num_k_heads, head_k_dim = q.shape
    num_v_heads = v.shape[2]
    if scale is None:
        scale = head_k_dim ** -0.5

    batch_size = B * T if cu_seqlens is None else len(cu_seqlens) - 1
    seq_length = 1 if cu_seqlens is not None else T

    output = torch.empty_like(v)
    dt_bias_bf16 = (
        dt_bias.to(torch.bfloat16) if dt_bias.dtype != torch.bfloat16 else dt_bias
    )
    indices_i32 = (
        initial_state_indices.to(torch.int32)
        if initial_state_indices.dtype != torch.int32
        else initial_state_indices
    )

    torch.ops.sgl_kernel.hip_gdn_decode_asm(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        a.contiguous(),
        b.contiguous(),
        dt_bias_bf16,
        A_log.contiguous(),
        indices_i32,
        initial_state_source,
        output,
        batch_size,
        seq_length,
        1,
        use_qk_l2norm_in_kernel,
        float(scale),
        num_k_heads,
        num_v_heads,
    )
    return output


def hip_state_transpose_inplace(
    state: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    num_v_heads: int,
) -> None:
    indices_i32 = indices.to(torch.int32) if indices.dtype != torch.int32 else indices
    torch.ops.sgl_kernel.hip_gdn_state_transpose(
        state,
        indices_i32,
        int(batch_size),
        int(num_v_heads),
    )


def hip_state_transpose_inplace_multi_layer(
    full_state: torch.Tensor,
    indices: torch.Tensor,
    slot_layout: torch.Tensor,
    target_layout: int,
    num_layers: int,
    num_v_heads: int,
) -> None:
    indices_i32 = indices.to(torch.int32) if indices.dtype != torch.int32 else indices
    if slot_layout.dtype != torch.int8:
        raise TypeError(f"slot_layout must be int8, got {slot_layout.dtype}")
    if not full_state.is_contiguous():
        raise ValueError("full_state must be contiguous")

    torch.ops.sgl_kernel.hip_gdn_state_transpose_multi_layer(
        full_state,
        indices_i32,
        slot_layout,
        int(target_layout),
        int(num_layers),
        int(indices_i32.shape[0]),
        int(num_v_heads),
        int(full_state.stride(0)),
    )


__all__ = [
    "LAYOUT_KV",
    "LAYOUT_VK",
    "hip_fused_sigmoid_gating_delta_rule_update",
    "hip_state_transpose_inplace",
    "hip_state_transpose_inplace_multi_layer",
]
