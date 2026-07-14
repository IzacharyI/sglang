# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os
from typing import Optional

import torch
import triton
from einops import rearrange

from sglang.srt.layers.attention.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from sglang.srt.layers.attention.fla.chunk_o import chunk_fwd_o
from sglang.srt.layers.attention.fla.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
from sglang.srt.layers.attention.fla.fused_cumsum_kkt import fused_cumsum_kkt
from sglang.srt.layers.attention.fla.fused_merge_recompute import fused_merge_recompute
from sglang.srt.layers.attention.fla.index import prepare_chunk_indices
from sglang.srt.layers.attention.fla.l2norm import fused_l2norm_qk, l2norm_fwd
from sglang.srt.layers.attention.fla.solve_tril import (
    solve_tril,
    solve_tril_16x16_kernel,
)
from sglang.srt.layers.attention.fla.utils import (
    SUPPRESS_LEVEL,
    autocast_custom_fwd,
    input_guard,
)
from sglang.srt.layers.attention.fla.wy_fast import recompute_w_u_fwd
from sglang.srt.utils import is_hip

_is_hip = is_hip()

try:
    from aiter.ops.chunk_gated_delta_rule_fwd_h import (
        chunk_gated_delta_rule_fwd_h_hip_fn as aiter_chunk_gated_delta_rule_fwd_h_hip_fn,
    )
    from aiter.ops.triton.gated_delta_net.gated_delta_rule import (
        chunk_gated_delta_rule_opt_vk as aiter_chunk_gated_delta_rule_opt_vk,
    )
    from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h_opt_vk as aiter_chunk_gated_delta_rule_fwd_h_triton_opt_vk,
    )
    from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_o import (
        chunk_fwd_o_opt_vk as aiter_chunk_fwd_o_opt_vk,
    )
    from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.fused_cumsum_kkt import (
        fused_chunk_local_cumsum_scaled_dot_kkt_fwd as aiter_fused_cumsum_kkt_opt_vk,
    )
    from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.fused_solve_tril_recompute import (
        fused_solve_tril_recompute_w_u as aiter_fused_solve_tril_recompute_w_u,
    )

    _aiter_prefill_opt_vk_available = True
except Exception:
    aiter_chunk_gated_delta_rule_fwd_h_hip_fn = None
    aiter_chunk_gated_delta_rule_opt_vk = None
    aiter_chunk_gated_delta_rule_fwd_h_triton_opt_vk = None
    aiter_chunk_fwd_o_opt_vk = None
    aiter_fused_cumsum_kkt_opt_vk = None
    aiter_fused_solve_tril_recompute_w_u = None
    _aiter_prefill_opt_vk_available = False


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "0").lower() in ("1", "true", "yes", "on")


def should_use_aiter_prefill_opt_vk_for_seq_len(seq_len: int) -> bool:
    return aiter_prefill_opt_vk_enabled()


def _can_use_aiter_prefill_opt_vk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    initial_state_indices: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
) -> bool:
    if not should_use_aiter_prefill_opt_vk_for_seq_len(q.shape[1]):
        return False
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        return False
    if k.shape[-1] != 128 or v.shape[-1] != 128:
        return False
    if g.dtype != torch.float32 or beta.dtype != torch.float32:
        return False
    if initial_state is None or initial_state_indices is None or cu_seqlens is None:
        return False
    if initial_state.dtype not in (torch.float32, torch.bfloat16):
        return False
    if initial_state.shape[-2:] != (128, 128):
        return False
    if initial_state.shape[1] != v.shape[2]:
        return False
    return True


def aiter_prefill_opt_vk_enabled() -> bool:
    return (
        _is_hip
        and _env_enabled("SGLANG_GDN_PREFILL_OPT_VK")
        and _aiter_prefill_opt_vk_available
    )


def selected_aiter_prefill_opt_vk_k5_backend() -> str:
    backend = os.getenv("SGLANG_GDN_PREFILL_OPT_VK_K5", "auto").lower()
    if backend not in ("auto", "hip", "triton"):
        raise ValueError(
            "SGLANG_GDN_PREFILL_OPT_VK_K5 must be 'auto', 'hip', or 'triton', "
            f"got {backend!r}."
        )
    return backend


def _resolved_aiter_prefill_opt_vk_k5_backend(seq_len: int) -> str:
    backend = selected_aiter_prefill_opt_vk_k5_backend()
    if backend != "auto":
        return backend
    hip_min_t = int(os.getenv("SGLANG_GDN_PREFILL_OPT_VK_HIP_MIN_T", "8192"))
    return "hip" if seq_len >= hip_min_t else "triton"


def _chunk_gated_delta_rule_fwd_aiter_opt_vk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state_layout: int = 0,
    output_state_layout: int = 0,
    return_intermediate_h: bool = True,
):
    """Experimental env-gated opt-vk prefill core."""
    state_indices = initial_state_indices.to(torch.long)
    k5_backend = _resolved_aiter_prefill_opt_vk_k5_backend(k.shape[1])
    pool_indexed_vk_state = initial_state_layout == 1 and output_state_layout == 1
    if pool_indexed_vk_state:
        state_vk = initial_state
    elif initial_state_layout == 1:
        state_vk = initial_state[state_indices].contiguous()
    else:
        state_vk = initial_state[state_indices].transpose(-1, -2).contiguous()
    if not return_intermediate_h:
        o, final_state_vk = aiter_chunk_gated_delta_rule_opt_vk(
            q=q,
            k=k,
            v=v,
            o=v.new_empty(v.shape),
            g=g,
            beta=beta,
            scale=scale,
            initial_state=state_vk,
            initial_state_indices=initial_state_indices if pool_indexed_vk_state else None,
            output_final_state=True,
            inplace_final_state=True if pool_indexed_vk_state else None,
            cu_seqlens=cu_seqlens,
            use_chunk_hip=k5_backend == "hip",
            state_dtype=state_vk.dtype,
            use_exp2=True,
        )
        if pool_indexed_vk_state:
            pass
        elif output_state_layout == 1:
            initial_state.index_copy_(
                0, state_indices, final_state_vk.to(initial_state.dtype)
            )
        else:
            final_state_kv = final_state_vk.transpose(-1, -2).contiguous()
            initial_state.index_copy_(
                0, state_indices, final_state_kv.to(initial_state.dtype)
            )
        empty_h = q.new_empty(0)
        return g.new_empty(0), o, None, None, empty_h, None

    g_cumsum, A = aiter_fused_cumsum_kkt_opt_vk(
        k=k,
        beta=beta,
        g=g,
        cu_seqlens=cu_seqlens,
        use_exp2=True,
    )
    w, u = aiter_fused_solve_tril_recompute_w_u(
        A_raw=A,
        k=k,
        v=v,
        beta=beta,
        g_cumsum=g_cumsum,
        cu_seqlens=cu_seqlens,
        use_exp2=True,
    )

    if k5_backend == "triton" or pool_indexed_vk_state:
        h_vk, v_new, final_state_vk = aiter_chunk_gated_delta_rule_fwd_h_triton_opt_vk(
            k=k,
            w=w,
            u=u,
            g=g_cumsum,
            initial_state=state_vk,
            initial_state_indices=initial_state_indices if pool_indexed_vk_state else None,
            output_final_state=True,
            inplace_final_state=True if pool_indexed_vk_state else None,
            cu_seqlens=cu_seqlens,
            state_dtype=state_vk.dtype,
            use_exp2=True,
        )
    else:
        h_vk, v_new, final_state_vk = aiter_chunk_gated_delta_rule_fwd_h_hip_fn(
            k=k,
            w=w,
            u=u,
            g=g_cumsum,
            initial_state=state_vk,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            state_dtype=state_vk.dtype,
            use_exp2=True,
            g_head_major=True,
        )

    o = v.new_empty(v.shape)
    o = aiter_chunk_fwd_o_opt_vk(
        q=q,
        k=k,
        v=v_new,
        o=o,
        h=h_vk,
        g=g_cumsum,
        scale=scale,
        cu_seqlens=cu_seqlens,
        use_exp2=True,
    )

    if pool_indexed_vk_state:
        h = h_vk
    elif output_state_layout == 1:
        initial_state.index_copy_(0, state_indices, final_state_vk.to(initial_state.dtype))
        h = h_vk
    else:
        final_state_kv = final_state_vk.transpose(-1, -2).contiguous()
        initial_state.index_copy_(0, state_indices, final_state_kv.to(initial_state.dtype))
        h = h_vk.transpose(-1, -2).contiguous()

    return g_cumsum.transpose(1, 2).contiguous(), o, A, w, h, v_new


def chunk_gated_delta_rule_prefill_opt_vk_no_h(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = True,
):
    """Fast inference-only opt-vk prefill path for canonical VK state pools."""
    if use_qk_l2norm_in_kernel:
        if _is_hip:
            q, k = fused_l2norm_qk(q, k)
        else:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)
    k5_backend = _resolved_aiter_prefill_opt_vk_k5_backend(k.shape[1])
    g_cumsum, A = aiter_fused_cumsum_kkt_opt_vk(
        k=k,
        beta=beta,
        g=g,
        cu_seqlens=cu_seqlens,
        use_exp2=True,
    )
    w, u = aiter_fused_solve_tril_recompute_w_u(
        A_raw=A,
        k=k,
        v=v,
        beta=beta,
        g_cumsum=g_cumsum,
        cu_seqlens=cu_seqlens,
        use_exp2=True,
    )
    if k5_backend == "hip":
        h, v_new, _ = aiter_chunk_gated_delta_rule_fwd_h_hip_fn(
            k=k,
            w=w,
            u=u,
            g=g_cumsum,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            output_final_state=True,
            inplace_final_state=True,
            cu_seqlens=cu_seqlens,
            state_dtype=initial_state.dtype,
            use_exp2=True,
            g_head_major=True,
        )
    else:
        h, v_new, _ = aiter_chunk_gated_delta_rule_fwd_h_triton_opt_vk(
            k=k,
            w=w,
            u=u,
            g=g_cumsum,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            output_final_state=True,
            inplace_final_state=True,
            cu_seqlens=cu_seqlens,
            state_dtype=initial_state.dtype,
            use_exp2=True,
        )
    o = aiter_chunk_fwd_o_opt_vk(
        q=q,
        k=k,
        v=v_new,
        o=v.new_empty(v.shape),
        h=h,
        g=g_cumsum,
        scale=scale,
        cu_seqlens=cu_seqlens,
        use_exp2=True,
    )
    return o.to(q.dtype), None, q.new_empty(0)


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
    initial_state_layout: int = 0,
    output_state_layout: int = 0,
    return_intermediate_h: bool = True,
):
    B, T = q.shape[0], q.shape[1]
    Hv = g.shape[2]

    if _can_use_aiter_prefill_opt_vk(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
    ):
        return _chunk_gated_delta_rule_fwd_aiter_opt_vk(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            cu_seqlens=cu_seqlens,
            initial_state_layout=initial_state_layout,
            output_state_layout=output_state_layout,
            return_intermediate_h=return_intermediate_h,
        )

    if _is_hip and T >= 64:
        g, A = fused_cumsum_kkt(g, k, beta, chunk_size=64, cu_seqlens=cu_seqlens)
        chunk_indices_16 = (
            prepare_chunk_indices(cu_seqlens, 16) if cu_seqlens is not None else None
        )
        NT_16 = len(chunk_indices_16) if cu_seqlens is not None else triton.cdiv(T, 16)
        Ai16 = torch.empty(B, T, Hv, 16, device=A.device, dtype=torch.float32)
        solve_tril_16x16_kernel[(NT_16, B * Hv)](
            A=A,
            Ad=Ai16,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices_16,
            T=T,
            H=Hv,
            BT=64,
            num_warps=1,
            num_stages=4,
        )
        w, u = fused_merge_recompute(
            k, v, beta, g, A, Ai16, chunk_size=64, cu_seqlens=cu_seqlens
        )
    else:
        g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
        A = chunk_scaled_dot_kkt_fwd(
            k=k,
            beta=beta,
            g_cumsum=g,
            cu_seqlens=cu_seqlens,
            output_dtype=torch.float32,
        )
        A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
        w, u = recompute_w_u_fwd(
            k=k,
            v=v,
            beta=beta,
            A=A,
            g_cumsum=g,
            cu_seqlens=cu_seqlens,
        )
    h, v_new = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    if SUPPRESS_LEVEL < 3:
        return g, o, A, None, h, None
    elif SUPPRESS_LEVEL >= 3:
        return g, o, A, w, h, v_new


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        initial_state_indices: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor] = None,
        use_qk_l2norm_in_kernel: bool = False,
        initial_state_layout: int = 0,
        output_state_layout: int = 0,
        return_intermediate_h: bool = True,
    ):
        q_orig = q
        k_orig = k

        if use_qk_l2norm_in_kernel:
            if _is_hip:
                q, k = fused_l2norm_qk(q, k)
            else:
                q = l2norm_fwd(q)
                k = l2norm_fwd(k)

        g, o, A, w, h, v_new = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            cu_seqlens=cu_seqlens,
            initial_state_layout=initial_state_layout,
            output_state_layout=output_state_layout,
            return_intermediate_h=return_intermediate_h,
        )
        return o.to(q.dtype), h


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    initial_state_indices: torch.Tensor = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    initial_state_layout: int = 0,
    output_state_layout: int = 0,
    return_intermediate_h: bool = True,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    assert q.dtype == k.dtype == v.dtype
    assert (
        q.dtype != torch.float32
    ), "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    assert (
        len(beta.shape) == 3
    ), "beta must be of shape [B, T, H] if head_first=False, or [B, H, T] otherwise."

    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
        q, k, v, beta, g = map(
            lambda x: rearrange(x, "b h t ... -> b t h ..."), (q, k, v, beta, g)
        )
    # if not head_first and q.shape[1] < q.shape[2]:
    #     warnings.warn(
    #         f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
    #         "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
    #         "when head_first=False was specified. "
    #         "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
    #     )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if (
            initial_state_indices is not None
            and initial_state_indices.shape[0] != len(cu_seqlens) - 1
        ):
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state_indices.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if not return_intermediate_h and _can_use_aiter_prefill_opt_vk(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
    ):
        if use_qk_l2norm_in_kernel:
            if _is_hip:
                q, k = fused_l2norm_qk(q, k)
            else:
                q = l2norm_fwd(q)
                k = l2norm_fwd(k)
        if initial_state_layout == 1 and output_state_layout == 1:
            k5_backend = _resolved_aiter_prefill_opt_vk_k5_backend(k.shape[1])
            o, _ = aiter_chunk_gated_delta_rule_opt_vk(
                q=q,
                k=k,
                v=v,
                o=v.new_empty(v.shape),
                g=g,
                beta=beta,
                scale=scale,
                initial_state=initial_state,
                initial_state_indices=initial_state_indices,
                output_final_state=True,
                inplace_final_state=True,
                cu_seqlens=cu_seqlens,
                use_chunk_hip=k5_backend == "hip",
                state_dtype=initial_state.dtype,
                use_exp2=True,
            )
            return o.to(q.dtype), None, q.new_empty(0)
        _, o, _, _, h, _ = _chunk_gated_delta_rule_fwd_aiter_opt_vk(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            cu_seqlens=cu_seqlens,
            initial_state_layout=initial_state_layout,
            output_state_layout=output_state_layout,
            return_intermediate_h=False,
        )
        return o.to(q.dtype), None, h
    o, h = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        initial_state_indices,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
        initial_state_layout,
        output_state_layout,
        return_intermediate_h,
    )
    if head_first:
        o = rearrange(o, "b t h ... -> b h t ...")
    return o, None, h
