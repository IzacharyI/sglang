"""Muse Spark attention block as ONE launch at target verify (TP4, gfx950, ``SGLANG_ROCM_MUSE_ATTNBLK_MEGA``).

QKV projection, QK norm / RoPE / FP8 KV write, verify attention, o norm / gate, o_proj, the TP4 reduction and the
attention boundary (post-norm, gated residual, FFN pre-norm, FP8 copy) run in one persistent kernel
(``aiter.ops.muse_attnblk_mega``) for verify batches of up to 8 requests. One kernel instance per layer kind
(full / sliding window); every rank issues the calls of an instance in the same order.
"""

from __future__ import annotations

import torch

from sglang.srt.model_executor.forward_context import get_attn_backend

_PAGE_SIZE = 16


class _LayerOperands:
    """Constant operands of one decoder layer. The FP8 QKV weight is made after ``post_load_weights``, so the
    weights bind on first use."""

    def __init__(self, layer):
        attn = layer.self_attn
        assert attn.num_heads == 32 and attn.num_kv_heads == 4 and attn.head_dim == 64
        self.attn = attn
        boundary = layer._fused_attn_boundary
        boundary._ensure_cached_weights(torch.bfloat16)
        self.alpha = boundary._cc_alpha.reshape(-1).float().contiguous()
        self.beta = boundary._cc_beta.reshape(-1).float().contiguous()
        self.pre_norm_weight = boundary._cc_pre_weight.reshape(-1).float().contiguous()
        self.post_eps, self.pre_eps = (
            float(boundary._post_norm.eps),
            float(boundary._pre_norm.eps),
        )
        self.norm_eps, self.q_scale, self.sm_scale = (
            float(attn.norm_eps),
            float(attn.scale_query_by),
            float(attn.scaling),
        )
        self.cos_sin = (
            attn.rotary_emb.cos_sin_cache.float().contiguous()
            if attn.rotary_emb is not None
            else None
        )
        window = attn.attn.sliding_window_size
        self.window = int(window) if window is not None and window > 0 else 0
        self.layer_id = attn.attn.layer_id
        self.w_qkv = self.w_qkv_scale = self.w_o = None

    def bind_weights(self):
        if self.w_qkv is None:
            w_qkv, w_qkv_scale = self.attn.qkv_proj.small_m_fp8_weight
            self.w_qkv, self.w_qkv_scale = (
                w_qkv.contiguous(),
                w_qkv_scale.float().contiguous(),
            )
            self.w_o = self.attn.o_proj.weight.data.contiguous()


def _verify_kv(layer_id: int, window: int, forward_batch):
    """The paged FP8 KV operands of this verify batch, or None when the backend is not the unified FP8 verify."""
    backend = get_attn_backend()
    md = backend.forward_metadata
    if not (
        backend.page_size == _PAGE_SIZE
        and backend.kv_cache_dtype == torch.float8_e4m3fn
        and md.max_q_len == 8
        and backend._use_unified_verify
    ):
        return None
    k_cache, v_cache = backend.token_to_kv_pool.get_kv_buffer(layer_id)
    page_table = md.kv_indices
    swa_mapping = None
    if window:
        if md.swa_page_table is not None:
            page_table = md.swa_page_table
        if backend.use_sliding_window_kv_pool:
            swa_mapping = backend.swa_kv_pool.full_to_swa_index_mapping
    seq_lens = (
        md.verify_seqlen_k
        if md.verify_seqlen_k is not None
        else forward_batch.seq_lens + md.max_q_len
    )
    bs = seq_lens.shape[0]
    assert page_table.dtype == torch.int32 and page_table.stride(1) == 1
    assert (
        forward_batch.out_cache_loc.dtype == torch.int64
        and forward_batch.positions.dtype == torch.int64
    )
    return dict(
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=page_table[:bs],
        seq_lens=seq_lens.to(torch.int32).contiguous(),
        out_cache_loc=forward_batch.out_cache_loc,
        positions=forward_batch.positions,
        swa_mapping=swa_mapping,
    )


class MuseAttnBlockMega:
    """The one-launch attention block of the given decoder layers (collective construction, outside capture)."""

    def __init__(self, cpu_group, device: torch.device, layers):
        from aiter.ops.muse_attnblk_mega import MuseAttnBlockMega as Kernel

        self._layers = {id(layer): _LayerOperands(layer) for layer in layers}
        self._kernels = {
            sliding: Kernel(cpu_group, device, sliding=sliding)
            for sliding in (False, True)
        }
        self._applies = Kernel.applies

    def __call__(
        self,
        layer,
        forward_batch,
        x_fp8: tuple[torch.Tensor, torch.Tensor],
        stream: torch.Tensor,
    ):
        """(FFN input bf16, new stream fp32, FFN input FP8, its scales), or None when the batch is not covered."""
        ops = self._layers.get(id(layer))
        if ops is None:
            return None
        kv = _verify_kv(ops.layer_id, ops.window, forward_batch)
        if kv is None or not self._applies(x_fp8[0].shape[0], kv["seq_lens"].shape[0]):
            return None
        ops.bind_weights()
        kernel = self._kernels[ops.window > 0]
        return kernel(
            x_fp8[0].contiguous(), x_fp8[1], stream.contiguous(),
            w_qkv=ops.w_qkv, w_qkv_scale=ops.w_qkv_scale, w_o=ops.w_o, cos_sin_cache=ops.cos_sin,
            alpha=ops.alpha, beta=ops.beta, pre_norm_weight=ops.pre_norm_weight, norm_eps=ops.norm_eps,
            q_scale=ops.q_scale, sm_scale=ops.sm_scale, post_eps=ops.post_eps, pre_eps=ops.pre_eps,
            window=ops.window, **kv,
        )  # fmt: skip
