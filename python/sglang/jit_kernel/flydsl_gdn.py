import functools
from typing import Optional

import torch
import triton
import triton.language as tl


def _is_flydsl_available() -> bool:
    try:
        from aiter.ops.flydsl import is_flydsl_available

        return is_flydsl_available()
    except (ImportError, ModuleNotFoundError):
        return False


@functools.cache
def _compile_gdr_kernel(
    dtype: torch.dtype,
    sq: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    use_qk_l2norm: bool,
):
    from aiter.ops.flydsl.kernels.gdr_decode import Args, get_func

    args_ = Args(
        dtype=dtype,
        b=1,
        sq=sq,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        use_qk_l2norm=use_qk_l2norm,
    )
    return get_func(args_)


def warmup_flydsl_gdr(
    dtype: torch.dtype,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    use_qk_l2norm: bool = True,
):
    """Pre-compile kernel to avoid JIT during graph capture."""
    exe = _compile_gdr_kernel(
        dtype=dtype,
        sq=1,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        use_qk_l2norm=use_qk_l2norm,
    )

    bs = 1
    device = "cuda"
    q = torch.zeros((bs, 1, num_k_heads, head_k_dim), dtype=dtype, device=device)
    k = torch.zeros_like(q)
    v = torch.zeros((bs, 1, num_v_heads, head_v_dim), dtype=dtype, device=device)
    a = torch.zeros((bs, 1, num_v_heads), dtype=dtype, device=device)
    b_gate = torch.zeros_like(a)
    dt_bias = torch.zeros(num_v_heads, dtype=dtype, device=device)
    A_log = torch.zeros(num_v_heads, dtype=torch.float32, device=device)
    indices = torch.zeros(bs, dtype=torch.int32, device=device)
    state = torch.zeros(
        (bs + 1, num_v_heads, head_v_dim, head_k_dim),
        dtype=torch.float32, device=device,
    )
    out = torch.zeros_like(v)
    scale = float(1.0 / (head_k_dim ** 0.5))

    stream = torch.cuda.current_stream().cuda_stream
    exe(q, k, v, a, b_gate, dt_bias, A_log, indices, state, out, bs, scale, stream)
    torch.cuda.synchronize()


def flydsl_fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
) -> torch.Tensor:
    """FlyDSL GDR decode kernel wrapper. Expects state in [HV, V, K] layout."""
    assert softplus_beta == 1.0 and softplus_threshold == 20.0, (
        "FlyDSL GDR kernel has softplus_beta=1.0, threshold=20.0 compiled-in. "
        f"Got beta={softplus_beta}, threshold={softplus_threshold}"
    )

    B_q, T_q, H_k, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    N = initial_state_indices.shape[0]

    q = q.reshape(N, 1, H_k, K)
    k = k.reshape(N, 1, H_k, K)
    v = v.reshape(N, 1, HV, V)

    if a.dim() == 2:
        a = a.unsqueeze(1)
    elif a.dim() == 3 and a.shape[0] == 1:
        a = a.reshape(N, 1, HV)
    if b.dim() == 2:
        b = b.unsqueeze(1)
    elif b.dim() == 3 and b.shape[0] == 1:
        b = b.reshape(N, 1, HV)

    if initial_state_source.dim() == 1:
        pool_size = initial_state_source.numel() // (HV * K * V)
        state = initial_state_source.view(pool_size, HV, K, V)
    elif initial_state_source.dim() == 4:
        state = initial_state_source
    else:
        raise ValueError(
            f"Unexpected initial_state_source shape: {initial_state_source.shape}"
        )

    exe = _compile_gdr_kernel(
        dtype=q.dtype,
        sq=1,
        num_k_heads=H_k,
        num_v_heads=HV,
        head_k_dim=K,
        head_v_dim=V,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
    )

    out = torch.empty((N, 1, HV, V), dtype=q.dtype, device=q.device)
    stream = torch.cuda.current_stream().cuda_stream
    exe(q, k, v, a, b, dt_bias, A_log, initial_state_indices, state, out, N,
        float(1.0 / (K ** 0.5)), stream)

    return out.reshape(1, N, HV, V)


@triton.jit
def _pool_square_transpose_kernel(
    ptr,
    idx_ptr,
    n_idx,
    stride_L, stride_S, stride_H, stride_R,
    N_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile_id = tl.program_id(0)
    matrix_id = tl.program_id(1)

    NT: tl.constexpr = DIM // BLOCK
    tr = tile_id // NT
    tc = tile_id % NT
    if tr > tc:
        return

    head = matrix_id % N_HEADS
    rest = matrix_id // N_HEADS
    idx_pos = rest % n_idx
    layer = rest // n_idx

    slot = tl.load(idx_ptr + idx_pos)

    sL = tl.cast(stride_L, tl.int64)
    sS = tl.cast(stride_S, tl.int64)
    sH = tl.cast(stride_H, tl.int64)
    sR = tl.cast(stride_R, tl.int64)
    base = (
        ptr
        + tl.cast(layer, tl.int64) * sL
        + tl.cast(slot, tl.int64) * sS
        + tl.cast(head, tl.int64) * sH
    )

    ri = tl.arange(0, BLOCK)
    ci = tl.arange(0, BLOCK)

    off = tl.cast(tr * BLOCK + ri[:, None], tl.int64) * sR + tl.cast(tc * BLOCK + ci[None, :], tl.int64)
    off_t = tl.cast(tr * BLOCK + ci[None, :], tl.int64) * sR + tl.cast(tc * BLOCK + ri[:, None], tl.int64)

    if tr == tc:
        tile_t = tl.load(base + off_t)
        tl.store(base + off, tile_t)
    else:
        off_m = tl.cast(tc * BLOCK + ri[:, None], tl.int64) * sR + tl.cast(tr * BLOCK + ci[None, :], tl.int64)
        off_m_t = tl.cast(tc * BLOCK + ci[None, :], tl.int64) * sR + tl.cast(tr * BLOCK + ri[:, None], tl.int64)

        tile_at = tl.load(base + off_t)
        tile_bt = tl.load(base + off_m_t)

        tl.store(base + off, tile_bt)
        tl.store(base + off_m, tile_at)


def flydsl_pool_transpose_inplace(
    temporal_pool: torch.Tensor,
    indices: torch.Tensor,
):
    num_layers = temporal_pool.shape[0]
    num_heads = temporal_pool.shape[2]
    DIM = temporal_pool.shape[3]
    n_idx = indices.shape[0]
    if n_idx == 0:
        return

    BLOCK = min(32, DIM)
    assert DIM == temporal_pool.shape[4] and DIM % BLOCK == 0
    NT = DIM // BLOCK

    grid = (NT * NT, num_layers * n_idx * num_heads)
    _pool_square_transpose_kernel[grid](
        temporal_pool,
        indices,
        n_idx,
        temporal_pool.stride(0),
        temporal_pool.stride(1),
        temporal_pool.stride(2),
        temporal_pool.stride(3),
        N_HEADS=num_heads,
        DIM=DIM,
        BLOCK=BLOCK,
    )
