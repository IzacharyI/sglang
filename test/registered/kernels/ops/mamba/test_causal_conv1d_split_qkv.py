"""Focused causal-conv QKV-split kernel parity tests.

This intentionally avoids ``test/registered/layers/mamba`` so it does not pull
the Mamba test conftest or ``sglang.test.test_utils`` dependency tree.
"""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkv_split_gdn_prefill,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    PAD_SLOT_ID,
    causal_conv1d_fn,
    causal_conv1d_fn_split_qkv,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=20, stage="jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Causal-conv Triton kernel tests require a CUDA or ROCm GPU.",
)


def _torch_varlen_causal_conv_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    initial_states: torch.Tensor,
    seq_lens_cpu: list[int],
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    activation: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent grouped ``F.conv1d`` reference with indexed state updates."""
    dim, _ = x.shape
    width = weight.shape[1]
    state_len = width - 1
    packed = torch.zeros(x.shape[1], dim, dtype=x.dtype, device=x.device)
    final_states = initial_states.clone()

    start = 0
    for seq_idx, seq_len in enumerate(seq_lens_cpu):
        end = start + seq_len
        cache_idx = int(cache_indices[seq_idx].item())
        if cache_idx == PAD_SLOT_ID:
            start = end
            continue

        x_seq = x[:, start:end].unsqueeze(0)
        use_initial_state = has_initial_state is not None and bool(
            has_initial_state[seq_idx].item()
        )
        history = (
            final_states[cache_idx].unsqueeze(0)
            if use_initial_state
            else x_seq.new_zeros((1, dim, state_len))
        )
        conv_input = torch.cat((history, x_seq), dim=-1)
        out = F.conv1d(
            conv_input,
            weight.unsqueeze(1),
            bias,
            padding=0,
            groups=dim,
        )[..., :seq_len]
        if activation in ("silu", "swish"):
            out = F.silu(out)
        packed[start:end] = out.squeeze(0).transpose(0, 1)
        final_states[cache_idx] = conv_input[..., -state_len:].squeeze(0)
        start = end

    return packed, final_states


def _split_torch_reference(
    packed: torch.Tensor,
    *,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_q_dim: int,
    head_k_dim: int,
    head_v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_dim = num_q_heads * head_q_dim
    k_dim = num_k_heads * head_k_dim
    q, k, v = torch.split(packed, [q_dim, k_dim, num_v_heads * head_v_dim], dim=-1)
    return (
        q.view(1, -1, num_q_heads, head_q_dim).contiguous(),
        k.view(1, -1, num_k_heads, head_k_dim).contiguous(),
        v.view(1, -1, num_v_heads, head_v_dim).contiguous(),
    )


@pytest.mark.parametrize(
    ("width", "activation", "with_bias", "with_initial_state"),
    [
        pytest.param(2, None, False, False, id="width2-linear-no-state"),
        pytest.param(3, "silu", True, True, id="width3-silu-initial-state"),
        pytest.param(4, "silu", True, False, id="width4-silu-no-state"),
    ],
)
def test_causal_conv1d_split_qkv_matches_packed_reference(
    width: int,
    activation: str | None,
    with_bias: bool,
    with_initial_state: bool,
) -> None:
    """Match packed causal-conv + GDN split for varlen indexed prefill."""
    torch.manual_seed(7)
    device = torch.device("cuda")

    num_q_heads, head_q_dim = 4, 32
    num_k_heads, head_k_dim = 3, 32
    num_v_heads, head_v_dim = 4, 40
    q_dim = num_q_heads * head_q_dim
    k_dim = num_k_heads * head_k_dim
    v_dim = num_v_heads * head_v_dim
    dim = q_dim + k_dim + v_dim

    # The final two entries are padded slots. Keep them after real requests so
    # their skipped output does not overlap the valid-token comparison.
    seq_lens_cpu = [3, 1, 4, 2, 3]
    valid_batch_size = 3
    valid_tokens = sum(seq_lens_cpu[:valid_batch_size])
    total_tokens = sum(seq_lens_cpu)
    query_start_loc = torch.tensor(
        [0, 3, 4, 8, 10, 13], dtype=torch.int32, device=device
    )
    cache_indices = torch.tensor(
        [4, 1, 8, PAD_SLOT_ID, PAD_SLOT_ID], dtype=torch.int32, device=device
    )
    has_initial_state = (
        torch.tensor([True, True, True, False, False], dtype=torch.bool, device=device)
        if with_initial_state
        else None
    )

    x_token_major = torch.randn(total_tokens, dim, dtype=torch.bfloat16, device=device)
    x = x_token_major.transpose(0, 1)
    weight = torch.randn(dim, width, dtype=torch.bfloat16, device=device)
    bias = torch.randn(dim, dtype=torch.bfloat16, device=device) if with_bias else None

    initial_states = torch.randn(
        11, dim, width - 1, dtype=torch.bfloat16, device=device
    )
    torch_reference_packed, torch_reference_states = (
        _torch_varlen_causal_conv_reference(
            x,
            weight,
            bias,
            initial_states,
            seq_lens_cpu,
            cache_indices,
            has_initial_state,
            activation,
        )
    )
    torch_reference_q, torch_reference_k, torch_reference_v = _split_torch_reference(
        torch_reference_packed,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_q_dim=head_q_dim,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
    )
    reference_states = initial_states.clone()
    split_states = initial_states.clone()

    packed = causal_conv1d_fn(
        x,
        weight,
        bias,
        reference_states,
        query_start_loc,
        seq_lens_cpu,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation=activation,
    ).transpose(0, 1)
    expected_q, expected_k, expected_v = fused_qkv_split_gdn_prefill(
        packed,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_q_dim,
        head_k_dim,
        head_v_dim,
    )

    q, k, v = causal_conv1d_fn_split_qkv(
        x,
        weight,
        bias,
        split_states,
        query_start_loc,
        seq_lens_cpu,
        q_dim=q_dim,
        k_dim=k_dim,
        v_dim=v_dim,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_q_dim=head_q_dim,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation=activation,
    )

    assert q.shape == expected_q.shape == (1, total_tokens, num_q_heads, head_q_dim)
    assert k.shape == expected_k.shape == (1, total_tokens, num_k_heads, head_k_dim)
    assert v.shape == expected_v.shape == (1, total_tokens, num_v_heads, head_v_dim)
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert torch.equal(q[:, :valid_tokens], expected_q[:, :valid_tokens])
    assert torch.equal(k[:, :valid_tokens], expected_k[:, :valid_tokens])
    assert torch.equal(v[:, :valid_tokens], expected_v[:, :valid_tokens])
    torch.testing.assert_close(
        packed[:valid_tokens],
        torch_reference_packed[:valid_tokens],
        rtol=1e-2,
        atol=5e-2,
    )
    torch.testing.assert_close(
        q[:, :valid_tokens],
        torch_reference_q[:, :valid_tokens],
        rtol=1e-2,
        atol=5e-2,
    )
    torch.testing.assert_close(
        k[:, :valid_tokens],
        torch_reference_k[:, :valid_tokens],
        rtol=1e-2,
        atol=5e-2,
    )
    torch.testing.assert_close(
        v[:, :valid_tokens],
        torch_reference_v[:, :valid_tokens],
        rtol=1e-2,
        atol=5e-2,
    )

    # The fused store path must retain bitwise-identical indexed state writes.
    assert torch.equal(split_states, reference_states)
    assert torch.equal(reference_states, torch_reference_states)
    unused_cache_lines = torch.tensor(
        [0, 2, 3, 5, 6, 7, 9, 10], dtype=torch.long, device=device
    )
    assert torch.equal(
        split_states[unused_cache_lines], initial_states[unused_cache_lines]
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
