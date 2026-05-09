import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROCM_AVAILABLE = torch.version.hip is not None and torch.cuda.is_available()


def _load_common_ops():
    package_dir = Path(__file__).resolve().parents[1] / "python" / "sgl_kernel"
    matches = list(package_dir.glob("common_ops*.so"))
    assert matches, "common_ops extension is not built"
    torch.ops.load_library(str(matches[0]))


def _load_gdn_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "python" / "sgl_kernel" / "gdn.py"
    )
    assert module_path.exists(), "native HIP GDN wrapper is missing"
    spec = importlib.util.spec_from_file_location("sgl_kernel_gdn_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_hip_gdn_wrapper_exposes_decode_and_transpose_api():
    gdn = _load_gdn_module()

    assert gdn.LAYOUT_KV == 0
    assert gdn.LAYOUT_VK == 1
    assert callable(gdn.hip_fused_sigmoid_gating_delta_rule_update)
    assert callable(gdn.hip_state_transpose_inplace)
    assert callable(gdn.hip_state_transpose_inplace_multi_layer)


def test_sglang_backend_uses_explicit_hip_gdn_env_gate():
    repo_root = Path(__file__).resolve().parents[2]
    environ_source = (repo_root / "python" / "sglang" / "srt" / "environ.py").read_text()
    backend_source = (
        repo_root
        / "python"
        / "sglang"
        / "srt"
        / "layers"
        / "attention"
        / "hybrid_linear_attn_backend.py"
    ).read_text()

    assert "SGLANG_USE_HIP_GDN_DECODE = EnvBool(False)" in environ_source
    assert "_use_hip_gdn_decode = _is_hip and Envs.SGLANG_USE_HIP_GDN_DECODE.get()" in backend_source
    assert "if _use_hip_gdn_decode:" in backend_source
    assert "USE_HIP_LINEAR_ATTN" not in backend_source
    assert "aiter.ops.hip.gated_delta_net" not in backend_source


def test_hip_gdn_decode_wrapper_calls_native_op(monkeypatch):
    gdn = _load_gdn_module()
    calls = []

    def fake_decode(*args):
        calls.append(args)

    monkeypatch.setattr(
        torch.ops,
        "sgl_kernel",
        SimpleNamespace(hip_gdn_decode_asm=fake_decode),
        raising=False,
    )

    q = torch.randn(2, 1, 2, 128, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(2, 1, 8, 128, dtype=torch.bfloat16)
    a = torch.randn(2, 1, 8, dtype=torch.bfloat16)
    b = torch.randn(2, 1, 8, dtype=torch.bfloat16)
    dt_bias = torch.randn(8, dtype=torch.float32)
    a_log = torch.randn(8, dtype=torch.float32)
    state = torch.randn(4, 8, 128, 128, dtype=torch.float32)
    indices = torch.tensor([0, 1], dtype=torch.int64)

    out = gdn.hip_fused_sigmoid_gating_delta_rule_update(
        a_log,
        a,
        dt_bias,
        1.0,
        20.0,
        q,
        k,
        v,
        b,
        state,
        indices,
        scale=None,
        use_qk_l2norm_in_kernel=True,
    )

    assert out.shape == v.shape
    assert out.dtype == v.dtype
    assert len(calls) == 1
    _, _, _, _, _, native_dt_bias, _, native_indices, native_state, native_out = calls[
        0
    ][:10]
    assert native_dt_bias.dtype == torch.bfloat16
    assert native_indices.dtype == torch.int32
    assert native_state is state
    assert native_out is out


def _gdn_decode_reference(
    A_log,
    a,
    dt_bias,
    q,
    k,
    v,
    b,
    state,
    indices,
    scale,
    use_qk_l2norm,
):
    output = torch.empty_like(v)
    batch_size, seq_length, num_k_heads, head_k_dim = q.shape
    num_v_heads = v.shape[2]
    gva_ratio = num_v_heads // num_k_heads

    for batch_idx in range(batch_size):
        pool_idx = int(indices[batch_idx].item())
        if pool_idx < 0:
            continue
        for hv_idx in range(num_v_heads):
            hk_idx = hv_idx // gva_ratio
            state_matrix = state[pool_idx, hv_idx]
            for sq in range(seq_length):
                q_vec = q[batch_idx, sq, hk_idx].float()
                k_vec = k[batch_idx, sq, hk_idx].float()
                if use_qk_l2norm:
                    q_vec = q_vec * (scale * torch.rsqrt(torch.sum(q_vec * q_vec) + 1e-6))
                    k_vec = k_vec * torch.rsqrt(torch.sum(k_vec * k_vec) + 1e-6)
                else:
                    q_vec = q_vec * scale

                gate_input = a[batch_idx, sq, hv_idx].float() + dt_bias[hv_idx].float()
                softplus = torch.where(
                    gate_input <= 20.0,
                    torch.log1p(torch.exp(gate_input)),
                    gate_input,
                )
                exp_g = torch.exp(-torch.exp(A_log[hv_idx].float()) * softplus)
                beta = torch.sigmoid(b[batch_idx, sq, hv_idx].float())

                state_before_decay = state_matrix
                dot_kq = torch.sum(k_vec * q_vec)
                res_hk = (state_before_decay * k_vec.unsqueeze(0)).sum(dim=1) * exp_g
                res_hq = (state_before_decay * q_vec.unsqueeze(0)).sum(dim=1) * exp_g
                vn = (v[batch_idx, sq, hv_idx].float() - res_hk) * beta
                output[batch_idx, sq, hv_idx] = (res_hq + vn * dot_kq).to(v.dtype)
                state_matrix.copy_(state_before_decay * exp_g + vn.unsqueeze(1) * k_vec)

    return output, state


@pytest.mark.skipif(not ROCM_AVAILABLE, reason="ROCm GPU is required")
def test_hip_gdn_multi_layer_transpose_round_trip():
    _load_common_ops()
    gdn = _load_gdn_module()

    torch.cuda.set_device(0)
    num_layers, num_slots, num_v_heads = 2, 4, 2
    state = torch.arange(
        num_layers * num_slots * num_v_heads * 128 * 128,
        device="cuda",
        dtype=torch.float32,
    ).reshape(num_layers, num_slots, num_v_heads, 128, 128)
    original = state.clone()
    indices = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    slot_layout = torch.zeros(num_slots, dtype=torch.int8, device="cuda")

    gdn.hip_state_transpose_inplace_multi_layer(
        state, indices, slot_layout, gdn.LAYOUT_VK, num_layers, num_v_heads
    )
    torch.cuda.synchronize()

    expected = original.clone()
    expected[:, [0, 2]] = expected[:, [0, 2]].transpose(-1, -2)
    torch.testing.assert_close(state, expected, rtol=0, atol=0)
    assert slot_layout.cpu().tolist() == [
        gdn.LAYOUT_VK,
        gdn.LAYOUT_KV,
        gdn.LAYOUT_VK,
        gdn.LAYOUT_KV,
    ]

    gdn.hip_state_transpose_inplace_multi_layer(
        state, indices, slot_layout, gdn.LAYOUT_KV, num_layers, num_v_heads
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(state, original, rtol=0, atol=0)
    assert slot_layout.cpu().tolist() == [
        gdn.LAYOUT_KV,
        gdn.LAYOUT_KV,
        gdn.LAYOUT_KV,
        gdn.LAYOUT_KV,
    ]


@pytest.mark.skipif(not ROCM_AVAILABLE, reason="ROCm GPU is required")
def test_hip_gdn_decode_matches_reference_for_supported_shape():
    _load_common_ops()
    gdn = _load_gdn_module()

    torch.cuda.set_device(0)
    torch.manual_seed(0)
    batch_size, seq_length = 2, 1
    num_k_heads, num_v_heads = 2, 4
    head_dim = 128
    scale = head_dim**-0.5

    q = torch.randn(
        batch_size, seq_length, num_k_heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    k = torch.randn_like(q)
    v = torch.randn(
        batch_size, seq_length, num_v_heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    a = torch.randn(batch_size, seq_length, num_v_heads, dtype=torch.bfloat16, device="cuda")
    b = torch.randn_like(a)
    dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device="cuda")
    A_log = torch.randn(num_v_heads, dtype=torch.float32, device="cuda")
    state = torch.randn(3, num_v_heads, head_dim, head_dim, dtype=torch.float32, device="cuda")
    indices = torch.tensor([0, 2], dtype=torch.int32, device="cuda")

    state_ref = state.clone()
    expected, expected_state = _gdn_decode_reference(
        A_log,
        a,
        dt_bias.to(torch.bfloat16),
        q,
        k,
        v,
        b,
        state_ref,
        indices,
        scale,
        True,
    )

    actual = gdn.hip_fused_sigmoid_gating_delta_rule_update(
        A_log,
        a,
        dt_bias,
        1.0,
        20.0,
        q,
        k,
        v,
        b,
        state,
        indices,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(state, expected_state, rtol=2e-2, atol=2e-2)

