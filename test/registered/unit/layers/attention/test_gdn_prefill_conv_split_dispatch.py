"""Dispatch-matrix tests for HIP GDN prefill causal-conv QKV stores."""

import sys

import pytest
import torch

from sglang.srt.layers.attention.linear.gdn_backend import (
    MAX_FUSED_QKV_SPLIT_DIM,
    can_use_hip_gdn_prefill_conv_qkv_split,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=5, stage="jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Dispatch eligibility tests require a CUDA or ROCm tensor.",
)


def _eligibility_kwargs(
    *,
    forward_mode: ForwardMode = ForwardMode.EXTEND,
    qkv_dim: int = 384,
    width: int = 4,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    num_q_heads, head_q_dim = 4, 32
    num_k_heads, head_k_dim = 3, 32
    num_v_heads, head_v_dim = 4, 40
    q_dim = num_q_heads * head_q_dim
    k_dim = num_k_heads * head_k_dim
    v_dim = qkv_dim - q_dim - k_dim
    assert v_dim > 0 and v_dim % num_v_heads == 0
    head_v_dim = v_dim // num_v_heads
    return {
        "forward_mode": forward_mode,
        "is_hip_platform": True,
        "mixed_qkv": torch.empty(3, qkv_dim, device=device, dtype=dtype).transpose(
            0, 1
        ),
        "weight": torch.empty(qkv_dim, width, device=device, dtype=dtype),
        "bias": torch.empty(qkv_dim, device=device, dtype=dtype),
        "conv_states": torch.empty(8, qkv_dim, width - 1, device=device, dtype=dtype),
        "query_start_loc": torch.tensor([0, 1, 3], dtype=torch.int32, device=device),
        "cache_indices": torch.tensor([0, 1], dtype=torch.int32, device=device),
        "has_initial_state": torch.tensor([False, True], device=device),
        "q_dim": q_dim,
        "k_dim": k_dim,
        "v_dim": v_dim,
        "num_q_heads": num_q_heads,
        "num_k_heads": num_k_heads,
        "num_v_heads": num_v_heads,
        "head_q_dim": head_q_dim,
        "head_k_dim": head_k_dim,
        "head_v_dim": head_v_dim,
        "enable_page_major_kv_layout": False,
        "needs_state_gather": False,
    }


def test_selects_only_ordinary_hip_extend_with_cuda_tensor() -> None:
    assert can_use_hip_gdn_prefill_conv_qkv_split(**_eligibility_kwargs())


@pytest.mark.parametrize(
    "forward_mode",
    [
        ForwardMode.TARGET_VERIFY,
        ForwardMode.MIXED,
        ForwardMode.SPLIT_PREFILL,
        ForwardMode.DLLM_EXTEND,
    ],
)
def test_rejects_nonordinary_extend_forward_modes(forward_mode: ForwardMode) -> None:
    assert not can_use_hip_gdn_prefill_conv_qkv_split(
        **_eligibility_kwargs(forward_mode=forward_mode)
    )


@pytest.mark.parametrize(
    "override",
    [
        {"is_hip_platform": False},
        {"enable_page_major_kv_layout": True},
        {"needs_state_gather": True},
        {"width": 5},
    ],
)
def test_rejects_unsafe_platform_or_layout(override: dict) -> None:
    kwargs = _eligibility_kwargs(
        width=override.get("width", 4),
    )
    kwargs.update({key: value for key, value in override.items() if key != "width"})
    assert not can_use_hip_gdn_prefill_conv_qkv_split(**kwargs)


def test_rejects_cpu_engine_tensor_on_rocm() -> None:
    assert not can_use_hip_gdn_prefill_conv_qkv_split(
        **_eligibility_kwargs(device="cpu")
    )


def test_rejects_oversized_qkv() -> None:
    qkv_dim = MAX_FUSED_QKV_SPLIT_DIM + 4
    assert not can_use_hip_gdn_prefill_conv_qkv_split(
        **_eligibility_kwargs(qkv_dim=qkv_dim)
    )


def test_rejects_unvalidated_activation_dtype() -> None:
    assert not can_use_hip_gdn_prefill_conv_qkv_split(
        **_eligibility_kwargs(dtype=torch.float16)
    )


def test_rejects_mismatched_weight_or_state_dtype() -> None:
    kwargs = _eligibility_kwargs()
    kwargs["weight"] = kwargs["weight"].float()
    assert not can_use_hip_gdn_prefill_conv_qkv_split(**kwargs)

    kwargs = _eligibility_kwargs()
    kwargs["conv_states"] = kwargs["conv_states"].float()
    assert not can_use_hip_gdn_prefill_conv_qkv_split(**kwargs)


def test_rejects_unvalidated_index_dtypes() -> None:
    kwargs = _eligibility_kwargs()
    kwargs["query_start_loc"] = kwargs["query_start_loc"].to(torch.int64)
    assert not can_use_hip_gdn_prefill_conv_qkv_split(**kwargs)

    kwargs = _eligibility_kwargs()
    kwargs["cache_indices"] = kwargs["cache_indices"].to(torch.int64)
    assert not can_use_hip_gdn_prefill_conv_qkv_split(**kwargs)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
