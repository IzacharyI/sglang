"""Benchmark GDN prefill causal-conv + direct QKV stores.

The baseline is the current packed causal-conv output followed by
``fused_qkv_split_gdn_prefill``. The fused path writes the same contiguous
``[1, T, H, D]`` Q/K/V tensors directly from causal-conv.

Example:
    PYTHONPATH=python python3 test/registered/kernels/benchmark/attention/\
bench_gdn_prefill_conv_split.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import triton
import triton.testing

from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkv_split_gdn_prefill,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_fn_split_qkv,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(
    est_time=30,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
)
register_amd_ci(est_time=30, stage="jit-kernel-benchmark", runner_config="amd")


@dataclass(frozen=True)
class Qwen35LocalShape:
    """Representative Qwen3.5 GDN local tensor-parallel dimensions."""

    name: str
    num_q_heads: int
    num_k_heads: int
    num_v_heads: int
    head_q_dim: int = 128
    head_k_dim: int = 128
    head_v_dim: int = 128

    @property
    def q_dim(self) -> int:
        return self.num_q_heads * self.head_q_dim

    @property
    def k_dim(self) -> int:
        return self.num_k_heads * self.head_k_dim

    @property
    def v_dim(self) -> int:
        return self.num_v_heads * self.head_v_dim

    @property
    def qkv_dim(self) -> int:
        return self.q_dim + self.k_dim + self.v_dim


@dataclass
class BenchmarkInputs:
    x: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor | None
    initial_states: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens_cpu: list[int]
    cache_indices: torch.Tensor
    has_initial_state: torch.Tensor
    shape: Qwen35LocalShape


QWEN35_LOCAL_SHAPES = (
    # Qwen3.5-35B-A3B style global Hq/Hk/Hv=16/16/32 split over TP=8 and TP=4.
    Qwen35LocalShape("qwen3.5-35b-tp8", num_q_heads=2, num_k_heads=2, num_v_heads=4),
    Qwen35LocalShape("qwen3.5-35b-tp4", num_q_heads=4, num_k_heads=4, num_v_heads=8),
    # Qwen3.5-397B-A17B global Hq/Hk/Hv=16/16/64.
    Qwen35LocalShape("qwen3.5-397b-tp8", num_q_heads=2, num_k_heads=2, num_v_heads=8),
    Qwen35LocalShape("qwen3.5-397b-tp4", num_q_heads=4, num_k_heads=4, num_v_heads=16),
)


def _make_uneven_seq_lens(total_tokens: int) -> list[int]:
    """Create a four-request varlen batch that sums to ``total_tokens``."""
    first = max(1, total_tokens // 8)
    second = max(1, total_tokens // 5)
    third = max(1, total_tokens // 3)
    fourth = total_tokens - first - second - third
    assert fourth > 0
    return [first, second, third, fourth]


def _make_inputs(
    shape: Qwen35LocalShape,
    total_tokens: int,
    width: int,
    device: torch.device,
) -> BenchmarkInputs:
    seq_lens_cpu = _make_uneven_seq_lens(total_tokens)
    query_start_loc = torch.tensor(
        [0, *torch.tensor(seq_lens_cpu).cumsum(0).tolist()],
        device=device,
        dtype=torch.int32,
    )
    batch_size = len(seq_lens_cpu)
    pool_size = batch_size * 4
    cache_indices = torch.randperm(pool_size, device=device)[:batch_size].to(
        torch.int32
    )
    x = torch.randn(
        total_tokens, shape.qkv_dim, device=device, dtype=torch.bfloat16
    ).transpose(0, 1)
    return BenchmarkInputs(
        x=x,
        weight=torch.randn(shape.qkv_dim, width, device=device, dtype=torch.bfloat16),
        # Qwen3.5 GDN convolution is biasless.
        bias=None,
        initial_states=torch.randn(
            pool_size, shape.qkv_dim, width - 1, device=device, dtype=torch.bfloat16
        ),
        query_start_loc=query_start_loc,
        seq_lens_cpu=seq_lens_cpu,
        cache_indices=cache_indices,
        has_initial_state=torch.ones(batch_size, device=device, dtype=torch.bool),
        shape=shape,
    )


def _baseline(
    inputs: BenchmarkInputs, conv_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Current packed causal-conv plus post-conv GDN QKV split."""
    mixed_qkv = causal_conv1d_fn(
        inputs.x,
        inputs.weight,
        inputs.bias,
        conv_states,
        inputs.query_start_loc,
        inputs.seq_lens_cpu,
        cache_indices=inputs.cache_indices,
        has_initial_state=inputs.has_initial_state,
        activation="silu",
    ).transpose(0, 1)
    shape = inputs.shape
    return fused_qkv_split_gdn_prefill(
        mixed_qkv,
        shape.num_q_heads,
        shape.num_k_heads,
        shape.num_v_heads,
        shape.head_q_dim,
        shape.head_k_dim,
        shape.head_v_dim,
    )


def _fused(
    inputs: BenchmarkInputs, conv_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Causal-conv with direct contiguous GDN QKV stores."""
    shape = inputs.shape
    return causal_conv1d_fn_split_qkv(
        inputs.x,
        inputs.weight,
        inputs.bias,
        conv_states,
        inputs.query_start_loc,
        inputs.seq_lens_cpu,
        shape.q_dim,
        shape.k_dim,
        shape.v_dim,
        num_q_heads=shape.num_q_heads,
        num_k_heads=shape.num_k_heads,
        num_v_heads=shape.num_v_heads,
        head_q_dim=shape.head_q_dim,
        head_k_dim=shape.head_k_dim,
        head_v_dim=shape.head_v_dim,
        cache_indices=inputs.cache_indices,
        has_initial_state=inputs.has_initial_state,
        activation="silu",
    )


def _validate(inputs: BenchmarkInputs) -> None:
    """Ensure the two timed paths still agree before measuring them."""
    baseline_states = inputs.initial_states.clone()
    fused_states = inputs.initial_states.clone()
    expected = _baseline(inputs, baseline_states)
    actual = _fused(inputs, fused_states)
    for actual_tensor, expected_tensor in zip(actual, expected):
        assert actual_tensor.is_contiguous()
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
    assert torch.equal(fused_states, baseline_states)


def _warmup(fn, iterations: int) -> None:
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()


def _median_us(fn, warmup: int, rep: int) -> float:
    """Return a median benchmark result in microseconds, not milliseconds."""
    median_ms = triton.testing.do_bench(
        fn,
        warmup=warmup,
        rep=rep,
        return_mode="median",
    )
    return float(median_ms) * 1000.0


def _run_case(
    shape: Qwen35LocalShape,
    total_tokens: int,
    width: int,
    warmup: int,
    rep: int,
    check_correctness: bool,
    device: torch.device,
) -> tuple[float, float]:
    inputs = _make_inputs(shape, total_tokens, width, device)
    if check_correctness:
        _validate(inputs)

    baseline_states = inputs.initial_states.clone()
    fused_states = inputs.initial_states.clone()

    def run_baseline() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _baseline(inputs, baseline_states)

    def run_fused() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _fused(inputs, fused_states)

    # Compile both specializations and make the measured runs independent of
    # first-use allocation/cache effects.
    _warmup(run_baseline, warmup)
    _warmup(run_fused, warmup)
    return _median_us(run_baseline, warmup, rep), _median_us(run_fused, warmup, rep)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark fused GDN prefill causal-conv QKV stores."
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=[64, 256, 1024, 8192])
    parser.add_argument("--width", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("-f", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Skip the per-shape exact Q/K/V and conv-state check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA or ROCm GPU is required for this benchmark.")

    torch.manual_seed(42)
    device = torch.device("cuda")
    runtime = torch.version.hip or torch.version.cuda or "unknown"
    print(f"device: {torch.cuda.get_device_name(device)}")
    print(f"torch: {torch.__version__}")
    print(f"triton: {triton.__version__}")
    print(f"runtime: {runtime}")
    print(
        "timing: triton.testing.do_bench(return_mode='median') returns ms; "
        "this table reports ms * 1000 as us"
    )
    print(f"warmup={args.warmup}, rep={args.rep}, varlen_batch=4, width={args.width}")
    print(
        f"{'shape':<14} {'tokens':>6} {'seq_lens':<24} "
        f"{'packed+split us':>16} {'fused us':>10} {'speedup':>9}"
    )

    for shape in QWEN35_LOCAL_SHAPES:
        for total_tokens in args.tokens:
            seq_lens = _make_uneven_seq_lens(total_tokens)
            baseline_us, fused_us = _run_case(
                shape,
                total_tokens,
                args.width,
                args.warmup,
                args.rep,
                not args.skip_correctness,
                device,
            )
            speedup = baseline_us / fused_us
            print(
                f"{shape.name:<14} {total_tokens:>6} {str(seq_lens):<24} "
                f"{baseline_us:>16.2f} {fused_us:>10.2f} {speedup:>8.2f}x"
            )


if __name__ == "__main__":
    main()
