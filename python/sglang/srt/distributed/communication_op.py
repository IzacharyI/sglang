# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/communication_op.py

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributed

from .parallel_state import get_tp_group


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_fused_allreduce_rmsnorm_quant(
    input_: torch.Tensor,
    residual_inp_: torch.Tensor,
    weight_: torch.Tensor,
    eps: float,
    *,
    emit_bf16: bool = False,
    use_old_ca: bool = False,
) -> Optional[Tuple[torch.Tensor, ...]]:
    """Try AITER fused all-reduce + residual add + RMSNorm + per-token FP8
    quant. Returns ``None`` if no fused path is available so callers can
    run the unfused fallback. See ``GroupCoordinator.fused_allreduce_rmsnorm_quant``
    for the precise return-tuple shape.

    When ``use_old_ca=True`` the AR half of the fused kernel is dispatched
    through the legacy ("old") custom_all_reduce primitive (start_sync +
    cross-rank load + end_sync), keeping the ``+ add + rmsnorm + quant``
    epilogue and the optional bf16 mirror unchanged. This is what the
    LayerCommunicator passes when SGLANG_USE_AITER_NEW_CA=false so the
    fused kernel can interleave safely with the rest of the model's
    old-CA all_reduce calls.
    """
    return get_tp_group().fused_allreduce_rmsnorm_quant(
        input_,
        residual_inp_,
        weight_,
        eps,
        emit_bf16=emit_bf16,
        use_old_ca=use_old_ca,
    )


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> Optional[torch.Tensor]:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
