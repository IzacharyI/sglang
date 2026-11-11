"""Muse Spark MoE block mega path at target verify (TP4 / EP4, gfx950, ``SGLANG_ROCM_MUSE_MOE_MEGA``).

Per call of 16..64 rows (multiples of 8), 6 launches instead of the production 7:

  1. ``_mega_front`` (one launch): router GEMM (replicated) + this rank's 1024-column pooler slab, slab exchange
     over xGMI IPC inboxes, routing / pre-expert RMSNorm / MXFP4 quantisation, compact-A4 scatter.
  2. Production compact-A4 expert GEMM1 and GEMM2 (``_expert_a4``).
  3. Production finalize + TP4 all-reduce (``MuseFinalizeAllreduce``).
  4. Tail: this rank's 2048-column slab of the unpool GEMM (``_slab_partial``) + BF16 slab all-gather
     (``aiter.ops.muse_moe_tail_gather``).

The pooler and unpool work that every rank used to replicate is split four ways. Construction is collective over
the TP CPU group and happens once, after the weights are loaded, outside graph capture; every rank must then issue
the same sequence of calls (the IPC rings are keyed by per-row call counters).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import triton

from sglang.kernels.ops.communication.muse_spark_moe_mega import (
    EP_INC,
    FLAG_STRIDE,
    HID,
    JOINT,
    ROUTER_COLS,
    _mega_front,
    _slab_partial,
)

_NR = 4
_ROW_BYTES = HID * 2
_OUTW = 8192
MIN_ROWS, MAX_ROWS = 16, 64
# Tile configurations of the front (GEMM BN, BK, split-K, warps, stages) and of the unpool slab GEMM; the tail
# all-gather uses NB blocks per row. Measured on MI355X at 16..64 rows.
_FRONT_CFG = (64, 128, 8, 4, 3)
_TAIL_CFG = (32, 256, 4, 4, 3)
_TAIL_NB = 4


class _DevPtr:
    """A raw IPC device address passed to Triton as an int16 pointer.

    ``ptr_range`` lets the AMD backend lower the accesses to buffer instructions; plain global write-through stores
    to peer memory are each followed by ``s_waitcnt vmcnt(0)`` (one xGMI round trip per store)."""

    dtype = torch.int16

    def __init__(self, addr: int, nbytes: int):
        self.addr = addr
        self.nbytes = nbytes

    def data_ptr(self):
        return self.addr

    def ptr_range(self):
        return self.nbytes


class _MegaFront:
    """One-launch front: (scores, ids, expert input, (q, q scales), sorted ids, expert ids, blocks)."""

    def __init__(
        self, cpu_group, device, bias, norm, mult, scale, eps, local_start, e_local
    ):
        assert e_local == 64
        self.rank = dist.get_rank(group=cpu_group)
        self.bias, self.norm = bias, norm
        self.mult, self.scale, self.eps = float(mult), float(scale), float(eps)
        self.local_start, self.e_local = int(local_start), int(e_local)
        nbytes = 3 * _NR * MAX_ROWS * _ROW_BYTES
        from aiter.ops.muse_moe_tail_gather import sentinel_inbox

        inbox = sentinel_inbox(cpu_group, device, nbytes)
        self.inbox = [_DevPtr(addr, nbytes) for addr in inbox]
        bn, _, sk, _, _ = _FRONT_CFG
        self.nt = ROUTER_COLS // bn + (HID // _NR) // bn
        assert ROUTER_COLS // bn * sk <= 64
        self.row_state = torch.zeros(MAX_ROWS, dtype=torch.int32, device=device)
        self.ct = torch.zeros(self.nt * 16, dtype=torch.int32, device=device)
        self.cnt = torch.zeros(512, dtype=torch.int32, device=device)
        # Router tile and routing row epoch flags, one per 128-byte line; monotone, never reset.
        self.flags = torch.zeros(128 * FLAG_STRIDE, dtype=torch.int32, device=device)
        torch.cuda.synchronize(device)
        dist.barrier(group=cpu_group)

    def __call__(self, x, packed):
        m, k = x.shape
        assert (
            MIN_ROWS <= m <= MAX_ROWS
            and k == 8192
            and m % 8 == 0
            and packed.shape == (JOINT, 8192)
        )
        bn, bk, sk, warps, stages = _FRONT_CFG
        dev = x.device
        nw = ROUTER_COLS + HID // _NR
        part = torch.empty((sk, m, nw), device=dev, dtype=torch.float32)
        js = torch.empty((m, HID), device=dev, dtype=torch.float32)
        scores = torch.empty((m, 8), device=dev, dtype=torch.float32)
        ids = torch.empty((m, 8), device=dev, dtype=torch.int32)
        y = torch.empty((m, HID), device=dev, dtype=torch.bfloat16)
        q = torch.empty((m, 2048), device=dev, dtype=torch.uint8)
        qs = torch.empty((m, 128), device=dev, dtype=torch.uint8)
        sbm, e_local = 16, self.e_local
        rows = m * 8
        cap = rows + e_local * (sbm - 1)
        nb = triton.cdiv(cap, sbm)
        sid = torch.empty(cap, device=dev, dtype=torch.int32)
        eid = torch.empty(nb, device=dev, dtype=torch.int32)
        s_pow2 = triton.next_power_of_2(rows)
        per_e = triton.next_power_of_2(triton.cdiv(rows, sbm))
        bm = max(16, 1 << (m - 1).bit_length())
        grid = (self.nt * sk + 2 * m + e_local,)
        _mega_front[grid](
            x, packed, part, js, *self.inbox, self.row_state, self.ct, self.cnt, self.flags,
            self.bias, self.norm, scores, ids, y, q, qs, sid, eid,
            self.rank, m, MAX_ROWS, nw, 8192, bm, bn, bk, sk,
            self.mult, self.scale, self.eps,
            self.local_start, e_local, s_pow2, min(s_pow2, 256), per_e, sbm, nb, triton.next_power_of_2(nb),
            FLAG_STRIDE, EP_INC,
            num_warps=warps, num_stages=stages, matrix_instr_nonkdim=16,
        )  # fmt: skip
        return scores, ids, y, (q, qs), sid, eid, nb


class _MegaTail:
    """Unpool: this rank's 2048-column slab of h @ Wu.T, then the BF16 all-gather of the four slabs."""

    def __init__(self, cpu_group, device):
        from aiter.ops.muse_moe_tail_gather import MuseMoETailGather

        self.rank = dist.get_rank(group=cpu_group)
        bn, _, sk, _, _ = _TAIL_CFG
        bounds = [i * (_OUTW // _NR) for i in range(_NR + 1)]
        assert all(v % bn == 0 for v in bounds)
        self.smax = _OUTW // _NR
        self.bounds = torch.tensor(bounds, dtype=torch.int32, device=device)
        self._gather = MuseMoETailGather(
            cpu_group, device, max_rows=MAX_ROWS, split_k=sk, blocks_per_row=_TAIL_NB
        )

    def __call__(self, h, weight):
        m, k = h.shape
        assert 0 < m <= MAX_ROWS and weight.shape == (_OUTW, k) and h.is_contiguous()
        bn, bk, sk, warps, stages = _TAIL_CFG
        part = torch.empty((sk, m, self.smax), device=h.device, dtype=torch.float32)
        bm = max(16, 1 << (m - 1).bit_length())
        _slab_partial[(self.smax // bn, sk)](
            h, weight, part, self.bounds, self.rank, m, self.smax, k, bm, bn, bk, sk,
            num_warps=warps, num_stages=stages, matrix_instr_nonkdim=16,
        )  # fmt: skip
        return self._gather(part, self.bounds)


def eligible(moe) -> bool:
    """The production compact-A4 verify path this mega path replaces: fused finalize all-reduce, preshuffled
    MXFP4 experts, packed router / pooler weight, unpool projection, EP4 over 256 experts."""
    experts = moe.experts
    return (
        moe._finalize_ar is not None
        and experts.native_dropped
        and moe.router_pooler_packed_weight is not None
        and moe.post_expert_proj is not None
        and experts.num_local_experts * _NR == 256
    )


class MuseMoEMega:
    """The mega path of one MoE layer, built from the live module after ``post_load_weights``."""

    def __init__(self, moe, cpu_group, post_expert_norm_args):
        assert dist.get_world_size(group=cpu_group) == _NR
        dev = torch.device("cuda", torch.cuda.current_device())
        self.moe = moe
        self.post_args = post_expert_norm_args
        self.front = _MegaFront(
            cpu_group, dev, moe.expert_bias, moe.pre_expert_norm._get_effective_weight(), moe.router_multiplier,
            moe.route_scale, moe.pre_expert_norm.eps, moe.local_expert_start, moe.experts.num_local_experts,
        )  # fmt: skip
        self.tail = _MegaTail(cpu_group, dev)

    @staticmethod
    def applies(rows: int) -> bool:
        return MIN_ROWS <= rows <= MAX_ROWS and rows % 8 == 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """bf16 [M, 8192] MoE block output."""
        from sglang.kernels.ops.moe.mxfp4_compact_a4_moe import _expert_a4
        from sglang.srt.rocm_muse_spark.mxfp4_moe import _compact_a4_operands

        moe, experts = self.moe, self.moe.experts
        m = x.shape[0]
        w13, w2, s13, s2 = _compact_a4_operands(experts)
        scores, ids, _, quant, sid, eid, nb = self.front(
            x, moe.router_pooler_packed_weight
        )
        rows = m * 8
        inter = x.new_empty((rows, 2048), dtype=torch.uint8)
        inter_scale = x.new_empty((rows, 128), dtype=torch.uint8)
        y = x.new_empty((rows, 4096), dtype=torch.bfloat16)
        grid = (4096 // 64, nb)
        options = dict(num_warps=4, num_stages=2, matrix_instr_nonkdim=16)
        shuffled = experts.native_dropped
        _expert_a4[grid](
            quant[0], quant[1], w13, s13, sid, eid, inter, inter_scale, rows, 64, 512, True, 16, shuffled, **options
        )  # fmt: skip
        _expert_a4[grid](
            inter,
            inter_scale,
            w2,
            s2,
            sid,
            eid,
            y,
            y,
            rows,
            64,
            512,
            False,
            16,
            shuffled,
            **options,
        )
        norm_weight, gain, _ = self.post_args
        out = moe._finalize_ar(
            y, norm_weight, scores, ids, moe.local_expert_start, experts.num_local_experts,
            moe.post_expert_norm.eps, gain,
        )  # fmt: skip
        return self.tail(out, moe.post_expert_proj.weight)
