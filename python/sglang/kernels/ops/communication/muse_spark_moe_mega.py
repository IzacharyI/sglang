"""Triton kernels of the Muse Spark MoE block mega path (TP4 / EP4, gfx950, target verify).

``_mega_front``: ONE launch for the whole front of the MoE block of up to 64 rows. Program roles by id:

  * ``[0, NT * SK)`` GEMM tiles over the joint weight (``router_pooler_packed_weight``): the 512 router columns
    (replicated, every rank) and only this rank's 1024-column pooler slab, split-K FP32 partials written through.
    Router tiles publish one epoch flag each; slab tiles reduce on the last arriver of their SK partials, round
    to BF16 and push the slab to all four ranks' IPC inboxes (the data is its own flag: BF16 -0.0 sentinel).
  * ``[NT * SK, + M)`` routing rows: wait for every router tile, sum the router partials (same order as the
    last-arriver reduction), sigmoid routing + biased top-8, normalised weights; one epoch flag per row.
  * ``[+ M, + 2M)`` norm rows: wait for the four slabs of the row in this rank's inbox, pre-expert RMSNorm,
    MXFP4 quantisation of the expert input (the production ``_joint_epilogue`` math with SK = 1), reset the
    ring slot two calls back.
  * ``[+ 2M, + 2M + E)`` scatter programs: wait for every routing row, build the compact-A4 ``sorted`` /
    ``expert`` block tables of this rank's 64 home experts (the production ``_scatter_a4`` body).

Synchronisation is data-as-flag / monotone epoch counters only (no fences, no resets inside a call), so the
launch is CUDA-graph replayable. ``_slab_partial`` is the split-K GEMM of the unpool column slab of the tail.
"""

import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op

ROUTER_COLS = 512
HID = 4096
JOINT = HID + ROUTER_COLS
FLAG_STRIDE = 32  # one epoch flag per 128-byte line: the M writers of the row flags do not serialise on one line
EP_INC = 64  # the epoch counter C_EP gains 64 per call (one per scatter program)


@triton.jit
def _vmcnt0(x):
    # s_waitcnt vmcnt(0): every write-through store of this wave completed before the following flag / atomic.
    # The memory clobber keeps the compiler from sinking the preceding stores below the wait.
    return tl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)",
        "=v,0,~{memory}",
        [x],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _nap(x):
    # s_sleep between polls: a spinning wave must not steal memory bandwidth from the GEMM tiles still streaming.
    return tl.inline_asm_elementwise(
        "s_sleep 2", "=v,0", [x], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def _spin_rows(F, M: tl.constexpr, target, STRIDE: tl.constexpr):
    # Wait until each of the first M flags (one per STRIDE ints) reached ``target``.
    r = tl.arange(0, 64)
    f = tl.load(
        F + r * STRIDE,
        mask=r < M,
        other=2147483647,
        volatile=True,
        cache_modifier=".cv",
    )
    n = 0
    while tl.min(f, axis=0) < target:
        f = tl.load(
            F + r * STRIDE + _nap(n) * 0,
            mask=r < M,
            other=2147483647,
            volatile=True,
            cache_modifier=".cv",
        )
        n += 1


@triton.jit
def _scatter_home(
    IDS, SORTED, EXPERT, e, M: tl.constexpr, START: tl.constexpr, E: tl.constexpr, S: tl.constexpr,
    CH: tl.constexpr, SBM: tl.constexpr, NB: tl.constexpr, NBB: tl.constexpr, PER_E: tl.constexpr,
):  # fmt: skip
    # The production _scatter_a4 body for local expert e (this rank's home experts, local indices).
    ee = tl.arange(0, E)
    counts = tl.zeros((E,), tl.int32)
    for c0 in tl.static_range(0, S, CH):
        sc = c0 + tl.arange(0, CH)
        idc = tl.load(
            IDS + sc, sc < M * 8, other=-1, volatile=True, cache_modifier=".cv"
        )
        counts += tl.sum(((idc[:, None] - START) == ee[None, :]).to(tl.int32), axis=0)
    blocks_all = (counts + SBM - 1) // SBM
    first = tl.sum(tl.where(ee < e, blocks_all, 0), axis=0)
    if e == E - 1:
        b = tl.arange(0, NBB)
        tl.store(EXPERT + b, -1, (b >= tl.sum(blocks_all, axis=0)) & (b < NB))
    s = tl.arange(0, S)
    ids = tl.load(IDS + s, s < M * 8, other=-1, volatile=True, cache_modifier=".cv")
    hit = (ids == START + e) & (s < M * 8)
    count = tl.sum(hit.to(tl.int32), axis=0)
    blocks = tl.cdiv(count, SBM)
    if blocks > 0:
        eb = tl.arange(0, PER_E)
        tl.store(EXPERT + first + eb, e, eb < blocks)
        rank = tl.cumsum(hit.to(tl.int32), axis=0) - 1
        base = first * SBM
        c = tl.arange(0, PER_E * SBM)
        tl.store(SORTED + base + c, M * 8, c < blocks * SBM)
        tl.debug_barrier()
        tl.store(SORTED + base + rank, s, hit)


@triton.jit
def _mega_front(
    X, W, P, JS, D0, D1, D2, D3, ROW_STATE, CT, CNT, FL,
    BIAS, NORM, SCORES, IDS, Y, Q, QS, SORTED, EXPERT,
    RANK: tl.constexpr, M: tl.constexpr, MAXR: tl.constexpr, NW: tl.constexpr,
    K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SK: tl.constexpr,
    MULT: tl.constexpr, SCALE: tl.constexpr, EPS: tl.constexpr,
    START: tl.constexpr, E: tl.constexpr, S: tl.constexpr, CH: tl.constexpr, PER_E: tl.constexpr,
    SBM: tl.constexpr, NB: tl.constexpr, NBB: tl.constexpr, FSTRIDE: tl.constexpr, EPI: tl.constexpr,
):  # fmt: skip
    pid = tl.program_id(0)
    NRT: tl.constexpr = 512 // BN
    SLAB: tl.constexpr = 4096 // 4
    NT: tl.constexpr = NRT + SLAB // BN
    NG: tl.constexpr = NT * SK
    C0: tl.constexpr = RANK * SLAB
    C_EP = CNT + 128  # call epoch counter (own 256-byte line)
    F_R = FL  # router tile flags (NRT * SK of them)
    F_IDS = FL + 64 * FSTRIDE  # routing row flags (M of them)
    if pid < NG:
        pn = pid % NT
        pk = pid // NT
        KT_PER: tl.constexpr = K // BK // SK
        is_router = pn < NRT
        # This call's epoch, read while the tile streams (no scatter program of this call can have advanced C_EP:
        # that needs every routing row, which needs this program's flag).
        ep_r = tl.atomic_add(C_EP, 0, sem="relaxed", scope="gpu") // EPI
        t = pn - NRT
        wrow0 = tl.where(is_router, 4096 + pn * BN, C0 + t * BN)
        pcol0 = tl.where(is_router, pn * BN, 512 + t * BN)
        m = tl.arange(0, BM)
        kk = tl.arange(0, BK)
        nn = tl.arange(0, BN)
        bp = (
            W
            + (wrow0 + nn)[None, :].to(tl.int64) * K
            + (pk * KT_PER * BK + kk)[:, None]
        )
        ap = X + m[:, None] * K + (pk * KT_PER * BK + kk)[None, :]
        acc = tl.zeros((BM, BN), tl.float32)
        for _ in range(KT_PER):
            a = tl.load(ap, mask=m[:, None] < M, other=0.0)
            b = tl.load(bp, cache_modifier=".cg")
            acc = tl.dot(a, b, acc)
            ap += BK
            bp += BK
        pptr = P + m[:, None] * NW + (pcol0 + nn)[None, :]
        rmask = m[:, None] < M
        tl.store(pptr + pk * M * NW, acc, mask=rmask, cache_modifier=".wt")
        tok = _vmcnt0(pid)
        use_ct = pn >= NRT
        if use_ct:
            old = tl.atomic_add(CT + pn * 16 + tok * 0, 1, sem="relaxed", scope="gpu")
        else:
            # Router partials go straight to the routing rows (they sum the SK partials themselves): one flag per
            # program instead of a last-arriver reduction.
            tl.store(
                F_R + (pn * SK + pk) * FSTRIDE + tok * 0, ep_r + 1, cache_modifier=".wt"
            )
            old = pid * 0
        if (old % SK == SK - 1) & use_ct:
            # Last arriver of a slab tile: sum the SK partials in split order, round to BF16 (-0.0 -> +0.0, the
            # sentinel) and push the tile to the four inboxes.
            tot = tl.load(
                pptr, mask=rmask, other=0.0, volatile=True, cache_modifier=".cv"
            )
            for s in tl.static_range(1, SK):
                tot += tl.load(
                    pptr + s * M * NW,
                    mask=rmask,
                    other=0.0,
                    volatile=True,
                    cache_modifier=".cv",
                )
            bits = tot.to(tl.bfloat16).to(tl.uint16, bitcast=True)
            bits = tl.where(bits == 0x8000, 0, bits).to(tl.uint16)
            slot = tl.load(ROW_STATE + m, mask=m < M, other=0) % 3
            col = C0 + t * BN + nn
            off = ((slot[:, None] * 4 + RANK) * MAXR + m[:, None]).to(
                tl.int64
            ) * 4096 + col[None, :]
            bits = bits.to(tl.int16, bitcast=True)
            tl.store(D0 + off, bits, mask=rmask, cache_modifier=".wt")
            tl.store(D1 + off, bits, mask=rmask, cache_modifier=".wt")
            tl.store(D2 + off, bits, mask=rmask, cache_modifier=".wt")
            tl.store(D3 + off, bits, mask=rmask, cache_modifier=".wt")
    elif pid < NG + M:
        r = pid - NG
        ep = tl.atomic_add(C_EP, 0, sem="relaxed", scope="gpu") // EPI
        n = tl.arange(0, 256)
        one = tl.arange(0, 1)
        _spin_rows(F_R, NRT * SK, ep + 1, FSTRIDE)
        # The router partials summed in split order (split 0 first).
        pr = P + r * NW + one[:, None] * 0 + n[None, :]
        high = tl.load(pr, volatile=True, cache_modifier=".cv")
        low = tl.load(pr + 256, volatile=True, cache_modifier=".cv")
        for s_ in tl.static_range(1, SK):
            high += tl.load(pr + s_ * M * NW, volatile=True, cache_modifier=".cv")
            low += tl.load(pr + s_ * M * NW + 256, volatile=True, cache_modifier=".cv")
        logits = tl.sum(high, axis=0) + tl.sum(low, axis=0)
        scores = tl.sigmoid(logits * MULT)
        biased = scores + tl.load(BIAS + n)
        slots = tl.arange(0, 8)
        ids = tl.full((8,), 0, tl.int32)
        denominator = tl.full((), 0, tl.float32)
        # One int64 max per pick: key = (order-preserving int32 image of the biased score) << 32 | (255 - expert),
        # so the max is the largest score with ties to the lowest id (argmax's tie rule).
        bits = biased.to(tl.int32, bitcast=True)
        mono = bits ^ ((bits >> 31) & 0x7FFFFFFF)
        key = (mono.to(tl.int64) << 32) | (255 - n).to(tl.int64)
        for j in tl.static_range(8):
            kmax = tl.max(key, axis=0)
            idx = 255 - (kmax & 0xFF).to(tl.int32)
            key = tl.where(n == idx, -9223372036854775807 - 1, key)
            ids = tl.where(slots == j, idx, ids)
        selected = tl.sum(
            tl.where(n[None, :] == ids[:, None], scores[None, :], 0.0), axis=1
        )
        for j in tl.static_range(8):
            denominator += tl.sum(tl.where(slots == j, selected, 0.0), axis=0)
        tl.store(IDS + r * 8 + slots, ids, cache_modifier=".wt")
        tl.store(
            SCORES + r * 8 + slots,
            selected * (SCALE / (denominator + 1e-15)),
            cache_modifier=".wt",
        )
        tok = _vmcnt0(r)
        tl.store(F_IDS + r * FSTRIDE + tok * 0, ep + 1, cache_modifier=".wt")
    elif pid < NG + 2 * M:
        row = pid - NG - M
        call = tl.load(ROW_STATE + row)
        slot = call % 3
        prev = (call + 2) % 3
        dn = tl.arange(0, 4096)
        sender = dn // SLAB
        if RANK == 0:
            my = D0
        elif RANK == 1:
            my = D1
        elif RANK == 2:
            my = D2
        else:
            my = D3
        rec = my + ((slot * 4 + sender) * MAXR + row).to(tl.int64) * 4096 + dn
        # Cheap poll first: the last column of every BN tile (each written by that tile's last arriver), with a
        # nap between polls; then the whole row (data is the flag; re-polled until no sentinel is left).
        pc = tl.arange(0, 4096 // BN) * BN + BN - 1
        prb = my + ((slot * 4 + pc // SLAB) * MAXR + row).to(tl.int64) * 4096 + pc
        pv = tl.load(prb, volatile=True, cache_modifier=".cv")
        npoll = 0
        while (tl.max((pv == -32768).to(tl.int32), axis=0) > 0) & (npoll < 50_000_000):
            pv = tl.load(prb + _nap(npoll) * 0, volatile=True, cache_modifier=".cv")
            npoll += 1
        v = tl.load(rec, volatile=True, cache_modifier=".cv")
        while tl.max((v == -32768).to(tl.int32), axis=0) > 0:
            v = tl.load(rec, volatile=True, cache_modifier=".cv")
        # Same [1, 4096] FP32 operand load as the production _joint_epilogue (SK = 1), so its reductions keep
        # their layout.
        tl.store(JS + row * 4096 + dn, v.to(tl.bfloat16, bitcast=True).to(tl.float32))
        tl.debug_barrier()
        one = tl.arange(0, 1)
        partial = tl.load(JS + one[:, None] * 0 + row * 4096 + dn[None, :])
        value = tl.sum(partial, axis=0)
        value = value.to(tl.bfloat16).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(value * value, axis=0) / 4096 + EPS)
        result = value * rstd * tl.load(NORM + dn).to(tl.float32)
        tl.store(Y + row * 4096 + dn, result)
        q, qs = _mxfp4_quant_op(
            result.to(tl.bfloat16).to(tl.float32)[None, :], 4096, 1, 32
        )
        tl.store(Q + row * 2048 + tl.arange(0, 2048), tl.reshape(q, (2048,)))
        tl.store(QS + row * 128 + tl.arange(0, 128), tl.reshape(qs, (128,)))
        sent = tl.full((4096,), -32768, tl.int16)
        tl.store(
            my + ((prev * 4 + sender) * MAXR + row).to(tl.int64) * 4096 + dn,
            sent,
            cache_modifier=".wt",
        )
        tl.debug_barrier()
        tl.store(ROW_STATE + row, call + 1)
    else:
        e = pid - NG - 2 * M
        ep = tl.atomic_add(C_EP, 0, sem="relaxed", scope="gpu") // EPI
        _spin_rows(F_IDS, M, ep + 1, FSTRIDE)
        _scatter_home(IDS, SORTED, EXPERT, e, M, START, E, S, CH, SBM, NB, NBB, PER_E)
        tl.atomic_add(C_EP, 1, sem="relaxed", scope="gpu")


@triton.jit
def _slab_partial(
    X, W, P, BOUNDS, RANK, M, NW, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    SK: tl.constexpr,
):  # fmt: skip
    """Split-K FP32 partials [SK, M, NW] of x @ W[c0:c1).T for this rank's column slab [c0, c1) = BOUNDS[RANK:+2]."""
    pn, pk = tl.program_id(0), tl.program_id(1)
    KT_PER: tl.constexpr = K // BK // SK
    c0 = tl.load(BOUNDS + RANK)
    c1 = tl.load(BOUNDS + RANK + 1)
    wrow0 = c0 + pn * BN
    if wrow0 < c1:
        m = tl.arange(0, BM)
        kk = tl.arange(0, BK)
        nn = tl.arange(0, BN)
        bp = (
            W
            + (wrow0 + nn)[None, :].to(tl.int64) * K
            + (pk * KT_PER * BK + kk)[:, None]
        )
        ap = X + m[:, None] * K + (pk * KT_PER * BK + kk)[None, :]
        acc = tl.zeros((BM, BN), tl.float32)
        for _ in range(KT_PER):
            a = tl.load(ap, mask=m[:, None] < M, other=0.0)
            b = tl.load(bp, cache_modifier=".cg")
            acc = tl.dot(a, b, acc)
            ap += BK
            bp += BK
        tl.store(
            P + pk * M * NW + m[:, None] * NW + (pn * BN + nn)[None, :],
            acc,
            mask=m[:, None] < M,
        )
