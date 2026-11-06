"""FLM q4_k support: read GGUF Q4_K natively, encode floats into Q4_K, and pack the
FLM q4_k block layout (4736 bytes per 32x256 chunk).

Newer FLM releases ship their 4-bit weights in this format instead of Q4_0/Q4_1
(e.g. Qwen3.8-27B-NPU2), so a converter that targets those engines has to emit it.

The read path, the (BF16, uint8) re-fit and the packer are ported from
FastFlowLM/FLM_Q4NX_Converter (Apache-2.0): ``GGUFTensor.unpack_q4_k``,
``_refit_one_side`` and ``__Q4NX_Converter._pack_q4k`` on its main branch, plus the
float encoder ``_q4k_triple_from_floats`` from its PR #26.  The search-based
encoder below is new.

Everything works on the "triple" form (t, u, q):
    t: (rows, cols // 32) float32   effective per-group scale
    u: (rows, cols // 32) float32   per-group min, *subtracted*:  w = t * q - u
    q: (rows, cols)       uint4     quants in [0, 15]
"""
from typing import Tuple

import numpy as np
import torch
from einops import rearrange
from gguf import GGMLQuantizationType
from gguf.constants import GGML_QUANT_SIZES

GROUP = 32          # weights per scale/min
SUPER_BLOCK = 256   # groups of 8 share one BF16 super-scale / super-min


# --------------------------------------------------------------------------- read
def unpack_q4_k(data: np.ndarray, columns: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack GGML Q4_K bytes into the (t, u, q) triple without re-quantizing.

    A Q4_K block is 144 bytes for 256 weights: fp16 d, fp16 dmin, 12 bytes of
    bit-packed uint6 scales/mins (get_scale_min_k4) and 128 bytes of nibbles;
    w = (d * s_j) q - (dmin * m_j).  t and u are those products, exact in float32.
    """
    block_size, type_size = GGML_QUANT_SIZES[GGMLQuantizationType.Q4_K]
    assert columns % block_size == 0, "Q4_K needs a multiple of 256 columns"
    raw = data.view(np.uint8)
    n_blocks = raw.size // type_size
    blocks = raw.reshape((n_blocks, type_size))

    d, rest = np.hsplit(blocks, [2])
    dmin, rest = np.hsplit(rest, [2])
    scales, qs = np.hsplit(rest, [12])
    S = d.view(np.float16).astype(np.float32).reshape(n_blocks)
    M = dmin.view(np.float16).astype(np.float32).reshape(n_blocks)

    lo, hi, top = scales[:, 0:4], scales[:, 4:8], scales[:, 8:12]
    s6 = np.concatenate([lo & np.uint8(0x3F),
                         (top & np.uint8(0x0F)) | ((lo >> np.uint8(6)) << np.uint8(4))], axis=1)
    m6 = np.concatenate([hi & np.uint8(0x3F),
                         (top >> np.uint8(4)) | ((hi >> np.uint8(6)) << np.uint8(4))], axis=1)

    qb = qs.reshape((n_blocks, 4, 32))
    q = np.stack([qb & np.uint8(0x0F), qb >> np.uint8(4)], axis=2).reshape((n_blocks, block_size))

    t = (S[:, None] * s6.astype(np.float32)).astype(np.float32)
    u = (M[:, None] * m6.astype(np.float32)).astype(np.float32)
    n_groups = columns // GROUP
    return (torch.from_numpy(t).reshape(-1, n_groups),
            torch.from_numpy(u).reshape(-1, n_groups),
            torch.from_numpy(q.astype(np.float32)).reshape(-1, columns))


# ------------------------------------------------------------------------- encode
def q4k_triple_from_floats(w: torch.Tensor, search: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode float weights (rows, cols) into a (t, u, q) triple, one asymmetric
    4-bit group per 32 columns.

    search=False is plain min/max (FastFlowLM PR #26).  search=True also tries
    a few shrunken ranges per group and keeps whichever minimizes the squared
    error. This is the same idea as llama.cpp's make_qkx search, and it gives a
    lower error than min/max at the same bit rate.
    """
    w = w.to(torch.float32)
    rows, cols = w.shape
    assert cols % GROUP == 0, "Q4_K needs a multiple of 32 columns"
    g = w.reshape(rows, cols // GROUP, GROUP)
    mn = g.amin(dim=-1)
    mx = g.amax(dim=-1)

    def fit(lo, hi):
        # Like llama.cpp's Q4_K, the min is never positive. The packer's re-fit
        # needs all 8 mins of a super-block to share one sign (u >= 0).
        lo = lo.clamp(max=0.0)
        hi = torch.maximum(hi, lo)
        scale = ((hi - lo) / 15.0).clamp(min=1e-12)
        qv = torch.round((g - lo.unsqueeze(-1)) / scale.unsqueeze(-1)).clamp(0, 15)
        err = ((qv * scale.unsqueeze(-1) + lo.unsqueeze(-1) - g) ** 2).sum(-1)
        return scale, lo, qv, err

    best = fit(mn, mx)
    if search:
        span = mx - mn
        for a in (0.02, 0.04, 0.06):         # shrink from the top, the bottom, or both
            for lo, hi in ((mn, mx - a * span), (mn + a * span, mx), (mn + a * span / 2, mx - a * span / 2)):
                cand = fit(lo, hi)
                take = cand[3] < best[3]
                best = tuple(torch.where(take.unsqueeze(-1), c, b) if c.dim() == 3 else torch.where(take, c, b)
                             for c, b in zip(cand, best))
    scale, lo, qv, _ = best
    # zero-point is the (clamped) group min: w = scale * q + lo = t * q - u  ->  u = -lo
    return scale.contiguous(), (-lo).contiguous(), qv.reshape(rows, cols).to(torch.uint8)


# --------------------------------------------------------------------------- pack
def _to_bf16_up(x: np.ndarray) -> np.ndarray:
    """Round the magnitude of a non-negative float32 UP to a BF16 value."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = np.where(u & np.uint32(0xFFFF), np.uint32(0x10000), np.uint32(0))
    return ((u + r) & np.uint32(0xFFFF0000)).view(np.float32)


def _bf16_next_up(x: np.ndarray, n: int) -> np.ndarray:
    """The n-th BF16 value above a BF16-representable, non-negative x (n=0 is x)."""
    if n == 0:
        return np.asarray(x, dtype=np.float32)
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return (u + np.uint32(n) * np.uint32(0x10000)).view(np.float32)


def refit_one_side(t: np.ndarray, search: int = 3, cap: float = 255.0) -> Tuple[np.ndarray, np.ndarray]:
    """Fit per-group values t_j (..., 8) of one super-block onto (BF16 P', uint8 p'_j)
    with P' p'_j ~ t_j.  P' is rounded away from zero so p'_j never overflows, and
    `search` extra BF16 candidates above it are scored by squared error."""
    t = np.ascontiguousarray(t, dtype=np.float32)
    tmax = np.abs(t).max(axis=-1)
    lead = np.take_along_axis(t, np.argmax(np.abs(t), axis=-1)[..., None], axis=-1)[..., 0]
    sigma = np.where(lead < 0, np.float32(-1.0), np.float32(1.0))
    t = sigma[..., None] * t
    live = tmax > 0
    base = _to_bf16_up(tmax / np.float32(cap))

    best_P = np.zeros_like(base)
    best_p = np.zeros_like(t)
    best_e = np.full(base.shape, np.inf, dtype=np.float64)
    for c in range(search + 1):
        Pc = np.where(live, _bf16_next_up(base, c), np.float32(0.0)).astype(np.float32)
        inv = np.where(live, 1.0 / np.where(live, Pc, np.float32(1.0)), np.float32(0.0)).astype(np.float32)
        pc = np.clip(np.rint(t * inv[..., None]), 0.0, cap).astype(np.float32)
        err = np.sum((Pc[..., None] * pc - t).astype(np.float64) ** 2, axis=-1)
        take = err < best_e
        best_e = np.where(take, err, best_e)
        best_P = np.where(take, Pc, best_P)
        best_p = np.where(take[..., None], pc, best_p)
    return (sigma * best_P).astype(np.float32), best_p


def pack_q4k(t: torch.Tensor, u: torch.Tensor, q: torch.Tensor, row_block_size: int = 32,
             col_block_size: int = 256, keep_block_in_2D: bool = True, search: int = 3) -> torch.Tensor:
    """Pack a (t, u, q) triple into FLM q4_k chunks (row_block_size x col_block_size).

    Per chunk, everything column major over the chunk's rows:
        uint8 scales[col/32][rows] | uint8 mins[col/32][rows] | uint4 q[col][rows/2]
        | bf16 S[col/256][rows] | bf16 -M[col/256][rows]
    = 256 + 256 + 4096 + 64 + 64 = 4736 bytes for 32x256.  The kernel adds the
    min path, so M is stored negated.
    """
    t = t.to(torch.float32).contiguous()
    u = u.to(torch.float32).contiguous()
    q = q.contiguous()
    rows, cols = q.shape
    assert cols % SUPER_BLOCK == 0, "Q4_K needs a multiple of 256 columns"
    assert col_block_size % SUPER_BLOCK == 0

    if rows % row_block_size:
        pad = row_block_size - rows % row_block_size
        t = torch.nn.functional.pad(t, (0, 0, 0, pad))
        u = torch.nn.functional.pad(u, (0, 0, 0, pad))
        q = torch.nn.functional.pad(q, (0, 0, 0, pad))
    if cols % col_block_size:
        pad = col_block_size - cols % col_block_size
        t = torch.nn.functional.pad(t, (0, pad // GROUP))
        u = torch.nn.functional.pad(u, (0, pad // GROUP))
        q = torch.nn.functional.pad(q, (0, pad))

    per_super = SUPER_BLOCK // GROUP
    S, s8 = refit_one_side(t.numpy().reshape(-1, per_super), search=search)
    M, m8 = refit_one_side(u.numpy().reshape(-1, per_super), search=search)
    S = torch.from_numpy(S).view(t.shape[0], -1)
    M = torch.from_numpy(M).view(u.shape[0], -1)
    s8 = torch.from_numpy(s8).view_as(t)
    m8 = torch.from_numpy(m8).view_as(u)

    r, cg, cs = row_block_size, col_block_size // GROUP, col_block_size // SUPER_BLOCK
    s8 = rearrange(s8, '(p r) (u c) -> p u (c r)', r=r, c=cg).contiguous()
    m8 = rearrange(m8, '(p r) (u c) -> p u (c r)', r=r, c=cg).contiguous()
    S = rearrange(S, '(p r) (u c) -> p u (c r)', r=r, c=cs).contiguous()
    M = rearrange(M, '(p r) (u c) -> p u (c r)', r=r, c=cs).contiguous()

    q = rearrange(q, '(p r) (u c) -> p u r c', r=r, c=col_block_size)
    q = rearrange(q, 'p u (r b) c -> p u c r b', b=2).contiguous().to(torch.int8)
    q[..., 1] = torch.bitwise_and(torch.bitwise_left_shift(q[..., 1], 4), 0xF0)
    q[..., 0] = torch.bitwise_or(torch.bitwise_and(q[..., 0], 0x0F), q[..., 1])
    q = rearrange(q[..., 0].contiguous(), 'p u c r -> p u (c r)').contiguous()

    merged = torch.from_numpy(np.concatenate([
        s8.to(torch.uint8).view(torch.int8).numpy(),
        m8.to(torch.uint8).view(torch.int8).numpy(),
        q.view(torch.int8).numpy(),
        S.to(torch.bfloat16).view(torch.int8).numpy(),
        (-M).to(torch.bfloat16).view(torch.int8).numpy(),
    ], axis=-1).copy())
    if not keep_block_in_2D:
        merged = merged.reshape(-1, merged.shape[-1])
    return merged
