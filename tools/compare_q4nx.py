"""Compare a converted model.q4nx against a reference one (e.g. the official
FastFlowLM build of the same base model), tensor by tensor.

    python tools/compare_q4nx.py <converted.q4nx> <reference.q4nx> [--min-cos 0.99] [--rows N]

Checks per tensor: present in both, same dtype, same shape (so the same packed
format), and, after decoding the packed blocks (q4k / q8nx / q4_1 / q4_0) or
reading the float tensor, the cosine similarity between the two.  A fine-tune of
the reference model should give cosine ~0.99+ everywhere; a wrong head order or
layout gives ~0.  --rows decodes only the first N rows of big tensors (fast).
Exit status 0 only if every tensor matches.

The block decoder is ported from FastFlowLM/FLM_Q4NX_Converter
(verify_minicpm_v_q4nx.py, Apache-2.0).
"""
import argparse
import sys

import numpy as np
from safetensors import safe_open

ROW, COL, GROUP = 32, 256, 32
GROUPS = COL // GROUP
FORMATS = {4736: "q4k", 8704: "q80", 5120: "q41", 4608: "q40"}


def _bf16(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def _nib(qb, signed):
    lo, hi = (qb & 0x0F).astype(np.int16), (qb >> 4).astype(np.int16)
    q = np.stack([lo, hi], axis=-1).reshape(*qb.shape[:-1], qb.shape[-1] * 2)
    return (q - 8 if signed else q).astype(np.float32)


def decode(blob):
    """int8 (row_blocks, col_blocks, chunk_bytes) -> float32 (rows, cols)."""
    fmt = FORMATS[blob.shape[-1]]
    b = np.ascontiguousarray(blob).view(np.uint8)
    p, u, _ = b.shape
    u16 = lambda lo, hi: np.ascontiguousarray(b[..., lo:hi]).view(np.uint16)
    if fmt == "q4k":
        s8 = b[..., 0:256].reshape(p, u, GROUPS, ROW).astype(np.float32)
        m8 = b[..., 256:512].reshape(p, u, GROUPS, ROW).astype(np.float32)
        q = _nib(b[..., 512:4608].reshape(p, u, COL, ROW // 2), signed=False)
        S = _bf16(u16(4608, 4672)).reshape(p, u, 1, ROW)
        nM = _bf16(u16(4672, 4736)).reshape(p, u, 1, ROW)
        w = np.repeat(S * s8, GROUP, axis=2) * q + np.repeat(nM * m8, GROUP, axis=2)
    elif fmt == "q80":
        s = _bf16(u16(0, 512)).reshape(p, u, GROUPS, ROW)
        q = b[..., 512:].view(np.int8).reshape(p, u, COL, ROW).astype(np.float32)
        w = np.repeat(s, GROUP, axis=2) * q
    elif fmt == "q41":
        s = _bf16(u16(0, 512)).reshape(p, u, GROUPS, ROW)
        m = _bf16(u16(512, 1024)).reshape(p, u, GROUPS, ROW)
        q = _nib(b[..., 1024:].reshape(p, u, COL, ROW // 2), signed=False)
        w = np.repeat(s, GROUP, axis=2) * q + np.repeat(m, GROUP, axis=2)
    else:
        s = _bf16(u16(0, 512)).reshape(p, u, GROUPS, ROW)
        q = _nib(b[..., 512:].reshape(p, u, COL, ROW // 2), signed=True)
        w = np.repeat(s, GROUP, axis=2) * q
    return fmt, w.transpose(0, 3, 1, 2).reshape(p * ROW, u * COL)


def load(f, name, rows):
    sl = f.get_slice(name)
    dtype = sl.get_dtype()
    if dtype == "I8":
        blob = sl[: max(1, rows // ROW)] if rows else sl[:]
        return decode(blob.numpy())
    import torch
    return dtype.lower(), f.get_tensor(name).to(torch.float32).numpy()


def cosine(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return 1.0 if den == 0 and not a.any() and not b.any() else float(a @ b / den) if den else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("converted")
    ap.add_argument("reference")
    ap.add_argument("--min-cos", type=float, default=0.99)
    ap.add_argument("--rows", type=int, default=0, help="decode only the first N rows of packed tensors")
    ap.add_argument("--only", default="", help="substring filter on tensor names")
    args = ap.parse_args()

    fa, fb = safe_open(args.converted, "pt"), safe_open(args.reference, "pt")
    ka, kb = set(fa.keys()), set(fb.keys())
    fails = 0
    for k in sorted(kb - ka):
        if args.only in k:
            print(f"MISSING  {k}"); fails += 1
    for k in sorted(ka - kb):
        if args.only in k:
            print(f"EXTRA    {k}"); fails += 1
    worst = (2.0, "")
    for k in sorted(ka & kb):
        if args.only not in k:
            continue
        sa, sb = fa.get_slice(k), fb.get_slice(k)
        if sa.get_dtype() != sb.get_dtype() or sa.get_shape() != sb.get_shape():
            print(f"LAYOUT   {k}: {sa.get_dtype()}{sa.get_shape()} vs {sb.get_dtype()}{sb.get_shape()}")
            fails += 1
            continue
        fmt, a = load(fa, k, args.rows)
        _, b = load(fb, k, args.rows)
        c = cosine(a, b)
        worst = min(worst, (c, k))
        ok = c >= args.min_cos
        fails += not ok
        print(f"{'OK  ' if ok else 'LOW '}     {k}  {fmt}  cos {c:.5f}", flush=True)
    print(f"\n{len(ka & kb)} common tensors, worst cos {worst[0]:.5f} ({worst[1]}), {fails} problem(s)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
