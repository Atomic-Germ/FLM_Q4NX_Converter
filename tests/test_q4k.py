"""Self-tests for the FLM q4_k path and the Qwen3.8 head reorders (synthetic data only).

    python -m pytest tests/test_q4k.py      (or: python tests/test_q4k.py)
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from q4nx.q4k import pack_q4k, q4k_triple_from_floats, unpack_q4_k  # noqa: E402
from q4nx.models.qwen38 import q_gate_split, tiled_to_grouped  # noqa: E402
from tools.compare_q4nx import decode  # noqa: E402


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def test_pack_layout_and_roundtrip():
    torch.manual_seed(0)
    w = torch.randn(96, 512) * 0.02
    packed = pack_q4k(*q4k_triple_from_floats(w))
    assert packed.dtype == torch.int8 and tuple(packed.shape) == (3, 2, 4736)
    fmt, back = decode(packed.numpy())
    assert fmt == "q4k"
    assert _rel(back, w.numpy()) < 0.12          # 4-bit asymmetric, ~0.08 expected


def test_search_beats_minmax():
    torch.manual_seed(1)
    w = torch.randn(64, 1024) * 0.02
    e_search = _rel(decode(pack_q4k(*q4k_triple_from_floats(w, search=True)).numpy())[1], w.numpy())
    e_naive = _rel(decode(pack_q4k(*q4k_triple_from_floats(w, search=False)).numpy())[1], w.numpy())
    assert e_search < e_naive


def test_positive_groups_survive_refit():
    # one all-positive group inside a super-block: its min must not be clipped away
    w = torch.randn(32, 256) * 0.02
    w[:, :32] = w[:, :32].abs() + 0.05
    back = decode(pack_q4k(*q4k_triple_from_floats(w)).numpy())[1]
    assert _rel(back, w.numpy()) < 0.12


def test_native_q4k_read_matches_gguf_dequant():
    from gguf import GGMLQuantizationType, dequantize
    from gguf.constants import GGML_QUANT_SIZES
    rng = np.random.default_rng(2)
    rows, cols = 4, 512
    bs, ts = GGML_QUANT_SIZES[GGMLQuantizationType.Q4_K]
    raw = rng.integers(0, 256, size=(rows, cols // bs * ts), dtype=np.uint8)
    raw.reshape(-1, ts)[:, 0:4] = np.frombuffer(np.array([0.01, 0.005], np.float16).tobytes(), np.uint8)
    t, u, q = unpack_q4_k(raw, cols)
    ours = (t.repeat_interleave(32, 1) * q - u.repeat_interleave(32, 1)).numpy()
    ref = dequantize(raw, GGMLQuantizationType.Q4_K)
    assert np.allclose(ours, ref, atol=1e-6)


def test_head_permutations():
    # tiled (r g p) -> grouped (g r p): head index r*G+g moves to g*R+r
    R, G, P = 3, 4, 2
    perm = tiled_to_grouped(R, G, P)
    tiled = np.array([[r, g] for r in range(R) for g in range(G) for _ in range(P)])
    grouped = tiled[perm]
    assert [tuple(x) for x in grouped[::P]] == [(r, g) for g in range(G) for r in range(R)]
    # q_proj: per-head [q | gate] -> [all q | all gate]
    sp = q_gate_split(3, 2)
    assert sp.tolist() == [0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
