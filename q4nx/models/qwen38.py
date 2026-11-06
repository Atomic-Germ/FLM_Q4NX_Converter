"""Qwen3.8-27B (FLM family ``qwen3.8-mtp``) language-weight converter.

What differs from the Qwen3.5 0.8B-9B converter:

* **Delta-net head ratio 3.**  48 value heads over 16 qk groups.  llama.cpp stores
  the value heads *tiled* (head = r * G + g); FLM wants them *grouped*
  (head = g * R + r, the HF order).  The reorder below is written for any ratio R,
  as row/column index permutations.  That keeps a native Q4_K source exact: the
  scales and quants move together and nothing is re-quantized.
* **The MTP head is kept.**  The qwen3.8-mtp engine runs speculative decoding with
  the next-token-prediction block that llama.cpp appends as the last ``blk``
  (``nextn_predict_layers`` = 1).  It is written under ``model.mtp.*``:
      blk.N.<attn/ffn/norms>        -> model.mtp.layers.0.<same as a full-attention layer>
      blk.N.nextn.eh_proj           -> model.mtp.fc
      blk.N.nextn.enorm / .hnorm    -> model.mtp.pre_fc_norm_embedding / _hidden
      blk.N.nextn.shared_head_norm  -> model.mtp.norm
* **FLM q4_k** (4736-byte chunks) for the 4-bit weights.  Q8NX for embed_tokens,
  lm_head and ssm_alpha/beta, as in the official Qwen3.8-27B-NPU2 build.

Validated tensor-by-tensor against the official FastFlowLM/Qwen3.8-27B-NPU2
model.q4nx with tools/compare_q4nx.py.
"""
import re

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFReader, dequantize, quantize

from ..constants import ModelArch
from ..gguf_tensor import GGUFTensor
from ..q4k import q4k_triple_from_floats, unpack_q4_k
from .qwen35 import Qwen35

Q4_K = GGMLQuantizationType.Q4_K
Q8_0 = GGMLQuantizationType.Q8_0
GROUP = 32

NEXTN_MAP = {
    "nextn.eh_proj.weight": "model.mtp.fc.weight",
    "nextn.enorm.weight": "model.mtp.pre_fc_norm_embedding.weight",
    "nextn.hnorm.weight": "model.mtp.pre_fc_norm_hidden.weight",
    "nextn.shared_head_norm.weight": "model.mtp.norm.weight",
}


def tiled_to_grouped(n_rep: int, n_groups: int, inner: int) -> np.ndarray:
    """Index permutation for '(r g p) -> (g r p)': new[k] = old[perm[k]]."""
    return np.arange(n_rep * n_groups * inner).reshape(n_rep, n_groups, inner).transpose(1, 0, 2).ravel()


def q_gate_split(n_heads: int, head_dim: int) -> np.ndarray:
    """Full-attention q_proj: llama.cpp interleaves [q_h | gate_h] per head;
    FLM wants [all q heads | all gate heads]: '(g p h) -> (p g h)', p = 2."""
    return np.arange(n_heads * 2 * head_dim).reshape(n_heads, 2, head_dim).transpose(1, 0, 2).ravel()


class Qwen38_27B(Qwen35, model_arch=ModelArch.QWEN38_27B):
    def __init__(self, gguf_reader: GGUFReader):
        print("[INFO] Using Qwen38_27B converter")
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()

    # ------------------------------------------------------------------ helpers
    def _field(self, key, default=None):
        f = self.gguf_reader.fields.get(key)
        return f.contents() if f is not None else default

    @staticmethod
    def _float(t: GGUFTensor) -> torch.Tensor:
        return torch.from_numpy(np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32).copy())

    def _triple(self, t: GGUFTensor, target: GGMLQuantizationType):
        """(scale, min/None, quants) for `target`; native when the source already is it."""
        cols = t.shape[0]
        if target == Q4_K:
            if t.tensor_type == Q4_K:
                return unpack_q4_k(t.data, cols)
            return q4k_triple_from_floats(self._float(t))
        if target == Q8_0:
            data = t.data if t.tensor_type == Q8_0 else quantize(self._float(t).numpy(), Q8_0).copy()
            d, _, qw = GGUFTensor.unpack_q8_0(data, cols)
            return d, None, qw
        raise ValueError(f"unsupported target {target.name} for {t.name}")

    @staticmethod
    def _permute(triple, rows=None, cols=None):
        out = []
        for x in triple:
            if x is None:
                out.append(None)
                continue
            if rows is not None:
                x = x[torch.from_numpy(rows)]
            if cols is not None:
                c = cols if x.shape[1] == len(cols) else cols[::GROUP] // GROUP  # per-group tensors
                x = x[:, torch.from_numpy(c)]
            out.append(x.contiguous())
        return tuple(out)

    def _target(self, gguf_name: str) -> GGMLQuantizationType:
        key = re.sub(r"^blk\.\d+\.", "blk.0.", gguf_name)
        return self.tensor_q4nx_type_map.get(gguf_name, self.tensor_q4nx_type_map.get(key, self.default_tensor_type))

    # ------------------------------------------------------------------ convert
    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        if weights_type != "language":
            return super().convert(q4nx_path, weights_type)   # vision: same tower layout as Qwen3.5

        n_mtp = int(self._field("qwen35.nextn_predict_layers", 0) or 0)
        n_main = self.num_layers - n_mtp
        interval = int(self._field("qwen35.full_attention_interval"))
        G = int(self._field("qwen35.ssm.group_count"))
        DS = int(self._field("qwen35.ssm.state_size"))
        n_v = int(self._field("qwen35.ssm.inner_size")) // DS
        if n_v % G:
            raise ValueError(f"{n_v} value heads do not split over {G} qk groups")
        R = n_v // G
        n_heads = int(self._field("qwen35.attention.head_count"))
        DH = int(self._field("qwen35.attention.value_length"))
        print(f"[INFO] {n_main} decoder layers + {n_mtp} MTP, full attention every {interval}, "
              f"delta-net {n_v} v heads / {G} groups (ratio {R})")
        if n_mtp == 0:
            print("[WARN] GGUF has no MTP block: the qwen3.8-mtp engine expects model.mtp.*; "
                  "use a GGUF exported with the nextn layer")

        v_rows = tiled_to_grouped(R, G, DS)                 # z gate rows, ssm_out cols
        head_rows = tiled_to_grouped(R, G, 1)               # alpha/beta/a/dt
        qk = 2 * G * DS
        qkv_rows = np.concatenate([np.arange(qk), qk + v_rows])
        q_rows = q_gate_split(n_heads, DH)

        out = {}
        if not self._has_lm_head():
            print("[INFO] tied embeddings: lm_head from token_embd")
            out["lm_head.weight"] = self._pack(*self._triple(self.gguf_tensors["token_embd.weight"], Q8_0), tensor_type=Q8_0)

        for name, t in self.gguf_tensors.items():
            m = re.match(r"^blk\.(\d+)\.(.+)$", name)
            layer = int(m.group(1)) if m else -1
            is_mtp = layer >= n_main
            if is_mtp:
                if layer != n_main:
                    print(f"[WARN] only one MTP layer is supported, skipping {name}")
                    continue
                suffix = m.group(2)
                if suffix in NEXTN_MAP:
                    new = NEXTN_MAP[suffix]
                elif suffix.startswith("nextn."):
                    print(f"[SKIP] {name}: shared with the main model in FLM")
                    continue
                else:
                    base = self.forward_name_map.get(f"blk.{interval - 1}.{suffix}")
                    if base is None:
                        print(f"[WARN] No name mapping for {name}, skipping")
                        continue
                    new = base.replace(f"model.layers.{interval - 1}.", "model.mtp.layers.0.")
            else:
                new = self.forward_name_map.get(name)
                if new is None:
                    print(f"[WARN] No name mapping for {name}, skipping")
                    continue

            full_attn = is_mtp or (layer >= 0 and layer % interval == interval - 1)
            print(f"Processing {name} ({t.tensor_type.name}) -> {new}", flush=True)

            # ---- non-matmul tensors (norms, conv, A, dt): float, stored as in the official build
            if len(t.shape) == 1 or "ssm_conv1d" in name:
                w = self._float(t)
                if "ssm_conv1d" in name:                    # (channels, kernel) -> (kernel, channels)
                    w = w[torch.from_numpy(qkv_rows)].T.contiguous().to(torch.bfloat16)
                elif name.endswith(".ssm_a") or name.endswith(".ssm_dt.bias"):
                    w = w[torch.from_numpy(head_rows)].contiguous()          # float32
                else:
                    w = w.to(torch.bfloat16)                                  # norms (llama.cpp +1 already applied)
                out[new] = w
                continue

            # ---- matmul weights
            target = self._target(name)
            if name == "token_embd.weight":
                target = Q8_0
            rows = cols = None
            if not full_attn and layer >= 0:
                if name.endswith(".attn_gate.weight"):
                    rows = v_rows
                elif name.endswith(".attn_qkv.weight"):
                    rows = qkv_rows
                elif name.endswith(".ssm_alpha.weight") or name.endswith(".ssm_beta.weight"):
                    rows = head_rows
                elif name.endswith(".ssm_out.weight"):
                    cols = v_rows
            elif full_attn and name.endswith(".attn_q.weight"):
                rows = q_rows

            if name.endswith(".ssm_alpha.weight") or name.endswith(".ssm_beta.weight"):
                # the prefill path reads an unquantized bf16 twin
                out[new.replace("_proj.weight", "_proj.bf16.weight")] = \
                    self._float(t)[torch.from_numpy(rows)].contiguous().to(torch.bfloat16)

            triple = self._permute(self._triple(t, target), rows=rows, cols=cols)
            out[new] = self._pack(*triple, tensor_type=target)

        self.q4nx_tensors = out
        self._extract_tokenizer_json(q4nx_path)
        self._export_weights(q4nx_path, weights_type)
