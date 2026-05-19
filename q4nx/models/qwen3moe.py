"""
Qwen3 MoE converter — targets pure-transformer MoE models such as
Qwen3-Coder-30B-A3B and Qwen3-30B-A3B.

Architecture specifics (confirmed from GGUF metadata):
  - 48 transformer layers, all with full multi-head attention
  - head_dim = 128  (compatible with existing Qwen3 attn.xclbin)
  - 128 experts per layer, top-8 routing (no shared expert)
  - expert_ffn = 768, hidden = 2048
  - Quantization in the wild: Q4_K (attn, gate/up exps), Q6_K (v_proj, down
    exps, lm_head), F32 (norms, router)

TODO (requires expert.xclbin analysis):
  - Post-pack expert weight merging/rearranging into ffn_gate_up_down_exps.weight
  - Until the expert.xclbin is available the gate/up/down tensors are stored
    separately in standard Q4NX flat format.
"""

import torch
from gguf import GGUFReader, dequantize, GGMLQuantizationType

from ..model_converter import __Q4NX_Converter
from ..constants import ModelArch


class Qwen3Moe(__Q4NX_Converter, model_arch=ModelArch.QWEN3_MOE):
    def __init__(self, gguf_reader: GGUFReader):
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()

    def initialize(self):
        super().initialize()

    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        self.q4nx_tensors = {}

        if not self._has_lm_head():
            print("[INFO] No lm_head found — reusing embedding weights")
            unpacked = self.gguf_tensors["token_embd.weight"].unpack(
                self.default_tensor_type
            )
            self.q4nx_tensors["lm_head.weight"] = self._pack_q4nx(*unpacked)

        for key, gguf_tensor in self.gguf_tensors.items():
            fw_name = self.forward_name_map[gguf_tensor.name]
            print(f"[INFO] {gguf_tensor.name} -> {fw_name}")

            # Embedding: keep in bf16 for full precision
            if gguf_tensor.name == "token_embd.weight":
                w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
                w = torch.from_numpy(w).contiguous().to(torch.bfloat16)
                self.q4nx_tensors[fw_name] = w
                continue

            # F32 tensors (norms, router): dequant to bf16
            if gguf_tensor.tensor_type == GGMLQuantizationType.F32:
                unpacked = gguf_tensor.unpack(self.default_tensor_type)
                assert len(unpacked) == 1
                self.q4nx_tensors[fw_name] = unpacked[0].to(torch.bfloat16)
                continue

            # All other tensors: dequant → requant to default_tensor_type → Q4NX
            unpacked = gguf_tensor.unpack(self.default_tensor_type)
            self.q4nx_tensors[fw_name] = self._pack_q4nx(*unpacked)

        self._export_q4nx_tensors(q4nx_path)
        self._extract_tokenizer_json(q4nx_path)
