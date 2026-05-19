"""
Qwen3.5 MoE / Qwen3.6-35B-A3B converter — hybrid SSM + sparse-MoE Transformer.

Architecture specifics (confirmed from GGUF metadata):
  - 40 layers total
  - full_attention_interval = 4  → layers 3, 7, 11, …, 39 have standard MHA
  - SSM layers (all others) have fused attn_qkv + attn_gate + mamba SSM block
  - head_dim = 256  (NOTE: different from Qwen3 dense — needs its own attn.xclbin)
  - 256 experts, top-8 routing, plus 1 shared expert per layer
  - expert_ffn = 512, hidden = 2048
  - Quantization: Q8_0 (SSM projections), Q4_K/Q6_K (attn/experts), F32 (norms,
    routers, most SSM parameters)

NPU support status:
  - Full-attention layers:  attn.xclbin needs head_dim=256 variant (TODO)
  - SSM layers:             NO NPU2 SSM kernel exists yet
  - Expert layers:          expert.xclbin needs analysis (TODO)
  - This converter produces a complete q4nx file for future hybrid execution.

SSM weights stored as bf16 (no Q4NX packing) — all other quantized tensors use
standard Q4NX packing (dequant → requant to Q4_1).
"""

import torch
from gguf import GGUFReader, dequantize, GGMLQuantizationType

from ..model_converter import __Q4NX_Converter
from ..constants import ModelArch

# Tensor names that belong to the SSM block — kept as bf16 for CPU fallback
_SSM_WEIGHT_SUFFIXES = (
    "ssm_a.weight",
    "ssm_alpha.weight",
    "ssm_beta.weight",
    "ssm_conv1d.weight",
    "ssm_dt.bias",
    "ssm_norm.weight",
    "ssm_out.weight",
    # SSM-layer attention projections (fused, Q8_0 — convert to bf16 for now)
    "attn_qkv.weight",
    "attn_gate.weight",
)


class Qwen35Moe(__Q4NX_Converter, model_arch=ModelArch.QWEN35_MOE):
    def __init__(self, gguf_reader: GGUFReader):
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()

    def initialize(self):
        super().initialize()

    def _is_ssm_tensor(self, gguf_name: str) -> bool:
        """Return True for SSM-block weights that should stay as bf16."""
        return any(gguf_name.endswith(s) for s in _SSM_WEIGHT_SUFFIXES)

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

            # Embedding: keep in bf16
            if gguf_tensor.name == "token_embd.weight":
                w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
                w = torch.from_numpy(w).contiguous().to(torch.bfloat16)
                self.q4nx_tensors[fw_name] = w
                continue

            # F32 tensors (norms, routers, ssm_a, ssm_alpha, …): store as bf16
            if gguf_tensor.tensor_type == GGMLQuantizationType.F32:
                unpacked = gguf_tensor.unpack(self.default_tensor_type)
                assert len(unpacked) == 1
                self.q4nx_tensors[fw_name] = unpacked[0].to(torch.bfloat16)
                continue

            # SSM projection weights (Q8_0): dequant to bf16 (no Q4NX)
            if self._is_ssm_tensor(gguf_tensor.name):
                w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
                w = torch.from_numpy(w).contiguous().to(torch.bfloat16)
                self.q4nx_tensors[fw_name] = w
                continue

            # All remaining tensors: dequant → requant → Q4NX pack
            unpacked = gguf_tensor.unpack(self.default_tensor_type)
            self.q4nx_tensors[fw_name] = self._pack_q4nx(*unpacked)

        self._export_q4nx_tensors(q4nx_path)
        self._extract_tokenizer_json(q4nx_path)
