from ..model_converter import __Q4NX_Converter
from ..constants import ModelArch
from gguf import GGUFReader, dequantize, quantize
from safetensors.torch import save_file
import torch
from gguf import dequantize
from einops import rearrange

class granite(__Q4NX_Converter, model_arch=ModelArch.granite):
    # Granite 4.1 architecture-specific scale factors (ibm-granite/granite-4.0-tiny-preview)
    EMBEDDING_SCALE  = 12.0   # applied to embedding lookup output before transformer layers
    RESIDUAL_SCALE   = 0.22   # applied to attn/MLP sub-block output before residual add
    LOGIT_SCALE      = 16.0   # applied to final logits before sampling
    HEAD_DIM         = 128    # rope.dimension_count from GGUF
    # Granite uses attention_scale = 1/head_dim; llama_npu uses 1/sqrt(head_dim).
    # Compensate by scaling Q and K weights by c = head_dim^(-1/4) so that
    # (cQ @ cK^T) * (1/sqrt(d)) == (Q @ K^T) * (1/d).
    ATTN_COMPENSATION = HEAD_DIM ** (-0.25)  # ≈ 0.2973

    def __init__(self, gguf_reader: GGUFReader):
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()

    def initialize(self):
        super().initialize()

    def _scale_q4nx_dm(self, unpacked: tuple, factor: float) -> tuple:
        """Multiply the d (scale) and m (min) tensors of an unpacked Q4_1 tensor by factor."""
        d, m, qw = unpacked
        d = (d.float() * factor).to(d.dtype)
        m = (m.float() * factor).to(m.dtype)
        return (d, m, qw)

    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        self.q4nx_tensors = {}

        if not self._has_lm_head():
            print("[INFO] Model does not have a lm_head, use embedding weights as lm_head")
            unpacked = self.gguf_tensors["token_embd.weight"].unpack(self.default_tensor_type)
            # Apply logit_scale to lm_head so logit distribution sharpness matches Granite
            unpacked = self._scale_q4nx_dm(unpacked, self.LOGIT_SCALE)
            self.q4nx_tensors["lm_head.weight"] = self._pack_q4nx(*unpacked)

        for key, gguf_tensor in self.gguf_tensors.items():
            if "token_embd.weight" in gguf_tensor.name: # this should be bf16
                w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
                w = torch.from_numpy(w).contiguous().to(torch.bfloat16)
                # Apply embedding_scale so the embedding magnitude matches Granite
                w = w * self.EMBEDDING_SCALE
                self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = w
                continue

            unpacked = gguf_tensor.unpack(self.default_tensor_type)

            if "q_proj" in self.forward_name_map[gguf_tensor.name] or "k_proj" in self.forward_name_map[gguf_tensor.name]: # for granite q_proj, the order is special
                # 0, 1, 2, .... 127
                # 0, 64, 1, ..., 127
                DH = self.gguf_reader.fields["granite.rope.dimension_count"].contents()
                pp = DH // 2
                d, m, qw = unpacked
                d = rearrange(d, '(g p q) c -> (g q p) c', p = pp, q = 2).contiguous()
                m = rearrange(m, '(g p q) c -> (g q p) c', p = pp, q = 2).contiguous()
                qw = rearrange(qw, '(g p q) c -> (g q p) c', p = pp, q = 2).contiguous()
                unpacked = (d, m, qw)
                # Apply attention scale compensation after reordering
                unpacked = self._scale_q4nx_dm(unpacked, self.ATTN_COMPENSATION)

            # Apply residual_scale to the last linear layer of each sub-block so
            # the residual stream is dampened by RESIDUAL_SCALE (0.22) as in Granite.
            q4nx_name = self.forward_name_map[gguf_tensor.name]
            if q4nx_name.endswith(".self_attn.o_proj.weight") or q4nx_name.endswith(".mlp.down_proj.weight"):
                unpacked = self._scale_q4nx_dm(unpacked, self.RESIDUAL_SCALE)

            self.q4nx_tensors[q4nx_name] = self._pack_q4nx(*unpacked)

        self._export_q4nx_tensors(q4nx_path)
        self._extract_tokenizer_json(q4nx_path)
