"""Qwen3-Omni-MoE (Qwen3-Omni-30B-A3B / qwen3-omni-moe) converter.

Targets the official FastFlowLM Qwen3.6-MoE NPU runtime layout (the same engine
that loads Qwen3.6-35B-A3B): the text stack is the Qwen3.5-MoE-style hybrid
(GDN + full attention) already handled by the ``Qwen35Moe`` converter, and the
vision tower is the Qwen3-VL-style ViT the ``libqwen3_6_moe_npu.so`` reads via
the exact ``model.visual.*`` key set. Audio (AuT) has no tensors in the
qwen3.6-moe runtime yet, so audio-weight conversion is deferred.

HF source layout (Qwen/Qwen3-Omni-30B-A3B-Thinking):
  model.thinker.text_model.*   -> language weights (this class text path)
  model.thinker.visual.*       -> vision weights (emitted as model.visual.*)
  model.thinker.audio_tower.*  -> audio encoder (deferred / not implemented)
  model.talker.* model.code2wav.* -> speech output (ignored, not a target)

The text MoE experts are stacked expert-major exactly like ``Qwen35Moe`` and
support every expert-layout variant found in Qwen3-MoE-family checkpoints:
per-expert split (gate/up/down_proj), per-expert fused (gate_up_proj), and
per-layer fused (mlp.experts.gate_up_proj + mlp.experts.down_proj).
"""

from pathlib import Path

import torch
from gguf import GGUFReader

from ..constants import ModelArch
from .qwen35moe import Qwen35Moe


class Qwen3OmniMoe(Qwen35Moe, model_arch=ModelArch.QWEN3_OMNI_MOE):
    # Ordered prefix strip for the thinker sub-model. Longest/most specific
    # first so text_model. wins over thinker. and model. wins over nothing.
    _HF_STRIP_PREFIXES = (
        "model.thinker.text_model.",
        "model.thinker.",
        "thinker.text_model.",
        "thinker.",
        "model.language_model.",
        "model.",
    )

    # Vision weights whose 2D matrix the NPU MM kernel reads block-wise.
    _VISION_MM_REARRANGE_SUFFIXES = (
        "attn.qkv.weight",
        "attn.proj.weight",
        "mlp.linear_fc1.weight",
        "mlp.linear_fc2.weight",
        "merger.linear_fc1.weight",
        "merger.linear_fc2.weight",
    )

    def __init__(self, source: str | Path, config_json_path: str | None = None):
        print("[INFO] Using Qwen3OmniMoe converter")
        self.gguf_reader = None
        self.gguf_tensors = {}
        self.hf_source = None
        self.hf_dir = None
        self.weight_map = {}
        self.hf_shards = {}
        self.q4nx_tensors = {}
        if isinstance(source, GGUFReader):
            print("[INFO] Qwen3OmniMoe converter (GGUF source)")
            self.gguf_reader = source
            self.gguf_tensors = {t.name: t for t in source.tensors}
            self.initialize(config_json_path=config_json_path)
        else:
            print("[INFO] Qwen3OmniMoe converter (HF safetensors source)")
            self.hf_source = source
            self.hf_dir = self._resolve_source(source)
            self.initialize(config_json_path=config_json_path)

    # ------------------------------------------------------------- dispatch

    def convert(self, q4nx_path: str, weights_type: str = "language"):
        self.q4nx_tensors = {}
        if self.gguf_reader is not None:
            if weights_type == "language":
                for name in sorted(self.gguf_tensors):
                    self._process_gguf_tensor(name)
                print(f"[INFO] Produced {len(self.q4nx_tensors)} Q4NX tensors")
                self._export_weights(q4nx_path, weights_type)
                self._extract_tokenizer_json(q4nx_path)
            else:
                raise NotImplementedError(
                    f"GGUF-source conversion for weights_type={weights_type} is not "
                    "implemented for the omni-moe converter; use an HF safetensors "
                    "source instead."
                )
        elif weights_type == "language":
            self._convert_hf(q4nx_path)
        elif weights_type == "vision":
            self._convert_hf_vision(q4nx_path)
        elif weights_type == "audio":
            self._convert_hf_audio(q4nx_path)
        else:
            raise ValueError(f"Unsupported weights_type: {weights_type}")

    # ------------------------------------------------------------ helpers

    def _strip_prefix(self, name: str) -> str:
        for prefix in self._HF_STRIP_PREFIXES:
            if name.startswith(prefix):
                return name[len(prefix):]
        return name

    @staticmethod
    def _is_non_text_key(key: str) -> bool:
        return key.startswith(
            ("visual.", "audio_tower.", "audio.", "talker.", "code2wav.")
        )

    def _process_tensor(self, hf_name: str):
        key = self._strip_prefix(hf_name)
        if key.startswith("layers."):
            self._process_layer_tensor(hf_name, key)
            return
        if key == "embed_tokens.weight":
            self.q4nx_tensors["model.embed_tokens.weight"] = self._bf16(self._load_tensor(hf_name))
        elif key == "lm_head.weight":
            self._store_q("lm_head.weight", self._load_tensor(hf_name))
        elif key == "norm.weight":
            w = self._bf16(self._load_tensor(hf_name).float() + 1)
            self.q4nx_tensors["model.norm.weight"] = w
        else:
            print(f"[WARN] Unhandled global tensor: {hf_name}")

    # ----------------------------------------------------- language (text)

    def _collect_expert(
        self, hf_name: str, bid: int, eid: int, kind: str,
        gate: dict, up: dict, down: dict,
    ):
        w = self._load_tensor(hf_name)
        if kind == "gate_up_proj":
            g, u = w.chunk(2, dim=1)
            gate.setdefault(bid, {})[eid] = g
            up.setdefault(bid, {})[eid] = u
        elif kind == "gate_proj":
            gate.setdefault(bid, {})[eid] = w
        elif kind == "up_proj":
            up.setdefault(bid, {})[eid] = w
        elif kind == "down_proj":
            down.setdefault(bid, {})[eid] = w
        else:
            print(f"[WARN] Unhandled expert tensor: {hf_name}")

    def _convert_hf(self, q4nx_path: str):
        """HF text path. Expert-major stacking; skip visual/audio/talker trees."""
        gate: dict[int, dict[int, torch.Tensor]] = {}
        up: dict[int, dict[int, torch.Tensor]] = {}
        down: dict[int, dict[int, torch.Tensor]] = {}

        for name in sorted(self.weight_map):
            key = self._strip_prefix(name)
            if self._is_non_text_key(key):
                continue
            if ".mlp.experts." in key and key.endswith(".weight"):
                parts = key.split(".")
                if len(parts) >= 6 and parts[2] == "mlp" and parts[3] == "experts":
                    bid = int(parts[1])
                    if parts[4].isdigit():
                        # per-expert: layers.{bid}.mlp.experts.{eid}.{kind}.weight
                        self._collect_expert(name, bid, int(parts[4]), parts[5], gate, up, down)
                    else:
                        # per-layer fused: layers.{bid}.mlp.experts.{kind}.weight
                        self._collect_expert(name, bid, -1, parts[4], gate, up, down)
                    continue
            self._process_tensor(name)

        for bid in sorted(set(gate) | set(up) | set(down)):
            prefix = f"model.layer.{bid}."
            if bid in gate:
                eids = sorted(gate[bid])
                g = torch.stack([gate[bid][e] for e in eids], dim=0)
                self._store_q(prefix + "mlp.gate_exps_proj.weight", g.reshape(-1, g.shape[-1]))
            if bid in up:
                eids = sorted(up[bid])
                u = torch.stack([up[bid][e] for e in eids], dim=0)
                self._store_q(prefix + "mlp.up_exps_proj.weight", u.reshape(-1, u.shape[-1]))
            if bid in down:
                eids = sorted(down[bid])
                d = torch.stack([down[bid][e] for e in eids], dim=0)
                self._store_q(prefix + "mlp.down_exps_proj.weight", d.reshape(-1, d.shape[-1]))

        print(f"[INFO] Produced {len(self.q4nx_tensors)} Q4NX tensors")
        if "lm_head.weight" not in self.q4nx_tensors and "model.embed_tokens.weight" in self.q4nx_tensors:
            print("[WARN] No lm_head.weight in source; using tied embed_tokens as lm_head")
            self._store_q("lm_head.weight", self.q4nx_tensors["model.embed_tokens.weight"])
        self._export_weights(q4nx_path, "language")

    # ----------------------------------------------------------- vision

    def _convert_hf_vision(self, q4nx_path: str):
        """HF vision path: thinker visual.* -> runtime model.visual.*, BF16."""
        for name in sorted(self.weight_map):
            key = self._strip_prefix(name)
            if not key.startswith("visual."):
                if any(t in key for t in ("visual", "vision_tower")):
                    print(f"[WARN] Unhandled vision tensor: {name}")
                continue
            new_name = "model." + key
            w = self._bf16(self._load_tensor(name))
            if new_name.endswith(self._VISION_MM_REARRANGE_SUFFIXES):
                w = self.vision_mm_weight_rearrange(w)
            self.q4nx_tensors[new_name] = w
        print(f"[INFO] Produced {len(self.q4nx_tensors)} Q4NX vision tensors")
        self._export_weights(q4nx_path, "vision")

    # ------------------------------------------------------------ audio

    def _convert_hf_audio(self, q4nx_path: str):
        audio_keys = [
            k for k in self.weight_map
            if "audio_tower" in self._strip_prefix(k)
        ]
        raise NotImplementedError(
            "Audio-weight conversion is not implemented for the omni-moe converter: "
            "the qwen3.6-moe runtime exposes no audio tensors yet, so an "
            "audio_weight.q4nx would have no target. Found "
            f"{len(audio_keys)} audio_tower.* tensors in the source "
            f"(e.g. {audio_keys[:3]}). Deferred until a Qwen3.6 AuT-capable "
            "runtime defines the audio layout."
        )
