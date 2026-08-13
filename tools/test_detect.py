import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeField:
    def __init__(self, name, value=None, is_str=True):
        self.name = name
        self._value = value
        self.parts = [value.encode() if isinstance(value, str) else None]
        self.data = [0] if value is not None else []
        self._is_str = is_str

    def contents(self):
        return self._value


class FakeTensor:
    def __init__(self, name):
        self.name = name


class FakeReader:
    def __init__(self, arch=None, basename=None, tensors=(), extra_fields=None):
        self.fields = {}
        if arch is not None:
            self.fields["general.architecture"] = FakeField("general.architecture", arch)
        if basename is not None:
            self.fields["general.basename"] = FakeField("general.basename", basename)
        for k, v in (extra_fields or {}).items():
            self.fields[k] = FakeField(k, v, is_str=False)
        self.tensors = [FakeTensor(t) for t in tensors]


from q4nx.arch_detect import detect_model_family
from q4nx.constants import ModelArch

CASES = [
    # (label, reader, expected arch(s))
    ("qwen35moe arch", FakeReader(arch="qwen35moe", tensors=("blk.0.ssm_conv1d.weight",)), {ModelArch.QWEN35MOE}),
    ("qwen35 dense arch + dim", FakeReader(arch="qwen35", tensors=("blk.0.attn_qkv.weight",), extra_fields={"qwen35.embedding_length": 2304}), {ModelArch.QWEN35_2B}),
    ("qwen35 dense 9b dim", FakeReader(arch="qwen3.5", tensors=("blk.0.ssm_conv1d.weight",), extra_fields={"qwen35.embedding_length": 4096}), {ModelArch.QWEN35_9B}),
    ("qwen35 dense unknown dim", FakeReader(arch="qwen35", tensors=("blk.0.ssm_conv1d.weight",), extra_fields={"qwen35.embedding_length": 7777}), {ModelArch.QWEN35_2B}),
    ("gpt-oss arch", FakeReader(arch="gpt-oss"), {ModelArch.GPT_OSS}),
    ("qwen3.6-moe basename", FakeReader(basename="Qwen3.6-MoE-A2B-Instruct", tensors=("blk.0.ffn_gate_exps.weight",)), {ModelArch.QWEN35MOE}),
    ("phi4 basename", FakeReader(basename="Phi-4-mini-instruct", tensors=("blk.0.rope_freqs_long.weight",)), {ModelArch.PHI4}),
    ("phi4 fingerprint only", FakeReader(tensors=("blk.0.rope_freqs_short.weight",)), {ModelArch.PHI4}),
    ("qwen3 fingerprint only", FakeReader(tensors=("blk.0.q_norm.weight", "blk.0.k_norm.weight", "blk.0.q_proj.weight")), {ModelArch.QWEN3}),
    ("qwen3vl fingerprint", FakeReader(tensors=("blk.0.q_norm.weight", "vision_patch_embd.weight")), {ModelArch.QWEN3VL}),
    ("qwen2 fingerprint", FakeReader(tensors=("blk.0.q_bias", "blk.0.k_bias", "blk.0.v_bias")), {ModelArch.QWEN2}),
    ("qwen2vl fingerprint", FakeReader(tensors=("blk.0.q_bias", "vision_patch_embd.weight", "vision_merger_mlp_fc1.weight")), {ModelArch.QWEN2VL}),
    ("gemma3 fingerprint", FakeReader(tensors=("blk.0.post_ffn_norm.weight", "blk.0.q_proj_norm.weight")), {ModelArch.GEMMA3}),
    ("gemma4 fingerprint", FakeReader(tensors=("blk.0.inp_gate.weight", "blk.0.per_layer_token_embedding.weight")), {ModelArch.GEMMA4}),
    ("lfm2 fingerprint", FakeReader(tensors=("blk.0.shortconv.weight", "blk.0.ssm_a.weight")), {ModelArch.LFM2}),
    ("llama keyword", FakeReader(basename="meta-llama-Llama-3.2-3B-Instruct", tensors=("blk.0.rope_freqs.weight",)), {ModelArch.LLAMA}),
    ("mistral -> llama fallback", FakeReader(basename="Mistral-7B-Instruct-v0.3", tensors=("blk.0.rope_freqs.weight",)), {ModelArch.LLAMA}),
    ("nanbeige keyword", FakeReader(basename="Nanbeige2.5-7B", tensors=("blk.0.rope_freqs.weight",)), {ModelArch.NANBEIGE}),
    ("qwen2.5 basename", FakeReader(basename="Qwen2.5-7B-Instruct", tensors=("blk.0.q_bias",)), {ModelArch.QWEN2}),
    ("qwen2.5vl basename", FakeReader(basename="Qwen2.5-VL-7B-Instruct", tensors=("vision_patch_embd.weight",)), {ModelArch.QWEN2VL}),
    ("gemma4 basename", FakeReader(basename="gemma-4-15b", tensors=("blk.0.inp_gate.weight",)), {ModelArch.GEMMA4}),
    ("medgemma basename", FakeReader(basename="google-medgemma-3b", tensors=("blk.0.post_ffn_norm.weight",)), {ModelArch.GEMMA3}),
    ("qwen3 basename", FakeReader(basename="Qwen3-30B-A3B-Instruct", tensors=("blk.0.q_norm.weight",)), {ModelArch.QWEN3}),
    ("unknown arch + unknown basename", FakeReader(arch="foobarbaz", basename="StrangeModel", tensors=("blk.0.some_tensor.weight",)), None),
    ("llama vs qwen3 ambiguity", FakeReader(tensors=("blk.0.q_norm.weight", "blk.0.rope_freqs.weight")), {ModelArch.QWEN3}),
]

fails = 0
for label, reader, expected in CASES:
    guesses = detect_model_family(reader)
    got = [g.arch for g in guesses]
    if expected is None:
        if got:
            print(f"FAIL {label}: expected no guess, got {[a.name for a in got]}")
            fails += 1
        else:
            print(f"ok   {label}: correctly no guess")
        continue
    top = got[0] if got else None
    if top not in expected:
        print(f"FAIL {label}: expected {[a.name for a in expected]}, top={top.name if top else None} all={[a.name for a in got]}")
        fails += 1
    else:
        extra = "".join(f"\n      reasons: {r}" for r in guesses[0].reasons)
        print(f"ok   {label}: -> {top.name} [{guesses[0].confidence}]{extra}")

print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
sys.exit(1 if fails else 0)
