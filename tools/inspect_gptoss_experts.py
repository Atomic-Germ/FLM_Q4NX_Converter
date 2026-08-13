import glob, os, sys
import numpy as np
import torch
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType

sys.path.insert(0, "/home/atomic-germ/Projects/Q4NX_Converter")
from q4nx.gguf_tensor import GGUFTensor

snap = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--mradermacher--gpt-oss-20b-i1-GGUF/snapshots/*/gpt-oss-20b.i1-Q4_1.gguf"))[0]
print("GGUF:", snap)
reader = GGUFReader(snap)

gguf_tensors = {}
for t in reader.tensors:
    gguf_tensors[t.name] = GGUFTensor(name=t.name, shape=tuple(t.shape.tolist()),
                                     data=t.data, tensor_type=t.tensor_type)

# Inspect every expert weight + its bias, layer 0 and 23.
for layer in (0, 23):
    print(f"\n=== layer {layer} ===")
    for suffix in ("ffn_up_exps", "ffn_gate_exps", "ffn_down_exps"):
        for kind in ("weight", "bias"):
            n = f"blk.{layer}.{suffix}.{kind}"
            if n not in gguf_tensors:
                print(f"  [missing] {n}")
                continue
            gt = gguf_tensors[n]
            print(f"  {n}: type={gt.tensor_type.name:6s} shape={gt.shape} data.shape={gt.data.shape}")
            if kind == "weight":
                # Reproduce what convert() dispatches to.
                unpacked = gt.unpack(GGMLQuantizationType.Q4_1)
                lens = len(unpacked)
                u_shapes = [tuple(x.shape) for x in unpacked]
                print(f"      unpack(Q4_1) -> len={lens}  shapes={u_shapes}  dtypes={[str(x.dtype) for x in unpacked]}")
            else:
                # bias: F32 passthrough after the fix
                r = gt.unpack(GGMLQuantizationType.F32)
                print(f"      unpack(F32)  -> len={len(r)}  shape={tuple(r[0].shape)}")

# Now simulate _pack_q4nx output shape for one expert weight (layer 0 down) to
# see what post_gpt_oss_process would actually index.
print("\n=== simulate _pack_q4nx output for blk.0.ffn_down_exps.weight ===")
gt = gguf_tensors["blk.0.ffn_down_exps.weight"]
import importlib
import q4nx.model_converter as mc
# load the converter to get row_block_size/col_block_size/keep_block_in_2D
from q4nx.constants import ModelArch
from q4nx.model_converter import _MODEL_REGISTRY
conv = _MODEL_REGISTRY[ModelArch.GPT_OSS](reader)
print(f"  row_block_size={conv.row_block_size} col_block_size={conv.col_block_size} keep_block_in_2D={conv.keep_block_in_2D}")
print(f"  default_tensor_type={conv.default_tensor_type}")
unpacked = gt.unpack(conv.default_tensor_type)
print(f"  unpacked len={len(unpacked)} shapes={[tuple(x.shape) for x in unpacked]}")
# Don't full-pack (expensive); just reason about the shapes the rearranges produce.
# _pack_q4nx with keep_block_in_2D merges to np.concatenate([d,m,qw], axis=-1)
# where d/m have shape (p, q, c*r) and qw has shape (p, q, ...).
# So output ndim = 3 for a 2D-style input, or higher if batched.
d, m, qw = unpacked
print(f"  d.shape={tuple(d.shape)} m.shape={tuple(m.shape)} qw.shape={tuple(qw.shape)}")
# emulate rearrange '(p r) (q c) -> p q (c r)' on d (treat qw as 2D first slice):
rows, cols = qw.shape
rb, cb = conv.row_block_size, conv.col_block_size
Q4g = 32
print(f"  qw rows={rows} cols={cols}  row_block_size={rb} col_block_size={cb}")
# how many extra leading dims beyond (rows, cols)?
lead = qw.dim() - 2
print(f"  qw leading (batch) dims = {lead}, leading shape = {tuple(qw.shape[:lead])}")
print(f"  -> _pack_q4nx would produce ndim = {3 + lead} (p q <merged>) + leading batch dims")
print(f"  -> post_gpt_oss_process indexes weight[exp_id][row_block_idx][0][...]")
print(f"     => needs at least ndim>=4 with weight.shape[0]==num_experts")
print(f"     => if lead==0, _pack_q4nx returns 3-D (p,q,bytes); weight[exp][rb][0] indexes a 0-D scalar -> IndexError")