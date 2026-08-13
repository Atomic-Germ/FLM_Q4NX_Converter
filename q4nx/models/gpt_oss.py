import os
import json
from pathlib import Path
import numpy as np
from ..model_converter import __Q4NX_Converter
from ..constants import ModelArch
from ..gguf_tensor import GGUFTensor
from gguf import GGUFReader, dequantize, quantize, GGMLQuantizationType
from safetensors import safe_open
from safetensors.torch import save_file
import torch
from gguf import dequantize
from einops import rearrange
import torch.nn.functional as F
from safetensors.torch import save_file, load_file
class GPTOSS(__Q4NX_Converter, model_arch=ModelArch.GPT_OSS):
    # Optional HF repo id / local file / local dir to pull the original
    # (high-precision) embed_tokens.weight from, instead of using the lossy
    # dequantized embed of a re-quantized GGUF. Distinct from the asset
    # source (-s FastFlowLM/...); set externally by convert.py before
    # convert() via the dedicated --embed-source flag.
    embed_source: str | None = None

    def __init__(self, gguf_reader: GGUFReader):
        self.gguf_reader = gguf_reader
        self.gguf_tensors = []
        self.initialize()
    def initialize(self):
        super().initialize()

    # def merge_expert_weights(self):
    #     """
    #     Merge all expert up, gate, down weights to run on NPU.
        
    #     :param self: Description

    #     We have to merge all expert up, gate, down weights to run on NPU,
    #     We also need to handle MXFP4 format
    #     """
    #     for layer_id in range(self.num_layers):
    #         print(f"[INFO] Merging expert weights for layer {layer_id}, new name model.layers.{layer_id}.ffn_gate_up_down_exps.weight")
    #         # Just delete it for now
    #         del self.gguf_tensors[f"blk.{layer_id}.ffn_up_exps.weight"]
    #         del self.gguf_tensors[f"blk.{layer_id}.ffn_gate_exps.weight"]
    #         del self.gguf_tensors[f"blk.{layer_id}.ffn_down_exps.weight"]

    def post_gpt_oss_process(self,result_tensors_map:dict[str, torch.Tensor], n_layers:int):
        #TODO: FIXME: parameterize it 
        NUM_CT_PER_COLUMN = 4
        
        for layer_idx in range(n_layers):
            weight_name_list = [
            f"model.layers.{layer_idx}.ffn_down_exps.weight",
            f"model.layers.{layer_idx}.ffn_up_exps.weight",
            f"model.layers.{layer_idx}.ffn_gate_exps.weight"
            ]
            bias_name_list = [f"model.layers.{layer_idx}.mlp.experts.down_proj_bias", 
                        f"model.layers.{layer_idx}.mlp.experts.up_proj_bias",                       
                        f"model.layers.{layer_idx}.mlp.experts.gate_proj_bias",
                        ]
            for i in range(len(weight_name_list)):
                weight = result_tensors_map[weight_name_list[i]]
                bias = result_tensors_map[bias_name_list[i]]            
                # first pad the shape[1] to be multiple of 4 (4 CT per column, and each column process separate Expert)

                if weight.shape[1] %NUM_CT_PER_COLUMN != 0:
                    pad_amount = (NUM_CT_PER_COLUMN - (weight.shape[1] % NUM_CT_PER_COLUMN)) % NUM_CT_PER_COLUMN
                    # F.pad expects padding for last dims: (last_left, last_right, mid_left, mid_right, ...)
                    # for a 3D tensor (batch, dim1, dim2) to pad dim1 on the right use (0,0,0,pad_amount)
                    weight = F.pad(weight, (0, 0, 0, 0, 0, pad_amount))
                
    
                # padd the bias to be same shape as weight[1]*Q4NX_BLOCK_ROW
                if bias.shape[1] != weight.shape[1]*self.row_block_size:
                    pad_amount = weight.shape[1]*self.row_block_size - bias.shape[1]
                    bias = F.pad(bias, (0, pad_amount))
                
                #NOTE: now do something unique for dequant stuff
                # This trick is to add thew bias into the weight matrix right after the scale values
                # So that at runtime, we can just load the weight matrix and get the bias values
                # without extra memory load
                # This works since MXFP4 size is smaller than q41 block size. 
                # recall mxfp4 only have 1 byte scale for every 32 value, while q41 has 2byte scale and 2byte bias for every 32 value
                bias = rearrange(
                    bias,
                    "batch (block Q4NX_ROW_SIZE) -> batch block Q4NX_ROW_SIZE",
                    Q4NX_ROW_SIZE = self.row_block_size
                ).contiguous()
                NUM_OF_32_set = self.col_block_size//32  # 32 becasuse MXFP4 share scale per 32 columns
                assert bias.dtype == torch.bfloat16
                bias_byte = bias.view(torch.uint8)
                for exp_id in range(weight.shape[0]):
                    for row_block_idx in range(weight.shape[1]):
                        
                        offset = self.row_block_size* NUM_OF_32_set
                        # add it right after to the scale value
                        weight[exp_id][row_block_idx][0][offset: offset+2*self.row_block_size] = bias_byte[exp_id][row_block_idx]
                    
                
                weight = rearrange(
                    weight, 
                    "batch (row_div_four four_row) (col_div one) data_block -> batch row_div_four col_div (four_row one) data_block",
                    one=1,
                    four_row=NUM_CT_PER_COLUMN

                ).contiguous()
                
                
                result_tensors_map[weight_name_list[i]] = weight.contiguous()    
                
        #Q, K, V, O weights projection
        for layer_idx in range(n_layers):
            weight_name_list = [
            f"model.layers.{layer_idx}.self_attn.k_proj.weight",
            f"model.layers.{layer_idx}.self_attn.q_proj.weight",
            f"model.layers.{layer_idx}.self_attn.v_proj.weight",
            f"model.layers.{layer_idx}.self_attn.o_proj.weight"
            ]

            for i in range(len(weight_name_list)):
                weight = result_tensors_map[weight_name_list[i]]       
                # first pad the shape[1] to be multiple of 4 (4 CT per column, and each column process separate Expert)

                if weight.shape[0] %16 != 0:
                    pad_amount = (16 - (weight.shape[0] % 16)) % 16  # NEED to divisible by 16 in this case due to full MVM array that is used for MVM on q, k, v, o project
                    # F.pad expects padding for last dims: (last_left, last_right, mid_left, mid_right, ...)
                    # for a 3D tensor (batch, dim1, dim2) to pad dim1 on the right use (0,0,0,pad_amount)
                    weight = F.pad(weight, (0, 0, 0, 0, 0, pad_amount))
                
    

                weight = rearrange(
                    weight, 
                    "(row_div_four four_row) (col_div one) data_block -> row_div_four col_div (four_row one) data_block",
                    one=1,
                    four_row=4

                ).contiguous()
                
                
                result_tensors_map[weight_name_list[i]] = weight.contiguous()    
        
        # reorder for inference
        for layer_idx in range(n_layers):

            gate_proj_weight = result_tensors_map[f"model.layers.{layer_idx}.ffn_gate_exps.weight"]
            up_proj_weight = result_tensors_map[f"model.layers.{layer_idx}.ffn_up_exps.weight"]

            
            down_weight =result_tensors_map[f"model.layers.{layer_idx}.ffn_down_exps.weight"]
            # bias is shape of [num_expert, intermediate_size]
            
            #note: weights is already being reorder for MVM
            # it reorder into blocks of row-major block wise, but with row_tiles=4
            # weights is shape of [num_expert,   (hidden_size/block_row)/4    ,intermediate_size/block_col ,4_row,byte_per_q4nx_block]
            
            # Goals is to combine the gate and up together for faster decode inference
            

            num_expert = gate_proj_weight.shape[0]
            # similary
            assert num_expert == gate_proj_weight.shape[0]
            weight_row_block_div_4 = gate_proj_weight.shape[1]
            weight_col_block = gate_proj_weight.shape[2]
            weight_block_4_row = gate_proj_weight.shape[3]
            weight_block_size = gate_proj_weight.shape[4]
            
            # #interleave the weights 
            # weights_concat = torch.empty(
            #     size=(num_expert, weight_row_block_div_4*2, weight_col_block, weight_block_4_row, weight_block_size),
            #     dtype=gate_proj_weight.dtype
            # ).contiguous()
            
            # weights_concat[:, 0::2, :, :, :] = gate_proj_weight
            # weights_concat[:, 1::2, :, :, :] = up_proj_weight
            
            
            weights_concat = torch.stack([gate_proj_weight, up_proj_weight], dim=1)  # [E, 2, R,  C, 4, B]
            
            weights_concat = rearrange(weights_concat, "e s r c m b -> e (r s) c m b")

            # do another stach
            weights_concat = torch.cat( [weights_concat, down_weight], dim=1)
            
            result_tensors_map[f"model.layers.{layer_idx}.ffn_gate_up_down_exps.weight"] = weights_concat.contiguous()    

            del result_tensors_map[f"model.layers.{layer_idx}.ffn_gate_exps.weight"]
            del result_tensors_map[f"model.layers.{layer_idx}.ffn_up_exps.weight"]
            del result_tensors_map[f"model.layers.{layer_idx}.ffn_down_exps.weight"]


    def process_gptoss_router_weights(self, weight:torch.Tensor, new_name:str, result_tensors_map:dict[str, torch.Tensor] ) :
        
        #TODO: FIXME: consider do the reorder at runtime shimtile, to avoid the redundant _prefil matrix here
        # This is done original for faster memory access at decode time, but maybe not worth it?
    
        # weight is (num_expert x hidden_size)

        BLOCK_ROWS=32
        BLOCK_COLS=64
        BLOCK_TILE_ROWS  = 16 

        
        # we split the matrix into blocks, each block is 32x64
        # 1. The blocks are in row-major in block order
        # 2. Within each block, the blocks are divide into tile of 16 rows, and each tile is in column-major-order
        
        original_weight = weight.clone()
        assert weight.shape[0] % BLOCK_TILE_ROWS == 0, f"Expected num_expert to be multiple of {BLOCK_TILE_ROWS}, but got {weight.shape[0]}"

        weight = rearrange(
            weight,
            "(num_block_row BLOCK_ROWS) (num_block_col BLOCK_COLS) -> (num_block_row num_block_col) BLOCK_ROWS BLOCK_COLS",
            BLOCK_ROWS=BLOCK_ROWS,
            BLOCK_COLS=BLOCK_COLS,
        ).contiguous()
        
        # # now, for each blocks, needs to be rearranged into tiles of 16 rows, and each tile is in column-major-order
        weight = rearrange(
            weight,
            "num_blocks (num_tile BLOCK_TILE_ROWS) (BLOCK_COLS one) -> num_blocks (num_tile BLOCK_COLS) (BLOCK_TILE_ROWS one)",
            BLOCK_TILE_ROWS=BLOCK_TILE_ROWS,
            BLOCK_COLS=BLOCK_COLS,
        ).contiguous()
        
        result_tensors_map[new_name] = weight.to(torch.bfloat16)
        
        
        # Then, we also have a padded, row_major weights for the MLP router at prefill stage
        #NOTE: The output of o-projection is Lx3072, thus, we need to padded tp 3072
        
        original_weight_col = original_weight.shape[1]
        if original_weight_col % self.col_block_size !=0:
            # padd to 3072
            pad_amount = self.col_block_size - (original_weight_col % self.col_block_size)
            original_weight = F.pad(original_weight, (0, pad_amount))
        original_weight= original_weight.contiguous()   
        result_tensors_map[new_name + "_prefill"] = original_weight.to(torch.bfloat16)
        
 
    def _requant_expert_to_mxfp4(self, gguf_tensor: GGUFTensor):
        # gpt-oss MoE expert weights (ffn_{up,gate,down}_exps.weight) must be
        # MXFP4-packed for the Q4NX runtime: post_gpt_oss_process injects each
        # per-expert bias into the 3-byte gap MXFP4 leaves per 32 values (1B
        # scale) vs Q4_1's full 4B (2B scale + 2B min). Re-quantized GGUFs
        # (mradermacher i1-Q4_1, Q4_K_M, BF16, ...) store experts in the
        # GGUF's house quant, so dequantize + re-quantize to MXFP4 here, then
        # split into the (scales, data) pair _pack_MXFP4_q4nx expects.
        w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
        w = np.ascontiguousarray(w, dtype=np.float32)
        w_mx = quantize(w, GGMLQuantizationType.MXFP4)
        scales_np, data_np = GGUFTensor.split_ggml_mxfpx_to_scale_blocks(w_mx)
        return torch.from_numpy(scales_np), torch.from_numpy(data_np)

    def _load_embed_from_safetensors(self, shard_path: Path) -> torch.Tensor | None:
        """Read model.embed_tokens.weight from one safetensors shard."""
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            if "model.embed_tokens.weight" in f.keys():
                return f.get_tensor("model.embed_tokens.weight")
        return None

    def _resolve_embed_from_source(self, embed_source: str | None) -> torch.Tensor | None:
        """Locate the original (high-precision) embed_tokens.weight.

        ``embed_source`` is the dedicated --embed-source argument -- distinct
        from --source-model (the asset repo). Resolution order:
          1. embed_source is a local safetensors file  -> read its embed tensor.
          2. embed_source is a local directory -> use model.safetensors
             or model.safetensors.index.json to find the right shard.
          3. embed_source looks like an HF repo id ('org/name') -> download
             model.safetensors.index.json (small), find the shard that holds
             the embed, then download ONLY that shard (not the whole model).
          4. Legacy fallback: model-00001-of-00001.safetensors in the CWD.
        Returns the BF16 embed tensor, or None if no source has it.
        """
        keys = ("model.embed_tokens.weight",)

        def try_dir(d: Path) -> torch.Tensor | None:
            idx = d / "model.safetensors.index.json"
            if idx.is_file():
                try:
                    weight_map = json.loads(idx.read_text()).get("weight_map", {})
                except Exception as e:
                    print(f"[WARN] Could not parse {idx}: {e}")
                    weight_map = {}
                shard_name = weight_map.get("model.embed_tokens.weight")
                if shard_name:
                    p = d / shard_name
                    if p.is_file():
                        return self._load_embed_from_safetensors(p)
            # single-shard
            single = d / "model.safetensors"
            if single.is_file():
                return self._load_embed_from_safetensors(single)
            # last resort: scan shards in the dir
            for p in sorted(d.glob("*.safetensors")):
                t = self._load_embed_from_safetensors(p)
                if t is not None:
                    return t
            return None

        def try_repo(repo_id: str) -> torch.Tensor | None:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError:
                print("[WARN] huggingface_hub not installed; cannot fetch BF16 embed from HF")
                return None
            # Cheap first step: the index (KB-scale) tells us which shard
            # has the embed so we only download that one shard, not the full
            # multi-GB BF16 model.
            try:
                idx_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors.index.json")
            except Exception:
                idx_path = None
            if idx_path is not None:
                try:
                    weight_map = json.loads(Path(idx_path).read_text()).get("weight_map", {})
                except Exception as e:
                    print(f"[WARN] Could not parse index.json from {repo_id}: {e}")
                    weight_map = {}
                shard_name = weight_map.get("model.embed_tokens.weight")
                if shard_name:
                    print(f"[INFO] Fetching embed shard '{shard_name}' from {repo_id}")
                    try:
                        sp = hf_hub_download(repo_id=repo_id, filename=shard_name)
                        return self._load_embed_from_safetensors(Path(sp))
                    except Exception as e:
                        print(f"[WARN] Could not download {shard_name} from {repo_id}: {e}")
            # single-shard repo
            print(f"[INFO] Fetching embed shard 'model.safetensors' from {repo_id}")
            try:
                sp = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
                return self._load_embed_from_safetensors(Path(sp))
            except Exception as e:
                print(f"[WARN] {repo_id} has no usable safetensors embed: {e}")
                return None

        # 1-2: local file/dir
        if embed_source:
            p = Path(embed_source)
            if p.is_file() and p.suffix == ".safetensors":
                t = self._load_embed_from_safetensors(p)
                if t is not None:
                    return t
            if p.is_dir():
                t = try_dir(p)
                if t is not None:
                    return t
            # 3: HF repo id (not a local path, single-slash 'org/name')
            if not os.path.exists(embed_source) and "/" in embed_source \
                    and not embed_source.startswith(("http://", "https://", "file:")) \
                    and len(embed_source.split("/")) == 2:
                return try_repo(embed_source)

        # 4: legacy CWD fallback (preserves the dev workflow)
        legacy = Path("model-00001-of-00001.safetensors")
        if legacy.is_file():
            return self._load_embed_from_safetensors(legacy)
        return None

    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        self.q4nx_tensors = {}

        print("Enter into GPTOSS convert function")

        # if not self._has_lm_head():
        #     print("[INFO] Model does not have a lm_head, use embedding weights as lm_head")
        #     unpacked = self.gguf_tensors["token_embd.weight"].unpack()
        #     self.q4nx_tensors["lm_head.weight"] = self._pack_q4nx(*unpacked)

        # self.merge_expert_weights()

        for key, gguf_tensor in self.gguf_tensors.items():
            print(f"[INFO] Converting tensor {gguf_tensor.name} to {self.forward_name_map[gguf_tensor.name]}")
            if "token_embd.weight"  ==  gguf_tensor.name: # this should be bf16
                w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
                w = torch.from_numpy(w).contiguous().to(torch.bfloat16)
                self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = w.contiguous()
                continue

            # MoE expert weights (ffn_{up,gate,down}_exps.weight) MUST be
            # MXFP4-packed: post_gpt_oss_process injects each per-expert bias
            # into the 3-byte MXFP4 scale gap (1B scale / 32 values) that Q4_1
            # (2B scale + 2B min) fills entirely. Source GGUFs that re-quantize
            # experts (mradermacher i1-Q4_1, Q4_K_M, BF16, ...) arrive here in
            # the GGUF's house quant, so dequantize + re-quantize to MXFP4 and
            # pack via _pack_MXFP4_q4nx -- the 4-D (num_experts, ...) layout the
            # runtime + post_gpt_oss_process expect. Native-MXFP4 sources take
            # the same pack path directly from unpacked (scales, data).
            if gguf_tensor.name.endswith("_exps.weight"):
                if gguf_tensor.tensor_type == GGMLQuantizationType.MXFP4:
                    scales, data = gguf_tensor.unpack(self.default_tensor_type)
                else:
                    scales, data = self._requant_expert_to_mxfp4(gguf_tensor)
                self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = self._pack_MXFP4_q4nx(scales, data)
                continue

            unpacked = gguf_tensor.unpack(self.default_tensor_type)
            #     continue
            if self.forward_name_map[gguf_tensor.name] == "lm_head.weight":
                qw = self._pack_q4nx(*unpacked)
                # do a reorder, #TODO: for now, 
                qw = rearrange(
                    tensor=qw,
                    pattern="(row_div_two two_row) (col_div one) data_block ->row_div_two col_div (two_row one) data_block",
                    one=1,
                    two_row=2
                ).contiguous()   
                self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = qw
            elif gguf_tensor.tensor_type == GGMLQuantizationType.MXFP4:
                self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = self._pack_MXFP4_q4nx(*unpacked)
            elif gguf_tensor.tensor_type == GGMLQuantizationType.F32:
                # Request F32 (not the Q4_1 default) so unpack() returns a
                # single float tensor for 2D F32 tensors too: gpt-oss MoE
                # expert biases (ffn_*_exps.bias, shape (num_experts, hidden))
                # and the router weight (ffn_gate_inp.weight) are both 2D F32.
                # With the global Q4_1 default, unpack() would instead
                # dequantize+requantize them and return a (d,m,q) 3-tuple,
                # tripping the len==1 asserts below.
                f32_unpacked = gguf_tensor.unpack(GGMLQuantizationType.F32)
                assert len(f32_unpacked) == 1
                f = f32_unpacked[0]
                if gguf_tensor.name.endswith("ffn_gate_inp.weight"):
                    new_name = self.forward_name_map[gguf_tensor.name]
                    self.process_gptoss_router_weights(weight=f, new_name=new_name, result_tensors_map=self.q4nx_tensors)
                elif gguf_tensor.name.endswith(".bias") or gguf_tensor.name.endswith(".weight"):
                    self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = f.to(torch.bfloat16)  # convert fp32 to bf16
                else:
                    raise ValueError(f"Unsupported F32 tensor {gguf_tensor.name} in GPTOSS model")
            else:
                self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = self._pack_q4nx(*unpacked)



        self.post_gpt_oss_process(self.q4nx_tensors, self.num_layers)

        # Override embed_tokens.weight with the original high-precision embed
        # from the source model when available. Re-quantized GGUFs
        # (mradermacher i1-Q4_1, etc.) store token_embd.weight in the GGUF's
        # house quant (Q4_1), so dequantizing it here yields a lossy bf16 embed.
        # The original BF16 embed lives in the upstream source; pull it from
        # -s <source> (local dir, safetensors file, or HF repo id) via
        # _resolve_embed_from_source.
        embed = self._resolve_embed_from_source(self.embed_source)
        if embed is not None:
            embed = embed.to(torch.bfloat16).contiguous()
            print(f"[INFO] Replaced model.embed_tokens.weight with original embed "
                  f"from source (shape={tuple(embed.shape)}, dtype={embed.dtype})")
            self.q4nx_tensors["model.embed_tokens.weight"] = embed
        else:
            gguf_type = self.gguf_tensors.get("token_embd.weight")
            gguf_type_name = gguf_type.tensor_type.name if gguf_type is not None else "?"
            print(
                f"[WARNING] No BF16 embed source available: model.embed_tokens.weight "
                f"is left as the lossy {gguf_type_name}-dequantized tensor from the GGUF. "
                f"Pass -e <original-bf16-repo> (e.g. openai/gpt-oss-20b) to override."
            )







        # Commet for now
        self._export_weights(q4nx_path, weights_type)
        self._extract_tokenizer_json(q4nx_path)
