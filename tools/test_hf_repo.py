import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/atomic-germ/Projects/Q4NX_Converter")


# --- find_repo_gguf priority selection (mocked network) --------------------
import huggingface_hub
import q4nx.model_assets as ma

class FakeModelInfo:
    def __init__(self):
        self.sha = "abc123"
        self.downloads = 12345
        self.tags = ["fp8", "gguf"]
        self.license = "apache-2.0"
        self.library_name = "transformers"
        self.pipeline_tag = "text-generation"
        self.cardData = {"base_model": "Qwen/Qwen3.5-9B", "language": ["en"]}
        self.config = {
            "model_type": "qwen35",
            "architectures": ["Qwen3.5ForCausalLM"],
            "quantization_config": {"quant_method": "gguf"},
        }

class FakeApi:
    def model_info(self, repo_id, **kw):
        return FakeModelInfo()


def run_case(files, expect_name):
    called = {}
    huggingface_hub.list_repo_files = lambda repo_id: files
    ma._hf_download_file = lambda repo_id, fname: (called.update(name=fname) or f"/tmp/{fname}")
    result = ma.find_repo_gguf("SomeOrg/SomeRepo")
    if expect_name is None:
        assert result is None, f"expected None, got {result}"
        print(f"ok   find_repo_gguf {files} -> None")
        return
    assert result == (f"/tmp/{expect_name}", expect_name), f"{result}"
    print(f"ok   find_repo_gguf {sorted(files)} -> {expect_name}")


run_case(
    ["config.json", "tokenizer.json", "model-q8_0.gguf", "model-q4_0.gguf", "model-q4_1.gguf"],
    "model-q4_1.gguf",
)
run_case(["model-q8_0.gguf", "model-q4_0.gguf"], "model-q4_0.gguf")
run_case(["model-q8_0.gguf"], "model-q8_0.gguf")
run_case(["model-Q4_K_M.gguf", "model-BF16.gguf"], None)
run_case(["a-q4_1.gguf", "b-q4_1.gguf"], "a-q4_1.gguf")
run_case(["config.json", "tokenizer.json"], None)
run_case(["nested/folder/model-q4_0.gguf"], "nested/folder/model-q4_0.gguf")


# --- fetch_hf_repo_info (mocked) ------------------------------------------
huggingface_hub.HfApi = FakeApi
info = ma.fetch_hf_repo_info("SomeOrg/SomeRepo")
assert info["sha"] == "abc123" and info["license"] == "apache-2.0"
assert info["base_model"] == "Qwen/Qwen3.5-9B" and info["quant_method"] == "gguf"
assert info["downloads"] == 12345 and info["model_type"] == "qwen35"
print("ok   fetch_hf_repo_info normalized fields")


# --- README banner with repo enrichment ------------------------------------
from q4nx.model_assets import _readme_banner, build_readme_meta, assemble_readme

meta = {
    "title": "qwen35-9b-q41",
    "tag": "qwen35-9b-q41",
    "modality": "language",
    "flm_version": "0.1.2",
    "date": "2026-08-11",
    "source": "SomeOrg/SomeRepo",
    "source_url": "https://huggingface.co/SomeOrg/SomeRepo",
    "source_file": "model-q4_1.gguf",
    "weight_file": "model.q4nx",
    "weight_size": "4.10 GB",
    "license": "apache-2.0",
    "base_model": "Qwen/Qwen3.5-9B",
    "library": "transformers",
    "model_type": "qwen35",
    "quant_method": "gguf",
    "downloads": 12345,
    "sha": "abc123",
}
banner = _readme_banner(meta)
assert "Source GGUF" in banner and "model-q4_1.gguf" in banner, banner
assert "## Source repository" in banner, banner
assert "apache-2.0" in banner and "abc123" in banner and "12,345" in banner, banner
print("ok   banner has Source GGUF + Source repository section")


# --- assemble_readme enrichment (mocked) -----------------------------------
ma.fetch_hf_repo_info = lambda repo_id: {"license": "mit", "downloads": 7, "sha": "s"}
ma._fetch_source_readme = lambda candidates: ("SomeOrg/SomeRepo", "# Original card\n\nbody")
with tempfile.TemporaryDirectory() as td:
    out = Path(td)
    (out / "model.q4nx").write_bytes(b"x")
    assemble_readme(out, ["SomeOrg/SomeRepo"], build_readme_meta(out, "0.1.2"))
    text = (out / "README.md").read_text()
    assert "| License | mit |" in text, text
    assert "| Source GGUF |" not in text
    assert "## Source model card" in text
print("ok   assemble_readme fetches repo info into banner")


# --- convert.py -i <repo> flow (mocked) ------------------------------------
import convert

calls = {}
convert.is_hf_repo_id = lambda p: p == "SomeOrg/SomeRepo"
convert.is_hf_source = lambda p: False
convert.find_repo_gguf = lambda repo_id: ("/cache/gguf/model-q4_1.gguf", "model-q4_1.gguf")

class FakeModel:
    gguf_reader = object()
    q4nx_config = {}
    model_arch = "QWEN35_2B"
    def convert(self, q4nx_path, weights_type):
        calls["convert"] = (q4nx_path, weights_type)
    hf_source = None

convert.create_converter = lambda path, override: (calls.update(gguf=path, f=override) or FakeModel())
convert.assemble_model_assets = lambda *a, **k: calls.update(assets=(a, k))
convert.get_default_flm_version = lambda: None
convert.deploy_model = lambda *a, **k: None  # unused here

convert.convert_gguf_to_q4nx(
    "SomeOrg/SomeRepo", "out/q4nx", "", weights_type="language", source_model=None, flm_version=None
)
assert calls["gguf"] == "/cache/gguf/model-q4_1.gguf", calls
assert calls["assets"][1]["source_file"] == "model-q4_1.gguf", calls
assert calls["assets"][1]["source_model"] == "SomeOrg/SomeRepo", calls
print("ok   -i <repo> downloads GGUF, uses repo as source_model + source_file")

# repo with no GGUF -> falls back to HF-safetensors path
calls.clear()
convert.find_repo_gguf = lambda repo_id: None
convert.is_hf_repo_id = lambda p: p == "SomeOrg/NoGGUF"
convert.is_hf_source = lambda p: p == "SomeOrg/NoGGUF"

class FakeHFModel:
    hf_source = "SomeOrg/NoGGUF"
    q4nx_config = {}
    def convert(self, q4nx_path, weights_type):
        calls["convert"] = True

convert.create_hf_converter = lambda src, override: (calls.update(hf_src=src) or FakeHFModel())
convert.assemble_model_assets_hf = lambda *a, **k: calls.update(hf_assets=(a, k))
convert.convert_gguf_to_q4nx("SomeOrg/NoGGUF", "out/q4nx", "")
assert calls["hf_src"] == "SomeOrg/NoGGUF", calls
assert calls["hf_assets"][1]["source_file"] is None, calls
print("ok   -i <repo without gguf> falls back to HF-safetensors path")

print("\nALL PASS")
