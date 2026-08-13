import sys, tempfile
from pathlib import Path

sys.path.insert(0, "/home/atomic-germ/Projects/Q4NX_Converter")


class FakeField:
    def __init__(self, name, value=None):
        self.name = name
        self._value = value
        self.parts = [value.encode() if isinstance(value, str) else None]
        self.data = [0] if value is not None else []

    def contents(self):
        return self._value


class FakeReader:
    def __init__(self, fields):
        self.fields = {k: FakeField(k, v) for k, v in fields.items()}


# Darwin-36B-Opus style GGUF: base model via repo_url, newer source repo id
fields = {
    "general.base_model.0.repo_url": "https://huggingface.co/FINAL-Bench/Darwin-36B-Opus",
    "general.base_model.0.organization": "FINAL-Bench",
    "general.base_model.0.name": "Darwin-36B-Opus",
    "general.source.huggingface.repo_id": "FINAL-Bench/Darwin-36B-Opus",
    "general.name": "Darwin-36B-Opus",
    "general.license": "apache-2.0",
    "general.repo_url": "https://huggingface.co/Atomic-Germ/Darwin-36B-Opus-NPU2",
    "tokenizer.chat_template": "{{ chat }}",
}
reader = FakeReader(fields)

import q4nx.model_assets as ma

prov = ma.read_provenance(reader)
assert prov["source_hf_repository"] == "FINAL-Bench/Darwin-36B-Opus", prov
assert prov["name"] == "Darwin-36B-Opus", prov
assert prov["license"] == "apache-2.0", prov
assert prov["base_models"][0]["repository"] == "FINAL-Bench/Darwin-36B-Opus", prov
ids = ma.provenance_repo_ids(reader)
assert ids[0] == "FINAL-Bench/Darwin-36B-Opus", ids
print("ok   read_provenance + provenance_repo_ids:", ids)

# Newer style: no .repository, only .repo_id and org/name
fields2 = {
    "general.base_model.0.organization": "FINAL-Bench",
    "general.base_model.0.name": "Darwin-36B-Opus",
    "general.source.huggingface.organization": "FINAL-Bench",
    "general.source.huggingface.name": "Darwin-36B-Opus",
}
reader2 = FakeReader(fields2)
prov2 = ma.read_provenance(reader2)
assert prov2["source_hf_repository"] == "FINAL-Bench/Darwin-36B-Opus", prov2
assert prov2["base_models"][0]["repository"] == "FINAL-Bench/Darwin-36B-Opus", prov2
print("ok   newer field variants (org+name composition)")

# build_readme_meta consumes provenance (offline, no network)
with tempfile.TemporaryDirectory() as td:
    out = Path(td)
    (out / "model.q4nx").write_bytes(b"x")
    meta = ma.build_readme_meta(out, "0.9.45", prov)
    assert meta["title"] == "Darwin-36B-Opus", meta
    assert meta["source"] == "FINAL-Bench/Darwin-36B-Opus", meta
    assert meta["source_url"] == "https://huggingface.co/FINAL-Bench/Darwin-36B-Opus", meta
    assert meta["license"] == "apache-2.0", meta
    assert meta["base_model"] == "FINAL-Bench/Darwin-36B-Opus", meta
    banner = ma._readme_banner(meta)
    assert "Darwin-36B-Opus" in banner
    assert "FINAL-Bench/Darwin-36B-Opus" in banner
    assert "apache-2.0" in banner
    assert "## Source repository" in banner
    print("ok   build_readme_meta + banner use provenance names/data")

# assemble_readme: provenance wins over -s repo for README source (no network)
ma._fetch_source_readme = lambda candidates: (candidates[-1] if candidates else None, "# Other card")
with tempfile.TemporaryDirectory() as td:
    out = Path(td)
    (out / "model.q4nx").write_bytes(b"x")
    # candidates: -s repo first, then provenance ids
    ma.assemble_readme(out, ["SomeQuantizer/Repo", "FINAL-Bench/Darwin-36B-Opus"], ma.build_readme_meta(out, "0.9.45", prov))
    text = (out / "README.md").read_text()
    # README source/banner should reflect provenance, NOT the -s quantizer repo
    assert "FINAL-Bench/Darwin-36B-Opus" in text
    assert "SomeQuantizer/Repo" not in text or "Original model card" not in text
    print("ok   assemble_readme: provenance source preferred over -s repo")

print("\nALL PASS")
