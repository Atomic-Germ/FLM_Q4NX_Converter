import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

ASSET_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
REQUIRED_ASSETS = ["config.json", "tokenizer.json", "tokenizer_config.json"]


def _gguf_field(reader, name: str):
    """Return the python value of a GGUF metadata field (scalar string/array included)."""
    field = reader.fields.get(name)
    if field is None:
        return None
    try:
        return field.contents()
    except Exception:
        pass
    try:
        if len(field.data) == 0:
            return None
        return field.parts[field.data[0]].decode("utf-8", errors="replace")
    except Exception:
        return None


def _gguf_token_list(reader) -> List[str]:
    field = reader.fields.get("tokenizer.ggml.tokens")
    if field is None:
        return []
    try:
        return [field.parts[i].decode("utf-8", errors="replace") for i in field.data]
    except Exception:
        return []


def _repo_id_from_url(url) -> Optional[str]:
    if not url:
        return None
    m = re.search(
        r"(?:huggingface\.co|hf\.co|modelscope\.cn/models)/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)",
        url,
    )
    return m.group(1) if m else None


def read_provenance(reader) -> dict:
    """Extract the provenance chain from GGUF metadata.

    llama.cpp converters record where a GGUF came from:
      - general.base_model.{i}.repo_url / .repository / .organization / .name
      - general.source.huggingface.repository
      - general.repo_url (the quantizer's repo)
      - tokenizer.chat_template
    """
    info: dict = {}
    base_models = []
    i = 0
    while i < 100:
        url = _gguf_field(reader, f"general.base_model.{i}.repo_url")
        repository = _gguf_field(reader, f"general.base_model.{i}.repository")
        if url is None and repository is None:
            break
        base_models.append(
            {
                "repo_url": url,
                "repository": repository,
                "organization": _gguf_field(reader, f"general.base_model.{i}.organization"),
                "name": _gguf_field(reader, f"general.base_model.{i}.name"),
            }
        )
        i += 1
    info["base_models"] = base_models
    info["source_hf_repository"] = _gguf_field(reader, "general.source.huggingface.repository")
    info["repo_url"] = _gguf_field(reader, "general.repo_url")
    info["chat_template"] = _gguf_field(reader, "tokenizer.chat_template")
    return info


def resolve_repo_candidates(reader, source_arg: Optional[str]) -> List[str]:
    """Ordered list of repo ids to try, from an explicit source then GGUF provenance."""
    candidates: List[str] = []
    if source_arg and "/" in source_arg and not os.path.exists(source_arg):
        candidates.append(source_arg)
    provenance = read_provenance(reader)
    for base in provenance["base_models"]:
        rid = base.get("repository") or _repo_id_from_url(base.get("repo_url"))
        if rid and rid not in candidates:
            candidates.append(rid)
    rid = provenance.get("source_hf_repository")
    if rid and rid not in candidates:
        candidates.append(rid)
    rid = _repo_id_from_url(provenance.get("repo_url"))
    if rid and rid not in candidates:
        candidates.append(rid)
    return candidates


def _cache_roots() -> List[Path]:
    roots = []
    for env in ("HF_HUB_CACHE", "HF_HOME"):
        value = os.environ.get(env)
        if not value:
            continue
        path = Path(value)
        if path.name == "hub":
            roots.append(path)
        else:
            roots.append(path / "hub")
    home = Path.home()
    roots.append(home / ".cache" / "huggingface" / "hub")
    roots.append(home / "cache" / "huggingface" / "hub")
    seen, result = set(), []
    for root in roots:
        if str(root) not in seen:
            seen.add(str(root))
            result.append(root)
    return result


def repo_folder_name(repo_id: str) -> str:
    return "models--" + "--".join(repo_id.split("/"))


def cached_snapshot_dir(repo_id: str) -> Optional[Path]:
    """Return the snapshot directory of a repo in the local HF cache, if present."""
    for root in _cache_roots():
        repo_dir = root / repo_folder_name(repo_id)
        if not repo_dir.is_dir():
            continue
        snapshots = repo_dir / "snapshots"
        if snapshots.is_dir():
            candidates = [s for s in snapshots.iterdir() if s.is_dir()]
            refs = repo_dir / "refs"
            if refs.is_dir():
                for ref in refs.iterdir():
                    try:
                        rev = ref.read_text().strip()
                    except Exception:
                        continue
                    rev_dir = snapshots / rev
                    if rev_dir.is_dir():
                        return rev_dir
            if candidates:
                return candidates[0]
    return None


def _copy_from_dir(source_dir: Path, output_dir: Path, files: List[str]) -> List[str]:
    copied = []
    for filename in files:
        src = source_dir / filename
        if src.is_file():
            shutil.copy2(src, output_dir / filename)
            copied.append(filename)
    return copied


def _hf_download_file(repo_id: str, filename: str) -> Optional[str]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[WARN] huggingface_hub not installed; cannot download missing model assets")
        return None
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception as e:
        print(f"[WARN] Could not download {filename} from {repo_id}: {e}")
        return None


def _source_dir_for_candidate(candidate: str) -> Optional[Path]:
    """Resolve a candidate (local dir or repo id) to a directory holding model files."""
    if os.path.isdir(candidate):
        return Path(candidate)
    snapshot = cached_snapshot_dir(candidate)
    if snapshot is not None:
        return snapshot
    return None


def _fetch_assets(candidate: str, output_dir: Path, files: List[str]) -> List[str]:
    """Fetch files for a candidate repo into output_dir. Uses cache, then SDK download."""
    source_dir = _source_dir_for_candidate(candidate)
    if source_dir is not None:
        copied = _copy_from_dir(source_dir, output_dir, files)
        if copied:
            print(f"[INFO] Copied model assets from local source: {source_dir}")
            return copied
        return []
    if "/" in candidate:
        downloaded = []
        for filename in files:
            path = _hf_download_file(candidate, filename)
            if path:
                shutil.copy2(path, output_dir / filename)
                downloaded.append(filename)
        if downloaded:
            print(f"[INFO] Downloaded model assets from {candidate}")
        return downloaded
    return []


def generate_config_from_gguf(reader) -> dict:
    """Best-effort HF-style config.json built from GGUF metadata (no-source fallback)."""
    cfg: dict = {}
    arch = _gguf_field(reader, "general.architecture")
    if arch:
        cfg["model_type"] = str(arch)
    mappings = [
        (".embedding_length", "hidden_size"),
        (".feed_forward_length", "intermediate_size"),
        (".block_count", "num_hidden_layers"),
        (".attention.head_count", "num_attention_heads"),
        (".attention.head_count_kv", "num_key_value_heads"),
        (".attention.layer_norm_rms_epsilon", "rms_norm_eps"),
        (".rope.dimension_count", "head_dim"),
        (".vocab_size", "vocab_size"),
    ]
    prefixes = [f"{arch}"] if arch else []
    for field_suffix, key in mappings:
        for prefix in prefixes:
            value = _gguf_field(reader, prefix + field_suffix)
            if value is not None:
                cfg[key] = value
                break
    general_vocab = _gguf_field(reader, "general.vocab_size")
    if "vocab_size" not in cfg and general_vocab is not None:
        cfg["vocab_size"] = general_vocab
    if "head_dim" in cfg and "num_attention_heads" in cfg and "hidden_size" in cfg:
        if cfg["head_dim"] * cfg["num_attention_heads"] == cfg["hidden_size"]:
            cfg.pop("head_dim", None)
    for key in list(cfg.keys()):
        if isinstance(cfg[key], bool):
            continue
        if isinstance(cfg[key], float) or isinstance(cfg[key], int):
            continue
        cfg[key] = str(cfg[key])
    return cfg


def generate_tokenizer_config(reader) -> dict:
    """Minimal tokenizer_config.json the FLM runtime can parse (no-source fallback)."""
    cfg: dict = {}
    tokens = _gguf_token_list(reader)
    id_map = {
        "bos_token_id": "tokenizer.ggml.bos_token_id",
        "eos_token_id": "tokenizer.ggml.eos_token_id",
        "unk_token_id": "tokenizer.ggml.unknown_token_id",
        "pad_token_id": "tokenizer.ggml.padding_token_id",
    }
    for hf_key, gguf_key in id_map.items():
        value = _gguf_field(reader, gguf_key)
        if value is not None:
            cfg[hf_key] = int(value)
    for token_key, id_key in [("bos_token", "bos_token_id"), ("eos_token", "eos_token_id")]:
        token_id = cfg.get(id_key)
        if token_id is not None and 0 <= token_id < len(tokens):
            cfg[token_key] = tokens[token_id]
    if "eos_token_id" in cfg:
        eos_id = cfg["eos_token_id"]
        cfg["eos_token_id"] = [eos_id] if not isinstance(eos_id, list) else eos_id
    chat_template = _gguf_field(reader, "tokenizer.chat_template")
    if chat_template:
        cfg["chat_template"] = chat_template
    return cfg


def ensure_runtime_tokenizer_ids(reader, output_dir: Path):
    """Backfill EOS/BOS/PAD token ids into the deployed tokenizer_config.json.

    The FLM runtime stops generation only when the sampled token id is in
    ``tokenizer_config["eos_token_id"]``. Several official sources (e.g.
    Qwen/Qwen3.5-9B) ship that field as null, which silently disables
    end-of-generation. Fill any null id from GGUF metadata, and normalize
    eos_token_id to a list, so converted models actually stop.
    """
    path = output_dir / "tokenizer_config.json"
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    tokens = _gguf_token_list(reader)
    changed = False
    id_map = {
        "eos_token_id": "tokenizer.ggml.eos_token_id",
        "bos_token_id": "tokenizer.ggml.bos_token_id",
        "pad_token_id": "tokenizer.ggml.padding_token_id",
    }
    for hf_key, gguf_key in id_map.items():
        if cfg.get(hf_key) is None:
            value = _gguf_field(reader, gguf_key)
            if value is not None:
                cfg[hf_key] = int(value)
                changed = True
    for token_key, id_key in [("eos_token", "eos_token_id"), ("bos_token", "bos_token_id"), ("pad_token", "pad_token_id")]:
        token_id = cfg.get(id_key)
        if cfg.get(token_key) is None and isinstance(token_id, int) and 0 <= token_id < len(tokens):
            cfg[token_key] = tokens[token_id]
            changed = True
    if cfg.get("eos_token_id") is not None:
        eos = cfg["eos_token_id"]
        if not isinstance(eos, list):
            cfg["eos_token_id"] = [eos]
            changed = True
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Patched {path.name} with token ids from GGUF metadata")


def _tokenizer_id_lookup(tokenizer_path: Path) -> Dict[str, int]:
    """Map token text -> token id from tokenizer.json (added tokens first, then vocab)."""
    if not tokenizer_path.exists():
        return {}
    try:
        with open(tokenizer_path, encoding="utf-8") as f:
            tokenizer = json.load(f)
    except Exception:
        return {}
    lookup: Dict[str, int] = {}
    for added in tokenizer.get("added_tokens", []):
        content = added.get("content")
        if content is not None:
            lookup[content] = int(added["id"])
    vocab = tokenizer.get("model", {}).get("vocab", {})
    for text, token_id in vocab.items():
        lookup.setdefault(text, int(token_id))
    return lookup


def ensure_hf_tokenizer_ids(output_dir: Path) -> None:
    """Backfill EOS/BOS/PAD token ids into a copied HF tokenizer_config.json.

    The FLM runtime stops generation only when the sampled token id appears in
    ``tokenizer_config["eos_token_id"]``. HF Qwen-family repos ship that field as
    null (the end-of-turn token lives in ``eos_token``), which silently disables
    end-of-generation. Resolve each token string against the tokenizer and
    normalize ``eos_token_id`` to a list so converted models actually stop.
    """
    path = output_dir / "tokenizer_config.json"
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    lookup = _tokenizer_id_lookup(output_dir / "tokenizer.json")
    config: dict = {}
    config_path = output_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            config = {}
    changed = False
    for token_key, id_key in [
        ("eos_token", "eos_token_id"),
        ("bos_token", "bos_token_id"),
        ("pad_token", "pad_token_id"),
    ]:
        token_text = cfg.get(token_key)
        current = cfg.get(id_key)
        if (current is None or current == [] or current == "") and isinstance(token_text, str):
            resolved = lookup.get(token_text)
            if resolved is None and config.get(id_key) is not None:
                resolved = int(config[id_key])
            if resolved is not None:
                cfg[id_key] = resolved
                changed = True
    if cfg.get("eos_token_id") is not None:
        eos = cfg["eos_token_id"]
        if not isinstance(eos, list):
            cfg["eos_token_id"] = [eos]
            changed = True
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Patched {path.name} with EOS/BOS/PAD token ids from HF tokenizer")


def inject_flm_keys(config: dict, q4nx_config: dict, output_dir: Path, flm_version: Optional[str]):
    """Restructure a source HF config.json into the shape the FLM runtime expects.

    - Flatten ``text_config`` into the top level (lm_config.hpp reads top-level
      keys such as hidden_size / num_attention_heads).
    - Keep the nested ``vision_config`` / ``audio_config`` objects (the runtime
      reads those via ``_vision_config`` / ``_audio_config``).
    - Inject flm_version and the weight-file names once the converted weights exist.
    """
    text_config = config.pop("text_config", None)
    if isinstance(text_config, dict):
        # Flatten text_config over top-level so LM_Config sees hidden_size etc.
        # Prefer text_config values (Ornith/VL wrappers put real LM hyperparams there).
        for key, value in text_config.items():
            if key in ("model_type",) or config.get(key) in (None, ""):
                config[key] = value
            else:
                config.setdefault(key, value)
        # Darwin-style text-only MoE: model_type becomes qwen3_5_moe_text.
        if text_config.get("model_type"):
            config["model_type"] = text_config["model_type"]
    # Drop HF vision blob when no vision weights were converted (text-only finetunes).
    if not (output_dir / "vision_weight.q4nx").exists():
        config.pop("vision_model_weight", None)
        # Keep Darwin parity: pure language configs omit nested vision_config.
        if config.get("model_type") in (
            "qwen3_5_moe_text", "qwen3_6_moe_text"
        ) or "text_config" not in config:
            # If original was a VL wrapper with nested text_config already popped,
            # strip unused vision_config so FLM stays text-only.
            vc = config.get("vision_config")
            if isinstance(vc, dict) and "vision_mm_engine_xclbin_name" not in vc:
                config.pop("vision_config", None)

    # qwen3.5/3.6 MoE: only moe_intermediate_size is declared upstream, but the
    # runtime LM_Config requires intermediate_size (== moe_intermediate_size).
    if (
        config.get("model_type") in (
            "qwen3_5_moe_text", "qwen3_5_moe", "qwen3_6_moe", "qwen3_6_moe_text"
        )
        and "moe_intermediate_size" in config
        and "intermediate_size" not in config
    ):
        config["intermediate_size"] = config["moe_intermediate_size"]
    # Engine memory-layout offsets (lm_config.hpp JSON_GETs addr_* with default 0).
    # Architecture-level, declared in the arch config. Darwin worked without them
    # on some FLM builds; still inject when present so MHA layouts are correct.
    for key in ("addr_qk", "addr_kv", "addr_kk", "addr_l_begin_mha", "addr_l_end_mha"):
        if key in q4nx_config:
            config.setdefault(key, q4nx_config[key])
    # Token ids: prefer text_config / generation defaults used by Darwin.
    if config.get("bos_token_id") is None and config.get("pad_token_id") is not None:
        config["bos_token_id"] = config["pad_token_id"]
    if config.get("eos_token_id") is None:
        config["eos_token_id"] = 248044
    # Darwin/Ornith engines need caching enabled at runtime.
    if config.get("model_type") in ("qwen3_5_moe", "qwen3_5_moe_text", "qwen3_6_moe", "qwen3_6_moe_text"):
        config["use_cache"] = True
    if flm_version:
        config["flm_version"] = flm_version
    vision_config = q4nx_config.get("vision_config", {})
    if vision_config:
        vision_file = vision_config.get("vision_file", "vision_weight.q4nx")
        if (output_dir / vision_file).exists():
            config["vision_model_weight"] = vision_file
            vc = config.setdefault("vision_config", {})
            if "vision_MM_K" in vision_config:
                vc["vision_MM_K"] = vision_config["vision_MM_K"]
            if "vision_MM_N" in vision_config:
                vc["vision_MM_N"] = vision_config["vision_MM_N"]
        else:
            config.pop("vision_model_weight", None)
    audio_config = q4nx_config.get("audio_config", {})
    if audio_config:
        audio_file = audio_config.get("audio_file", "audio_weight.q4nx")
        if (output_dir / audio_file).exists():
            config["audio_model_weight"] = audio_file
        else:
            config.pop("audio_model_weight", None)
    return config


def get_default_flm_version() -> Optional[str]:
    """Best-effort: read the installed FLM version for the config's flm_version."""
    try:
        output = subprocess.run(
            ["flm", "--version"], capture_output=True, text=True, timeout=3
        ).stdout
        m = re.search(r"v?(\d+\.\d+\.\d+)", output)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None


def assemble_model_assets_hf(
    hf_source: str,
    q4nx_config: dict,
    output_dir: str,
    source_model: Optional[str] = None,
    flm_version: Optional[str] = None,
) -> None:
    """Build a complete model directory from an HF safetensors source.

    Same result as assemble_model_assets but sourced straight from an HF repo
    (no GGUF provenance involved): config.json / tokenizer.json /
    tokenizer_config.json / chat_template.jinja are copied from the HF model,
    then the config is restructured for the FLM runtime.
    """
    output_dir = Path(output_dir)
    if output_dir.suffix == ".q4nx":
        output_dir = output_dir.parent
    os.makedirs(output_dir, exist_ok=True)

    candidate = source_model or hf_source
    fetched = _fetch_assets(candidate, output_dir, ASSET_FILES)
    missing = [f for f in REQUIRED_ASSETS if f not in fetched]
    if missing:
        print(f"[WARN] HF source {candidate} missing required assets: {missing}")

    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {}
    inject_flm_keys(config, q4nx_config, output_dir, flm_version)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    ensure_hf_tokenizer_ids(output_dir)

    print(f"[INFO] Model directory ready: {output_dir}")


def assemble_model_assets(
    reader,
    q4nx_config: dict,
    output_dir: str,
    source_model: Optional[str] = None,
    flm_version: Optional[str] = None,
) -> None:
    """Build a complete, uploadable model directory.

    Weights (model.q4nx / vision_weight.q4nx / audio_weight.q4nx) are produced by the
    converter itself. Everything else - tokenizer.json, tokenizer_config.json,
    config.json and optionally chat_template.jinja - is copied from the original
    HF/ModelScope model (local cache first, SDK download as backup), following the
    provenance recorded in the GGUF. If no source can be found, best-effort files
    are generated from GGUF metadata with a loud warning.
    """
    output_dir = Path(output_dir)
    if output_dir.suffix == ".q4nx":
        output_dir = output_dir.parent
    os.makedirs(output_dir, exist_ok=True)

    candidates = resolve_repo_candidates(reader, source_model)
    if source_model and not candidates:
        candidates = [source_model]

    fetched = []
    for candidate in candidates:
        fetched = _fetch_assets(candidate, output_dir, ASSET_FILES)
        missing = [f for f in REQUIRED_ASSETS if f not in fetched]
        if not missing:
            break
        if fetched:
            print(f"[WARN] Source {candidate} is missing required files: {missing}")

    if fetched:
        print(f"[INFO] Model assets present: {fetched}")

    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    else:
        print("[WARN] No source model found; generating config.json from GGUF metadata.")
        print("[WARN] tokenizer files may not exactly match the official model.")
        config = generate_config_from_gguf(reader)

    inject_flm_keys(config, q4nx_config, output_dir, flm_version)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    tokenizer_config_path = output_dir / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        generated = generate_tokenizer_config(reader)
        if generated:
            print("[WARN] Generating tokenizer_config.json from GGUF metadata (runtime-required).")
            with open(tokenizer_config_path, "w", encoding="utf-8") as f:
                json.dump(generated, f, indent=2, ensure_ascii=False)

    ensure_runtime_tokenizer_ids(reader, output_dir)

    chat_template_path = output_dir / "chat_template.jinja"
    if not chat_template_path.exists():
        chat_template = _gguf_field(reader, "tokenizer.chat_template")
        if chat_template:
            print("[INFO] Writing chat_template.jinja from GGUF metadata.")
            chat_template_path.write_text(chat_template, encoding="utf-8")

    print(f"[INFO] Model directory ready: {output_dir}")
