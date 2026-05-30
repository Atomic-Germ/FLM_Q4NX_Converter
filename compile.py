#!/usr/bin/env python3
"""
Discover, pre-check, and convert a HuggingFace GGUF model to Q4NX format.

Fetches architecture parameters from HF (config.json or GGUF header) before
downloading the full model, so incompatible models are rejected fast.

Usage:
    python compile.py -hf tencent/Hy-MT2-1.8B-GGUF
    python compile.py -hf Qwen/Qwen3-8B-GGUF -o ./output/
    python compile.py -hf some/model --check-only
    python compile.py -hf some/model --quant Q8_0 --keep-gguf
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# Hard constraints derived from libllama_npu.so and Q4NX tile layout
SUPPORTED_HIDDEN_SIZES = (2048, 3072, 4096)
QUANT_PREFERENCE = ["Q4_1", "Q4_0", "Q8_0"]


# ---------------------------------------------------------------------------
# HuggingFace helpers
# ---------------------------------------------------------------------------

def _hf_api():
    try:
        from huggingface_hub import HfApi
        return HfApi()
    except ImportError:
        print("[ERROR] huggingface_hub not found. Install with: pip install huggingface-hub")
        sys.exit(1)


def find_gguf_candidates(repo_id: str) -> dict:
    """Return GGUF filenames grouped by quantization type, in preference order."""
    api = _hf_api()
    try:
        files = list(api.list_repo_files(repo_id))
    except Exception as e:
        print(f"[ERROR] Could not list files in {repo_id!r}: {e}")
        sys.exit(1)

    gguf_files = [f for f in files if f.endswith('.gguf')]
    if not gguf_files:
        print(f"[ERROR] No GGUF files found in {repo_id!r}")
        print(f"  Files present: {files[:20]}")
        sys.exit(1)

    candidates = {}
    for name in gguf_files:
        # mmproj files are the vision encoder — kept separate from language model GGUFs
        if Path(name).name.lower().startswith('mmproj'):
            candidates.setdefault('mmproj', []).append(name)
            continue
        name_upper = name.upper()
        for quant in QUANT_PREFERENCE:
            if quant in name_upper:
                candidates.setdefault(quant, []).append(name)
                break
        else:
            candidates.setdefault('other', []).append(name)

    return candidates


def _fetch_json_from_repo(repo_id: str, filename: str) -> dict | None:
    """Download a small JSON file from a HF repo."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo_id, filename=filename,
                               local_dir=tempfile.mkdtemp())
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _get_base_model_id(repo_id: str) -> str | None:
    """Try to find the non-GGUF base model repo for config lookup."""
    api = _hf_api()
    try:
        info = api.model_info(repo_id)
        card = getattr(info, 'cardData', None) or {}
        base = card.get('base_model')
        if isinstance(base, str):
            return base
        if isinstance(base, list) and base:
            entry = base[0]
            return entry if isinstance(entry, str) else entry.get('name')
    except Exception:
        pass
    # Heuristic: strip common GGUF suffixes
    for suffix in ('-GGUF', '-gguf'):
        if repo_id.endswith(suffix):
            return repo_id[:-len(suffix)]
    return None


def fetch_chat_template(gguf_repo_id: str) -> str | None:
    """
    Find the best available chat template for a model, trying in order:
      1. chat_template.jinja directly in the GGUF repo
      2. chat_template.jinja in the base model repo
      3. chat_template field in the base model's tokenizer_config.json
         (if a list, pick the entry named 'default', else the only string)
    Returns the template string, or None if nothing is found.
    """
    from huggingface_hub import hf_hub_download

    def _try_jinja(repo_id: str) -> str | None:
        try:
            path = hf_hub_download(repo_id=repo_id, filename='chat_template.jinja',
                                   local_dir=tempfile.mkdtemp())
            return open(path, encoding='utf-8').read()
        except Exception:
            return None

    def _try_tokenizer_config(repo_id: str) -> str | None:
        cfg = _fetch_json_from_repo(repo_id, 'tokenizer_config.json')
        if not cfg:
            return None
        ct = cfg.get('chat_template')
        if isinstance(ct, str) and ct:
            return ct
        if isinstance(ct, list):
            # pick 'default' entry, or first entry if no default named
            default = next((e['template'] for e in ct if isinstance(e, dict) and e.get('name') == 'default'), None)
            if default:
                return default
            first = next((e['template'] for e in ct if isinstance(e, dict) and 'template' in e), None)
            return first
        return None

    # 1. jinja directly in GGUF repo
    tmpl = _try_jinja(gguf_repo_id)
    if tmpl:
        return tmpl

    base_id = _get_base_model_id(gguf_repo_id)
    if base_id:
        # 2. jinja in base model repo
        tmpl = _try_jinja(base_id)
        if tmpl:
            return tmpl
        # 3. tokenizer_config chat_template in base model repo
        tmpl = _try_tokenizer_config(base_id)
        if tmpl:
            return tmpl

    return None


def fetch_tokenizer_config(gguf_repo_id: str) -> dict | None:
    """
    Fetch tokenizer_config.json from the GGUF repo or its base model.
    The `chat_template` key is stripped if present because we write a
    separate chat_template.jinja file and the C++ runtime prefers that.
    Returns the config dict, or None if not found anywhere.
    """
    def _clean(cfg: dict) -> dict:
        cfg.pop('chat_template', None)
        return cfg

    cfg = _fetch_json_from_repo(gguf_repo_id, 'tokenizer_config.json')
    if cfg:
        return _clean(cfg)

    base_id = _get_base_model_id(gguf_repo_id)
    if base_id:
        cfg = _fetch_json_from_repo(base_id, 'tokenizer_config.json')
        if cfg:
            return _clean(cfg)

    return None


def _fetch_gguf_header_metadata(repo_id: str, filename: str) -> dict:
    """
    Download the first 2 MB of a GGUF via HTTP range request and parse
    all metadata fields from it (tensor data is not needed for this).
    """
    try:
        import urllib.request
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        req = urllib.request.Request(url, headers={"Range": "bytes=0-2097151"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
    except Exception as e:
        print(f"[WARN] Could not fetch GGUF header: {e}")
        return {}

    with tempfile.NamedTemporaryFile(delete=False, suffix='.gguf') as f:
        f.write(data)
        tmp = f.name

    try:
        from gguf import GGUFReader
        r = GGUFReader(tmp)
        result = {}
        for field in r.fields.values():
            try:
                val = field.parts[field.data[0]]
                result[field.name] = (
                    bytes(val).decode('utf-8') if field.types[0] == 8
                    else int(list(val)[0])
                )
            except Exception:
                pass
        return result
    except Exception:
        return {}
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# Parameter discovery
# ---------------------------------------------------------------------------

def gather_arch_params(repo_id: str, candidates: dict) -> dict:
    """
    Collect hidden_size, vocab_size, intermediate_size, and arch string.
    Tries (in order):
      1. config.json from the GGUF repo itself
      2. config.json from the detected base model repo
      3. First 2 MB of the smallest available GGUF (header-only parse)
    Returns a dict with keys: arch, hidden_size, vocab_size, intermediate_size, source.
    """
    result = dict(arch=None, hidden_size=None, vocab_size=None,
                  intermediate_size=None, layer_types=None, source=None)

    def _from_config(cfg: dict, label: str) -> dict | None:
        if not cfg:
            return None
        # For multimodal models (VLMs), text params live under text_config
        text_cfg = cfg.get('text_config') or cfg
        arch = text_cfg.get('model_type') or cfg.get('model_type') or (cfg.get('architectures') or [None])[0]
        hs = text_cfg.get('hidden_size')
        vs = text_cfg.get('vocab_size')
        is_ = text_cfg.get('intermediate_size')
        layer_types = text_cfg.get('layer_types')
        if hs or vs:
            return dict(arch=arch, hidden_size=hs, vocab_size=vs,
                        intermediate_size=is_, layer_types=layer_types,
                        source=f"config.json ({label})")
        return None

    # 1. Config from GGUF repo
    cfg = _fetch_json_from_repo(repo_id, 'config.json')
    if cfg:
        r = _from_config(cfg, repo_id)
        if r:
            result.update(r)
            return result

    # 2. Config from base model repo
    base_id = _get_base_model_id(repo_id)
    if base_id and base_id != repo_id:
        print(f"[INFO] Fetching base model config: {base_id}")
        cfg = _fetch_json_from_repo(base_id, 'config.json')
        if cfg:
            r = _from_config(cfg, base_id)
            if r:
                result.update(r)
                # Also try to get GGUF arch string from header (more reliable for converter check)
                # but don't block on it
                _enrich_with_gguf_arch(repo_id, candidates, result)
                return result

    # 3. GGUF header fallback
    first_quant = next((q for q in QUANT_PREFERENCE + ['other'] if q in candidates), None)
    if first_quant:
        first_file = candidates[first_quant][0]
        print(f"[INFO] Fetching GGUF header from {first_file}...")
        meta = _fetch_gguf_header_metadata(repo_id, first_file)
        if meta:
            arch = meta.get('general.architecture')
            if arch:
                result['arch'] = arch
                result['hidden_size'] = meta.get(f'{arch}.embedding_length')
                result['vocab_size'] = meta.get(f'{arch}.vocab_size')
                result['intermediate_size'] = meta.get(f'{arch}.feed_forward_length')
                result['source'] = f"GGUF header ({first_file})"

    return result


def _enrich_with_gguf_arch(repo_id: str, candidates: dict, result: dict):
    """If we got params from config.json, also peek at GGUF arch for converter check."""
    first_quant = next((q for q in QUANT_PREFERENCE + ['other'] if q in candidates), None)
    if not first_quant:
        return
    first_file = candidates[first_quant][0]
    print(f"[INFO] Fetching GGUF arch string from {first_file}...")
    meta = _fetch_gguf_header_metadata(repo_id, first_file)
    gguf_arch = meta.get('general.architecture')
    if gguf_arch:
        result['arch'] = gguf_arch  # prefer GGUF arch for converter lookup


# ---------------------------------------------------------------------------
# Compatibility checks
# ---------------------------------------------------------------------------

def check_compatibility(params: dict) -> list:
    """
    Return a list of (issue_text, is_blocking) tuples.
    is_blocking=True means conversion will definitely fail.
    is_blocking=False is a warning (may still work).
    """
    issues = []
    hidden_size = params.get('hidden_size')
    vocab_size = params.get('vocab_size')
    intermediate_size = params.get('intermediate_size')
    arch = params.get('arch')

    # hidden_size: each architecture's NPU library has its own compiled-in constraints,
    # so this is informational only — an unusual value is worth noting but not a blocker.
    if hidden_size is None:
        issues.append((
            "Could not determine hidden_size — check the model architecture manually.",
            False
        ))
    elif hidden_size not in SUPPORTED_HIDDEN_SIZES:
        issues.append((
            f"hidden_size={hidden_size} is outside the common set {SUPPORTED_HIDDEN_SIZES}.\n"
            f"    This may be fine if the architecture has its own NPU library (e.g. gemma4e_npu).\n"
            f"    Conversion will proceed — the kernel will catch it at runtime if unsupported.",
            False
        ))

    # vocab_size: lm_head tile row alignment
    if vocab_size is not None and vocab_size % 32 != 0:
        issues.append((
            f"vocab_size={vocab_size} is not divisible by 32.\n"
            f"    The lm_head NPU kernel tiles in rows of 32  (hard tile constraint).",
            True
        ))

    # intermediate_size: gate/up projection tile row alignment
    if intermediate_size is not None and intermediate_size % 32 != 0:
        issues.append((
            f"intermediate_size={intermediate_size} is not divisible by 32.\n"
            f"    FFN gate/up projection rows cannot be tiled.",
            True
        ))

    # Architecture: converter registration
    # Normalize config.json model_type strings to match GGUF arch names:
    #   - strip VLM tower suffixes: _text, _vision, _audio
    #   - collapse digit_digit sequences: qwen3_5 → qwen35, gemma_3 → gemma3
    if arch:
        import re
        arch_normalized = re.sub(r'_(text|vision|audio)$', '', arch, flags=re.IGNORECASE)
        arch_normalized = re.sub(r'(\d)_(\d)', r'\1\2', arch_normalized)
        try:
            from q4nx.constants import ModelArchNames
            # Match if normalized arch equals or is a prefix of any registered name
            registered = any(
                arch_normalized.lower() == name.lower() or
                name.lower().startswith(arch_normalized.lower())
                for names in ModelArchNames.values()
                for name in names
            )
            if not registered:
                hint = f" (normalized: {arch_normalized!r})" if arch_normalized != arch else ""
                issues.append((
                    f"Architecture {arch!r}{hint} has no registered converter.\n"
                    f"    A new model class would need to be added to FLM_Q4NX_Converter.",
                    True
                ))
        except ImportError:
            pass

    # Hybrid attention: SSM / linear_attention layers require kernels not present in FastFlowLM.
    # Sliding-window variants (sliding_attention, local_attention, etc.) are standard attention
    # with a reduced window — FastFlowLM handles these fine (e.g. Gemma4).
    # Only block known SSM / state-space / linear-attention compute patterns.
    _SSM_TYPES = {'linear_attention', 'mamba', 'mamba2', 'ssm', 'recurrent', 'rwkv', 'h3', 's6'}
    layer_types = params.get('layer_types')
    if layer_types:
        non_standard = sorted(set(t for t in layer_types if t not in _SSM_TYPES and 'attention' not in t.lower()))
        ssm_types = sorted(set(t for t in layer_types if t in _SSM_TYPES))
        if ssm_types:
            issues.append((
                f"Model uses SSM / linear-attention layer types: {ssm_types}.\n"
                f"    FastFlowLM has no SSM or state-space kernels.\n"
                f"    Hybrid SSM / linear-attention models cannot be converted.",
                True
            ))
        elif non_standard:
            issues.append((
                f"Model uses unrecognised layer types: {non_standard}.\n"
                f"    These may or may not be supported — check FastFlowLM release notes.",
                False  # non-blocking warning
            ))

    return issues


# ---------------------------------------------------------------------------
# Download + convert
# ---------------------------------------------------------------------------

def download_gguf(repo_id: str, filename: str, dest_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download
    print(f"[INFO] Downloading {filename}...")
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(dest_dir),
    )
    return Path(local_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Discover, pre-check, and convert a HuggingFace GGUF model to Q4NX.',
        epilog=(
            'Examples:\n'
            '  python compile.py -hf tencent/Hy-MT2-1.8B-GGUF\n'
            '  python compile.py -hf Qwen/Qwen3-8B-GGUF -o ./output/\n'
            '  python compile.py -hf some/model --check-only\n'
            '  python compile.py -hf some/model --quant Q8_0 --keep-gguf\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('-hf', '--huggingface', required=True, metavar='REPO_ID',
                        help='HuggingFace GGUF repo (e.g. tencent/Hy-MT2-1.8B-GGUF)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output folder (default: ./<model-name>/)')
    parser.add_argument('--check-only', action='store_true',
                        help='Check compatibility only — no download or conversion')
    parser.add_argument('--quant', choices=['Q4_1', 'Q4_0', 'Q8_0'], default=None,
                        help='Force a specific quantization level (default: auto, prefers Q4_1)')
    parser.add_argument('--keep-gguf', action='store_true',
                        help='Keep the downloaded GGUF file after conversion')
    parser.add_argument('--vision', action='store_true',
                        help='Also convert the vision encoder (mmproj*.gguf) to vision_weight.q4nx')

    args = parser.parse_args()
    repo_id = args.huggingface.strip('/')

    # ---- Discover files ----
    print(f"[INFO] Probing {repo_id!r}...")
    candidates = find_gguf_candidates(repo_id)

    print(f"[INFO] Available GGUF files:")
    for quant in QUANT_PREFERENCE + ['other']:
        for fname in candidates.get(quant, []):
            print(f"  [{quant:5s}] {fname}")
    for fname in candidates.get('mmproj', []):
        print(f"  [mmproj] {fname}")

    # ---- Gather architecture parameters ----
    params = gather_arch_params(repo_id, candidates)

    print(f"\n[INFO] Architecture parameters (source: {params.get('source', 'unknown')}):")
    print(f"  arch              = {params.get('arch')}")
    print(f"  hidden_size       = {params.get('hidden_size')}")
    print(f"  vocab_size        = {params.get('vocab_size')}")
    print(f"  intermediate_size = {params.get('intermediate_size')}")

    # ---- Compatibility checks ----
    issues = check_compatibility(params)
    blocking = [t for t, is_block in issues if is_block]
    warnings = [t for t, is_block in issues if not is_block]

    print()
    if warnings:
        print("[WARN] Non-blocking notes:")
        for w in warnings:
            for line in w.splitlines():
                print(f"  ! {line}")

    if blocking:
        print("[RESULT] This model cannot be converted to Q4NX:")
        for issue in blocking:
            lines = issue.splitlines()
            print(f"  x {lines[0]}")
            for line in lines[1:]:
                print(f"    {line}")
        print()
        print("[HINT] FastFlowLM NPU Q4NX hard constraints:")
        print(f"  * vocab_size must be divisible by 32 (universal tile constraint)")
        print(f"  * intermediate_size must be divisible by 32 (universal tile constraint)")
        print(f"  * Architecture must have a registered converter")
        print(f"  * hidden_size constraints vary per architecture's NPU library")
        sys.exit(1)

    print("[RESULT] Pre-check passed — model looks convertible.")

    if args.check_only:
        print("[INFO] --check-only specified, stopping here.")
        sys.exit(0)

    # ---- Pick GGUF to download ----
    chosen_quant = args.quant or next(
        (q for q in QUANT_PREFERENCE if q in candidates), None
    )
    if not chosen_quant or chosen_quant not in candidates:
        print(f"[ERROR] No suitable GGUF found. Available: {list(candidates.keys())}")
        sys.exit(1)

    chosen_file = candidates[chosen_quant][0]
    print(f"[INFO] Selected: [{chosen_quant}] {chosen_file}")

    # ---- Set up output directory ----
    model_name = repo_id.split('/')[-1]
    output_dir = Path(args.output) if args.output else Path(f'./{model_name}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Download ----
    gguf_path = download_gguf(repo_id, chosen_file, output_dir)

    # ---- Convert ----
    print(f"[INFO] Converting {gguf_path.name} -> {output_dir}/...")
    try:
        from q4nx import create_converter
        model = create_converter(str(gguf_path), '')
        model.convert(q4nx_path=str(output_dir), weights_type='language')
    except Exception as e:
        print(f"\n[ERROR] Conversion failed: {e}")
        if not args.keep_gguf:
            gguf_path.unlink(missing_ok=True)
        sys.exit(1)

    if not args.keep_gguf:
        gguf_path.unlink(missing_ok=True)
        print(f"[INFO] Removed temporary GGUF.")

    # ---- Fetch chat template ----
    print(f"[INFO] Fetching chat template...")
    tmpl = fetch_chat_template(repo_id)
    if tmpl:
        jinja_path = output_dir / 'chat_template.jinja'
        jinja_path.write_text(tmpl, encoding='utf-8')
        print(f"[INFO] chat_template.jinja written ({len(tmpl)} chars)")
    else:
        print(f"[WARN] No chat template found — chat_template.jinja not written.")

    # ---- Fetch tokenizer config ----
    print(f"[INFO] Fetching tokenizer_config.json...")
    tok_cfg = fetch_tokenizer_config(repo_id)
    if tok_cfg:
        tok_cfg_path = output_dir / 'tokenizer_config.json'
        tok_cfg_path.write_text(json.dumps(tok_cfg, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"[INFO] tokenizer_config.json written")
    else:
        print(f"[WARN] No tokenizer_config.json found — skipping.")

    print(f"\n[INFO] Done. Output: {output_dir}/model.q4nx")

    # ---- Optional vision + audio encoder conversion ----
    # Both vision and audio weights live in the same mmproj*.gguf file.
    # The converter filters by weights_type ('vision' vs 'audio') using the name_map.
    if args.vision:
        mmproj_files = candidates.get('mmproj', [])
        if not mmproj_files:
            print("[WARN] --vision specified but no mmproj*.gguf found in repo — skipping vision/audio conversion.")
        else:
            import shutil
            mmproj_file = mmproj_files[0]
            mmproj_path = download_gguf(repo_id, mmproj_file, output_dir)

            for wtype, out_name in [('vision', 'vision_weight.q4nx'), ('audio', 'audio_weight.q4nx')]:
                tmp_dir = output_dir / f'_{wtype}_tmp'
                tmp_dir.mkdir(exist_ok=True)
                print(f"[INFO] Converting {wtype} encoder -> {output_dir}/{out_name} ...")
                try:
                    from q4nx import create_converter
                    enc_model = create_converter(str(mmproj_path), '')
                    enc_model.convert(q4nx_path=str(tmp_dir), weights_type=wtype)
                    (tmp_dir / 'model.q4nx').rename(output_dir / out_name)
                    print(f"[INFO] {wtype.capitalize()} encoder saved: {output_dir}/{out_name}")
                except Exception as e:
                    print(f"[WARN] {wtype.capitalize()} conversion failed (skipping): {e}")
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

            if not args.keep_gguf:
                mmproj_path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
