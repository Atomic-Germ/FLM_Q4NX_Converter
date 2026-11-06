"""Install/registration tests for flm-add.py and q4nx.deploy against a fake FLM
install in a temp directory.  Runs on Linux and Windows; each platform checks its
own behaviour (Linux: ~/.config/flm + symlinked xclbins; Windows: %USERPROFILE%/.flm
+ xclbin_folder, or a junction when the registry has no xclbin_folder entries).

    python -m pytest tests/test_install_paths.py   (or: python tests/test_install_paths.py)
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from q4nx import flm_paths  # noqa: E402

IS_WINDOWS = os.name == "nt"
OFFICIAL = "Qwen3-4B-NPU2"


def _fake_install(base: Path, with_xclbin_folder: bool) -> Path:
    """<base>/flm with model_list.json + xclbins/<OFFICIAL>/, like an FLM install."""
    inst = base / "flm"
    (inst / "xclbins" / OFFICIAL).mkdir(parents=True)
    (inst / "xclbins" / OFFICIAL / "layer.xclbin").write_bytes(b"x")
    entry = {"name": OFFICIAL, "size": 4000000000, "files": ["config.json", "model.q4nx"],
             "details": {"format": "NPU2", "family": "qwen3"}, "max_prefill_len": 4096}
    models = {"qwen3": {"4b": entry}}
    if with_xclbin_folder:
        models["qwen3-drafter"] = {"4b": dict(entry, name="Qwen3-4B-Drafter-NPU2", xclbin_folder=OFFICIAL)}
    (inst / "model_list.json").write_text(json.dumps({"model_path": "models", "models": models}))
    return inst


def _fake_model(base: Path, name: str) -> Path:
    d = base / "src" / name
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    for f in ("model.q4nx", "tokenizer.json", "tokenizer_config.json"):
        (d / f).write_bytes(b"0" * 64)
    return d


def _env(home: Path) -> dict:
    env = {k: v for k, v in os.environ.items()
           if k not in ("FLM_CONFIG_PATH", "FLM_MODEL_PATH", "FLM_XCLBIN_PATH", "FLM_MODELINFO_PATH")}
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    return env


def _xclbins_dir_for_flm_add(inst):
    # flm-add looks for `flm` on PATH to find the xclbins; give it a stub
    stub = inst / ("flm.exe" if IS_WINDOWS else "flm")
    stub.write_text("")
    if not IS_WINDOWS:
        stub.chmod(0o755)


def test_user_root_defaults():
    old = os.environ.pop("FLM_MODEL_PATH", None)
    try:
        want = Path.home() / (".flm" if IS_WINDOWS else ".config/flm")
        assert flm_paths.user_flm_root() == want
        os.environ["FLM_MODEL_PATH"] = str(Path("somewhere"))
        assert flm_paths.user_flm_root() == Path("somewhere")
    finally:
        os.environ.pop("FLM_MODEL_PATH", None)
        if old is not None:
            os.environ["FLM_MODEL_PATH"] = old


def test_link_dir_and_safe_removal():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        target = t / "target"
        target.mkdir()
        (target / "keep.txt").write_text("keep")
        link = t / "link"
        kind = flm_paths.link_dir(target, link)
        assert kind in ("symlink", "junction")
        assert flm_paths.is_dir_link(link) and (link / "keep.txt").read_text() == "keep"
        assert flm_paths.link_dir(target, link) == "existing"
        flm_paths.remove_dir_link(link)
        assert not link.exists() and (target / "keep.txt").exists(), "removing a link must not touch its target"


def test_place_file_hardlink_and_copy():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        (t / "a").write_bytes(b"abc")
        assert flm_paths.place_file(t / "a", t / "b", hardlink=True) in ("hardlink", "copy")
        assert flm_paths.place_file(t / "a", t / "c") == "copy"
        assert (t / "b").read_bytes() == (t / "c").read_bytes() == b"abc"


def test_flm_add_platform_defaults():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        home = t / "home"
        home.mkdir()
        inst = _fake_install(t, with_xclbin_folder=True)
        _xclbins_dir_for_flm_add(inst)
        src = _fake_model(t, "Qwen3-4B-Test-NPU2")
        env = _env(home)
        env["PATH"] = str(inst) + os.pathsep + env.get("PATH", "")
        out = subprocess.run([sys.executable, str(ROOT / "flm-add.py"), str(src), "--tag", "qwen3-test:4b",
                              "--system-list", str(inst / "model_list.json")],
                             env=env, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr + out.stdout
        root = home / (".flm" if IS_WINDOWS else ".config/flm")
        reg = json.loads((root / "model_list.json").read_text())
        entry = reg["models"]["qwen3-test"]["4b"]
        assert entry["name"] == "Qwen3-4B-Test-NPU2"
        assert (root / "models" / "Qwen3-4B-Test-NPU2" / "model.q4nx").is_file()
        link = root / "xclbins" / "Qwen3-4B-Test-NPU2"
        if IS_WINDOWS:
            # registry supports xclbin_folder -> no link, entry points at the official kernels
            assert entry.get("xclbin_folder") == OFFICIAL
            assert not link.exists()
            assert "setx FLM_CONFIG_PATH" in out.stdout
        else:
            # Linux: unchanged symlink flow, no xclbin_folder added
            assert "xclbin_folder" not in entry
            assert link.is_symlink() and (link / "layer.xclbin").is_file()
            assert "export FLM_CONFIG_PATH" in out.stdout


def test_flm_add_without_xclbin_folder_support_links():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        home = t / "home"
        home.mkdir()
        inst = _fake_install(t, with_xclbin_folder=False)
        _xclbins_dir_for_flm_add(inst)
        src = _fake_model(t, "Qwen3-4B-Test-NPU2")
        env = _env(home)
        env["PATH"] = str(inst) + os.pathsep + env.get("PATH", "")
        out = subprocess.run([sys.executable, str(ROOT / "flm-add.py"), str(src), "--tag", "qwen3-test:4b",
                              "--system-list", str(inst / "model_list.json")],
                             env=env, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr + out.stdout
        root = home / (".flm" if IS_WINDOWS else ".config/flm")
        entry = json.loads((root / "model_list.json").read_text())["models"]["qwen3-test"]["4b"]
        assert "xclbin_folder" not in entry
        link = root / "xclbins" / "Qwen3-4B-Test-NPU2"
        assert flm_paths.is_dir_link(link) and (link / "layer.xclbin").is_file()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
