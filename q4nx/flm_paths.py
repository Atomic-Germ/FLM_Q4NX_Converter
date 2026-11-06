"""Where FastFlowLM keeps things, on Linux and on Windows.

FLM resolves its files like this (checked against FLM 1.0.12 on both platforms):

* install dir:  next to the ``flm`` executable.  Linux packages use
  ``/opt/fastflowlm/share/flm`` (or ``/usr[/local]/share/flm``).  The Windows
  installer uses ``%ProgramFiles%\\flm`` and does not put it on PATH.
  ``model_list.json``, ``model_info.json`` and ``xclbins/`` live there.
* user root:    ``$FLM_MODEL_PATH`` if set, else ``~/.config/flm`` (Linux) or
  ``%USERPROFILE%\\.flm`` (Windows).  Models are read from ``<user root>/models/<name>``.
* overrides:    ``FLM_CONFIG_PATH`` (model_list.json), ``FLM_MODELINFO_PATH``
  (model_info.json), ``FLM_XCLBIN_PATH`` (directory that contains ``xclbins/``).
"""
import os
import shutil
from pathlib import Path
from typing import Optional

IS_WINDOWS = os.name == "nt"


def find_flm_executable() -> Optional[str]:
    exe = shutil.which("flm")
    if exe:
        return exe
    if IS_WINDOWS:
        for base in (os.environ.get("ProgramW6432"), os.environ.get("ProgramFiles"), r"C:\Program Files"):
            if base and (Path(base) / "flm" / "flm.exe").is_file():
                return str(Path(base) / "flm" / "flm.exe")
    return None


def user_flm_root() -> Path:
    """Directory that holds ``models/`` (FLM_MODEL_PATH semantics)."""
    env = os.environ.get("FLM_MODEL_PATH")
    if env:
        return Path(env)
    return Path.home() / (".flm" if IS_WINDOWS else ".config/flm")


def is_dir_link(path: Path) -> bool:
    """True for a symlink or (Windows) a directory junction."""
    path = Path(path)
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)          # Python 3.12+
    if is_junction is not None:
        return is_junction()
    if IS_WINDOWS and path.exists():
        try:
            os.readlink(path)                                  # also reads junctions (3.8+)
            return True
        except OSError:
            return False
    return False


def _norm(p) -> str:
    p = str(p)
    for prefix in ("\\\\?\\", "\\??\\"):    # junction targets can read back as \\?\C:\... or \??\C:\...
        if p.startswith(prefix):
            p = p[len(prefix):]
    return os.path.normcase(os.path.normpath(p))


def link_target(path: Path) -> Optional[str]:
    try:
        return _norm(os.readlink(path))
    except OSError:
        return None


def remove_dir_link(path: Path) -> None:
    """Remove a symlink/junction itself, never what it points to."""
    path = Path(path)
    if not is_dir_link(path):
        raise ValueError(f"{path} is not a link; refusing to remove it")
    if path.is_symlink() and not IS_WINDOWS:
        path.unlink()
    else:
        os.rmdir(path)      # removes a junction or a directory symlink without touching the target


def link_dir(target: Path, link: Path) -> str:
    """Point ``link`` at directory ``target``.  Returns 'symlink' or 'junction'.

    Windows symlinks need admin rights or Developer Mode; a junction needs
    neither, so it is the fallback there.
    """
    target, link = Path(target), Path(link)
    if is_dir_link(link):
        if link_target(link) == _norm(target):
            return "existing"
        remove_dir_link(link)
    elif link.exists():
        raise FileExistsError(f"{link} exists and is not a link")
    try:
        os.symlink(target, link, target_is_directory=True)
        return "symlink"
    except OSError:
        if not IS_WINDOWS:
            raise
    import _winapi
    _winapi.CreateJunction(str(target), str(link))
    return "junction"


def place_file(src: Path, dst: Path, hardlink: bool = False) -> str:
    """Copy ``src`` to ``dst``, or hardlink it when asked and possible (same volume).

    A hardlink costs no disk space, but both names then share one file:
    rewriting the source in place also changes the deployed copy.
    """
    src, dst = Path(src), Path(dst)
    if dst.exists():
        if hardlink and os.path.samefile(src, dst):
            return "existing"
        dst.unlink()
    if hardlink:
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


def registry_supports_xclbin_folder(registry: dict) -> bool:
    """The FLM release ships entries with ``xclbin_folder`` (an entry may reuse
    another model's kernel folder), so a custom entry can use it too."""
    for bucket in registry.get("models", {}).values():
        if isinstance(bucket, dict):
            for entry in bucket.values():
                if isinstance(entry, dict) and "xclbin_folder" in entry:
                    return True
    return False


def env_hint(name: str, value) -> str:
    """How to set an environment variable for future shells, per platform."""
    if IS_WINDOWS:
        return (f'setx {name} "{value}"        '
                f'(new terminals; this PowerShell session: $env:{name} = "{value}")')
    return f'export {name}="{value}"        (add to ~/.bashrc)'
