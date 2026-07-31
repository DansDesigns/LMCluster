"""Finding model files.

The first version of this made you type a folder path into the settings
page before anything worked, which is a poor way to greet somebody who has
just spent half an hour compiling llama.cpp. Most people already have GGUF
files somewhere, and there are only a handful of places they are likely to
be, so it is better to go and look.

Two things are handled here that a naive scan gets wrong.

The first is that llama.cpp has its own cache, which is where anything
downloaded with `llama-server -hf` ends up, and which its router mode reads
by default. Its location depends on the platform and on several environment
variables, in a documented order of preference.

The second is that model files are frequently not sitting flat in a folder.
llama.cpp's own convention is that a model split across several files goes
in a subdirectory of its own, and the Hugging Face cache nests everything
several levels deep under names like
`models--unsloth--Qwen3-235B-GGUF/snapshots/<hash>/`. A scan that only
looks at the top level of a directory finds nothing at all in either case,
which is exactly the bug this module was written to fix.
"""

import os
import platform

# Deeper than the Hugging Face cache layout needs, shallow enough that
# pointing this at a large disk by mistake does not take all afternoon.
MAX_DEPTH = 5

# The projector file for a multimodal model. It sits next to the model it
# belongs to and cannot be loaded on its own, so listing it as though it
# were a model just invites a confusing failure.
_NOT_MODELS = ("mmproj",)


def _home(*parts) -> str:
    return os.path.join(os.path.expanduser("~"), *parts)


def llamacpp_cache() -> str:
    """Where llama.cpp keeps models downloaded with -hf.

    Order of preference follows llama.cpp's own: LLAMA_CACHE wins, then the
    Hugging Face cache variables that newer builds also consult, then the
    platform default.
    """
    explicit = os.environ.get("LLAMA_CACHE")
    if explicit:
        return explicit

    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(var)
        if value:
            return value
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return os.path.join(hf_home, "hub")

    system = platform.system()
    if system == "Windows":
        base = os.environ.get("LOCALAPPDATA") or _home("AppData", "Local")
        return os.path.join(base, "llama.cpp")
    if system == "Darwin":
        return _home("Library", "Caches", "llama.cpp")
    xdg = os.environ.get("XDG_CACHE_HOME")
    return os.path.join(xdg, "llama.cpp") if xdg else _home(".cache", "llama.cpp")


def candidates(project_root: str | None = None) -> list[dict]:
    """Places worth looking, best first, with what is actually in each.

    Returned even when empty or missing, because telling somebody which
    folders were checked is far more useful than an unqualified "no models
    found".
    """
    seen, out = set(), []

    def add(path: str, label: str):
        if not path:
            return
        real = os.path.abspath(os.path.expanduser(path))
        if real in seen:
            return
        seen.add(real)
        exists = os.path.isdir(real)
        out.append({
            "path": real,
            "label": label,
            "exists": exists,
            "count": len(find(real)) if exists else 0,
        })

    add(llamacpp_cache(), "llama.cpp cache")
    if project_root:
        add(os.path.join(project_root, "models"), "models folder here")
    add(_home("models"), "models in your home folder")

    system = platform.system()
    if system == "Windows":
        add(_home(".lmstudio", "models"), "LM Studio")
        add(os.path.join(os.environ.get("USERPROFILE", _home()),
                         "Documents", "models"), "Documents/models")
    else:
        add(_home(".lmstudio", "models"), "LM Studio")
        add(_home(".cache", "lm-studio", "models"), "LM Studio (older layout)")
    if system == "Linux":
        add("/srv/models", "/srv/models")
        add("/opt/models", "/opt/models")

    return out


def default_dir(project_root: str | None = None) -> str | None:
    """The first candidate folder that actually contains models."""
    for c in candidates(project_root):
        if c["exists"] and c["count"]:
            return c["path"]
    return None


def _is_model(name: str) -> bool:
    if not name.endswith(".gguf"):
        return False
    lowered = name.lower()
    return not any(bad in lowered for bad in _NOT_MODELS)


def _walk(directory: str, depth: int = 0):
    """Yield .gguf files, descending a bounded number of levels."""
    if depth > MAX_DEPTH:
        return
    try:
        entries = list(os.scandir(directory))
    except OSError:
        return
    for entry in entries:
        try:
            if entry.is_file(follow_symlinks=False) and _is_model(entry.name):
                yield entry
            elif entry.is_dir(follow_symlinks=False):
                # Hugging Face's cache keeps a blobs directory of content
                # addressed files alongside the readable snapshots tree.
                # The same bytes appear in both, so skipping blobs avoids
                # listing every model twice under an unreadable name.
                if entry.name in ("blobs", ".git", ".no_exist"):
                    continue
                yield from _walk(entry.path, depth + 1)
        except OSError:
            continue


def find(directory: str) -> list[dict]:
    """Every loadable model under a folder.

    A model split into several files is reported once, under its first
    part, with the total size of all its parts, because that first part is
    what you hand to llama.cpp and the total is what has to fit in memory.
    """
    if not directory or not os.path.isdir(directory):
        return []

    parts: dict[str, list] = {}
    singles = []

    for entry in _walk(directory):
        name = entry.name
        if "-of-" in name:
            # e.g. model-00002-of-00004.gguf -> group key "model"
            stem = name.rsplit("-", 3)[0]
            parts.setdefault(os.path.join(os.path.dirname(entry.path), stem),
                             []).append(entry)
        else:
            singles.append(entry)

    models = []
    for entry in singles:
        try:
            size = entry.stat().st_size
        except OSError:
            continue
        models.append({"name": entry.name, "path": entry.path,
                       "size": size, "files": 1,
                       "folder": os.path.dirname(entry.path)})

    for key, group in parts.items():
        group.sort(key=lambda e: e.name)
        total = 0
        for entry in group:
            try:
                total += entry.stat().st_size
            except OSError:
                pass
        first = group[0]
        models.append({"name": first.name, "path": first.path,
                       "size": total, "files": len(group),
                       "folder": os.path.dirname(first.path)})

    models.sort(key=lambda m: m["name"].lower())
    return models


def resolve(configured: str, project_root: str | None = None) -> dict:
    """Work out which folder to use and report how that was decided.

    An explicitly configured folder always wins, even if it turns out to be
    empty or missing, because silently searching elsewhere when somebody has
    told you where to look is worse than showing them an empty list.
    """
    if configured:
        path = os.path.abspath(os.path.expanduser(configured))
        if not os.path.isdir(path):
            return {"dir": path, "models": [], "source": "configured",
                    "error": f"That folder does not exist: {path}",
                    "candidates": candidates(project_root)}
        models = find(path)
        return {"dir": path, "models": models, "source": "configured",
                "error": None if models else
                         f"No .gguf files found in {path}",
                "candidates": candidates(project_root)}

    found = candidates(project_root)
    for c in found:
        if c["exists"] and c["count"]:
            return {"dir": c["path"], "models": find(c["path"]),
                    "source": c["label"], "error": None,
                    "candidates": found}

    return {"dir": None, "models": [], "source": None,
            "error": "No model files found yet. Put a .gguf file in one of "
                     "the folders below, or set your own in Settings.",
            "candidates": found}
