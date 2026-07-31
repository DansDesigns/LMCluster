"""Skills: small Python functions the cluster can call.

Ported from LMCluster's skills_manager. Round-Table had no equivalent, so
this is the main thing the merge carries over from that side.

A skill is a .py file in the state directory exposing `run(inputs) ->
dict`, with metadata in its module docstring:

    \"\"\"
    Skill: Word Count
    Description: Count words in a string
    Version: 1.0
    Inputs: {"text": "string"}
    Outputs: {"count": "number"}
    Tags: text, built-in
    \"\"\"

    def run(inputs):
        return {"count": len(inputs.get("text", "").split())}


SECURITY — READ THIS
--------------------
Skills are NOT sandboxed. They run as ordinary Python in this process,
with this process's privileges, and can do anything your user can do.

The original implementation swapped in a "restricted_builtins" dict, which
looks like a sandbox and is not one: it still exposed `os` and `sys`, and
replacing __builtins__ does not stop a module from importing whatever it
likes. Rather than keep a defence that only creates false confidence, this
port drops the pretence and states the real position: **a skill is code
you have chosen to run**.

That matters more here than it did in standalone LMCluster, because this
package also exposes an HTTP API on the LAN. Skill execution therefore
sits behind the cluster token, and `validate()` is a lint that catches
obvious mistakes — not a security boundary. Treat the ability to write a
skill as equivalent to shell access on the node, and only put skills on a
cluster whose token you control.
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time

_LOCK = threading.Lock()
_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def skills_dir() -> str:
    if os.name == "nt":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
    else:
        base = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    path = os.path.join(base, "lmcluster", "skills")
    os.makedirs(path, exist_ok=True)
    return path


def _path_for(skill_id: str) -> str:
    """Resolve a skill id to a path, refusing anything that escapes the
    skills directory. The id comes off an HTTP route, so traversal is a
    live concern rather than a theoretical one."""
    if not _ID_RE.match(skill_id):
        raise ValueError("skill id must be 1-64 chars of [A-Za-z0-9_-]")
    return os.path.join(skills_dir(), f"{skill_id}.py")


# -- metadata -------------------------------------------------------------

_FIELDS = {
    "skill": "name",
    "description": "description",
    "version": "version",
    "author": "author",
    "tags": "tags",
}


def parse_metadata(source: str) -> dict:
    meta = {"name": "Untitled", "description": "", "version": "1.0",
            "author": "cluster", "inputs": {}, "outputs": {}, "tags": []}

    lines, doc, inside = source.split("\n"), [], False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if inside:
                break
            inside = True
            # Handle a docstring that opens with content on the same line.
            rest = stripped[3:].strip()
            if rest:
                doc.append(rest)
            continue
        if inside:
            doc.append(stripped)

    for entry in doc:
        if ":" not in entry:
            continue
        key, value = entry.split(":", 1)
        key, value = key.strip().lower(), value.strip()
        if key in _FIELDS:
            if key == "tags":
                meta["tags"] = [t.strip() for t in value.split(",") if t.strip()]
            else:
                meta[_FIELDS[key]] = value
        elif key in ("inputs", "outputs"):
            try:
                meta[key] = json.loads(value)
            except ValueError:
                meta[key] = {}
    return meta


# -- loading --------------------------------------------------------------

def load(skill_id: str, with_module: bool = True) -> dict | None:
    path = _path_for(skill_id)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        source = f.read()

    info = parse_metadata(source)
    info.update({
        "id": skill_id,
        "path": path,
        "size": len(source),
        "modified": os.path.getmtime(path),
        "status": "ok",
        "has_run": "def run(" in source,
    })
    if not with_module:
        return info

    spec = importlib.util.spec_from_file_location(f"lmcluster_skill_{skill_id}",
                                                  path)
    if spec is None or spec.loader is None:
        info["status"] = "error: cannot build module spec"
        info["_module"] = None
        return info

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        info["_module"] = module
        info["has_run"] = callable(getattr(module, "run", None))
    except Exception as e:  # a skill is arbitrary code; anything can happen
        info["status"] = f"error: {type(e).__name__}: {e}"
        info["_module"] = None
    return info


def _public(info: dict) -> dict:
    return {k: v for k, v in info.items() if not k.startswith("_")}


def list_all() -> list[dict]:
    with _LOCK:
        out = []
        for name in sorted(os.listdir(skills_dir())):
            if not name.endswith(".py") or name.startswith("_"):
                continue
            info = load(name[:-3], with_module=False)
            if info:
                out.append(_public(info))
        return out


def get(skill_id: str) -> dict | None:
    """One skill, including its source.

    The source is included here but deliberately not in list_all(), which
    would otherwise return every skill's full text on each poll just to
    render a list of names.
    """
    with _LOCK:
        info = load(skill_id)
        if info is None:
            return None
        out = _public(info)
        try:
            with open(info["path"], encoding="utf-8") as f:
                out["source"] = f.read()
        except OSError:
            out["source"] = ""
        return out


def save(skill_id: str, source: str) -> dict:
    with _LOCK:
        path = _path_for(skill_id)
        with open(path, "w", encoding="utf-8") as f:
            f.write(source)
        info = load(skill_id)
        return _public(info)


def delete(skill_id: str) -> bool:
    with _LOCK:
        path = _path_for(skill_id)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False


# -- execution ------------------------------------------------------------

_RUNNER = r'''
import json, sys, importlib.util
path, skill_id = sys.argv[1], sys.argv[2]
payload = json.loads(sys.stdin.read() or "{}")
spec = importlib.util.spec_from_file_location("skill_" + skill_id, path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
result = mod.run(payload)
sys.stdout.flush()
sys.stderr.write("\x00RESULT\x00" + json.dumps(result, default=str))
'''


def execute(skill_id: str, inputs: dict | None = None,
            timeout: float = 30.0) -> dict:
    """Run a skill's run(inputs) in a subprocess, capturing its output.

    A subprocess rather than a thread, for three reasons found the hard
    way:

      * The original signature accepted `timeout` and then ignored it, so
        a skill containing `while True:` wedged the node permanently. A
        subprocess can actually be killed.
      * `contextlib.redirect_stdout` swaps the *process-global* sys.stdout.
        Capturing a skill's output on a worker thread therefore swallows
        every other thread's output too, including the web server's, for
        as long as the skill runs. On a spinning skill that means forever.
      * A crashing skill takes its own process down instead of the node's.

    The subprocess still runs with full user privileges — see the module
    docstring. This buys robustness against accidents, not safety against
    malice.
    """
    inputs = inputs or {}
    with _LOCK:
        info = load(skill_id, with_module=False)

    if info is None:
        return {"success": False, "error": f"skill '{skill_id}' not found"}
    if not info["has_run"]:
        return {"success": False,
                "error": "skill has no run(inputs) function"}

    started = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _RUNNER, info["path"], skill_id],
            input=json.dumps(inputs), capture_output=True, text=True,
            timeout=timeout,
            # Skills import from the package (see cluster_info), so the
            # project root has to be importable in the child.
            env={**os.environ, "PYTHONPATH": _project_root()
                 + os.pathsep + os.environ.get("PYTHONPATH", "")})
    except subprocess.TimeoutExpired as e:
        return {"success": False, "skill": skill_id,
                "seconds": round(time.time() - started, 3),
                "error": f"timed out after {timeout:.0f}s and was killed",
                "stdout": (e.stdout or b"").decode(errors="replace")
                          if isinstance(e.stdout, bytes) else (e.stdout or ""),
                "stderr": ""}
    except OSError as e:
        return {"success": False, "skill": skill_id,
                "error": f"could not start skill process: {e}"}

    common = {"skill": skill_id,
              "seconds": round(time.time() - started, 3),
              "stdout": proc.stdout}

    stderr, _, encoded = proc.stderr.partition("\x00RESULT\x00")
    if proc.returncode != 0:
        tail = stderr.strip().splitlines()
        return {"success": False, "error": tail[-1] if tail else
                f"skill exited with status {proc.returncode}",
                "stderr": stderr, **common}
    try:
        result = json.loads(encoded) if encoded else None
    except ValueError:
        return {"success": False, "stderr": stderr,
                "error": "skill returned something that is not JSON "
                         "serialisable", **common}
    return {"success": True, "result": result, "stderr": stderr, **common}


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# -- authoring ------------------------------------------------------------

def template(name: str, description: str,
             inputs: dict, outputs: dict) -> str:
    return f'''"""
Skill: {name}
Description: {description}
Version: 1.0
Author: cluster
Inputs: {json.dumps(inputs)}
Outputs: {json.dumps(outputs)}
Tags: generated
"""


def run(inputs):
    """Execute the skill.

    Args:
        inputs: dict of input values
    Returns:
        dict of output values
    """
    raise NotImplementedError("fill this in")
'''


def validate(source: str) -> tuple[bool, list[str]]:
    """Lint a skill before saving.

    Explicitly a lint, not a sandbox. It catches the mistakes people
    actually make — no run(), a syntax error, an accidental rmtree — and
    makes no attempt to stop someone determined to do damage, because a
    blocklist of import names cannot.
    """
    problems = []

    if "def run(" not in source:
        problems.append("missing required run(inputs) function")

    try:
        compile(source, f"<skill>", "exec")
    except SyntaxError as e:
        problems.append(f"syntax error on line {e.lineno}: {e.msg}")

    risky = {
        "shutil.rmtree": "recursive delete",
        "os.remove": "file delete",
        "os.rmdir": "directory delete",
        "subprocess": "spawns processes",
        "eval(": "evaluates arbitrary strings",
        "exec(": "executes arbitrary strings",
    }
    for needle, why in risky.items():
        if needle in source:
            problems.append(f"warning: contains {needle!r} ({why}) — allowed, "
                            f"but be sure you meant it")

    # Warnings alone should not block a save; only hard errors do.
    hard = [p for p in problems if not p.startswith("warning:")]
    return len(hard) == 0, problems


BUILTIN = {
    "word_count": '''"""
Skill: Word Count
Description: Count words, characters and lines in a block of text
Version: 1.0
Author: cluster
Inputs: {"text": "string"}
Outputs: {"words": "number", "characters": "number", "lines": "number"}
Tags: text, built-in
"""


def run(inputs):
    text = inputs.get("text", "")
    return {
        "words": len(text.split()),
        "characters": len(text),
        "lines": len(text.splitlines()) or (1 if text else 0),
    }
''',
    "cluster_info": '''"""
Skill: Cluster Info
Description: Report this node's hardware, as the shard planner sees it
Version: 1.0
Author: cluster
Inputs: {}
Outputs: {"ram_total": "number", "ram_free": "number", "gpu": "string"}
Tags: diagnostics, built-in
"""

from lmcluster import hardware


def run(inputs):
    hw = hardware.probe()
    return {
        "os": hw["os"],
        "cpu_count": hw["cpu_count"],
        "ram_total_gb": hardware.gb(hw["ram_total"]),
        "ram_free_gb": hardware.gb(hw["ram_free"]),
        "gpu": hw["gpu_backend"],
        "gpu_name": hw["gpu_name"],
        "disk_free_gb": hardware.gb(hw["disk_free"]),
    }
''',
}


def ensure_builtins():
    """Write the built-in skills if they are not already present."""
    for skill_id, source in BUILTIN.items():
        path = _path_for(skill_id)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(source)
