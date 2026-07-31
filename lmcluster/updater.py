"""Version check against the GitHub repository.

Compares the local version.txt (project root) with the one on GitHub at
DansDesigns/Round-Table. Network failure, missing file, or a repo that
doesn't exist yet are all reported honestly rather than guessed at.
"""

import os

import httpx

REPO = "DansDesigns/LMCluster"
BRANCHES = ("main", "master")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def local_version() -> str | None:
    """This installation's version, if it has one.

    Read from version.txt beside the code. A missing file is not an error:
    a checkout that has not been tagged simply has no version, and the
    update check reports that it cannot compare rather than inventing a
    number.
    """
    try:
        with open(os.path.join(_ROOT, "version.txt"), encoding="utf-8") as f:
            text = f.read().strip()
            return text or None
    except OSError:
        return None


def _parse(v: str) -> tuple:
    """Tolerant version parse: '0.1.0' -> (0,1,0). Non-numeric parts
    compare as strings after the numeric prefix."""
    parts = []
    for p in v.strip().lstrip("v").split("."):
        parts.append(int(p) if p.isdigit() else p)
    return tuple(parts)


async def check() -> dict:
    local = local_version()
    result = {
        "local": local,
        "remote": None,
        "update_available": False,
        "repo_url": f"https://github.com/{REPO}",
        "error": None,
    }
    if local is None:
        result["error"] = "this installation has no version.txt, so there is nothing to compare against. Add one to the repository and to this folder if you want update checking."
        return result

    last_err = None
    for branch in BRANCHES:
        url = (f"https://raw.githubusercontent.com/{REPO}/{branch}"
               "/version.txt")
        try:
            async with httpx.AsyncClient(timeout=10,
                                         follow_redirects=True) as client:
                r = await client.get(url)
            if r.status_code == 404:
                last_err = f"version.txt not found on {branch} branch"
                continue
            r.raise_for_status()
            remote = r.text.strip()
            if not remote:
                last_err = "remote version.txt is empty"
                continue
            result["remote"] = remote
            try:
                result["update_available"] = _parse(remote) > _parse(local)
            except TypeError:
                # mixed numeric/string parts; fall back to inequality
                result["update_available"] = remote != local
            return result
        except httpx.HTTPError as e:
            last_err = f"could not reach GitHub: {e}"
    result["error"] = last_err or "unknown error"
    return result


# -- installing an update -------------------------------------------------

import io
import os as _os
import shutil
import stat
import sys
import threading
import time
import zipfile

# Never overwritten by an update: your settings, your cluster identity, the
# downloaded llama.cpp, the virtual environment, and the update workspace
# itself.
KEEP = {"lmcluster.toml", "lmcluster.toml.bak", ".venv", "venv", "vendor",
        "tmp", ".git", "__pycache__"}

TMP_DIR = _os.path.join(_ROOT, "tmp")


def _archive_urls() -> list:
    return [f"https://github.com/{REPO}/archive/refs/heads/{b}.zip"
            for b in BRANCHES]


def _force_remove(path: str):
    if not _os.path.isdir(path):
        return

    def on_error(func, target, exc_info):
        try:
            _os.chmod(target, stat.S_IWRITE)
            func(target)
        except OSError:
            pass

    shutil.rmtree(path, onerror=on_error)


async def download_update() -> dict:
    """Fetch the current code into tmp/ and check it looks like LMCluster.

    Nothing outside tmp/ is touched here. The unpacked copy is left in
    place so that install_update can move it in as a separate, quick step —
    the same reasoning as the llama.cpp downloader: a long operation that
    can fail should not be the one holding your working installation open.
    """
    info = await check()
    if info["error"]:
        return {"ok": False, "message": info["error"], **info}

    _os.makedirs(TMP_DIR, exist_ok=True)
    staging = _os.path.join(TMP_DIR, "update")
    _force_remove(staging)

    data, last_err = None, None
    for url in _archive_urls():
        try:
            async with httpx.AsyncClient(timeout=120,
                                         follow_redirects=True) as client:
                r = await client.get(url)
            if r.status_code == 404:
                last_err = f"no archive at {url}"
                continue
            r.raise_for_status()
            data = r.content
            break
        except httpx.HTTPError as e:
            last_err = f"could not download: {e}"

    if data is None:
        return {"ok": False, "message": last_err or "download failed", **info}

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            bad = zf.testzip()
            if bad:
                return {"ok": False, "message": f"archive damaged at {bad}",
                        **info}
            zf.extractall(staging)
    except zipfile.BadZipFile as e:
        return {"ok": False, "message": f"not a valid archive: {e}", **info}

    # GitHub wraps everything in one folder named after the branch.
    entries = _os.listdir(staging)
    root = (_os.path.join(staging, entries[0])
            if len(entries) == 1 and
            _os.path.isdir(_os.path.join(staging, entries[0]))
            else staging)

    # Confirm this is actually LMCluster before letting it near anything.
    for required in ("lmcluster", "install.py", "version.txt"):
        if not _os.path.exists(_os.path.join(root, required)):
            return {"ok": False,
                    "message": f"what downloaded does not look like "
                               f"LMCluster — no {required} in it", **info}

    try:
        with open(_os.path.join(root, "version.txt"), encoding="utf-8") as f:
            downloaded = f.read().strip()
    except OSError:
        downloaded = None

    return {"ok": True, "message": f"version {downloaded} is ready to install",
            "staged_at": root, "downloaded_version": downloaded, **info}


def install_update(staged_root: str) -> dict:
    """Copy a downloaded update over the running installation.

    The previous version is kept in tmp/backup first, so that a botched
    update can be undone by copying it back rather than by reinstalling
    from scratch.
    """
    if not _os.path.isdir(staged_root):
        return {"ok": False, "message": "nothing downloaded to install"}

    backup = _os.path.join(TMP_DIR, "backup")
    _force_remove(backup)
    _os.makedirs(backup, exist_ok=True)

    replaced = []
    try:
        for name in _os.listdir(staged_root):
            if name in KEEP:
                continue
            src = _os.path.join(staged_root, name)
            dst = _os.path.join(_ROOT, name)

            if _os.path.exists(dst):
                keep = _os.path.join(backup, name)
                if _os.path.isdir(dst):
                    shutil.copytree(dst, keep, dirs_exist_ok=True,
                                    ignore=shutil.ignore_patterns(
                                        "__pycache__", "*.pyc"))
                else:
                    shutil.copy2(dst, keep)

            if _os.path.isdir(src):
                _force_remove(dst)
                shutil.copytree(src, dst,
                                ignore=shutil.ignore_patterns("__pycache__",
                                                              "*.pyc"))
            else:
                shutil.copy2(src, dst)
            replaced.append(name)
    except (OSError, shutil.Error) as e:
        return {"ok": False, "replaced": replaced,
                "message": f"update failed part way through: {e}. The "
                           f"previous version is in tmp/backup."}

    return {"ok": True, "replaced": replaced, "backup": backup,
            "message": f"replaced {len(replaced)} item(s)"}


def restart_soon(node=None, delay: float = 1.5):
    """Restart this node, after giving the browser its answer first.

    Replacing the running process is the only reliable way to pick up new
    Python files: the old ones are already loaded, and reimporting a live
    application does not work in practice. Any llama.cpp processes are
    stopped first, since replacing this process would otherwise leave them
    running with nothing to talk to.
    """
    def go():
        time.sleep(delay)
        if node is not None:
            try:
                node.master.stop()
                node.rpc_worker.stop()
            except Exception:
                pass
        python = sys.executable
        args = [python, "-m", "lmcluster"] + sys.argv[1:]
        try:
            _os.execv(python, args)
        except OSError:
            # execv failing leaves the process alive but running old code,
            # so the honest outcome is to stop and let whatever supervises
            # this start it again.
            _os._exit(0)

    threading.Thread(target=go, daemon=True).start()
