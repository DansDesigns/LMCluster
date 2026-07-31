"""Fetching prebuilt llama.cpp binaries.

Compiling llama.cpp is the slowest and by far the most fragile part of
setting this up. On Windows it means installing several gigabytes of Visual
Studio Build Tools; on Linux it means a working toolchain and twenty
minutes of waiting. None of that is necessary, because the llama.cpp
project publishes prebuilt binaries for every release, and those archives
contain `ggml-rpc-server`, which is the one program this whole project
depends on.

So the normal path is now to download rather than to build. Compiling is
still there for anyone who wants a build tuned to their own machine, or
whose platform has no published binary.

A word on the archive layout, because it dictates how this works. The
executables are small stubs that load their real implementation from a DLL
or shared object sitting beside them — on Windows `llama-server.exe` is
about nine kilobytes and `llama-server-impl.dll` is about ten megabytes.
Extracting two files out of the archive would therefore give you two
programs that cannot start. The whole folder is kept together, and the
recorded paths point into it.
"""

import hashlib
import io
import json
import os
import platform
import shutil
import stat
import subprocess
import urllib.error
import urllib.request
import zipfile

RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"

# What we need out of whichever archive we end up with. The RPC server has
# had two names across versions, so both are accepted.
RPC_NAMES = ("ggml-rpc-server", "rpc-server")
SERVER_NAME = "llama-server"


class DownloadError(RuntimeError):
    pass


def _platform_tokens(gpu_backend: str) -> tuple[list[list[str]], str]:
    """Substrings identifying a suitable archive, best choice first.

    Matching on substrings rather than reconstructing exact filenames,
    because the naming has changed before (build numbers, CUDA versions,
    the ubuntu/linux switch) and a rigid pattern would silently stop
    matching one day.
    """
    system, machine = platform.system(), platform.machine().lower()
    arm = machine in ("arm64", "aarch64")

    if system == "Windows":
        arch = ["arm64"] if arm else ["x64"]
        preferred = {
            "cuda": [["win", "cuda"] + arch],
            "rocm": [["win", "hip"] + arch, ["win", "rocm"] + arch],
            "vulkan": [["win", "vulkan"] + arch],
        }.get(gpu_backend, [])
        # CPU always last, and always present, so there is a fallback.
        return preferred + [["win", "cpu"] + arch], "Windows"

    if system == "Darwin":
        return ([["macos", "arm64"]] if arm else [["macos", "x64"]]), "macOS"

    arch = ["arm64"] if arm else ["x64"]
    preferred = {
        "cuda": [["ubuntu", "cuda"] + arch, ["linux", "cuda"] + arch],
        "vulkan": [["ubuntu", "vulkan"] + arch, ["linux", "vulkan"] + arch],
    }.get(gpu_backend, [])
    return (preferred
            + [["ubuntu"] + arch, ["linux"] + arch]), "Linux"


def latest_release(timeout: int = 30) -> dict:
    req = urllib.request.Request(
        RELEASES_API, headers={"Accept": "application/vnd.github+json",
                               "User-Agent": "LMCluster-installer"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 403:
            raise DownloadError(
                "GitHub is rate-limiting this network, which it does by IP "
                "address and without warning. Wait an hour, or build from "
                "source instead with --build-from-source.") from e
        raise DownloadError(f"could not reach GitHub: {e}") from e
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise DownloadError(f"could not reach GitHub: {e}") from e


def pick_asset(release: dict, gpu_backend: str) -> tuple[dict | None, str]:
    """Choose the archive that suits this machine."""
    assets = [a for a in release.get("assets", [])
              if a["name"].endswith((".zip", ".tar.gz"))
              and "cudart" not in a["name"]]
    patterns, os_label = _platform_tokens(gpu_backend)

    for tokens in patterns:
        for asset in assets:
            lowered = asset["name"].lower()
            if all(t in lowered for t in tokens):
                return asset, os_label
    return None, os_label


def cuda_runtime_asset(release: dict) -> dict | None:
    """The separate CUDA runtime archive, when a CUDA build was chosen.

    NVIDIA's runtime libraries are shipped alongside rather than inside the
    main archive, and without them a CUDA build refuses to start.
    """
    for asset in release.get("assets", []):
        if "cudart" in asset["name"].lower() and asset["name"].endswith(".zip"):
            return asset
    return None


def download(url: str, on_progress=None, timeout: int = 60,
             expect_size: int | None = None,
             expect_digest: str | None = None) -> bytes:
    """Fetch a URL, checking that what arrived is what was promised.

    Verification is done while the bytes stream past rather than in a
    separate pass afterwards, which costs nothing and means a truncated or
    substituted download is caught before anything is written to disk.

    GitHub gives a declared size for every release asset and, on newer API
    responses, a `digest` field of the form "sha256:...". The size is
    always checked. The digest is checked when offered, and its absence is
    not treated as a failure, because an older API response is not evidence
    of tampering.

    This idea is lifted from the NIGHTRUN project's installer, which
    verifies its model downloads by SHA-256 and then re-reads the media
    afterwards to confirm what it wrote. Its reasoning applies here: an
    installer that reports success without checking has told you nothing.
    """
    req = urllib.request.Request(
        url, headers={"User-Agent": "LMCluster-installer"})
    hasher = hashlib.sha256()
    chunks, got = [], 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            total = int(resp.headers.get("Content-Length") or 0) or expect_size
            while True:
                chunk = resp.read(256 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
                hasher.update(chunk)
                got += len(chunk)
                if on_progress:
                    on_progress(got, total or 0)
    except (urllib.error.URLError, OSError) as e:
        raise DownloadError(f"download failed: {e}") from e

    if expect_size and got != expect_size:
        raise DownloadError(
            f"download is the wrong size: expected {expect_size} bytes, got "
            f"{got}. The connection probably dropped part way through; try "
            f"again.")

    actual = hasher.hexdigest()
    if expect_digest:
        wanted = expect_digest.split(":", 1)[-1].strip().lower()
        if wanted and wanted != actual:
            raise DownloadError(
                "the downloaded file does not match the checksum GitHub "
                "published for it. Do not use it. Try again, and if it "
                "fails the same way, something between you and GitHub is "
                "altering the download.")

    return b"".join(chunks)


def extract(data: bytes, name: str, dest: str) -> str:
    """Unpack an archive into dest, flattening any single wrapper folder.

    Some archives put everything at the top level and some wrap it in one
    directory. Flattening that away means the rest of the code does not
    have to care which it got.
    """
    os.makedirs(dest, exist_ok=True)
    if name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(dest)
    else:
        import tarfile
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            tf.extractall(dest)

    entries = os.listdir(dest)
    if len(entries) == 1:
        inner = os.path.join(dest, entries[0])
        if os.path.isdir(inner):
            for item in os.listdir(inner):
                shutil.move(os.path.join(inner, item),
                            os.path.join(dest, item))
            os.rmdir(inner)
    return dest


def locate(folder: str) -> dict:
    """Find the programs we need, wherever in the folder they landed."""
    exe = ".exe" if os.name == "nt" else ""
    found = {}
    for root, _dirs, files in os.walk(folder):
        for filename in files:
            base = filename[:-4] if filename.endswith(".exe") else filename
            if base in RPC_NAMES and "rpc" not in found:
                found["rpc"] = os.path.join(root, filename)
            elif base == SERVER_NAME and "server" not in found:
                found["server"] = os.path.join(root, filename)
    # Extraction does not preserve the executable bit on every platform.
    for path in found.values():
        try:
            os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR
                     | stat.S_IXGRP | stat.S_IXOTH)
        except OSError:
            pass
    del exe
    return found


def _binary_names() -> set:
    """Programs that hold these files open while they run."""
    suffix = ".exe" if os.name == "nt" else ""
    return {n + suffix for n in RPC_NAMES} | {SERVER_NAME + suffix}


def running_llama_processes() -> list[str]:
    """Which of our programs are running right now.

    Worth knowing before touching the folder they live in. On Windows a
    file cannot be replaced while a process has it open — a DLL loaded by a
    running RPC server is locked, and any attempt to overwrite it fails
    with a permission error that says nothing about the real cause.
    """
    names = _binary_names()
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/fo", "csv", "/nh"],
                                 capture_output=True, text=True, timeout=20,
                                 check=False).stdout
            found = set()
            for line in out.splitlines():
                first = line.split(",")[0].strip('"').strip()
                if first in names:
                    found.add(first)
            return sorted(found)
        out = subprocess.run(["ps", "-eo", "comm"], capture_output=True,
                             text=True, timeout=20, check=False).stdout
        return sorted({line.strip() for line in out.splitlines()
                       if line.strip() in names})
    except (OSError, subprocess.SubprocessError):
        return []


def _force_remove(path: str):
    """Delete a tree, dealing with read-only files, and complain if it can't.

    The version this replaces passed ignore_errors=True, which meant a file
    that could not be deleted was skipped in silence. The extraction that
    followed then tried to overwrite that same file and failed — so the
    error surfaced several steps away from its cause, wearing the wrong
    name. Anything that cannot be removed needs saying out loud.
    """
    if not os.path.isdir(path):
        return

    def on_error(func, target, exc_info):
        # Commonly just a read-only flag, which is trivially fixable.
        try:
            os.chmod(target, stat.S_IWRITE)
            func(target)
        except OSError:
            pass  # collected and reported below

    shutil.rmtree(path, onerror=on_error)

    if os.path.isdir(path):
        stuck = []
        for root, _dirs, files in os.walk(path):
            for name in files:
                stuck.append(os.path.join(root, name))
        raise DownloadError(
            "could not clear the old llama.cpp folder. "
            + (f"{len(stuck)} file(s) are still there, including "
               f"{os.path.basename(stuck[0])}. " if stuck else "")
            + "On Windows this means a program still has them open.")


def fetch(dest: str, gpu_backend: str = "cpu", say=print) -> dict:
    """Download and unpack llama.cpp, returning the paths we care about.

    The unpacking is done into a temporary folder beside the destination
    and only swapped into place once everything needed has been confirmed
    present. Extracting straight over a working installation, which is what
    this used to do, means any failure part way through leaves you with
    neither the old version nor the new one.

    Raises DownloadError with something explanatory if any step fails, so
    the caller can offer to build from source instead.
    """
    # Checked before downloading rather than after, because finding out
    # that the folder is locked is not worth thirty megabytes of waiting.
    busy = running_llama_processes()
    if busy and os.path.isdir(dest):
        raise DownloadError(
            f"{' and '.join(busy)} {'is' if len(busy) == 1 else 'are'} "
            f"running, and llama.cpp's files cannot be replaced while they "
            f"are open.\n"
            f"    Stop the LMCluster node on this machine — close its window, "
            f"or press Leave the pool on the dashboard — and run this again. "
            f"The existing installation has not been touched.")

    say("  asking GitHub for the latest llama.cpp release...")
    release = latest_release()
    tag = release.get("tag_name", "?")

    asset, os_label = pick_asset(release, gpu_backend)
    if asset is None:
        raise DownloadError(
            f"no prebuilt archive published for {os_label} with a "
            f"{gpu_backend} backend in release {tag}")

    size_mb = asset.get("size", 0) / 1e6
    say(f"  {tag}: {asset['name']} ({size_mb:.0f} MB)")

    state = {"last": -1}

    def progress(got, total):
        if not total:
            return
        pct = int(got * 100 / total)
        if pct >= state["last"] + 10:
            state["last"] = pct
            say(f"    {pct}%")

    data = download(asset["browser_download_url"], on_progress=progress,
                    expect_size=asset.get("size"),
                    expect_digest=asset.get("digest"))
    digest = hashlib.sha256(data).hexdigest()
    say(f"    sha256 {digest[:16]}…"
        + ("  (matches GitHub)" if asset.get("digest") else
           "  (GitHub published no checksum to compare against)"))

    # A valid archive is its own integrity check: a corrupted download
    # fails to open long before anything is written anywhere.
    try:
        if asset["name"].endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                bad = zf.testzip()
            if bad:
                raise DownloadError(f"the archive is damaged at {bad}")
    except zipfile.BadZipFile as e:
        raise DownloadError(
            f"what arrived is not a valid zip file ({e}). GitHub may have "
            f"served an error page instead of the download.") from e

    staging = dest + ".new"
    _force_remove(staging)
    try:
        extract(data, asset["name"], staging)

        # CUDA builds need NVIDIA's runtime libraries, published as a
        # separate archive. Without them the binaries will not start.
        if "cuda" in asset["name"].lower():
            runtime = cuda_runtime_asset(release)
            if runtime:
                say(f"  and the CUDA runtime "
                    f"({runtime.get('size', 0) / 1e6:.0f} MB)")
                try:
                    extract(download(runtime["browser_download_url"],
                                     expect_size=runtime.get("size"),
                                     expect_digest=runtime.get("digest")),
                            runtime["name"], staging)
                except DownloadError as e:
                    say(f"  warning: could not fetch the CUDA runtime "
                        f"({e}). The binaries may not start.")

        found = locate(staging)
        if "rpc" not in found:
            raise DownloadError(
                f"{asset['name']} unpacked, but there is no RPC server in "
                f"it. This release may not ship one for your platform; "
                f"build from source instead with --build-from-source.")
        if "server" not in found:
            raise DownloadError(
                f"{asset['name']} unpacked, but there is no llama-server "
                f"in it.")

        # Everything needed is confirmed present, so the old version can go.
        try:
            _force_remove(dest)
        except DownloadError as e:
            busy = running_llama_processes()
            raise DownloadError(
                f"{e}\n"
                + (f"    {' and '.join(busy)} started while this was "
                   f"downloading. Stop it and run this again.\n"
                   if busy else
                   "    Close anything using llama.cpp on this machine, or "
                   "check whether antivirus software is holding the "
                   "folder.\n")
                + f"    The new version is waiting in {staging} — your "
                  f"working installation is untouched.") from e

        os.replace(staging, dest)
    except Exception:
        # Leave the staging folder alone on a locking failure, since it
        # holds a good copy someone may want to move by hand. Clear it for
        # anything else, so a half-unpacked archive is not mistaken for a
        # usable one.
        if not os.path.isdir(dest):
            _force_remove(staging)
        raise

    found = locate(dest)
    found["release"] = tag
    found["asset"] = asset["name"]
    found["sha256"] = digest
    return found
