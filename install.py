#!/usr/bin/env python3
"""LMCluster installer.

Sets up one node. Run it on every machine that will join the cluster.

  python3 install.py                       interactive
  python3 install.py --yes                 accept defaults, no prompts
  python3 install.py --with-rpc            fetch llama.cpp without asking
  python3 install.py --build-from-source   compile it instead of downloading
  python3 install.py --token abc123...     join an existing cluster
  python3 install.py --with-rpc --gpu cuda --jobs 4
  python3 install.py --new-token           issue a fresh cluster key

On Windows use `python` rather than `python3`.

Two things get installed:

  1. A local virtual environment with the Python dependencies.
  2. llama.cpp, including the RPC server that makes a cluster possible.

The second is downloaded from the llama.cpp project by default, which takes
a minute or two and needs no compiler. Pass --build-from-source to compile
it yourself instead. A machine without it can still run the dashboard and
watch the cluster, but it cannot lend its memory to a model or load one.
"""

import argparse
import json
import re
import os
import platform
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(ROOT, "vendor")
LLAMA_SRC = os.path.join(VENDOR, "llama.cpp")
LLAMA_BUILD = os.path.join(LLAMA_SRC, "build")
LLAMA_BIN = os.path.join(VENDOR, "llama.cpp-bin")   # downloaded, not built
LLAMA_REPO = "https://github.com/ggml-org/llama.cpp"

IS_WINDOWS = os.name == "nt"
EXE = ".exe" if IS_WINDOWS else ""

# The two programs we need out of llama.cpp.
#
# The RPC server has been called two different things. Older llama.cpp built
# `rpc-server`; current versions build `ggml-rpc-server`. Looking for only
# one of those is how this installer came to report a successful build and
# then insist the binary did not exist, so both names are accepted
# everywhere: as a cmake target, and as a file on disk.
RPC_NAMES = ("ggml-rpc-server", "rpc-server")
SERVER_NAME = "llama-server"

# Where CMake leaves things, which varies by generator: Ninja and Make put
# binaries in bin/, while the Visual Studio generator adds a config folder.
BIN_DIRS = ("bin", os.path.join("bin", "Release"),
            os.path.join("bin", "RelWithDebInfo"), "Release", ".")


class Abort(RuntimeError):
    pass


class NeedsRestart(RuntimeError):
    """Tools were installed, but this shell cannot see them yet.

    Worth distinguishing from a plain failure: nothing is wrong, the
    person just has to open a new terminal, and telling them that is far
    more useful than repeating the whole list of ways to install a
    compiler they have in fact just installed.
    """


# -- console --------------------------------------------------------------

PY = "python" if IS_WINDOWS else "python3"


def say(msg=""):
    print(msg, flush=True)


def step(n, total, msg):
    say(f"\n[{n}/{total}] {msg}")
    say("-" * 60)


def ask(question, default=True, assume_yes=False) -> bool:
    if assume_yes:
        say(f"{question} [auto: yes]")
        return True
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        answer = input(f"{question} {suffix}: ").strip().lower()
    except EOFError:
        return default
    if not answer:
        return default
    return answer in ("y", "yes")


def run(cmd, cwd=None, check=True, quiet=False) -> int:
    printable = cmd if isinstance(cmd, str) else " ".join(cmd)
    say(f"  $ {printable}")
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, shell=isinstance(cmd, str),
            stdout=subprocess.DEVNULL if quiet else None,
            stderr=subprocess.STDOUT if quiet else None)
    except OSError as e:
        if check:
            raise Abort(f"could not run {printable}: {e}") from e
        return 1
    if check and proc.returncode != 0:
        raise Abort(f"command failed ({proc.returncode}): {printable}")
    return proc.returncode


def have(binary) -> bool:
    return shutil.which(binary) is not None


# -- step 1: python environment -------------------------------------------

def venv_python() -> str:
    return os.path.join(ROOT, ".venv",
                        "Scripts" if IS_WINDOWS else "bin",
                        "python" + EXE)


def install_python_env(args):
    if sys.version_info < (3, 10):
        raise Abort(f"Python 3.10+ required, found {platform.python_version()}")

    venv_dir = os.path.join(ROOT, ".venv")
    if os.path.isdir(venv_dir) and os.path.exists(venv_python()):
        say("  venv already present, reusing it")
    else:
        try:
            import venv  # noqa: F401
        except ImportError:
            raise Abort("python3-venv is missing. On Debian/Devuan/Ubuntu: "
                        "sudo apt install python3-venv")
        run([sys.executable, "-m", "venv", venv_dir])

    py = venv_python()
    run([py, "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
    req = os.path.join(ROOT, "requirements.txt")
    run([py, "-m", "pip", "install", "--quiet", "-r", req])
    say("  ✓ dependencies installed")


# -- step 3: llama.cpp with RPC -------------------------------------------

def detect_gpu_backend() -> str:
    """auto -> the cmake flag we should add alongside -DGGML_RPC=ON."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "metal"
    if have("nvidia-smi"):
        return "cuda"
    if have("hipcc") or have("rocm-smi"):
        return "rocm"
    if have("vulkaninfo"):
        return "vulkan"
    return "cpu"


GPU_CMAKE_FLAGS = {
    "cuda": ["-DGGML_CUDA=ON"],
    "rocm": ["-DGGML_HIP=ON"],
    "vulkan": ["-DGGML_VULKAN=ON"],
    "metal": ["-DGGML_METAL=ON"],
    "cpu": [],
    "none": [],
}


# What each tool is called in the package managers we can drive.
TOOL_PACKAGES = {
    "cmake": {"winget": "Kitware.CMake", "brew": "cmake",
              "apt": "cmake", "dnf": "cmake"},
    "git": {"winget": "Git.Git", "brew": "git",
            "apt": "git", "dnf": "git"},
    # Deliberately not listed on Windows: see offer_build_tools.
    "a C++ compiler (gcc or clang)": {"apt": "build-essential",
                                      "dnf": "gcc-c++",
                                      "brew": "gcc"},
}


def find_windows_compiler() -> tuple[str | None, list[str]]:
    """What C++ compiler is available on Windows, and how to drive it.

    Returns (description, extra cmake arguments), or (None, []) if there is
    no compiler at all.

    This looks in several places because no single check is reliable.
    vswhere is the official way to find Visual Studio, but it only reports
    an installation once the C++ workload has finished installing, which
    can be twenty minutes after the installer said "successfully
    installed". So a filesystem check for the toolchain itself is done as
    well, and cl.exe on PATH is accepted, and MinGW or clang from MSYS2 are
    accepted too.
    """
    # 1. Visual Studio, located the way Microsoft intends.
    for base in (os.environ.get("ProgramFiles(x86)"),
                 os.environ.get("ProgramFiles")):
        if not base:
            continue
        vswhere = os.path.join(base, "Microsoft Visual Studio", "Installer",
                               "vswhere.exe")
        if not os.path.exists(vswhere):
            continue
        out = _run_capture([
            vswhere, "-latest", "-products", "*",
            "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property", "displayName"])
        if out and out.strip():
            # CMake finds Visual Studio by itself once it is installed, so
            # there is no generator to force here.
            return f"Visual Studio ({out.strip().splitlines()[0]})", []

    # 2. The toolchain on disk. vswhere can be absent or can under-report
    #    while an install is still finishing, but the compiler either
    #    exists or it does not.
    for base in (os.environ.get("ProgramFiles(x86)"),
                 os.environ.get("ProgramFiles")):
        if not base:
            continue
        root = os.path.join(base, "Microsoft Visual Studio")
        if not os.path.isdir(root):
            continue
        for year in sorted(os.listdir(root), reverse=True):
            for edition in ("BuildTools", "Community", "Professional",
                            "Enterprise"):
                msvc = os.path.join(root, year, edition, "VC", "Tools", "MSVC")
                if os.path.isdir(msvc) and os.listdir(msvc):
                    return f"Visual Studio {year} {edition}", []

    # 3. Already in a developer command prompt.
    if have("cl"):
        return "MSVC (cl.exe is on PATH)", []

    # 4. MSYS2 or a standalone MinGW. These need the generator naming,
    #    because CMake otherwise still reaches for NMake.
    for name, label in (("gcc", "MinGW gcc"), ("clang", "clang")):
        if have(name):
            gen = ["-G", "Ninja"] if have("ninja") else ["-G", "MinGW Makefiles"]
            return f"{label} (using {gen[1]})", gen

    return None, []


def _run_capture(cmd, timeout=20) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def check_build_tools() -> list[str]:
    missing = []
    if not have("git"):
        missing.append("git")
    if not have("cmake"):
        missing.append("cmake")
    if IS_WINDOWS:
        compiler, _ = find_windows_compiler()
        if compiler is None:
            missing.append("a C++ compiler")
    elif not (have("cc") or have("gcc") or have("clang")):
        missing.append("a C++ compiler (gcc or clang)")
    return missing


def _package_manager() -> str | None:
    if IS_WINDOWS:
        return "winget" if have("winget") else None
    if platform.system() == "Darwin":
        return "brew" if have("brew") else None
    if have("apt-get"):
        return "apt"
    if have("dnf"):
        return "dnf"
    return None


def offer_build_tools(missing: list[str], args) -> list[str]:
    """Try to install the missing toolchain, then re-check.

    Refusing to help with a missing cmake, when a package manager is
    sitting right there that could install it, just sends the person off to
    do by hand what we could have done for them. The C++ compiler is the
    exception: on Windows that means Visual Studio Build Tools, which is a
    multi-gigabyte interactive installer nobody should launch on someone
    else's behalf.
    """
    manager = _package_manager()
    installable = [m for m in missing
                   if m in TOOL_PACKAGES
                   and manager in TOOL_PACKAGES.get(m, {})]
    manual = [m for m in missing if m not in installable]

    if "a C++ compiler" in manual and manager == "winget":
        say("")
        say("  A C++ compiler on Windows means Visual Studio Build Tools.")
        say("  It is a large download, several gigabytes, and it installs")
        say("  system-wide, so I would rather ask than assume.")
        if ask("  Install it now with winget?", default=False,
               assume_yes=False):
            say("")
            say("  This will take a while. winget downloads a small")
            say("  bootstrapper first, then Visual Studio downloads the")
            say("  actual compiler, which is the slow part. Leave it be.")
            # --wait is what makes the Visual Studio bootstrapper block
            # until the real install has finished. Without it the
            # bootstrapper exits after a few seconds, winget cheerfully
            # reports success, and the compiler is still downloading in the
            # background — which is exactly why this step used to say
            # "successfully installed" and then "still missing" one line
            # apart.
            run(["winget", "install",
                 "Microsoft.VisualStudio.2022.BuildTools", "-e",
                 "--accept-package-agreements", "--accept-source-agreements",
                 "--override",
                 "--quiet --wait --norestart "
                 "--add Microsoft.VisualStudio.Workload.NativeDesktop"
                 " --includeRecommended"], check=False)

            compiler, _ = find_windows_compiler()
            if compiler:
                say(f"  ✓ found it: {compiler}")
                return [m for m in missing if m != "a C++ compiler"]
            raise NeedsRestart(
                "Visual Studio Build Tools has been installed, but this "
                "window was started before it existed and cannot see it.")

    if not installable:
        return missing

    if manager is None:
        say(f"  no package manager available to install "
            f"{', '.join(installable)} automatically")
        return missing

    say(f"  {', '.join(installable)} can be installed with {manager}.")
    if not ask(f"  Install {', '.join(installable)} now?", default=True,
               assume_yes=args.yes):
        return missing

    for tool in installable:
        pkg = TOOL_PACKAGES[tool][manager]
        say(f"  installing {tool}...")
        if manager == "winget":
            run(["winget", "install", "--id", pkg, "-e", "--silent",
                 "--accept-package-agreements", "--accept-source-agreements"],
                check=False)
        elif manager == "brew":
            run(["brew", "install", pkg], check=False)
        elif manager == "apt":
            run(["sudo", "apt-get", "install", "-y", pkg], check=False)
        elif manager == "dnf":
            run(["sudo", "dnf", "install", "-y", pkg], check=False)

    still = check_build_tools()
    installed = [m for m in installable if m not in still]
    if installed:
        say(f"  ✓ installed {', '.join(installed)}")
    if any(t in still for t in installable) and manager == "winget":
        # winget puts new tools on the machine PATH, not this process's.
        say("  ⚠ newly installed tools are not on this shell's PATH yet.")
        say("    Close this window, open a new one, and re-run:")
        say(f"      {PY} install.py --with-rpc")
    if manual:
        say(f"  ✗ still need {', '.join(manual)} — this one is manual:")
        say(build_hint())
    return still


def build_hint() -> str:
    system = platform.system()
    if system == "Linux":
        return ("  Debian, Devuan or Ubuntu:  sudo apt install "
                "build-essential cmake git\n"
                "  Fedora:                    sudo dnf install gcc-c++ "
                "cmake git")
    if system == "Darwin":
        return ("  xcode-select --install\n"
                "  brew install cmake git")
    return (
        "  You need a C++ compiler. There are two ways to get one:\n"
        "\n"
        "  1. Visual Studio Build Tools, which is the official route but a\n"
        "     large download. Either install it from\n"
        "     https://visualstudio.microsoft.com/downloads/ picking the\n"
        "     'Desktop development with C++' workload, or run:\n"
        "       winget install Microsoft.VisualStudio.2022.BuildTools "
        "--override \"--quiet --add "
        "Microsoft.VisualStudio.Workload.NativeDesktop\"\n"
        "\n"
        "  2. MSYS2, which is smaller. Install it from https://msys2.org,\n"
        "     then in the UCRT64 shell:\n"
        "       pacman -S mingw-w64-ucrt-x86_64-gcc "
        "mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja git\n"
        "\n"
        "  Either way, open a new terminal afterwards so the new tools are\n"
        "  on your PATH, then re-run this installer.")


def find_binaries(build_dir) -> dict:
    """Locate what the build produced.

    Returns a dict with the keys "rpc" and "server" when found. The RPC
    server is searched for under both of its historical names, and as a
    last resort by walking the build tree, because a generator we have not
    anticipated putting it somewhere unexpected should not look like a
    failed build.
    """
    found = {}

    def look(names, key):
        for name in names:
            for sub in BIN_DIRS:
                candidate = os.path.join(build_dir, sub, name + EXE)
                if os.path.exists(candidate):
                    found[key] = os.path.abspath(candidate)
                    return True
        return False

    look(RPC_NAMES, "rpc")
    look((SERVER_NAME,), "server")

    if "rpc" not in found or "server" not in found:
        wanted = {n + EXE for n in RPC_NAMES} | {SERVER_NAME + EXE}
        for root, _dirs, files in os.walk(build_dir):
            for filename in files:
                if filename not in wanted:
                    continue
                key = "server" if filename.startswith(SERVER_NAME) else "rpc"
                found.setdefault(key, os.path.abspath(
                    os.path.join(root, filename)))
    return found


def get_llamacpp(args) -> dict:
    """Obtain llama.cpp, downloading it unless told to compile.

    Downloading is the default because compiling is the slowest and most
    fragile step in the whole setup, and because the llama.cpp project
    already publishes binaries containing exactly the two programs needed
    here. Building from source remains available for anyone who wants a
    binary tuned to their own machine, or whose platform has no published
    build.
    """
    if args.build_from_source:
        return build_llamacpp(args)

    backend = args.gpu if args.gpu != "auto" else detect_gpu_backend()
    say(f"  hardware backend: {backend}"
        + (" (detected)" if args.gpu == "auto" else " (you chose this)"))

    sys.path.insert(0, ROOT)
    try:
        from lmcluster import prebuilt
    except ImportError as e:
        say(f"  could not load the downloader ({e}); building instead")
        return build_llamacpp(args)

    # Say this up front rather than letting it surface as a permission
    # error halfway through unpacking. A node that is lending its memory
    # has llama.cpp's libraries open, and on Windows an open file cannot
    # be replaced.
    busy = prebuilt.running_llama_processes()
    if busy and os.path.isdir(LLAMA_BIN):
        say(f"  ⚠ {' and '.join(busy)} "
            f"{'is' if len(busy) == 1 else 'are'} still running.")
        say("    llama.cpp cannot be replaced while its files are open.")
        say("    Stop the LMCluster node on this machine first — close its")
        say("    window, or press Leave the pool on its dashboard.")
        say("")
        say("    If you carry on, llama.cpp will NOT be replaced — you will")
        say("    keep whatever build is there now.")
        if not ask("  Try anyway?", default=False, assume_yes=False):
            raise Abort(
                "stopped so you can shut the node down first. Nothing has "
                "been changed; run this again once it is closed.")

    try:
        found = prebuilt.fetch(LLAMA_BIN, gpu_backend=backend, say=say)
    except prebuilt.DownloadError as e:
        say(f"  ✗ {e}")
        say("")
        # A locked folder is not a reason to spend half an hour compiling:
        # the download worked, and it will work again once whatever is
        # holding the files has been stopped.
        if "cannot be replaced" in str(e) or "still has them open" in str(e):
            raise Abort("stop the running node and try again")
        if ask("  Build it from source instead? This needs a compiler and "
               "takes a while.", default=True, assume_yes=args.yes):
            return build_llamacpp(args)
        raise Abort("no llama.cpp, and you declined to build it")

    say(f"  ✓ llama.cpp {found['release']} ready, no compiler needed")
    say(f"    RPC server:   {found['rpc']}")
    say(f"    llama-server: {found['server']}")

    # Confirm what actually arrived rather than trusting the archive name.
    # Asking for a GPU build and quietly ending up with a CPU one is the
    # kind of thing that is only noticed days later, when the machine keeps
    # telling everyone else it has no graphics support.
    from lmcluster import rpc as rpc_mod
    got = rpc_mod.build_backends(found["rpc"])
    accel = [b for b in got if b != "cpu"]
    if backend in ("cpu", "none"):
        say("    this is a CPU build, as asked")
    elif backend in got:
        say(f"    confirmed: {rpc_mod.describe_build(found['rpc'])}")
    else:
        say(f"  ⚠ you asked for {backend}, but what downloaded is a "
            f"{rpc_mod.describe_build(found['rpc']).lower()}"
            + (f" ({', '.join(accel)})" if accel else "") + ".")
        say(f"    There may be no {backend} build published for this "
            f"platform in that release.")
    return {"rpc": found["rpc"], "server": found["server"]}


def build_llamacpp(args) -> dict:
    """Clone and build llama.cpp with the RPC backend enabled.

    -DGGML_RPC=ON is the whole point: without it llama.cpp has no
    rpc-server binary and --rpc is not a recognised flag, so shard mode
    cannot work at all.
    """
    missing = check_build_tools()
    if missing:
        say(f"  missing build tools: {', '.join(missing)}")
        missing = offer_build_tools(missing, args)
    if missing:
        say(f"  ✗ still missing: {', '.join(missing)}")
        say("")
        say(build_hint())
        # Checked before cloning on purpose: there is no sense downloading
        # a few hundred megabytes of source that cannot be compiled.
        raise Abort("cannot build llama.cpp without a compiler")

    generator = []
    if IS_WINDOWS:
        compiler, generator = find_windows_compiler()
        say(f"  compiler: {compiler}")

    os.makedirs(VENDOR, exist_ok=True)

    if os.path.isdir(os.path.join(LLAMA_SRC, ".git")):
        say("  llama.cpp already cloned")
        if ask("  Pull latest before building?", default=False,
               assume_yes=False if not args.yes else True):
            run(["git", "-C", LLAMA_SRC, "pull", "--ff-only"], check=False)
    else:
        run(["git", "clone", "--depth", "1", LLAMA_REPO, LLAMA_SRC])

    backend = args.gpu if args.gpu != "auto" else detect_gpu_backend()
    say(f"  GPU backend: {backend}"
        + (" (auto-detected)" if args.gpu == "auto" else " (forced)"))

    cmake_cfg = [
        "cmake", "-B", LLAMA_BUILD, "-S", LLAMA_SRC,
        "-DGGML_RPC=ON",            # everything here depends on this
        "-DLLAMA_CURL=OFF",         # avoids needing libcurl headers
        "-DCMAKE_BUILD_TYPE=Release",
    ] + generator + GPU_CMAKE_FLAGS.get(backend, [])

    say("  configuring, which is where the RPC backend gets switched on")
    rc = run(cmake_cfg, check=False)
    if rc != 0:
        # A stale build folder from a previous failed attempt keeps its
        # broken cache and will fail identically forever, so it is worth
        # clearing once before giving up.
        if os.path.isdir(LLAMA_BUILD):
            say("  configure failed; clearing the build folder and retrying")
            shutil.rmtree(LLAMA_BUILD, ignore_errors=True)
            rc = run(cmake_cfg, check=False)
    if rc != 0:
        raise Abort(
            "cmake could not set up the build. If it complained about "
            "nmake, or about CMAKE_C_COMPILER not being set, the real "
            "problem is that there is no C++ compiler installed.")

    jobs = args.jobs or (os.cpu_count() or 4)
    say(f"  building with {jobs} jobs — this takes a while, go and make tea")

    # Try the current target name, then the old one. Naming a target that
    # does not exist is a hard error in cmake, so this cannot be a single
    # combined command.
    built = False
    for rpc_target in RPC_NAMES:
        rc = run(["cmake", "--build", LLAMA_BUILD, "--config", "Release",
                  "-j", str(jobs), "--target", rpc_target, SERVER_NAME],
                 check=False)
        if rc == 0:
            built = True
            break
        say(f"  (no target called {rpc_target} in this version, trying "
            f"another name)")

    if not built:
        say("  building everything instead, which takes longer")
        run(["cmake", "--build", LLAMA_BUILD, "--config", "Release",
             "-j", str(jobs)])

    found = find_binaries(LLAMA_BUILD)
    if found.get("rpc"):
        say(f"  ✓ RPC server: {found['rpc']}")
    else:
        say("  ✗ no RPC server binary was produced")
    if found.get("server"):
        say(f"  ✓ llama-server: {found['server']}")
    else:
        say("  ✗ no llama-server binary was produced")

    if not found.get("rpc"):
        raise Abort(
            "the build finished but produced no RPC server. That usually "
            "means -DGGML_RPC=ON was not honoured; look for a line about "
            "the RPC backend in the cmake output above.")
    if not found.get("server"):
        raise Abort("the build finished but produced no llama-server.")
    return found


# -- step: where the models are ------------------------------------------

def choose_model_dir(args) -> str:
    """Find where GGUF files already are, and confirm it.

    Leaving this blank, as the installer used to, meant the dashboard
    greeted you with an empty list and a box to type a path into. Most
    people already have models somewhere and there are only a handful of
    likely places, so it is better to look first and ask second.
    """
    if args.model_dir:
        path = os.path.abspath(os.path.expanduser(args.model_dir))
        say(f"  using {path} (given on the command line)")
        return path

    sys.path.insert(0, ROOT)
    try:
        from lmcluster import models as finder
    except ImportError as e:
        say(f"  could not search for models ({e}); set the folder later in "
            f"the dashboard")
        return ""

    found = finder.candidates(ROOT)
    with_models = [c for c in found if c["exists"] and c["count"]]

    say("  Looked in the usual places:")
    for c in found:
        if c["exists"] and c["count"]:
            note = f"{c['count']} model(s)"
        elif c["exists"]:
            note = "exists, empty"
        else:
            note = "not there"
        say(f"    {c['label']:30} {note:16} {c['path']}")

    if not with_models:
        say("")
        say("  No model files anywhere yet, which is fine — this is the")
        say("  folder LMCluster will watch, so you can point it at where")
        say("  you intend to put them.")
        suggested = found[0]["path"] if found else ""
    else:
        suggested = with_models[0]["path"]
        say("")
        say(f"  Found {with_models[0]['count']} model(s) in "
            f"{with_models[0]['label']}.")

    if args.yes:
        say(f"  using {suggested}")
        return suggested

    say("")
    say(f"  Press enter to use: {suggested}")
    say("  Or type a different folder.")
    try:
        answer = input("  Model folder: ").strip()
    except EOFError:
        answer = ""
    chosen = os.path.abspath(os.path.expanduser(answer)) if answer else suggested

    if chosen and not os.path.isdir(chosen):
        if ask(f"  {chosen} does not exist. Create it?", default=True,
               assume_yes=args.yes):
            try:
                os.makedirs(chosen, exist_ok=True)
                say(f"  ✓ created {chosen}")
            except OSError as e:
                say(f"  ✗ could not create it: {e}")
                return ""
    return chosen


# -- step: firewall -------------------------------------------------------

def open_firewall(args):
    """Offer to open the ports the cluster needs.

    Done during installation because this is the moment somebody is already
    sitting at the machine expecting to be asked things. Discovering later
    that a machine is invisible, and tracing that back to a firewall, is a
    much worse afternoon.
    """
    sys.path.insert(0, ROOT)
    try:
        from lmcluster import firewall
        from lmcluster.config import Config
    except ImportError as e:
        say(f"  could not check the firewall ({e}); skipping")
        return

    cfg = Config(os.path.join(ROOT, "lmcluster.toml"))
    info = firewall.detect()

    if info["kind"] == "none":
        say("  no firewall found, so nothing is being blocked")
        return
    if info.get("inactive"):
        say(f"  {info['name']} is not switched on, so nothing is blocked")
        return
    if not info["manageable"]:
        say(f"  {info['name']} cannot be set up automatically.")
        for cmd in firewall.commands(cfg):
            say(f"    {cmd}")
        return

    state = firewall.status(cfg)
    if state.get("all_open"):
        say(f"  {info['name']}: the ports are already open")
        return

    say(f"  {info['name']} is active. Three ports need to be reachable:")
    for label, port, proto in firewall.ports(cfg):
        say(f"    {port}/{proto:3}  {label}")
    say("")
    say("  Without these, machines either cannot see each other or can see")
    say("  each other and then fail to share memory, which is a confusing")
    say("  thing to debug later.")

    if not ask("  Open them now?", default=True, assume_yes=args.yes):
        say("  skipped. You can do it later from the dashboard, under")
        say("  Settings, or run these by hand:")
        for cmd in firewall.commands(cfg):
            say(f"    {cmd}")
        return

    result = firewall.apply(cfg)
    say(f"  {'✓' if result['ok'] else '✗'} {result['message']}")
    if not result["ok"] and result.get("commands"):
        say("  Run these by hand instead:")
        for cmd in result["commands"]:
            say(f"    {cmd}")


# -- config ---------------------------------------------------------------

def state_dir() -> str:
    if IS_WINDOWS:
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
    else:
        base = os.environ.get("XDG_CONFIG_HOME",
                              os.path.expanduser("~/.config"))
    path = os.path.join(base, "lmcluster")
    os.makedirs(path, exist_ok=True)
    return path


def setup_token(args) -> str:
    path = os.path.join(state_dir(), "cluster_token")

    if args.new_token:
        import secrets
        token = secrets.token_hex(16)
        with open(path, "w") as f:
            f.write(token)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        say("  ✓ new cluster token generated (--new-token)")
        say("  ⚠ every other node is now on the old token and has left the")
        say("    cluster. Re-run their installer with --token <new token>,")
        say("    or rotate from the dashboard instead, which pushes the new")
        say("    key to online nodes for you.")
        return token

    if args.token:
        with open(path, "w") as f:
            f.write(args.token.strip())
        say("  ✓ joined existing cluster using the token you supplied")
        return args.token.strip()

    if os.path.exists(path):
        with open(path) as f:
            existing = f.read().strip()
        if existing:
            say("  ✓ existing cluster token found, keeping it")
            return existing

    if not args.yes:
        say("  If this node is JOINING an existing cluster, paste that")
        say("  cluster's token now. Leave blank to start a NEW cluster.")
        try:
            supplied = input("  Cluster token: ").strip()
        except EOFError:
            supplied = ""
        if supplied:
            with open(path, "w") as f:
                f.write(supplied)
            say("  ✓ joined existing cluster")
            return supplied

    import secrets
    token = secrets.token_hex(16)
    with open(path, "w") as f:
        f.write(token)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    say("  ✓ new cluster token generated")
    return token


def _existing_tables(text: str) -> set:
    """Top-level table names already present in a TOML file."""
    return {m.group(1) for m in re.finditer(r"^\s*\[([^\[\]]+)\]", text,
                                           re.M)}


def _trailing_comment(line: str) -> str:
    """The comment at the end of a config line, if there really is one.

    Finding the first hash and calling the rest a comment is wrong, because
    a hash inside a quoted value is just a character — Windows paths and
    llama.cpp argument strings can both contain one. Doing that turned

        extra_args = "--file C:\\models\\tmpl#1.jinja"

    into a mangled line with a stray quote in it. So this walks the line
    and only treats a hash as a comment when it is outside quotes.
    """
    in_quotes = False
    escaped = False
    for i, ch in enumerate(line):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
        elif ch == '"':
            in_quotes = not in_quotes
        elif ch == "#" and not in_quotes:
            return "  " + line[i:].rstrip()
    return ""


def _set_key(text: str, table: str, key: str, value: str) -> tuple[str, bool]:
    """Rewrite `key = ...` inside `[table]`, leaving the rest byte-identical.

    Surgical rather than a full re-emit, because the config carries the
    explanatory comments the installer wrote and re-emitting from a parsed
    dict would silently throw them away.
    """
    lines = text.splitlines(keepends=True)
    in_table = False
    for i, line in enumerate(lines):
        header = re.match(r"^\s*\[([^\[\]]+)\]", line)
        if header:
            in_table = header.group(1) == table
            continue
        if in_table and re.match(rf"^\s*{re.escape(key)}\s*=", line):
            lines[i] = f"{key} = {value}{_trailing_comment(line)}\n"
            return "".join(lines), True
    return text, False


SHARD_BLOCK = """
# Where llama.cpp lives, and how this machine takes part in the cluster.
#
# rpc_port is the port this machine listens on when it is lending its
# memory to a model loaded elsewhere. master_port is the port llama.cpp
# serves the model on when this is the machine that loaded it. They only
# need changing if something else is already using them.
[shard]
enabled = {enabled}
rpc_port = 50052
master_port = 8080
# Lend this machine's memory to the cluster as soon as it starts up,
# rather than waiting for someone to press the button on the dashboard.
auto_start_worker = {auto}
rpc_server = "{rpc_server}"
llama_server = "{llama_server}"
# The folder holding your .gguf model files. Leave it blank and LMCluster
# will search the usual places, including llama.cpp's own download cache.
model_dir = "{model_dir}"
"""

CLUSTER_BLOCK = """
# Machines must present the shared cluster key to ask anything of this one.
# Turning this off means anybody on your network can use this machine.
[cluster]
require_token = true
"""


def write_config(binaries: dict, model_dir: str = ""):
    """Create or update lmcluster.toml.

    An earlier version refused to touch an existing config and printed a
    block for the operator to paste in by hand — which is both annoying and
    error-prone, especially for the binary paths a successful RPC build
    needs to record. This merges instead: missing sections are appended,
    and the shard binary paths are updated in place when a build succeeds.
    Existing values and comments are left alone.
    """
    path = os.path.join(ROOT, "lmcluster.toml")
    enabled = "true" if binaries else "false"
    auto = "true" if binaries else "false"
    shard = SHARD_BLOCK.format(
        enabled=enabled, auto=auto,
        rpc_server=binaries.get("rpc", "").replace("\\", "\\\\"),
        llama_server=binaries.get("server", "").replace("\\", "\\\\"),
        model_dir=model_dir.replace("\\", "\\\\"))

    if not os.path.exists(path):
        content = f"""# LMCluster configuration for this machine.
#
# The defaults are fine for most setups. The one thing you will almost
# certainly want to set is model_dir, at the bottom.

[node]
# Blank means use this computer's hostname.
name = ""
port = 8470
open_browser = true
{CLUSTER_BLOCK}
# How machines find each other. The port has to match on every machine.
[discovery]
port = 8471
interval = 3.0         # seconds between announcements
timeout = 15.0         # a machine counts as gone after this long in silence

{shard}"""
        with open(path, "w") as f:
            f.write(content)
        say(f"  ✓ wrote {path}")
        return

    with open(path) as f:
        text = f.read()
    original = text
    tables = _existing_tables(text)
    added = []

    if "cluster" not in tables:
        text = text.rstrip() + "\n" + CLUSTER_BLOCK
        added.append("[cluster]")

    if "shard" not in tables:
        text = text.rstrip() + "\n" + shard
        added.append("[shard]")
    elif binaries:
        # Section is there but a build just produced new binaries: point
        # the existing keys at them rather than appending a duplicate table.
        changed = []
        updates = [
            ("rpc_server", '"%s"' % binaries.get("rpc", "")
                                    .replace("\\", "\\\\")),
            ("llama_server", '"%s"' % binaries.get("server", "")
                                      .replace("\\", "\\\\")),
            ("enabled", "true"),
            ("auto_start_worker", "true"),
        ]
        for key, value in updates:
            text, ok = _set_key(text, "shard", key, value)
            if ok:
                changed.append(key)
        if changed:
            added.append("shard binary paths (" + ", ".join(changed) + ")")

    if model_dir and re.search(r'^\s*model_dir\s*=\s*""', text, re.M):
        text, ok = _set_key(text, "shard", "model_dir", '"%s"'
                            % model_dir.replace("\\", "\\\\"))
        if ok:
            added.append("model folder")

    if text == original:
        configured = re.search(r'^\s*rpc_server\s*=\s*"[^"]+"', text, re.M)
        if configured:
            say(f"  ✓ {os.path.basename(path)} is already set up correctly")
        else:
            # Saying "already complete" here was misleading: the file is
            # unchanged because there was nothing new to write, not because
            # everything is in order.
            say(f"  {os.path.basename(path)} left as it is, but llama.cpp is")
            say("  still not configured in it — see the summary below.")
        return

    backup = path + ".bak"
    with open(backup, "w") as f:
        f.write(original)
    with open(path, "w") as f:
        f.write(text)
    say(f"  ✓ updated {os.path.basename(path)}: {', '.join(added)}")
    say(f"    previous version saved as {os.path.basename(backup)}")


# -- main -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Install one LMCluster node.")
    ap.add_argument("--yes", "-y", action="store_true",
                    help="accept defaults, never prompt")
    ap.add_argument("--with-rpc", dest="with_rpc", action="store_true",
                    help="fetch llama.cpp without prompting first")
    ap.add_argument("--build-from-source", dest="build_from_source",
                    action="store_true",
                    help="compile llama.cpp rather than downloading it "
                         "(needs a C++ compiler)")
    ap.add_argument("--no-rpc", dest="no_rpc", action="store_true",
                    help="skip llama.cpp entirely (dashboard only)")
    ap.add_argument("--gpu", default="auto",
                    choices=["auto", "cuda", "rocm", "vulkan", "metal",
                             "cpu", "none"],
                    help="hardware backend to use (default: detect)")
    ap.add_argument("--jobs", "-j", type=int, default=0,
                    help="parallel jobs when building from source")
    ap.add_argument("--token", default="",
                    help="cluster token to join an existing cluster")
    ap.add_argument("--check-tools", dest="check_tools", action="store_true",
                    help="report which build tools are visible and stop")
    ap.add_argument("--model-dir", dest="model_dir", default="",
                    help="folder holding your .gguf files (default: search "
                         "llama.cpp's cache and other usual places)")
    ap.add_argument("--new-token", dest="new_token", action="store_true",
                    help="discard the stored token and generate a new one "
                         "(other nodes must be given it too)")
    args = ap.parse_args()

    if args.check_tools:
        say("Build tools on this machine")
        say("-" * 40)
        for tool in ("git", "cmake", "ninja"):
            say(f"  {tool:8} {'found' if have(tool) else 'not found'}")
        if IS_WINDOWS:
            compiler, generator = find_windows_compiler()
            say(f"  compiler {compiler or 'not found'}")
            if generator:
                say(f"           cmake would use {' '.join(generator)}")
        else:
            found = [c for c in ("cc", "gcc", "clang") if have(c)]
            say(f"  compiler {', '.join(found) if found else 'not found'}")
        missing = check_build_tools()
        say("")
        if missing:
            say(f"Missing: {', '.join(missing)}")
            say("")
            say(build_hint())
        else:
            say("Everything needed is present. Run:")
            say(f"  {PY} install.py --with-rpc")
        return 0

    say("=" * 60)
    say("      LMCluster — node installer")
    say("=" * 60)
    say(f"  {platform.system()} {platform.machine()} · "
        f"Python {platform.python_version()}")

    total = 6
    binaries: dict = {}
    model_dir = ""
    token = None
    problems = []

    def attempt(number, title, fn, fatal=False):
        """Run one step, and carry on if it fails unless it is fatal.

        Steps used to run in a single try block, so anything going wrong in
        the middle skipped everything after it. That is how a Devuan
        machine ended up with no lmcluster.toml at all: the firewall step
        raised on a missing dependency, and writing the config — the one
        genuinely essential output — never happened. A step that cannot do
        its job should cost you that step and nothing else.
        """
        step(number, total, title)
        try:
            return fn()
        except (NeedsRestart, KeyboardInterrupt):
            raise
        except Abort as e:
            say(f"  ✗ {e}")
            problems.append(f"{title}: {e}")
        except Exception as e:
            say(f"  ✗ this step failed: {type(e).__name__}: {e}")
            problems.append(f"{title}: {e}")
            if fatal:
                raise Abort(str(e))
        return None

    try:
        attempt(1, "Python environment",
                lambda: install_python_env(args), fatal=True)

        def do_llamacpp():
            if args.no_rpc:
                say("  skipped (--no-rpc)")
                return None
            wanted = args.with_rpc or args.build_from_source
            if not wanted:
                say("  This is the part that lets one model run across")
                say("  several machines. It gets downloaded from the")
                say("  llama.cpp project, which takes a minute or two and")
                say("  needs no compiler.")
                wanted = ask("  Fetch it now?", default=True,
                             assume_yes=args.yes)
            if not wanted:
                say("  skipped — run this again with --with-rpc when ready")
                return None
            try:
                return get_llamacpp(args)
            except NeedsRestart as e:
                say("")
                say(f"  {e}")
                say("")
                say("  Nothing has gone wrong. Windows only picks up new")
                say("  tools in terminals opened after they were")
                say("  installed, so this one cannot see the compiler.")
                say("")
                say("  Close this window, open a new one, go to")
                say(f"    {ROOT}")
                say("  and run:")
                say(f"    {PY} install.py --with-rpc")
                say("")
                say("  The rest of the setup below will still finish.")
                return None

        binaries = attempt(2, "llama.cpp", do_llamacpp) or {}
        model_dir = attempt(3, "Where your models are",
                            lambda: choose_model_dir(args)) or ""
        attempt(4, "Network ports", lambda: open_firewall(args))
        token = attempt(5, "Cluster key", lambda: setup_token(args))
        attempt(6, "Configuration",
                lambda: write_config(binaries, model_dir), fatal=True)

    except NeedsRestart as e:
        say(f"\n{e}")
        say("Open a new terminal and run this installer again.")
        return 0
    except Abort as e:
        say(f"\n✗ Install failed: {e}")
        return 1
    except KeyboardInterrupt:
        say("\n\nInterrupted. Re-run install.py to pick up where it stopped.")
        return 130

    say("\n" + "=" * 60)
    if problems:
        say("  Finished, but some steps did not work.")
    elif binaries:
        say("  Done. This machine is ready.")
    else:
        # The old version printed "Installed." whatever had happened, so a
        # failed llama.cpp build still ended on a cheerful note several
        # lines below the error that mattered.
        say("  Partly done — llama.cpp is NOT built.")
    say("=" * 60)
    if problems:
        say("=" * 60)
        for p in problems:
            say(f"  • {p}")
        say("")

    # The config file is what everything else reads, so its absence is
    # worth shouting about rather than leaving to be discovered later.
    cfg_file = os.path.join(ROOT, "lmcluster.toml")
    if not os.path.exists(cfg_file):
        say("  ✗ No lmcluster.toml was written. The node will start with")
        say("    built-in defaults, which means it will not lend memory or")
        say("    load models. Fix whatever is listed above and run this")
        say("    installer again.")
        say("")
    else:
        say(f"  Config: {cfg_file}")

    if token:
        say(f"\n  Cluster key: {token}")
        say("  Every other machine needs this. Install them with:")
        say(f"    {PY} install.py --token {token}")
    say("\n  Start this node:")
    say("    ./run.sh" if not IS_WINDOWS else "    run.bat")
    if binaries:
        say("\n  This machine can lend its memory and load models.")
        say("  Set up the others the same way, using the key above.")
    else:
        say("\n  What works now: the dashboard will start and this machine")
        say("  will find the others on your network.")
        say("")
        say("  What does not: this machine cannot lend its memory to a")
        say("  model, and cannot load one itself. Both need llama.cpp,")
        say("  and the build above did not finish.")
        say("")
        say("  Fix the problem it reported, then run:")
        say(f"    {PY} install.py --with-rpc")
    if model_dir:
        say(f"\n  Models will be read from {model_dir}")
        say("  You can change that in the dashboard under Settings.")
    else:
        say("\n  No model folder set, so LMCluster will search llama.cpp's")
        say("  cache and the other usual places. Set one in Settings if you")
        say("  keep yours somewhere unusual.")
    say()
    return 0


if __name__ == "__main__":
    sys.exit(main())
