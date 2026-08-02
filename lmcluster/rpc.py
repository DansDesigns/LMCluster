"""Splitting one model across several machines.

This is the whole point of LMCluster. llama.cpp can hold different layers
of a single model on different machines and pass the intermediate results
between them over the network, using what it calls its RPC backend. One
machine loads the model and drives the others; the rest simply lend their
memory.

The thing to understand before relying on it: this pools memory, it does
not divide up the work. The layers still run one after another, so while
the second machine is working on layers thirty to sixty, the first one is
sitting idle. What you gain is the ability to run a model that would not
fit on any single machine you own. What you do not gain is speed, and if a
model already fits on one machine then adding others will actually make it
slower.

So this is for the model that is too big, not the model that is too slow.

Requires llama.cpp built with -DGGML_RPC=ON, which the installer does.
"""

import asyncio
import os
import re
import shlex
import threading
import signal
import socket
import subprocess
import time

DEFAULT_RPC_PORT = 50052
DEFAULT_MASTER_PORT = 8080

# The master reserves this much of a node's free RAM for the OS, KV cache
# and its own working set before offering the rest to the plan.
HEADROOM_BYTES = 2 * 1024 ** 3


class RpcError(RuntimeError):
    pass


def _drain(proc, into: list, cap: int = 500):
    """Continuously move a process's output into a list, on a thread.

    Reading a subprocess's output without blocking looks like a job for
    select, and that is a trap. Once Python's buffered reader has pulled
    data out of the pipe, select on the underlying descriptor reports
    nothing further to read even though there are several complete lines
    sitting in the buffer. The result is that you read one line and
    conclude there is no more, which is exactly the bug that made this
    function necessary. A thread doing ordinary blocking reads has no such
    problem.
    """
    def run():
        try:
            for line in proc.stdout:
                into.append(line.rstrip())
                if len(into) > cap:
                    del into[:len(into) - cap]
        except (OSError, ValueError):
            pass  # the pipe closed, which is how this thread ends

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


# -- worker side ----------------------------------------------------------

class RpcWorker:
    """Runs llama.cpp's RPC server here, lending this machine's memory.

    Any machine can do this. The one you load a model from is the one
    holding the conversation; the rest are simply holding weights and doing
    arithmetic when asked. A worker holds no model file of its own — the
    machine that loaded the model pushes it whatever layers it assigned.

    The binary's command-line options have changed across versions, and it
    has been called two different things, so rather than assuming a set of
    flags this asks the binary what it accepts before starting it.
    """

    def __init__(self, binary: str, port: int = DEFAULT_RPC_PORT,
                 host: str = "0.0.0.0", use_cache: bool = True,
                 use_gpu: bool = True, server_binary: str = ""):
        self.binary = binary
        # llama-server, used only to ask what devices exist. The RPC server
        # can list them too, but only by starting up and binding a port.
        self.server_binary = server_binary
        self.port = int(port)
        self.host = host
        self.use_cache = use_cache
        # Whether to offer this machine's graphics memory as well as its
        # system memory. Switchable while running rather than needing a
        # different build, because the right answer is not obvious: a
        # graphics card is faster but holds far less, so on a large model
        # spread across machines the same card can be the reason it will
        # not fit.
        self.use_gpu = use_gpu
        self.proc: subprocess.Popen | None = None
        self.started_at: float | None = None
        self.last_error: str | None = None
        self.command: list[str] | None = None
        self.banner: list[str] = []
        self.devices: list[dict] = []
        self._flags: set | None = None
        self._devices_cache = None

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _supported_flags(self) -> set:
        """Ask the binary which options it understands.

        Guessing here is how you end up with a worker that refuses to
        start, or worse, one that starts and quietly listens on localhost
        only, so that every other machine sees a refused connection and the
        cluster never forms.
        """
        if self._flags is not None:
            return self._flags
        flags = set()
        try:
            r = subprocess.run([self.binary, "--help"], capture_output=True,
                               text=True, timeout=15, check=False)
            text = (r.stdout or "") + (r.stderr or "")
            for match in re.finditer(r"(--?[a-zA-Z][\w-]*)", text):
                flags.add(match.group(1))
        except (OSError, subprocess.SubprocessError):
            pass
        self._flags = flags
        return flags

    def available_devices(self) -> list[dict]:
        """Devices this machine could offer, cached for a short while."""
        now = time.time()
        cached = getattr(self, "_devices_cache", None)
        if cached and now - cached[0] < 60:
            return cached[1]
        found = list_devices(self.server_binary or self.binary)
        self._devices_cache = (now, found)
        return found

    def build_command(self) -> list[str]:
        flags = self._supported_flags()
        cmd = [self.binary]

        # Without an explicit host it binds to localhost, and every other
        # machine on the network sees the port refuse connections.
        if "--host" in flags:
            cmd += ["--host", self.host]
        elif "-H" in flags:
            cmd += ["-H", self.host]

        if "--port" in flags:
            cmd += ["--port", str(self.port)]
        else:
            cmd += ["-p", str(self.port)]

        # The local cache keeps large tensors on disk here instead of
        # pulling them across the network every time a model is loaded,
        # which makes a substantial difference on a big model.
        if self.use_cache and ("-c" in flags or "--cache" in flags):
            cmd += ["-c"]

        # Offering only the CPU means offering system memory only. The RPC
        # server exposes every device it finds unless told otherwise, and
        # --device is the documented way to narrow that.
        if not self.use_gpu and self.accelerator_backends and (
                "--device" in flags or "-d" in flags):
            cmd += ["--device", "CPU"]

        return cmd

    @property
    def accelerator_backends(self) -> list:
        return [b for b in build_backends(self.binary) if b != "cpu"]

    def environment(self) -> dict:
        """Environment for the worker process.

        A fallback for builds too old to have --device: every ggml backend
        honours a visible-devices variable, and an empty one leaves nothing
        visible. Harmless to set when --device is doing the work anyway.
        """
        env = dict(os.environ)
        if not self.use_gpu:
            for var in ("GGML_VK_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                        "HIP_VISIBLE_DEVICES", "GGML_SYCL_VISIBLE_DEVICES"):
                env[var] = ""
        return env

    def start(self) -> dict:
        if self.running:
            return self.status()
        if not self.binary:
            raise RpcError(
                "llama.cpp is not built on this machine, so it cannot lend "
                "its memory. Re-run the installer with --with-rpc.")
        if not os.path.exists(self.binary):
            raise RpcError(
                f"the RPC server should be at {self.binary} but is not "
                "there. If you moved or rebuilt llama.cpp, re-run the "
                "installer with --with-rpc to update the path.")

        cmd = self.build_command()
        self.command = cmd
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=self.environment())
        except OSError as e:
            self.last_error = str(e)
            raise RpcError(f"could not start the RPC server: {e}") from e

        # It either binds within a second or it has failed. Reporting
        # success and letting the operator discover otherwise from a
        # refused connection on another machine is not helpful.
        time.sleep(1.2)
        if self.proc.poll() is not None:
            output = ""
            try:
                output = (self.proc.stdout.read() or "")[:400]
            except (OSError, ValueError):
                pass
            self.last_error = output.strip() or "it exited immediately"
            self.proc = None
            raise RpcError(
                f"the RPC server started and stopped again: "
                f"{self.last_error}")

        self.started_at = time.time()
        self.last_error = None
        _drain(self.proc, self.banner)
        time.sleep(0.6)          # let the startup banner arrive
        self._parse_devices()
        return self.status()

    def _parse_devices(self):
        """Work out which devices the RPC server is actually offering.

        Worth reading from the program itself rather than guessing, because
        it reports what it can genuinely use, which is not the same as the
        hardware present in the machine. It exposes whichever accelerators
        the binary was built to support, and falls back to a single CPU
        device when it finds none. So a machine with a perfectly good
        graphics card, running a build compiled without that card's backend,
        will honestly report one CPU device — and the gap between that and
        what the hardware probe sees is precisely the thing worth telling
        somebody about.
        """
        self.devices = []
        in_devices = False
        for line in self.banner:
            if line.lower().startswith("devices:"):
                in_devices = True
                continue
            if not in_devices:
                continue
            # e.g. "  ROCm0: AMD Radeon RX 7900 XTX (24560 MiB, 24102 MiB free)"
            m = re.match(
                r"\s*(\w+):\s*(.+?)\s*\((\d+)\s*MiB,\s*(\d+)\s*MiB free\)",
                line)
            if m:
                self.devices.append({
                    "id": m.group(1), "name": m.group(2),
                    "total": int(m.group(3)) * 1024 * 1024,
                    "free": int(m.group(4)) * 1024 * 1024})
                continue
            simple = re.match(r"\s*(\w+):\s*(\S.*)$", line)
            if simple:
                self.devices.append({
                    "id": simple.group(1), "name": simple.group(2).strip(),
                    "total": None, "free": None})
                continue
            in_devices = False

    @property
    def accelerators(self) -> list[dict]:
        """Exposed devices that are not the plain CPU."""
        return [d for d in getattr(self, "devices", [])
                if not d["id"].upper().startswith("CPU")]

    def stop(self) -> dict:
        if self.proc is not None and self.proc.poll() is None:
            try:
                if os.name == "nt":
                    self.proc.terminate()
                else:
                    self.proc.send_signal(signal.SIGTERM)
                self.proc.wait(timeout=10)
            except (subprocess.TimeoutExpired, OSError):
                self.proc.kill()
        self.proc = None
        self.started_at = None
        self.devices = []
        self.banner = []
        return self.status()

    def status(self) -> dict:
        return {
            "running": self.running,
            "port": self.port,
            "pid": self.proc.pid if self.running else None,
            "uptime": (round(time.time() - self.started_at)
                       if self.started_at and self.running else None),
            "binary": self.binary,
            "command": " ".join(self.command) if self.command else None,
            "devices": self.devices,
            "accelerators": len(self.accelerators),
            "use_gpu": self.use_gpu,
            "can_use_gpu": bool(self.accelerator_backends),
            "banner": self.banner[:20],
            "error": self.last_error,
        }


# -- master side ----------------------------------------------------------

class ShardMaster:
    """Runs `llama-server` with --rpc pointed at the enlisted workers.

    The result is an ordinary OpenAI-compatible endpoint on this node, so
    the existing LlamaCppBackend consumes it with no changes — the rest of
    the cluster cannot tell a sharded model from a local one.
    """

    def __init__(self, binary: str, port: int = DEFAULT_MASTER_PORT):
        self.binary = binary
        self.port = int(port)
        self.proc: subprocess.Popen | None = None
        self.plan: dict | None = None
        self.log: list[str] = []
        self.started_at: float | None = None
        self.last_failure: str | None = None
        self.use_gpu: bool = True
        self._help: str | None = None

    def environment(self) -> dict:
        """Environment for llama-server.

        When this machine is set to lend system memory only, its own
        graphics device is hidden here with the visible-devices variables
        rather than with --device.

        The distinction matters. --device names the complete set of devices
        llama.cpp may use, and the RPC workers are in that set — so
        restricting it to CPU would exclude every other machine in the
        cluster and quietly turn a shared model into a local one. The
        environment variables only affect which local graphics devices the
        backends find, and leave the RPC devices alone.
        """
        env = dict(os.environ)
        if not self.use_gpu:
            for var in ("GGML_VK_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                        "HIP_VISIBLE_DEVICES", "GGML_SYCL_VISIBLE_DEVICES"):
                env[var] = ""
        return env

    def version(self) -> str | None:
        """Which llama.cpp build this is.

        Worth showing, because a good share of load failures are the model
        file being newer than the build rather than anything to do with the
        cluster, and the first useful question in that case is what build
        you are on.
        """
        text = self.help_text()
        match = re.search(r"version:\s*(\S+)", text)
        if match:
            return match.group(1)
        try:
            r = subprocess.run([self.binary, "--version"],
                               capture_output=True, text=True, timeout=8,
                               check=False)
            out = ((r.stdout or "") + (r.stderr or "")).strip()
            match = re.search(r"version:\s*(\S+)", out)
            if match:
                return match.group(1)
            return out.splitlines()[0][:60] if out else None
        except (OSError, subprocess.SubprocessError):
            return None

    def help_text(self) -> str:
        """What this llama-server build says about its own options.

        Asked rather than assumed, because the options change under us.
        Flash attention is the case that caught this out: it used to be a
        bare switch, `-fa`, and is now `-fa on|off|auto`. Give a current
        build the old form and it stops immediately with "expected value
        for argument"; give an older build the new form and it fails just
        as hard. Reading the help costs a fraction of a second and settles
        it either way.
        """
        if self._help is not None:
            return self._help
        try:
            # Short, because this is on the path of an ordinary page load.
            # A binary that does not answer promptly is treated as one that
            # said nothing, which costs a version number rather than
            # holding the dashboard open waiting.
            r = subprocess.run([self.binary, "--help"], capture_output=True,
                               text=True, timeout=8, check=False)
            self._help = (r.stdout or "") + (r.stderr or "")
        except subprocess.TimeoutExpired:
            self._help = ""
        except (OSError, subprocess.SubprocessError):
            self._help = ""
        return self._help

    def _flash_attn_args(self, wanted: bool) -> list:
        help_text = self.help_text()
        takes_value = bool(re.search(r"--flash-attn\s*\[?\s*on\s*\|",
                                     help_text))
        if takes_value:
            return ["-fa", "on" if wanted else "off"]
        # Older builds treat it as a plain switch, and there is no way to
        # ask for it off beyond leaving it out.
        return ["-fa"] if wanted else []

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def build_command(self, plan: dict, extra_args: str = "") -> list[str]:
        workers = plan["workers"]
        cmd = [
            self.binary,
            "-m", plan["model_path"],
            "--host", "0.0.0.0",
            "--port", str(self.port),
            "-c", str(plan.get("ctx", 4096)),
        ]
        if workers:
            cmd += ["--rpc", ",".join(f"{w['host']}:{w['port']}"
                                      for w in workers)]
        # Layer placement is deliberately left to llama.cpp unless somebody
        # asks otherwise.
        #
        # This used to force "-ngl 999", meaning "put every layer on a
        # device", on the reasoning that RPC endpoints count as devices and
        # so that is what spreads the model about. It does — and it also
        # switches off llama.cpp's own fitting, which is the part that
        # works out what will actually go where. Current builds say so
        # plainly when they give up:
        #
        #   common_fit_params: failed to fit params to free device memory:
        #   n_gpu_layers already set by user to 999, abort
        #
        # On a machine with a graphics card, 999 means trying to put a
        # twenty-four gigabyte model into a few gigabytes of video memory,
        # and it stops rather than falling back. Left alone, llama.cpp
        # measures every device it has, the RPC workers included, and fills
        # them in proportion.
        if plan.get("ngl") is not None:
            cmd += ["-ngl", str(plan["ngl"])]
        cmd += self._flash_attn_args(plan.get("flash_attn", True))
        # How much of the model each machine holds. Left alone, llama.cpp
        # divides the layers in proportion to each device's free memory,
        # which is a sensible default. Setting it by hand is worth doing
        # when memory is a poor guide to how useful a machine is — an old
        # laptop with plenty of free RAM and a slow processor will happily
        # accept a third of the model and then hold everything else up.
        if plan.get("tensor_split"):
            cmd += ["--tensor-split", plan["tensor_split"]]
        if plan.get("n_cpu_moe"):
            cmd += ["--n-cpu-moe", str(plan["n_cpu_moe"])]
        if extra_args:
            cmd += shlex.split(extra_args)
        return cmd

    def start(self, plan: dict, extra_args: str = "") -> dict:
        if self.running:
            raise RpcError("a shard is already running; stop it first")
        if not self.binary or not os.path.exists(self.binary):
            raise RpcError(
                f"llama-server binary not found at {self.binary!r}. Re-run "
                "install.py with --with-rpc.")
        if not os.path.exists(plan["model_path"]):
            raise RpcError(f"model not found: {plan['model_path']}")

        cmd = self.build_command(plan, extra_args)
        self.last_failure = None
        self.log = [f"$ {' '.join(shlex.quote(c) for c in cmd)}"]
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=self.environment())
        except OSError as e:
            raise RpcError(f"could not start llama-server: {e}") from e

        self.plan = plan
        self.started_at = time.time()
        _drain(self.proc, self.log)

        # A model that is too large, or a file llama.cpp cannot read, fails
        # during loading rather than at startup, so this waits long enough
        # to catch that rather than only catching a rejected option.
        deadline = time.time() + 6.0
        while time.time() < deadline:
            if self.proc.poll() is not None:
                break
            time.sleep(0.25)

        if self.proc.poll() is not None:
            code = self.proc.returncode
            # Give the reader thread a moment to collect whatever was
            # printed on the way out; the useful part is often the very
            # last line, written just before the process died.
            time.sleep(0.4)
            detail = self._failure_detail()
            meaning = self.explain_failure(detail)
            self.proc = None
            self.plan = None
            self.started_at = None
            raise RpcError(
                f"llama.cpp stopped while loading the model "
                f"(exit code {code})."
                + (f"\n\n{meaning}" if meaning else "")
                + f"\n\n{detail}\n\n"
                  f"The full output, including the command that was run, "
                  f"is under Load log.")

        return self.status()

    # Lines llama.cpp prints on the way up that say nothing about a
    # failure. Showing these instead of the error, which is what happened
    # before, tells you only that the program started — which you knew.
    _NOISE = ("common_param", "CORS is set to allow", "this can be a security",
              "more info: https", "-----", "build info", "system info",
              "llama_server: ---")

    _INTERESTING = ("error", "failed", "cannot", "unable", "not enough",
                    "out of memory", "oom", "no such file", "invalid",
                    "unsupported", "terminate", "assert", "abort",
                    "insufficient", "exception")

    # Things llama.cpp says when it gives up, and what they actually mean
    # in the context of a cluster. The raw message is accurate and assumes
    # you already know how llama.cpp allocates memory.
    _EXPLANATIONS = (
        ("failed to fit params to free device memory",
         "The layer count was forced, so llama.cpp could not work out what "
         "would fit and stopped instead. Clear 'Layers on devices' in "
         "Settings and let it decide."),
        ("erroroutofdevicememory",
         "It ran out of graphics memory. A graphics card holds far less "
         "than system memory, so a large model needs either more machines "
         "lending memory or a build without GPU support on this machine "
         "— re-run the installer here with --gpu cpu."),
        ("unable to allocate vulkan",
         "It ran out of graphics memory. Either bring more machines into "
         "the pool, or re-run the installer on this machine with --gpu cpu "
         "so the model uses system memory instead."),
        ("unable to allocate cuda",
         "It ran out of graphics memory. Either bring more machines into "
         "the pool, or re-run the installer on this machine with --gpu cpu."),
        ("not enough memory",
         "There is not enough memory for this model. Bring more machines "
         "into the pool, reduce the context window in Settings, or pick a "
         "smaller model."),
        ("no such file",
         "The model file could not be opened. If it is on a network drive "
         "or an external disk, check it is still attached."),
        ("unknown model architecture",
         "llama.cpp does not recognise this model at all. It probably needs "
         "a newer build — try Update llama.cpp in Settings."),
        ("wrong type",
         "The model file and this llama.cpp build disagree about the shape "
         "of the model's metadata, which means the file is newer than the "
         "build, or is a variant the build does not handle yet. Nothing to "
         "do with the cluster — the same file would fail on one machine on "
         "its own. Try Update llama.cpp in Settings, and if that does not "
         "help, a different quantisation or a more mainstream model."),
        ("error loading model hyperparameters",
         "llama.cpp could not make sense of this model's metadata, so the "
         "file is either newer than the build or a variant it does not "
         "handle. Try Update llama.cpp in Settings."),
        ("failed to allocate compute buffers",
         "The context window is too large for the memory available. Reduce "
         "it in Settings and try again."),
    )

    def explain_failure(self, detail: str) -> str | None:
        lowered = detail.lower()
        for needle, meaning in self._EXPLANATIONS:
            if needle in lowered:
                return meaning
        return None

    def _failure_detail(self, lines: int = 14) -> str:
        """The part of the output worth reading.

        Errors appear at the end. The previous version showed the first six
        hundred characters, which is the startup banner — the CORS notice
        and the verbosity setting — and cut off before anything that
        explained the failure.
        """
        body = [ln for ln in self.log[1:] if ln.strip()]
        if not body:
            return "It printed nothing on the way out."

        flagged = [ln for ln in body
                   if any(word in ln.lower() for word in self._INTERESTING)]
        if flagged:
            return "\n".join(flagged[-lines:])

        useful = [ln for ln in body
                  if not any(n in ln for n in self._NOISE)]
        tail = (useful or body)[-lines:]
        return "\n".join(tail)

    def stop(self) -> dict:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=15)
            except (subprocess.TimeoutExpired, OSError):
                self.proc.kill()
        self.proc = None
        self.plan = None
        self.started_at = None
        return self.status()

    # Lines llama.cpp prints while loading that actually say something
    # about how far along it is.
    _PROGRESS = ("load_tensors", "llama_model_loader", "print_info",
                 "loading model", "offloading", "model buffer size",
                 "init_tokenizer", "llama_context", "kv cache",
                 "compute buffer", "graph nodes", "warming up")

    def progress(self) -> str | None:
        """The most recent line that says something about loading.

        A large model spread over a network takes minutes to load, and a
        counter of elapsed seconds tells you nothing about whether it is
        working or wedged. llama.cpp is describing what it is doing the
        whole time — naming each tensor buffer as it is filled — and the
        last such line is a far better answer to "is this going anywhere".
        """
        for line in reversed(self.log[-80:]):
            text = line.strip()
            if not text or text.startswith("$"):
                continue
            if any(marker in text for marker in self._PROGRESS):
                # Strip llama.cpp's timestamp and severity prefix, which
                # take up half the width and say nothing.
                cleaned = re.sub(r"^[\d.]+\s+[IWED]\s+\S*\s*", "", text)
                return cleaned[:160]
        return None

    def drain_log(self, limit: int = 200) -> list[str]:
        """Whatever llama.cpp has printed so far.

        Collected by a background thread, so this is just a read.
        """
        return self.log[-limit:]

    def check_still_running(self):
        """Notice a model that died after we reported it loaded.

        A large model can take minutes to read off disk, so the check at
        start time cannot wait for the whole of it — six seconds catches a
        rejected option or a file it will not open, and then we let it get
        on with it. But it can still fail at forty seconds, having read
        twenty gigabytes and found there is nowhere to put the rest, and
        without this that failure would show up as nothing more than the
        model quietly not being loaded.
        """
        if self.proc is None or self.proc.poll() is None:
            return
        code = self.proc.returncode
        detail = self._failure_detail()
        meaning = self.explain_failure(detail)
        elapsed = (f" after {round(time.time() - self.started_at)}s"
                   if self.started_at else "")
        self.last_failure = (f"llama.cpp stopped{elapsed} (exit code {code})."
                             + (f"\n\n{meaning}" if meaning else "")
                             + f"\n\n{detail}")
        self.proc = None
        self.plan = None
        self.started_at = None

    def status(self) -> dict:
        self.check_still_running()
        return {
            "running": self.running,
            "port": self.port,
            "pid": self.proc.pid if self.running else None,
            "uptime": (round(time.time() - self.started_at)
                       if self.started_at and self.running else None),
            "plan": self.plan,
            "progress": self.progress(),
            "last_failure": self.last_failure,
            "url": f"http://127.0.0.1:{self.port}" if self.running else None,
        }


# -- planning -------------------------------------------------------------

# Why a worker probe failed, and what the operator should do about it.
# Each entry: (short label, what it means, how to fix it, is it fixable
# remotely from the dashboard).
PROBE_DIAGNOSIS = {
    "ok": ("Ready", "Accepting connections.", "", False),
    "serving": (
        "Holding the model",
        "This machine is busy with part of the model that is loaded, which "
        "is exactly what it is meant to be doing.",
        "", False),
    "not_offered": (
        "Worker not started",
        "This machine has llama.cpp built and ready, but is not currently "
        "lending its memory to the cluster.",
        "Start it from here — one click, nothing to log in to.",
        True),
    "refused": (
        "Worker died",
        "This machine said it was lending its memory, but nothing is "
        "listening on the port now, so the RPC server has stopped.",
        "Restart it from here. If it stops again, look at that machine's "
        "own dashboard: a worker that exits straight away usually means "
        "llama.cpp was built without RPC support.",
        True),
    "timeout": (
        "Port blocked",
        "This machine answers on its dashboard port but not on the one used "
        "for lending memory, and the connection is being discarded rather "
        "than refused. That is a firewall rule allowing one port and not "
        "the other.",
        "Open port {port} on that machine. Its own dashboard can do it: "
        "Settings, then Network ports. On Windows check the rule covers the "
        "network profile in use — a rule for Private networks does nothing "
        "if Windows has decided your network is Public.",
        False),
    "unreachable": (
        "Cannot be reached",
        "This machine is announcing itself, so it is alive and on the "
        "network, but no connection to it succeeds at all.",
        "Something between the two machines is blocking traffic — a "
        "firewall set to block everything inbound, client isolation on a "
        "Wi-Fi access point, or the two being on different subnets.",
        False),
    "no_rpc_build": (
        "No RPC build",
        "This node is in the cluster but its llama.cpp was not built with "
        "RPC support, so it cannot contribute memory.",
        "On that machine, run: install.py --with-rpc",
        False),
}


# Which ggml library means which backend. Checking for these beside the
# binary is the one wholly reliable way to know what a build supports: it
# needs nothing run and nothing parsed, and it is true whether or not the
# machine happens to have a suitable device in it.
_BACKEND_LIBS = {
    "vulkan": ("ggml-vulkan",),
    "cuda": ("ggml-cuda",),
    "rocm": ("ggml-hip", "ggml-rocm"),
    "metal": ("ggml-metal",),
    "sycl": ("ggml-sycl",),
    "opencl": ("ggml-opencl",),
}


def build_backends(binary: str) -> list[str]:
    """Which accelerators this llama.cpp build was compiled for.

    Read from the library files sitting beside the binary. The alternative
    — starting the RPC server and reading what it says about itself — only
    works if it is running, only reports devices actually present, and
    depends on the exact wording of its output. This does not: a build with
    ggml-vulkan beside it is a Vulkan build, full stop, and that is the
    question being asked when somebody reinstalls with --gpu vulkan and
    wants to know whether it took.
    """
    if not binary or not os.path.exists(binary):
        return []
    folder = os.path.dirname(os.path.abspath(binary))
    try:
        names = [n.lower() for n in os.listdir(folder)]
    except OSError:
        return []

    found = []
    for backend, stems in _BACKEND_LIBS.items():
        if any(n.startswith(stem) for stem in stems for n in names):
            found.append(backend)
    if any(n.startswith("ggml-cpu") or n.startswith("libggml-cpu")
           for n in names):
        found.append("cpu")
    return found


# Which device identifier belongs to which backend. llama.cpp names them
# by backend and index — CUDA0, Vulkan0, ROCm0 — with the plain CPU always
# called CPU.
_DEVICE_PREFIX = {"cuda": "CUDA", "vulkan": "Vulkan", "rocm": "ROCm",
                  "metal": "Metal", "sycl": "SYCL", "opencl": "OpenCL"}


def list_devices(binary: str, timeout: int = 20) -> list[dict]:
    """Every device this build can see, without starting a server.

    llama.cpp will list them on request, which is better than starting the
    RPC server and reading what it prints: that binds a port, and it has to
    be stopped again afterwards.

    Output looks like:

        Available devices:
          Vulkan0: AMD Radeon 680M (12288 MiB, 11900 MiB free)
          CPU: AMD Ryzen 7 (32768 MiB, 24110 MiB free)
    """
    if not binary or not os.path.exists(binary):
        return []
    try:
        r = subprocess.run([binary, "--list-devices"], capture_output=True,
                           text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return []

    text = (r.stdout or "") + (r.stderr or "")
    devices = []
    for line in text.splitlines():
        m = re.match(r"\s*(\w+):\s*(.+?)\s*\((\d+)\s*MiB,\s*(\d+)\s*MiB free\)",
                     line)
        if not m:
            continue
        ident = m.group(1)
        # The listing is preceded by other lines that can look similar;
        # only accept identifiers that name a backend we know.
        if ident.upper() != "CPU" and not any(
                ident.startswith(p) for p in _DEVICE_PREFIX.values()):
            continue
        devices.append({
            "id": ident,
            "name": m.group(2),
            "total": int(m.group(3)) * 1024 * 1024,
            "free": int(m.group(4)) * 1024 * 1024,
            "accelerator": ident.upper() != "CPU",
        })
    return devices


def describe_build(binary: str) -> str:
    """A short phrase for what this build can use, for the dashboard."""
    backends = [b for b in build_backends(binary) if b != "cpu"]
    if not backends:
        return "CPU build"
    return " and ".join(b.upper() if b in ("cuda", "rocm") else b.title()
                        for b in backends) + " build"


async def probe_worker(host: str, port: int,
                       timeout: float = 3.0,
                       attempts: int = 2) -> tuple[bool, str]:
    """Is an RPC server accepting connections here, and if not, why not?

    A bare TCP connect, because the protocol is binary and we only need
    liveness, not a handshake.

    The distinction that matters is refused versus timed out. Refused means
    nothing is listening, which the dashboard can fix by starting the
    worker. A timeout means packets are being discarded, which is a
    firewall the operator has to open themselves.

    Retried before giving up, and with a more forgiving timeout than the
    two seconds this used to allow. A machine that is busy loading a model,
    or reached over Wi-Fi, can easily take longer than that to answer, and
    reporting a slow machine as firewalled sends somebody off to fix
    something that was never broken.
    """
    last = "unreachable"
    for attempt in range(attempts):
        try:
            fut = asyncio.open_connection(host, port)
            _, writer = await asyncio.wait_for(fut, timeout=timeout)
            writer.close()
            try:
                await writer.wait_closed()
            except (OSError, asyncio.CancelledError):
                pass
            return True, "ok"
        except asyncio.TimeoutError:
            last = "timeout"
        except ConnectionRefusedError:
            # Refusal is a definite answer; nothing is listening, and
            # trying again will not change that.
            return False, "refused"
        except OSError:
            last = "unreachable"
        if attempt + 1 < attempts:
            await asyncio.sleep(0.4)
    return False, last


async def reachable(host: str, port: int, timeout: float = 3.0) -> bool:
    """Can we open any connection to this machine at all?

    Used to tell a single blocked port from a machine that cannot be
    reached in general. The dashboard port is the natural thing to try,
    since a beacon has already proved the machine is alive and on the
    network.
    """
    try:
        fut = asyncio.open_connection(host, port)
        _, writer = await asyncio.wait_for(fut, timeout=timeout)
        writer.close()
        try:
            await writer.wait_closed()
        except (OSError, asyncio.CancelledError):
            pass
        return True
    except (OSError, asyncio.TimeoutError):
        return False


def pool_summary(local: dict, workers: list[dict]) -> dict:
    """What the cluster has to offer, itemised.

    Two machines will often disagree about the total, and the reason is
    worth understanding rather than treating as a fault.

    A machine always counts its own memory, because whichever machine you
    load a model from holds part of that model itself — it does not need to
    lend anything to anybody to do that. But it counts another machine only
    if that machine is lending and can be reached. So if one of three
    machines has not joined the pool, it will show a larger total than the
    other two: it is counting itself, and they are not counting it.

    That is correct in each case, and the figure that matters is the one on
    the machine you actually load from. The per-machine breakdown returned
    here lets the dashboard show the arithmetic instead of asking anyone to
    take the total on trust.
    """
    def usable(node: dict) -> int:
        ram = node.get("ram_free") or 0
        vram = node.get("vram_free") or 0
        if node.get("gpu_backend") == "metal":
            vram = 0  # unified memory: already counted as RAM
        return max(0, ram + vram - HEADROOM_BYTES)

    live = usable(local)
    potential = live
    ready, fixable, blocked = [], [], []
    breakdown = [{
        "name": local.get("name", "this machine"),
        "self": True,
        "bytes": usable(local),
        "counted": True,
        "why": "counted because a model loaded here is held here",
    }]

    for w in workers:
        share = usable(w)
        # "serving" means the machine is carrying part of the model right
        # now, which is as much a member of the pool as one merely standing
        # ready — more so, in fact.
        counted = w["reason"] in ("ok", "serving")
        if counted:
            live += share
            potential += share
            ready.append(w)
        elif w.get("remotely_fixable"):
            potential += share
            fixable.append(w)
        else:
            blocked.append(w)
        breakdown.append({
            "name": w.get("name", w.get("ip", "?")),
            "self": False,
            "bytes": share,
            "counted": counted,
            "why": (("holding part of the model" if w["reason"] == "serving"
                     else "lending its memory") if counted
                    else f"not counted: {w.get('label', 'unavailable').lower()}"),
        })

    return {
        "live_bytes": live,
        "potential_bytes": potential,
        "node_count": 1 + len(ready),
        "ready": len(ready),
        "fixable": len(fixable),
        "blocked": len(blocked),
        "breakdown": breakdown,
        "slowest_link": _slowest_link(local, ready),
    }


def _slowest_link(local: dict, ready: list[dict]) -> dict | None:
    """The worst link in the pool. In a pipeline every node waits on the
    slowest hop, so this single figure predicts the experience better than
    any average would."""
    candidates = []
    for node in [local, *ready]:
        link = node.get("link") or {}
        sev = [w["severity"] for w in link.get("warnings", [])]
        rank = 0 if "bad" in sev else 1 if "warn" in sev else 2
        candidates.append((rank, node.get("name", "this node"), link))
    if not candidates:
        return None
    rank, name, link = min(candidates, key=lambda c: c[0])
    if rank >= 2:
        return None  # everything is fine, nothing to report
    return {"node": name, "type": link.get("type"), "band": link.get("band"),
            "speed_mbps": link.get("speed_mbps"),
            "warnings": link.get("warnings", [])}


def model_size(path: str) -> int | None:
    """Bytes of a GGUF model, following the multi-part convention where a
    model is split into -00001-of-0000N files."""
    if not os.path.exists(path):
        return None
    if os.path.isdir(path):
        total = 0
        for entry in os.scandir(path):
            if entry.is_file() and entry.name.endswith(".gguf"):
                total += entry.stat().st_size
        return total or None

    size = os.path.getsize(path)
    base = os.path.basename(path)
    if "-of-" in base:
        directory = os.path.dirname(path) or "."
        stem = base.rsplit("-", 3)[0]
        total = 0
        for entry in os.scandir(directory):
            if entry.name.startswith(stem) and entry.name.endswith(".gguf"):
                total += entry.stat().st_size
        return total or size
    return size


def plan_shard(model_path: str, local: dict, peers: list[dict],
               ctx: int = 4096, rpc_port: int = DEFAULT_RPC_PORT) -> dict:
    """Decide which nodes to enlist for a model.

    Greedy and deliberately simple: sort candidates by usable memory,
    take nodes until the model fits, and stop. A smarter placement would
    weigh interconnect speed and per-node compute, but greedy-by-capacity
    is the honest baseline and it is easy to reason about when it goes
    wrong.

    `local` and `peers` are hardware.probe() snapshots; peers additionally
    carry ip/name/rpc fields from the discovery beacon.
    """
    size = model_size(model_path)
    if size is None:
        raise RpcError(f"cannot size model at {model_path!r} — does it exist?")

    def usable(node: dict) -> int:
        ram = node.get("ram_free") or 0
        vram = node.get("vram_free") or 0
        # Unified-memory Macs would double-count RAM as VRAM.
        if node.get("gpu_backend") == "metal":
            vram = 0
        return max(0, ram + vram - HEADROOM_BYTES)

    local_usable = usable(local)
    candidates = sorted(
        [p for p in peers if p.get("rpc_available")],
        key=usable, reverse=True)

    workers: list[dict] = []
    pooled = local_usable
    for peer in candidates:
        if pooled >= size:
            break
        share = usable(peer)
        if share <= 0:
            continue
        workers.append({
            "host": peer["ip"],
            "port": peer.get("rpc_port", rpc_port),
            "name": peer.get("name", peer["ip"]),
            "node_id": peer.get("id"),
            "usable": share,
        })
        pooled += share

    return {
        "model_path": model_path,
        "model_size": size,
        "ctx": ctx,
        "workers": workers,
        "local_usable": local_usable,
        "pooled_usable": pooled,
        "fits": pooled >= size,
        "shortfall": max(0, size - pooled),
        # None means "do not pass -ngl at all", which lets llama.cpp fit
        # the layers to the memory it can actually see.
        "ngl": None,
        "flash_attn": True,
    }


def check_tensor_split(value: str, machine_count: int) -> tuple[bool, str]:
    """Is this a usable split for the machines we have?

    Returns (ok, message). Checked before launching rather than after,
    because llama.cpp's own complaint about a bad split arrives buried in
    several hundred lines of startup output.
    """
    value = (value or "").strip()
    if not value:
        return True, ""
    parts = [p.strip() for p in value.replace(" ", ",").split(",") if p.strip()]
    try:
        numbers = [float(p) for p in parts]
    except ValueError:
        return False, (f"the split should be numbers separated by commas, "
                       f"like 3,2,1 — not {value!r}")
    if any(n < 0 for n in numbers):
        return False, "the split cannot contain negative numbers"
    if sum(numbers) <= 0:
        return False, "the split has to add up to more than zero"
    if len(numbers) != machine_count:
        return False, (f"the split has {len(numbers)} number(s) but there "
                       f"{'is' if machine_count == 1 else 'are'} "
                       f"{machine_count} machine(s) in the pool. Give one "
                       f"number per machine, this one first.")
    return True, ""


def explain_plan(plan: dict) -> str:
    """One paragraph a human can sanity-check before committing."""
    gb = lambda b: f"{b / 1e9:.1f} GB"  # noqa: E731
    if not plan["workers"]:
        if plan["fits"]:
            return (f"{gb(plan['model_size'])} model fits in this node's "
                    f"{gb(plan['local_usable'])} on its own — no RPC workers "
                    "needed, and adding them would only slow it down.")
        return (f"{gb(plan['model_size'])} model does not fit in this node's "
                f"{gb(plan['local_usable'])}, and no RPC workers are "
                f"available. Short by {gb(plan['shortfall'])}.")

    names = ", ".join(w["name"] for w in plan["workers"])
    head = (f"{gb(plan['model_size'])} model across this node "
            f"({gb(plan['local_usable'])}) plus {len(plan['workers'])} "
            f"worker(s): {names}. Pooled: {gb(plan['pooled_usable'])}.")
    if not plan["fits"]:
        return head + (f" Still short by {gb(plan['shortfall'])} — it will "
                       "spill to disk and crawl, or fail to load.")
    return head + " Expect capacity, not speed: layers run in sequence."
