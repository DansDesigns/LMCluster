"""The node: one per machine, all of them equal.

Every machine on the cluster runs this. It does three things:

  1. Announces itself on the LAN and keeps track of the other machines,
     including how much memory each has spare and what sort of network
     connection it is on.
  2. Offers its memory to the cluster by running llama.cpp's rpc-server,
     so a model loaded elsewhere can put some of its layers here.
  3. Optionally acts as the machine that loads a model and drives the
     others. Any node can do this; whichever one you load a model from
     becomes the one holding the conversation.

There is no master to configure and no fixed roles.
"""

import asyncio
import json
import os
import secrets
import time

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from . import (auth, engine as engine_mod, firewall, hardware,
               models as model_finder, rpc, skills, updater)
from .config import Config
from .discovery import Discovery
from .store import Store

STATIC = os.path.join(os.path.dirname(__file__), "static")

# Regenerated every time this process starts. The page uses it to tell an
# actual restart from a version number that merely changed on disk: an
# update writes the new version.txt seconds before the process is replaced,
# so watching the version alone would have the browser reload into a server
# that is about to exit.
BOOT_ID = secrets.token_hex(8)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class LoadModel(BaseModel):
    model_path: str
    ctx: int | None = None          # None means use the saved default
    extra_args: str | None = None
    tensor_split: str | None = None
    workers: list[str] | None = None


class ChatRequest(BaseModel):
    prompt: str
    system: str | None = None       # None means use the saved system prompt
    history: list[dict] = []
    temperature: float | None = None
    max_tokens: int | None = None
    chat_id: str | None = None


class SettingsUpdate(BaseModel):
    name: str | None = None
    model_dir: str | None = None
    auto_start_worker: bool | None = None
    cluster_token: str | None = None
    # How the model behaves when you talk to it
    system_prompt: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    max_tokens: int | None = None
    # How models get loaded
    ctx: int | None = None
    tensor_split: str | None = None
    extra_args: str | None = None


class KeyRotate(BaseModel):
    push_to_peers: bool = True


class KeySet(BaseModel):
    token: str


class SkillSource(BaseModel):
    source: str


class SkillRun(BaseModel):
    inputs: dict = {}
    timeout: float = 30.0


class SkillGenerate(BaseModel):
    description: str
    temperature: float = 0.3


class Node:
    def __init__(self, config: Config):
        self.cfg = config
        self.rpc_worker = rpc.RpcWorker(
            binary=config.shard.get("rpc_server", ""),
            port=int(config.shard.get("rpc_port", rpc.DEFAULT_RPC_PORT)))
        self.master = rpc.ShardMaster(
            binary=config.shard.get("llama_server", ""),
            port=int(config.shard.get("master_port", rpc.DEFAULT_MASTER_PORT)))
        # The engine holds the same dict the settings page edits, so a
        # change takes effect on the next message rather than needing a
        # restart.
        self.engine = engine_mod.Engine(self.master, defaults=config.chat)
        self.store = Store(auth._state_dir())
        self.discovery = Discovery(config, caps_fn=self.capabilities)

        # Checking every machine can take several seconds when one of them
        # is unreachable, since that means waiting for connections to time
        # out. Doing it while the dashboard waits would make the page feel
        # broken, and the page polls more often than the answer changes, so
        # it runs on its own schedule and the page reads the last result.
        self.peer_status: list[dict] = []
        self.peer_status_at: float = 0.0
        self._probe_lock = asyncio.Lock()

    def capabilities(self) -> dict:
        """What this machine tells the others about itself.

        Kept small because it goes out as a UDP datagram every few seconds.
        Only the three raw facts about the network link travel; the advice
        that goes with them is worked out by whichever node displays it, so
        we are not sending the same paragraph of prose over and over.
        """
        hw = hardware.probe(self.cfg.shard.get("model_dir") or None)
        link = hw.get("link") or {}
        worker = self.rpc_worker
        return {
            "rpc": worker.running,
            "rpc_capable": self.can_shard(),
            "rpc_port": worker.port,
            "holding_model": self.master.running,
            # What llama.cpp is actually able to use here, which is not the
            # same as what the machine physically contains.
            "devices": [{"id": d["id"], "name": d["name"], "free": d["free"]}
                        for d in worker.devices],
            "accelerators": len(worker.accelerators),
            "link": {"type": link.get("type"), "band": link.get("band"),
                     "speed_mbps": link.get("speed_mbps")},
            "ram_free": hw["ram_free"],
            "ram_total": hw["ram_total"],
            "vram_free": hw["vram_free"],
            "gpu": hw["gpu_backend"],
            "gpu_name": hw["gpu_name"],
            "gpu_integrated": hw.get("gpu_integrated", False),
            "cpu_count": hw["cpu_count"],
        }

    def can_shard(self) -> bool:
        return self.shard_problem() is None

    def shard_problem(self) -> str | None:
        """Why this machine cannot lend memory or load a model, if it can't.

        Returns None when everything is in order. Worth having as a reason
        rather than a bare no, because every one of these looks identical
        from another machine — "No RPC build" — while needing a completely
        different response.
        """
        binary = self.cfg.shard.get("rpc_server", "")
        if not self.cfg.shard.get("enabled"):
            return (f"shard mode is switched off in "
                    f"{os.path.basename(self.cfg.path)}. If llama.cpp is "
                    f"installed, re-run the installer with --with-rpc to "
                    f"record it.")
        if not binary:
            return (f"no path to the RPC server is recorded in "
                    f"{os.path.basename(self.cfg.path)}. Re-run the "
                    f"installer with --with-rpc.")
        if not os.path.exists(binary):
            # A recorded path that no longer exists reported as capable
            # before, and then failed only when someone tried to use it.
            return (f"the RPC server should be at {binary} but is not "
                    f"there. Re-run the installer with --with-rpc.")
        return None


def gpu_mismatch(info: dict) -> dict | None:
    """A graphics card present but invisible to llama.cpp.

    The RPC server exposes whichever accelerators its binary was built for,
    so a machine with a discrete card running a CPU-only build lends only
    its system memory. Nothing appears broken — the card is simply missing
    from the pool — and without saying so there is nothing to suggest why.

    Integrated graphics are deliberately not reported. They have no memory
    of their own, so adding one cannot increase how large a model the
    cluster can hold, and telling somebody to go and reinstall for no gain
    in capacity would be advice that wastes their afternoon.
    """
    if not info.get("rpc") and not info.get("rpc_available"):
        return None
    gpu = info.get("gpu") or info.get("gpu_backend")
    if not gpu or gpu == "none":
        return None
    if info.get("gpu_integrated"):
        return None
    if info.get("accelerators"):
        return None
    return {
        "severity": "warn",
        "message": f"has a {gpu} device that llama.cpp cannot see, so only "
                   f"its system memory is in the pool",
        "fix": f"its llama.cpp was built for the CPU only — re-run the "
               f"installer there with --gpu {gpu}",
    }


async def peer_status(node: Node, force: bool = False) -> list[dict]:
    """The last known state of every machine, refreshed in the background.

    Returns immediately with whatever was last measured. Pass force=True
    after doing something that changes the answer, such as starting another
    machine's worker, so the page does not show a stale result for the few
    seconds until the next sweep.
    """
    if force or not node.peer_status_at:
        return await refresh_peer_status(node)
    return node.peer_status


async def refresh_peer_status(node: Node) -> list[dict]:
    # One sweep at a time. Without this, a slow sweep overlapping with the
    # next one would double the number of connections being opened to a
    # machine that is already struggling to answer.
    async with node._probe_lock:
        node.peer_status = await diagnose_peers(node)
        node.peer_status_at = time.time()
        return node.peer_status


async def peer_status_loop(node: Node, interval: float = 6.0):
    while True:
        try:
            await refresh_peer_status(node)
        except Exception as e:
            print(f"[pool] could not check the other machines: {e}")
        await asyncio.sleep(interval)


async def diagnose_peers(node: Node) -> list[dict]:
    """Every other machine, with whether it can take layers and why not.

    A beacon tells us a worker was running a few seconds ago. Opening a TCP
    connection tells us whether it is running right now, which is what
    matters when we are about to hand it part of a model. But quietly
    dropping the machines that fail that check is no help either, because
    then a machine sits on the dashboard looking perfectly fine with no
    explanation of why its memory is missing from the pool. So every
    machine comes back either way, carrying a reason and something to do
    about it.
    """
    peers = node.discovery.registry.online()
    if not peers:
        return []

    async def check(peer):
        port = peer.get("rpc_port") or rpc.DEFAULT_RPC_PORT
        link = dict(peer.get("link") or {})
        link["warnings"] = hardware.link_warnings(link)
        entry = {**peer, "rpc_port": port, "link": link}

        if not peer.get("rpc_capable", True):
            ok, reason = False, "no_rpc_build"

        elif not peer.get("rpc_available"):
            # The machine's own announcement says it is not lending its
            # memory, and it is the authority on that. Probing anyway and
            # reading the result as a firewall problem — which is what this
            # used to do — accuses a machine of being misconfigured when it
            # has simply not been asked to join. On Windows that mistake is
            # guaranteed rather than occasional: a port with nothing
            # listening is silently discarded rather than refused, so every
            # idle machine looked firewalled from every other machine.
            ok, reason = False, "not_offered"

        else:
            ok, reason = await rpc.probe_worker(peer["ip"], port)
            if not ok and reason in ("timeout", "unreachable"):
                # Before blaming a firewall, check whether this machine can
                # be reached at all. Its dashboard port is a fair test,
                # since a beacon has already proved it is alive. If that
                # answers and the RPC port does not, the block really is on
                # that one port. If neither answers, something broader is
                # wrong and saying "firewall" would send somebody looking
                # in the wrong place.
                if await rpc.reachable(peer["ip"], peer.get("port", 8470)):
                    reason = "timeout"
                else:
                    reason = "unreachable"

        label, meaning, fix, fixable = rpc.PROBE_DIAGNOSIS[reason]
        entry.update({"in_pool": ok, "reason": reason, "label": label,
                      "meaning": meaning, "fix": fix.format(port=port),
                      "remotely_fixable": fixable,
                      "gpu_warning": gpu_mismatch(peer)})
        return entry

    results = await asyncio.gather(*(check(p) for p in peers),
                                  return_exceptions=True)
    return [r for r in results if isinstance(r, dict)]


def create_app(config: Config) -> FastAPI:
    node = Node(config)
    app = FastAPI(title="LMCluster")
    app.state.node = node

    # Proves the caller is part of this cluster. Nothing uses it on its own
    # at present — admin_guard accepts the same token and additionally lets
    # the machine's own dashboard through — but it is what a
    # machine-to-machine endpoint would want if one is added.
    peer_guard = [Depends(auth.make_dependency(config))]  # noqa: F841
    # Accepts either the cluster key or a request from this machine, which
    # is what the dashboard is. Used for anything that changes this node.
    admin_guard = [Depends(auth.make_local_or_token_dependency(config))]

    def require_local(request: Request):
        if not auth.is_loopback(request):
            raise HTTPException(
                403, "the cluster key can only be read or changed from the "
                     "machine itself. Open the dashboard at "
                     f"http://localhost:{config.port} on that machine, or "
                     "use an SSH tunnel.")

    @app.on_event("startup")
    async def startup():
        skills.ensure_builtins()
        node.discovery.start()
        asyncio.create_task(peer_status_loop(node))
        if node.can_shard() and config.shard.get("auto_start_worker"):
            try:
                node.rpc_worker.start()
                print(f"[pool] offering this machine's memory on port "
                      f"{node.rpc_worker.port}")
            except rpc.RpcError as e:
                print(f"[pool] could not offer memory: {e}")
        problem = node.shard_problem()
        if problem:
            print(f"[node] {config.name} ({config.node_id}) on port "
                  f"{config.port}")
            print(f"[node] cannot lend memory or load models: {problem}")
        else:
            print(f"[node] {config.name} ({config.node_id}) on port "
                  f"{config.port}, llama.cpp ready")

    @app.on_event("shutdown")
    async def shutdown():
        # Leaving a llama-server behind holding tens of gigabytes would be
        # the rudest possible way to fail, so tear both down explicitly.
        node.master.stop()
        node.rpc_worker.stop()

    # -- dashboard -------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index():
        with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as f:
            return f.read()

    # -- this machine ----------------------------------------------------

    @app.get("/api/health")
    async def health():
        return {"id": config.node_id, "name": config.name,
                "can_shard": node.can_shard(),
                "shard_problem": node.shard_problem(),
                "config_path": os.path.abspath(config.path),
                "worker": node.rpc_worker.status(),
                "capabilities": node.capabilities(),
                "version": updater.local_version(),
                "boot_id": BOOT_ID}

    @app.get("/api/cluster")
    async def cluster():
        """Every machine and its state, for the strip along the top.

        Deliberately cheap: this is read straight from the announcements
        machines broadcast, with no connections opened to check anything.
        The Pool page does the real probing, but the header is polled from
        every page and should cost nothing.

        Machines that stop announcing themselves are kept and marked
        offline rather than quietly disappearing, because a machine
        vanishing from the list is exactly the thing you want to notice.
        """
        def state(entry, is_self=False):
            if not entry.get("online", True):
                return "offline"
            if entry.get("rpc_available"):
                return "lending"
            return "idle"

        machines = [{
            "id": config.node_id,
            "name": config.name,
            "self": True,
            "state": ("lending" if node.rpc_worker.running
                      else "idle" if node.can_shard() else "unable"),
            "holding_model": node.master.running,
        }]
        for peer in node.discovery.registry.snapshot():
            machines.append({
                "id": peer["id"],
                "name": peer["name"],
                "self": False,
                "state": ("offline" if not peer["online"]
                          else "lending" if peer.get("rpc_available")
                          else "idle" if peer.get("rpc_capable")
                          else "unable"),
                "holding_model": False,
            })
        return {"machines": machines}

    @app.get("/api/settings")
    async def get_settings():
        return {"id": config.node_id, "name": config.name, "port": config.port,
                "model_dir": config.shard.get("model_dir", ""),
                "model_dir_in_use": model_finder.resolve(
                    config.shard.get("model_dir") or "",
                    project_root=PROJECT_ROOT)["dir"],
                "auto_start_worker": bool(
                    config.shard.get("auto_start_worker")),
                "can_shard": node.can_shard(),
                "shard_problem": node.shard_problem(),
                "config_path": os.path.abspath(config.path),
                "rpc_server": config.shard.get("rpc_server", ""),
                "llama_server": config.shard.get("llama_server", ""),
                "version": updater.local_version(),
                "chat": dict(config.chat),
                "loading": {
                    "ctx": config.shard.get("ctx", 4096),
                    "tensor_split": config.shard.get("tensor_split", ""),
                    "extra_args": config.shard.get("extra_args", ""),
                },
                "model_loaded": node.master.running}

    @app.post("/api/settings", dependencies=admin_guard)
    async def set_settings(upd: SettingsUpdate):
        if upd.name is not None and upd.name.strip():
            config.name = upd.name.strip()
        if upd.model_dir is not None:
            config.shard["model_dir"] = upd.model_dir.strip()
        if upd.auto_start_worker is not None:
            config.shard["auto_start_worker"] = upd.auto_start_worker
        if upd.cluster_token is not None and upd.cluster_token.strip():
            try:
                config.set_cluster_token(upd.cluster_token)
            except ValueError as e:
                raise HTTPException(400, str(e))

        # An empty system prompt is a real choice, so these are checked for
        # being absent rather than for being falsy.
        for field in ("system_prompt", "temperature", "top_p", "top_k",
                      "min_p", "repeat_penalty", "max_tokens"):
            value = getattr(upd, field)
            if value is not None:
                config.chat[field] = value

        if upd.ctx is not None:
            config.shard["ctx"] = max(512, int(upd.ctx))
        if upd.extra_args is not None:
            config.shard["extra_args"] = upd.extra_args.strip()
        if upd.tensor_split is not None:
            split = upd.tensor_split.strip()
            if split:
                # Validated against the pool as it stands now, which is a
                # snapshot: a machine joining later will make the split the
                # wrong length, and the check at load time will say so.
                pool = len([p for p in node.discovery.registry.online()
                            if p.get("rpc_available")]) + 1
                ok, why = rpc.check_tensor_split(split, pool)
                if not ok:
                    raise HTTPException(400, why)
            config.shard["tensor_split"] = split

        try:
            config.save()
        except OSError as e:
            raise HTTPException(500, f"could not write {config.path}: {e}")
        return await get_settings()

    @app.get("/api/version/check")
    async def version_check():
        return await updater.check()

    @app.post("/api/update/install", dependencies=admin_guard)
    async def update_install():
        """Download the current version, install it, and restart.

        Done in two steps on purpose. The download goes into tmp/ and is
        checked for actually being LMCluster before anything is replaced,
        so a failed or interrupted download cannot leave you with a broken
        installation.
        """
        staged = await updater.download_update()
        if not staged["ok"]:
            raise HTTPException(400, staged["message"])

        result = await asyncio.to_thread(updater.install_update,
                                         staged["staged_at"])
        if not result["ok"]:
            raise HTTPException(500, result["message"])

        updater.restart_soon(node)
        return {"ok": True,
                "from_version": staged.get("local"),
                "to_version": staged.get("downloaded_version"),
                "replaced": result["replaced"],
                "boot_id": BOOT_ID,
                "message": "Installed. This node is restarting; the page "
                           "will come back on its own."}

    # -- firewall --------------------------------------------------------

    @app.get("/api/firewall")
    async def firewall_status():
        return await asyncio.to_thread(firewall.status, config)

    @app.post("/api/firewall/open", dependencies=admin_guard)
    async def firewall_open():
        """Open the ports a cluster needs, asking the system to elevate.

        The prompt appears on the machine running the node, not in the
        browser — which matters when you are looking at one machine's
        dashboard from another. The reply says so.
        """
        result = await asyncio.to_thread(firewall.apply, config)
        return result

    # -- the pool --------------------------------------------------------

    @app.get("/api/pool")
    async def pool():
        peers = await peer_status(node)
        local = {**node.capabilities(), "name": config.name}
        return {
            "local": {
                "id": config.node_id, "name": config.name,
                "ram_free": local.get("ram_free"),
                "ram_total": local.get("ram_total"),
                "vram_free": local.get("vram_free"),
                "gpu": local.get("gpu"),
                "devices": local.get("devices", []),
                "accelerators": local.get("accelerators", 0),
                "gpu_warning": gpu_mismatch(local),
                "link": {**(local.get("link") or {}),
                         "warnings": hardware.link_warnings(
                             local.get("link") or {})},
                "can_shard": node.can_shard(),
                "shard_problem": node.shard_problem(),
                "offering_memory": node.rpc_worker.running,
            },
            "peers": peers,
            "summary": rpc.pool_summary(local, peers),
            "model": await node.engine.info(),
        }

    @app.post("/api/pool/offer/{action}", dependencies=admin_guard)
    async def offer_memory(action: str):
        """Start or stop offering this machine's memory to the cluster."""
        if action not in ("start", "stop"):
            raise HTTPException(400, "action must be start or stop")
        try:
            return (node.rpc_worker.start() if action == "start"
                    else node.rpc_worker.stop())
        except rpc.RpcError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/pool/peers/{node_id}/{action}", dependencies=admin_guard)
    async def fix_peer(node_id: str, action: str):
        """Start or stop another machine's worker from here.

        Without this, a machine that is not contributing has to be fixed by
        walking over to it, which rather defeats the point of a dashboard
        that already knows exactly what is wrong with it. The cluster key
        is what makes it reasonable to offer, since we can only do this to
        machines that share our key.
        """
        if action not in ("start", "stop"):
            raise HTTPException(400, "action must be start or stop")
        peer = next((p for p in node.discovery.registry.online()
                     if p["id"] == node_id), None)
        if peer is None:
            raise HTTPException(404, f"no machine on this cluster with id "
                                     f"{node_id}")
        url = f"http://{peer['ip']}:{peer['port']}/api/pool/offer/{action}"
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(url, headers=auth.client_headers(config))
        except httpx.HTTPError as e:
            raise HTTPException(502, f"could not reach {peer['name']}: {e}")
        if r.status_code == 401:
            raise HTTPException(
                502, f"{peer['name']} rejected our cluster key, so it is on a "
                     "different cluster. Re-run its installer with this "
                     "cluster's key.")
        if r.status_code >= 400:
            detail = r.json().get("detail", r.text) if r.text else r.text
            raise HTTPException(502, f"{peer['name']}: {detail}")
        # Give the worker a moment to bind, then look again so the page
        # reflects the change rather than the state before it.
        await asyncio.sleep(1.2)
        asyncio.create_task(refresh_peer_status(node))
        return {"node": peer["name"], "action": action, "result": r.json()}

    # -- loading a model -------------------------------------------------

    @app.get("/api/models")
    async def list_models():
        """Models this machine can load, and where they were found."""
        found = model_finder.resolve(config.shard.get("model_dir") or "",
                                     project_root=PROJECT_ROOT)
        return {"model_dir": found["dir"], "models": found["models"],
                "source": found["source"], "error": found["error"],
                "candidates": found["candidates"]}

    @app.get("/api/models/folders")
    async def model_folders():
        """Every folder worth looking in, with how many models are in each.

        Offered to the dashboard so choosing a folder is a matter of picking
        one off a list rather than typing a path from memory.
        """
        return {"configured": config.shard.get("model_dir") or "",
                "candidates": model_finder.candidates(PROJECT_ROOT)}

    @app.post("/api/plan")
    async def plan(req: LoadModel):
        """Work out which machines would carry a model, without loading it."""
        # Loading is worth a fresh look: a stale view could hand layers to
        # a machine that has just dropped off.
        peers = await peer_status(node, force=True)
        local = {**node.capabilities(), "name": config.name}
        ctx = req.ctx if req.ctx is not None else config.shard.get("ctx", 4096)
        try:
            result = rpc.plan_shard(req.model_path, local,
                                    [p for p in peers if p["in_pool"]],
                                    ctx=ctx,
                                    rpc_port=node.rpc_worker.port)
        except rpc.RpcError as e:
            raise HTTPException(400, str(e))
        split = (req.tensor_split if req.tensor_split is not None
                 else config.shard.get("tensor_split", ""))
        split_ok, split_why = rpc.check_tensor_split(
            split, len(result["workers"]) + 1)
        return {"plan": result,
                "explanation": rpc.explain_plan(result),
                "summary": rpc.pool_summary(local, peers),
                "excluded": [p for p in peers if not p["in_pool"]],
                "split_problem": None if split_ok else split_why}

    @app.post("/api/load", dependencies=admin_guard)
    async def load(req: LoadModel):
        if not node.can_shard():
            raise HTTPException(
                400, "This machine cannot load models, because llama.cpp was "
                     "not built with RPC support here. Re-run the installer "
                     "with --with-rpc.")
        peers = await peer_status(node, force=True)
        local = {**node.capabilities(), "name": config.name}
        ctx = req.ctx if req.ctx is not None else config.shard.get("ctx", 4096)
        try:
            result = rpc.plan_shard(req.model_path, local,
                                    [p for p in peers if p["in_pool"]],
                                    ctx=ctx,
                                    rpc_port=node.rpc_worker.port)
            if req.workers is not None:
                result["workers"] = [
                    {"host": w.split(":")[0],
                     "port": int(w.split(":")[1]) if ":" in w
                             else node.rpc_worker.port,
                     "name": w, "node_id": None, "usable": 0}
                    for w in req.workers]

            split = (req.tensor_split if req.tensor_split is not None
                     else config.shard.get("tensor_split", ""))
            if split:
                ok, why = rpc.check_tensor_split(
                    split, len(result["workers"]) + 1)
                if not ok:
                    raise HTTPException(400, why)
                result["tensor_split"] = split

            extra = (req.extra_args if req.extra_args is not None
                     else config.shard.get("extra_args", ""))
            node.master.start(result, extra_args=extra)
        except rpc.RpcError as e:
            raise HTTPException(400, str(e))
        await asyncio.sleep(1.5)  # let llama-server bind before reporting
        return {"status": node.master.status(),
                "explanation": rpc.explain_plan(result)}

    @app.post("/api/unload", dependencies=admin_guard)
    async def unload():
        return node.master.stop()

    @app.get("/api/load/log")
    async def load_log():
        """What llama-server has printed since it started.

        Worth surfacing, because when a large model fails to load the
        reason is almost always in here and nowhere else.
        """
        return {"running": node.master.running,
                "log": node.master.drain_log(200)}

    # -- talking to the model --------------------------------------------

    @app.get("/api/model")
    async def model_info():
        return await node.engine.info()

    @app.post("/api/chat", dependencies=admin_guard)
    async def chat(req: ChatRequest):
        """Streamed reply, as newline-delimited JSON.

        Guarded by loopback-or-token like everything else the dashboard
        touches. It was previously token-only, which meant the page served
        by this very node could not use it: the browser has no token and no
        way to be given one. Every test passed because every test sent a
        token explicitly, and the fault showed up as asking a question and
        getting silence.

        Streaming is not a nicety here. A large model spread over a home
        network may produce a token every second or two, so a non-streaming
        reply would leave the page looking frozen for minutes.
        """
        chat_id = req.chat_id or node.store.new_chat(req.prompt)
        node.store.add_message(chat_id, "user", req.prompt)

        async def body():
            yield json.dumps({"chat_id": chat_id}) + "\n"
            collected = []
            try:
                async for chunk in node.engine.stream(
                        req.prompt, system=req.system, history=req.history,
                        temperature=req.temperature,
                        max_tokens=req.max_tokens):
                    if chunk.get("delta"):
                        collected.append(chunk["delta"])
                    yield json.dumps(chunk) + "\n"
            except engine_mod.NoModelLoaded as e:
                yield json.dumps({"done": True, "error": str(e)}) + "\n"
                return
            except httpx.HTTPError as e:
                yield json.dumps({
                    "done": True,
                    "error": f"lost contact with the model: {e}. Check the "
                             "load log, because a machine holding part of it "
                             "may have dropped off."}) + "\n"
                return
            if collected:
                node.store.add_message(chat_id, "assistant",
                                       "".join(collected).strip())

        return StreamingResponse(body(), media_type="application/x-ndjson")

    @app.get("/api/chats")
    async def chats():
        return node.store.chats()

    @app.get("/api/chats/{chat_id}")
    async def chat_history(chat_id: str):
        return node.store.messages(chat_id)

    @app.delete("/api/chats/{chat_id}", dependencies=admin_guard)
    async def delete_chat(chat_id: str):
        return {"deleted": node.store.delete_chat(chat_id)}

    # -- skills ----------------------------------------------------------
    # Writing a skill amounts to getting a shell on this machine, so
    # everything that creates, changes or runs one is guarded. Reading the
    # list is not.

    @app.get("/api/skills")
    async def skill_list():
        return skills.list_all()

    @app.get("/api/skills/{skill_id}")
    async def skill_get(skill_id: str):
        try:
            skill = skills.get(skill_id)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if skill is None:
            raise HTTPException(404, f"no skill called '{skill_id}'")
        return skill

    @app.put("/api/skills/{skill_id}", dependencies=admin_guard)
    async def skill_save(skill_id: str, req: SkillSource):
        ok, problems = skills.validate(req.source)
        if not ok:
            raise HTTPException(400, {"errors": problems})
        try:
            saved = skills.save(skill_id, req.source)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {**saved, "warnings": problems}

    @app.delete("/api/skills/{skill_id}", dependencies=admin_guard)
    async def skill_delete(skill_id: str):
        try:
            removed = skills.delete(skill_id)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if not removed:
            raise HTTPException(404, f"no skill called '{skill_id}'")
        return {"id": skill_id, "deleted": True}

    @app.post("/api/skills/{skill_id}/run", dependencies=admin_guard)
    async def skill_run(skill_id: str, req: SkillRun):
        return await asyncio.to_thread(
            skills.execute, skill_id, req.inputs, req.timeout)

    @app.post("/api/skills/validate", dependencies=admin_guard)
    async def skill_validate(req: SkillSource):
        ok, problems = skills.validate(req.source)
        return {"valid": ok, "problems": problems}

    @app.post("/api/skills/generate", dependencies=admin_guard)
    async def skill_generate(req: SkillGenerate):
        """Have the loaded model write a skill, streamed as it works.

        Streamed rather than returned in one piece because this takes a
        while — a minute or two on a cluster is normal — and a spinner for
        that long tells you nothing. Watching the model work is more
        useful than watching a spinner: when the result is wrong you have
        already seen where it went wrong, rather than being handed a
        verdict about a file you never saw being written.

        The reply is saved as a conversation like any other, so it can be
        read again afterwards.
        """
        chat_id = node.store.new_chat(f"Write a skill: {req.description}")
        node.store.add_message(chat_id, "user",
                               f"Write a skill that does the following:\n\n"
                               f"{req.description}")

        async def body():
            yield json.dumps({"chat_id": chat_id}) + "\n"
            queue: asyncio.Queue = asyncio.Queue()

            async def on_delta(text):
                await queue.put(text)

            async def work():
                try:
                    return await engine_mod.generate_skill(
                        node.engine, req.description,
                        temperature=req.temperature, on_delta=on_delta)
                finally:
                    await queue.put(None)

            task = asyncio.create_task(work())
            collected = []
            while True:
                piece = await queue.get()
                if piece is None:
                    break
                collected.append(piece)
                yield json.dumps({"delta": piece}) + "\n"

            try:
                result = await task
            except engine_mod.NoModelLoaded as e:
                yield json.dumps({"done": True, "error": str(e)}) + "\n"
                return
            except httpx.HTTPError as e:
                yield json.dumps({"done": True,
                                  "error": f"the model stopped "
                                           f"answering: {e}"}) + "\n"
                return

            if collected:
                node.store.add_message(chat_id, "assistant",
                                       "".join(collected).strip())
            yield json.dumps({"done": True, **result}) + "\n"

        return StreamingResponse(body(), media_type="application/x-ndjson")

    # -- the cluster key -------------------------------------------------
    # Restricted to this machine rather than to the key itself. Guarding
    # these with the key would be circular, since the dashboard would need
    # the secret in order to show you the secret, and leaving them open
    # would let anybody on the network read the key out of the dashboard.

    @app.get("/api/key")
    async def get_key(request: Request):
        require_local(request)
        return {"key": config.cluster_token,
                "fingerprint": config.token_fingerprint,
                "peers_online": len(node.discovery.registry.online())}

    @app.post("/api/key/new")
    async def new_key(request: Request, req: KeyRotate):
        """Issue a new cluster key and hand it to the machines that are on.

        The order is the whole difficulty. If this machine changed its own
        key first it would immediately lose the authority to tell anybody
        else, and the cluster would fall apart. So the new key goes out to
        every machine using the current key, and only then does this one
        switch over.

        A machine that is asleep or switched off cannot be told anything,
        so it keeps the old key and drops out of the cluster. There is no
        way round that, so those machines are named in the reply rather
        than glossed over.
        """
        require_local(request)
        peers = node.discovery.registry.online()
        old, new = config.cluster_token, secrets.token_hex(16)

        results = []
        if req.push_to_peers:
            async def push(peer):
                url = f"http://{peer['ip']}:{peer['port']}/api/settings"
                try:
                    async with httpx.AsyncClient(timeout=20) as client:
                        r = await client.post(
                            url, json={"cluster_token": new},
                            headers={auth.TOKEN_HEADER: old})
                    if r.status_code == 401:
                        return {"node": peer["name"], "ok": False,
                                "error": "was already on a different key"}
                    if r.status_code >= 400:
                        return {"node": peer["name"], "ok": False,
                                "error": f"replied {r.status_code}"}
                    return {"node": peer["name"], "ok": True}
                except httpx.HTTPError as e:
                    return {"node": peer["name"], "ok": False, "error": str(e)}

            results = list(await asyncio.gather(*(push(p) for p in peers)))

        config.set_cluster_token(new)
        missed = [r for r in results if not r["ok"]]
        return {
            "key": new,
            "fingerprint": config.token_fingerprint,
            "updated": [r["node"] for r in results if r["ok"]],
            "missed": missed,
            "warning": (
                "These machines kept the old key and have left the cluster. "
                "Give them the new one from their own dashboards, or re-run "
                "their installers with it: "
                + ", ".join(r["node"] for r in missed)) if missed else None,
        }

    @app.post("/api/key")
    async def set_key(request: Request, req: KeySet):
        """Join a different cluster by adopting its key."""
        require_local(request)
        try:
            config.set_cluster_token(req.token)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"fingerprint": config.token_fingerprint,
                "note": "This machine now looks for others using the new key. "
                        "Any that are still on the old one will disappear "
                        "from the pool within a few seconds."}

    return app
