"""Configuration loading and persistent node identity.

Each node has a stable random ID generated on first run and stored in
~/.config/lmcluster/node_id so it survives IP changes and reinstalls.
"""

import os
import sys
import uuid
import socket

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore

DEFAULTS = {
    "node": {
        "name": "",            # defaults to hostname
        "port": 8470,
        "open_browser": True,  # open the dashboard automatically on launch
    },
    "discovery": {
        "port": 8471,          # UDP beacon port, shared across the LAN
        "interval": 3.0,       # seconds between beacons
        "timeout": 15.0,       # peer considered offline after this
    },
    "cluster": {
        "require_token": True,  # reject unauthenticated node-to-node calls
    },
    # How the model behaves when you talk to it. These are defaults; a
    # single request can override any of them without changing them here.
    "chat": {
        "system_prompt": "",
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "min_p": 0.05,
        "repeat_penalty": 1.1,
        "max_tokens": 0,        # 0 means no limit beyond the context window
    },
    "shard": {
        "enabled": False,           # set by install.py --with-rpc
        "rpc_port": 50052,
        "master_port": 8080,
        "auto_start_worker": False, # offer memory to the cluster on boot
        "rpc_server": "",           # path to the rpc-server binary
        "llama_server": "",         # path to the llama-server binary
        "model_dir": "",            # where .gguf files live
        "ctx": 4096,                # context window to load models with
        "tensor_split": "",         # how to weight layers across machines
        "extra_args": "",           # anything else to pass to llama-server
    },
}


def _state_dir() -> str:
    if os.name == "nt":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
    else:
        base = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    path = os.path.join(base, "lmcluster")
    os.makedirs(path, exist_ok=True)
    return path


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _toml_dump(data: dict, indent: str = "") -> str:
    """Minimal TOML emitter for our flat config shape (str/int/float/bool
    values, nested dict tables). Avoids an extra dependency."""
    lines = []
    scalars = {k: v for k, v in data.items() if not isinstance(v, dict)}
    tables = {k: v for k, v in data.items() if isinstance(v, dict)}

    def emit(value):
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        return '"' + str(value).replace('"', '\\"') + '"'

    for k, v in scalars.items():
        lines.append(f"{k} = {emit(v)}")

    def walk(prefix, table):
        subs = {k: v for k, v in table.items() if isinstance(v, dict)}
        flats = {k: v for k, v in table.items() if not isinstance(v, dict)}
        if flats or not subs:
            lines.append(f"\n[{prefix}]")
            for k, v in flats.items():
                lines.append(f"{k} = {emit(v)}")
        for k, v in subs.items():
            walk(f"{prefix}.{k}", v)

    for k, v in tables.items():
        walk(k, v)
    return "\n".join(lines) + "\n"


class Config:
    def __init__(self, path: str = "lmcluster.toml"):
        self.path = path
        raw = {}
        if os.path.exists(path):
            with open(path, "rb") as f:
                raw = tomllib.load(f)
        else:
            print(f"[config] no {path} found, using defaults", file=sys.stderr)
        self.data = _deep_merge(DEFAULTS, raw)
        self.node_id = self._load_identity(int(self.data["node"]["port"]))
        self.name = self.data["node"]["name"] or socket.gethostname()
        self.port = int(self.data["node"]["port"])
        self.discovery = self.data["discovery"]
        self.shard = self.data["shard"]
        self.chat = self.data["chat"]
        self.cluster = self.data["cluster"]

        # Imported here rather than at module scope: auth imports fastapi,
        # and config is also read by install.py before deps are installed.
        from . import auth
        self.cluster_token = auth.load_or_create_token()
        self.token_fingerprint = auth.fingerprint(self.cluster_token)

    @property
    def require_token(self) -> bool:
        return bool(self.cluster.get("require_token", True))

    def set_cluster_token(self, token: str) -> str:
        """Move this node onto a different cluster. Peers on the old token
        drop out of the registry once their beacons stop matching."""
        from . import auth
        self.cluster_token = auth.set_token(token)
        self.token_fingerprint = auth.fingerprint(self.cluster_token)
        return self.cluster_token

    @staticmethod
    def _load_identity(port: int) -> str:
        # Keyed by port so several nodes can share one machine (and one user)
        # without their beacons colliding on a shared identity.
        id_file = os.path.join(_state_dir(), f"node_id_{port}")
        if os.path.exists(id_file):
            with open(id_file) as f:
                nid = f.read().strip()
                if nid:
                    return nid
        nid = uuid.uuid4().hex[:12]
        with open(id_file, "w") as f:
            f.write(nid)
        return nid

    def save(self):
        """Persist current settings back to the TOML config file.

        The name is only written if it differs from the hostname, so nodes
        that were never explicitly named keep following their hostname
        (important for cloned images)."""
        if self.name != socket.gethostname():
            self.data["node"]["name"] = self.name
        self.data["node"]["port"] = self.port
        self.data["shard"] = self.shard
        self.data["cluster"] = self.cluster
        self.data["chat"] = self.chat
        with open(self.path, "w") as f:
            f.write(_toml_dump(self.data))

    def regenerate_id(self) -> str:
        """Issue a fresh node identity. Applied live; peers drop the old
        entry once its beacon times out."""
        id_file = os.path.join(_state_dir(), f"node_id_{self.port}")
        nid = uuid.uuid4().hex[:12]
        with open(id_file, "w") as f:
            f.write(nid)
        self.node_id = nid
        return nid
