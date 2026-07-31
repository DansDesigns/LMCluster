"""Cluster token: shared secret that decides who is in this cluster.

Round-Table had no authentication at all — any process on the LAN could
answer a beacon and be handed prompts, or POST to /api/infer and use your
hardware. LMCluster's token model fixes that, and this is the merged
version of it.

Two uses, deliberately different:

  * Beacon fingerprint. Nodes broadcast sha256(token)[:16], never the
    token itself. Peers on a different token are ignored rather than
    rejected, so two clusters can share a LAN without seeing each other.
  * Request authentication. Node-to-node HTTP carries the real token in
    the X-Cluster-Token header, compared with hmac.compare_digest.

This is LAN-perimeter security, not transport security: the token crosses
the wire in the clear on plain HTTP, so it keeps honest machines apart
rather than defending against someone already sniffing your network. Put
the cluster on a trusted subnet.
"""

import hashlib
import hmac
import os
import secrets

# FastAPI is imported inside the functions that need it rather than here.
# Everything above those — reading and writing the cluster key, hashing it,
# comparing it — is plain standard library, and the installer uses exactly
# those parts while running under the system Python, which on Debian and
# Devuan has no FastAPI and cannot easily be given any. Importing it at the
# top made a module the installer depends on unimportable on precisely the
# systems where that matters.

TOKEN_HEADER = "X-Cluster-Token"
_TOKEN_FILE = "cluster_token"


def _state_dir() -> str:
    if os.name == "nt":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
    else:
        base = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    path = os.path.join(base, "lmcluster")
    os.makedirs(path, exist_ok=True)
    return path


def load_or_create_token() -> str:
    """Read the cluster token, generating one on first run.

    The first node to start invents the token; every other node is given
    it by the operator (installer prompt, or LMCLUSTER_TOKEN in the
    environment). An environment variable always wins so containers and
    provisioning scripts can set it without touching disk.
    """
    env = os.environ.get("LMCLUSTER_TOKEN", "").strip()
    if env:
        return env

    path = os.path.join(_state_dir(), _TOKEN_FILE)
    if os.path.exists(path):
        with open(path) as f:
            token = f.read().strip()
            if token:
                return token

    token = secrets.token_hex(16)
    with open(path, "w") as f:
        f.write(token)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass  # Windows and some filesystems: best effort
    return token


def set_token(token: str) -> str:
    """Join a different cluster. Written to the state dir, effective for
    new requests immediately and for beacons on the next interval."""
    token = token.strip()
    if not token:
        raise ValueError("cluster token cannot be empty")
    path = os.path.join(_state_dir(), _TOKEN_FILE)
    with open(path, "w") as f:
        f.write(token)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return token


def fingerprint(token: str) -> str:
    """Public, broadcastable identifier for a cluster token."""
    return hashlib.sha256(token.encode()).hexdigest()[:16]


def matches(token: str, candidate: str | None) -> bool:
    if not candidate:
        return False
    return hmac.compare_digest(token, candidate)


def rotate_token() -> str:
    """Generate and store a brand new cluster token.

    Rotating is destructive to cluster membership: every node still holding
    the old token stops matching our beacon fingerprint and drops out. The
    coordinated rotation in the node API pushes the new token to online
    peers *before* switching locally, so this low-level function should
    generally not be called on its own.
    """
    return set_token(secrets.token_hex(16))


def make_dependency(config):
    """Build a FastAPI dependency that guards node-to-node endpoints.

    Returned as a closure over config so the token can be rotated at
    runtime without re-registering routes.
    """
    from fastapi import Header, HTTPException

    async def require_token(
        x_cluster_token: str | None = Header(default=None,
                                             alias=TOKEN_HEADER),
    ):
        if not config.require_token:
            return  # open cluster, explicitly configured
        if not matches(config.cluster_token, x_cluster_token):
            raise HTTPException(
                401,
                f"bad or missing {TOKEN_HEADER}. Run the installer on this "
                "node with the cluster token from an existing node, or set "
                "LMCLUSTER_TOKEN.")

    return require_token


def is_loopback(request) -> bool:
    host = request.client.host if request.client else ""
    return host in ("127.0.0.1", "::1", "localhost")


def make_local_or_token_dependency(config):
    """Guard for endpoints that change a node's own configuration.

    Two callers legitimately need these: the dashboard the node itself
    serves, and another node in the cluster (a coordinated key rotation has
    to be able to hand the new key to its peers). So either proof is
    accepted — you are sitting at the machine, or you can demonstrate you
    are already in its cluster.

    This exists because leaving these routes open was a real hole: POST
    /api/settings accepts a new cluster_token, so before this guard any
    machine on the LAN could rewrite a node's key and walk it into a
    different cluster, unauthenticated. Cross-origin JSON POSTs are
    separately blocked by the absence of CORS middleware, which is what
    keeps a web page from abusing the loopback allowance.
    """

    # Imported here, before the nested function is defined, so its
    # annotations resolve. Without the Request annotation FastAPI treats
    # the parameter as a query string field and rejects every request to a
    # guarded endpoint with "field required".
    from fastapi import Header, HTTPException, Request

    async def require_local_or_token(
        request: Request,
        x_cluster_token: str | None = Header(default=None,
                                             alias=TOKEN_HEADER),
    ):
        if is_loopback(request):
            return
        if not config.require_token:
            return
        if matches(config.cluster_token, x_cluster_token):
            return
        raise HTTPException(
            401,
            f"this endpoint changes node configuration. Send a valid "
            f"{TOKEN_HEADER}, or use the dashboard on the machine itself.")

    return require_local_or_token


def client_headers(config) -> dict:
    """Headers for outbound node-to-node requests."""
    return {TOKEN_HEADER: config.cluster_token}
