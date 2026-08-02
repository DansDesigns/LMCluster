"""LAN node discovery via UDP broadcast beacons.

Every node broadcasts a small JSON beacon on a shared UDP port every few
seconds and listens for beacons from others. No broker, no mDNS daemon,
works the same on Devuan, Debian, or anything else with a network stack.

Each beacon says who the machine is and what it can contribute: whether
its rpc-server is running, how much memory it has spare, and what sort of
network connection it is on. The planner needs those figures to decide
where a model's layers should go, and a beacon repeated every few seconds
is the cheapest way to keep them current.

It also carries a fingerprint of the cluster token — sha256(token)[:16],
never the token itself. Beacons whose fingerprint does not match ours are
ignored, so two clusters can share a LAN without seeing each other. This
is separation, not security; the HTTP layer does the actual authenticating.
"""

import json
import socket
import threading
import time


class PeerRegistry:
    """Thread-safe table of known peers, keyed by node_id."""

    def __init__(self, timeout: float):
        self._peers: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._timeout = timeout

    def update(self, beacon: dict, addr: str):
        caps = beacon.get("caps") or {}
        with self._lock:
            self._peers[beacon["id"]] = {
                "id": beacon["id"],
                "name": beacon.get("name", "?"),
                "ip": addr,
                "port": beacon.get("port", 8470),
                "last_seen": time.time(),
                # shard-mode capability
                "rpc_available": bool(caps.get("rpc")),
                "rpc_capable": bool(caps.get("rpc_capable")),
                "accelerators": caps.get("accelerators", 0),
                "build": caps.get("build", []),
                "build_label": caps.get("build_label"),
                "use_gpu": caps.get("use_gpu", True),
                "can_use_gpu": caps.get("can_use_gpu", False),
                "devices": caps.get("devices", []),
                "link": caps.get("link") or {},
                "rpc_port": caps.get("rpc_port"),
                "ram_free": caps.get("ram_free"),
                "ram_total": caps.get("ram_total"),
                "vram_free": caps.get("vram_free"),
                "gpu_backend": caps.get("gpu", "none"),
                "gpu_name": caps.get("gpu_name"),
                "cpu_count": caps.get("cpu_count"),
            }

    def snapshot(self) -> list[dict]:
        now = time.time()
        with self._lock:
            out = []
            for peer in self._peers.values():
                p = dict(peer)
                p["online"] = (now - p["last_seen"]) < self._timeout
                p["url"] = f"http://{p['ip']}:{p['port']}"
                out.append(p)
            return sorted(out, key=lambda p: p["name"])

    def online(self) -> list[dict]:
        return [p for p in self.snapshot() if p["online"]]


class Discovery:
    def __init__(self, config, caps_fn=None):
        self.cfg = config
        self.caps_fn = caps_fn or (lambda: {})  # callable -> dict
        self.registry = PeerRegistry(timeout=config.discovery["timeout"])
        self._stop = threading.Event()
        self._udp_port = int(config.discovery["port"])

    def start(self):
        threading.Thread(target=self._beacon_loop, daemon=True).start()
        threading.Thread(target=self._listen_loop, daemon=True).start()

    def stop(self):
        self._stop.set()

    def broadcast_targets(self) -> list[str]:
        """Every address a beacon should be sent to.

        Sending once to 255.255.255.255, which is what this used to do,
        delivers on exactly one interface — whichever the routing table
        happens to pick. On a machine carrying virtual adapters, and a
        Windows machine with WSL, Hyper-V, VirtualBox or Docker installed
        carries several, that is frequently a virtual one, so the beacon
        never reaches the real network at all.

        The failure this produces is thoroughly confusing, because
        receiving still works perfectly: the listener is bound to every
        interface. So that machine sees the whole cluster while the rest of
        the cluster cannot see it, and the fault looks like it lies with
        the machines that cannot see, rather than the one that cannot be
        heard.

        The answer is to send to every interface's own broadcast address as
        well.
        """
        targets = ["255.255.255.255"]
        try:
            import psutil
            for _name, addrs in psutil.net_if_addrs().items():
                for addr in addrs:
                    if addr.family != socket.AF_INET:
                        continue
                    if addr.broadcast and addr.broadcast not in targets:
                        targets.append(addr.broadcast)
        except (ImportError, OSError, AttributeError):
            # psutil is optional. Without it the fallback below still
            # covers the common case of a single ordinary network.
            pass

        if len(targets) == 1:
            for guess in self._guess_broadcasts():
                if guess not in targets:
                    targets.append(guess)
        return targets

    def _guess_broadcasts(self) -> list[str]:
        """Broadcast addresses worked out without psutil.

        Assumes a /24, which is what home networks almost always are. Wrong
        on an unusual netmask, but a wrong broadcast address is simply a
        packet that goes nowhere, and 255.255.255.255 is still being tried
        alongside it.
        """
        out = []
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None,
                                           socket.AF_INET):
                ip = info[4][0]
                if ip.startswith("127."):
                    continue
                out.append(".".join(ip.split(".")[:3]) + ".255")
        except OSError:
            pass
        return out

    def _beacon_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        interval = float(self.cfg.discovery["interval"])
        targets, refreshed = self.broadcast_targets(), time.time()

        while not self._stop.is_set():
            # Interfaces come and go — a cable is plugged in, a VPN starts —
            # so the list is rebuilt occasionally rather than once at boot.
            if time.time() - refreshed > 30:
                targets, refreshed = self.broadcast_targets(), time.time()

            beacon = json.dumps({
                "lmcluster": 1,
                "id": self.cfg.node_id,
                "name": self.cfg.name,
                "port": self.cfg.port,
                "fp": self.cfg.token_fingerprint,
                "caps": self.caps_fn(),
            }).encode()

            for target in targets:
                try:
                    sock.sendto(beacon, (target, self._udp_port))
                except OSError:
                    continue  # this interface is down; the others may not be

            # Anyone we have heard from also gets the beacon directly. This
            # makes discovery repair itself: if our broadcasts are not
            # reaching a machine but its broadcasts reach us, we now answer
            # it personally, and the two become visible to each other
            # without anybody touching a network setting.
            for peer in self.registry.snapshot():
                if not peer.get("online"):
                    continue
                try:
                    sock.sendto(beacon, (peer["ip"], self._udp_port))
                except OSError:
                    continue

            self._stop.wait(interval)

    def _listen_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", self._udp_port))
        except OSError as e:
            print(f"[discovery] cannot bind UDP {self._udp_port}: {e}")
            return
        sock.settimeout(1.0)
        while not self._stop.is_set():
            try:
                data, (ip, _) = sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                continue
            try:
                beacon = json.loads(data.decode())
            except (ValueError, UnicodeDecodeError):
                continue
            # Older versions used a different marker word. Accepting both
            # means a cluster still forms while you are part way through
            # upgrading the machines one at a time.
            if not (beacon.get("lmcluster") == 1 or beacon.get("council") == 1):
                continue
            if beacon.get("id") == self.cfg.node_id:
                continue  # our own echo
            fp = beacon.get("fp")
            if fp is not None and fp != self.cfg.token_fingerprint:
                continue  # different cluster sharing this LAN
            self.registry.update(beacon, ip)
