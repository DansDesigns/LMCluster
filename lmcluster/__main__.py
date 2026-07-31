"""Entry point: python -m lmcluster [config.toml]

Starts the node and, unless disabled in config, opens the dashboard in the
default browser once the server is listening.

The config path, when not given, is resolved against the project directory
rather than the current one. It used to be the bare relative name
"lmcluster.toml", which works when started through run.sh — that changes
directory first — and silently does not when started any other way. A
desktop autostart entry, a systemd unit, or simply running the command from
your home directory would find no config, fall back to built-in defaults,
and come up with sharding switched off. Nothing looked wrong locally; the
machine just told everyone else it had no RPC support.
"""

import os
import socket
import sys
import threading
import time
import webbrowser

import uvicorn

from .config import Config
from .node import create_app


def _open_dashboard(port: int):
    """Wait for the server to accept connections, then open the browser.

    Failures are silent by design: headless machines (no display, no
    browser) just won't open anything, and the node keeps running.
    """
    url = f"http://127.0.0.1:{port}"
    for _ in range(40):  # up to ~10s
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.25)
    else:
        return
    try:
        webbrowser.open(url)
    except Exception:
        pass


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_config_path() -> str:
    """Where to look for lmcluster.toml when not told.

    Prefers the current directory if there is one there, so working on a
    copy still behaves as expected, and otherwise uses the one beside the
    code.
    """
    here = os.path.abspath("lmcluster.toml")
    if os.path.exists(here):
        return here
    return os.path.join(ROOT, "lmcluster.toml")


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else default_config_path()
    config = Config(cfg_path)
    app = create_app(config)
    if config.data["node"].get("open_browser", True):
        threading.Thread(target=_open_dashboard, args=(config.port,),
                         daemon=True).start()
    print(f"[node] config: {os.path.abspath(config.path)}")
    print(f"[node] dashboard: http://127.0.0.1:{config.port}")
    uvicorn.run(app, host="0.0.0.0", port=config.port, log_level="warning")


if __name__ == "__main__":
    main()
