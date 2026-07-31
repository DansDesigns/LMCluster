"""Opening the ports a cluster needs, without making anyone read a manual.

Three ports have to be reachable on every machine:

  8470/tcp   the dashboard and the API one machine uses to ask another
             to do something
  8471/udp   the announcements machines send so they can find each other
  50052/tcp  the RPC server, through which a machine lends its memory

A blocked port produces symptoms that point in entirely the wrong
direction. A machine whose UDP is blocked simply never appears, so it looks
switched off. A machine whose RPC port is blocked appears perfectly healthy
and contributes nothing, and the connection times out rather than being
refused — which is why the pool page distinguishes those two cases.

Changing firewall rules needs administrator rights, and expecting somebody
to find their way to the right settings dialogue, on three machines, one of
which may have no screen, is not a reasonable thing to ask. So this builds
the exact commands and then asks the operating system to run them with
elevation: a UAC prompt on Windows, a password prompt on Linux.

Nothing here runs without the person agreeing to it at that prompt. If they
decline, the commands are shown so they can be pasted somewhere else, which
is the answer for a headless machine reached over SSH.
"""

import os
import platform
import shutil
import subprocess
import tempfile

RULE_PREFIX = "LMCluster"


def ports(config) -> list[tuple[str, int, str]]:
    """(label, port, protocol) for everything that must be reachable."""
    return [
        ("dashboard and API", int(config.port), "tcp"),
        ("finding other machines", int(config.discovery["port"]), "udp"),
        ("lending memory", int(config.shard.get("rpc_port", 50052)), "tcp"),
    ]


def _run(cmd, timeout=30) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, check=False)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.SubprocessError) as e:
        return 1, str(e)


# -- what kind of firewall is this? --------------------------------------

def detect() -> dict:
    """Which firewall is in charge here, and can we drive it?"""
    system = platform.system()

    if system == "Windows":
        return {"kind": "windows", "name": "Windows Defender Firewall",
                "manageable": True}

    if system == "Darwin":
        # The macOS application firewall filters by program rather than by
        # port, and a program the user has run is normally allowed already.
        return {"kind": "macos", "name": "macOS application firewall",
                "manageable": False}

    if shutil.which("ufw"):
        code, out = _run(["ufw", "status"])
        inactive = "inactive" in out.lower()
        return {"kind": "ufw", "name": "ufw", "manageable": True,
                "inactive": inactive}
    if shutil.which("firewall-cmd"):
        code, out = _run(["firewall-cmd", "--state"])
        return {"kind": "firewalld", "name": "firewalld", "manageable": True,
                "inactive": "running" not in out}
    if shutil.which("nft"):
        return {"kind": "nft", "name": "nftables", "manageable": False}
    if shutil.which("iptables"):
        return {"kind": "iptables", "name": "iptables", "manageable": True}

    return {"kind": "none", "name": "no firewall found", "manageable": False}


# -- building the commands -----------------------------------------------

def commands(config) -> list[str]:
    """The exact commands that would be run, as text.

    Returned so they can be shown to somebody before anything happens, and
    so a headless machine can be dealt with by pasting them over SSH.
    """
    kind = detect()["kind"]
    out = []

    for label, port, proto in ports(config):
        name = f"{RULE_PREFIX} {label}"
        if kind == "windows":
            # Every profile, not just private and domain. Windows decides
            # for itself whether a network is Public or Private, gets it
            # wrong often, and a rule that does not cover the profile in
            # use does nothing at all — while looking, in the firewall
            # settings, exactly like a rule that is working. That failure
            # is invisible from the machine itself and shows up only as
            # another machine being unable to reach it.
            out.append(
                f'netsh advfirewall firewall add rule name="{name}" '
                f'dir=in action=allow protocol={proto.upper()} '
                f'localport={port} profile=any')
        elif kind == "ufw":
            out.append(f"ufw allow {port}/{proto} comment '{name}'")
        elif kind == "firewalld":
            out.append(f"firewall-cmd --permanent --add-port={port}/{proto}")
        elif kind == "iptables":
            out.append(f"iptables -I INPUT -p {proto} --dport {port} "
                       f"-j ACCEPT")

    if kind == "firewalld":
        out.append("firewall-cmd --reload")
    return out


def _windows_script(config) -> str:
    lines = ["@echo off",
             "echo Opening the ports LMCluster needs...",
             "echo."]
    for label, port, proto in ports(config):
        name = f"{RULE_PREFIX} {label}"
        # Removed first so re-running does not pile up duplicate rules.
        lines.append(f'netsh advfirewall firewall delete rule '
                     f'name="{name}" >nul 2>&1')
        lines.append(
            f'netsh advfirewall firewall add rule name="{name}" '
            f'dir=in action=allow protocol={proto.upper()} '
            f'localport={port} profile=any')
    lines += ["echo.", "echo Done.", "timeout /t 3 >nul"]
    return "\r\n".join(lines)


def _posix_script(config) -> str:
    lines = ["#!/bin/sh", "set -e"]
    lines += commands(config)
    return "\n".join(lines) + "\n"


# -- applying them --------------------------------------------------------

def is_admin() -> bool:
    if os.name == "nt":
        try:
            import ctypes
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return False
    return os.geteuid() == 0


def apply(config) -> dict:
    """Open the ports, asking the operating system to elevate if needed.

    Returns {"ok": bool, "message": str, "commands": [...]}. The commands
    are included whatever happens, so that somebody who cannot use the
    prompt — over SSH, say — has something to paste.
    """
    info = detect()
    cmds = commands(config)

    if not info["manageable"]:
        return {"ok": False, "commands": cmds,
                "message": f"{info['name']} cannot be configured "
                           f"automatically. "
                           + ("On macOS, allow llama.cpp and Python through "
                              "the firewall in System Settings when asked."
                              if info["kind"] == "macos" else
                              "Open the ports listed below by hand.")}

    if info.get("inactive"):
        return {"ok": True, "commands": cmds,
                "message": f"{info['name']} is not switched on, so nothing "
                           f"is being blocked and there is nothing to do."}

    if os.name == "nt":
        return _apply_windows(config, cmds)
    return _apply_posix(config, cmds, info)


def _apply_windows(config, cmds) -> dict:
    """Write a script and ask Windows to run it as administrator.

    ShellExecuteW with the "runas" verb is what raises the UAC prompt. It
    cannot be done by simply calling netsh, because a process cannot grant
    itself administrator rights — a new one has to be started with them.
    """
    script = os.path.join(tempfile.gettempdir(), "lmcluster-firewall.bat")
    try:
        with open(script, "w", encoding="utf-8") as f:
            f.write(_windows_script(config))
    except OSError as e:
        return {"ok": False, "commands": cmds,
                "message": f"could not write the script: {e}"}

    if is_admin():
        code, out = _run(["cmd", "/c", script], timeout=90)
        return {"ok": code == 0, "commands": cmds,
                "message": "Ports opened." if code == 0
                           else f"the commands failed: {out[:300]}"}

    try:
        import ctypes
        rc = ctypes.windll.shell32.ShellExecuteW(
            None, "runas", "cmd.exe", f'/c "{script}"', None, 1)
    except Exception as e:
        return {"ok": False, "commands": cmds,
                "message": f"could not ask Windows for permission: {e}"}

    # Anything above 32 means it started. 5 is the specific code for the
    # user pressing No, which deserves saying plainly rather than being
    # reported as a failure.
    if rc == 5:
        return {"ok": False, "commands": cmds,
                "message": "You declined the administrator prompt, so "
                           "nothing was changed."}
    if rc <= 32:
        return {"ok": False, "commands": cmds,
                "message": f"Windows would not start the script (code {rc})."}
    return {"ok": True, "commands": cmds,
            "message": "Accept the administrator prompt that has appeared. "
                       "The ports will be open a moment later."}


def _apply_posix(config, cmds, info) -> dict:
    script = os.path.join(tempfile.gettempdir(), "lmcluster-firewall.sh")
    try:
        with open(script, "w", encoding="utf-8") as f:
            f.write(_posix_script(config))
        os.chmod(script, 0o755)
    except OSError as e:
        return {"ok": False, "commands": cmds,
                "message": f"could not write the script: {e}"}

    if is_admin():
        code, out = _run(["sh", script], timeout=90)
        return {"ok": code == 0, "commands": cmds,
                "message": "Ports opened." if code == 0
                           else f"the commands failed: {out[:300]}"}

    # pkexec raises a graphical password prompt on a desktop. On a headless
    # machine it has nothing to draw on and fails immediately, which is why
    # the commands come back in the reply for pasting over SSH.
    if shutil.which("pkexec") and os.environ.get("DISPLAY"):
        code, out = _run(["pkexec", "sh", script], timeout=120)
        if code == 0:
            return {"ok": True, "commands": cmds, "message": "Ports opened."}
        if code == 126:
            return {"ok": False, "commands": cmds,
                    "message": "You dismissed the password prompt, so "
                               "nothing was changed."}

    # sudo without a password, which is how a lot of single-user machines
    # are set up. -n means it fails immediately rather than sitting there
    # waiting for a password nobody can type into a web page.
    if shutil.which("sudo"):
        code, out = _run(["sudo", "-n", "sh", script], timeout=90)
        if code == 0:
            return {"ok": True, "commands": cmds, "message": "Ports opened."}

    return {"ok": False, "commands": cmds,
            "message": "This needs a password, and a web page is not the "
                       "place to type one. Run this in a terminal on that "
                       f"machine:\n    sudo sh {script}\n"
                       "Or paste the commands below."}


# -- checking ------------------------------------------------------------

def network_profile() -> str | None:
    """Which profile Windows has assigned to the active network.

    Worth surfacing because it explains a whole class of confusion. If
    Windows has decided a home network is Public, the machine hides itself
    much more aggressively, and rules written for Private networks have no
    effect whatsoever.
    """
    if platform.system() != "Windows":
        return None
    code, out = _run(["powershell", "-NoProfile", "-Command",
                      "(Get-NetConnectionProfile).NetworkCategory"],
                     timeout=25)
    if code != 0 or not out.strip():
        return None
    return out.strip().splitlines()[0].strip()


def status(config) -> dict:
    """Are the rules already in place?

    Only checked where it can be done without administrator rights, which
    rules out iptables. Where we cannot tell, that is what gets reported,
    rather than a guess dressed up as a fact.
    """
    info = detect()
    wanted = ports(config)
    profile = network_profile()
    out = {"firewall": info["name"], "kind": info["kind"],
           "manageable": info["manageable"],
           "inactive": bool(info.get("inactive")),
           "network_profile": profile,
           "profile_warning": (
               "Windows has classified this network as Public, which makes "
               "the machine hide from others on it. Setting it to Private "
               "in Windows network settings is worth doing regardless of "
               "the rules below."
               if profile and profile.lower() == "public" else None),
           "ports": [{"label": l, "port": p, "protocol": proto,
                      "open": None} for l, p, proto in wanted]}

    if info.get("inactive") or info["kind"] == "none":
        for entry in out["ports"]:
            entry["open"] = True
        return out

    if info["kind"] == "windows":
        code, text = _run(["netsh", "advfirewall", "firewall", "show", "rule",
                           f"name=all"], timeout=45)
        if code == 0:
            for entry in out["ports"]:
                entry["open"] = f"{RULE_PREFIX} {entry['label']}" in text
    elif info["kind"] == "ufw":
        code, text = _run(["ufw", "status"])
        if code == 0:
            for entry in out["ports"]:
                entry["open"] = (f"{entry['port']}/{entry['protocol']}"
                                 in text)
    elif info["kind"] == "firewalld":
        code, text = _run(["firewall-cmd", "--list-ports"])
        if code == 0:
            for entry in out["ports"]:
                entry["open"] = (f"{entry['port']}/{entry['protocol']}"
                                 in text)

    known = [e["open"] for e in out["ports"] if e["open"] is not None]
    out["all_open"] = bool(known) and all(known)
    out["unknown"] = not known
    return out
