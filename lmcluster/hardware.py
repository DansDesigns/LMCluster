"""Hardware capability probing.

Nodes advertise what they can contribute so the shard planner can decide
where a model's layers should live. Everything here degrades gracefully:
a probe that cannot answer returns None rather than guessing, and the
planner treats None as "unknown, assume nothing".

No hard dependency on psutil — it is used when present and the stdlib
fallbacks cover Linux, Windows and macOS well enough for planning.
"""

import os
import platform
import re
import shutil
import subprocess
import time

_CACHE: dict = {}
_CACHE_TTL = 10.0


def _run(cmd, timeout=5) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


# -- memory ---------------------------------------------------------------

def ram_bytes() -> tuple[int | None, int | None]:
    """Return (total, available) system RAM in bytes."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.total, vm.available
    except ImportError:
        pass

    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo") as f:
                info = f.read()
            total = re.search(r"MemTotal:\s+(\d+)", info)
            avail = re.search(r"MemAvailable:\s+(\d+)", info)
            return (int(total.group(1)) * 1024 if total else None,
                    int(avail.group(1)) * 1024 if avail else None)
        except (OSError, AttributeError):
            return None, None

    if platform.system() == "Darwin":
        total_s = _run(["sysctl", "-n", "hw.memsize"])
        total = int(total_s.strip()) if total_s else None
        # vm_stat pages are 4096 bytes on every shipping Apple platform.
        vm = _run(["vm_stat"])
        avail = None
        if vm:
            free = re.search(r"Pages free:\s+(\d+)", vm)
            inactive = re.search(r"Pages inactive:\s+(\d+)", vm)
            if free and inactive:
                avail = (int(free.group(1)) + int(inactive.group(1))) * 4096
        return total, avail

    if platform.system() == "Windows":
        import ctypes

        class MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        stat = MemStatus()
        stat.dwLength = ctypes.sizeof(MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return stat.ullTotalPhys, stat.ullAvailPhys
        return None, None

    return None, None


# -- GPU ------------------------------------------------------------------

_INTEGRATED_HINTS = (
    "iris", "uhd graphics", "hd graphics", "intel(r) graphics",
    "radeon graphics", "radeon vega", "vega 3", "vega 5", "vega 6",
    "vega 7", "vega 8", "vega 11", "integrated",
)


def looks_integrated(name: str | None) -> bool:
    """Is this graphics chip sharing the machine's system memory?

    A heuristic on the device name, which is unreliable in principle and
    good enough in practice for the chips people actually have. The reason
    it matters is that integrated graphics have no memory of their own —
    they carve it out of system RAM — so adding one to the pool cannot
    increase how large a model the cluster can hold. Treating it like a
    discrete card would double-count the same gigabytes twice and produce a
    pool figure that is simply wrong.
    """
    if not name:
        return False
    lowered = name.lower()
    return any(hint in lowered for hint in _INTEGRATED_HINTS)


def detect_gpu() -> dict:
    """Identify the best available GPU backend.

    Returns {"backend": cuda|rocm|metal|vulkan|none, "name": str|None,
             "vram_total": int|None, "vram_free": int|None,
             "integrated": bool}

    Where the memory figures are left as None, that is deliberate and means
    "this device has no memory budget of its own", not "we failed to find
    out". The planner reads None as nothing to add.
    """
    out = {"backend": "none", "name": None, "vram_total": None,
           "vram_free": None, "integrated": False}

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        # Unified memory: the GPU draws on system RAM, so there is no
        # separate budget to add.
        out["backend"] = "metal"
        out["integrated"] = True
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        out["name"] = name.strip() if name else "Apple Silicon"
        return out

    smi = _run(["nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits"])
    if smi and smi.strip():
        first = smi.strip().splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        out["backend"] = "cuda"
        out["name"] = parts[0]
        try:
            out["vram_total"] = int(float(parts[1])) * 1024 * 1024
            out["vram_free"] = int(float(parts[2])) * 1024 * 1024
        except (ValueError, IndexError):
            pass
        return out

    if shutil.which("rocm-smi"):
        out["backend"] = "rocm"
        out["name"] = "AMD ROCm device"
        return out

    if shutil.which("vulkaninfo") or shutil.which("vulkaninfoSDK"):
        out["backend"] = "vulkan"
        out["name"] = _vulkan_device_name() or "Vulkan device"
        out["integrated"] = looks_integrated(out["name"])
        return out

    return out


def _vulkan_device_name() -> str | None:
    """The name of the first Vulkan device, if it can be had cheaply.

    Worth knowing rather than reporting a bare "Vulkan device", because
    whether it is a discrete card or the graphics built into the processor
    changes the advice entirely.
    """
    out = _run(["vulkaninfo", "--summary"], timeout=10)
    if not out:
        return None
    match = re.search(r"deviceName\s*=\s*(.+)", out)
    return match.group(1).strip() if match else None


# -- network link ---------------------------------------------------------
# Intermediate results cross the network on every single token, so the
# connection each machine is on matters a great deal. A machine on 2.4 GHz
# Wi-Fi will hold up everything else, and its owner usually has no idea,
# because as far as anyone can tell it "has Wi-Fi" and looks perfectly fine
# on the dashboard.

def _default_interface() -> str | None:
    """Interface carrying the default route."""
    system = platform.system()

    if system == "Linux":
        try:
            with open("/proc/net/route") as f:
                next(f)  # header
                for line in f:
                    parts = line.split()
                    # Destination 00000000 == default route
                    if len(parts) > 2 and parts[1] == "00000000":
                        return parts[0]
        except (OSError, StopIteration):
            return None
        return None

    if system == "Darwin":
        out = _run(["route", "-n", "get", "default"])
        if out:
            m = re.search(r"interface:\s*(\S+)", out)
            if m:
                return m.group(1)
        return None

    if system == "Windows":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "(Get-NetRoute -DestinationPrefix '0.0.0.0/0' | "
                    "Sort-Object RouteMetric | Select-Object -First 1)"
                    ".InterfaceAlias"], timeout=15)
        return out.strip() if out and out.strip() else None

    return None


def _linux_link(iface: str) -> dict:
    out = {"type": "ethernet", "speed_mbps": None, "band": None, "ssid": None}

    if os.path.isdir(f"/sys/class/net/{iface}/wireless") or iface.startswith(
            ("wl", "wlan", "wlp")):
        out["type"] = "wifi"
        link = _run(["iw", "dev", iface, "link"])
        if link:
            ssid = re.search(r"SSID:\s*(.+)", link)
            if ssid:
                out["ssid"] = ssid.group(1).strip()
            freq = re.search(r"freq:\s*(\d+)", link)
            if freq:
                out["band"] = _band_from_mhz(int(freq.group(1)))
            rate = re.search(r"tx bitrate:\s*([\d.]+)", link)
            if rate:
                out["speed_mbps"] = int(float(rate.group(1)))
        if out["band"] is None:
            iwc = _run(["iwconfig", iface])
            if iwc:
                # "Frequency:5.18 GHz"
                ghz = re.search(r"Frequency[:=]\s*([\d.]+)\s*GHz", iwc)
                if ghz:
                    out["band"] = _band_from_mhz(float(ghz.group(1)) * 1000)
        return out

    try:
        with open(f"/sys/class/net/{iface}/speed") as f:
            speed = int(f.read().strip())
            # -1 means "no carrier / unknown", not a real speed.
            out["speed_mbps"] = speed if speed > 0 else None
    except (OSError, ValueError):
        pass
    return out


def _windows_link(iface: str | None) -> dict:
    out = {"type": "unknown", "speed_mbps": None, "band": None, "ssid": None}

    wlan = _run(["netsh", "wlan", "show", "interfaces"], timeout=15)
    if wlan and re.search(r"State\s*:\s*connected", wlan, re.I):
        out["type"] = "wifi"
        ssid = re.search(r"^\s*SSID\s*:\s*(.+)$", wlan, re.M)
        if ssid:
            out["ssid"] = ssid.group(1).strip()
        band = re.search(r"Band\s*:\s*([\d.]+)\s*GHz", wlan, re.I)
        if band:
            out["band"] = _band_from_mhz(float(band.group(1)) * 1000)
        else:
            chan = re.search(r"Channel\s*:\s*(\d+)", wlan)
            if chan:
                out["band"] = _band_from_channel(int(chan.group(1)))
        rate = re.search(r"Receive rate \(Mbps\)\s*:\s*([\d.]+)", wlan)
        if rate:
            out["speed_mbps"] = int(float(rate.group(1)))
        return out

    if iface:
        speed = _run(["powershell", "-NoProfile", "-Command",
                      f"(Get-NetAdapter -Name '{iface}').LinkSpeed"],
                     timeout=15)
        if speed:
            m = re.search(r"([\d.]+)\s*(G|M)bps", speed, re.I)
            if m:
                value = float(m.group(1))
                out["speed_mbps"] = int(value * 1000 if m.group(2).upper() == "G"
                                        else value)
        out["type"] = "ethernet"
    return out


def _darwin_link(iface: str) -> dict:
    out = {"type": "ethernet", "speed_mbps": None, "band": None, "ssid": None}

    ports = _run(["networksetup", "-listallhardwareports"])
    is_wifi = False
    if ports:
        blocks = ports.split("Hardware Port:")
        for block in blocks:
            if f"Device: {iface}" in block and "Wi-Fi" in block:
                is_wifi = True
                break
    if not is_wifi:
        return out

    out["type"] = "wifi"
    # system_profiler is the one interface Apple has not removed. Slow
    # (~2s), which is why probe() caches.
    prof = _run(["system_profiler", "SPAirPortDataType"], timeout=12)
    if prof:
        chan = re.search(r"Channel:\s*(\d+)\s*\((\d)(?:\.\d)?\s*GHz", prof)
        if chan:
            out["band"] = _band_from_mhz(int(chan.group(2)) * 1000)
        else:
            chan2 = re.search(r"Channel:\s*(\d+)", prof)
            if chan2:
                out["band"] = _band_from_channel(int(chan2.group(1)))
        ssid = re.search(r"Current Network Information:\s*\n\s*(.+?):", prof)
        if ssid:
            out["ssid"] = ssid.group(1).strip()
        rate = re.search(r"Transmit Rate:\s*(\d+)", prof)
        if rate:
            out["speed_mbps"] = int(rate.group(1))
    return out


def _band_from_mhz(mhz: float) -> str:
    if mhz < 3000:
        return "2.4GHz"
    if mhz < 5900:
        return "5GHz"
    return "6GHz"


def _band_from_channel(channel: int) -> str | None:
    if 1 <= channel <= 14:
        return "2.4GHz"
    if 32 <= channel <= 177:
        return "5GHz"
    return None


def link_warnings(link: dict) -> list[dict]:
    """Advice about this node's link, worst first.

    Severity drives colour in the dashboard: 'bad' means this node will
    visibly hold up the pipeline, 'warn' means it is workable but not what
    you want, 'info' is a nudge.
    """
    out = []
    kind, band = link.get("type"), link.get("band")
    speed = link.get("speed_mbps")

    if kind == "wifi":
        if band == "2.4GHz":
            out.append({
                "severity": "bad",
                "message": "On 2.4 GHz Wi-Fi. Shared with every microwave "
                           "and neighbour on the channel, and typically "
                           "10-50 Mbps in practice.",
                "fix": "Switch this node to your 5 GHz SSID, or plug it in. "
                       "Ethernet is worth more here than any other change.",
            })
        elif band in ("5GHz", "6GHz"):
            out.append({
                "severity": "info",
                "message": f"On {band} Wi-Fi. Workable for shard mode.",
                "fix": "Ethernet is still steadier — Wi-Fi latency varies "
                       "with interference, and every token pays it.",
            })
        else:
            out.append({
                "severity": "warn",
                "message": "On Wi-Fi, band unknown.",
                "fix": "Make sure this node is on the 5 GHz SSID rather than "
                       "2.4 GHz. Many routers publish both under one name, "
                       "in which case the node may silently pick 2.4.",
            })
        if speed is not None and speed < 100:
            out.append({
                "severity": "bad",
                "message": f"Wi-Fi negotiated only {speed} Mbps.",
                "fix": "Move the node closer to the access point, or use "
                       "ethernet.",
            })

    elif kind == "ethernet":
        if speed is not None and speed <= 100:
            out.append({
                "severity": "warn",
                "message": f"Ethernet linked at {speed} Mbps, not 1000.",
                "fix": "Usually an old cable (Cat5 not Cat5e), a 100 Mbps "
                       "switch port, or a bad crimp. Swapping the cable "
                       "fixes it most of the time.",
            })
        elif speed is not None and speed >= 1000:
            out.append({
                "severity": "good",
                "message": f"Ethernet at {speed} Mbps.",
                "fix": "",
            })

    return out


def network_link() -> dict:
    iface = _default_interface()
    system = platform.system()

    if iface is None and system != "Windows":
        link = {"interface": None, "type": "unknown", "speed_mbps": None,
                "band": None, "ssid": None}
    elif system == "Linux":
        link = {"interface": iface, **_linux_link(iface)}
    elif system == "Darwin":
        link = {"interface": iface, **_darwin_link(iface)}
    elif system == "Windows":
        link = {"interface": iface, **_windows_link(iface)}
    else:
        link = {"interface": iface, "type": "unknown", "speed_mbps": None,
                "band": None, "ssid": None}

    link["warnings"] = link_warnings(link)
    return link


# -- disk -----------------------------------------------------------------

def disk_free(path: str | None = None) -> int | None:
    try:
        return shutil.disk_usage(path or os.getcwd()).free
    except OSError:
        return None


# -- aggregate ------------------------------------------------------------

def probe(model_dir: str | None = None) -> dict:
    """Full capability snapshot, cached briefly so beacon and dashboard
    polling do not re-shell out several times a second."""
    now = time.time()
    cached = _CACHE.get("probe")
    if cached and now - cached[0] < _CACHE_TTL:
        return cached[1]

    total, avail = ram_bytes()
    gpu = detect_gpu()
    link = network_link()
    snap = {
        "os": platform.system(),
        "arch": platform.machine(),
        "cpu_count": os.cpu_count(),
        "ram_total": total,
        "ram_free": avail,
        "gpu_backend": gpu["backend"],
        "gpu_name": gpu["name"],
        "gpu_integrated": gpu.get("integrated", False),
        "vram_total": gpu["vram_total"],
        "vram_free": gpu["vram_free"],
        "disk_free": disk_free(model_dir),
        "link": link,
    }
    _CACHE["probe"] = (now, snap)
    return snap


def gb(value: int | None) -> float | None:
    """Bytes to GB, one decimal. None passes through."""
    return None if value is None else round(value / 1e9, 1)
