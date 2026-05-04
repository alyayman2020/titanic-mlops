"""
System metrics collector — CPU, RAM, GPU.
Logged as MLflow tags/params at the start of every run.
"""
from __future__ import annotations

import platform
import time
from typing import Any

import psutil

try:
    import nvidia_ml_py as pynvml  # nvidia-ml-py (replaces deprecated pynvml)

    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

try:
    import GPUtil

    _GPUTIL_AVAILABLE = True
except Exception:
    _GPUTIL_AVAILABLE = False


def get_system_info() -> dict[str, Any]:
    """Return a flat dict of system info suitable for MLflow logging."""
    info: dict[str, Any] = {}

    # ── Platform ──────────────────────────────────────────────────
    info["sys.platform"] = platform.system()
    info["sys.python_version"] = platform.python_version()
    info["sys.cpu_count_logical"] = psutil.cpu_count(logical=True)
    info["sys.cpu_count_physical"] = psutil.cpu_count(logical=False)

    # ── RAM ───────────────────────────────────────────────────────
    vm = psutil.virtual_memory()
    info["sys.ram_total_gb"] = round(vm.total / 1e9, 2)
    info["sys.ram_available_gb"] = round(vm.available / 1e9, 2)

    # ── GPU (pynvml) ──────────────────────────────────────────────
    if _NVML_AVAILABLE:
        try:
            gpu_count = pynvml.nvmlDeviceGetCount()
            info["sys.gpu_count"] = gpu_count
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info[f"sys.gpu_{i}_name"] = pynvml.nvmlDeviceGetName(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                info[f"sys.gpu_{i}_vram_total_gb"] = round(mem.total / 1e9, 2)
                info[f"sys.gpu_{i}_vram_free_gb"] = round(mem.free / 1e9, 2)
                info[f"sys.gpu_{i}_driver"] = pynvml.nvmlSystemGetDriverVersion()
        except Exception as e:
            info["sys.gpu_error"] = str(e)
    elif _GPUTIL_AVAILABLE:
        try:
            gpus = GPUtil.getGPUs()
            info["sys.gpu_count"] = len(gpus)
            for g in gpus:
                info[f"sys.gpu_{g.id}_name"] = g.name
                info[f"sys.gpu_{g.id}_vram_total_gb"] = round(g.memoryTotal / 1024, 2)
                info[f"sys.gpu_{g.id}_vram_free_gb"] = round(g.memoryFree / 1024, 2)
        except Exception as e:
            info["sys.gpu_error"] = str(e)
    else:
        info["sys.gpu_count"] = 0
        info["sys.gpu_available"] = False

    return info


class RuntimeTimer:
    """Simple wall-clock timer."""

    def __init__(self):
        self._start = None

    def start(self):
        self._start = time.perf_counter()
        return self

    def elapsed(self) -> float:
        return round(time.perf_counter() - self._start, 3)

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        pass
