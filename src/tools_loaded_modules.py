"""Show which DLLs are actually in the process once a task has run on the NPU.

    python tools_loaded_modules.py

Evidence for the software-stack diagram: prints every NVIDIA / CUDA / DLA module loaded into
this process after a cuDLA task ran, with the folder it was loaded from.
"""
from __future__ import annotations

import ctypes as C
import ctypes.wintypes as W
from pathlib import Path

from npu_common import ROOT
from npu_dla import DlaDevice


def loaded_modules() -> list[str]:
    psapi, k32 = C.WinDLL("psapi"), C.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = W.HANDLE
    psapi.EnumProcessModulesEx.argtypes = [W.HANDLE, C.POINTER(W.HMODULE), W.DWORD, C.POINTER(W.DWORD), W.DWORD]
    psapi.GetModuleFileNameExW.argtypes = [W.HANDLE, W.HMODULE, W.LPWSTR, W.DWORD]
    proc, needed = k32.GetCurrentProcess(), W.DWORD()
    arr = (W.HMODULE * 2048)()
    psapi.EnumProcessModulesEx(proc, arr, C.sizeof(arr), C.byref(needed), 3)
    out = []
    for h in arr[: needed.value // C.sizeof(W.HMODULE)]:
        buf = C.create_unicode_buffer(1024)
        psapi.GetModuleFileNameExW(proc, h, buf, 1024)
        out.append(buf.value)
    return out


def main() -> int:
    loadable = ROOT / "models" / "convnet_fp16_io16.dla.loadable"
    before = set(loaded_modules())
    with DlaDevice() as dev:
        after_device = set(loaded_modules())
        mod = dev.load(loadable.read_bytes())
        mod.run()
        after_run = set(loaded_modules())
        mod.close()

    def show(title: str, mods: set[str]) -> None:
        print(title)
        for m in sorted(mods, key=lambda p: Path(p).name.lower()):
            p = Path(m)
            if any(s in p.name.lower() for s in ("nv", "cud", "dla")) or "DriverStore" in m:
                where = ("driver store" if "DriverStore" in m else "System32" if "System32" in m or "system32" in m
                         else str(p.parent.relative_to(ROOT)) if ROOT in p.parents else str(p.parent))
                print(f"  {p.name:28} {p.stat().st_size / 1e6:7.1f} MB   {where}")

    show("loaded by cudlaCreateDevice (and CUDA init):", after_device - before)
    show("additionally loaded by module load + first task:", after_run - after_device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
