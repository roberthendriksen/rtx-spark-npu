"""Step 9 - does a busy GPU slow the NPU down, and the other way round?

    python 09_gpu_contention.py

The NPU only works in cuDLA "hybrid" mode: tasks are submitted on a CUDA stream and tensors live
in CUDA memory. So the question is whether NPU latency survives another process saturating the GPU.

A child process (tools_gpu_load.py, PyTorch FP16 matmuls in .venv-gpuload) keeps the GPU busy.
The parent times three transformer encoders on the NPU - e5-small and e5-base class (one loadable
each) and e5-large class (two chained loadables) - alone and under that load, called back to back and
called once every 2 s (how a once-per-turn model is really used).
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from npu_common import ROOT
from npu_dla import DlaDevice, compile_loadable

RESULTS = ROOT / "results"
GPU_PY = ROOT / ".venv-gpuload" / "Scripts" / "python.exe"


def voice_models():
    spec = importlib.util.spec_from_file_location("voice_models", Path(__file__).with_name("08_voice_models.py"))
    vm = importlib.util.module_from_spec(spec)
    sys.modules["voice_models"] = vm
    spec.loader.exec_module(vm)
    return vm


def stats(lat: list[float]) -> dict:
    a = np.array(lat) * 1e3
    return {"n": len(lat), "median_ms": round(float(np.median(a)), 2), "p95_ms": round(float(np.percentile(a, 95)), 2),
            "max_ms": round(float(a.max()), 2)}


def back_to_back(run, seconds: float) -> dict:
    lat, t_end = [], time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        t = time.perf_counter()
        run()
        lat.append(time.perf_counter() - t)
    return stats(lat)


def spaced(run, calls: int, period: float) -> dict:
    lat = []
    for _ in range(calls):
        time.sleep(period)
        t = time.perf_counter()
        run()
        lat.append(time.perf_counter() - t)
    return stats(lat)


def main() -> int:
    if not GPU_PY.exists():
        raise SystemExit(f"missing {GPU_PY}: create it with\n  uv venv --python 3.13 .venv-gpuload\n"
                         "  uv pip install --python .venv-gpuload/Scripts/python.exe torch==2.15.0+cu134 "
                         "--extra-index-url https://pypi.nvidia.com/nvtorch_oot/ --index-strategy unsafe-best-match")
    RESULTS.mkdir(exist_ok=True)
    vm = voice_models()
    out = {"date": time.strftime("%Y-%m-%d %H:%M"),
           "gpu_load": "PyTorch FP16 6144x6144 matmul loop in a second process"}
    with DlaDevice() as dev:
        small = dev.load(compile_loadable(vm.encoder(vm.F16, 12, 384, 12, 1536, 64, "conv")))
        base = dev.load(compile_loadable(vm.encoder(vm.F16, 12, 768, 12, 3072, 64, "conv")))
        halves = [dev.load(compile_loadable(vm.encoder(vm.F16, 12, 1024, 16, 4096, 64, "conv", pool=False, seed=0))),
                  dev.load(compile_loadable(vm.encoder(vm.F16, 12, 1024, 16, 4096, 64, "conv", pool=True, seed=1)))]
        xs = (np.random.default_rng(1).standard_normal(small.inputs[0].shape) * 0.5).astype(np.float16)
        xb = (np.random.default_rng(1).standard_normal(base.inputs[0].shape) * 0.5).astype(np.float16)
        xl = (np.random.default_rng(1).standard_normal(halves[0].inputs[0].shape) * 0.5).astype(np.float16)

        def run_small():
            small.write(0, xs)
            small.run()
            return small.read(0)

        def run_base():
            base.write(0, xb)
            base.run()
            return base.read(0)

        def run_large():
            halves[0].write(0, xl)
            halves[0].run()
            halves[1].write(0, halves[0].read(0))
            halves[1].run()
            return halves[1].read(0)

        models = {"e5small_64tok": run_small, "e5base_64tok": run_base, "e5large_64tok_split2": run_large}
        for run in models.values():
            for _ in range(5):
                run()

        def measure(tag: str) -> None:
            for name, run in models.items():
                out[f"{name}.{tag}.back_to_back"] = back_to_back(run, 4)
                out[f"{name}.{tag}.every_2s"] = spaced(run, 6, 2.0)
                print(f"{name:22} {tag:14} back to back {out[f'{name}.{tag}.back_to_back']} | "
                      f"every 2 s {out[f'{name}.{tag}.every_2s']}", flush=True)

        measure("gpu_idle")

        child = subprocess.Popen([str(GPU_PY), str(Path(__file__).with_name("tools_gpu_load.py")), "120"],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        ips: list[tuple[float, float]] = []
        ready = threading.Event()

        def reader():
            for line in child.stdout:
                if line.startswith("READY"):
                    out["gpu_load_env"] = line.strip()[6:]
                    ready.set()
                elif line.startswith("IPS"):
                    _, ts, v = line.split()
                    ips.append((float(ts), float(v)))

        threading.Thread(target=reader, daemon=True).start()
        if not ready.wait(180):
            child.kill()
            raise SystemExit("GPU load process never became ready: " + child.stderr.read()[-800:])
        time.sleep(8)                                    # GPU load alone
        t_a = time.time()
        measure("gpu_saturated")
        t_b = time.time()
        child.terminate()
        for m in [small, base, *halves]:
            m.close()
    alone = [v for ts, v in ips if ts <= t_a][1:]
    during = [v for ts, v in ips if t_a + 1 < ts <= t_b]
    out["gpu_load_ips_alone"] = round(float(np.mean(alone)), 2) if alone else None
    out["gpu_load_ips_while_npu_measured"] = round(float(np.mean(during)), 2) if during else None
    out["gpu_load_ips_min_while_npu_measured"] = round(float(np.min(during)), 2) if during else None
    print(f"GPU load: {out['gpu_load_ips_alone']} it/s alone, {out['gpu_load_ips_while_npu_measured']} it/s "
          f"(min {out['gpu_load_ips_min_while_npu_measured']}) while the NPU was being measured")
    (RESULTS / "gpu_contention.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
