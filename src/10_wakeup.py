"""Step 10 - the idle wake-up cost, and how to avoid it.

    python 10_wakeup.py

A model that runs once per turn sits idle for seconds between calls. This measures the latency of
an e5-small-class encoder (3.7 ms when called back to back) after different idle gaps, and with
two keep-warm strategies running during the gap:

    npu-tick   a near-empty NPU task every 100 ms
    gpu-tick   a tiny CUDA memset every 100 ms (no NPU work)

It also samples nvidia-smi power for the NPU (device 1) and GPU (device 0) while each strategy idles.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from npu_common import ROOT
from npu_dla import DlaDevice, compile_loadable

RESULTS = ROOT / "results"


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def power() -> dict:
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,power.draw.instant", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout
    out = {}
    for line in q.splitlines():
        try:
            i, w = line.split(",")
            out["gpu_w" if i.strip() == "0" else "npu_w"] = float(w)
        except ValueError:
            pass
    return out


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    vm, tops = load("voice_models", "08_voice_models.py"), load("tops", "07_measure_tops.py")
    out = {"date": time.strftime("%Y-%m-%d %H:%M"), "power": tops.power_state(), "model": "encoder e5-small class, 64 tokens"}
    with DlaDevice() as dev:
        mod = dev.load(compile_loadable(vm.encoder(vm.F16, 12, 384, 12, 1536, 64, "conv")))
        tiny = dev.load(compile_loadable(tops.model(tops.Case("tick", 32, 32, 3, 1))))
        x = (np.random.default_rng(1).standard_normal(mod.inputs[0].shape) * 0.5).astype(np.float16)
        mod.write(0, x)
        scratch = dev.cuda.alloc(4096)

        def call() -> float:
            t = time.perf_counter()
            mod.run()
            return (time.perf_counter() - t) * 1e3

        def idle(seconds: float, strategy: str, watts: list | None = None) -> None:
            t_end = time.perf_counter() + seconds
            if strategy == "none":
                if watts is not None and seconds >= 1.5:
                    time.sleep(seconds / 2)
                    watts.append(power())
                time.sleep(max(0.0, t_end - time.perf_counter()))
                return
            n = 0
            while time.perf_counter() < t_end:
                if strategy == "npu-tick":
                    tiny.run()
                else:
                    dev.cuda.zero(scratch, 4096)
                    dev.cuda.sync()
                n += 1
                if watts is not None and n == int(seconds / 0.2):
                    watts.append(power())
                time.sleep(0.1)

        for _ in range(20):
            call()
        hot = sorted(call() for _ in range(200))
        out["back_to_back_ms"] = round(hot[len(hot) // 2], 2)
        print(f"back to back: {out['back_to_back_ms']} ms")

        out["idle_gap"] = {}
        for gap in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0):
            lat = []
            for _ in range(6):
                idle(gap, "none")
                lat.append(call())
            out["idle_gap"][str(gap)] = {"median_ms": round(float(np.median(lat)), 1), "max_ms": round(max(lat), 1)}
            print(f"idle {gap:4.2f} s, no keep-warm : median {np.median(lat):7.1f} ms  max {max(lat):7.1f} ms", flush=True)

        out["keep_warm"] = {}
        for strategy in ("none", "npu-tick", "gpu-tick"):
            lat, watts = [], []
            for _ in range(8):
                idle(3.0, strategy, watts)
                lat.append(call())
            w = {k: round(float(np.mean([s[k] for s in watts if k in s])), 2) for k in ("npu_w", "gpu_w")}
            out["keep_warm"][strategy] = {"median_ms": round(float(np.median(lat)), 1), "max_ms": round(max(lat), 1),
                                          "idle_power": w}
            print(f"3 s gaps, keep-warm={strategy:8}: median {np.median(lat):7.1f} ms  max {max(lat):7.1f} ms  "
                  f"idle power NPU {w.get('npu_w')} W, GPU {w.get('gpu_w')} W", flush=True)
        dev.cuda.free(scratch)
        tiny.close()
        mod.close()
    (RESULTS / "wakeup.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
