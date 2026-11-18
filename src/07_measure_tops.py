"""Step 7 - measure how many operations per second the NPU really delivers.

    python 07_measure_tops.py [--quick] [--only NAME] [--seconds 4]

Method: build convolution stacks whose multiply-accumulate (MAC) count is known exactly, run
them on the NPU through raw cuDLA and divide:

    ops per inference = 2 x MACs      (one multiply + one add per MAC, the usual TOPS convention)
    TOPS              = ops per inference x inferences per second / 1e12

Everything the supported stack runs is FP16, so these are FP16 operations ("TFLOPS").
Results are written to results/tops.json and results/tops.md.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from npu_common import ROOT
from npu_dla import DlaBuildError, DlaDevice, compile_loadable

RESULTS = ROOT / "results"


@dataclass
class Case:
    name: str
    c: int            # channels in and out of every layer
    hw: int           # square feature map (multiple of 32 keeps DLA_LINEAR rows unpadded)
    k: int            # kernel size
    depth: int        # number of conv+ReLU layers
    groups: int = 1
    sparsity: str = "dense"   # dense | unstructured50 | unstructured90 | struct2of4 | prunedhalf

    @property
    def macs(self) -> int:
        return self.depth * self.hw * self.hw * self.c * (self.c // self.groups) * self.k * self.k


def weights(case: Case, rng: np.random.Generator) -> np.ndarray:
    cin = case.c // case.groups
    w = rng.standard_normal((case.c, cin, case.k, case.k)) * np.sqrt(2.0 / (cin * case.k * case.k))
    if case.sparsity.startswith("unstructured"):
        keep = 1 - int(case.sparsity[len("unstructured"):]) / 100
        w *= rng.random(w.shape) < keep
    elif case.sparsity == "struct2of4":      # 2 of every 4 consecutive input channels are zero
        mask = np.zeros(cin, dtype=bool)
        mask[0::4] = mask[1::4] = True
        w *= mask[None, :, None, None]
    elif case.sparsity == "prunedhalf":      # half of the output kernels are entirely zero
        w[case.c // 2:] = 0
    return w.astype(np.float16)


def model(case: Case, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    nodes, inits, cur = [], [], "input"
    for i in range(case.depth):
        inits.append(numpy_helper.from_array(weights(case, rng), f"w{i}"))
        inits.append(numpy_helper.from_array((rng.standard_normal(case.c) * 0.01).astype(np.float16), f"b{i}"))
        nodes.append(helper.make_node("Conv", [cur, f"w{i}", f"b{i}"], [f"c{i}"], kernel_shape=[case.k] * 2,
                                      pads=[case.k // 2] * 4, strides=[1, 1], group=case.groups, name=f"conv{i}"))
        cur = "output" if i == case.depth - 1 else f"r{i}"
        nodes.append(helper.make_node("Relu", [f"c{i}"], [cur], name=f"relu{i}"))
    shape = [1, case.c, case.hw, case.hw]
    graph = helper.make_graph(nodes, case.name,
                              [helper.make_tensor_value_info("input", TensorProto.FLOAT16, shape)],
                              [helper.make_tensor_value_info("output", TensorProto.FLOAT16, shape)], inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    m.ir_version = 10
    return m.SerializeToString()


class Sampler(threading.Thread):
    """Samples the NPU (nvidia-smi device 1) while a case runs."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.rows: list[tuple[float, float, float]] = []
        self.proc = subprocess.Popen(
            ["nvidia-smi", "-i", "1", "--query-gpu=utilization.gpu,power.draw.instant,temperature.gpu",
             "--format=csv,noheader,nounits", "-lms", "250"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

        self.clocks: list[int] = []
        self.max_clock = 0
        self._stop = threading.Event()
        self._clock_thread = threading.Thread(target=self._poll_clock, daemon=True)
        self._clock_thread.start()

    def _poll_clock(self) -> None:
        """The DLA engine clock is only in `nvidia-smi -q` ("DLA Clocks" section), not in --query-gpu."""
        while not self._stop.is_set():
            try:
                q = subprocess.run(["nvidia-smi", "-i", "1", "-q"], capture_output=True, text=True, timeout=10).stdout
                m = re.search(r"DLA Clocks\s+Engine\s+:\s+(\d+) MHz\s+Max Engine\s+:\s+(\d+) MHz", q)
                if m:
                    self.clocks.append(int(m.group(1)))
                    self.max_clock = int(m.group(2))
            except (subprocess.SubprocessError, OSError):
                pass
            self._stop.wait(0.4)

    def run(self) -> None:
        for line in self.proc.stdout:
            try:
                self.rows.append(tuple(float(p) for p in line.split(",")))
            except ValueError:
                pass

    def stop(self) -> dict:
        self._stop.set()
        self.proc.terminate()
        self.join(timeout=2)
        self._clock_thread.join(timeout=12)
        out = {}
        if self.rows:
            a = np.array(self.rows)
            out.update(util_pct=round(float(a[:, 0].mean()), 1), power_w=round(float(a[:, 1].mean()), 2),
                       power_w_max=round(float(a[:, 1].max()), 2), temp_c_max=round(float(a[:, 2].max()), 1))
        if self.clocks:
            out.update(clock_mhz=round(float(np.mean(self.clocks))), clock_mhz_peak=max(self.clocks),
                       clock_mhz_limit=self.max_clock)
        return out


def measure(dev: DlaDevice, case: Case, seconds: float) -> dict:
    out = {**asdict(case), "gmacs": round(case.macs / 1e9, 3)}
    t0 = time.perf_counter()
    try:
        loadable = compile_loadable(model(case))
    except DlaBuildError as e:
        return {**out, "status": "build failed", "error": str(e)[:300]}
    out["compile_s"] = round(time.perf_counter() - t0, 1)
    out["loadable_mb"] = round(len(loadable) / 1e6, 1)
    try:
        mod = dev.load(loadable)
    except RuntimeError as e:
        return {**out, "status": "load failed", "error": str(e)}
    try:
        x = np.random.default_rng(1).standard_normal(mod.inputs[0].shape).astype(np.float16)
        mod.write(0, x)
        for _ in range(5):
            mod.run()
        y = mod.read(0).astype(np.float32)
        out["output_finite"] = bool(np.isfinite(y).all())
        out["output_nonzero_pct"] = round(float(np.count_nonzero(y)) / y.size * 100, 1)

        sampler = Sampler()
        sampler.start()
        lat, t_end = [], time.perf_counter() + seconds
        while time.perf_counter() < t_end or len(lat) < 10:
            t = time.perf_counter()
            mod.run()
            lat.append(time.perf_counter() - t)
        lat = np.array(lat)
        # Throughput with the queue kept full: submit 4 tasks, then wait once.
        n_pipe, t_pipe = 0, time.perf_counter()
        while time.perf_counter() - t_pipe < seconds / 2:
            for _ in range(4):
                mod.submit()
            mod.wait()
            n_pipe += 4
        pipe_ips = n_pipe / (time.perf_counter() - t_pipe)
        out.update(sampler.stop())
        ops = 2 * case.macs
        out.update(
            status="ok",
            latency_ms=round(float(lat.mean() * 1e3), 3),
            latency_ms_min=round(float(lat.min() * 1e3), 3),
            ips=round(1 / float(lat.mean()), 1),
            tops=round(ops / float(lat.mean()) / 1e12, 3),
            tops_best=round(ops / float(lat.min()) / 1e12, 3),
            tops_pipelined=round(ops * pipe_ips / 1e12, 3),
            last_error=dev.last_error(),
        )
        if out.get("clock_mhz"):
            # MACs the hardware completes per clock cycle, and the rate that would give at the clock limit.
            out["macs_per_cycle"] = round(case.macs / float(lat.mean()) / (out["clock_mhz"] * 1e6))
            out["tops_at_clock_limit"] = round(out["tops"] * out["clock_mhz_limit"] / out["clock_mhz"], 3)
    finally:
        mod.close()
    return out


CASES = [
    Case("floor_c32_k3_hw32_d1", 32, 32, 3, 1),          # how long does an almost-empty task take?
    Case("c64_k3_hw128_d8", 64, 128, 3, 8),
    Case("c128_k3_hw64_d8", 128, 64, 3, 8),
    Case("c256_k3_hw32_d8", 256, 32, 3, 8),
    Case("c256_k3_hw64_d16", 256, 64, 3, 16),
    Case("c256_k3_hw64_d32", 256, 64, 3, 32),
    Case("c512_k3_hw32_d16", 512, 32, 3, 16),
    Case("c512_k3_hw64_d8", 512, 64, 3, 8),
    Case("c1024_k3_hw32_d4", 1024, 32, 3, 4),
    Case("c192_k3_hw64_d16", 192, 64, 3, 16),
    Case("c256_k1_hw64_d16", 256, 64, 1, 16),
    Case("c512_k1_hw64_d16", 512, 64, 1, 16),
    Case("c1024_k1_hw32_d16", 1024, 32, 1, 16),
    Case("c128_k5_hw64_d8", 128, 64, 5, 8),
    Case("depthwise_c256_k3_hw64_d16", 256, 64, 3, 16, groups=256),
    Case("c256_k3_hw64_d16_unstructured50", 256, 64, 3, 16, sparsity="unstructured50"),
    Case("c256_k3_hw64_d16_unstructured90", 256, 64, 3, 16, sparsity="unstructured90"),
    Case("c256_k3_hw64_d16_struct2of4", 256, 64, 3, 16, sparsity="struct2of4"),
    Case("c256_k3_hw64_d16_prunedhalf", 256, 64, 3, 16, sparsity="prunedhalf"),
]
QUICK = {"floor_c32_k3_hw32_d1", "c256_k3_hw64_d16", "c256_k1_hw64_d16"}


def power_state() -> str:
    """'AC', 'battery' or 'unknown', plus the active Windows power plan."""
    ps = (r"$b = Get-CimInstance -Namespace root\wmi -ClassName BatteryStatus -ErrorAction SilentlyContinue | "
          r"Select-Object -First 1; if ($b) { if ($b.PowerOnline) { 'AC' } else { 'battery' } } else { 'unknown' }; "
          r"(powercfg /getactivescheme) -replace '.*\((.*)\).*', '$1'")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=30)
        parts = [p.strip() for p in out.stdout.splitlines() if p.strip()]
        return ", ".join(parts[:2]) + " plan" if len(parts) >= 2 else (parts[0] if parts else "unknown")
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--tag", default="", help="suffix for the results file, e.g. 'ac' or 'battery'")
    args = ap.parse_args()
    cases = [c for c in CASES if (not args.quick or c.name in QUICK) and (not args.only or args.only in c.name)]
    RESULTS.mkdir(exist_ok=True)
    power_start = power_state()
    rows = []
    with DlaDevice() as dev:
        for case in cases:
            r = measure(dev, case, args.seconds)
            rows.append(r)
            if r["status"] == "ok":
                print(f"{case.name:36} {r['gmacs']:8.2f} GMAC  {r['latency_ms']:8.3f} ms  {r['tops']:6.2f} TOPS "
                      f"(pipelined {r['tops_pipelined']:.2f})  clock {r.get('clock_mhz')} MHz  "
                      f"{r.get('macs_per_cycle')} MAC/cycle  at-limit {r.get('tops_at_clock_limit')}  "
                      f"util {r.get('util_pct')}%  {r.get('power_w')} W  finite={r['output_finite']}", flush=True)
            else:
                print(f"{case.name:36} {r['status']}: {r.get('error', '')[:160]}", flush=True)
    tag = (f"_{args.tag}" if args.tag else "") + ("" if not (args.quick or args.only) else "_partial")
    power_end = power_state()
    meta = {"date": time.strftime("%Y-%m-%d %H:%M"), "power_at_start": power_start, "power_at_end": power_end,
            "seconds_per_case": args.seconds}
    (RESULTS / f"tops{tag}.json").write_text(json.dumps({"meta": meta, "cases": rows}, indent=1))
    ok = [r for r in rows if r["status"] == "ok"]
    lines = [f"Measured {meta['date']}. Power at start: {power_start}; at end: {power_end}.", "",
             "| case | GMACs/inference | latency ms | TOPS (FP16) | DLA clock MHz | MACs/cycle | TOPS at clock limit | util % | W |",
             "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['name']} | {r['gmacs']} | {r['latency_ms']} | {r['tops']} | {r.get('clock_mhz', '')} | "
              f"{r.get('macs_per_cycle', '')} | {r.get('tops_at_clock_limit', '')} | {r.get('util_pct', '')} | "
              f"{r.get('power_w', '')} |" for r in ok]
    (RESULTS / f"tops{tag}.md").write_text("\n".join(lines) + "\n")
    if ok:
        best = max(ok, key=lambda r: max(r["tops"], r["tops_pipelined"]))
        print(f"\npeak: {max(best['tops'], best['tops_pipelined']):.2f} TOPS (FP16) on {best['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
