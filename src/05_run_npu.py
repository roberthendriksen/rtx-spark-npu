"""Step 5 - run a DLA engine on the NPU and prove the NPU did the work.

    python 05_run_npu.py models/convnet_fp16.dla.engine [--iters 300]

Inference goes TensorRT -> cuDLA (hybrid mode) -> nvcudla.dll -> nvdla_runtime.dll -> NPU.
I/O tensors live in CUDA device memory and the task is submitted on a CUDA stream.
While the benchmark loops, nvidia-smi is sampled for both devices (0 = GPU, 1 = NPU).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from npu_common import CUDLA_BIN, ROOT, Cuda

for d in [CUDLA_BIN, *(ROOT / "vendor" / "tensorrt").glob("TensorRT-*/bin")]:
    os.add_dll_directory(str(d))
    os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]

import tensorrt_bindings as trt  # noqa: E402


class SmiSampler(threading.Thread):
    """Streams `nvidia-smi` samples for every device while a workload runs."""

    QUERY = "index,utilization.gpu,power.draw,temperature.gpu"

    def __init__(self, period_ms: int = 200) -> None:
        super().__init__(daemon=True)
        self.samples: dict[int, list[tuple[float, float, float]]] = {}
        self.proc = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={self.QUERY}", "--format=csv,noheader,nounits", "-lms", str(period_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )

    @staticmethod
    def _num(s: str) -> float:
        try:
            return float(s)
        except ValueError:
            return float("nan")

    def run(self) -> None:
        for line in self.proc.stdout:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4 and parts[0].isdigit():
                self.samples.setdefault(int(parts[0]), []).append(tuple(self._num(p) for p in parts[1:]))

    def stop(self) -> dict[int, dict[str, float]]:
        self.proc.terminate()
        self.join(timeout=2)
        out = {}
        for idx, rows in self.samples.items():
            a = np.array(rows, dtype=np.float64)
            out[idx] = {
                "n": len(rows),
                "util_mean": float(np.nanmean(a[:, 0])), "util_max": float(np.nanmax(a[:, 0])),
                "power_mean": float(np.nanmean(a[:, 1])), "power_max": float(np.nanmax(a[:, 1])),
                "temp_max": float(np.nanmax(a[:, 2])),
            }
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", nargs="?", default=str(ROOT / "models" / "convnet_fp16.dla.engine"))
    ap.add_argument("--iters", type=int, default=300)
    args = ap.parse_args()
    engine_path = Path(args.engine).resolve()
    stem = engine_path.name.removesuffix(".dla.engine")

    cuda = Cuda()
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    runtime.DLA_core = 0
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise SystemExit("could not deserialize engine")
    context = engine.create_execution_context()

    print(f"engine {engine_path.name}: {engine.num_layers} layer(s), device memory {engine.device_memory_size_v2 / 1e6:.1f} MB")
    insp = engine.create_engine_inspector()
    print("engine layer info:", insp.get_engine_information(trt.LayerInformationFormat.ONELINE).strip()[:600])

    bufs: dict[str, tuple] = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
        nbytes = int(np.prod(shape)) * dtype.itemsize
        ptr = cuda.alloc(nbytes)
        cuda.zero(ptr, nbytes)
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        bufs[name] = (ptr, shape, dtype, nbytes, is_input)
        context.set_tensor_address(name, ptr.value)
        print(f"  {'in ' if is_input else 'out'} {name:8} {shape} {dtype} {engine.get_tensor_format_desc(name)}")

    in_name = next(n for n, b in bufs.items() if b[4])
    out_name = next(n for n, b in bufs.items() if not b[4])
    ref_in, ref_out = (ROOT / "models" / f"{stem}.input.npy"), (ROOT / "models" / f"{stem}.cpu_ref.npy")
    x = np.load(ref_in) if ref_in.exists() else np.random.default_rng(1).standard_normal(bufs[in_name][1]).astype(np.float32)
    x = np.ascontiguousarray(x.astype(bufs[in_name][2]))

    stream = cuda.stream()
    cuda.to_device(bufs[in_name][0], x.tobytes())
    cuda.sync()
    if not context.execute_async_v3(stream.value):
        raise SystemExit("execute_async_v3 failed")
    cuda.sync(stream)
    optr, oshape, odtype, onbytes, _ = bufs[out_name]
    y = np.frombuffer(cuda.from_device(optr, onbytes), dtype=odtype).reshape(oshape).astype(np.float32)
    print(f"\nNPU output: mean={y.mean():.5f} max={y.max():.4f} nonzero={np.count_nonzero(y) / y.size:.1%}")
    if ref_out.exists():
        ref = np.load(ref_out).astype(np.float32)
        err = np.abs(y - ref)
        print(f"vs CPU (onnxruntime) reference: max abs err={err.max():.4f}, mean abs err={err.mean():.5f}, "
              f"relative L2 err={np.linalg.norm(y - ref) / np.linalg.norm(ref):.4%}")

    print(f"\nbenchmark: {args.iters} inferences ...")
    sampler = SmiSampler()
    sampler.start()
    time.sleep(1.0)
    idle = {k: (v[-1] if v else None) for k, v in sampler.samples.items()}
    sampler.samples.clear()
    lat = []
    t_start = time.perf_counter()
    for _ in range(args.iters):
        t0 = time.perf_counter()
        context.execute_async_v3(stream.value)
        cuda.sync(stream)
        lat.append(time.perf_counter() - t0)
    wall = time.perf_counter() - t_start
    stats = sampler.stop()
    lat_ms = np.array(lat) * 1e3
    print(f"latency ms: mean={lat_ms.mean():.3f} p50={np.percentile(lat_ms, 50):.3f} p99={np.percentile(lat_ms, 99):.3f}"
          f" -> {args.iters / wall:.1f} inferences/s")
    names = {0: "GPU (RTX Spark N1X)", 1: "NPU (DLA)"}
    for idx in sorted(stats):
        s = stats[idx]
        print(f"  nvidia-smi dev {idx} {names.get(idx, ''):20} idle={idle.get(idx)}  under load: util mean {s['util_mean']:.0f}% "
              f"max {s['util_max']:.0f}%, power mean {s['power_mean']:.1f} W max {s['power_max']:.1f} W, temp max {s['temp_max']:.0f} C "
              f"({s['n']} samples)")

    for name, (ptr, *_rest) in bufs.items():
        context.set_tensor_address(name, 0)
        cuda.free(ptr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
