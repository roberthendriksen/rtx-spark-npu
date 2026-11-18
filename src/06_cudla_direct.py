"""Step 6 - drive the NPU with nothing but cuDLA (no TensorRT at run time).

    python 06_cudla_direct.py [--iters 2000]

1. TensorRT is used once, offline, as a compiler: EngineCapability.DLA_STANDALONE makes it emit
   the raw DLA loadable instead of a TensorRT engine.
2. The loadable is then executed through cudla.dll -> nvcudla.dll (cudlaDrv*) by hand:
   cudlaCreateDevice -> cudlaModuleLoadFromMemory -> cudlaMemRegister -> cudlaSubmitTask.
"""
from __future__ import annotations

import argparse
import ctypes as C
import os
import time
from pathlib import Path

import numpy as np

from npu_common import (
    CUDLA_BIN, CUDLA_CUDA_DLA, CUDLA_INPUT_TENSOR_DESCRIPTORS, CUDLA_NUM_INPUT_TENSORS,
    CUDLA_NUM_OUTPUT_TENSORS, CUDLA_OUTPUT_TENSOR_DESCRIPTORS, ROOT, Cuda, CudlaTask,
    CudlaTensorDesc, cudla_name, load_cudla,
)

MODELS = ROOT / "models"
ONNX = MODELS / "convnet_fp16_io16.onnx"
LOADABLE = MODELS / "convnet_fp16_io16.dla.loadable"


def build_loadable() -> None:
    for d in [CUDLA_BIN, *(ROOT / "vendor" / "tensorrt").glob("TensorRT-*/bin")]:
        os.add_dll_directory(str(d))
        os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]
    import importlib.util

    import onnx
    import onnxruntime as ort
    import tensorrt_bindings as trt

    spec = importlib.util.spec_from_file_location("make_model", Path(__file__).with_name("03_make_model.py"))
    mm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mm)
    onnx.save(mm.convnet(io_fp32=False), ONNX)
    x = np.random.default_rng(1).standard_normal((1, 3, 224, 224)).astype(np.float16)
    y = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
    np.save(MODELS / "convnet_fp16_io16.input.npy", x)
    np.save(MODELS / "convnet_fp16_io16.cpu_ref.npy", y)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    config = builder.create_builder_config()
    config.default_device_type = trt.DeviceType.DLA
    config.DLA_core = 0
    config.engine_capability = trt.EngineCapability.DLA_STANDALONE
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(ONNX.read_bytes()):
        raise SystemExit([str(parser.get_error(i)) for i in range(parser.num_errors)])
    fmt = 1 << int(trt.TensorFormat.DLA_LINEAR)
    network.get_input(0).allowed_formats = fmt
    network.get_output(0).allowed_formats = fmt
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise SystemExit("standalone loadable build failed")
    LOADABLE.write_bytes(bytes(blob))
    print(f"compiled DLA loadable: {LOADABLE.relative_to(ROOT)} ({LOADABLE.stat().st_size / 1e6:.2f} MB)")


def ok(rc: int, what: str) -> None:
    print(f"  {what:34} {cudla_name(rc)}")
    if rc != 0:
        raise SystemExit(f"{what} failed: {cudla_name(rc)}")


def describe(cudla, module, count_attr, desc_attr) -> list[CudlaTensorDesc]:
    n = C.c_uint64(0)
    ok(cudla.cudlaModuleGetAttributes(module, count_attr, C.byref(n)), f"cudlaModuleGetAttributes({count_attr})")
    arr = (CudlaTensorDesc * (n.value & 0xFFFFFFFF))()
    holder = C.c_void_p(C.addressof(arr))
    ok(cudla.cudlaModuleGetAttributes(module, desc_attr, C.byref(holder)), f"cudlaModuleGetAttributes({desc_attr})")
    return list(arr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    if args.rebuild or not LOADABLE.exists():
        build_loadable()

    cuda = Cuda()
    cudla = load_cudla()
    dev = C.c_void_p()
    ok(cudla.cudlaCreateDevice(0, C.byref(dev), CUDLA_CUDA_DLA), "cudlaCreateDevice(hybrid)")
    blob = LOADABLE.read_bytes()
    buf = (C.c_ubyte * len(blob)).from_buffer_copy(blob)
    module = C.c_void_p()
    ok(cudla.cudlaModuleLoadFromMemory(dev, buf, len(blob), C.byref(module), 0), "cudlaModuleLoadFromMemory")

    ins = describe(cudla, module, CUDLA_NUM_INPUT_TENSORS, CUDLA_INPUT_TENSOR_DESCRIPTORS)
    outs = describe(cudla, module, CUDLA_NUM_OUTPUT_TENSORS, CUDLA_OUTPUT_TENSOR_DESCRIPTORS)
    for kind, descs in (("in ", ins), ("out", outs)):
        for d in descs:
            print(f"  {kind} tensor '{d.name.decode()}' size={d.size} nchw=({d.n},{d.c},{d.h},{d.w}) "
                  f"dataFormat={d.dataFormat} dataType={d.dataType} dataCategory={d.dataCategory} "
                  f"stride={list(d.stride)[:4]}")

    def alloc_registered(d: CudlaTensorDesc):
        ptr = cuda.alloc(d.size)
        cuda.zero(ptr, d.size)
        reg = C.c_void_p()
        ok(cudla.cudlaMemRegister(dev, ptr, d.size, C.byref(reg), 0), f"cudlaMemRegister('{d.name.decode()}')")
        return ptr, reg

    in_bufs = [alloc_registered(d) for d in ins]
    out_bufs = [alloc_registered(d) for d in outs]

    x = np.load(MODELS / "convnet_fp16_io16.input.npy")
    ref = np.load(MODELS / "convnet_fp16_io16.cpu_ref.npy").astype(np.float32)
    assert x.nbytes == ins[0].size, f"input layout is not plain NCHW ({x.nbytes} vs {ins[0].size})"
    cuda.to_device(in_bufs[0][0], x.tobytes())
    cuda.sync()

    in_arr = (C.c_void_p * len(ins))(*[r for _, r in in_bufs])
    out_arr = (C.c_void_p * len(outs))(*[r for _, r in out_bufs])
    task = CudlaTask(
        moduleHandle=module,
        outputTensor=C.cast(out_arr, C.POINTER(C.c_void_p)), numOutputTensors=len(outs),
        numInputTensors=len(ins), inputTensor=C.cast(in_arr, C.POINTER(C.c_void_p)),
        waitEvents=None, signalEvents=None,
    )
    stream = cuda.stream()
    ok(cudla.cudlaSubmitTask(dev, C.byref(task), 1, stream, 0), "cudlaSubmitTask")
    cuda.sync(stream)
    ok(cudla.cudlaGetLastError(dev), "cudlaGetLastError")

    y = np.frombuffer(cuda.from_device(out_bufs[0][0], outs[0].size), dtype=np.float16)
    y = y.reshape(outs[0].n, outs[0].c, outs[0].h, -1)[..., : outs[0].w].astype(np.float32)
    print(f"\nNPU output via raw cuDLA: mean={y.mean():.5f} max={y.max():.4f}")
    print(f"vs CPU reference: max abs err={np.abs(y - ref).max():.4f}, "
          f"relative L2 err={np.linalg.norm(y - ref) / np.linalg.norm(ref):.4%}")

    lat = []
    t_start = time.perf_counter()
    for _ in range(args.iters):
        t0 = time.perf_counter()
        rc = cudla.cudlaSubmitTask(dev, C.byref(task), 1, stream, 0)
        cuda.sync(stream)
        lat.append(time.perf_counter() - t0)
        if rc != 0:
            raise SystemExit(f"cudlaSubmitTask failed: {cudla_name(rc)}")
    wall = time.perf_counter() - t_start
    lat_ms = np.array(lat) * 1e3
    print(f"{args.iters} raw cuDLA submissions: mean={lat_ms.mean():.3f} ms p50={np.percentile(lat_ms, 50):.3f} ms "
          f"p99={np.percentile(lat_ms, 99):.3f} ms -> {args.iters / wall:.1f} inferences/s")

    for (ptr, _), _d in zip(in_bufs + out_bufs, ins + outs):
        cudla.cudlaMemUnregister(dev, ptr)
        cuda.free(ptr)
    ok(cudla.cudlaModuleUnload(module, 0), "cudlaModuleUnload")
    ok(cudla.cudlaDestroyDevice(dev), "cudlaDestroyDevice")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
