"""Reusable helpers for the NVIDIA NPU (DLA) on Windows on Arm.

    from npu_dla import compile_loadable, DlaDevice

    loadable = compile_loadable(onnx_bytes)      # offline: TensorRT + nvdla_compiler.dll
    with DlaDevice() as dev:                     # run time: cudla.dll only
        mod = dev.load(loadable)
        mod.write(0, x)                          # NCHW numpy array
        mod.run()
        y = mod.read(0)

Run time needs only `cudla.dll` and the display driver. TensorRT is needed to compile.
"""
from __future__ import annotations

import ctypes as C
import os
from dataclasses import dataclass

import numpy as np

from npu_common import (
    CUDLA_BIN, CUDLA_CUDA_DLA, CUDLA_INPUT_TENSOR_DESCRIPTORS, CUDLA_NUM_INPUT_TENSORS,
    CUDLA_NUM_OUTPUT_TENSORS, CUDLA_OUTPUT_TENSOR_DESCRIPTORS, ROOT, Cuda, CudlaTask,
    CudlaTensorDesc, cudla_name, load_cudla,
)

_trt = None


def tensorrt():
    """Import TensorRT with cudla.dll and nvdla_compiler.dll on the DLL search path."""
    global _trt
    if _trt is None:
        for d in [CUDLA_BIN, *(ROOT / "vendor" / "tensorrt").glob("TensorRT-*/bin")]:
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]
        import tensorrt_bindings as trt

        _trt = trt
    return _trt


class DlaBuildError(RuntimeError):
    pass


def compile_loadable(onnx_bytes: bytes, verbose: bool = False, adjust_for_dla: bool = True) -> bytes:
    """Compile an ONNX model (static shapes, batch 1, FP16 tensors) into a raw DLA loadable."""
    trt = tensorrt()
    messages: list[str] = []

    class Log(trt.ILogger):
        def log(self, severity, msg):
            if severity <= trt.Logger.WARNING:
                messages.append(msg)
            if verbose:
                print("[TRT]", msg)

    logger = Log()
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    config = builder.create_builder_config()
    config.default_device_type = trt.DeviceType.DLA
    config.DLA_core = 0
    config.engine_capability = trt.EngineCapability.DLA_STANDALONE
    parser = trt.OnnxParser(network, logger)
    if adjust_for_dla and hasattr(trt.OnnxParserFlag, "ADJUST_FOR_DLA"):
        parser.set_flag(trt.OnnxParserFlag.ADJUST_FOR_DLA)   # lets the parser reshape tensors to 4-D for the DLA
    if not parser.parse(onnx_bytes):
        raise DlaBuildError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    fmt = 1 << int(trt.TensorFormat.DLA_LINEAR)
    for i in range(network.num_inputs):
        network.get_input(i).allowed_formats = fmt
    for i in range(network.num_outputs):
        network.get_output(i).allowed_formats = fmt
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise DlaBuildError(" | ".join(m[:300] for m in messages[-4:]) or "build failed")
    return bytes(blob)


@dataclass
class Tensor:
    name: str
    shape: tuple[int, int, int, int]  # N, C, H, W
    nbytes: int
    row_elems: int                    # DLA_LINEAR pads each row to a 64-byte multiple
    ptr: C.c_void_p
    reg: C.c_void_p


class DlaModule:
    def __init__(self, dev: "DlaDevice", loadable: bytes) -> None:
        self.dev, cudla, cuda = dev, dev.cudla, dev.cuda
        buf = (C.c_ubyte * len(loadable)).from_buffer_copy(loadable)
        self.handle = C.c_void_p()
        dev.check(cudla.cudlaModuleLoadFromMemory(dev.handle, buf, len(loadable), C.byref(self.handle), 0),
                  "cudlaModuleLoadFromMemory")
        self.inputs = self._tensors(CUDLA_NUM_INPUT_TENSORS, CUDLA_INPUT_TENSOR_DESCRIPTORS)
        self.outputs = self._tensors(CUDLA_NUM_OUTPUT_TENSORS, CUDLA_OUTPUT_TENSOR_DESCRIPTORS)
        self._in = (C.c_void_p * len(self.inputs))(*[t.reg for t in self.inputs])
        self._out = (C.c_void_p * len(self.outputs))(*[t.reg for t in self.outputs])
        self.task = CudlaTask(
            moduleHandle=self.handle,
            outputTensor=C.cast(self._out, C.POINTER(C.c_void_p)), numOutputTensors=len(self.outputs),
            numInputTensors=len(self.inputs), inputTensor=C.cast(self._in, C.POINTER(C.c_void_p)),
            waitEvents=None, signalEvents=None,
        )
        self.stream = cuda.stream()

    def _tensors(self, count_attr: int, desc_attr: int) -> list[Tensor]:
        dev, cudla, cuda = self.dev, self.dev.cudla, self.dev.cuda
        n = C.c_uint64(0)
        dev.check(cudla.cudlaModuleGetAttributes(self.handle, count_attr, C.byref(n)), "cudlaModuleGetAttributes")
        arr = (CudlaTensorDesc * (n.value & 0xFFFFFFFF))()
        holder = C.c_void_p(C.addressof(arr))
        dev.check(cudla.cudlaModuleGetAttributes(self.handle, desc_attr, C.byref(holder)), "cudlaModuleGetAttributes")
        out = []
        for d in arr:
            row_elems = d.stride[1] // d.stride[0] if d.stride[0] else d.w
            if d.n * d.c * d.h * row_elems * d.stride[0] != d.size:
                raise RuntimeError(f"unexpected layout for tensor {d.name!r}: size={d.size} stride={list(d.stride)}")
            ptr = cuda.alloc(d.size)
            cuda.zero(ptr, d.size)
            reg = C.c_void_p()
            dev.check(cudla.cudlaMemRegister(dev.handle, ptr, d.size, C.byref(reg), 0), "cudlaMemRegister")
            out.append(Tensor(d.name.decode(), (d.n, d.c, d.h, d.w), d.size, row_elems, ptr, reg))
        return out

    def write(self, index: int, array: np.ndarray) -> None:
        t = self.inputs[index]
        a = np.asarray(array, dtype=np.float16).reshape(t.shape)
        if t.row_elems != t.shape[3]:
            padded = np.zeros((*t.shape[:3], t.row_elems), dtype=np.float16)
            padded[..., : t.shape[3]] = a
            a = padded
        self.dev.cuda.to_device(t.ptr, np.ascontiguousarray(a).tobytes())
        self.dev.cuda.sync()  # finish the copy before the DLA task can start (avoids a known race)

    def read(self, index: int = 0) -> np.ndarray:
        t = self.outputs[index]
        raw = np.frombuffer(self.dev.cuda.from_device(t.ptr, t.nbytes), dtype=np.float16)
        return raw.reshape(*t.shape[:3], t.row_elems)[..., : t.shape[3]].copy()

    def submit(self) -> None:
        self.dev.check(self.dev.cudla.cudlaSubmitTask(self.dev.handle, C.byref(self.task), 1, self.stream, 0),
                       "cudlaSubmitTask")

    def wait(self) -> None:
        self.dev.cuda.sync(self.stream)

    def run(self) -> None:
        self.submit()
        self.wait()

    def close(self) -> None:
        self.wait()
        for t in self.inputs + self.outputs:
            self.dev.cudla.cudlaMemUnregister(self.dev.handle, t.ptr)
            self.dev.cuda.free(t.ptr)
        self.dev.cudla.cudlaModuleUnload(self.handle, 0)


class DlaDevice:
    def __init__(self) -> None:
        self.cuda = Cuda()
        self.cudla = load_cudla()
        self.handle = C.c_void_p()
        self.check(self.cudla.cudlaCreateDevice(0, C.byref(self.handle), CUDLA_CUDA_DLA), "cudlaCreateDevice")

    @staticmethod
    def check(rc: int, what: str) -> None:
        if rc != 0:
            raise RuntimeError(f"{what} failed: {cudla_name(rc)}")

    def load(self, loadable: bytes) -> DlaModule:
        return DlaModule(self, loadable)

    def last_error(self) -> str:
        return cudla_name(self.cudla.cudlaGetLastError(self.handle))

    def close(self) -> None:
        self.cudla.cudlaDestroyDevice(self.handle)

    def __enter__(self) -> "DlaDevice":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
