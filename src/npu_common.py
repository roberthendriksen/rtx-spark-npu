"""Shared ctypes bindings for the NVIDIA NPU (DLA) on Windows on Arm.

Two libraries are involved:

* ``nvcuda.dll``  - the CUDA driver API. Ships with the display driver in System32.
* ``cudla.dll``   - NVIDIA's public cuDLA front end (CUDA redist component ``libcudla``).
                    It forwards every call to ``nvcudla.dll`` (``cudlaDrv*``) inside the
                    driver store, which in turn drives ``nvdla_runtime.dll``.

The struct layouts and enum values below are taken from the official ``cudla.h``.
"""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CUDLA_BIN = ROOT / "vendor" / "libcudla" / "libcudla-windows-arm64-13.5.25-archive" / "bin" / "arm64"

# --------------------------------------------------------------------------- cuDLA
CUDLA_STATUS = {
    0: "cudlaSuccess",
    1: "cudlaErrorInvalidParam",
    2: "cudlaErrorOutOfResources",
    3: "cudlaErrorCreationFailed",
    4: "cudlaErrorInvalidAddress",
    5: "cudlaErrorOs",
    6: "cudlaErrorCuda",
    7: "cudlaErrorUmd",
    8: "cudlaErrorInvalidDevice",
    9: "cudlaErrorInvalidAttribute",
    10: "cudlaErrorIncompatibleDlaSWVersion",
    11: "cudlaErrorMemoryRegistered",
    12: "cudlaErrorInvalidModule",
    13: "cudlaErrorUnsupportedOperation",
    14: "cudlaErrorNvSci",
    15: "cudlaErrorDriverNotFound",
    0x40000001: "cudlaErrorDlaErrInvalidInput",
    0x40000002: "cudlaErrorDlaErrInvalidPreAction",
    0x40000003: "cudlaErrorDlaErrNoMem",
    0x40000004: "cudlaErrorDlaErrProcessorBusy",
    0x40000005: "cudlaErrorDlaErrTaskStatusMismatch",
    0x40000006: "cudlaErrorDlaErrEngineTimeout",
    0x40000007: "cudlaErrorDlaErrDataMismatch",
}

CUDLA_CUDA_DLA = 0  # hybrid: tasks are submitted on a CUDA stream
CUDLA_STANDALONE = 1

CUDLA_UNIFIED_ADDRESSING = 0
CUDLA_DEVICE_VERSION = 1

CUDLA_NUM_INPUT_TENSORS = 0
CUDLA_NUM_OUTPUT_TENSORS = 1
CUDLA_INPUT_TENSOR_DESCRIPTORS = 2
CUDLA_OUTPUT_TENSOR_DESCRIPTORS = 3

NAME_MAX = 80  # CUDLA_RUNTIME_TENSOR_DESC_NAME_MAX_LEN
NUM_STRIDES = 8  # CUDLA_LOADABLE_TENSOR_DESC_NUM_STRIDES


class CudlaTensorDesc(C.Structure):
    _fields_ = [
        ("name", C.c_char * (NAME_MAX + 1)),
        ("size", C.c_uint64),
        ("n", C.c_uint64),
        ("c", C.c_uint64),
        ("h", C.c_uint64),
        ("w", C.c_uint64),
        ("dataFormat", C.c_uint8),
        ("dataType", C.c_uint8),
        ("dataCategory", C.c_uint8),
        ("pixelFormat", C.c_uint8),
        ("pixelMapping", C.c_uint8),
        ("stride", C.c_uint32 * NUM_STRIDES),
    ]


class CudlaTask(C.Structure):
    _fields_ = [
        ("moduleHandle", C.c_void_p),
        ("outputTensor", C.POINTER(C.c_void_p)),
        ("numOutputTensors", C.c_uint32),
        ("numInputTensors", C.c_uint32),
        ("inputTensor", C.POINTER(C.c_void_p)),
        ("waitEvents", C.c_void_p),
        ("signalEvents", C.c_void_p),
    ]


def cudla_name(status: int) -> str:
    return CUDLA_STATUS.get(status & 0xFFFFFFFF, f"unknown(0x{status & 0xFFFFFFFF:x})")


def load_cudla() -> C.CDLL:
    os.add_dll_directory(str(CUDLA_BIN))
    lib = C.CDLL(str(CUDLA_BIN / "cudla.dll"))
    lib.cudlaGetVersion.argtypes = [C.POINTER(C.c_uint64)]
    lib.cudlaDeviceGetCount.argtypes = [C.POINTER(C.c_uint64)]
    lib.cudlaCreateDevice.argtypes = [C.c_uint64, C.POINTER(C.c_void_p), C.c_uint32]
    lib.cudlaDestroyDevice.argtypes = [C.c_void_p]
    lib.cudlaDeviceGetAttribute.argtypes = [C.c_void_p, C.c_int, C.c_void_p]
    lib.cudlaGetLastError.argtypes = [C.c_void_p]
    lib.cudlaMemRegister.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.POINTER(C.c_void_p), C.c_uint32]
    lib.cudlaMemUnregister.argtypes = [C.c_void_p, C.c_void_p]
    lib.cudlaModuleLoadFromMemory.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.POINTER(C.c_void_p), C.c_uint32]
    lib.cudlaModuleGetAttributes.argtypes = [C.c_void_p, C.c_int, C.c_void_p]
    lib.cudlaModuleUnload.argtypes = [C.c_void_p, C.c_uint32]
    lib.cudlaSubmitTask.argtypes = [C.c_void_p, C.POINTER(CudlaTask), C.c_uint32, C.c_void_p, C.c_uint32]
    lib.cudlaSetTaskTimeoutInMs.argtypes = [C.c_void_p, C.c_uint32]
    for name in (
        "cudlaGetVersion", "cudlaDeviceGetCount", "cudlaCreateDevice", "cudlaDestroyDevice",
        "cudlaDeviceGetAttribute", "cudlaGetLastError", "cudlaMemRegister", "cudlaMemUnregister",
        "cudlaModuleLoadFromMemory", "cudlaModuleGetAttributes", "cudlaModuleUnload",
        "cudlaSubmitTask", "cudlaSetTaskTimeoutInMs",
    ):
        getattr(lib, name).restype = C.c_uint32
    return lib


# --------------------------------------------------------------------------- CUDA driver API
class Cuda:
    """Minimal CUDA driver-API wrapper (primary context on GPU 0)."""

    def __init__(self) -> None:
        self.lib = C.CDLL("nvcuda.dll")
        L = self.lib
        L.cuInit.argtypes = [C.c_uint]
        L.cuDeviceGet.argtypes = [C.POINTER(C.c_int), C.c_int]
        L.cuDeviceGetName.argtypes = [C.c_char_p, C.c_int, C.c_int]
        L.cuDeviceTotalMem_v2.argtypes = [C.POINTER(C.c_size_t), C.c_int]
        L.cuDevicePrimaryCtxRetain.argtypes = [C.POINTER(C.c_void_p), C.c_int]
        L.cuDevicePrimaryCtxRelease_v2.argtypes = [C.c_int]
        L.cuCtxSetCurrent.argtypes = [C.c_void_p]
        L.cuCtxSynchronize.argtypes = []
        L.cuMemAlloc_v2.argtypes = [C.POINTER(C.c_void_p), C.c_size_t]
        L.cuMemFree_v2.argtypes = [C.c_void_p]
        L.cuMemcpyHtoD_v2.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]
        L.cuMemcpyDtoH_v2.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]
        L.cuMemsetD8_v2.argtypes = [C.c_void_p, C.c_ubyte, C.c_size_t]
        L.cuStreamCreate.argtypes = [C.POINTER(C.c_void_p), C.c_uint]
        L.cuStreamSynchronize.argtypes = [C.c_void_p]
        L.cuStreamDestroy_v2.argtypes = [C.c_void_p]
        L.cuDriverGetVersion.argtypes = [C.POINTER(C.c_int)]
        self.check(L.cuInit(0), "cuInit")
        self.device = C.c_int()
        self.check(L.cuDeviceGet(C.byref(self.device), 0), "cuDeviceGet")
        self.ctx = C.c_void_p()
        self.check(L.cuDevicePrimaryCtxRetain(C.byref(self.ctx), self.device), "cuDevicePrimaryCtxRetain")
        self.check(L.cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")

    @staticmethod
    def check(rc: int, what: str) -> None:
        if rc != 0:
            raise RuntimeError(f"{what} failed with CUDA error {rc}")

    def name(self) -> str:
        buf = C.create_string_buffer(256)
        self.check(self.lib.cuDeviceGetName(buf, 256, self.device), "cuDeviceGetName")
        return buf.value.decode()

    def driver_version(self) -> int:
        v = C.c_int()
        self.check(self.lib.cuDriverGetVersion(C.byref(v)), "cuDriverGetVersion")
        return v.value

    def alloc(self, nbytes: int) -> C.c_void_p:
        p = C.c_void_p()
        self.check(self.lib.cuMemAlloc_v2(C.byref(p), nbytes), "cuMemAlloc")
        return p

    def free(self, p: C.c_void_p) -> None:
        self.lib.cuMemFree_v2(p)

    def to_device(self, p: C.c_void_p, data: bytes | bytearray | memoryview) -> None:
        buf = (C.c_ubyte * len(data)).from_buffer_copy(bytes(data))
        self.check(self.lib.cuMemcpyHtoD_v2(p, buf, len(data)), "cuMemcpyHtoD")

    def from_device(self, p: C.c_void_p, nbytes: int) -> bytes:
        buf = (C.c_ubyte * nbytes)()
        self.check(self.lib.cuMemcpyDtoH_v2(buf, p, nbytes), "cuMemcpyDtoH")
        return bytes(buf)

    def zero(self, p: C.c_void_p, nbytes: int) -> None:
        self.check(self.lib.cuMemsetD8_v2(p, 0, nbytes), "cuMemsetD8")

    def stream(self) -> C.c_void_p:
        s = C.c_void_p()
        self.check(self.lib.cuStreamCreate(C.byref(s), 0), "cuStreamCreate")
        return s

    def sync(self, stream: C.c_void_p | None = None) -> None:
        if stream is None:
            self.check(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")
        else:
            self.check(self.lib.cuStreamSynchronize(stream), "cuStreamSynchronize")
