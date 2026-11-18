"""Step 1 - find the NPU: enumerate it through cuDLA and open a device handle."""
import ctypes as C
import sys

from npu_common import (
    CUDLA_CUDA_DLA, CUDLA_DEVICE_VERSION, CUDLA_STANDALONE, CUDLA_UNIFIED_ADDRESSING,
    Cuda, cudla_name, load_cudla,
)


def main() -> int:
    cuda = Cuda()
    print(f"CUDA driver API version : {cuda.driver_version()}")
    print(f"CUDA device 0           : {cuda.name()}")

    cudla = load_cudla()
    ver = C.c_uint64()
    rc = cudla.cudlaGetVersion(C.byref(ver))
    print(f"cudlaGetVersion         : {cudla_name(rc)} -> {ver.value} "
          f"({ver.value // 1000000}.{ver.value // 1000 % 1000}.{ver.value % 1000})")

    count = C.c_uint64()
    rc = cudla.cudlaDeviceGetCount(C.byref(count))
    print(f"cudlaDeviceGetCount     : {cudla_name(rc)} -> {count.value} DLA device(s)")
    if rc != 0 or count.value == 0:
        return 1

    for mode, label in ((CUDLA_CUDA_DLA, "hybrid (CUDA_DLA)"), (CUDLA_STANDALONE, "standalone")):
        dev = C.c_void_p()
        rc = cudla.cudlaCreateDevice(0, C.byref(dev), mode)
        print(f"cudlaCreateDevice[{label:18}] : {cudla_name(rc)}")
        if rc != 0:
            continue
        attr = C.c_uint64(0)
        rc = cudla.cudlaDeviceGetAttribute(dev, CUDLA_UNIFIED_ADDRESSING, C.byref(attr))
        print(f"    unified addressing  : {cudla_name(rc)} -> {attr.value & 0xFF}")
        attr = C.c_uint64(0)
        rc = cudla.cudlaDeviceGetAttribute(dev, CUDLA_DEVICE_VERSION, C.byref(attr))
        print(f"    DLA hardware version: {cudla_name(rc)} -> {attr.value & 0xFFFFFFFF}")
        print(f"    cudlaGetLastError   : {cudla_name(cudla.cudlaGetLastError(dev))}")
        print(f"    cudlaDestroyDevice  : {cudla_name(cudla.cudlaDestroyDevice(dev))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
