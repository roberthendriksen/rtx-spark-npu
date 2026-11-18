"""Step 2 - ask TensorRT whether it can see the DLA (needs cudla.dll + nvdla_compiler.dll)."""
import os
import sys

from npu_common import CUDLA_BIN, ROOT

os.add_dll_directory(str(CUDLA_BIN))
os.environ["PATH"] = str(CUDLA_BIN) + os.pathsep + os.environ["PATH"]
trt_sdk_bin = next((ROOT / "vendor" / "tensorrt").glob("TensorRT-*/bin"), None)
if trt_sdk_bin:
    os.add_dll_directory(str(trt_sdk_bin))
    os.environ["PATH"] = str(trt_sdk_bin) + os.pathsep + os.environ["PATH"]

import tensorrt_bindings as trt  # noqa: E402  (bindings-only install: module is tensorrt_bindings)

print("TensorRT", trt.__version__)
logger = trt.Logger(trt.Logger.VERBOSE if "-v" in sys.argv else trt.Logger.INFO)
builder = trt.Builder(logger)
print("num_DLA_cores           :", builder.num_DLA_cores)
print("max_DLA_batch_size      :", getattr(builder, "max_DLA_batch_size", "n/a"))
print("platform_has_tf32       :", getattr(builder, "platform_has_tf32", "n/a"))
runtime = trt.Runtime(logger)
print("runtime.num_DLA_cores   :", runtime.num_DLA_cores)
