"""Step 4 - compile an ONNX model for the NPU with TensorRT's DLA backend.

    python 04_build_engine.py models/convnet_fp16.onnx [-v]

Produces <model>.dla.engine. Every layer is forced onto the DLA (there is no GPU fallback on
Windows on Arm). Needs cudla.dll (CUDA redist) and nvdla_compiler.dll (TensorRT Arm64 zip).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from npu_common import CUDLA_BIN, ROOT


def add_dll_dirs() -> None:
    dirs = [CUDLA_BIN]
    dirs += list((ROOT / "vendor" / "tensorrt").glob("TensorRT-*/bin"))
    for d in dirs:
        os.add_dll_directory(str(d))
        os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]


add_dll_dirs()
import tensorrt_bindings as trt  # noqa: E402


def build(onnx_path: Path, verbose: bool = False) -> Path:
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)
    print(f"TensorRT {trt.__version__}; DLA cores visible: {builder.num_DLA_cores}")
    if builder.num_DLA_cores < 1:
        raise SystemExit("TensorRT sees no DLA core - is cudla.dll on the DLL search path?")

    network = builder.create_network(0)
    config = builder.create_builder_config()
    config.default_device_type = trt.DeviceType.DLA
    config.DLA_core = 0
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    parser = trt.OnnxParser(network, logger)
    for flag in ("ADJUST_FOR_DLA", "REPORT_CAPABILITY_DLA"):
        if hasattr(trt.OnnxParserFlag, flag):
            parser.set_flag(getattr(trt.OnnxParserFlag, flag))
    if hasattr(parser, "set_builder_config"):
        parser.set_builder_config(config)
    if not parser.parse(onnx_path.read_bytes()):
        for i in range(parser.num_errors):
            print("ONNX parse error:", parser.get_error(i))
        raise SystemExit(1)

    print("layers and where TensorRT is willing to put them:")
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        print(f"  {layer.name:14} {str(layer.type).split('.')[-1]:14} can_run_on_DLA={config.can_run_on_DLA(layer)}")

    t0 = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise SystemExit("engine build failed (see TensorRT log above)")
    out = onnx_path.with_suffix(".dla.engine")
    out.write_bytes(bytes(plan))
    print(f"built {out.relative_to(ROOT)} in {time.perf_counter() - t0:.1f}s ({out.stat().st_size / 1e6:.2f} MB)")
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    build(Path(args[0] if args else ROOT / "models" / "convnet_fp16.onnx").resolve(), "-v" in sys.argv)
