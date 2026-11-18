"""Step 3 - author small ONNX models the Windows-on-Arm DLA can take.

DLA rules on this platform (TensorRT 11.5 beta): static shapes, batch 1, FP16 compute,
no GPU fallback (every layer must run on the DLA), FP32 linear I/O is allowed.

    python 03_make_model.py            -> models/convnet_fp16.onnx (+ reference input/output)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def convnet(channels=(3, 32, 64, 64, 16), hw=(224, 224), seed=0, io_fp32=True) -> onnx.ModelProto:
    """conv3x3+ReLU stack. Compute is FP16; I/O is FP32 (cast at the edges) unless io_fp32=False."""
    rng = np.random.default_rng(seed)
    io_type = TensorProto.FLOAT if io_fp32 else TensorProto.FLOAT16
    nodes, inits = [], []
    cur = "input"
    if io_fp32:
        nodes.append(helper.make_node("Cast", ["input"], ["x_fp16"], to=TensorProto.FLOAT16, name="cast_in"))
        cur = "x_fp16"
    for i in range(len(channels) - 1):
        cin, cout = channels[i], channels[i + 1]
        w = (rng.standard_normal((cout, cin, 3, 3)) * np.sqrt(2.0 / (cin * 9))).astype(np.float16)
        b = (rng.standard_normal(cout) * 0.05).astype(np.float16)
        inits += [numpy_helper.from_array(w, f"w{i}"), numpy_helper.from_array(b, f"b{i}")]
        nodes.append(helper.make_node("Conv", [cur, f"w{i}", f"b{i}"], [f"conv{i}"], name=f"conv{i}",
                                      kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]))
        last = i == len(channels) - 2
        # With FP16 I/O the final ReLU is the graph output (the DLA has no Identity layer).
        relu_out = "output" if (last and not io_fp32) else f"relu{i}"
        nodes.append(helper.make_node("Relu", [f"conv{i}"], [relu_out], name=f"relu{i}"))
        cur = relu_out
    if io_fp32:
        nodes.append(helper.make_node("Cast", [cur], ["output"], to=TensorProto.FLOAT, name="cast_out"))
    graph = helper.make_graph(
        nodes, "convnet_fp16",
        [helper.make_tensor_value_info("input", io_type, [1, channels[0], *hw])],
        [helper.make_tensor_value_info("output", io_type, [1, channels[-1], *hw])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)], producer_name="sparky-npu")
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def main() -> int:
    MODELS.mkdir(exist_ok=True)
    model = convnet()
    path = MODELS / "convnet_fp16.onnx"
    onnx.save(model, path)

    import onnxruntime as ort

    x = np.random.default_rng(1).standard_normal((1, 3, 224, 224)).astype(np.float32)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    y = sess.run(None, {"input": x})[0]
    np.save(MODELS / "convnet_fp16.input.npy", x)
    np.save(MODELS / "convnet_fp16.cpu_ref.npy", y)
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} KiB)")
    print(f"input {x.shape} {x.dtype} -> output {y.shape} {y.dtype}; CPU reference mean={y.mean():.5f} max={y.max():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
