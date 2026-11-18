"""Step 8 - do voice-assistant-shaped models fit the NPU, and are they faster than the CPU?

    python 08_voice_models.py [--only NAME] [--seconds 3] [--cpu-threads 8]

Builds synthetic stand-ins (random weights, real layer shapes) for the model families a voice
pipeline uses, compiles each for the DLA, and times NPU vs CPU (onnxruntime) on the same graph.
Random weights answer "does it compile" and "how fast"; they say nothing about accuracy.

    encoder_*   transformer text/audio encoders (embedding models, turn detectors)
    tdnn_*      conv1d speaker-embedding trunks (ECAPA-TDNN-like)
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from npu_common import ROOT
from npu_dla import DlaBuildError, DlaDevice, compile_loadable

RESULTS = ROOT / "results"
F16, F32 = TensorProto.FLOAT16, TensorProto.FLOAT


class G:
    """Tiny ONNX graph builder. dtype is FLOAT16 for the NPU graph and FLOAT for the CPU baseline."""

    def __init__(self, dtype: int, seed: int = 0) -> None:
        self.dtype, self.np = dtype, (np.float16 if dtype == F16 else np.float32)
        self.rng = np.random.default_rng(seed)
        self.nodes, self.inits, self.n = [], [], 0

    def name(self, p: str) -> str:
        self.n += 1
        return f"{p}_{self.n}"

    def const(self, arr, p="c") -> str:
        nm = self.name(p)
        self.inits.append(numpy_helper.from_array(np.asarray(arr), nm))
        return nm

    def w(self, *shape, scale=None) -> str:
        scale = scale if scale is not None else 1.0 / np.sqrt(shape[-1] if len(shape) > 1 else shape[0])
        return self.const((self.rng.standard_normal(shape) * scale).astype(self.np), "w")

    def op(self, kind: str, inputs: list[str], **attrs) -> str:
        out = self.name(kind.lower())
        self.nodes.append(helper.make_node(kind, inputs, [out], name=out, **attrs))
        return out

    def model(self, name: str, inp: str, in_shape, out: str, out_shape, extra_inputs=()) -> bytes:
        inputs = [helper.make_tensor_value_info(inp, self.dtype, in_shape)]
        inputs += [helper.make_tensor_value_info(n, self.dtype, shp) for n, shp in extra_inputs]
        graph = helper.make_graph(self.nodes, name, inputs,
                                  [helper.make_tensor_value_info(out, self.dtype, out_shape)], self.inits)
        m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
        m.ir_version = 10
        onnx.checker.check_model(m)
        return m.SerializeToString()


# ----------------------------------------------------------------------------- transformer encoder
def encoder(dtype: int, layers: int, d: int, heads: int, ffn: int, seq: int, style: str,
            pool: bool = True, seed: int = 0, masked: bool = False) -> bytes:
    """Post-LN transformer encoder (BERT/e5 layout). Input is the already-embedded sequence.

    style "matmul": tensors are [1, seq, d], linear layers are MatMul + Add (what PyTorch exports).
    style "conv":   tensors are [1, d, seq, 1], linear layers are 1x1 convolutions (DLA-native).
    pool=False leaves out the final mean pooling, so two halves of a big model can be chained.
    masked=True adds a second input "mask" of shape [1, heads, seq, seq] that is added to the attention
    scores (0 = attend, -1e4 = padding). It is full-size on purpose: the DLA cannot broadcast.
    """
    g = G(dtype, seed)
    hd = d // heads
    if style == "matmul":
        x, in_shape = "input", [1, seq, d]

        def linear(t, cin, cout, gain=1.0):
            return g.op("Add", [g.op("MatMul", [t, g.w(cin, cout, scale=gain / np.sqrt(cin))]),
                                g.const(np.zeros(cout, g.np), "b")])

        def norm(t):
            return g.op("LayerNormalization", [t, g.const(np.ones(d, g.np)), g.const(np.zeros(d, g.np))], axis=-1)

        def split_heads(t):   # [1, seq, d] -> [1, heads, seq, hd]
            r = g.op("Reshape", [t, g.const(np.array([1, seq, heads, hd], np.int64))])
            return g.op("Transpose", [r], perm=[0, 2, 1, 3])

        def merge_heads(t):   # [1, heads, seq, hd] -> [1, seq, d]
            r = g.op("Transpose", [t], perm=[0, 2, 1, 3])
            return g.op("Reshape", [r, g.const(np.array([1, seq, d], np.int64))])
    else:
        x, in_shape = "input", [1, d, seq, 1]

        def linear(t, cin, cout, gain=1.0):
            return g.op("Conv", [t, g.w(cout, cin, 1, 1, scale=gain / np.sqrt(cin)), g.const(np.zeros(cout, g.np), "b")],
                        kernel_shape=[1, 1])

        def norm(t):          # layer norm over the channel axis of [1, d, seq, 1]
            r = g.op("Transpose", [t], perm=[0, 2, 3, 1])
            r = g.op("LayerNormalization", [r, g.const(np.ones((1, 1, 1, d), g.np)),
                                            g.const(np.zeros((1, 1, 1, d), g.np))], axis=-1)
            return g.op("Transpose", [r], perm=[0, 3, 1, 2])

        def split_heads(t):   # [1, d, seq, 1] -> [1, heads, seq, hd]
            r = g.op("Reshape", [t, g.const(np.array([1, heads, hd, seq], np.int64))])
            return g.op("Transpose", [r], perm=[0, 1, 3, 2])

        def merge_heads(t):   # [1, heads, seq, hd] -> [1, d, seq, 1]
            r = g.op("Transpose", [t], perm=[0, 1, 3, 2])
            return g.op("Reshape", [r, g.const(np.array([1, d, seq, 1], np.int64))])

    for _ in range(layers):
        q = split_heads(linear(x, d, d, gain=1.0 / np.sqrt(hd)))   # attention scale folded into Q
        k, v = split_heads(linear(x, d, d)), split_heads(linear(x, d, d))
        scores = g.op("MatMul", [q, g.op("Transpose", [k], perm=[0, 1, 3, 2])])
        if masked:
            scores = g.op("Add", [scores, "mask"])
        ctx = merge_heads(g.op("MatMul", [g.op("Softmax", [scores], axis=-1), v]))
        x = norm(g.op("Add", [x, linear(ctx, d, d)]))
        x = norm(g.op("Add", [x, linear(g.op("Gelu", [linear(x, d, ffn)]), ffn, d)]))
    if not pool:
        out_shape = in_shape
    elif style == "matmul":
        g.op("ReduceMean", [x, g.const(np.array([1], np.int64))], keepdims=1)
        out_shape = [1, 1, d]
    else:
        g.op("GlobalAveragePool", [x])          # mean pooling over the sequence
        out_shape = [1, d, 1, 1]
    g.nodes[-1].output[0] = "output"
    extra = [("mask", [1, heads, seq, seq])] if masked else []
    return g.model(f"encoder_{style}", "input", in_shape, "output", out_shape, extra)


# ----------------------------------------------------------------------------- conv1d speaker trunk
def tdnn(dtype: int, feats: int, frames: int, c: int, blocks: int, emb: int) -> bytes:
    """ECAPA-TDNN-like trunk as [1, C, 1, frames] convolutions: k=5 stem, residual k=3 blocks with
    squeeze-excitation, 1x1 head, mean pooling, linear embedding. Dilation is 1 (the DLA limit)."""
    g = G(dtype)

    def conv(t, cin, cout, k):
        return g.op("Conv", [t, g.w(cout, cin, 1, k, scale=np.sqrt(2 / (cin * k))), g.const(np.zeros(cout, g.np), "b")],
                    kernel_shape=[1, k], pads=[0, k // 2, 0, k // 2])

    x = g.op("Relu", [conv("input", feats, c, 5)])
    for _ in range(blocks):
        y = g.op("Relu", [conv(x, c, c, 1)])
        y = g.op("Relu", [conv(y, c, c, 3)])
        y = g.op("Relu", [conv(y, c, c, 1)])
        s = g.op("GlobalAveragePool", [y])
        s = g.op("Sigmoid", [conv(g.op("Relu", [conv(s, c, c // 4, 1)]), c // 4, c, 1)])
        x = g.op("Add", [x, g.op("Mul", [y, s])])
    x = g.op("Relu", [conv(x, c, c * 3, 1)])
    x = g.op("GlobalAveragePool", [x])
    x = conv(x, c * 3, emb, 1)
    g.nodes[-1].output[0] = "output"
    return g.model("tdnn", "input", [1, feats, 1, frames], "output", [1, emb, 1, 1])


@dataclass
class Case:
    name: str
    what: str
    build: callable          # dtype -> onnx bytes


CASES = [
    Case("encoder_e5small_seq64_matmul", "12 layers, d=384, 12 heads, 64 tokens (e5-small / MiniLM class), MatMul form",
         lambda t: encoder(t, 12, 384, 12, 1536, 64, "matmul")),
    Case("encoder_e5small_seq64_conv", "same, 1x1-conv form",
         lambda t: encoder(t, 12, 384, 12, 1536, 64, "conv")),
    Case("encoder_e5small_seq64_masked", "same, plus an attention-mask input (last 24 of 64 tokens are padding), no pooling",
         lambda t: encoder(t, 12, 384, 12, 1536, 64, "conv", pool=False, masked=True)),
    Case("encoder_e5small_seq128_conv", "same, 128 tokens",
         lambda t: encoder(t, 12, 384, 12, 1536, 128, "conv")),
    Case("encoder_e5base_seq64_conv", "12 layers, d=768, 12 heads, 64 tokens (e5-base class)",
         lambda t: encoder(t, 12, 768, 12, 3072, 64, "conv")),
    Case("encoder_e5large_seq64_conv", "24 layers, d=1024, 16 heads, 64 tokens (e5-large / bge-m3 class), one loadable",
         lambda t: encoder(t, 24, 1024, 16, 4096, 64, "conv")),
    Case("encoder_whispertiny_seq400_conv", "4 layers, d=384, 6 heads, 400 frames = 8 s (Smart Turn class)",
         lambda t: encoder(t, 4, 384, 6, 1536, 400, "conv")),
    Case("tdnn_c512_3s", "speaker trunk, 80 mel x 300 frames (3 s), C=512, 3 SE blocks, 192-d embedding",
         lambda t: tdnn(t, 80, 300, 512, 3, 192)),
    Case("tdnn_c1024_3s", "speaker trunk, C=1024 (ECAPA-TDNN large class)",
         lambda t: tdnn(t, 80, 300, 1024, 3, 192)),
]


def time_cpu(model_fp32: bytes, threads: int, seconds: float) -> tuple[float, dict, np.ndarray]:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(model_fp32, so, providers=["CPUExecutionProvider"])
    feeds = {}
    for inp in sess.get_inputs():
        if inp.name == "mask":
            m = np.zeros(inp.shape, np.float32)
            m[..., -24:] = -1e4          # treat the last 24 positions as padding
            feeds[inp.name] = m
        else:
            feeds[inp.name] = (np.random.default_rng(1).standard_normal(inp.shape) * 0.5).astype(np.float32)
    for _ in range(3):
        y = sess.run(None, feeds)[0]
    lat, t_end = [], time.perf_counter() + seconds
    while time.perf_counter() < t_end or len(lat) < 5:
        t = time.perf_counter()
        sess.run(None, feeds)
        lat.append(time.perf_counter() - t)
    return float(np.median(lat) * 1e3), feeds, y


def split_large(dev: DlaDevice, threads: int, seconds: float) -> dict:
    """e5-large / bge-m3 class (24 layers, d=1024) as two chained 12-layer loadables.

    The single 24-layer loadable (~660 MB) is rejected by the DLA compiler; two halves compile.
    The first half's output is copied through the host into the second half's input."""
    import onnxruntime as ort

    name = "encoder_e5large_seq64_split2"
    halves = [lambda t: encoder(t, 12, 1024, 16, 4096, 64, "conv", pool=False, seed=0),
              lambda t: encoder(t, 12, 1024, 16, 4096, 64, "conv", pool=True, seed=1)]
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    cpu = [ort.InferenceSession(h(F32), so, providers=["CPUExecutionProvider"]) for h in halves]
    x = (np.random.default_rng(1).standard_normal([1, 1024, 64, 1]) * 0.5).astype(np.float32)

    def cpu_run():
        return cpu[1].run(None, {"input": cpu[0].run(None, {"input": x})[0]})[0]

    y_cpu = cpu_run()
    lat, t_end = [], time.perf_counter() + seconds
    while time.perf_counter() < t_end or len(lat) < 5:
        t = time.perf_counter()
        cpu_run()
        lat.append(time.perf_counter() - t)
    cpu_ms = float(np.median(lat) * 1e3)
    r = {"name": name, "what": "24 layers, d=1024, 16 heads, 64 tokens, as two chained 12-layer loadables",
         "cpu_ms": round(cpu_ms, 2)}
    try:
        t0 = time.perf_counter()
        blobs = [compile_loadable(h(F16)) for h in halves]
        r["compile_s"] = round(time.perf_counter() - t0, 1)
        r["loadable_mb"] = round(sum(len(b) for b in blobs) / 1e6, 1)
        mods = [dev.load(b) for b in blobs]
    except (DlaBuildError, RuntimeError) as e:
        r.update(status="does not compile", error=str(e)[:400])
        print(f"{name:34} CPU {cpu_ms:8.2f} ms | NPU: does not compile: {str(e)[:200]}", flush=True)
        return r

    def npu_run():
        mods[0].write(0, x)
        mods[0].run()
        mods[1].write(0, mods[0].read(0))
        mods[1].run()
        return mods[1].read(0)

    try:
        for _ in range(3):
            y = npu_run().astype(np.float32).reshape(y_cpu.shape)
        lat, t_end = [], time.perf_counter() + seconds
        while time.perf_counter() < t_end or len(lat) < 5:
            t = time.perf_counter()
            npu_run()
            lat.append(time.perf_counter() - t)
        cos = float(np.dot(y.ravel(), y_cpu.ravel()) / (np.linalg.norm(y) * np.linalg.norm(y_cpu)))
        r.update(status="ok", npu_ms=round(float(np.median(lat) * 1e3), 2),
                 npu_ms_p99=round(float(np.percentile(lat, 99) * 1e3), 2),
                 speedup=round(cpu_ms / float(np.median(lat) * 1e3), 2),
                 rel_l2_err_pct=round(float(np.linalg.norm(y - y_cpu) / np.linalg.norm(y_cpu)) * 100, 3),
                 cosine=round(cos, 5), finite=bool(np.isfinite(y).all()),
                 note="NPU time includes host copies of input, intermediate and output")
        print(f"{name:34} CPU {cpu_ms:8.2f} ms | NPU {r['npu_ms']:8.2f} ms (p99 {r['npu_ms_p99']}) | "
              f"x{r['speedup']:<5} | rel err {r['rel_l2_err_pct']}% cos {r['cosine']} | {r['loadable_mb']} MB", flush=True)
    finally:
        for m in mods:
            m.close()
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--cpu-threads", type=int, default=8)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    rows = []
    with DlaDevice() as dev:
        for case in [c for c in CASES if not args.only or args.only in c.name]:
            r = {"name": case.name, "what": case.what}
            cpu_ms, feeds, y_cpu = time_cpu(case.build(F32), args.cpu_threads, args.seconds)
            r["cpu_ms"] = round(cpu_ms, 2)
            try:
                t0 = time.perf_counter()
                loadable = compile_loadable(case.build(F16))
                r["compile_s"] = round(time.perf_counter() - t0, 1)
                r["loadable_mb"] = round(len(loadable) / 1e6, 1)
                mod = dev.load(loadable)
            except (DlaBuildError, RuntimeError) as e:
                r.update(status="does not compile", error=str(e)[:400])
                rows.append(r)
                print(f"{case.name:34} CPU {cpu_ms:8.2f} ms | NPU: does not compile: {str(e)[:200]}", flush=True)
                continue
            try:
                for i, t in enumerate(mod.inputs):
                    mod.write(i, feeds[t.name].reshape(t.shape))
                for _ in range(3):
                    mod.run()
                y = mod.read(0).astype(np.float32).reshape(y_cpu.shape)
                lat, t_end = [], time.perf_counter() + args.seconds
                while time.perf_counter() < t_end or len(lat) < 5:
                    t = time.perf_counter()
                    mod.run()
                    lat.append(time.perf_counter() - t)
                denom = float(np.linalg.norm(y_cpu)) or 1.0
                cos = float(np.dot(y.ravel(), y_cpu.ravel()) / ((np.linalg.norm(y) * np.linalg.norm(y_cpu)) or 1.0))
                r.update(status="ok", npu_ms=round(float(np.median(lat) * 1e3), 2),
                         npu_ms_p99=round(float(np.percentile(lat, 99) * 1e3), 2),
                         speedup=round(cpu_ms / float(np.median(lat) * 1e3), 2),
                         rel_l2_err_pct=round(float(np.linalg.norm(y - y_cpu)) / denom * 100, 3),
                         cosine=round(cos, 5), finite=bool(np.isfinite(y).all()))
                print(f"{case.name:34} CPU {cpu_ms:8.2f} ms | NPU {r['npu_ms']:8.2f} ms (p99 {r['npu_ms_p99']}) | "
                      f"x{r['speedup']:<5} | rel err {r['rel_l2_err_pct']}% cos {r['cosine']} | {r['loadable_mb']} MB",
                      flush=True)
            finally:
                mod.close()
            rows.append(r)
        if not args.only or "split" in args.only:
            rows.append(split_large(dev, args.cpu_threads, args.seconds))
    tag = "_partial" if args.only else ""
    (RESULTS / f"voice_models{tag}.json").write_text(json.dumps(
        {"meta": {"date": time.strftime("%Y-%m-%d %H:%M"), "cpu_threads": args.cpu_threads}, "cases": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
