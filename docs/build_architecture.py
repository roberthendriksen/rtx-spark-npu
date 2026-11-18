"""Build docs/architecture.html: the NPU's hardware and software architecture as pictures.

    python docs/build_architecture.py

One self-contained HTML file: inline SVG diagrams, charts drawn from results/*.json, no scripts,
no network. Charts read the measurement files, so re-running after a new benchmark updates them.
"""
from __future__ import annotations

import json
import math
from html import escape as esc
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


# --------------------------------------------------------------------------- SVG helpers
def defs(p: str) -> str:
    m = ('<marker id="{id}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" '
         'orient="auto-start-reverse"><path d="M0 0L8 4L0 8z" class="{cls}"/></marker>')
    return "<defs>" + m.format(id=p + "a", cls="mk") + m.format(id=p + "n", cls="mk-npu") + "</defs>"


def rect(x, y, w, h, cls="bx", rx=6) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" class="{cls}"/>'


def text(x, y, s, cls="dt", anchor="start") -> str:
    a = "" if anchor == "start" else f' text-anchor="{anchor}"'
    return f'<text x="{x}" y="{y}" class="{cls}"{a}>{esc(s)}</text>'


def line(d: str, p: str, cls="ln", end=True, start=False) -> str:
    mk = "n" if "npu" in cls else "a"
    me = f' marker-end="url(#{p}{mk})"' if end else ""
    ms = f' marker-start="url(#{p}{mk})"' if start else ""
    return f'<path d="{d}" class="{cls}"{me}{ms}/>'


def node(x, y, w, h, title, sub="", cls="bx", tag="", title_cls="dt-f") -> str:
    out = [rect(x, y, w, h, cls), text(x + 14, y + 22, title, title_cls)]
    if sub:
        out.append(text(x + 14, y + 40, sub, "dt-s"))
    if tag:
        out.append(text(x + w - 12, y + 21, tag.upper(), "dt-h", "end"))
    return "".join(out)


def svg(w: int, h: int, label: str, body: str, min_w: int = 760) -> str:
    return (f'<div class="scroll"><svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}" '
            f'style="min-width:{min_w}px">{body}</svg></div>')


# --------------------------------------------------------------------------- diagrams
def diagram_where() -> str:
    p = "A"
    b = [defs(p)]
    b.append(rect(16, 28, 612, 348, "bx-out", 10))
    b.append(text(32, 52, "ONE CHIP · NVIDIA RTX SPARK N1X", "dt-h"))
    # CPU
    b.append(rect(36, 70, 170, 160))
    b += [text(50, 96, "20 Arm CPU cores", "dt-b"), text(50, 116, "your process", "dt-s"),
          text(50, 132, "and the drivers", "dt-s")]
    # GPU
    b.append(rect(226, 70, 180, 160))
    b += [text(240, 96, "Blackwell GPU", "dt-b"), text(240, 116, "6144 CUDA cores", "dt-s"),
          text(240, 132, "holds the CUDA context", "dt-s"), text(240, 148, "the NPU path needs", "dt-s")]
    # NPU
    b.append(rect(426, 70, 186, 160, "bx-npu"))
    b += [text(440, 96, "NPU (a DLA)", "dt-b"), text(440, 116, "≈ 1,940 MACs per clock cycle", "dt-s"),
          text(440, 132, "clock 1105 of 1562 MHz", "dt-s"), text(440, 148, "1.0–2.3 W", "dt-s"),
          text(440, 174, "0x19000000  256 KiB", "dt-m"), text(440, 190, "0x31380000   64 KiB", "dt-m"),
          text(440, 212, "2 interrupts · \\_SB.DLA0", "dt-s")]
    # memory
    b.append(rect(36, 290, 576, 64))
    b += [text(50, 316, "Unified LPDDR5X memory", "dt-b"),
          text(50, 336, "weights and tensors for all three; the NPU only adds a small on-chip buffer", "dt-s")]
    for x in (121, 316):
        b.append(line(f"M{x} 232V288", p, start=True))
    b.append(line("M519 232V288", p, "ln-npu", start=True))
    b.append(text(308, 264, "CUDA memory", "dt-s", "end"))
    b.append(text(511, 264, "DMA, behind the IOMMU", "dt-s", "end"))
    # Windows view
    b.append(text(656, 52, "WINDOWS SEES TWO ADAPTERS", "dt-h"))
    b.append(rect(656, 70, 288, 84))
    b += [text(670, 96, "NVIDIA RTX Spark N1X", "dt-b"), text(670, 116, "WDDM · nvidia-smi device 0", "dt-s"),
          text(670, 134, "engines: 3D, Copy, Video, …", "dt-s")]
    b.append(rect(656, 170, 288, 104, "bx-npu"))
    b += [text(670, 196, "NVIDIA NPU", "dt-b"), text(670, 216, "MCDM, compute only · nvidia-smi device 1", "dt-s"),
          text(670, 234, "one engine: NPU_0", "dt-s"), text(670, 256, "ACPI\\NVDA200A", "dt-m")]
    b.append(rect(656, 300, 288, 54))
    b += [text(670, 323, "nvlddmkm.sys", "dt-f"), text(670, 341, "one kernel driver serves both", "dt-s")]
    b.append(line("M800 274V298", p))
    b.append(line("M944 112H953V327H946", p))
    b.append(line("M612 150H634V222H654", p, "ln-npu"))
    return svg(960, 392, "The NPU sits on the same chip as the CPU and GPU and shares their memory; "
                         "Windows lists it as a second adapter of the GPU driver.", "".join(b))


def diagram_inside() -> str:
    p = "B"
    b = [defs(p), text(16, 50, "MAIN PIPELINE", "dt-h"), text(16, 200, "SIDE UNITS", "dt-h")]
    b.append(rect(16, 64, 116, 80))
    b += [text(30, 98, "Unified", "dt-b"), text(30, 116, "memory", "dt-b")]
    b.append(rect(176, 64, 140, 80, "bx rep"))
    b += [text(190, 90, "Buffer", "dt-b"), text(190, 108, "on-chip SRAM", "dt-s"), text(190, 124, "2.5 MiB (reported)", "dt-s")]
    b.append(rect(360, 64, 190, 80, "bx-npu"))
    b += [text(374, 90, "Convolution core", "dt-b"), text(374, 108, "multiply-accumulate array", "dt-s"),
          text(374, 124, "≈ 1,940 MACs per clock cycle", "dt-s")]
    b.append(rect(594, 64, 150, 80))
    b += [text(608, 90, "SDP", "dt-b"), text(608, 108, "activation, add,", "dt-s"), text(608, 124, "multiply, bias", "dt-s")]
    b.append(rect(788, 64, 156, 80))
    b += [text(802, 90, "PDP · CDP", "dt-b"), text(802, 108, "pooling,", "dt-s"), text(802, 124, "normalisation", "dt-s")]
    for x0, x1 in ((132, 174), (316, 358), (550, 592), (744, 786)):
        b.append(line(f"M{x0} 104H{x1}", p, "ln-npu"))
    b.append(text(154, 94, "in", "dt-s", "middle"))
    b.append(line("M866 144V172H74V146", p, "ln-npu"))
    b.append(text(470, 166, "results written back to memory", "dt-s", "middle"))
    b.append(rect(176, 214, 140, 60))
    b += [text(190, 238, "Rubik", "dt-b"), text(190, 256, "reshape, transpose", "dt-s")]
    b.append(rect(360, 214, 140, 60))
    b += [text(374, 238, "BDMA", "dt-b"), text(374, 256, "memory copy", "dt-s")]
    b.append(rect(544, 214, 180, 60))
    b += [text(558, 238, "RVV vector unit", "dt-b"), text(558, 256, "other math (reported)", "dt-s")]
    b.append(rect(768, 214, 176, 60, "bx rep"))
    b += [text(782, 238, "Control processor", "dt-b"), text(782, 256, "RISC-V (reported)", "dt-s")]
    return svg(960, 290, "Inside the NPU: data flows from memory through a buffer, the convolution core, "
                         "an activation stage and a pooling stage, then back to memory.", "".join(b))


def diagram_stack() -> str:
    p = "C"
    b = [defs(p), text(30, 34, "COMPILE · ONCE, OFFLINE", "dt-h"), text(372, 34, "RUN · EVERY INFERENCE", "dt-h"),
         text(748, 34, "ROUTES THAT DO NOT WORK", "dt-h"), text(748, 204, "GPU SIDE", "dt-h")]
    # compile column
    b.append(node(30, 50, 290, 52, "model.onnx", "static shapes · batch 1 · FP16 · 4-D tensors"))
    b.append(node(30, 134, 290, 52, "nvinfer_11.dll", "TensorRT 11.5 parses and checks every layer", "bx-trt", "TensorRT"))
    b.append(node(30, 218, 290, 52, "nvdla_compiler.dll", "the DLA compiler", "bx-trt", "TensorRT"))
    b.append(node(30, 302, 290, 52, "loadable", "one file per model, up to about 400 MB"))
    for y in (102, 186, 270):
        b.append(line(f"M175 {y}V{y + 30}", p))
    b.append(line("M320 328H346V160H370", p))
    # run column
    b.append(node(372, 50, 320, 52, "your code", "Python (ctypes), C or C++"))
    b.append(node(372, 134, 320, 52, "cudla.dll", "public cuDLA API · 18 functions", "bx-cuda", "CUDA redist"))
    b.append(node(372, 218, 320, 52, "nvcudla.dll", "cudlaDrv* · the same 18 functions", "bx-drv", "driver"))
    b.append(node(372, 302, 320, 52, "nvdla_runtime.dll", "loads the loadable, binds memory, submits", "bx-drv", "driver"))
    b.append(node(372, 398, 320, 52, "dxgkrnl.sys", "Windows graphics kernel · schedules NPU_0", "bx", "Windows"))
    b.append(node(372, 470, 568, 52, "nvlddmkm.sys", "NVIDIA kernel driver · the NPU is its compute-only (MCDM) adapter",
                  "bx-drv", "driver"))
    b.append(rect(372, 548, 320, 40, "bx-npu"))
    b.append(text(386, 573, "NPU hardware", "dt-b"))
    b.append(text(678, 573, "\\_SB.DLA0", "dt-m", "end"))
    b.append(rect(748, 548, 192, 40))
    b.append(text(762, 573, "GPU", "dt-b"))
    for y0, y1, lab in ((102, 132, "cudlaSubmitTask, …"), (186, 216, "forwards 1:1"), (270, 300, "load · bind · submit"),
                        (354, 396, "D3DKMT calls through gdi32.dll"), (450, 468, ""), (522, 546, "")):
        b.append(line(f"M532 {y0}V{y1}", p, "ln-npu"))
        if lab:
            b.append(text(542, y0 + (16 if y1 - y0 > 34 else (y1 - y0) / 2 + 4), lab, "dt-s"))
    b.append(f'<path d="M372 376H940" class="bound"/>')
    b.append(text(940, 370, "USER MODE", "dt-h", "end"))
    b.append(text(940, 390, "KERNEL MODE", "dt-h", "end"))
    # GPU side
    b.append(node(748, 218, 192, 52, "nvcuda.dll", "context · stream · memory", "bx-drv", "driver"))
    b.append(line("M692 244H746", p))
    b.append(text(720, 236, "uses", "dt-s", "middle"))
    b.append(line("M844 270V468", p))
    b.append(text(852, 300, "tensors live in", "dt-s"))
    b.append(text(852, 315, "CUDA memory", "dt-s"))
    b.append(line("M844 522V546", p))
    # dead ends
    for i, s in enumerate(("DirectML, Windows ML (reported)", "cuDLA standalone mode", "TensorRT GPU engines")):
        y = 62 + i * 22
        b.append(f'<path d="M750 {y - 9}l8 8m0 -8l-8 8" class="cross"/>')
        b.append(text(766, y, s, "dt-s"))
    return svg(960, 604, "Software stack: TensorRT and the DLA compiler turn an ONNX model into a loadable; at run "
                         "time your code calls cudla.dll, which goes through nvcudla.dll, nvdla_runtime.dll, the "
                         "Windows graphics kernel and the NVIDIA kernel driver to the NPU.", "".join(b))


def diagram_hybrid() -> str:
    p = "D"
    b = [defs(p)]
    for y in (44, 114, 184):
        b.append(rect(150, y, 794, 56, "lane", 6))
    b += [text(16, 77, "Your process (CPU)", "dt-b"), text(16, 147, "GPU, CUDA stream", "dt-b"), text(16, 217, "NPU", "dt-b")]
    b.append(rect(166, 54, 140, 36))
    b.append(text(236, 77, "copy input in", "dt", "middle"))
    b.append(rect(800, 54, 128, 36))
    b.append(text(864, 77, "copy output out", "dt", "middle"))
    b.append('<path d="M312 72H794" class="wait"/>')
    b.append(text(553, 66, "waits in cuStreamSynchronize", "dt-s", "middle"))
    b.append(rect(326, 124, 120, 36))
    b.append(text(386, 147, "queues the task", "dt", "middle"))
    b.append(rect(680, 124, 110, 36))
    b.append(text(735, 147, "signals done", "dt", "middle"))
    b.append(rect(466, 194, 194, 36, "bx-npu"))
    b.append(text(563, 217, "runs the loadable", "dt-b", "middle"))
    b.append(line("M286 90L344 123", p))
    b.append(text(340, 110, "cudlaSubmitTask", "dt-s"))
    b.append(line("M426 160L484 193", p, "ln-npu"))
    b.append(line("M640 194L698 161", p, "ln-npu"))
    b.append(line("M770 124L828 91", p))
    return svg(960, 252, "One inference in hybrid mode: the process copies the input into CUDA memory and submits the "
                         "task on a CUDA stream, the NPU runs it, the stream signals completion, and the process "
                         "copies the output back.", "".join(b))


def diagram_split() -> str:
    p = "E"
    b = [defs(p)]
    b.append(rect(16, 40, 110, 64))
    b += [text(30, 66, "token ids", "dt-b"), text(30, 86, "64 integers", "dt-s")]
    b.append(rect(176, 40, 170, 64))
    b += [text(190, 66, "embedding lookup", "dt-b"), text(190, 86, "NumPy, on the CPU", "dt-s")]
    b.append(rect(396, 40, 170, 64, "bx-npu"))
    b += [text(410, 66, "layers 1–12", "dt-b"), text(410, 86, "loadable A, on the NPU", "dt-s")]
    b.append(rect(596, 40, 170, 64, "bx-npu"))
    b += [text(610, 66, "layers 13–24", "dt-b"), text(610, 86, "loadable B, on the NPU", "dt-s")]
    b.append(rect(816, 40, 128, 64))
    b += [text(830, 66, "mean pooling", "dt-b"), text(830, 86, "on the CPU", "dt-s")]
    for x0, x1, cls in ((126, 174, "ln"), (346, 394, "ln-npu"), (566, 594, "ln-npu"), (766, 814, "ln-npu")):
        b.append(line(f"M{x0} 72H{x1}", p, cls))
    b += [text(371, 128, "[1, d, 64, 1]", "dt-m", "middle"), text(581, 128, "copy", "dt-s", "middle"),
          text(791, 128, "[1, d, 64, 1]", "dt-m", "middle")]
    return svg(960, 142, "A large encoder on the NPU: the embedding lookup runs on the CPU, two chained loadables run "
                         "on the NPU, and pooling runs on the CPU.", "".join(b))


# --------------------------------------------------------------------------- charts
def bar_path(x0: float, x1: float, y: float, h: float, r: float = 4) -> str:
    """Horizontal bar: square at the baseline (x0), rounded at the data end (x1)."""
    if x1 - x0 < r + 1:
        return f"M{x0:.1f} {y}H{max(x1, x0 + 1.5):.1f}V{y + h}H{x0:.1f}Z"
    return (f"M{x0:.1f} {y}H{x1 - r:.1f}Q{x1:.1f} {y} {x1:.1f} {y + r}V{y + h - r}"
            f"Q{x1:.1f} {y + h} {x1 - r:.1f} {y + h}H{x0:.1f}Z")


def axis(x0, x1, y0, y1, ticks, fx, fmt=str) -> str:
    out = []
    for t in ticks:
        x = fx(t)
        out.append(f'<path d="M{x:.1f} {y0}V{y1}" class="grid"/>')
        out.append(text(round(x, 1), y1 + 16, fmt(t), "dt-t", "middle"))
    out.append(f'<path d="M{x0} {y0}V{y1}" class="base"/>')
    return "".join(out)


def chart_tops(rows) -> str:
    x0, x1, top, rh, bh = 300, 880, 44, 30, 14
    fx = lambda v: x0 + v / 7 * (x1 - x0)
    y1 = top + len(rows) * rh
    b = [axis(x0, x1, top, y1, range(8), fx)]
    for v, lab, yy in ((4.3, "measured peak 4.3", 34), (6.1, "6.1 at the 1562 MHz clock limit (extrapolated)", 18)):
        b.append(f'<path d="M{fx(v):.1f} {yy + 4}V{y1}" class="ref"/>')
        b.append(text(round(fx(v) - 5, 1), yy, lab, "dt-s", "end"))
    for i, (label, lo, hi, src) in enumerate(rows):
        y = top + i * rh + (rh - bh) / 2
        tip = rng(lo, hi)
        b.append(f"<g><title>{esc(label)}: {tip} TOPS</title>")
        b.append(text(x0 - 12, y + bh - 3, label, "dt", "end"))
        if hi is not None:
            b.append(f'<path d="{bar_path(x0, fx(hi), y, bh)}" class="bar-npu wash"/>')
        b.append(f'<path d="{bar_path(x0, fx(lo), y, bh)}" class="bar-npu"/>')
        b.append(text(round(fx(hi if hi is not None else lo) + 7, 1), y + bh - 3,
                      tip + (" · varies run to run" if src == "all runs" else ""), "dt-v"))
        b.append("</g>")
    b.append(text((x0 + x1) / 2, y1 + 36, "TOPS (FP16) = 2 × multiply-accumulates per second ÷ 10¹²", "dt-t", "middle"))
    return svg(960, y1 + 46, "Throughput by layer type: ordinary convolutions reach 3.2 to 4.3 TOPS, wide layers and "
                             "transformer encoders 0.7 to 2.7, depthwise convolutions 0.07.", "".join(b))


def chart_gap() -> str:
    x0, x1, top, rh, bh = 300, 880, 14, 34, 16
    fx = lambda v: x0 + v / 60 * (x1 - x0)
    rows = [("Advertised, press coverage", 50, 55, "bar-claim", "50–55"),
            ("At the 1562 MHz clock limit, extrapolated", 6.1, None, "bar-npu wash", "6.1"),
            ("Measured on Sparky today", 4.3, None, "bar-npu", "4.3")]
    y1 = top + len(rows) * rh
    b = [axis(x0, x1, top, y1, range(0, 61, 10), fx)]
    for i, (label, lo, hi, cls, val) in enumerate(rows):
        y = top + i * rh + (rh - bh) / 2
        b.append(f"<g><title>{esc(label)}: {val} TOPS</title>")
        b.append(text(x0 - 12, y + bh - 4, label, "dt", "end"))
        if hi is not None:
            b.append(f'<path d="{bar_path(x0, fx(hi), y, bh)}" class="{cls} wash"/>')
        b.append(f'<path d="{bar_path(x0, fx(lo), y, bh)}" class="{cls}"/>')
        b.append(text(round(fx(hi or lo) + 7, 1), y + bh - 4, val, "dt-v"))
        b.append("</g>")
    b.append(text((x0 + x1) / 2, y1 + 36, "TOPS", "dt-t", "middle"))
    return svg(960, y1 + 46, "The advertised rating is 50 to 55 TOPS; 4.3 were measured and about 6.1 would be "
                             "reached at the clock limit.", "".join(b))


def chart_latency(rows) -> str:
    x0, x1, top, rh, bh, gap = 330, 880, 14, 66, 12, 4
    fx = lambda v: x0 + v / 100 * (x1 - x0)
    y1 = top + len(rows) * rh
    b = [axis(x0, x1, top, y1, range(0, 101, 10), fx)]
    series = (("cpu", "bar-s1", "CPU, 8 threads"), ("npu", "bar-s2", "NPU, GPU idle"), ("busy", "bar-s3", "NPU, GPU saturated"))
    for i, r in enumerate(rows):
        yr = top + i * rh + (rh - (3 * bh + 2 * gap)) / 2
        b.append(text(x0 - 12, yr + 16, r["label"], "dt", "end"))
        b.append(text(x0 - 12, yr + 32, r["sub"], "dt-s", "end"))
        for j, (key, cls, name) in enumerate(series):
            y = yr + j * (bh + gap)
            v = r.get(key)
            if v is None:
                b.append(text(x0 + 6, y + bh - 2, "not measured", "dt-t"))
                continue
            lo, hi = (v, None) if not isinstance(v, tuple) else v
            val = rng(lo, hi)
            b.append(f"<g><title>{esc(r['label'])} · {name}: {val} ms</title>")
            if hi is not None:
                b.append(f'<path d="{bar_path(x0, fx(hi), y, bh)}" class="{cls} wash"/>')
            b.append(f'<path d="{bar_path(x0, fx(lo), y, bh)}" class="{cls}"/>')
            b.append(text(round(fx(hi or lo) + 7, 1), y + bh - 2, val, "dt-v"))
            b.append("</g>")
        if i:
            b.append(f'<path d="M{x0 - 300} {top + i * rh}H{x1}" class="grid"/>')
    b.append(text((x0 + x1) / 2, y1 + 36, "milliseconds per inference, median (lower is better)", "dt-t", "middle"))
    return svg(960, y1 + 46, "Latency of six model shapes on eight CPU threads, on the NPU with an idle GPU, and on the "
                             "NPU with a saturated GPU.", "".join(b))


def chart_wake(w: dict) -> str:
    x0, x1, top, rh = 300, 880, 30, 28
    fx = lambda v: x0 + math.log10(v) / 3 * (x1 - x0)
    rows = [("called back to back", w["back_to_back_ms"], None, 0)]
    for gap, d in w["idle_gap"].items():
        rows.append((f"after {float(gap):g} s idle", d["median_ms"], d["max_ms"], 0))
    kw = w["keep_warm"]
    rows += [("3 s idle, no keep-warm", kw["none"]["median_ms"], kw["none"]["max_ms"], 1),
             ("3 s idle, NPU tick every 100 ms", kw["npu-tick"]["median_ms"], kw["npu-tick"]["max_ms"], 1),
             ("3 s idle, CUDA tick every 100 ms", kw["gpu-tick"]["median_ms"], kw["gpu-tick"]["max_ms"], 1)]
    sep = 40
    n0 = sum(1 for r in rows if not r[3])
    y1 = top + len(rows) * rh + sep
    b = [axis(x0, x1, top - 14, y1, (1, 10, 100, 1000), fx, lambda t: f"{t:,} ms")]
    b.append(text(16, top - 2, "NO KEEP-WARM", "dt-h"))
    y_sep = top + n0 * rh + 10
    b.append(f'<path d="M16 {y_sep}H{x1}" class="grid"/>')
    b.append(text(16, y_sep + 18, "3 S GAPS, WITH AND WITHOUT A TICK", "dt-h"))
    for i, (label, med, worst, grp) in enumerate(rows):
        y = top + i * rh + rh / 2 + (sep if grp else 0)
        b.append(f"<g><title>{esc(label)}: median {num(med)} ms" + (f", worst {num(worst)} ms" if worst else "") + "</title>")
        b.append(text(x0 - 12, y + 4, label, "dt", "end"))
        xm = fx(med)
        if worst:
            xw = fx(worst)
            b.append(f'<path d="M{xm:.1f} {y}H{xw:.1f}" class="rng"/>')
            b.append(f'<circle cx="{xw:.1f}" cy="{y}" r="5" class="dot-open"/>')
            b.append(text(round(xw + 10, 1), y + 4, num(worst), "dt-v"))
        b.append(f'<circle cx="{xm:.1f}" cy="{y}" r="5.5" class="dot"/>')
        b.append(text(round(xm - 10, 1), y + 4, num(med), "dt-v", "end"))
        b.append("</g>")
    b.append(text((x0 + x1) / 2, y1 + 36, "latency of the next call, logarithmic scale", "dt-t", "middle"))
    return svg(960, y1 + 46, "Latency of the first call after an idle gap: from half a second of idleness the worst "
                             "case reaches hundreds of milliseconds, and a tick every 100 ms keeps it near 10.", "".join(b))


def num(v: float) -> str:
    """Chart value: two decimals below 1, one below 100, none above."""
    return f"{v:.2f}".rstrip("0") if v < 1 else f"{v:.1f}" if v < 100 else f"{v:,.0f}"


def rng(lo, hi=None) -> str:
    return num(lo) if hi is None else f"{num(lo)}–{num(hi)}"


def table(head, rows) -> str:
    th = "".join(f"<th>{esc(h)}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{esc(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f'<details><summary>Show the numbers</summary><div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div></details>'


# --------------------------------------------------------------------------- data
def load():
    tops = {c["name"]: c for c in json.loads((RESULTS / "tops_ac.json").read_text())["cases"] if c["status"] == "ok"}
    vm = {c["name"]: c for c in json.loads((RESULTS / "voice_models.json").read_text())["cases"]}
    gc = json.loads((RESULTS / "gpu_contention.json").read_text())
    wake = json.loads((RESULTS / "wakeup.json").read_text())
    t = lambda n: tops[n]["tops"]
    best256 = max(max(c["tops"], c["tops_pipelined"]) for n, c in tops.items() if n.startswith("c256_k3_hw64_d16"))
    # (label, value or low end, high end of a run-to-run range, source). Ranges combine tops_ac.json with
    # the earlier runs recorded in ARCHITECTURE.md section 4.2; encoder rates are MACs / measured latency.
    tops_rows = [
        ("3×3 conv, 256 channels", round(best256, 1), None, "tops_ac.json"),
        ("5×5 conv, 128 channels", round(t("c128_k5_hw64_d8"), 1), None, "tops_ac.json"),
        ("1×1 conv, 512–1024 channels", round(t("c512_k1_hw64_d16"), 1), None, "tops_ac.json"),
        ("3×3 conv, 64–128 channels", round(t("c128_k3_hw64_d8"), 1), None, "tops_ac.json"),
        ("1×1 conv, 256 channels", round(t("c256_k1_hw64_d16"), 1), None, "tops_ac.json"),
        ("3×3 conv, 192 channels", round(t("c192_k3_hw64_d16"), 1), None, "tops_ac.json"),
        ("3×3 conv, 512 channels", 0.9, 2.7, "all runs"),
        ("Transformer encoder, 768–1024 wide", 1.5, 1.8, "voice_models.json"),
        ("3×3 conv, 1024 channels", 0.7, 1.4, "all runs"),
        ("Transformer encoder, 384 wide", 0.75, None, "voice_models.json"),
        ("Depthwise 3×3, 256 channels", round(t("depthwise_c256_k3_hw64_d16"), 2), None, "tops_ac.json"),
    ]
    g = lambda k: gc[k]["median_ms"]
    lat_rows = [
        {"label": "Encoder, 12 layers × 384 wide", "sub": "64 tokens · e5-small class",
         "cpu": vm["encoder_e5small_seq64_conv"]["cpu_ms"], "npu": g("e5small_64tok.gpu_idle.back_to_back"),
         "busy": g("e5small_64tok.gpu_saturated.back_to_back")},
        {"label": "Encoder, 12 layers × 768 wide", "sub": "64 tokens · e5-base class",
         "cpu": vm["encoder_e5base_seq64_conv"]["cpu_ms"], "npu": g("e5base_64tok.gpu_idle.back_to_back"),
         "busy": g("e5base_64tok.gpu_saturated.back_to_back")},
        {"label": "Encoder, 24 layers × 1024 wide", "sub": "64 tokens · e5-large class · two loadables",
         "cpu": (vm["encoder_e5large_seq64_conv"]["cpu_ms"], vm["encoder_e5large_seq64_split2"]["cpu_ms"]),
         "npu": g("e5large_64tok_split2.gpu_idle.back_to_back"), "busy": g("e5large_64tok_split2.gpu_saturated.back_to_back")},
        {"label": "Encoder, 4 layers × 384 wide", "sub": "400 frames · Whisper-tiny class",
         "cpu": vm["encoder_whispertiny_seq400_conv"]["cpu_ms"], "npu": vm["encoder_whispertiny_seq400_conv"]["npu_ms"], "busy": None},
        {"label": "Conv1d speaker trunk, 512 channels", "sub": "300 frames",
         "cpu": vm["tdnn_c512_3s"]["cpu_ms"], "npu": vm["tdnn_c512_3s"]["npu_ms"], "busy": None},
        {"label": "Conv1d speaker trunk, 1024 channels", "sub": "300 frames",
         "cpu": vm["tdnn_c1024_3s"]["cpu_ms"], "npu": vm["tdnn_c1024_3s"]["npu_ms"], "busy": None},
    ]
    return tops_rows, lat_rows, wake


def fmt(v) -> str:
    if v is None:
        return "not measured"
    return rng(*v) if isinstance(v, tuple) else num(v)


# --------------------------------------------------------------------------- page
CSS = r"""
/* Layout: one reading column; every section is a claim, a picture, then two or three short notes. */
:root{
  --bg:#f2f3f4; --panel:#fcfcfb; --panel-2:#eef0f1; --ink:#0b0b0b; --ink-2:#52514e; --ink-3:#7c7b76; --rule:#dadbdc;
  --npu:#eb6834; --npu-wash:#fdebe2; --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a;
  --drv-wash:#e3eefb; --cuda-wash:#dff4ec; --trt-wash:#fbf0cf; --claim:#8d8c87;
  --font:"Segoe UI Variable Text","Segoe UI",system-ui,-apple-system,"Helvetica Neue",Arial,sans-serif;
  --display:"Segoe UI Variable Display","Segoe UI",system-ui,-apple-system,"Helvetica Neue",Arial,sans-serif;
  --mono:"Cascadia Code","Cascadia Mono",Consolas,ui-monospace,"SF Mono",Menlo,monospace;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#111312; --panel:#1a1a19; --panel-2:#242523; --ink:#f0efec; --ink-2:#c3c2b7; --ink-3:#8f8e86; --rule:#353633;
  --npu:#d95926; --npu-wash:#3b2217; --s1:#3987e5; --s2:#d95926; --s3:#199e70;
  --drv-wash:#172a43; --cuda-wash:#12312a; --trt-wash:#3a2f12; --claim:#77766f; color-scheme:dark } }
:root[data-theme="dark"]{
  --bg:#111312; --panel:#1a1a19; --panel-2:#242523; --ink:#f0efec; --ink-2:#c3c2b7; --ink-3:#8f8e86; --rule:#353633;
  --npu:#d95926; --npu-wash:#3b2217; --s1:#3987e5; --s2:#d95926; --s3:#199e70;
  --drv-wash:#172a43; --cuda-wash:#12312a; --trt-wash:#3a2f12; --claim:#77766f; color-scheme:dark }
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 var(--font)}
.wrap{max-width:1040px;margin:0 auto;padding-block:40px 72px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));display:flex;flex-direction:column;gap:56px}
header{display:flex;flex-direction:column;gap:14px}
.eyebrow{font:600 11.5px/1 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3)}
h1{font:600 clamp(28px,4.4vw,40px)/1.12 var(--display);letter-spacing:-.015em;margin:0;text-wrap:balance}
h2{font:600 23px/1.2 var(--display);letter-spacing:-.01em;margin:0;text-wrap:balance}
p{margin:0}
.lead{font-size:17px;max-width:68ch;color:var(--ink-2)}
.lead b{color:var(--ink);font-weight:600}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:0;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}
.fact{padding:16px 18px 16px 0;display:flex;flex-direction:column;gap:2px}
.fact b{font:600 30px/1.1 var(--display);letter-spacing:-.02em}
.fact b small{font-size:15px;font-weight:600;color:var(--ink-2);letter-spacing:0}
.fact span{font-size:13px;color:var(--ink-2)}
section{display:flex;flex-direction:column;gap:16px}
.claim{max-width:72ch;color:var(--ink-2)}
figure{margin:0;background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:18px 18px 14px;display:flex;flex-direction:column;gap:12px;min-width:0}
figcaption{font-size:13px;color:var(--ink-2);max-width:86ch}
.scroll{overflow-x:auto;min-width:0}
svg{display:block;width:100%;height:auto;color:var(--ink)}
.notes{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px 28px}
.notes p{font-size:14px;color:var(--ink-2);min-width:0}
.notes b{color:var(--ink);font-weight:600}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:12.5px;color:var(--ink-2)}
.legend span{display:inline-flex;align-items:center;gap:7px}
.sw{width:14px;height:14px;border-radius:3px;border:1px solid var(--ink-2);flex:none}
.sw.line{height:0;border:0;border-top:1.5px solid var(--ink-2);border-radius:0;width:22px}
.sw.dash{border-top-style:dashed}
.sw.dotc{width:11px;height:11px;border-radius:50%;border:0;background:var(--npu)}
.sw.ring{width:11px;height:11px;border-radius:50%;border:2px solid var(--npu);background:var(--panel)}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}
.rw{width:100%;border-collapse:collapse;font-size:13.5px}
.rw th{font:600 11px/1.3 var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3);text-align:left;padding:0 12px 8px 0}
.rw td{padding:9px 12px 9px 0;border-top:1px solid var(--rule);vertical-align:top}
.rw td:first-child{color:var(--ink-2)}
.rw code,.notes code,figcaption code,li code{font:12.5px var(--mono)}
.no::before{content:"✕  ";color:var(--ink-3)} .yes::before{content:"✓  ";color:var(--npu);font-weight:700}
.fits{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}
.fits div{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:16px 18px}
.fits h3{font:600 11.5px/1 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3);margin:0 0 10px}
.fits ul{margin:0;padding-left:18px;display:flex;flex-direction:column;gap:6px;font-size:14px}
details{font-size:13px;color:var(--ink-2)}
summary{cursor:pointer;width:max-content}
summary:focus-visible,a:focus-visible{outline:2px solid var(--s1);outline-offset:3px;border-radius:3px}
table{border-collapse:collapse;margin-top:8px;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:4px 16px 4px 0;white-space:nowrap}
thead th{color:var(--ink-3);font-weight:600;border-bottom:1px solid var(--rule)}
footer{font-size:13px;color:var(--ink-2);display:flex;flex-direction:column;gap:8px;border-top:1px solid var(--rule);padding-top:20px}
a{color:inherit;text-decoration-color:var(--ink-3);text-underline-offset:3px}
/* drawing */
.dt{font:12.5px var(--font);fill:var(--ink)} .dt-b{font:600 13px var(--font);fill:var(--ink)}
.dt-s{font:11.5px var(--font);fill:var(--ink-2)} .dt-m{font:11px var(--mono);fill:var(--ink-2)}
.dt-f{font:600 12.5px var(--mono);fill:var(--ink)} .dt-h{font:600 10px var(--mono);letter-spacing:.09em;fill:var(--ink-3)}
.dt-t{font:11px var(--font);fill:var(--ink-3);font-variant-numeric:tabular-nums} .dt-v{font:600 11.5px var(--font);fill:var(--ink);font-variant-numeric:tabular-nums}
.bx{fill:var(--panel-2);stroke:var(--ink-2);stroke-width:1} .bx-out{fill:none;stroke:var(--ink-3);stroke-width:1}
.bx-npu{fill:var(--npu-wash);stroke:var(--npu);stroke-width:1.6}
.bx-drv{fill:var(--drv-wash);stroke:var(--ink-2);stroke-width:1} .bx-cuda{fill:var(--cuda-wash);stroke:var(--ink-2);stroke-width:1}
.bx-trt{fill:var(--trt-wash);stroke:var(--ink-2);stroke-width:1} .rep{stroke-dasharray:6 4}
.lane{fill:var(--panel-2);stroke:none}
.ln{fill:none;stroke:var(--ink-2);stroke-width:1.3} .ln-npu{fill:none;stroke:var(--npu);stroke-width:1.9}
.mk{fill:var(--ink-2)} .mk-npu{fill:var(--npu)}
.bound{fill:none;stroke:var(--ink-3);stroke-width:1;stroke-dasharray:3 4} .wait{fill:none;stroke:var(--ink-3);stroke-width:1.2;stroke-dasharray:2 5}
.cross{fill:none;stroke:var(--ink-3);stroke-width:1.6;stroke-linecap:round}
.grid{fill:none;stroke:var(--rule);stroke-width:1} .base{fill:none;stroke:var(--ink-3);stroke-width:1}
.ref{fill:none;stroke:var(--ink-3);stroke-width:1}
.bar-npu{fill:var(--npu)} .bar-s1{fill:var(--s1)} .bar-s2{fill:var(--s2)} .bar-s3{fill:var(--s3)} .bar-claim{fill:var(--claim)}
.wash{opacity:.28}
.rng{fill:none;stroke:var(--npu);stroke-width:2;opacity:.4}
.dot{fill:var(--npu);stroke:var(--panel);stroke-width:2} .dot-open{fill:var(--panel);stroke:var(--npu);stroke-width:2}
g:hover>.bar-npu:not(.wash),g:hover>.bar-s1:not(.wash),g:hover>.bar-s2:not(.wash),g:hover>.bar-s3:not(.wash),g:hover>.bar-claim:not(.wash){filter:brightness(.88)}
@media (max-width:560px){ body{font-size:14.5px} .fact b{font-size:26px} figure{padding:14px 12px 12px} }
"""


def build() -> str:
    tops_rows, lat_rows, wake = load()
    kw = wake["keep_warm"]
    tops_table = table(["Layer type", "TOPS (FP16)", "Source"],
                       [(l, rng(lo, hi), s) for l, lo, hi, s in tops_rows])
    lat_table = table(["Model shape", "CPU, 8 threads (ms)", "NPU, GPU idle (ms)", "NPU, GPU saturated (ms)"],
                      [(f"{r['label']}, {r['sub']}", fmt(r["cpu"]), fmt(r["npu"]), fmt(r["busy"])) for r in lat_rows])
    wake_rows = [("called back to back", num(wake["back_to_back_ms"]), "")]
    wake_rows += [(f"after {float(g):g} s idle", num(d["median_ms"]), num(d["max_ms"])) for g, d in wake["idle_gap"].items()]
    wake_rows += [(lab, num(kw[k]["median_ms"]), num(kw[k]["max_ms"])) for lab, k in
                  (("3 s idle, no keep-warm", "none"), ("3 s idle, NPU tick every 100 ms", "npu-tick"),
                   ("3 s idle, CUDA tick every 100 ms", "gpu-tick"))]
    wake_table = table(["Condition", "Median (ms)", "Worst (ms)"], wake_rows)

    evidence = ('<div class="legend" aria-label="How to read the diagrams">'
                '<span><i class="sw" style="background:var(--npu-wash);border-color:var(--npu)"></i>the NPU and its path</span>'
                '<span><i class="sw line"></i>measured or read on this machine</span>'
                '<span><i class="sw line dash"></i>reported by others, not re-checked</span></div>')
    origin = ('<div class="legend" aria-label="Where each file comes from">'
              '<span><i class="sw" style="background:var(--drv-wash)"></i>ships with the NVIDIA driver</span>'
              '<span><i class="sw" style="background:var(--cuda-wash)"></i>CUDA redistributable (libcudla), separate download</span>'
              '<span><i class="sw" style="background:var(--trt-wash)"></i>TensorRT, separate download</span>'
              '<span><i class="sw" style="background:var(--panel-2)"></i>yours, or Windows</span></div>')
    lat_legend = ('<div class="legend"><span><i class="sw" style="background:var(--s1);border:0"></i>CPU, onnxruntime, 8 threads, FP32</span>'
                  '<span><i class="sw" style="background:var(--s2);border:0"></i>NPU, GPU idle</span>'
                  '<span><i class="sw" style="background:var(--s3);border:0"></i>NPU, another process saturating the GPU</span></div>')
    wake_legend = ('<div class="legend"><span><i class="sw dotc"></i>median</span><span><i class="sw ring"></i>worst of the calls</span></div>')

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Sparky NPU Architecture</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<header>
  <div class="eyebrow">ASUS ProArt P16 · NVIDIA RTX Spark N1X · Windows 11 on Arm · measured 18 Nov 2026</div>
  <h1>Sparky's NPU: hardware and software architecture</h1>
  <p class="lead">The NPU is an NVIDIA <b>Deep Learning Accelerator</b>, the fixed-function accelerator of the Jetson boards. <b>One software stack reaches it</b>: TensorRT's DLA backend compiles a model, and cuDLA runs it through a CUDA stream.</p>
  <div class="facts">
    <div class="fact"><b>4.3 <small>TOPS</small></b><span>measured, FP16, as the machine runs today</span></div>
    <div class="fact"><b>6.1 <small>TOPS</small></b><span>if the clock reached its 1562 MHz limit</span></div>
    <div class="fact"><b>1–2 <small>W</small></b><span>under load, 1.0 W idle</span></div>
    <div class="fact"><b>2–4×</b><span>faster than 8 CPU threads on mid-size encoders</span></div>
  </div>
  {evidence}
</header>

<section>
  <h2>Where it sits</h2>
  <p class="claim">The NPU is a block on the same chip as the CPU and GPU. It has no memory of its own, and Windows lists it as a second adapter of the GPU's driver.</p>
  <figure>{diagram_where()}
    <figcaption>Left: the chip. Right: what Windows enumerates. The NPU's two register windows, its interrupts and its ACPI name were read from Device Manager; the clock and power from <code>nvidia-smi</code>.</figcaption>
  </figure>
  <div class="notes">
    <p><b>Shared memory.</b> Weights, tensors and the compiled model live in the same LPDDR5X as everything else. The NPU reaches them by DMA through the IOMMU.</p>
    <p><b>Clock.</b> The limit is 1562 MHz. Under full load it reached 1105 MHz, on AC and on battery, with the Windows Standard power plan.</p>
    <p><b>One driver.</b> There is no separate NPU driver to install. <code>nvlddmkm.sys</code> serves the GPU as a display adapter and the NPU as a compute-only one.</p>
  </div>
</section>

<section>
  <h2>Inside the accelerator</h2>
  <p class="claim">A convolution pipeline with a few side units. The convolution core does almost all the work, at about 1,940 multiply-accumulates per clock cycle.</p>
  <figure>{diagram_inside()}
    <figcaption>Stage names were read from the DLA compiler shipped with TensorRT. The wiring follows NVIDIA's open NVDLA design and is inferred, not verified. Dashed boxes come from third-party reporting.</figcaption>
  </figure>
  <div class="notes">
    <p><b>Why wide layers are slow.</b> A 3×3 layer with 512 channels has 4.7 MB of weights, more than the reported 2.5 MiB buffer. Throughput falls from 4.3 to 1–2.7 TOPS.</p>
    <p><b>Why depthwise is slow.</b> A depthwise convolution keeps only about 30 of the roughly 2,000 multipliers busy per cycle.</p>
    <p><b>FP16 only.</b> The compiler knows INT8 and sparse weight formats, but on Windows on Arm only FP16 compute is offered. Sparse weights ran no faster.</p>
  </div>
</section>

<section>
  <h2>The software stack</h2>
  <p class="claim">Compiling and running are separate. TensorRT is only needed to compile. At run time one small DLL from the CUDA toolkit talks to two DLLs that ship with the driver.</p>
  <figure>{diagram_stack()}
    {origin}
    <figcaption>The orange path is one inference. Your code loads the compiled file with <code>cudlaModuleLoadFromMemory</code> and submits it with <code>cudlaSubmitTask</code>. Neither <code>cudla.dll</code> nor <code>nvdla_compiler.dll</code> ships with the driver or the TensorRT pip wheel, which is why TensorRT reports zero DLA cores out of the box.</figcaption>
  </figure>
  <div class="notes">
    <p><b>How it was found.</b> REA's export and import inspectors showed that <code>nvcudla.dll</code> has one <code>cudlaDrv*</code> function for each function of NVIDIA's public cuDLA API. Nothing was decompiled.</p>
    <p><b>Kernel entry.</b> <code>nvdla_runtime.dll</code> calls the Windows graphics kernel (<code>D3DKMT*</code>): open the adapter, create a context, submit, wait on a sync object.</p>
    <p><b>Dead ends.</b> cuDLA's standalone mode returns "unsupported". TensorRT could not build a GPU engine on this platform. DirectML and Windows ML are reported not to work.</p>
  </div>
</section>

<section>
  <h2>One inference, in hybrid mode</h2>
  <p class="claim">Every NPU task passes through the GPU's CUDA stream on the way in and on the way out. That is the only mode that works, and it explains the two timing surprises below.</p>
  <figure>{diagram_hybrid()}
    <figcaption>Tensors live in CUDA memory that cuDLA registers for the NPU. While the NPU runs at 99%, <code>nvidia-smi</code> shows the GPU at 1–2%.</figcaption>
  </figure>
  <div class="notes">
    <p><b>A busy GPU slows the NPU.</b> With another process saturating the GPU, the two hops through the stream wait their turn: NPU latency roughly doubles. The GPU job lost about 5%.</p>
    <p><b>It wakes slowly.</b> After half a second with nothing on the GPU, the next task can take 0.2–0.9 s. A tiny task every 100 ms prevents it.</p>
    <p><b>Floor.</b> A near-empty task takes 0.36 ms, so models that already run in under a millisecond on the CPU gain nothing.</p>
  </div>
</section>

<section>
  <h2>How fast it is</h2>
  <p class="claim">About 4.3 TOPS on ordinary convolutions, much less on very wide layers, transformer encoders and depthwise convolutions. The advertised figure is ten times higher and not reachable.</p>
  <figure>{chart_tops(tops_rows)}
    <figcaption>Each bar is a stack of identical layers with an exactly known operation count, run through cuDLA on AC power. A pale extension shows a range: between runs, or between model sizes for the encoders.</figcaption>
    {tops_table}
  </figure>
  <figure>{chart_gap()}
    <figcaption>NVIDIA rates its Jetson accelerators in sparse INT8 TOPS, and the press figure for this NPU is most likely the same kind of rating. INT8 compute is not exposed on Windows on Arm.</figcaption>
  </figure>
</section>

<section>
  <h2>Against the CPU</h2>
  <p class="claim">Mid-size and large networks run two to four times faster than on eight CPU threads. A small one keeps its lead only while the GPU is idle.</p>
  <figure>{lat_legend}{chart_latency(lat_rows)}
    <figcaption>Synthetic weights in real layer shapes. NPU output matched the FP32 CPU result to 0.05–0.2%. The 24-layer encoder does not compile as one file; it runs as two chained loadables.</figcaption>
    {lat_table}
  </figure>
</section>

<section>
  <h2>Waking up</h2>
  <p class="claim">Left alone for half a second or more, the next call can take hundreds of milliseconds. A tick during the gap keeps it near ten.</p>
  <figure>{wake_legend}{chart_wake(wake)}
    <figcaption>Twelve-layer encoder, 384 wide, six to eight calls per row. The tick is either a near-empty NPU task or a 4 KB CUDA memset. A GPU kept busy by another process has the same effect.</figcaption>
    {wake_table}
  </figure>
</section>

<section>
  <h2>What a model has to look like</h2>
  <p class="claim">Static shapes, batch 1, FP16, 4-D tensors, and every layer on the NPU. A transformer as PyTorch exports it is rejected; the same network rewritten with 1×1 convolutions compiles.</p>
  <figure>
    <div class="scroll"><table class="rw">
      <thead><tr><th>Part</th><th>As exported: rejected</th><th>Rewritten: compiles</th></tr></thead>
      <tbody>
        <tr><td>Tensor layout</td><td class="no"><code>[1, seq, d]</code>, 3-D</td><td class="yes"><code>[1, d, seq, 1]</code>, 4-D</td></tr>
        <tr><td>Linear layer</td><td class="no">MatMul with constant weights</td><td class="yes">1×1 convolution</td></tr>
        <tr><td>Attention scale</td><td class="no">multiply by a scalar</td><td class="yes">folded into the query weights</td></tr>
        <tr><td>Padding mask</td><td class="no"><code>[1, 1, 1, seq]</code>, broadcast</td><td class="yes"><code>[1, heads, seq, seq]</code> input, full size</td></tr>
        <tr><td>Layer norm</td><td class="no">on a 3-D tensor</td><td class="yes">on <code>[1, seq, 1, d]</code></td></tr>
        <tr><td>Pooling</td><td class="no">ReduceMean</td><td class="yes">GlobalAveragePool, or on the CPU</td></tr>
        <tr><td>Size</td><td class="no">one 660 MB loadable</td><td class="yes">two loadables of about 330 MB, chained</td></tr>
      </tbody>
    </table></div>
    <figcaption>A working reference is <code>encoder()</code> in <code>src/08_voice_models.py</code>.</figcaption>
  </figure>
  <figure>{diagram_split()}
    <figcaption>What stays on the CPU. The embedding table is a lookup the NPU has no layer for, and it costs microseconds in NumPy.</figcaption>
  </figure>
</section>

<section>
  <h2>What to use it for</h2>
  <div class="fits">
    <div><h3>Good fits</h3><ul>
      <li>Text or audio embedding encoders with a fixed length: 4–25 ms.</li>
      <li>Speaker, keyword or audio classifiers built from ordinary convolutions.</li>
      <li>ResNet-style image classifiers and detectors at a fixed resolution (not run here).</li>
      <li>Any of these while the GPU is busy with a language model, at about half speed.</li>
    </ul></div>
    <div><h3>Poor fits</h3><ul>
      <li>MobileNet-style networks: depthwise convolutions are about 60 times less efficient.</li>
      <li>LSTM or GRU models, autoregressive decoders, language models, dynamic shapes.</li>
      <li>Models that take under a millisecond on the CPU.</li>
      <li>Occasional calls without a keep-warm tick.</li>
    </ul></div>
  </div>
</section>

<footer>
  <p>Software under test: driver 621.24 (32.0.16.2124), CUDA UMD 13.5, cuDLA 13.5.25, TensorRT 11.5.0.63. DLA support in TensorRT on Windows on Arm is a beta.</p>
  <p>Full text, tables and sources: <a href="ARCHITECTURE.md">ARCHITECTURE.md</a>. Raw measurements: <code>results/</code>. This page is generated by <code>docs/build_architecture.py</code> from those files.</p>
  <p>An independent project. Not affiliated with or endorsed by NVIDIA.</p>
</footer>

</div>
</body>
</html>
"""


if __name__ == "__main__":
    out = Path(__file__).with_name("architecture.html")
    out.write_text(build(), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f} KiB)")
