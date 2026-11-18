# Sparky's NPU: hardware and software architecture

The NPU in the NVIDIA RTX Spark N1X is an NVIDIA Deep Learning Accelerator (DLA), the fixed-function
accelerator family NVIDIA ships in Jetson Xavier and Orin. On Windows on Arm it is reachable through
exactly one stack: **TensorRT's DLA backend to compile, cuDLA in hybrid mode to run**. It delivers
about 4.3 TOPS of FP16 compute on this machine today.

The same content as pictures: <https://roberthendriksen.github.io/rtx-spark-npu/>.

Everything here was measured on Sparky (ASUS ProArt P16 H7607BA) on 18 Nov 2026 unless a row says
otherwise. Each claim is tagged:

- **measured**: produced by a script in `src/`, raw output in `results/` or `evidence/`. The
  `evidence/` folder (export tables of NVIDIA's DLLs) is kept locally and is not in the repository.
- **read**: read from the OS, the driver package or an NVIDIA header on this machine.
- **reported**: from NVIDIA documentation or third-party reporting, not re-checked here.
- **inferred**: my reading of the evidence.

Software under test: Windows 11 Pro build 28000 (Arm64), NVIDIA driver 621.24 (32.0.16.2124),
CUDA UMD 13.5, cuDLA 13.5.25 (API 1.5.0), TensorRT 11.5.0.63. DLA support in this TensorRT is beta.

## 1. Hardware

### 1.1 Where it sits

```
 NVIDIA RTX Spark N1X system-on-chip
 +--------------------------------------------------------------------+
 |  20 Arm CPU cores      Blackwell GPU (6144 CUDA cores)     NPU     |
 |                        compute capability 12.1             (DLA)   |
 |        |                         |                           |     |
 |        +-----------+-------------+---------------------------+     |
 |                    |  unified LPDDR5X memory, behind an IOMMU      |
 +--------------------------------------------------------------------+
```

The NPU has no memory of its own beyond a small on-chip buffer. It reads weights and tensors from
the same unified memory as the CPU and GPU.

### 1.2 Identity

| Property | Value | Tag |
|---|---|---|
| Device name | `NVIDIA NPU` | read |
| Hardware ID | `ACPI\VEN_NVDA&DEV_200A&REV_0034` (compatible ID `NVDA2042`) | read |
| ACPI firmware node | `\_SB.DLA0` | read |
| Memory-mapped registers | `0x19000000`–`0x1903FFFF` (256 KiB) and `0x31380000`–`0x3138FFFF` (64 KiB) | read |
| Interrupts | 2 (IRQ 463 and 800) | read |
| DMA remapping policy | 2 (the driver supports IOMMU remapping) | read |
| Kernel driver | `nvlddmkm.sys`, the GPU driver; the NPU is a second adapter of it | read |
| Driver model | MCDM (compute-only), device class Display | read |
| DXCore attribute | `{d46140c4-add7-451b-9e56-06fe8c3b58ed}`, the NPU hardware-type attribute | read |
| nvidia-smi | device index 1, brand `NPU`, architecture `NPU3`, UUID prefix `DLA-` | read |
| Windows scheduler | one engine, `NPU_0`, on its own adapter LUID | read |
| cuDLA | 1 device, hardware version 2 (the header says Xavier is 1, Orin is 2) | measured |
| Unified addressing with CUDA | not supported | measured |

### 1.3 Clock, power, temperature

| Property | Value | Tag |
|---|---|---|
| DLA engine clock limit | 1562 MHz (`nvidia-smi -q`, "DLA Clocks") | read |
| Clock actually reached under full load | 1053–1105 MHz, on AC and on battery, Windows "Standard" plan, also after 40 s sustained | measured |
| Clock under light load | 880–940 MHz | measured |
| Power, idle | 1.0 W (`power.draw.instant`) | measured |
| Power, full load | 1.4–2.3 W | measured |
| Temperature under load | 45–52 °C | measured |

The clock never reached its limit in any run. What caps it at about 1.1 GHz was not determined; the
Windows power mode and the ASUS performance profile were left as they were.

### 1.4 Inside the accelerator

Verified here, by reading strings in the driver's `nvdla_runtime.dll` and TensorRT's
`nvdla_compiler.dll` (**read**):

- The compiler knows four targets: `t194` (Xavier), `t234` (Orin), `t254` and `t256`.
- It models the classic NVDLA stages: convolution core, SDP (activation and element-wise), PDP
  (pooling), CDP (normalisation), Rubik (reshape), BDMA (memory copy).
- It has a capability flag `supportsRVV`, a RISC-V vector unit, and limits named after the
  convolution buffer (`bufBankAllotted`, `requiresCbufSplitsForNonConv`).
- Its precision list includes INT8, INT16, FP16 and an internal FP19. Its weight formats include
  compressed weights and "structured sparse INT8 compressed" weights.

Derived from the benchmark (**inferred**): at best the hardware completes about 1,940
multiply-accumulates per clock cycle. That fits an array of 2,048 FP16 multipliers running at
about 95% occupancy.

From third-party reporting, not re-checked (**reported**): four convolution MAC units, a 32-bit
RISC-V control processor, firmware named `firmware-t35x`, 2.5 MiB of on-chip SRAM, and a DLA
compiler version of 3.16. The 512- and 1024-channel results in section 4.2 are consistent with an
SRAM of that size: one such layer's weights no longer fit.

## 2. Software stack

### 2.1 Layers

```
 your code (Python via ctypes, or C/C++)
        |                                   compile time only
        |                          +-------------------------------+
        |                          | nvinfer_11.dll   TensorRT      |
        |                          | nvonnxparser     ONNX parser   |
        |                          | nvdla_compiler   DLA compiler  |
        |                          +---------------+---------------+
        |                                          | "loadable" (FlatBuffers, magic NVDA)
        v                                          v
 cudla.dll            public cuDLA API (18 functions)            CUDA redist, libcudla
        |  forwards 1:1
        v
 nvcudla.dll          cudlaDrv* (18 functions), "NVIDIA CUDLA Driver"      driver store
        |                     |
        |                     +--> nvcuda.dll / nvcudaa64.dll   CUDA driver API (context, host memory)
        v
 nvdla_runtime.dll    NVDLA runtime: nvdla::IRuntime, NvDlaRuntimeSubmit   driver store
        |  D3DKMT calls through gdi32.dll (OpenAdapterFromLuid, CreateDevice,
        |  CreateContextVirtual, SubmitCommand, Escape, sync objects)
        v
 dxgkrnl.sys          Windows graphics kernel, schedules engine NPU_0
        v
 nvlddmkm.sys         NVIDIA kernel driver (MCDM adapter for the NPU)
        v
 NPU hardware         \_SB.DLA0
```

### 2.2 Components

| File | Comes from | Role | Evidence |
|---|---|---|---|
| `nvlddmkm.sys` 32.0.16.2124 | driver | Kernel driver for both GPU and NPU | PnP properties |
| `nvdla_runtime.dll` (7.0 MB) | driver store | Loads loadables, binds memory, submits tasks. Exports a C++ `nvdla::IRuntime` and a C `NvDla*` API | `evidence/nvdla_runtime.exports.json` |
| `nvcudla.dll` (1.4 MB) | driver store | cuDLA's driver half. One `cudlaDrv*` export for each public function. Imports 12 functions from `nvcuda.dll` | `evidence/nvcudla.*.json` |
| `nvcuda.dll` | System32 | CUDA driver API. cuDLA needs a CUDA context | loaded-module list |
| `cudla.dll` (0.1 MB) | NVIDIA CUDA redist, component `libcudla`, windows-arm64 | Public cuDLA API and header `cudla.h`. Not shipped with the driver | `evidence/cudla.*.json` |
| `nvdla_compiler.dll` (26 MB) | TensorRT Windows-Arm64 zip | Compiles a network into a loadable. 666 exports, 474 of them a C `NvDla*` API | `evidence/nvdla_compiler.exports.json` |
| `nvinfer_11.dll` (337 MB) | TensorRT wheel or zip | TensorRT. Loads `cudla.dll` and `nvdla_compiler.dll` by name at run time | string scan |

The module list after a task has run on the NPU (`src/tools_loaded_modules.py`, **measured**):
`cudla.dll`, `nvcuda.dll`, `nvcudaa64.dll`, `nvcudla.dll`, `nvdla_runtime.dll`, `nvcuextenda64.dll`,
`nvdxgdmala64.dll`, `nvobjectloadera64.dll`, `nvapia64.dll`. TensorRT is not among them: it is
needed to compile, not to run.

How these were found: REA (`rea inspect-native-exports`, `rea inspect-pe-imports`) on copies of the
driver DLLs, plus a plain string scan. Nothing was decompiled.

### 2.3 Hybrid mode

cuDLA has two modes. Standalone mode returns `cudlaErrorUnsupportedOperation` on this platform
(**measured**). The only mode that works is hybrid ("CUDA + DLA"):

1. The process creates a CUDA context on the GPU.
2. Input and output tensors are allocated as CUDA device memory and registered with cuDLA
   (`cudlaMemRegister`). There is no shared virtual addressing, so cuDLA maps them for the NPU.
3. A task is submitted on a CUDA stream (`cudlaSubmitTask`). The stream orders it against GPU work.
4. `cuStreamSynchronize` waits for the NPU to finish.

Consequences, all **measured** (section 4):

- The GPU is lightly involved in every NPU inference. `nvidia-smi` shows it at 1–2% while the NPU
  is at 99%.
- A saturated GPU slows the NPU by about a factor of two.
- When both are idle for half a second or more, the next NPU task can take 0.2–0.9 s to start.

### 2.4 Compile-time flow

```
ONNX (static shapes, batch 1, FP16 tensors)
  -> TensorRT ONNX parser
  -> TensorRT builder, default device = DLA, GPU fallback off
  -> nvdla_compiler.dll
  -> either a TensorRT engine that wraps the loadable   (EngineCapability.STANDARD)
     or the raw loadable itself                        (EngineCapability.DLA_STANDALONE)
```

`src/npu_dla.py` wraps this as `compile_loadable(onnx_bytes)`.

### 2.5 Two ways to run

| Route | Needs at run time | Notes |
|---|---|---|
| TensorRT execution context | TensorRT, `cudla.dll`, `nvdla_compiler.dll` | FP32 or FP16 I/O, plain layout. `src/05_run_npu.py` |
| Raw cuDLA | `cudla.dll` only | FP16 I/O in `DLA_LINEAR` layout (rows padded to 64 bytes). About 10% faster. `src/06_cudla_direct.py`, `src/npu_dla.py` |

### 2.6 What does not reach the NPU

- **DirectML and Windows ML**: the driver registers the adapter with DXCore as NPU hardware, but
  creating a Direct3D 12 device on it reportedly fails, and Microsoft lists no NVIDIA NPU execution
  provider (**reported**, not re-tested).
- **TensorRT on the GPU**: on Windows on Arm, TensorRT 11.5 could not build a GPU engine here
  (`myelin_resource64_sm110_2.dll` is missing). On this platform it is a DLA compiler only
  (**measured**).
- **INT8 compute**: NVIDIA documents FP16 compute only on Windows on Arm; quantize/dequantize
  layers are a storage format (**reported**).

## 3. What a model must look like

Rules that were hit in practice (**measured**):

| Rule | Detail |
|---|---|
| Static shapes, batch 1 | No dynamic dimensions |
| FP16 tensors | The network is strongly typed. Weights and activations must be FP16 in the ONNX file. FP32 I/O is possible through TensorRT with casts at the edges |
| Every layer on the NPU | No GPU fallback. One unsupported layer fails the build |
| 4-D tensors | LayerNorm, MatMul and reshapes on 3-D tensors are rejected. Scalars and 1-D constants that need broadcasting are rejected |
| No `Identity`, no `ReduceMean` | Use the previous layer's output directly; use `GlobalAveragePool` |
| Loadable size | A 390 MB loadable compiled; a 660 MB one did not. Split large models into chained loadables |

Layers that compiled and ran: Conv (1×1, 3×3, 5×5, 1×k), ReLU, Sigmoid, GELU, Add, Mul,
MatMul between two activations, Softmax, LayerNormalization, Reshape, Transpose,
GlobalAveragePool, Cast.

A transformer encoder as PyTorch exports it (3-D tensors, MatMul with constant weights) does not
compile. The same encoder rewritten in 4-D form does: tensors as `[1, d, seq, 1]`, linear layers as
1×1 convolutions, the attention scale folded into the query weights, the padding mask as a
full-size `[1, heads, seq, seq]` input. `encoder()` in `src/08_voice_models.py` is a working
reference.

## 4. Performance

### 4.1 TOPS

Method (`src/07_measure_tops.py`): build convolution stacks with an exactly known number of
multiply-accumulates (MACs), run them through raw cuDLA, and compute
`TOPS = 2 × MACs per inference × inferences per second / 10^12`. All compute is FP16.

| Figure | Value | Tag |
|---|---|---|
| Peak, as this machine runs today | **4.3 TOPS** (FP16, dense 3×3 convolution, 256 channels, clock 1105 MHz) | measured |
| Hardware rate | about 1,940 MACs per clock cycle | measured |
| At the 1562 MHz clock limit | about 6.1 TOPS | inferred (rate × clock limit); a third party measured 6.07 at 1561 MHz |
| Advertised | 50–55 TOPS in press coverage; NVIDIA's specification page gives no NPU figure | reported |

Why the advertised figure is not reachable: NVIDIA rates its DLAs in *sparse INT8* TOPS (Jetson AGX
Orin: 105 for two DLAs). On this platform INT8 compute is not exposed, and sparse weights gave no
speed-up: 50% and 90% random zeros, a 2-of-4 channel pattern and half-pruned kernels all ran in the
same time as dense weights. They only lowered power, from 1.7 W to 1.1 W at 90% zeros (**measured**).

### 4.2 By layer type

AC power, Windows "Standard" plan, DLA clock 1053–1105 MHz. Raw table: `results/tops_ac.md`.

| Workload | TOPS (FP16) | MACs per cycle |
|---|---|---|
| 3×3 conv, 64–256 channels | 3.7–4.3 | 1,730–1,940 |
| 5×5 conv, 128 channels | 4.0 | 1,910 |
| 1×1 conv, 256–1024 channels | 3.4–3.9 | 1,560–1,790 |
| 3×3 conv, 192 channels | 3.2 | 1,460 |
| 3×3 conv, 512 channels | 0.9–2.7, varies run to run | 450–1,200 |
| 3×3 conv, 1024 channels | 0.7–1.4, varies run to run | 360–650 |
| Depthwise 3×3, 256 channels | 0.07 | 32 |
| Transformer encoder, d=768 or 1024 | about 1.5–1.8 | |
| Near-empty task | 0.36 ms per task | |

Reading it:

- Ordinary convolutions up to 256 channels use the hardware almost fully.
- Wide 3×3 layers lose most of it, and the NPU's own utilisation drops to 13–60%. Their weights
  (4.7 MB per layer at 512 channels) exceed the reported on-chip buffer.
- Depthwise convolutions are about 60 times less efficient than regular ones. MobileNet-style
  networks are a poor fit.
- A task costs at least 0.36 ms, so very small models gain nothing.

### 4.3 Model-shaped tests

Synthetic weights, real layer shapes. CPU is onnxruntime 1.32, FP32, 8 threads. NPU is raw cuDLA,
FP16. Source: `src/08_voice_models.py`, `results/voice_models.json`.

| Model shape | CPU | NPU | Speed-up | Error vs FP32 |
|---|---|---|---|---|
| Encoder, 12 layers, d=384, 64 tokens | 8.2 ms | 3.9 ms | 2.1× | 0.12% |
| Same, 128 tokens | 14.0 ms | 5.0 ms | 2.8× | 0.14% |
| Same, with a padding-mask input | 8.1 ms | 4.0 ms | 2.0× | 0.20% |
| Encoder, 12 layers, d=768, 64 tokens | 22.3 ms | 7.0 ms | 3.2× | 0.13% |
| Encoder, 24 layers, d=1024, 64 tokens, one loadable | 73.6 ms | does not compile | | |
| Same, as two chained loadables | 94.4 ms | 24.6 ms | 3.8× | 0.21% |
| Encoder, 4 layers, d=384, 400 frames | 14.6 ms | 5.8 ms | 2.5× | 0.07% |
| Conv1d speaker trunk, 512 channels, 300 frames | 5.2 ms | 1.3 ms | 3.9× | 0.05% |
| Same, 1024 channels | 17.6 ms | 9.9 ms | 1.8× | 0.05% |

### 4.4 Sharing with the GPU

A second process saturated the GPU with FP16 matrix multiplies (`src/09_gpu_contention.py`).

| Encoder | GPU idle | GPU saturated | GPU saturated, p95 |
|---|---|---|---|
| d=384 | 3.8 ms | 8.4 ms | 11.4 ms |
| d=768 | 7.3 ms | 12.7 ms | 17.1 ms |
| d=1024, two loadables | 22.1 ms | 40.8 ms | 45.3 ms |

The GPU process lost about 5% of its throughput while the NPU ran back to back.

### 4.5 Waking up

Called back to back, the d=384 encoder takes 3.3–3.8 ms. After an idle gap with nothing else
running on the GPU (`src/10_wakeup.py`):

| Idle gap | Median | Worst |
|---|---|---|
| 0.1–0.25 s | 5 ms | 7 ms |
| 0.5 s | 6 ms | 796 ms |
| 1–5 s | 9–200 ms | 550–830 ms |

Keeping the path warm removes it. With either a near-empty NPU task or a 4 KB CUDA memset every
100 ms during the gap, the call took 6–8 ms with a worst case of 11 ms. A busy GPU in another
process has the same effect. The remaining gap between 3.5 ms and 6–8 ms is the clock ramping back
up (**inferred**).

## 5. Not determined

- What limits the DLA clock to about 1.1 GHz, and the throughput at the full 1562 MHz.
- Whether INT8 compute or sparse acceleration can be reached through the compiler's own C API.
  A third party reports doing so and seeing no gain over FP16.
- GPU memory held by the CUDA context that hybrid mode requires.
- The exact loadable size limit between 390 MB and 660 MB.
- Power at the wall. Only `nvidia-smi` readings were taken.

## 6. Reproduce

```powershell
.venv\Scripts\python.exe src\01_find_npu.py
.venv\Scripts\python.exe src\07_measure_tops.py --tag ac
.venv\Scripts\python.exe src\08_voice_models.py
.venv\Scripts\python.exe src\09_gpu_contention.py
.venv\Scripts\python.exe src\10_wakeup.py
```

## 7. Sources

- NVIDIA, DLA supported layers for Windows on ARM: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-windows-on-arm.html
- NVIDIA, working with DLA: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dla.html
- NVIDIA, CUDA redistributable manifest 13.5.0: https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.5.0.json
- NVIDIA, Jetson Orin DLA ratings: https://developer.nvidia.com/blog/maximizing-deep-learning-performance-on-nvidia-jetson-orin-with-dla/
- NVIDIA, RTX Spark specifications: https://www.nvidia.com/en-us/products/rtx-spark/specifications/
- XDA Developers, independent investigation of the same NPU: https://www.xda-developers.com/nvidia-rtx-spark-npu-barely-mentions-spent-week-figuring-out/
- REA, used for the static inspection: https://github.com/morluto/rea
