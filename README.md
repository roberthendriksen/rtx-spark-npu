# The NPU on Sparky

Sparky is an ASUS ProArt P16 (H7607BA) with an NVIDIA RTX Spark N1X, running Windows 11 on Arm64.
Its NPU is an NVIDIA **DLA** (Deep Learning Accelerator, the Jetson accelerator family). This
folder finds it, runs models on it from Python, and measures what it can do.

- [docs/architecture.html](docs/architecture.html): the architecture in pictures: block diagrams, the
  software stack, and charts of every measurement. Download it and open it in a browser.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): the same in text, with every claim tagged measured,
  read, reported or inferred.
- `results/`: raw benchmark output.

Verified on 18 Nov 2026 with driver 621.24, CUDA UMD 13.5, cuDLA 13.5.25, TensorRT 11.5.0.63.
DLA support in TensorRT on Windows on Arm is a beta, so treat this as a snapshot of that date.

This is an independent project. It is not affiliated with or endorsed by NVIDIA.

## In one screen

| Question | Answer |
|---|---|
| What is it? | `NVIDIA NPU`, `ACPI\NVDA200A`, firmware node `\_SB.DLA0`; a second adapter of the GPU driver; nvidia-smi device 1 |
| How fast? | 4.3 TOPS (FP16) measured; about 6.1 TOPS if the clock reached its 1562 MHz limit (it stayed at 1105 MHz) |
| What about the advertised 50–55 TOPS? | Not reachable. It is most likely a sparse-INT8 rating; INT8 compute is not exposed and sparse weights gave no speed-up |
| Power | 1.0 W idle, 1.4–2.3 W under load (nvidia-smi) |
| What runs on it? | Static-shape, batch-1, FP16 networks of convolutions, activations, MatMul, Softmax, LayerNorm, in 4-D tensors |
| Which stack? | TensorRT's DLA backend to compile, cuDLA (hybrid mode, via a CUDA stream) to run. Not DirectML, not Windows ML |
| What is it good at? | Mid-size CNNs and transformer encoders: 2–4× faster than 8 CPU threads |
| What is it bad at? | Depthwise convolutions, very wide layers, tiny models, anything recurrent, dynamic or autoregressive |

## Good and poor fits

Good:
- Text or audio **embedding encoders** with a fixed length (12–24 layers, 384–1024 wide): 4–25 ms.
- **Speaker, keyword or audio classifiers** built from ordinary 1-D or 2-D convolutions.
- **Image classifiers and detectors with ResNet-style blocks** at a fixed resolution (not run here;
  a third party reports ResNet-50 at 11.8 ms after folding BatchNorm into the convolutions).
- Any of these **while the GPU is busy** with an LLM: the NPU keeps working, about 2× slower.

Poor:
- MobileNet-style networks: depthwise convolutions run about 60× less efficiently.
- LSTM/GRU models, autoregressive decoders, LLMs, anything with dynamic shapes.
- Models that take under a millisecond on the CPU: one NPU task costs at least 0.36 ms.
- Once-in-a-while calls without a keep-warm tick: the first call after idling can take 0.2–0.9 s.

## Set up

Needs Windows 11 on Arm64 on an RTX Spark machine, NVIDIA driver 621.24 or later, and
[uv](https://docs.astral.sh/uv/). Everything lands inside this folder.

```powershell
.\setup.ps1
```

That creates the Python environment and downloads cuDLA from NVIDIA's CUDA redistributables. The
DLA compiler is only in NVIDIA's TensorRT zip for Windows on Arm64, which sits behind a licence
agreement. Accept it at <https://developer.nvidia.com/tensorrt/download/11x>, download the
"Windows on Arm64, CUDA 13.5" zip, and run:

```powershell
.\setup.ps1 -TensorRtZip C:\path\to\TensorRT-Enterprise-11.5.0.63-Windows-arm64-cuda-13.5-Release-external.zip
```

Without the compiler you can still enumerate the NPU (`src\01_find_npu.py`) but not compile models.

## Use it

```python
from npu_dla import compile_loadable, DlaDevice    # src/npu_dla.py

loadable = compile_loadable(onnx_bytes)            # offline; needs TensorRT + nvdla_compiler.dll
with DlaDevice() as dev:                           # run time; needs cudla.dll only
    mod = dev.load(loadable)
    mod.write(0, x)                                # NCHW array, converted to FP16
    mod.run()
    y = mod.read(0)
```

## Scripts

```powershell
.venv\Scripts\python.exe src\01_find_npu.py            # enumerate the NPU through cuDLA
.venv\Scripts\python.exe src\02_tensorrt_dla_check.py  # TensorRT should report 1 DLA core
.venv\Scripts\python.exe src\03_make_model.py          # write a small test model
.venv\Scripts\python.exe src\04_build_engine.py        # compile it as a TensorRT DLA engine
.venv\Scripts\python.exe src\05_run_npu.py             # run it, check accuracy and utilisation
.venv\Scripts\python.exe src\06_cudla_direct.py        # the same network through raw cuDLA
.venv\Scripts\python.exe src\07_measure_tops.py        # TOPS by layer type
.venv\Scripts\python.exe src\08_voice_models.py        # encoders and a speaker trunk, NPU vs CPU
.venv\Scripts\python.exe src\09_gpu_contention.py      # NPU latency while the GPU is saturated
.venv\Scripts\python.exe src\10_wakeup.py              # wake-up cost after idling, keep-warm
.venv\Scripts\python.exe docs\build_architecture.py    # redraw docs\architecture.html from results\
```

## What ends up in this folder, and what is not in the repository

| Path | What | Source |
|---|---|---|
| `vendor/libcudla` | `cudla.dll`, `cudla.h`, `cudla.lib` 13.5.25 | NVIDIA CUDA redist; `setup.ps1` checks the SHA-256 against NVIDIA's manifest |
| `vendor/tensorrt` | `nvdla_compiler.dll` from the TensorRT 11.5.0.63 Windows-Arm64 zip | developer.nvidia.com, NVIDIA TensorRT License Agreement |
| `.venv` | Python 3.13 Arm64: TensorRT 11.5.0.63 wheels, numpy, onnx, onnxruntime | pypi.nvidia.com, PyPI |
| `.venv-gpuload` | optional: NVIDIA's PyTorch 2.15.0+cu134, only as GPU load for `09_gpu_contention.py` | pypi.nvidia.com/nvtorch_oot |
| `models/` | generated ONNX files, engines and loadables | the scripts in `src/` |
| `tools/rea`, `targets/`, `evidence/` | optional: REA, copies of driver DLLs, and their export tables, for re-inspecting the driver | npm; your own driver store |

Nothing is installed system-wide and the driver store is not modified. All of these paths are in
`.gitignore`: they hold NVIDIA's and third parties' binaries, or files derived from them, and must
not be redistributed. The code in this repository is under the MIT licence; NVIDIA's software stays
under NVIDIA's licences.

## How it was found

[REA](https://github.com/morluto/rea) ("Reverse Engineer Anything") was pointed at the NVIDIA
driver package. Its export and import inspectors showed that the driver ships `nvcudla.dll` and
`nvdla_runtime.dll`, that `nvcudla.dll` implements NVIDIA's public cuDLA API one to one, and that
TensorRT loads `cudla.dll` and `nvdla_compiler.dll` by name. Those two files are not in the driver
or the pip wheel, which is why TensorRT reports zero DLA cores out of the box. REA's decompiler
back ends do not run on Windows Arm64, so nothing was decompiled.
