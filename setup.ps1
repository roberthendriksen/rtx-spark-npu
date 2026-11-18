# Sets up everything the scripts in src/ need, inside this folder. Nothing is installed system-wide.
#
#   .\setup.ps1                                   # Python environment + cuDLA, then explains the TensorRT step
#   .\setup.ps1 -TensorRtZip C:\path\to\TensorRT-Enterprise-11.5.0.63-Windows-arm64-cuda-13.5-Release-external.zip
#
# Needs: Windows 11 on Arm64 with an NVIDIA RTX Spark, NVIDIA driver 621.24 or later, and uv (https://docs.astral.sh/uv/).
param([string]$TensorRtZip = "")
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

# 1. Python environment (kept if it already exists)
if (-not (Test-Path $python)) {
    uv venv --python 3.13 (Join-Path $root ".venv")
}
uv pip install --python $python -r (Join-Path $root "requirements.txt") --index-strategy unsafe-best-match

# 2. cuDLA: cudla.dll, cudla.h and cudla.lib from NVIDIA's CUDA redistributables.
#    By downloading it you accept NVIDIA's CUDA Toolkit licence. The hash is the one in NVIDIA's manifest
#    https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.5.0.json
$cudla = "libcudla-windows-arm64-13.5.25-archive"
$cudlaSha = "92e36696406562635e24a079822cf3b0c559932622e743a5bb4ecf073fae6a24"
$cudlaDir = Join-Path $root "vendor\libcudla"
if (-not (Test-Path (Join-Path $cudlaDir "$cudla\bin\arm64\cudla.dll"))) {
    New-Item -ItemType Directory -Force $cudlaDir | Out-Null
    $zip = Join-Path $cudlaDir "$cudla.zip"
    Invoke-WebRequest "https://developer.download.nvidia.com/compute/cuda/redist/libcudla/windows-arm64/$cudla.zip" -OutFile $zip
    if ((Get-FileHash $zip -Algorithm SHA256).Hash.ToLower() -ne $cudlaSha) { throw "cuDLA archive: SHA-256 mismatch" }
    Expand-Archive $zip -DestinationPath $cudlaDir -Force
}

# 3. The DLA compiler, nvdla_compiler.dll, is only in NVIDIA's TensorRT zip for Windows on Arm64.
#    That download sits behind NVIDIA's TensorRT licence agreement, so this script does not fetch it:
#    accept the licence at https://developer.nvidia.com/tensorrt/download/11x, download the
#    "Windows on Arm64, CUDA 13.5" zip, and pass its path with -TensorRtZip.
$trtBin = Join-Path $root "vendor\tensorrt\TensorRT-11.5.0.63\bin"
if ($TensorRtZip) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [IO.Compression.ZipFile]::OpenRead((Resolve-Path $TensorRtZip))
    try {
        $entry = $archive.Entries | Where-Object { $_.FullName -like "*/bin/nvdla_compiler.dll" } | Select-Object -First 1
        if (-not $entry) { throw "nvdla_compiler.dll not found in $TensorRtZip" }
        New-Item -ItemType Directory -Force $trtBin | Out-Null
        [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, (Join-Path $trtBin "nvdla_compiler.dll"), $true)
    } finally { $archive.Dispose() }
}
if (-not (Test-Path (Join-Path $trtBin "nvdla_compiler.dll"))) {
    Write-Host ""
    Write-Host "Not done yet: the DLA compiler is missing, so models cannot be compiled." -ForegroundColor Yellow
    Write-Host "  1. Accept the licence and download the Windows on Arm64 (CUDA 13.5) zip from"
    Write-Host "     https://developer.nvidia.com/tensorrt/download/11x"
    Write-Host "  2. Run:  .\setup.ps1 -TensorRtZip <path to that zip>"
}

# 4. Check that the NPU answers (needs only the driver and cudla.dll)
Write-Host ""
& $python (Join-Path $root "src\01_find_npu.py")
