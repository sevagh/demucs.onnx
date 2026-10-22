#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Configure + build the demucs.onnx native CLI on Windows (MSVC).

.DESCRIPTION
    Upstream only ships ./scripts/build-ort-linux.sh, which builds a minimal
    ONNX Runtime with the ORT builder on Linux. On Windows this script instead
    stages the official prebuilt Microsoft.ML.OnnxRuntime NuGet package into
    ./deps/ort (headers + import lib + DLL) and builds src_cli against it.

.PARAMETER Config
    CMake build type (default Release).

.PARAMETER Jobs
    Parallel compile jobs (default: number of logical CPUs).

.PARAMETER OrtVersion
    ONNX Runtime NuGet version to stage (default 1.21.0 = version vendored in
    ./vendor/onnxruntime).

.PARAMETER StageOrtOnly
    Only stage deps/ort and exit.

.PARAMETER Clean
    Wipe the build directory before configuring.
#>
[CmdletBinding()]
param(
    [string]$Config = 'Release',
    [int]$Jobs = 0,
    [string]$OrtVersion = '1.21.0',
    [switch]$StageOrtOnly,
    [switch]$Clean,
    # Which ONNX Runtime build to stage+link: auto|cpu|dml|cuda.
    # auto prefers cuda, then dml, then cpu (whatever is already staged).
    [ValidateSet('auto', 'cpu', 'dml', 'cuda')]
    [string]$Variant = 'auto'
)

$ErrorActionPreference = 'Stop'

$RepoRoot  = Split-Path -Parent $PSScriptRoot
$DepsDir   = Join-Path $RepoRoot 'deps'
$OrtRoot   = Join-Path $DepsDir 'ort'
$OrtDmlRoot = Join-Path $DepsDir 'ort-dml'
$OrtCudaRoot = Join-Path $DepsDir 'ort-cuda'
$BuildDir  = Join-Path $RepoRoot 'build/build-cli-win'

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }

function Get-NuGetPackage {
    param([string]$Id, [string]$Version, [string]$NupkgPath)
    if (-not (Test-Path $NupkgPath)) {
        Write-Step "Downloading $Id $Version from NuGet"
        $url = "https://api.nuget.org/v3-flatcontainer/$($Id.ToLower())/$Version/$($Id.ToLower()).$Version.nupkg"
        Invoke-WebRequest -Uri $url -OutFile $NupkgPath -TimeoutSec 1800
    }
    return $NupkgPath
}

# ---------------------------------------------------------------- ONNX Runtime
function Stage-OnnxRuntime {
    $lib = Join-Path $OrtRoot 'lib/onnxruntime.lib'
    $dll = Join-Path $OrtRoot 'bin/onnxruntime.dll'
    if ((Test-Path $lib) -and (Test-Path $dll)) {
        Write-Step "ONNX Runtime already staged at deps/ort"
        return
    }

    Write-Step "Staging ONNX Runtime $OrtVersion (CPU) from NuGet"
    New-Item -ItemType Directory -Force -Path $DepsDir | Out-Null

    $nupkg = Get-NuGetPackage -Id 'Microsoft.ML.OnnxRuntime' -Version $OrtVersion `
        -NupkgPath (Join-Path $DepsDir "microsoft.ml.onnxruntime.$OrtVersion.nupkg")

    $extract = Join-Path $DepsDir "ort-$OrtVersion"
    if (Test-Path $extract) { Remove-Item $extract -Recurse -Force }
    $zip = Join-Path $DepsDir 'ort.zip'
    Copy-Item $nupkg $zip -Force
    Expand-Archive $zip -DestinationPath $extract -Force

    $inc = Join-Path $OrtRoot 'include/onnxruntime/core/session'
    New-Item -ItemType Directory -Force -Path $inc, (Join-Path $OrtRoot 'lib'), (Join-Path $OrtRoot 'bin') | Out-Null
    # the NuGet headers are flat, the sources include onnxruntime/core/session/...
    Copy-Item (Join-Path $extract 'build/native/include/*.h') $inc -Force
    Copy-Item (Join-Path $extract 'runtimes/win-x64/native/onnxruntime.lib') (Join-Path $OrtRoot 'lib') -Force
    Copy-Item (Join-Path $extract 'runtimes/win-x64/native/onnxruntime.dll') (Join-Path $OrtRoot 'bin') -Force
    Copy-Item (Join-Path $extract 'runtimes/win-x64/native/onnxruntime_providers_shared.dll') (Join-Path $OrtRoot 'bin') -Force -ErrorAction SilentlyContinue

    Remove-Item $zip -Force -ErrorAction SilentlyContinue
    Write-Step "Staged deps/ort (ORT $OrtVersion, win-x64)"
}

# DirectML build of the same runtime: same C API, plus the DirectML EP so the
# network can run on the GPU. Selected by CMake when deps/ort-dml is present.
function Stage-DirectML {
    $lib = Join-Path $OrtDmlRoot 'lib/onnxruntime.lib'
    $dll = Join-Path $OrtDmlRoot 'bin/onnxruntime.dll'
    if ((Test-Path $lib) -and (Test-Path $dll)) {
        Write-Step "DirectML runtime already staged at deps/ort-dml"
        return
    }

    Write-Step "Staging ONNX Runtime DirectML $OrtVersion (GPU) from NuGet"
    $nupkg = Get-NuGetPackage -Id 'Microsoft.ML.OnnxRuntime.DirectML' -Version $OrtVersion `
        -NupkgPath (Join-Path $DepsDir "microsoft.ml.onnxruntime.directml.$OrtVersion.nupkg")

    $extract = Join-Path $DepsDir "ort-dml-$OrtVersion"
    if (Test-Path $extract) { Remove-Item $extract -Recurse -Force }
    $zip = Join-Path $DepsDir 'ort-dml.zip'
    Copy-Item $nupkg $zip -Force
    Expand-Archive $zip -DestinationPath $extract -Force

    $sess = Join-Path $OrtDmlRoot 'include/onnxruntime/core/session'
    $prov = Join-Path $OrtDmlRoot 'include/onnxruntime/core/providers/dml'
    New-Item -ItemType Directory -Force -Path $sess, $prov, (Join-Path $OrtDmlRoot 'lib'), (Join-Path $OrtDmlRoot 'bin') | Out-Null
    Copy-Item (Join-Path $extract 'build/native/include/*.h') $sess -Force
    Copy-Item (Join-Path $extract 'build/native/include/dml_provider_factory.h') $prov -Force
    Copy-Item (Join-Path $extract 'runtimes/win-x64/native/onnxruntime.lib') (Join-Path $OrtDmlRoot 'lib') -Force
    Copy-Item (Join-Path $extract 'runtimes/win-x64/native/onnxruntime.dll') (Join-Path $OrtDmlRoot 'bin') -Force

    Remove-Item $zip -Force -ErrorAction SilentlyContinue
    Write-Step "Staged deps/ort-dml (ORT $OrtVersion DirectML, win-x64)"
}

# CUDA build: same C API plus the CUDA EP (onnxruntime_providers_cuda.dll).
# The Windows NuGet payload ships runtime binaries only, so the headers are
# reused from the CPU package (identical version) and cuda_provider_factory is
# not needed at all - demucs.cpp uses OrtCUDAProviderOptions from the C API.
function Stage-Cuda {
    $lib = Join-Path $OrtCudaRoot 'lib/onnxruntime.lib'
    $dll = Join-Path $OrtCudaRoot 'bin/onnxruntime_providers_cuda.dll'
    if ((Test-Path $lib) -and (Test-Path $dll)) {
        Write-Step "CUDA runtime already staged at deps/ort-cuda"
        return
    }

    Write-Step "Staging ONNX Runtime GPU (CUDA) $OrtVersion from NuGet"
    # the meta package only has nuspec deps; the payload lives in .Gpu.Windows
    $nupkg = Get-NuGetPackage -Id 'Microsoft.ML.OnnxRuntime.Gpu.Windows' -Version $OrtVersion `
        -NupkgPath (Join-Path $DepsDir "microsoft.ml.onnxruntime.gpu.windows.$OrtVersion.nupkg")

    $extract = Join-Path $DepsDir "ort-gpu-win-$OrtVersion"
    if (Test-Path $extract) { Remove-Item $extract -Recurse -Force }
    $zip = Join-Path $DepsDir 'ort-gpu-win.zip'
    Copy-Item $nupkg $zip -Force
    Expand-Archive $zip -DestinationPath $extract -Force

    Stage-OnnxRuntime   # headers (.h) are shared with the CPU package
    $sess = Join-Path $OrtCudaRoot 'include/onnxruntime/core/session'
    New-Item -ItemType Directory -Force -Path $sess, (Join-Path $OrtCudaRoot 'lib'), (Join-Path $OrtCudaRoot 'bin') | Out-Null
    Copy-Item (Join-Path $OrtRoot 'include/onnxruntime/core/session/*.h') $sess -Force
    Copy-Item (Join-Path $extract 'runtimes/win-x64/native/onnxruntime.lib') (Join-Path $OrtCudaRoot 'lib') -Force
    foreach ($f in 'onnxruntime.dll', 'onnxruntime_providers_cuda.dll', 'onnxruntime_providers_shared.dll') {
        Copy-Item (Join-Path $extract "runtimes/win-x64/native/$f") (Join-Path $OrtCudaRoot 'bin') -Force
    }

    Remove-Item $zip -Force -ErrorAction SilentlyContinue
    Write-Step "Staged deps/ort-cuda (ORT $OrtVersion CUDA, win-x64)"
    Write-Host "    NOTE: CUDA 12.x + cuDNN 9 DLLs must be on PATH at runtime (see WINDOWS.md)." -ForegroundColor Yellow
}


function Import-VcVars {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    $vcvars = $null
    if (Test-Path $vswhere) {
        $installs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        foreach ($i in $installs) {
            $candidate = Join-Path $i 'VC/Auxiliary/Build/vcvars64.bat'
            if (Test-Path $candidate) { $vcvars = $candidate; break }
        }
    }
    if (-not $vcvars) {
        $fallback = Get-ChildItem 'C:\','D:\' -Directory -ErrorAction SilentlyContinue -Depth 0 |
            ForEach-Object { Join-Path $_.FullName 'VC/Auxiliary/Build/vcvars64.bat' } |
            Where-Object { Test-Path $_ } | Select-Object -First 1
        $vcvars = $fallback
    }
    if (-not $vcvars) { throw 'Could not locate vcvars64.bat (Visual Studio C++ build tools).' }

    Write-Step "Importing MSVC environment from $vcvars"
    $lines = & cmd.exe /c "`"$vcvars`" >nul 2>&1 && set"
    if ($LASTEXITCODE -ne 0) { throw "vcvars64.bat failed with exit code $LASTEXITCODE" }
    foreach ($line in $lines) {
        if ($line -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
    }
}

function Find-CMake {
    $cmake = (Get-Command cmake -ErrorAction SilentlyContinue).Source
    if (-not $cmake) {
        $cmake = Get-ChildItem 'C:\Program Files (x86)\CMake\bin\cmake.exe','C:\Program Files\CMake\bin\cmake.exe' -ErrorAction SilentlyContinue |
            Select-Object -First 1 -ExpandProperty FullName
    }
    if (-not $cmake) { throw 'cmake not found on PATH.' }
    return $cmake
}

function Find-Ninja {
    $ninja = (Get-Command ninja -ErrorAction SilentlyContinue).Source
    if ($ninja) { return $ninja }
    $ninja = Get-ChildItem 'C:\','D:\' -Recurse -Filter ninja.exe -ErrorAction SilentlyContinue -Depth 6 |
        Select-Object -First 1 -ExpandProperty FullName
    return $ninja
}

# ------------------------------------------------------------------- main flow
if ($Variant -eq 'cuda') {
    Stage-Cuda
} elseif ($Variant -eq 'dml') {
    Stage-DirectML
} elseif ($Variant -eq 'cpu') {
    Stage-OnnxRuntime
} else {
    # auto: stage the CPU payload (idempotent). CMake then links whichever
    # GPU-capable build is present, preferring cuda over dml.
    Stage-OnnxRuntime
}
if ($StageOrtOnly) { Write-Step 'Done (StageOrtOnly).'; exit 0 }

Import-VcVars

$cmake = Find-CMake
$ninja = Find-Ninja
if (-not $ninja) { throw 'ninja not found; pass -G "Visual Studio 17 2022" manually or install ninja.' }
# NOTE: do not prepend the ninja directory to PATH. A MinGW/MSYS ninja sits next
# to a full gcc toolchain, and putting that directory first makes CMake pick
# gcc/g++ instead of MSVC (the prebuilt onnxruntime.lib is MSVC-only).

if ($Jobs -le 0) { $Jobs = [Environment]::ProcessorCount }
if ($Clean -and (Test-Path $BuildDir)) { Write-Step "Removing $BuildDir"; Remove-Item $BuildDir -Recurse -Force }

Write-Step "Configuring ($Config, Ninja, $Jobs jobs, variant=$Variant)"
& $cmake -S (Join-Path $RepoRoot 'src_cli') -B $BuildDir -G Ninja `
    "-DCMAKE_BUILD_TYPE=$Config" `
    "-DCMAKE_MAKE_PROGRAM=$ninja" `
    "-DDEMUCS_ORT_VARIANT=$Variant"
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed ($LASTEXITCODE)" }

Write-Step 'Building'
& $cmake --build $BuildDir --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw "cmake build failed ($LASTEXITCODE)" }

$exe = Join-Path $BuildDir 'demucs.exe'
if (Test-Path $exe) {
    Write-Step "Built: $exe"
    Write-Host "Usage: $exe <model.onnx> <input.wav> <out dir>"
} else {
    Write-Warning "Build finished but $exe is missing."
}
