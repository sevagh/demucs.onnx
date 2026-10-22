# Windows 构建与使用说明

上游仓库只提供 Linux 构建路径（`scripts/build-ort-linux.sh` + `make cli`，链接静态精简版
`.so`）。在 Windows 上用 **MSVC + 预编译 ONNX Runtime** 跑通需要若干一次性改动；
本文件记录「改了什么、怎么用、什么能信、什么不能信」。

实测环境：

| 组件 | 版本 / 型号 |
| --- | --- |
| CPU | Intel i7-13650HX（14C/20T） |
| GPU | NVIDIA RTX 5060 Laptop（8 GB，Blackwell，驱动 621.01） |
| Python (venv) | 3.14.6 |
| PyTorch / torchaudio | 2.14.1+cpu / 2.11.0+cpu |
| ONNX Runtime (C++) | 1.21.0（NuGet `Microsoft.ML.OnnxRuntime` / `.DirectML` / `.Gpu.Windows`，与 `vendor/onnxruntime` 同版本） |
| ONNX Runtime (Python，验证用) | 1.31.0（CPU）、1.24.4（DirectML）、1.24.1（CUDA） |
| CUDA / cuDNN（pip 轮子） | CUDA 12.9（cublas 12.9.2 / cufft 11.4.1 / curand 10.3.10 / cudart 12.9.79）+ cuDNN 9.27.0.42 |
| 编译器 | MSVC v14.51.36231（VS 2022）+ Ninja |
| CMake | 4.4.2 |

---

## 1. 克隆与子模块

```powershell
git clone --recurse-submodules https://github.com/sevagh/demucs.onnx
# 已 clone 但子模块是空的（本机出现过 vendor/demucs 只剩 .git 指针的情况）：
git submodule update --init --recursive --force
```

## 2. Python 环境（导出 ONNX 用）

```powershell
cd C:\Users\f12fd\OneDrive\Desktop\demucs.onnx
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# **不需要** CUDA 版 torch：ONNX 推理走 ORT，不走 torch
.\.venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

.\.venv\Scripts\python.exe -m pip install einops openunmix dora-search omegaconf pyyaml tqdm numpy lameenc julius onnx onnxruntime

# diffq 需要编译，必须在 vcvars 环境里装
cmd /c "call ""D:\VisualStudioProducts\VC\Auxiliary\Build\vcvars64.bat"" >nul && .\.venv\Scripts\python.exe -m pip install diffq"

.\.venv\Scripts\python.exe -m pip install --no-build-isolation -e .\demucs-for-onnx
```

`scripts/requirements.txt` 里的 `museval` / `musdb` 只在跑评估时需要。

## 3. 导出 ONNX 模型

```powershell
.\.venv\Scripts\python.exe .\scripts\convert-pth-to-onnx.py .\onnx-models
```

首次会下载官方权重 `955717e8-8726e21a.th`（80 MB）到 `~\.cache\torch\hub\checkpoints\`，
生成 `onnx-models\htdemucs.onnx`（约 166 MB）。图（opset 17）：

```
IN   input                [1, 2, 343980]      # 时域分支
IN   onnx::ReduceMean_1   [1, 4, 2048, 336]   # 幅度谱（复数当通道）
OUT  output               [1, 4, 4, 2048, 336]
OUT  <dynamic>            [1, 4, 2, 343980]   # 时域分支输出
```

与 `src/demucs.hpp` 的 `x_onnx_in_shape` / `xt_onnx_in_shape` 一致。6 源模型
（`--six-source`）同理，C++ 端从输出第 2 维自动读出 `nb_sources`。

## 4. 编译 native CLI

```powershell
.\scripts\build-win.ps1                 # auto：优先 CUDA，其次 DirectML，最后 CPU
.\scripts\build-win.ps1 -Variant cuda   # 只装/链 CUDA 版运行时
.\scripts\build-win.ps1 -Variant dml    # 只装/链 DirectML 版
.\scripts\build-win.ps1 -Variant cpu    # 纯 CPU
.\scripts\build-win.ps1 -Clean          # 清掉 build\build-cli-win 重建
```

产物 `build\build-cli-win\demucs.exe`。构建脚本会把运行时摆到（均在 `.gitignore` 里）：

| 目录 | 内容 | NuGet 来源 |
| --- | --- | --- |
| `deps\ort` | CPU：头文件 + `onnxruntime.lib` + `onnxruntime.dll` | `Microsoft.ML.OnnxRuntime` |
| `deps\ort-dml` | CPU + DirectML EP | `Microsoft.ML.OnnxRuntime.DirectML` |
| `deps\ort-cuda` | CPU + CUDA EP（+`onnxruntime_providers_cuda.dll`） | `Microsoft.ML.OnnxRuntime.Gpu.Windows` |

CMake 用 `DEMUCS_ORT_VARIANT`（`auto|cpu|dml|cuda`）选择，只能链一套（各自的
`onnxruntime.dll` 不同）。配置阶段会打印

```
-- ONNX Runtime: CUDA (GPU) build from deps/ort-cuda
```

两者互斥，选了 CUDA 就没有 DirectML，反之亦然。

> `Microsoft.ML.OnnxRuntime.Gpu` 本身只是元包，Windows 载荷在 `.Gpu.Windows`；
> 且它**只带运行时、不带头文件**，脚本会从 `deps\ort` 复用同版本头文件。

---

## 5. 执行提供器（CPU / CUDA / DirectML）

### 5.1 为什么上游默认是 CPU

`src_cli/demucs.cpp` 建了 `Ort::SessionOptions` 却**从不调用** `AppendExecutionProvider`，
ONNX Runtime 于是用默认 CPU EP。不是 GPU 不能用，只是没人接这根线。

### 5.2 怎么切换

```powershell
# 默认（CPU）
.\build\build-cli-win\demucs.exe .\onnx-models\htdemucs.onnx in.wav out

# GPU：CUDA（推荐）
$env:DEMUCS_EP='cuda'          # 'gpu' 是 cuda 的别名
.\build\build-cli-win\demucs.exe .\onnx-models\htdemucs.onnx in.wav out
Remove-Item Env:DEMUCS_EP      # 用完清掉

# GPU：DirectML（不要用于本模型，原因见 5.4）
$env:DEMUCS_EP='dml'
```

启动时打印 `Execution provider: CUDA (device 0)` / `DirectML (device 0)` / `CPU`。

**CUDA 需要 CUDA 12.x + cuDNN 9 的 DLL 在 `PATH` 里**（见 5.6）。若加载失败，程序会打印
`[WARN] CUDA provider unavailable (...); falling back to CPU`，不会静默跑错。

### 5.3 实测数据（同一首 238.6 s 的歌）

| 提供器 | 单段推理 | 整首端到端 | 结果正确性 |
| --- | --- | --- | --- |
| CPU（20 线程） | 2548–3692 ms | **115.2 s**（2.07× 实时） | ✅ |
| **CUDA（RTX 5060）** | **122–139 ms** | **16.4 s**（14.6× 实时，**7.0×**） | ✅ 与 CPU 相关性 0.994–0.998 |
| DirectML（RTX 5060） | 144–148 ms | 17 s | ❌ **算错**（见 5.4） |

CUDA 与 CPU 输出的 stem 差异（RMS 6.2–8.2%、相关性 0.994–0.998）来自**每次运行随机取
的 shift 偏移**，不是精度问题。同输入的模型级对比：相对误差 0.03–0.10%、相关性 1.000000。

整首时间构成（CUDA）：CUDA 上下文/cuDNN 初始化约 2–4 s + 34 段 × ~0.15 s。
首次运行会略慢（冷缓存）。

### 5.4 DirectML 在 htdemucs.onnx 上产出的是垃圾

速度同样快，数值彻底错，ONNX Runtime **不报错**。三个层面验证：

| 验证 | CPU | DirectML |
| --- | --- | --- |
| 模型输出 `absmean`（时域分支） | 0.0918 | 53.06（关掉全部图优化）/ 1761（默认优化） |
| 与 CPU 输出相关系数 | — | 0.0007 / 0.0016 |
| 端到端 drums RMS | 0.107 | 2684.6 |
| 端到端与 CPU 相关性 | — | −0.0003 |

关掉 `ORT_DISABLE_ALL` 仍然错，所以不是某个 fusion 的问题，而是该 EP 对这张图的实现缺陷。
**结论：用 CUDA，不要用 DirectML。**

### 5.5 输出自检（防止静默出错）

坏 EP 会"成功"返回垃圾，所以 CLI 在写盘前有一道数值闸门（`outputs_are_plausible()`）：
输出必须全为有限值、峰值不超过输入峰值的 100 倍。不通过则打印原因、**一个文件都不写**、
退出码 `3`。CPU/CUDA 都能正常通过。

### 5.6 CUDA 运行时依赖（本机已验证）

ORT 的 CUDA EP 在 Windows 上是延迟加载的：`onnxruntime_providers_cuda.dll` 会依赖这些 DLL，
缺一个就整体回落到 CPU。

| 需要 | 提供者 |
| --- | --- |
| `cudart64_12.dll` | `nvidia-cuda-runtime-cu12` |
| `cublas64_12.dll` / `cublasLt64_12.dll` / `nvblas64_12.dll` | `nvidia-cublas-cu12` |
| `cudnn64_9.dll` + `cudnn_ops64_9.dll`、`cudnn_cnn64_9.dll`、`cudnn_graph64_9.dll`、`cudnn_engines_*64_9.dll` 等 | `nvidia-cudnn-cu12` |
| `cufft64_11.dll` / `cufftw64_11.dll` | `nvidia-cufft-cu12` |
| `curand64_10.dll` | `nvidia-curand-cu12` |
| `nvJitLink_120_0.dll` | `nvidia-nvjitlink-cu12` |

装法与 PATH（不需要装完整 CUDA Toolkit，pip 轮子就够）：

```powershell
python -m venv .cuda-runtime
.\.cuda-runtime\Scripts\python.exe -m pip install numpy nvidia-cudnn-cu12 nvidia-cublas-cu12 `
    nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 nvidia-curand-cu12

# 每次开新终端时把这些 bin 目录加进 PATH
$nv = ".\.cuda-runtime\Lib\site-packages\nvidia"
$env:PATH = ((Get-ChildItem $nv -Directory | ForEach-Object { Join-Path $_.FullName 'bin' } |
    Where-Object { Test-Path $_ }) -join ';') + ';' + $env:PATH
```

> 轮子装在哪里都行，只要把这些 `nvidia\*\bin` 加进 `PATH`。本机验证时用的是
> `build\venv-cuda\Lib\site-packages\nvidia`（见第 7 节），下文示例也用这个路径；
> 若你按上面装到 `.cuda-runtime`，把 `$nv` 指过去即可。

**版本匹配很重要**：

* ORT **1.24.x 及更早**的 GPU 版对应 **CUDA 12.x + cuDNN 9**
* ORT **1.31** 的 GPU 版要求 **CUDA 13**（报错会写 `Require cuDNN 9.* and CUDA 13.*`）
  —— 用 CUDA 12 轮子就会看到 `cublasLt64_13.dll which is missing`
* 本仓库 C++ 侧用 1.21.0，配 CUDA 12.9 + cuDNN 9.27 实测正常

诊断技巧：如果只是"静默回落 CPU"，把日志级别调高就能看到缺哪个 DLL：

```python
so = ort.SessionOptions(); so.log_severity_level = 0   # 0=verbose
ort.InferenceSession(model, so, providers=[('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider'])
```

### 5.7 GPU 与进程退出（已修）

接上 CUDA 后出现过两个只在 GPU 下发作的问题，都已修掉并写在代码注释里：

1. **退出时段错误 0xC0000005**：`demucs_model` 里 `Ort::Env` 原来声明在 `sess` 之后，
   成员按声明的**逆序**析构，于是 Session 比 Env 活得久；另外 `allocator` / `run_options`
   两个进程级全局 Ort 对象会在静态析构期才释放，那时 GPU provider 已经拆了。
   修法：`Ort::Env` 提前到首位、两个全局对象改为**故意泄漏**的单例（引用绑定）。
2. 即使这样，ORT CUDA provider 在进程静态拆除阶段仍会踩内存。CLI 在**所有 stem 写盘
   并 flush 之后**直接 `std::_Exit(0)` 跳过静态析构，退出码回到 0。
   如果要把这套代码编成库、需要正常析构链，请把 `std::_Exit(0)` 换回 `return 0` 并自行
   管理生命周期（CPU EP 下两者都没问题）。

## 6. 采样率：进 44.1 kHz 与回采样

demucs.onnx 内部 STFT 固定 44.1 kHz（`demucsonnx::SUPPORTED_SAMPLE_RATE`），
libnyquist **只解码不重采样**。非 44.1 kHz 素材必须先转。

### 6.1 输入侧：任意 WAV → 44.1 kHz

```powershell
.\.venv\Scripts\python.exe .\scripts\prepare-input.py "E:\Music\许嵩 何曼婷 - 素颜.wav" .\work\in_44k.wav
```

做的事（`scripts/prepare-input.py`，逻辑在 `scripts/wavio.py`）：

* 解 PCM16 / PCM24 / PCM32 / IEEE-float WAV（`wave` 模块读不了 float WAV，这里自己解析 RIFF）
* 单声道复制成双声道；>2 声道直接拒绝
* 带限有理重采样到 44.1 kHz
* 峰值 > 1.0 时整体归一（打印实际系数）
* 输出 16-bit PCM（libnyquist 读得最稳）

**为什么重采样要分块**：torchaudio 的 `resample` 内部是一次 `conv1d`，对整条 4 分钟
192 kHz 素材会按"长度 × 抽头"分配 im2col 缓冲（几十 GB）。`wavio.resample()` 按
`up/down` 化简后的比率切片，块起点与 margin 都取 `down` 的整数倍，每块因此落在与整段
相同的采样相位上，裁掉两侧 margin 后**直接拼接、无需交叉淡化**。验证：同一段真实音频
分别用分块（含 200 块）与整段一次重采样，**逐样本差值 = 0**。

例：192 kHz / 24-bit → 44.1 kHz / 16-bit，238.61 s 保持（10 522 554 帧）。

### 6.2 输出侧：stem → 交付格式（可选回采样到源采样率）

C++ CLI 写出的是 libnyquist 的 32-bit IEEE-float WAV（`wFormatTag=3`），且 Demucs 不保能量，
单轨峰值常超 1.0（本曲 drums 1.076、vocals 1.042），直接转 PCM 会削瞬态。收尾用：

```powershell
# 交付 44.1 kHz / 24-bit PCM（默认）
.\.venv\Scripts\python.exe .\scripts\render-stems.py .\work\stems "E:\Music\Output" "许嵩 何曼婷 - 素颜"

# 回采样到源文件采样率（192 kHz），与原始工程对齐
.\.venv\Scripts\python.exe .\scripts\render-stems.py .\work\stems "E:\Music\Output" "许嵩 何曼婷 - 素颜" `
    --match-rate "E:\Music\许嵩 何曼婷 - 素颜.wav"

# 或直接指定
... --out-rate 48000
```

`scripts/render-stems.py` 做的事：

1. **共用一个增益**：取四轨峰值压到 `--peak`（默认 0.995）。回采样时峰值取**回采样之后**
   的（带限插值有 Gibbs 过冲：本曲 192 kHz 渲染实际峰值 1.1159，比 44.1 kHz 的 1.0757 高，
   若按 44.1 kHz 峰值定增益，192 kHz 版就会削）。全轨同一系数 → 混音相对电平不变，
   并打印还原系数（本曲 `x1.081144`）。
2. 可选重采样（复用同一个分块重采样器，反向同样成立：44100→192000 是 640/147）。
3. 写 **24-bit PCM**（兼容性比 float32 好，比 16-bit 留更多混音余量）。

**诚实说明**：回采样到 192 kHz 只对齐采样网格，**不会**恢复 22.05 kHz 以上内容 ——
网络是在 44.1 kHz 上跑完的，带宽已经没了。

## 7. 完整流水线（本机那一首）

```powershell
$R = "C:\Users\f12fd\OneDrive\Desktop\demucs.onnx"
$PY = "$R\.venv\Scripts\python.exe"
$SRC = "E:\Music\许嵩 何曼婷 - 素颜.wav"      # 192 kHz / 24-bit

# CUDA 运行时 DLL（本机路径；按 5.6 装在别处就改这里）
$nv = "$R\build\venv-cuda\Lib\site-packages\nvidia"
$env:PATH = ((Get-ChildItem $nv -Directory | % { Join-Path $_.FullName 'bin' } | ? { Test-Path $_ }) -join ';') + ';' + $env:PATH

# 1. 转 44.1 kHz
& $PY "$R\scripts\prepare-input.py" $SRC "$R\build\work\in_44k.wav"

# 2. 分离（CUDA 16.4 s / CPU 115.2 s）
$env:DEMUCS_EP = 'cuda'
& "$R\build\build-cli-win\demucs.exe" "$R\onnx-models\htdemucs.onnx" "$R\build\work\in_44k.wav" "$R\build\work\stems"
Remove-Item Env:DEMUCS_EP

# 3. 渲染交付（24-bit PCM + 统一增益）
& $PY "$R\scripts\render-stems.py" "$R\build\work\stems" "E:\Music\Output" "许嵩 何曼婷 - 素颜"
```

4 个 stem：`- drums.wav` / `- bass.wav` / `- other.wav` / `- vocals.wav`
（6 源模型多 `guitar` / `piano`）。本曲实测：bass 99.8% 能量 <200 Hz，vocals 90.4% 在
200–2 kHz，四轨相加与原混音相对误差 8.1%，削波采样 0。

---

## 相对上游的改动

| 文件 | 改动 | 原因 |
| --- | --- | --- |
| `src_cli/CMakeLists.txt` | `cmake_minimum_required` 3.0→3.16；`if(MSVC)` 选项分支（`/O2 /fp:fast /EHsc /bigobj` ↔ 上游 GCC 的 `-O3 -march=native -ffast-math`）；`DEMUCS_ORT_VARIANT`（auto/cpu/dml/cuda）选择运行时目录并定义 `DEMUCS_HAS_CUDA`/`DEMUCS_HAS_DML`；POST_BUILD 复制 `onnxruntime*.dll`；session 头目录加入 include 路径 | 上游硬编码 `.so` 与 GCC flag；provider 头以兄弟路径包含 C API 头 |
| `src_cli/demucs.cpp` | 补 `#include <iomanip>`；`p_target.string()`；`DEMUCS_EP` 选择 EP（cpu/cuda/dml）；`outputs_are_plausible()` 自检（失败退出码 3）；结尾 `std::_Exit(0)` | MSVC 类型/头文件差异；上游固定 CPU；坏 EP 会静默产垃圾；GPU 拆除崩溃 |
| `src/demucs.hpp` | `Ort::Env` 移到成员首位；`allocator`/`run_options` 改为引用 | 析构顺序：Session 不能比 Env 活得久 |
| `src/model_inference.cpp` | 两个全局 Ort 对象改为故意泄漏的单例 | 静态析构期 provider 已卸载，会踩内存 |
| `scripts/convert-pth-to-onnx.py` | `torch.onnx.export(..., dynamo=False)` | torch ≥ 2.9 默认 dynamo 导出器（还需 `onnxscript`），图与 `src/` 期望的 opset17 TorchScript 图不匹配 |
| `scripts/build-win.ps1` | 新增 | 摆 CPU/DirectML/CUDA 运行时、导 vcvars、Ninja 配置编译一条龙 |
| `scripts/prepare-input.py` | 新增 | 无 ffmpeg 把任意 WAV 转 44.1 kHz（分块重采样） |
| `scripts/render-stems.py` | 新增 | float32 stem → 24-bit PCM，统一增益防削波，可回采样到源采样率 |
| `scripts/wavio.py` | 新增 | WAV 读写 + 分块重采样器（上面两脚本共用） |
| `.gitignore` | 增加 `deps/`、`.venv/`、`.cuda-runtime/` | 本地运行时与虚拟环境不入库 |

## 已知注意点

* **用 CUDA，不要用 DirectML**（5.4）。
* CUDA 运行时 DLL 必须在 `PATH`（5.6），否则静默回落 CPU —— 看启动打印的
  `Execution provider:` 那一行确认。
* ORT 版本 ↔ CUDA 版本要配对：1.24.x 及更早 = CUDA 12.x，1.31 = CUDA 13（5.6）。
* `build-win.ps1` **不会**把 ninja 目录塞进 `PATH`。WinLibs 的 `ninja.exe` 和整套 MinGW
  gcc 同目录，排在最前会让 CMake 选 `g++` 而不是 MSVC，而 MSVC 格式的 `onnxruntime.lib`
  只能在 MSVC 下链接（表现为 `_BitScanReverse` 等 intrinsic 未声明）。
* 输入必须 44.1 kHz（用 6.1 的脚本转）；单/双声道可以，>2 声道被拒。
* C++ CLI 输出是 32-bit IEEE **float** WAV，入库前用 6.2 的脚本转 24-bit PCM。
* 每次运行取随机 shift 偏移（日志 `shift offset is: ...`），两次结果不会逐样本相同。
* 上游 `.ort` 精简格式（`convert-model-to-ort.sh`）与 `onnx-models/model/*.ort.c`（bin2c
  内嵌，上游自己说 1.2 GB c 文件太离谱）在 Windows 路径都不需要：`deps\ort*` 是完整运行时。
