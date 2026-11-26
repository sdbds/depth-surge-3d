# Installation Guide

Install the application first, then prepare the weights for **one** depth backend. The wheel contains the CLI and the compiled C++17 Quality extension. Model weights, the separate V2 source tree, FFmpeg and the Web UI checkout are not supplied by the wheel.

## Windows x64 (Recommended)

1. Install [Git for Windows](https://git-scm.com/downloads/win) and [uv](https://docs.astral.sh/uv/getting-started/installation/). The installer requests 64-bit CPython 3.11; uv can download it if needed. The project metadata allows 3.10 through 3.12, but the Windows accelerator wheels differ by Python version.
2. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with **Desktop development with C++**, the MSVC x64 compiler, and a Windows SDK. These compile the project's Quality extension. It does **not** need CUDA Toolkit or a preinstalled copy of PyTorch to build.
3. Install [FFmpeg](https://ffmpeg.org/download.html). Add the directory containing both `ffmpeg.exe` and `ffprobe.exe` to `PATH`. Reopen PowerShell and check `uv --version`, `git --version`, `ffmpeg -version` and `ffprobe -version`.
4. Use a volume with space for the environment, uv cache, and the chosen model. The Windows runtime includes large CUDA-enabled PyTorch wheels even if you plan to run inference on the CPU. For GPU use you need a driver compatible with the locked CUDA 13.0 wheels; a CUDA toolkit installation alone is not a driver.

From PowerShell:

```powershell
git clone https://github.com/sdbds/depth-surge-3d.git
cd depth-surge-3d
.\1.install-uv-qinglong.ps1
.\.venv\Scripts\depth-surge-3d.exe --help
```

The script installs the **project itself**, non-editably, with `uv sync --locked --no-editable --no-dev --inexact --python 3.11`. It checks dependencies in a new environment and verifies that both the imported package and compiled Quality module belong to the installed wheel. No activation, `pip install -r`, or unpacked `.pyd` in the checkout is required. It does not install system tools, set permanent environment variables, or force cache symlinks. The first installation needs access to PyPI, GitHub and the PyTorch wheel index unless their artifacts are already cached. The locked Windows 3.11 FlashAttention wheel is sourced from the [third-party release](https://github.com/sdbds/flash-attention-for-windows/releases/tag/2.8.4), not from PyTorch.

**Keep an existing environment:** The script defaults to the checkout's `.venv` (or honors `UV_PROJECT_ENVIRONMENT`). It reuses a valid x64 CPython 3.11 venv without pruning unrelated packages; dependencies the project needs may be updated to their locked versions. It refuses an incompatible interpreter, a non-venv directory or a linked target instead of replacing it. For an independent installation, choose a new directory, without touching `.venv`:

```powershell
$runtime = Join-Path $env:TEMP ("depth-surge-" + [guid]::NewGuid())
.\1.install-uv-qinglong.ps1 -EnvironmentPath $runtime
$cli = Join-Path $runtime 'Scripts\depth-surge-3d.exe'
$python = Join-Path $runtime 'Scripts\python.exe'
Push-Location $env:TEMP
try {
    & $cli --help
    & $python -I -c 'import _quality_kd_native as kd; print(kd.__file__)'
} finally { Pop-Location }
```

Both commands should succeed outside the checkout, and the extension should come from the selected environment. Use that same `$cli` for conversions. For the Web UI, run `& $python app.py` from the checkout, then open [localhost:5000](http://localhost:5000).

## Linux

Install Git, uv, a C++17 compiler, the development headers for your Python, and FFmpeg with `ffprobe`. For Linux x86_64, use the same project install from a fresh checkout with no existing `.venv`:

```bash
uv sync --locked --no-editable --no-dev --inexact --python 3.11
.venv/bin/depth-surge-3d --help
```

To preserve an existing `.venv`, **before** syncing set `export UV_PROJECT_ENVIRONMENT="$(mktemp -d)/runtime"` and use `"$UV_PROJECT_ENVIRONMENT/bin/depth-surge-3d"`. For GPU acceleration the driver must support the locked PyTorch wheel. Linux FlashAttention is optional: only add `--extra flash-attn` to the sync command when you have the compatible CUDA build toolchain. The base installation does not require that optional build. The Windows x64/3.11 lock profile already includes its matching accelerator wheel. Do not infer support for every OS and architecture from the Python version range.

## First Conversion and Weights

**V3 Small is the shortest first run.** Its [inference code](https://github.com/ByteDance-Seed/Depth-Anything-3) is installed at the Git revision in `uv.lock`. The [DA3-SMALL weights](https://huggingface.co/depth-anything/DA3-SMALL) are separate. This example fetches them into the Hugging Face cache on first use; replace `input.mp4` with a real clip:

```powershell
.\.venv\Scripts\depth-surge-3d.exe input.mp4 --depth-model-version v3 --model-size vits --end 00:01 --vr-resolution custom:640x360 --stereo-render-mode quality
```

`--help` needs no weights; conversion does. For an offline or repeatable run, stage a specific snapshot while online, then point `--model` at the local directory, **without** `--model-size`. For example:

```powershell
.\.venv\Scripts\hf.exe download depth-anything/DA3-SMALL --revision e08cab65ca0ec38e7826075418411ab90cab4da3 --local-dir models/DA3-SMALL
Get-FileHash models/DA3-SMALL/model.safetensors -Algorithm SHA256
.\.venv\Scripts\depth-surge-3d.exe input.mp4 --depth-model-version v3 --model .\models\DA3-SMALL --end 00:01 --vr-resolution custom:640x360 --stereo-render-mode quality
```

The [pinned snapshot](https://huggingface.co/depth-anything/DA3-SMALL/tree/e08cab65ca0ec38e7826075418411ab90cab4da3) contains `config.json` and the 137,248,940-byte `model.safetensors` (SHA-256 `364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf`). This pin applies to the explicit example, not to every V3 model name.

### Video-Depth-Anything V2

The CLI defaults to V2 when no backend is supplied. Unlike V3, V2 needs a separate [upstream source checkout](https://github.com/DepthAnything/Video-Depth-Anything) under `vendor/Video-Depth-Anything` **relative to the working directory** and a matching video checkpoint. Run these commands from the Depth Surge checkout for a pinned **Small, relative-depth** example:

```powershell
git clone https://github.com/DepthAnything/Video-Depth-Anything.git vendor/Video-Depth-Anything
git -C vendor/Video-Depth-Anything checkout --detach 4f5ae23172ba60fd7bc11ef671cca678842c7072
.\.venv\Scripts\hf.exe download depth-anything/Video-Depth-Anything-Small video_depth_anything_vits.pth --revision 256875362cff76724b920335dfb4b29dd611f66e --local-dir models/Video-Depth-Anything-Small
Get-FileHash models/Video-Depth-Anything-Small/video_depth_anything_vits.pth -Algorithm SHA256
.\.venv\Scripts\depth-surge-3d.exe input.mp4 --depth-model-version v2 --model-size vits
```

The [Small video weight](https://huggingface.co/depth-anything/Video-Depth-Anything-Small/tree/256875362cff76724b920335dfb4b29dd611f66e) has SHA-256 `13379300b739e659f076a59d52e9801bd8d38c541a7e71f73bbca4dcfb013609`. Do not substitute the standalone *image* Depth-Anything-V2 weights produced by `scripts/download_models.sh`. Without `--model-size vits` the V2 CLI requests the Large checkpoint instead. Review upstream model licenses for your selected size and use.

### Optional MoGe-2

The base install does not request MoGe. To opt in, use the same checkout and environment (`-EnvironmentPath` if customized):

```powershell
.\1.install-uv-qinglong.ps1 -WithMoGe2
```

On Linux the corresponding command is `uv sync --extra moge2 --locked --no-editable --no-dev --inexact --python 3.11`. Microsoft MoGe's source is pinned to [`925b8ed835a7a9cdb7578ba15c658a0afc969030`](https://github.com/microsoft/MoGe/tree/925b8ed835a7a9cdb7578ba15c658a0afc969030). Its weights resolve at these immutable revisions:

| UI/setting | Parameters | Repository | Revision |
| --- | ---: | --- | --- |
| Small / `vits` | 35M | `Ruicheng/moge-2-vits-normal` | `679230677b4d282c6f304189a93e98e14f085902` |
| Base / `vitb` default | 104M | `Ruicheng/moge-2-vitb-normal` | `54ad3a693e61907ea4633d13dec6ee682fa09419` |
| Large / `vitl` | 326M | `Ruicheng/moge-2-vitl` | `39c4d5e957afe587e04eec59dc2bcc3be5ecd968` |

Offline use requires the exact cached snapshot or a local model directory containing `model.pt`. MoGe runs in float32 on CPU and float16 on CUDA; an out-of-memory error does not silently change variant, geometry mode, resolution, or device. See the [parameters reference](PARAMETERS.md) and [third-party notices](../THIRD_PARTY_NOTICES.md).

### Optional VDPP Checkpoint

VDPP is disabled by default. The first run with `--temporal-postprocessor vdpp` verifies CUDA and disk capacity, then downloads the pinned v1.0 checkpoint to `models/VDPP/vdpp.pth`. The expected file is 116,485,370 bytes with SHA-256 `7368315b126093f0335147f42a1920f255d529613bfffc5c6cf4ef832deb73a7`. Downloads are locked and verified before publication. Generation is CUDA-only; CPU and MPS may still render an already complete, validated stabilized cache. The vendored inference source is pinned to upstream revision `73cc2b4dc6b3b5cfb2e37f51e452461e03fe26f5` and retains its Apache-2.0 license.

## Troubleshooting

- **PowerShell blocks the script:** Review the local file first. For a trusted checkout, `powershell -NoProfile -ExecutionPolicy Bypass -File .\1.install-uv-qinglong.ps1` affects that process only; no machine-wide policy change is needed.
- **Compiler, `cl.exe`, SDK or header error:** Install the x64 C++ workload and Windows SDK, then retry in a fresh **Developer PowerShell for Visual Studio**. On Linux install the C++ compiler and matching Python development headers. Do not copy a `.pyd` from another interpreter or turn off build isolation.
- **Missing FFmpeg or probe:** Both `ffmpeg -version` and `ffprobe -version` must work in the launching shell. A Python wrapper is not the system `ffprobe` executable.
- **Lock or download failure:** Use `pyproject.toml` and `uv.lock` from the same revision. Check access to the named host and provide sufficient cache/target disk space. Do not bypass the lock, relax index security, disable TLS, or replace a wheel with a similarly named download.
- **Only the first conversion fails:** Check the backend, model directory, file name, revision and digest above. A missing V2 source tree or an unavailable model download is not fixed by reinstalling the Quality extension.
