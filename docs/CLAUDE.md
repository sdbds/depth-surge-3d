# Depth Surge 3D - Development Guide

**Version**: 0.9.0
**Purpose**: 2D to 3D VR video converter using AI depth estimation (Depth Anything V3, Video-Depth-Anything V2, See-Through, MoGe-2)

---

## Quick Start

Install the project using the [installation guide](INSTALLATION.md), then activate `.venv`:

```bash
./test.sh           # Verify installation (CUDA, dependencies, models)
./run_ui.sh         # Launch web UI (http://localhost:5000)

# CLI usage
python depth_surge_3d.py input.mp4
python depth_surge_3d.py input.mp4 --vr-resolution 16x9-1080p -s 1:00 -e 2:00
```

---

## Code Quality (Pre-Commit Gate)

```bash
# Run all checks at once (mirrors CI)
./scripts/pre-commit-checks.sh          # Linux/macOS
.\scripts\pre-commit-checks.ps1         # Windows

# Or run individually
black src/ tests/ app.py
flake8 src/ tests/ app.py --count --show-source --statistics
mypy src/depth_surge_3d --ignore-missing-imports
pytest tests/unit -q
```

Rebuild and install the native Quality C++17 extension after modifying `native/quality_kd_native.cpp`:

```powershell
uv run --no-project --python .venv\Scripts\python.exe --with pybind11==3.0.4 --with "setuptools>=84" python scripts/build_quality_task0_native.py
```

Coding standards ([CODING_GUIDE.md](CODING_GUIDE.md)):
- Modern Python 3.10+ type hints (`dict`, `list`, `X | None`); `mypy src/depth_surge_3d --ignore-missing-imports` must report 0 errors.
- Max McCabe complexity: 10 (enforced by `flake8`).
- `black` formatting and `flake8` 0 errors across `src/`, `tests/`, and `app.py`.

---

## Tech Stack

- **Backend**: Flask + Socket.IO (threading)
- **AI Models**: Depth Anything V3 (Web UI default), Video-Depth-Anything V2 (CLI default, 32-frame shot windows), See-Through Marigold, optional MoGe-2 (metric depth + focal)
- **Temporal Stabilization**: Optional VDPP post-processor (`03_disparity_stabilized/`)
- **Stereo Rendering**: Torch/Triton Fast renderer + compiled C++17 Quality renderer (`_quality_kd_native`)
- **Video**: FFmpeg with `hevc_nvenc`, `libx265`, or `libx264` and zero-copy streamed `bgr24` piping

---

## Architecture Essentials

### Processing Pipeline
1. Frame Extraction → 2. Scene Analysis → 3. Native Raw Depth → 4. Canonical Disparity (+ optional VDPP) or Metric Geometry → 5. DIBR Stereo Pairs (`fast` or `quality`) → 6. Optional Projection/Crop/Upscale → 7. VR Assembly / Streamed `bgr24` Encoding → 8. Audio Integration & Lightweight `ffprobe` Validation

### Output Structure
```text
output/<batch>/
├── 00_original_frames/metadata.json
├── 01_scene_data/
├── 02_depth_raw/
├── 03_disparity_maps/            # or 03_metric_geometry/
├── 03_disparity_stabilized/      # optional VDPP
├── 04_left_frames/ + 04_right_frames/
├── 04_stereo_diagnostics/summary.json
├── 99_vr_frames/                 # only when keep_intermediates=true
├── processing-settings-v5.json
└── <videoname>_3D_<format>.mp4
```

### Memory & Performance
- Host memory budgets (`core/host_memory.py`) scale dynamically to available RAM (2–4 GiB for Quality host preflight, unblocking 4K Quality; 512 MiB–2 GiB for raw depth prefetch).
- Default CUDA stereo splat band budgets scale up to 1.5 GiB (`free_vram // 4`).
- Native C++ Quality kernels share a persistent 7-worker thread pool and use a parallel O(H·W) L1 distance transform for edge bands.

---

## Documentation Map

- `README.md` / `README.zh-CN.md`: Quick start and overview
- `docs/INSTALLATION.md`: Platform setup and model weights
- `docs/USAGE.md` & `docs/PARAMETERS.md`: CLI/UI workflows and flags
- `docs/ARCHITECTURE.md`: Technical pipeline and module architecture
- `docs/CODING_GUIDE.md` & `docs/CONTRIBUTING.md`: Standards and CI/CD
- `docs/TROUBLESHOOTING.md`: Common issues and tips
