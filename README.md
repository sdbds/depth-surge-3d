# Depth Surge 3D

[![CI](https://github.com/Tok/depth-surge-3d/actions/workflows/ci.yml/badge.svg)](https://github.com/Tok/depth-surge-3d/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Tok/depth-surge-3d/branch/main/graph/badge.svg)](https://codecov.io/gh/Tok/depth-surge-3d)
[![Python 3.9-3.12](https://img.shields.io/badge/python-3.9--3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Convert 2D videos to 3D VR format using AI depth estimation.**

Depth Surge 3D transforms flat videos into stereoscopic 3D for VR headsets using **Depth Anything V3**, **Video-Depth-Anything V2**, **See-Through Marigold**, or the optional **MoGe-2** backend. It predicts depth with the selected backend, then generates left and right eye views for immersive stereoscopic viewing. V2 applies model-native temporal inference independently within each detected shot.

## Key Features

- **Dual Depth Models**:
  - **Depth Anything V3** (default): 50% lower VRAM, faster processing, optimized for modern GPUs
  - **Video-Depth-Anything V2**: Shot-aware temporal consistency with fixed 32-frame windows, 10-frame overlap, and state resets at detected cuts
- **Optional MoGe-2 Backend**: Pinned Small, Base, and Large variants with metric
  depth and focal output. Relative geometry remains the default.
- **AI Upscaling**: Optional Real-ESRGAN enhancement (2x/4x) for higher output resolution
- **CUDA Hardware Acceleration**: NVENC H.265 encoding and GPU-accelerated frame decoding
- **Configurable Depth Quality**: Adjustable depth map resolution (518px to 4K) for quality vs. speed
- **Optional VDPP Stabilization**: Experimental depth-only temporal post-processing for any video depth backend using relative geometry
- **Multiple VR Formats**: Side-by-side and over-under stereoscopic formats
- **Flexible Resolutions**: Square (VR-optimized), 16:9 (standard), cinema, and custom resolutions up to 8K
- **Resume Capability**: Intelligent step-level resume for interrupted processing
- **Audio Preservation**: Maintains original audio synchronization with lossless FLAC extraction
- **Web Interface**: Modern browser-based UI with real-time progress tracking and live previews
- **Wide Format Support**: Cinema, ultra-wide, and standard aspect ratios

## Quick Start

### Installation

```bash
git clone https://github.com/Tok/depth-surge-3d.git depth-surge-3d
cd depth-surge-3d
chmod +x setup.sh
./setup.sh
```

The setup script automatically installs all dependencies, downloads the Video-Depth-Anything model (~1.3GB), and verifies your system.

**See [Installation Guide](docs/INSTALLATION.md) for detailed setup instructions.**

### Usage

**Web UI (Recommended):**
```bash
./run_ui.sh
# Opens http://localhost:5000 in your browser
```

**Command Line:**
```bash
# Basic usage
python depth_surge_3d.py input_video.mp4

# Process specific time range with custom settings
python depth_surge_3d.py input_video.mp4 -s 01:30 -e 03:45 -f over_under --resolution 4k

# Add experimental generic temporal stabilization (CUDA generation only)
python depth_surge_3d.py input_video.mp4 --temporal-postprocessor vdpp
```

**UV Command Line**

In WSL required to export a UV variable

```bash
export UV_LINK_MODE=copy
```

```bash
# Basic usage
uv run python depth_surge_3d.py input_video.mp4
```

**Quick Start Script:**
```bash
# Process a clip with optimized settings
./start.sh 1:11 2:22
```

**See [Usage Guide](docs/USAGE.md) for comprehensive usage examples.**

### Experimental MoGe-2 Metric Camera

MoGe-2 can drive the existing relative geometry path or the Experimental
`metric_camera` path. Flat rectilinear side-by-side output remains the main
playback path. Metric camera mode is limited to `side_by_side` with distortion
disabled and square-pixel (`1:1` sample-aspect-ratio) sources. It does not
establish calibrated physical scale, physically correct reconstruction,
improved stereo quality, viewing comfort or safety, or superiority over the
relative default.

MoGe-2 performs per-frame depth and focal estimation. Temporal stability on video is not guaranteed; depth or focal drift may be visible across frames.

Install the optional backend with `uv sync --extra moge2`; see the
[Installation Guide](docs/INSTALLATION.md) for the pinned artifacts and the
[Parameters Reference](docs/PARAMETERS.md) for the projection controls.

## Requirements

- Python 3.9, 3.10, 3.11, or 3.12 (Python 3.13+ not yet supported due to dependency limitations)
- FFmpeg
- CUDA 13.0+ (required for GPU acceleration)
- CUDA-compatible GPU (optional but strongly recommended; required to generate VDPP output)

## Documentation

**User Guides:**
- **[Installation Guide](docs/INSTALLATION.md)** - Detailed setup instructions and troubleshooting
- **[Usage Guide](docs/USAGE.md)** - Complete usage examples and workflows
- **[Parameters Reference](docs/PARAMETERS.md)** - All command-line options and settings explained
- **[VR Headset Compatibility](docs/VR_HEADSET_COMPATIBILITY.md)** - Specs and optimal settings for top 10 VR devices
- **[Performance Benchmarks](docs/PERFORMANCE.md)** - GPU benchmarks, VRAM usage, and optimization guide
- **[Web UI Guide](docs/WEB_GUI.md)** - VDPP controls, resume behavior, and progress
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and performance tips

**Technical Documentation:**
- **[Architecture](docs/ARCHITECTURE.md)** - Technical details and processing pipeline
- **[Development Guide](docs/DEVELOPMENT.md)** - VDPP vendoring, tests, and quality gate
- **[Contributing Guide](docs/CONTRIBUTING.md)** - Development workflow and CI/CD setup
- **[Coding Standards](docs/CODING_GUIDE.md)** - Code quality requirements and best practices

**Development:**
- **[Development Notes](docs/CLAUDE.md)** - Quick reference for development
- **[Project Roadmap](docs/TODO.md)** - Planned features and improvements

## Output Structure

Each processing session creates a self-contained timestamped directory:

```
output/
└── timestamp_videoname_timestamp/
    ├── original_video.mp4      # Source video
    ├── original_audio.flac     # Pre-extracted audio
    ├── 00_original_frames/    # Extracted frames
    ├── 03_disparity_maps/     # Canonical relative disparity
    ├── 03_disparity_stabilized/ # Optional VDPP artifact
    ├── 99_vr_frames/          # Final VR frames
    └── videoname_3D_side_by_side.mp4  # Final 3D video
```

## VR Viewing

Generated videos work with:
- VR headsets (Meta Quest, HTC Vive, etc.)
- Cardboard VR viewers
- 3D video players supporting side-by-side or over-under formats

## Performance

- **GPU Processing**: ~2-4 seconds per output frame (RTX 4070+ class)
- **CPU Processing**: ~30-60 seconds per output frame
- **Typical 1-minute clip**: ~2-4 hours on modern GPU at 60fps output

## Attribution

This project uses state-of-the-art depth estimation models:

- **[Depth Anything V3](https://github.com/ByteDance-Seed/Depth-Anything-3)** - Default model with improved memory efficiency and performance
- **[Video-Depth-Anything V2](https://github.com/DepthAnything/Video-Depth-Anything)** - Shot-aware temporal depth estimation with fixed 32-frame windows and 10-frame overlap
- **[Microsoft MoGe](https://github.com/microsoft/MoGe)** - Optional pinned
  MoGe-2 metric depth backend
- **[VDPP](https://github.com/injun-baek/VDPP)** - Optional video depth post-processing; the pinned inference subset and Apache-2.0 notice ship under `src/depth_surge_3d/_vendor/vdpp`

These models use vision transformer architectures for monocular depth prediction.

## License

MIT License - see [LICENSE](LICENSE) file for details.

**Third-Party Components**: Please review the [third-party notices](THIRD_PARTY_NOTICES.md)
and the [Video-Depth-Anything license](https://github.com/DepthAnything/Video-Depth-Anything)
for the licenses that apply to upstream components.

## Quality Expectations, Parallax-Glitchwave and Z-Collapse Slopcore Aesthetics

This tool converts monocular video to pseudo-stereo using AI depth estimation. Results can be compelling for many types of content but will never match true stereo cameras or specialized VR equipment.

**Best results with:**
- Clear depth variation (landscapes, interiors, people)
- Good lighting and detail
- Source resolution 1080p or higher
- Steady camera movement

**May struggle with:**
- Mirrors, glass, water reflections
- Very dark or low-contrast scenes
- Fast motion or rapid camera movements

**Artifact Embracement**: Expect algorithmic stereo divergence, synthetic depth layers, and monocular hallucinations. These AI-generated depth discontinuities create a unique aesthetic - depth-drift, disparity shimmer, and temporal judder may become part of the experience.

See the [Troubleshooting Guide](docs/TROUBLESHOOTING.md) for detailed quality expectations and optimization tips.

## Screenshots

**[Full Application Walkthrough (v0.9.1)](docs/Depth-Surge-3D-0.9.1-screenshot.png)** - Complete web interface screenshot showing all processing steps (large file: 2.1 MB, 1371x8733px)
