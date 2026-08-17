# Changelog

All notable changes to Depth Surge 3D will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Optional MoGe-2 Small, Base, and Large backends with immutable source and
  weight revisions.
- Experimental metric-camera geometry for flat rectilinear side-by-side output,
  with clip-global convergence and a retained-output disparity cap. Relative
  geometry remains the default.
- Raw depth schema v3 camera/focal persistence and independent restartable
  relative and metric Stage-3 stores.
- Third-party notices for Microsoft MoGe and its bundled/derived DINOv2 code.
- Optional experimental VDPP depth-only temporal stabilization for every depth
  backend in relative geometry mode, with fixed shot-aware 32/4 inference,
  bounded memory, content-addressed artifacts, and shot-atomic resume.
- Web and CLI controls, lazy integrity-checked checkpoint download, cache-only CPU
  rendering, and a deterministic quality-gate harness.

### Fixed

- Metric-camera rendering no longer treats MoGe's depth-confidence mask as
  image transparency. Pixels without trusted metric depth are retained as an
  infinite-background layer instead of becoming black cutouts.
- Metric jobs that keep intermediate files now persist the same viewable
  8-bit depth visualizations used by the live Web preview alongside the metric
  geometry payloads.

The [MoGe-2 release-evidence checklist](docs/release/moge2-release-checklist.md)
defines the separate three-variant evidence process that will be added with the
release tooling. This changelog entry does not claim that real-model evidence
has been executed.

### Changed
- Processing settings schema is now v4. Earlier jobs, including MoGe-era v3
  jobs, migrate to VDPP `off`; current jobs preserve omission separately from
  an explicit resume override.
- Source distributions explicitly exclude downloaded model and runtime artifacts.

## [0.9.2] - 2026-01-19

### Fixed
- **Critical bug fix**: Added null checks for `progress_tracker` in depth_processor.py
  - Fixes crash when using CLI without web interface (#14)
  - Error: `'NoneType' object has no attribute 'update_progress'`
  - Added guards at lines 111 and 399 in `src/depth_surge_3d/processing/frames/depth_processor.py`
  - All 770 tests passing

### Added
- **Python version pinning**: Added `.python-version` file pinned to Python 3.12
  - Improves development environment consistency
  - Recommended by contributor for uv project stability
- **Windows test script**: Added `test.ps1` for PowerShell users
  - Matches functionality of `test.sh` for cross-platform consistency
  - Verifies Python dependencies, CUDA, model files, and FFmpeg

### Changed
- **UI reorganization**: Improved logical grouping of settings in web UI
  - **Step 7 (VR Assembly)**: Now contains VR Format, Headset Preset, and VR Resolution
  - **Step 8 (Video Encoding & Output)**: Focused on encoding, audio, and file management
  - Clearer separation: assembly settings vs. output encoding settings
- **Dependency configuration**: Merged PR #13 from @danrossi
  - Added `depth-anything-3` package with git source configuration to `pyproject.toml`
  - Enables proper installation via uv package manager
  - Resolves installation issues for Python 3.9-3.12 users (#11)
- **Script colors**: Updated all user-facing scripts to use exact CSS colors
  - Lime green: `#39ff14` (RGB 57, 255, 20) - matches `--accent-lime`
  - Cyan: `#00d9ff` (RGB 0, 217, 255) - matches info/cyan
  - Consistent branding across CLI scripts and web UI

### Documentation
- Reorganized project structure
  - Moved `TODO.md` to `docs/` directory for better organization
  - Archived `codex-review.md` to `docs/archive/`
  - Updated contributor documentation links
- Rewrote CONTRIBUTING.md with separate sections for human vs AI contributors
  - Human contributors: Relaxed requirements, focus on ideas over perfection
  - AI contributors: Strict requirements, points to CLAUDE.md
  - Acknowledges AI may refactor human contributions later
- Updated example_settings.json to v0.9.2 with current settings

### Contributors
- Special thanks to @danrossi for identifying and helping resolve installation issues

---

## Previous Releases

For changelog entries prior to v0.9.2, see [docs/archive/CHANGELOG.md](docs/archive/CHANGELOG.md)

[Unreleased]: https://github.com/Tok/depth-surge-3d/compare/v0.9.2...HEAD
[0.9.2]: https://github.com/Tok/depth-surge-3d/compare/v0.9.1...v0.9.2
