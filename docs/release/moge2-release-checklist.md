# MoGe-2 Three-Variant Release Checklist

This is the non-CI release procedure for the experimental MoGe-2 metric-camera path. Run it on the approved NVIDIA release machine. CI runs only fake unit tests and never downloads model weights or release media.

## 1. Approve and Freeze the Corpus

- [ ] Capture `indoor-near.mp4` with a stable near foreground object, visible occlusion edges, and useful background depth.
- [ ] Capture `outdoor-far.mp4` with distant structure, a broad depth range, and enough texture for visual inspection.
- [ ] Capture `scene-cut.mp4` with at least one clear cut between materially different scenes; do not edit frames after approval.
- [ ] Keep all three clips at their native frame order, dimensions, cadence, and encoding after approval.
- [ ] Select one fixed image with nontrivial near/far structure, a supported image format, and no post-approval edits.
- [ ] Confirm that every clip is square-SAR (`1:1`). Missing or `N/A` SAR is normalized to `1:1`; malformed, zero, overflowing, or explicit non-square SAR is rejected.
- [ ] Review clips for baked-in letterboxing. If bars are present, record them in the release ticket and keep each ROI out of the bars. Square SAR does not remove letterbox bias.

Create lowercase SHA-256 values from the final bytes:

```powershell
(Get-FileHash D:\moge2-corpus\fixed-image.png -Algorithm SHA256).Hash.ToLowerInvariant()
(Get-FileHash D:\moge2-corpus\indoor-near.mp4 -Algorithm SHA256).Hash.ToLowerInvariant()
(Get-FileHash D:\moge2-corpus\outdoor-far.mp4 -Algorithm SHA256).Hash.ToLowerInvariant()
(Get-FileHash D:\moge2-corpus\scene-cut.mp4 -Algorithm SHA256).Hash.ToLowerInvariant()
```

- [ ] Put those exact lowercase hashes and paths in `D:\moge2-corpus\corpus.json` using the schema documented in the implementation plan.
- [ ] Store the approved `corpus.json` immutably with the release ticket or controlled corpus location.
- [ ] Re-run `Get-FileHash` immediately before the gate and compare every value byte-for-byte with `corpus.json`.

## 2. Select Static ROIs

- [ ] Select one source-coordinate `[x, y, width, height]` ROI for each clip.
- [ ] Keep every ROI wholly inside the source frame and the retained center crop; exclude letterbox bars.
- [ ] Use fixed scene content that is representative and expected to contain valid positive metric depth.
- [ ] Avoid moving frame borders, subtitles, watermarks, transition edges, and regions dominated by holes.
- [ ] Do not change an ROI between variants, geometry modes, retries, or reports.
- [ ] Record the approved ROIs in the immutable corpus configuration and release ticket.

## 3. Prepare the Release Machine

- [ ] Record the project Git commit and whether the tree is dirty.
- [ ] Record the OS, Python, PyTorch, CUDA runtime, NVIDIA driver, and exact GPU identity.
- [ ] Confirm sufficient free disk for extracted frames, raw schema-v3 data, both Stage-3 representations, stereo frames, 18 videos, three NPZ files, and temporary sibling files.
- [ ] Confirm sufficient VRAM for the pinned Large variant at depth resolution 1080. The runner does not fall back to a smaller model, another device, or another precision policy.
- [ ] Close unrelated GPU workloads and confirm `nvidia-smi` shows the intended device.
- [ ] Confirm FFmpeg and ffprobe are available on `PATH`.

The approved gate uses CUDA. An exploratory `--device cpu` run records `peak_vram_bytes` as numeric zero because CUDA allocation is unavailable; it is not a substitute for the CUDA release gate.

Install the pinned optional dependency:

```powershell
uv sync --extra moge2
```

## 4. Run the Gate

Use the approved corpus and a new, empty output directory:

```powershell
uv run --extra moge2 python scripts/verify_moge2_release.py `
  --corpus-config D:\moge2-corpus\corpus.json `
  --output-dir artifacts\moge2-release\2026-08-16 `
  --device cuda `
  --depth-resolution 1080
```

- [ ] Confirm the runner processes exactly `vits`, `vitb`, and `vitl` in that order.
- [ ] Confirm each model is loaded once, used for the fixed image and all three clips/two modes, then unloaded before the next model.
- [ ] Confirm the command exits zero and `report.json` has top-level `status: "complete"` with an empty failures list.

## 5. Verify Machine Evidence

- [ ] Confirm all 18 A/B videos exist: three clips times two modes times three variants.
- [ ] Confirm all three `fixed-image-depth.npz` files exist and each contains only `depth.npy`, `valid.npy`, and `focal_x_normalized.npy`.
- [ ] Confirm `report.json` records every input hash, all 21 output hashes, the three exact repositories/revisions, MoGe source commit `925b8ed835a7a9cdb7578ba15c658a0afc969030`, adapter level `9`, and depth resolution `1080`.
- [ ] Recompute SHA-256 for every reported video and NPZ and compare it with the complete JSON hash set.
- [ ] Confirm every relative/metric pair used identical output, crop, packing, and distortion settings and that each clip reports one inference pass reused by both modes.
- [ ] Confirm `report.md` contains the same machine, input, setting, measurement, output, and failure identities as `report.json`.

## 6. Human Inspection and Sign-Off

Review every A/B pair for every variant and clip. Complete every unchecked row in `report.md`:

- [ ] Edge tearing
- [ ] Foreground sign
- [ ] Scale pumping
- [ ] Focal breathing
- [ ] Convergence placement
- [ ] Viewing discomfort

- [ ] Record reviewer names, review date, every disposition, and any linked defect IDs in the release ticket.
- [ ] Record the artifact location and the SHA-256 of `report.json` in the release ticket.
- [ ] Preserve the approved corpus configuration, report files, three NPZs, and 18 videos together under the ticket's retention policy.

The reported timing, VRAM, focal, ROI depth, disparity, hole, and clamp values are structural observations, not portable quality, calibration, comfort, or temporal-stability thresholds. A successful structural gate does not establish physical calibration, better quality, viewing comfort, or temporal stability. Removing the Experimental label requires a separate compatibility decision.
