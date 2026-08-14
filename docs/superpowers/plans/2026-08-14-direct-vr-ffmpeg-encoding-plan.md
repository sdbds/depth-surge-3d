# Direct VR FFmpeg Encoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in final-encoding path in which FFmpeg reads the validated left and right PNG sequences, resizes and stacks them in one filter graph, and atomically publishes the final video without creating `99_vr_frames`.

**Architecture:** `ProcessingOrchestrator` owns the strategy branch. `VRFrameAssembler` remains the single source-selection authority and exposes its current resolver. `VideoEncoder` owns canonical image2 validation, direct FFmpeg command construction, merged-stream progress handling, and atomic output publication. The legacy assembled-PNG path remains the default and is kept unchanged.

**Tech Stack:** Python 3, `pathlib`, `subprocess`, FFmpeg/ffprobe, OpenCV, NumPy, Bootstrap/vanilla JavaScript, pytest.

## Global Constraints

- `direct_vr_encode` is a strict boolean and defaults to `false` without a settings-schema-version bump.
- The option is independent of `keep_intermediates`; direct mode never creates an active `vr_frames` directory.
- Never delete a pre-existing `99_vr_frames` merely because direct mode is selected.
- Keep the disabled branch's method calls, FFmpeg command, progress, previews, manifests, and recovery behavior unchanged.
- Resolve cropped versus upscaled eye sources through `VRFrameAssembler`, not in the orchestrator or encoder.
- Validate canonical, matching, gap-free `frame_%06d.png` sequences before starting FFmpeg. Read only the first IHDR from each eye.
- If either eye differs from the requested per-eye dimensions, scale both with `bicubic+accurate_rnd`; otherwise omit scale filters.
- Reuse `_resolve_output_fps`, `_build_audio_input_args`, `_build_encoder_cmd`, and `generate_output_filename` so FPS, trim, encoder, pixel format, and quality arguments do not drift.
- Run direct FFmpeg with one merged output stream, incremental `frame=` progress parsing, a 50-line bounded diagnostic tail, final `wait()`, and terminate/kill cleanup on reader errors.
- Encode to a sibling `.mp4` temporary path. Replace the published output only after return code zero and a non-empty temporary file.
- A failed encode leaves all eye stages, manifests, and any previous valid final video intact.
- Direct mode emits progress but no assembled-frame preview or VR-stage metadata.
- Do not add this execution-only flag to frame-stage identity settings or `_VR_SETTING_KEYS`.
- Preserve unrelated worktree changes and use `apply_patch` for manual edits.

## File And Interface Map

- `src/depth_surge_3d/core/constants.py`: add the persisted default.
- `src/depth_surge_3d/core/settings.py`: add strict boolean validation without changing `PROCESSING_SETTINGS_SCHEMA_VERSION`.
- `src/depth_surge_3d/io/operations.py`: add `omitted_intermediates: Collection[str] | None = None` to directory setup.
- `src/depth_surge_3d/processing/frames/vr_assembler.py`: expose `resolve_vr_source_files(...)` and consume it from the legacy assembler.
- `src/depth_surge_3d/processing/video/video_encoder.py`: add `_DirectStereoSequence`, validation, command construction, merged progress execution, and `create_video_from_stereo_sequences(...)`.
- `src/depth_surge_3d/processing/orchestration/pipeline_orchestrator.py`: omit the VR directory and branch Steps 7-8 only when enabled.
- `templates/index.html`: add the unchecked checkbox to request, persistence, reset, and autosave flows.
- `tests/unit/`: lock every contract above with focused tests.
- `tests/integration/test_direct_vr_ffmpeg_encoding.py`: exercise real FFmpeg layouts, audio, output metadata, and resize diagnostics.

---

### Task 1: Strict Setting Contract

**Files:**
- Modify: `src/depth_surge_3d/core/constants.py`
- Modify: `src/depth_surge_3d/core/settings.py`
- Modify: `tests/unit/test_settings.py`

**Interfaces:**
- Produces: `DEFAULT_SETTINGS["direct_vr_encode"] is False`.
- Preserves: `PROCESSING_SETTINGS_SCHEMA_VERSION == 2`.
- Produces: explicit and legacy-disk normalization through the existing `validate_settings` API.

- [ ] **Step 1: Write failing default, strictness, and legacy-normalization tests**

```python
def test_direct_vr_encode_defaults_off() -> None:
    assert DEFAULT_SETTINGS["direct_vr_encode"] is False
    assert validate_settings({}, source="legacy_disk")["direct_vr_encode"] is False


@pytest.mark.parametrize("value", [0, 1, "false", "true", None])
def test_direct_vr_encode_rejects_non_booleans(value: object) -> None:
    with pytest.raises(ValueError, match="direct_vr_encode"):
        validate_settings({"direct_vr_encode": value}, source="explicit")


@pytest.mark.parametrize("value", [False, True])
def test_direct_vr_encode_accepts_booleans(value: bool) -> None:
    assert validate_settings(
        {"direct_vr_encode": value}, source="explicit"
    )["direct_vr_encode"] is value


def test_direct_vr_encode_does_not_bump_settings_schema() -> None:
    assert PROCESSING_SETTINGS_SCHEMA_VERSION == 2
```

- [ ] **Step 2: Run the focused tests and verify the missing setting fails**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_settings.py -q`

Expected: failures show that the default and known boolean field do not yet exist.

- [ ] **Step 3: Add the default and strict boolean registration**

```python
# constants.py, beside preserve_audio/keep_intermediates
"direct_vr_encode": False,

# settings.py
_EXISTING_BOOLEAN_SETTINGS = {
    "preserve_audio",
    "keep_intermediates",
    "direct_vr_encode",
    "apply_distortion",
    "experimental_frame_interpolation",
}
```

Do not change `PROCESSING_SETTINGS_SCHEMA_VERSION`.

- [ ] **Step 4: Run the setting tests**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_settings.py -q`

Expected: all tests pass.

- [ ] **Step 5: Checkpoint the focused diff**

Run: `git diff --check -- src/depth_surge_3d/core/constants.py src/depth_surge_3d/core/settings.py tests/unit/test_settings.py`

Clean-worktree commit: `git commit -m "feat: add direct VR encoding setting"`

### Task 2: Omit Only The Active VR Working Directory

**Files:**
- Modify: `src/depth_surge_3d/io/operations.py`
- Modify: `tests/unit/test_io_operations.py`

**Interfaces:**
- Changes: `create_output_directories(base_path, keep_intermediates=True, omitted_intermediates=None) -> dict[str, Path]`.
- Preserves: all current two-argument and default calls.
- Guarantees: omitted keys are absent from the returned map and are neither created nor removed.

- [ ] **Step 1: Write failing real-filesystem omission tests**

```python
def test_create_output_directories_can_omit_vr_stage(tmp_path):
    result = create_output_directories(
        tmp_path,
        keep_intermediates=True,
        omitted_intermediates={"vr_frames"},
    )

    assert "vr_frames" not in result
    assert not (tmp_path / INTERMEDIATE_DIRS["vr_frames"]).exists()
    assert result["left_cropped"].is_dir()
    assert result["right_cropped"].is_dir()


def test_omission_never_deletes_preexisting_vr_stage(tmp_path):
    vr_dir = tmp_path / INTERMEDIATE_DIRS["vr_frames"]
    vr_dir.mkdir(parents=True)
    marker = vr_dir / "frame_000001.png"
    marker.write_bytes(b"existing")

    result = create_output_directories(
        tmp_path,
        keep_intermediates=False,
        omitted_intermediates={"vr_frames"},
    )

    assert "vr_frames" not in result
    assert marker.read_bytes() == b"existing"
```

- [ ] **Step 2: Run the focused tests and verify the new keyword is rejected**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_io_operations.py -q`

Expected: the two new tests fail with an unexpected keyword argument.

- [ ] **Step 3: Implement key-based omission without deletion**

```python
from collections.abc import Collection


def create_output_directories(
    base_path: Path,
    keep_intermediates: bool = True,
    omitted_intermediates: Collection[str] | None = None,
) -> dict[str, Path]:
    omitted = frozenset(omitted_intermediates or ())
    directories = {"base": base_path}
    base_path.mkdir(parents=True, exist_ok=True)
    for dir_name, dir_path in INTERMEDIATE_DIRS.items():
        if dir_name in omitted:
            continue
        full_path = base_path / dir_path
        full_path.mkdir(exist_ok=True)
        directories[dir_name] = full_path
    return directories
```

Keep `keep_intermediates` in the signature because it is part of the current public API and still documents post-success retention.

- [ ] **Step 4: Run directory tests**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_io_operations.py -q`

Expected: all tests pass, including current retention tests.

- [ ] **Step 5: Checkpoint the focused diff**

Run: `git diff --check -- src/depth_surge_3d/io/operations.py tests/unit/test_io_operations.py`

Clean-worktree commit: `git commit -m "feat: allow omitted processing directories"`

### Task 3: Share VR Eye-Source Resolution

**Files:**
- Modify: `src/depth_surge_3d/processing/frames/vr_assembler.py`
- Modify: `tests/unit/test_vr_assembler.py`

**Interfaces:**
- Produces: `VRFrameAssembler.resolve_vr_source_files(directories, settings, total_frames) -> tuple[list[Path], list[Path]] | None`.
- Preserves: legacy source choice, sorting, completeness checks, and failure messages.
- Consumed by: `assemble_vr_frames` and the direct orchestrator branch.

- [ ] **Step 1: Write failing public-resolver tests**

```python
def _write_source_pair(left_dir: Path, right_dir: Path, index: int) -> None:
    image = np.full((4, 6, 3), index, dtype=np.uint8)
    assert cv2.imwrite(str(left_dir / f"frame_{index:06d}.png"), image)
    assert cv2.imwrite(str(right_dir / f"frame_{index:06d}.png"), image)


def test_resolve_vr_source_files_uses_cropped_sources(tmp_path):
    left = tmp_path / "left_cropped"
    right = tmp_path / "right_cropped"
    left.mkdir()
    right.mkdir()
    _write_source_pair(left, right, 1)

    result = VRFrameAssembler().resolve_vr_source_files(
        {"left_cropped": left, "right_cropped": right},
        {"upscale_model": "none"},
        total_frames=1,
    )

    assert result == ([left / "frame_000001.png"], [right / "frame_000001.png"])


def test_resolve_vr_source_files_requires_upscaled_sources(tmp_path):
    cropped_left = tmp_path / "left_cropped"
    cropped_right = tmp_path / "right_cropped"
    cropped_left.mkdir()
    cropped_right.mkdir()
    _write_source_pair(cropped_left, cropped_right, 1)

    assert VRFrameAssembler().resolve_vr_source_files(
        {"left_cropped": cropped_left, "right_cropped": cropped_right},
        {"upscale_model": "x2"},
        total_frames=1,
    ) is None
```

Retain or add the existing unequal-count, total-count, and stem-mismatch cases against the public method.

- [ ] **Step 2: Run VR tests and verify the public method is missing**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_vr_assembler.py -q`

Expected: new tests fail with `AttributeError`.

- [ ] **Step 3: Rename the private resolver and update the legacy caller**

```python
def resolve_vr_source_files(
    self,
    directories: dict[str, Path],
    settings: dict[str, Any],
    total_frames: int,
) -> tuple[list[Path], list[Path]] | None:
    source_dirs = self._get_vr_assembly_source_dirs(directories, settings)
    if source_dirs is None:
        return None
    left_files = sorted(source_dirs[0].glob("*.png"))
    right_files = sorted(source_dirs[1].glob("*.png"))
    if not self._source_frame_manifest_is_complete(left_files, right_files, total_frames):
        print("Error: VR source frame manifest is incomplete")
        return None
    return left_files, right_files
```

Change only `assemble_vr_frames` from `_vr_source_files(...)` to `resolve_vr_source_files(...)`; do not tighten its accepted filenames.

- [ ] **Step 4: Run VR regression tests**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_vr_assembler.py -q`

Expected: all tests pass.

- [ ] **Step 5: Checkpoint the focused diff**

Run: `git diff --check -- src/depth_surge_3d/processing/frames/vr_assembler.py tests/unit/test_vr_assembler.py`

Clean-worktree commit: `git commit -m "refactor: expose VR source resolution"`

### Task 4: Validate Image2 Sequences And Build The Direct Command

**Files:**
- Modify: `src/depth_surge_3d/processing/video/video_encoder.py`
- Modify: `tests/unit/test_video_encoder.py`

**Interfaces:**
- Produces private immutable `_DirectStereoSequence` with numerically sorted files, patterns, start number, count, and first-eye headers.
- Produces: `_validate_direct_stereo_sequence(left_files, right_files, total_frames) -> _DirectStereoSequence`; invalid input raises `ValueError` before process launch.
- Produces: `_build_direct_stereo_command(sequence, temporary_output, original_video, settings) -> list[str]`.
- Preserves: existing `_resolve_output_fps`, `_build_audio_input_args`, and `_build_encoder_cmd` ownership.

- [ ] **Step 1: Add deterministic PNG-sequence helpers and failing validation tests**

```python
def _write_eye_sequence(
    directory: Path,
    indices: list[int],
    *,
    shape: tuple[int, int] = (4, 6),
) -> list[Path]:
    directory.mkdir()
    files = []
    for index in indices:
        path = directory / f"frame_{index:06d}.png"
        image = np.full((*shape, 3), index % 255, dtype=np.uint8)
        assert cv2.imwrite(str(path), image)
        files.append(path)
    return files


@pytest.mark.parametrize("start", [0, 1, 1_000_000])
def test_validate_direct_sequence_accepts_canonical_consecutive_names(tmp_path, start):
    left = _write_eye_sequence(tmp_path / "left", [start, start + 1])
    right = _write_eye_sequence(tmp_path / "right", [start, start + 1])

    sequence = VideoEncoder()._validate_direct_stereo_sequence(left, right, 2)

    assert sequence.start_number == start
    assert sequence.frame_count == 2
    assert sequence.left_pattern == tmp_path / "left" / "frame_%06d.png"
    assert sequence.right_pattern == tmp_path / "right" / "frame_%06d.png"


@pytest.mark.parametrize(
    "left_names,right_names,total",
    [
        ([], [], 0),
        (["frame_000001.png"], ["frame_000002.png"], 1),
        (["frame_000001.png", "frame_000003.png"], ["frame_000001.png", "frame_000003.png"], 2),
        (["frame_1.png"], ["frame_1.png"], 1),
        (["frame_0000001.png"], ["frame_0000001.png"], 1),
    ],
)
def test_validate_direct_sequence_rejects_invalid_manifests(
    tmp_path, left_names, right_names, total
):
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()
    left_files = []
    right_files = []
    for directory, names, destination in (
        (left_dir, left_names, left_files),
        (right_dir, right_names, right_files),
    ):
        for name in names:
            path = directory / name
            assert cv2.imwrite(str(path), np.zeros((4, 6, 3), np.uint8))
            destination.append(path)

    with pytest.raises(ValueError):
        VideoEncoder()._validate_direct_stereo_sequence(
            left_files, right_files, total
        )
```

Add separate cases for multiple parent directories, a positive `total_frames` mismatch, and an unreadable first PNG. Patch `read_png_header` and assert exactly two calls for a valid multi-frame sequence.

- [ ] **Step 2: Run focused validation tests and verify the APIs are missing**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_video_encoder.py -q`

Expected: failures identify the missing validator/dataclass.

- [ ] **Step 3: Implement numeric canonical validation and first-header reads**

```python
@dataclass(frozen=True)
class _DirectStereoSequence:
    left_files: tuple[Path, ...]
    right_files: tuple[Path, ...]
    left_pattern: Path
    right_pattern: Path
    start_number: int
    frame_count: int
    left_header: PngHeader
    right_header: PngHeader


_DIRECT_FRAME_NAME = re.compile(r"^frame_(\d{6,})\.png$")


@staticmethod
def _direct_frame_index(path: Path) -> int:
    match = _DIRECT_FRAME_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Noncanonical direct frame name: {path.name}")
    index = int(match.group(1))
    if path.name != f"frame_{index:06d}.png":
        raise ValueError(f"Noncanonical direct frame padding: {path.name}")
    return index
```

Numerically sort each eye, require a single parent per eye, identical names, an exact positive expected count when supplied, and `next_index == previous_index + 1`. Read only each sorted list's first IHDR and reject `None`.

- [ ] **Step 4: Add failing command-construction tests**

```python
def _direct_settings(**overrides):
    settings = {
        "target_fps": "24000/1001",
        "vr_format": "side_by_side",
        "vr_resolution": "custom",
        "per_eye_width": 6,
        "per_eye_height": 4,
        "preserve_audio": False,
        "video_encoder": "libx264",
    }
    settings.update(overrides)
    return settings


def test_direct_command_omits_scale_for_exact_size_and_uses_hstack(tmp_path):
    left = _write_eye_sequence(tmp_path / "left", [1, 2])
    right = _write_eye_sequence(tmp_path / "right", [1, 2])
    encoder = VideoEncoder()
    sequence = encoder._validate_direct_stereo_sequence(left, right, 2)
    command = encoder._build_direct_stereo_command(
        sequence,
        tmp_path / ".output.direct.tmp.mp4",
        "source.mp4",
        _direct_settings(),
    )

    graph = command[command.index("-filter_complex") + 1]
    assert graph == "[0:v][1:v]hstack=inputs=2:shortest=1[vr]"
    assert "scale=" not in graph
    assert command.count("-framerate") == 2
    assert command.count("24000/1001") == 2
    assert command.count("-start_number") == 2
    starts = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "-start_number"
    ]
    assert starts == ["1", "1"]
    assert command[command.index("-frames:v") + 1] == "2"
    assert command[command.index("-map") + 1] == "[vr]"


def test_direct_command_scales_both_eyes_and_uses_vstack(tmp_path):
    left = _write_eye_sequence(tmp_path / "left", [0], shape=(6, 8))
    right = _write_eye_sequence(tmp_path / "right", [0], shape=(5, 8))
    encoder = VideoEncoder()
    sequence = encoder._validate_direct_stereo_sequence(left, right, 1)
    command = encoder._build_direct_stereo_command(
        sequence,
        tmp_path / ".output.direct.tmp.mp4",
        "source.mp4",
        _direct_settings(vr_format="over_under", per_eye_width=8, per_eye_height=6),
    )

    graph = command[command.index("-filter_complex") + 1]
    assert graph.count("scale=8:6:flags=bicubic+accurate_rnd") == 2
    assert "[left][right]vstack=inputs=2:shortest=1[vr]" in graph
```

Add exact assertions that software encoding ends with `-c:v libx264 -pix_fmt yuv420p -crf 18 -preset medium <temp>`. Patch NVENC availability and assert the existing NVENC argument list is reused unchanged.

- [ ] **Step 5: Add failing explicit-audio-index and invalid-setting tests**

```python
def test_direct_command_maps_audio_from_third_input(tmp_path):
    left = _write_eye_sequence(tmp_path / "left", [1])
    right = _write_eye_sequence(tmp_path / "right", [1])
    (tmp_path / "original_audio.flac").write_bytes(b"audio")
    encoder = VideoEncoder()
    sequence = encoder._validate_direct_stereo_sequence(left, right, 1)

    command = encoder._build_direct_stereo_command(
        sequence,
        tmp_path / ".output.direct.tmp.mp4",
        "source.mp4",
        _direct_settings(preserve_audio=True),
    )

    assert ["-map", "2:a:0?", "-c:a", "aac", "-shortest"] == command[
        command.index("-map", command.index("-map") + 1) :
        command.index("-map", command.index("-map") + 1) + 5
    ]
```

Also assert non-positive dimensions and unknown VR formats raise `ValueError`, and that audio arguments are entirely absent when `preserve_audio` is false.

- [ ] **Step 6: Implement the direct filter graph and command builder**

Construct both image2 inputs with the same exact FPS and explicit start. Append the optional audio input next, so every input declaration precedes the output filter/mapping region. If either header is not the target dimensions, create two scale chains. Then append:

```python
if settings.get("preserve_audio", True):
    command.extend(self._build_audio_input_args(audio_source, settings))
command.extend(["-filter_complex", filter_graph, "-map", "[vr]"])
if settings.get("preserve_audio", True):
    command.extend(["-map", "2:a:0?", "-c:a", "aac", "-shortest"])
command.extend(
    ["-frames:v", str(sequence.frame_count), "-progress", "pipe:1"]
)
encoder_args, _ = self._build_encoder_cmd(
    settings.get("video_encoder", "auto"), temporary_output
)
command.extend(encoder_args)
```

Place `-loglevel error -nostats` in the command's global option region. Keep output-path ownership in `_build_encoder_cmd`.

- [ ] **Step 7: Run video-encoder unit tests**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_video_encoder.py -q`

Expected: validation and command tests pass; no existing `create_video` test changes.

- [ ] **Step 8: Checkpoint the focused diff**

Run: `git diff --check -- src/depth_surge_3d/processing/video/video_encoder.py tests/unit/test_video_encoder.py`

Clean-worktree commit: `git commit -m "feat: build direct stereo FFmpeg commands"`

### Task 5: Stream Progress And Publish Atomically

**Files:**
- Modify: `src/depth_surge_3d/processing/video/video_encoder.py`
- Modify: `tests/unit/test_video_encoder.py`

**Interfaces:**
- Produces: `_run_ffmpeg_with_progress(command, total_frames, progress_tracker) -> tuple[int, tuple[str, ...]]`.
- Produces: `create_video_from_stereo_sequences(left_files, right_files, output_dir, original_video, settings, *, total_frames, progress_tracker=None) -> bool`.
- Guarantees: one merged pipe, bounded diagnostics, deterministic reaping, and atomic publication.

- [ ] **Step 1: Add a controllable fake process and failing progress tests**

```python
class _FakeProcess:
    def __init__(self, output: str, returncode: int = 0):
        self.stdout = io.StringIO(output)
        self.returncode = returncode
        self.waited = False
        self.terminated = False
        self.killed = False

    def wait(self, timeout=None):
        self.waited = True
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def communicate(self, timeout=None):
        self.waited = True
        return ("", None)


def test_direct_runner_merges_stderr_tracks_frames_and_waits():
    process = _FakeProcess("frame=1\nnoise\nframe=2\n")
    tracker = Mock()
    with patch(
        "src.depth_surge_3d.processing.video.video_encoder.subprocess.Popen",
        return_value=process,
    ) as popen:
        returncode, diagnostics = VideoEncoder()._run_ffmpeg_with_progress(
            ["ffmpeg"], 2, tracker
        )

    assert returncode == 0
    assert diagnostics == ("noise",)
    assert process.waited is True
    assert [call.kwargs["frame_num"] for call in tracker.update_progress.call_args_list] == [1, 2]
    assert popen.call_args.kwargs == {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,
    }


def test_direct_runner_retains_only_last_fifty_diagnostic_lines():
    process = _FakeProcess("".join(f"line-{index}\n" for index in range(75)), 1)
    with patch(
        "src.depth_surge_3d.processing.video.video_encoder.subprocess.Popen",
        return_value=process,
    ):
        returncode, diagnostics = VideoEncoder()._run_ffmpeg_with_progress(
            ["ffmpeg"], 1, None
        )

    assert returncode == 1
    assert len(diagnostics) == 50
    assert diagnostics[0] == "line-25"
    assert diagnostics[-1] == "line-74"
```

- [ ] **Step 2: Add failing callback-isolation and reader-error cleanup tests**

Use a tracker whose `update_progress` raises and assert the runner still reaches `wait()` and returns FFmpeg's code. Use a stream iterator that raises after one line and assert the runner terminates, drains with bounded `communicate(timeout=5)`, and reaps the process. Add a `subprocess.TimeoutExpired` fake to assert `kill()` and a second `communicate()` occur.

- [ ] **Step 3: Run the focused runner tests and verify the method is missing**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_video_encoder.py -q`

Expected: progress-runner tests fail with `AttributeError`.

- [ ] **Step 4: Implement one-stream progress handling**

```python
diagnostics: deque[str] = deque(maxlen=50)
process = subprocess.Popen(
    command,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding="utf-8",
    errors="replace",
    bufsize=1,
)
try:
    if process.stdout is None:
        raise OSError("FFmpeg progress stream is unavailable")
    for raw_line in process.stdout:
        line = raw_line.strip()
        if line.startswith("frame="):
            try:
                frame = min(max(int(line.partition("=")[2]), 0), total_frames)
            except ValueError:
                diagnostics.append(line)
                continue
            try:
                if progress_tracker is not None:
                    progress_tracker.update_progress(
                        f"Encoding VR frame {frame}/{total_frames}",
                        phase="video_encoding",
                        frame_num=frame,
                        step_name="Direct VR Encoding",
                        step_progress=frame,
                        step_total=total_frames,
                    )
            except Exception as error:
                print(f"Warning: Direct encoding progress update failed: {error}")
        elif line:
            diagnostics.append(line)
    return process.wait(), tuple(diagnostics)
except Exception:
    process.terminate()
    try:
        remaining, _ = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        remaining, _ = process.communicate()
    for line in (remaining or "").splitlines():
        if line.strip():
            diagnostics.append(line.strip())
    raise
```

The public method catches launch/reader exceptions and returns `False`; the helper itself raises only so unit tests can verify cleanup precisely.

- [ ] **Step 5: Add failing atomic-success and failure-preservation tests**

```python
def test_direct_create_video_atomically_replaces_final_output(tmp_path):
    left = _write_eye_sequence(tmp_path / "left", [1])
    right = _write_eye_sequence(tmp_path / "right", [1])
    settings = _direct_settings()
    final = tmp_path / generate_output_filename(
        "source.mp4", settings["vr_format"], settings["vr_resolution"]
    )
    final.write_bytes(b"old-valid-video")

    def launch(command, **_kwargs):
        Path(command[-1]).write_bytes(b"new-valid-video")
        return _FakeProcess("frame=1\nprogress=end\n")

    with (
        patch(
            "src.depth_surge_3d.processing.video.video_encoder.verify_ffmpeg_installation",
            return_value=True,
        ),
        patch(
            "src.depth_surge_3d.processing.video.video_encoder.subprocess.Popen",
            side_effect=launch,
        ),
    ):
        assert VideoEncoder().create_video_from_stereo_sequences(
            left,
            right,
            tmp_path,
            "source.mp4",
            settings,
            total_frames=1,
        )

    assert final.read_bytes() == b"new-valid-video"
    assert not (tmp_path / f".{final.stem}.direct.tmp.mp4").exists()


def test_direct_create_video_failure_preserves_old_final_and_removes_temp(tmp_path):
    left = _write_eye_sequence(tmp_path / "left", [1])
    right = _write_eye_sequence(tmp_path / "right", [1])
    settings = _direct_settings()
    final = tmp_path / generate_output_filename(
        "source.mp4", settings["vr_format"], settings["vr_resolution"]
    )
    final.write_bytes(b"old-valid-video")

    def launch(command, **_kwargs):
        Path(command[-1]).write_bytes(b"partial")
        return _FakeProcess("fatal encode error\n", returncode=1)

    with (
        patch(
            "src.depth_surge_3d.processing.video.video_encoder.verify_ffmpeg_installation",
            return_value=True,
        ),
        patch(
            "src.depth_surge_3d.processing.video.video_encoder.subprocess.Popen",
            side_effect=launch,
        ),
    ):
        assert not VideoEncoder().create_video_from_stereo_sequences(
            left,
            right,
            tmp_path,
            "source.mp4",
            settings,
            total_frames=1,
        )

    assert final.read_bytes() == b"old-valid-video"
    assert not (tmp_path / f".{final.stem}.direct.tmp.mp4").exists()
```

Add cases for validation failure before `Popen`, launch exception, zero-byte temporary output, and `Path.replace` failure. Capture output and assert the diagnostic tail is reported on nonzero exit.

- [ ] **Step 6: Implement the public atomic method**

```python
def create_video_from_stereo_sequences(
    self,
    left_files: Sequence[Path],
    right_files: Sequence[Path],
    output_dir: Path,
    original_video: str,
    settings: dict[str, Any],
    *,
    total_frames: int,
    progress_tracker=None,
) -> bool:
    if not verify_ffmpeg_installation():
        print("Error: FFmpeg not found. Cannot create output video.")
        return False
    output_filename = generate_output_filename(
        Path(original_video).name,
        settings["vr_format"],
        settings["vr_resolution"],
    )
    output_path = output_dir / output_filename
    temporary = output_path.with_name(f".{output_path.stem}.direct.tmp.mp4")
    temporary.unlink(missing_ok=True)
    try:
        sequence = self._validate_direct_stereo_sequence(
            left_files, right_files, total_frames
        )
        command = self._build_direct_stereo_command(
            sequence, temporary, original_video, settings
        )
        returncode, diagnostics = self._run_ffmpeg_with_progress(
            command, sequence.frame_count, progress_tracker
        )
        if returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
            self._print_direct_ffmpeg_failure(returncode, diagnostics)
            return False
        temporary.replace(output_path)
        return True
    except Exception as error:
        print(f"Error creating direct VR output video: {error}")
        return False
    finally:
        temporary.unlink(missing_ok=True)
```

- [ ] **Step 7: Run all video-encoder tests**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_video_encoder.py -q`

Expected: all existing and direct tests pass.

- [ ] **Step 8: Checkpoint the focused diff**

Run: `git diff --check -- src/depth_surge_3d/processing/video/video_encoder.py tests/unit/test_video_encoder.py`

Clean-worktree commit: `git commit -m "feat: stream and publish direct VR encoding"`

### Task 6: Branch The Pipeline At The Orchestration Boundary

**Files:**
- Modify: `src/depth_surge_3d/processing/orchestration/pipeline_orchestrator.py`
- Modify: `tests/unit/test_pipeline_orchestrator.py`

**Interfaces:**
- Consumes: `settings["direct_vr_encode"]`, omitted directory setup, shared source resolution, and the direct encoder method.
- Preserves: the exact default Step 7 and Step 8 path.
- Guarantees: direct source failure prevents FFmpeg; direct encode failure enters existing failed finalization without cleanup.

- [ ] **Step 1: Add failing direct directory-setup test while preserving the old assertion**

```python
def test_setup_processing_omits_vr_directory_in_direct_mode(self):
    orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())
    with (
        patch(
            "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.create_output_directories",
            return_value={"base": Path("/output/dir")},
        ) as create_dirs,
        patch(
            "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.save_processing_settings"
        ),
    ):
        orchestrator._setup_processing(
            "/input/video.mp4",
            "/output/dir",
            {"keep_intermediates": True, "direct_vr_encode": True},
            {"fps": 30, "frame_count": 1},
        )

    create_dirs.assert_called_once_with(
        Path("/output/dir"),
        True,
        omitted_intermediates={"vr_frames"},
    )
```

Keep `test_setup_processing` asserting the current two-argument call for settings where direct mode is absent/false.

- [ ] **Step 2: Add failing direct-branch orchestration tests**

Build the same successful stereo/distortion/crop mocks used by the current pipeline test. Then assert:

```python
vr_assembler.resolve_vr_source_files.return_value = ([left], [right])
video_encoder.create_video_from_stereo_sequences.return_value = True
settings["direct_vr_encode"] = True

assert orchestrator._execute_remaining_steps(
    directories,
    settings,
    [frame],
    [depth],
    24.0,
    "source.mp4",
    directories["base"],
    progress_tracker,
)

vr_assembler.assemble_vr_frames.assert_not_called()
video_encoder.create_video.assert_not_called()
video_encoder.create_video_from_stereo_sequences.assert_called_once_with(
    [left],
    [right],
    directories["base"],
    "source.mp4",
    settings,
    total_frames=1,
    progress_tracker=progress_tracker,
)
```

Add a resolver-returns-`None` case asserting neither encoder method runs. Add an encoder-false case asserting `_finalize_processing(False, ...)` runs and `progress_tracker.finish` keeps the current finalization behavior. Retain a default-mode assertion that the direct method is never called.

- [ ] **Step 3: Run orchestration tests and verify direct mode still assembles**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_pipeline_orchestrator.py -q`

Expected: direct setup and branch assertions fail.

- [ ] **Step 4: Implement the setup branch without perturbing default call shape**

```python
if settings.get("direct_vr_encode", False):
    directories = create_output_directories(
        output_path,
        settings["keep_intermediates"],
        omitted_intermediates={"vr_frames"},
    )
else:
    directories = create_output_directories(
        output_path, settings["keep_intermediates"]
    )
```

- [ ] **Step 5: Implement the Step 7-8 strategy branch**

```python
if settings.get("direct_vr_encode", False):
    source_files = self.vr_assembler.resolve_vr_source_files(
        directories, settings, num_frames
    )
    if source_files is None:
        return self._handle_step_error("Direct VR source validation failed")
    left_files, right_files = source_files
    print(step_complete("Step 7: Deferred VR assembly to direct FFmpeg encoding"))
    success = self.video_encoder.create_video_from_stereo_sequences(
        left_files,
        right_files,
        directories["base"],
        video_path,
        settings,
        total_frames=num_frames,
        progress_tracker=progress_tracker,
    )
else:
    # Keep the current assemble_vr_frames and create_video block unchanged here.
```

Do not call `complete_stage` or print a `vr_frames` save location in direct mode. Leave the existing common success message, `finish`, `_finalize_processing`, and cleanup flow after the branch.

- [ ] **Step 6: Run orchestration and neighboring pipeline tests**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_pipeline_orchestrator.py tests/unit/test_video_processor_new.py tests/unit/test_resume.py -q`

Expected: all tests pass and the default branch retains its current call sequence.

- [ ] **Step 7: Checkpoint the focused diff**

Run: `git diff --check -- src/depth_surge_3d/processing/orchestration/pipeline_orchestrator.py tests/unit/test_pipeline_orchestrator.py`

Clean-worktree commit: `git commit -m "feat: branch final VR encoding strategy"`

### Task 7: Add The Browser Control Without A Persistence Migration

**Files:**
- Modify: `templates/index.html`
- Create: `tests/unit/test_direct_vr_encode_template.py`

**Interfaces:**
- Produces: unchecked `#directVrEncode` checkbox labeled `Direct FFmpeg VR Encoding`.
- Produces request key: `direct_vr_encode`.
- Produces persisted/reset/autosaved browser key: `directVrEncode`.
- Preserves: browser `SETTINGS_VERSION = 3`; an older same-version save without the key leaves the HTML default unchecked.

- [ ] **Step 1: Write failing static template-contract tests**

```python
from pathlib import Path


TEMPLATE = Path("templates/index.html").read_text(encoding="utf-8")


def test_direct_vr_encode_checkbox_defaults_off():
    assert 'id="directVrEncode"' in TEMPLATE
    checkbox = TEMPLATE.split('id="directVrEncode"', 1)[1].split(">", 1)[0]
    assert "checked" not in checkbox
    assert "Direct FFmpeg VR Encoding" in TEMPLATE


def test_direct_vr_encode_is_sent_and_persisted():
    assert "direct_vr_encode: document.getElementById('directVrEncode').checked" in TEMPLATE
    assert "directVrEncode: document.getElementById('directVrEncode').checked" in TEMPLATE
    assert "directVrEncode: false" in TEMPLATE
    assert "'directVrEncode'" in TEMPLATE


def test_direct_vr_encode_does_not_force_browser_settings_reset():
    assert TEMPLATE.count("const SETTINGS_VERSION = 3") == 2
```

- [ ] **Step 2: Run the focused template test and verify every direct-control assertion fails**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_direct_vr_encode_template.py -q`

Expected: failures for the missing checkbox and mappings.

- [ ] **Step 3: Add the control near audio/intermediate settings**

```html
<div class="form-check mb-2">
    <input class="form-check-input" type="checkbox" id="directVrEncode">
    <label class="form-check-label" for="directVrEncode">
        Direct FFmpeg VR Encoding
        <i class="fas fa-info-circle" data-bs-toggle="tooltip"
           title="Advanced: encode directly from left/right frames and skip assembled VR-frame files."></i>
    </label>
</div>
```

Add `direct_vr_encode` to the process request, `directVrEncode` to `saveSettings`, `directVrEncode: false` to reset defaults, and `'directVrEncode'` to `setupAutoSave`. Do not increment either `SETTINGS_VERSION` constant.

- [ ] **Step 4: Run template, settings, and route validation tests**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_direct_vr_encode_template.py tests/unit/test_settings.py tests/unit/test_final_schema.py -q`

Expected: all tests pass.

- [ ] **Step 5: Inspect the control at desktop and mobile widths**

Start the existing application only if its route cannot be inspected as static HTML. At approximately 1440x900 and 390x844, verify the checkbox label and tooltip do not overlap neighboring controls and remain keyboard-focusable.

- [ ] **Step 6: Checkpoint the focused diff**

Run: `git diff --check -- templates/index.html tests/unit/test_direct_vr_encode_template.py`

Clean-worktree commit: `git commit -m "feat: expose direct VR encoding option"`

### Task 8: Real FFmpeg Layout, Audio, And Quality Evidence

**Files:**
- Create: `tests/integration/test_direct_vr_ffmpeg_encoding.py`

**Interfaces:**
- Exercises: public `create_video_from_stereo_sequences` with `libx264` and real PNGs.
- Verifies: dimensions, FPS, frame count, placement, optional audio, atomic final file, and absence of `99_vr_frames`.
- Reports: pre-compression OpenCV-versus-FFmpeg resize PSNR and SSIM, warning below 30 dB or 0.95 without failing solely on those floors.

- [ ] **Step 1: Write deterministic fixture and tool guards**

```python
pytestmark = pytest.mark.integration


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg and ffprobe are required")


def _write_stereo_fixture(root: Path, frame_count: int = 3):
    left_dir = root / "06_left_cropped"
    right_dir = root / "06_right_cropped"
    left_dir.mkdir()
    right_dir.mkdir()
    left_files = []
    right_files = []
    for index in range(1, frame_count + 1):
        left = np.zeros((48, 64, 3), np.uint8)
        left[:, :] = (20 * index, 40, 220)
        cv2.putText(left, "L", (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        right = np.zeros((48, 64, 3), np.uint8)
        right[:, :] = (220, 40, 20 * index)
        cv2.putText(right, "R", (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        left_path = left_dir / f"frame_{index:06d}.png"
        right_path = right_dir / f"frame_{index:06d}.png"
        assert cv2.imwrite(str(left_path), left)
        assert cv2.imwrite(str(right_path), right)
        left_files.append(left_path)
        right_files.append(right_path)
    return left_files, right_files
```

- [ ] **Step 2: Write SBS and OU end-to-end tests**

Parametrize `("side_by_side", 128, 48)` and `("over_under", 64, 96)`. Encode at `24000/1001`, probe JSON with `-count_frames`, and assert width, height, `avg_frame_rate`, and `nb_read_frames == 3`. Decode the first frame with OpenCV and compare each half's dominant colors and visible `L`/`R` placement. Assert `99_vr_frames` and its metadata were never created.

```python
assert encoder.create_video_from_stereo_sequences(
    left_files,
    right_files,
    tmp_path,
    str(tmp_path / "source.mp4"),
    settings,
    total_frames=3,
)
assert not (tmp_path / "99_vr_frames").exists()
```

- [ ] **Step 3: Write a real optional-audio mapping test**

Generate a one-second FLAC with:

```text
ffmpeg -y -f lavfi -i sine=frequency=440:duration=1 -c:a flac original_audio.flac
```

Encode the three-frame sequence with `preserve_audio=true`, probe streams, and assert exactly one video stream plus one AAC audio stream. Keep the fixture tiny enough for routine local execution.

- [ ] **Step 4: Write the calibrated pre-compression resize diagnostic**

Create a structured RGB fixture containing smooth gradients, sharp rectangles, deterministic fine texture, and text-like strokes. Produce the legacy reference with `cv2.resize(..., interpolation=cv2.INTER_CUBIC)`. Produce a lossless FFmpeg PNG with `scale=<w>:<h>:flags=bicubic+accurate_rnd`. Compute:

```python
def _psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    mse = np.mean((reference.astype(np.float64) - candidate.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 20 * math.log10(255.0 / math.sqrt(mse))


def _ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    values = []
    for channel in range(3):
        x = reference[:, :, channel].astype(np.float64)
        y = candidate[:, :, channel].astype(np.float64)
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
        sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
        numerator = (2 * mu_x * mu_y + 6.5025) * (2 * sigma_xy + 58.5225)
        denominator = (mu_x * mu_x + mu_y * mu_y + 6.5025) * (
            sigma_x + sigma_y + 58.5225
        )
        values.append(float(np.mean(numerator / denominator)))
    return float(np.mean(values))
```

Always print both values. Use `warnings.warn` when PSNR is below 30 dB or SSIM below 0.95, but still enforce exact target dimensions, the approved FFmpeg scaler flags, and successful image decode.

- [ ] **Step 5: Run the integration tests**

Run: `.venv\Scripts\python.exe -m pytest tests/integration/test_direct_vr_ffmpeg_encoding.py -v -m integration -s`

Expected: SBS, OU, and audio tests pass; quality metrics are printed. Any warning below the calibrated floors must be investigated and reported before completion.

- [ ] **Step 6: Checkpoint the integration test**

Run: `git diff --check -- tests/integration/test_direct_vr_ffmpeg_encoding.py`

Clean-worktree commit: `git commit -m "test: verify direct VR FFmpeg encoding"`

### Task 9: Full Verification And Scope Audit

**Files:**
- Verify all changed files; production changes are allowed only to fix failures caused by this feature.

**Interfaces:**
- Produces: repository-wide correctness evidence, quality metrics, and a concise residual-risk report.

- [ ] **Step 1: Scan for incomplete implementation markers and accidental identity changes**

Run: `rg -n "TODO|TBD|pass$|NotImplemented|direct_vr_encode" src templates tests docs/superpowers`

Expected: no new placeholder implementation; `direct_vr_encode` appears only in settings, orchestration, UI, and tests, never in stage identities or `_VR_SETTING_KEYS`.

- [ ] **Step 2: Run whitespace and formatter checks**

Run: `git diff --check`

Run: `.venv\Scripts\python.exe -m black --check src/ tests/ app.py`

Expected: both commands pass.

- [ ] **Step 3: Run lint and static typing**

Run: `.venv\Scripts\python.exe -m flake8 src/ tests/ app.py --count --show-source --statistics`

Run: `.venv\Scripts\python.exe -m mypy src/depth_surge_3d --ignore-missing-imports`

Expected: both commands pass without new suppressions for this feature.

- [ ] **Step 4: Run all unit tests**

Run: `.venv\Scripts\python.exe -m pytest tests/unit -q`

Expected: the complete unit suite passes; the default-path tests prove no legacy regression.

- [ ] **Step 5: Run direct FFmpeg integration tests**

Run: `.venv\Scripts\python.exe -m pytest tests/integration/test_direct_vr_ffmpeg_encoding.py -v -m integration -s`

Expected: both layouts and audio pass, and PSNR/SSIM values are captured in the completion report.

- [ ] **Step 6: Run the full repository suite**

Run: `.venv\Scripts\python.exe -m pytest -q`

Expected: all tests pass or environment-only skips are explicitly identified.

- [ ] **Step 7: Review FFmpeg command and failure invariants manually**

Inspect the final command for exactly two image inputs, identical fractional rates and start numbers, explicit `[vr]` mapping, optional `2:a:0?`, exact frame limit, approved encoder arguments, and a temporary `.mp4` output. Inject a nonzero fake FFmpeg result and reconfirm that the old final file and upstream PNGs remain untouched.

- [ ] **Step 8: Review the final diff and commit**

Run: `git status --short`

Run: `git diff --stat origin/main...HEAD`

Run: `git diff --check origin/main...HEAD`

Confirm the feature remains opt-in, no unrelated files changed, and no generated media or temporary output was staged.

Clean-worktree commit for any final test-only correction: `git commit -m "test: complete direct VR encoding coverage"`
