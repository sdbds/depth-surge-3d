# MoGe-2 Flat SBS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the three pinned MoGe-2 variants as an optional depth backend and add an opt-in, bounded metric-camera projection for flat rectified SBS video without changing existing relative output bytes.

**Architecture:** A single backend registry owns estimator identity, availability, variants, capabilities, and construction. MoGe-2 returns typed metric depth plus normalized horizontal focal length, raw schema v3 commits both atomically, and a restartable `03_metric_geometry` stage derives inverse depth, validity, and one clip-global automatic convergence value. Relative and metric builders both produce `StereoGeometryFrame`; the existing banded forward splat consumes that backend-neutral geometry and never branches on model identity.

**Tech Stack:** Python 3.10-3.12, NumPy, PyTorch 2.13, OpenCV, Flask/Socket.IO, FFmpeg/ffprobe, pytest, Hugging Face Hub, optional `moge` 2 source dependency.

## Global Constraints

- Execute this plan in an isolated `codex/moge2-flat-sbs` worktree created with `superpowers:using-git-worktrees`; the current workspace contains unrelated uncommitted edits in overlapping files.
- Preserve the setting name `depth_model_version`; accepted IDs are exactly `v2`, `v3`, `see_through`, and `moge2`. Unknown IDs fail and never fall through to V3.
- Pin MoGe source to `925b8ed835a7a9cdb7578ba15c658a0afc969030` and install it only through the `moge2` optional extra. The supported command is exactly `uv sync --extra moge2`.
- Pin variants exactly: `vits` -> `Ruicheng/moge-2-vits-normal@679230677b4d282c6f304189a93e98e14f085902` (35M), `vitb` -> `Ruicheng/moge-2-vitb-normal@54ad3a693e61907ea4633d13dec6ee682fa09419` (104M), and `vitl` -> `Ruicheng/moge-2-vitl@39c4d5e957afe587e04eec59dc2bcc3be5ecd968` (326M). MoGe-2 defaults to `vitb`.
- Keep `MOGE_RESOLUTION_LEVEL = 9` internal. It is fingerprinted and reported, but there is no setting, HTML control, or CLI flag for it.
- Call `moge.model.v2.MoGeModel.infer(force_projection=False, apply_mask=True, resolution_level=9, use_fp16=self.device.startswith("cuda") and not fp32)`; pass a concrete `model.pt` file to `from_pretrained`, not a snapshot directory.
- MoGe preprocessing is exactly BGR uint8 -> RGB float32 `[0,1]` -> one aspect-preserving OpenCV `INTER_AREA` downscale whose longest edge is at most `depth_resolution`, with no upscaling. Its identity is `moge2-rgb-area-max-edge-v1`.
- The first MoGe release uses `max_batch_size = 1`; it never lowers resolution, changes model, changes device, or falls back to another backend after OOM.
- The global geometry default remains `relative`. `metric_camera` is Experimental, supports only `vr_format=side_by_side`, requires `apply_distortion=false`, requires square-pixel source SAR, and is never exposed for a backend without `pinhole_fx`.
- Metric defaults and bounds are exact: `virtual_baseline_mm=63.0` in `[0,100]`, `metric_convergence_distance="auto"` or `[0.1,1000]` metres, and `max_disparity_percent=2.0` in `[0,5]`.
- Explicit SAR syntax is ASCII unsigned decimal `numerator:denominator`; each component is in `1..2147483647`, the pair is reduced by GCD, and metric mode accepts only reduced `1:1`. Missing and `N/A` normalize to `1:1`; zero, signs, overflow, trailing data, and non-square ratios fail before model loading.
- New raw writes use schema v3. A v3 `camera_model=none` NPZ contains exactly `values.npy`; a v3 `camera_model=pinhole_fx` NPZ contains exactly `values.npy` and zero-dimensional float32 `focal_x_normalized.npy`. Valid schema-v2 depth-only payloads remain readable and are not rewritten.
- Metric geometry lives in `03_metric_geometry`; relative geometry remains in `03_disparity_maps`. Only the active stage is generated, compatible inactive stages are preserved, and successful `keep_intermediates=false` cleanup removes both.
- Automatic convergence is one deterministic float32 median for the whole clip: at most 32 source-ordered frames per candidate scene and at most a 64x64 source-ordered grid of valid positive metric depths per selected frame. It is resolved even when the active render setting is explicit, so changing explicit/auto never rebuilds metric geometry.
- The metric disparity formula, crop transform, sign, and cap are exactly those in `docs/superpowers/specs/2026-08-15-moge2-flat-sbs-design.md`. Z-order always uses unclamped inverse depth.
- A frame with no valid metric pixels has clamp fraction zero. The earliest source frame above 5 percent emits one warning per job. The final summary persists affected-frame count and mean/maximum per-frame clamp fraction; no histogram is added.
- The live Web preview remains in-process. Metric stereo preview starts only after stage 3 has persisted the clip-global convergence; no pre-processing preview endpoint or provisional frame convergence is added.
- Show this exact copy in CLI, Web, and documentation: `MoGe-2 performs per-frame depth and focal estimation. Temporal stability on video is not guaranteed; depth or focal drift may be visible across frames.`
- Do not add point maps, normal maps, full intrinsics, confidence dictionaries, VR180, dome, equirectangular, over-under metric output, temporal stabilization, or a public MoGe resolution-level control.
- The fixed CPU relative regression corpus must remain byte-identical for both eye images and all four masks.

---

## File Map

### New production files

- `src/depth_surge_3d/inference/depth/backend_registry.py`: immutable backend/variant/capability registry, availability checks, estimator construction, and backend/geometry request validation.
- `src/depth_surge_3d/inference/depth/video_depth_estimator_moge2.py`: pinned MoGe artifact resolution, preprocessing, inference, validation, typed camera output, and OOM reporting.
- `src/depth_surge_3d/processing/frames/metric_geometry.py`: metric frame contract, deterministic convergence sampler, atomic stage store, disk bound, allocation-unit probe, and ENOSPC recovery.
- `src/depth_surge_3d/rendering/stereo_geometry.py`: immutable common geometry, relative builder, mask-aware metric resize, metric builder, crop-aware disparity formula, and clamp statistics.
- `requirements-moge2.txt`: pip-oriented source pin for the optional backend.
- `THIRD_PARTY_NOTICES.md`: MoGe MIT and bundled DINOv2 Apache-2.0 attribution.
- `scripts/verify_moge2_release.py`: non-CI three-variant image/video evidence runner.
- `docs/release/moge2-release-checklist.md`: fixed-corpus commands and evidence interpretation rules.

### New tests

- `tests/unit/test_backend_registry.py`
- `tests/unit/test_video_depth_estimator_moge2.py`
- `tests/unit/test_metric_geometry.py`
- `tests/unit/test_stereo_geometry.py`
- `tests/unit/test_cli_moge2.py`
- `tests/unit/test_web_moge2.py`
- `tests/unit/test_moge2_docs.py`
- `tests/integration/test_moge2_pipeline.py`
- `tests/unit/test_moge2_release_script.py`

### Existing files with focused changes

- Dependency/export contracts: `pyproject.toml`, `uv.lock`, `src/depth_surge_3d/inference/depth/__init__.py`, `src/depth_surge_3d/inference/__init__.py`.
- Typed data and artifact resolution: `src/depth_surge_3d/inference/depth/types.py`, `src/depth_surge_3d/inference/depth/model_artifact.py`.
- Existing estimator reporting: `src/depth_surge_3d/inference/depth/video_depth_estimator.py`, `src/depth_surge_3d/inference/depth/video_depth_estimator_da3.py`, `src/depth_surge_3d/inference/depth/video_depth_estimator_see_through.py`.
- Settings/constants/SAR/resolution: `src/depth_surge_3d/core/settings.py`, `src/depth_surge_3d/core/constants.py`, `src/depth_surge_3d/io/operations.py`, `src/depth_surge_3d/utils/domain/resolution.py`.
- Persistence and stage derivation: `src/depth_surge_3d/processing/frames/depth_storage.py`, `src/depth_surge_3d/processing/frames/depth_processor.py`, `src/depth_surge_3d/processing/frames/scene_analyzer.py`.
- Common renderer: `src/depth_surge_3d/rendering/forward_splat.py`, `src/depth_surge_3d/rendering/stereo_renderer.py`, `src/depth_surge_3d/processing/frames/stereo_generator.py`, `src/depth_surge_3d/utils/imaging/image_processing.py`.
- Pipeline/resume: `src/depth_surge_3d/io/resume.py`, `src/depth_surge_3d/processing/orchestration/pipeline_orchestrator.py`, `src/depth_surge_3d/processing/video/video_encoder.py`, `src/depth_surge_3d/rendering/stereo_projector.py`.
- Product surfaces: `depth_surge_3d.py`, `app.py`, `templates/index.html`.
- Documentation: `README.md`, `docs/INSTALLATION.md`, `docs/PARAMETERS.md`, `docs/ARCHITECTURE.md`, `docs/USAGE.md`, `docs/TROUBLESHOOTING.md`, `example_settings.json`, `CHANGELOG.md`.

## Shared Interfaces

Define these names once and keep every later task consistent with them:

```python
# inference/depth/backend_registry.py
StereoGeometryMode = Literal["relative", "metric_camera"]

@dataclass(frozen=True)
class BackendCapabilities:
    metric_depth: bool
    pinhole_fx: bool
    stereo_geometry_modes: frozenset[StereoGeometryMode]

@dataclass(frozen=True)
class ModelVariantSpec:
    setting: str
    display_name: str
    backend_value: str | None = None
    repo_id: str | None = None
    revision: str | None = None
    parameters_millions: int | None = None

@dataclass(frozen=True)
class BackendAvailability:
    available: bool
    reason: str | None = None
    install_command: str | None = None

@dataclass(frozen=True)
class EstimatorRequest:
    model_path: str | None
    model_size: str | None
    device: str
    metric: bool
    temporal_window_overlap: int

@dataclass(frozen=True)
class DepthBackendSpec:
    backend_id: str
    display_name: str
    default_model_size: str
    variants: Mapping[str, ModelVariantSpec]
    capabilities: BackendCapabilities
    factory: Callable[[EstimatorRequest], Any]
    availability_probe: Callable[[], BackendAvailability]

def get_backend_spec(backend_id: str) -> DepthBackendSpec: ...
def list_backend_specs() -> tuple[DepthBackendSpec, ...]: ...
def backend_availability(backend_id: str) -> BackendAvailability: ...
def resolve_model_variant(backend_id: str, model_size: str | None) -> ModelVariantSpec: ...
def create_registered_depth_estimator(backend_id: str, request: EstimatorRequest) -> Any: ...
def validate_backend_geometry_request(
    settings: Mapping[str, Any], video_properties: Mapping[str, Any]
) -> None: ...

TEMPORAL_STABILITY_WARNING = (
    "MoGe-2 performs per-frame depth and focal estimation. Temporal stability "
    "on video is not guaranteed; depth or focal drift may be visible across frames."
)

def build_effective_depth_run_report(
    settings: Mapping[str, Any], estimator: Any
) -> dict[str, Any]: ...
```

```python
# inference/depth/types.py
@dataclass(frozen=True)
class PinholeCameraBatch:
    focal_x_normalized: np.ndarray  # float32 [N]

@dataclass(frozen=True)
class DepthBatch:
    values: np.ndarray              # float32 [N,H,W]
    representation: DepthRepresentation
    camera: PinholeCameraBatch | None = None
```

```python
# rendering/stereo_geometry.py
@dataclass(frozen=True)
class StereoGeometryFrame:
    near_score: np.ndarray                 # float32 [H,W]
    total_disparity_fraction: np.ndarray   # float64 [H,W]
    source_valid: np.ndarray               # bool [H,W]

@dataclass(frozen=True)
class MetricProjectionStats:
    valid_pixel_count: int
    clamped_pixel_count: int
    clamped_fraction: float

@dataclass(frozen=True)
class StereoSplatSettings:
    max_eye_shift_fraction: float
    occlusion_fill: Literal["none", "background"] = "background"

def build_relative_geometry(
    canonical: np.ndarray,
    render_shape: tuple[int, int],
    *,
    stereo_strength: float,
    convergence: float,
) -> StereoGeometryFrame: ...

def build_metric_geometry(
    inverse_depth: np.ndarray,
    valid: np.ndarray,
    focal_x_normalized: np.float32,
    render_shape: tuple[int, int],
    *,
    virtual_baseline_mm: float,
    convergence_distance_m: float,
    max_disparity_percent: float,
    retained_crop_width: int,
) -> tuple[StereoGeometryFrame, MetricProjectionStats]: ...

def resize_metric_geometry(
    inverse_depth: np.ndarray,
    valid: np.ndarray,
    render_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]: ...

def calculate_geometry_eye_sample_offsets(
    total_disparity_fraction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]: ...
```

```python
# processing/frames/metric_geometry.py
@dataclass(frozen=True)
class MetricGeometryFrame:
    inverse_depth: np.ndarray       # float32 [H,W], invalid is zero
    valid: np.ndarray               # bool [H,W]
    focal_x_normalized: np.float32  # scalar

@dataclass(frozen=True)
class ClipConvergence:
    distance_m: np.float32
    selected_frame_indexes: tuple[int, ...]
    sample_count: int

class MetricGeometryDiskFullError(OSError):
    required_bytes: int
    free_bytes: int
    failing_path: Path

class MetricGeometryStore:
    @classmethod
    def open_existing(
        cls,
        directory: Path,
        *,
        frame_names: list[str],
        source_raw_fingerprint: str,
        source_frame_fingerprint: str,
        candidate_scene_fingerprint: str,
    ) -> "MetricGeometryStore": ...

    @classmethod
    def open_or_create(
        cls,
        directory: Path,
        *,
        frame_names: list[str],
        native_shape: tuple[int, int],
        source_raw_fingerprint: str,
        source_frame_fingerprint: str,
        candidate_scene_fingerprint: str,
        preflight_required_bytes: int,
    ) -> "MetricGeometryStore": ...

    def path_for(self, frame_name: str) -> Path: ...
    @property
    def complete_files(self) -> tuple[Path, ...]: ...
    def write_frame(self, frame_name: str, frame: MetricGeometryFrame) -> Path: ...
    def load(self, path: Path) -> MetricGeometryFrame: ...
    def finalize(self, convergence: ClipConvergence) -> dict[str, Any]: ...
    def validate_payloads(self) -> int: ...

def sample_clip_convergence(
    raw_store: RawDepthStore,
    raw_files: Sequence[Path],
    candidate_scene_ids: Sequence[int],
) -> ClipConvergence: ...

def estimate_metric_geometry_disk_bytes(
    frame_shapes: Sequence[tuple[int, int]], *, allocation_unit: int
) -> int: ...

def filesystem_allocation_unit(directory: Path) -> int: ...
def require_metric_geometry_disk_space(directory: Path, required_bytes: int) -> None: ...
def metric_frame_from_depth(
    depth: np.ndarray, focal_x_normalized: np.float32
) -> MetricGeometryFrame: ...

# processing/frames/depth_storage.py
def estimate_raw_depth_only_bytes(
    *,
    frame_count: int,
    native_width: int,
    native_height: int,
    storage_bytes: int,
    camera_bytes_per_frame: int,
) -> int: ...
```

---

## Slice 1: Backend and Raw Contract

### Task 1: Freeze the Relative CPU Baseline

**Files:**
- Modify: `tests/unit/test_stereo_renderer.py`

**Interfaces:**
- Consumes: current `StereoRenderer.render(frame, canonical, StereoRenderSettings)` behavior before any renderer refactor.
- Produces: a deterministic SHA-256 regression that later tasks must keep green.

- [ ] **Step 1: Add the characterization test before changing production code**

```python
import hashlib


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def test_relative_cpu_regression_corpus_is_byte_frozen() -> None:
    generator = np.random.default_rng(20260816)
    frame = generator.integers(0, 256, size=(7, 19, 3), dtype=np.uint8)
    canonical = generator.random((5, 13), dtype=np.float32)
    result = StereoRenderer(
        device="cpu",
        temporary_budget_bytes=19 * SPLAT_BYTES_PER_PIXEL * 2,
    ).render(
        frame,
        canonical,
        StereoRenderSettings(
            stereo_strength=3.75,
            convergence=0.42,
            occlusion_fill="background",
        ),
    )

    assert _array_sha256(result.left_image) == (
        "48f5e73497d9adea81c5a0dc1444dfa23776f71af291c4a8995470435052d0ce"
    )
    assert _array_sha256(result.right_image) == (
        "5fbd0cb67a9a31f7f18957d597979a0c88db49d5df7b29e2ce731bfd7e3aae05"
    )
    assert _array_sha256(result.left_valid_mask) == (
        "2be1e207cb3c363ebec163e3de22b4b60a5357f10f7e3afc1e10a7e47c4dc03a"
    )
    assert _array_sha256(result.right_valid_mask) == (
        "2be1e207cb3c363ebec163e3de22b4b60a5357f10f7e3afc1e10a7e47c4dc03a"
    )
    assert _array_sha256(result.left_hole_mask) == (
        "57ffc9ca3beb6ee6226c28248ab9c77b2076ef6acffba839cec21fac28a8fd1f"
    )
    assert _array_sha256(result.right_hole_mask) == (
        "57ffc9ca3beb6ee6226c28248ab9c77b2076ef6acffba839cec21fac28a8fd1f"
    )
```

- [ ] **Step 2: Run the characterization test on the pre-refactor renderer**

Run: `uv run pytest tests/unit/test_stereo_renderer.py::test_relative_cpu_regression_corpus_is_byte_frozen -v`

Expected: PASS. A failure means the worktree is not based on the approved renderer and execution must stop before recording new hashes.

- [ ] **Step 3: Commit the frozen baseline alone**

```bash
git add tests/unit/test_stereo_renderer.py
git commit -m "test: freeze relative stereo CPU output"
```

### Task 2: Add the Optional MoGe Dependency and Immutable Artifact Resolution

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `requirements-moge2.txt`
- Modify: `src/depth_surge_3d/inference/depth/model_artifact.py`
- Modify: `tests/unit/test_model_artifact.py`

**Interfaces:**
- Consumes: existing `resolve_hf_snapshot(repo_id, cache_dir=None)` callers.
- Produces: backward-compatible `resolve_hf_snapshot(repo_id, *, revision=None, cache_dir=None) -> tuple[str, str]`.

- [ ] **Step 1: Write failing tests for revision forwarding and cache-only retry**

```python
def test_resolve_hf_snapshot_forwards_immutable_revision(monkeypatch, tmp_path):
    snapshot = tmp_path / "models--owner--repo" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)

    path, identity = resolve_hf_snapshot("owner/repo", revision="pinned123")

    assert Path(path) == snapshot.resolve()
    assert calls == [{"repo_id": "owner/repo", "revision": "pinned123"}]
    assert identity == "hf:owner/repo@abc123"


def test_resolve_hf_snapshot_keeps_revision_during_offline_retry(monkeypatch, tmp_path):
    snapshot = tmp_path / "models--owner--repo" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        if not kwargs.get("local_files_only"):
            raise ConnectionError("offline")
        return str(snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    resolve_hf_snapshot("owner/repo", revision="pinned123")

    assert calls[1] == {
        "repo_id": "owner/repo",
        "revision": "pinned123",
        "local_files_only": True,
    }
```

- [ ] **Step 2: Run the focused tests and verify the new keyword is rejected**

Run: `uv run pytest tests/unit/test_model_artifact.py -v`

Expected: FAIL with `TypeError: resolve_hf_snapshot() got an unexpected keyword argument 'revision'`.

- [ ] **Step 3: Extend artifact resolution without changing local-directory identity**

```python
def resolve_hf_snapshot(
    repo_id: str,
    *,
    revision: str | None = None,
    cache_dir: Path | str | None = None,
) -> tuple[str, str]:
    local_path = Path(repo_id).expanduser()
    if local_path.is_dir():
        resolved = local_path.resolve()
        return str(resolved), f"local:{_hash_directory(resolved)}"

    from huggingface_hub import snapshot_download

    kwargs: dict[str, Any] = {"repo_id": repo_id}
    if revision is not None:
        kwargs["revision"] = revision
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    try:
        snapshot = Path(snapshot_download(**kwargs)).resolve()
    except Exception as online_error:
        try:
            snapshot = Path(snapshot_download(**kwargs, local_files_only=True)).resolve()
        except Exception:
            raise online_error
    resolved_revision = (
        snapshot.name if snapshot.parent.name == "snapshots" else _hash_directory(snapshot)
    )
    return str(snapshot), f"hf:{repo_id}@{resolved_revision}"
```

- [ ] **Step 4: Add the optional source pin and regenerate the lockfile**

```toml
[project.optional-dependencies]
moge2 = [
    "moge @ git+https://github.com/microsoft/MoGe.git@925b8ed835a7a9cdb7578ba15c658a0afc969030",
]
```

`requirements-moge2.txt` contains exactly:

```text
git+https://github.com/microsoft/MoGe.git@925b8ed835a7a9cdb7578ba15c658a0afc969030#egg=moge
```

Run: `uv lock`

Expected: `uv.lock` records the MoGe Git source at commit `925b8ed835a7a9cdb7578ba15c658a0afc969030`, while `uv sync` without `--extra moge2` does not install it.

- [ ] **Step 5: Verify the default and optional dependency paths**

Run: `uv run pytest tests/unit/test_model_artifact.py -v`

Expected: PASS.

Run: `uv sync --no-dev`

Run: `uv pip show moge`

Expected: `uv pip show moge` exits nonzero and reports `Package(s) not found for: moge`,
confirming that the default installation state excludes the optional dependency.

- [ ] **Step 6: Commit dependency and resolver changes**

```bash
git add pyproject.toml uv.lock requirements-moge2.txt src/depth_surge_3d/inference/depth/model_artifact.py tests/unit/test_model_artifact.py
git commit -m "build: add pinned optional MoGe-2 dependency"
```

### Task 3: Introduce the Backend Registry

**Files:**
- Create: `src/depth_surge_3d/inference/depth/backend_registry.py`
- Create: `tests/unit/test_backend_registry.py`
- Modify: `src/depth_surge_3d/inference/depth/__init__.py`
- Modify: `src/depth_surge_3d/inference/__init__.py`
- Modify: `src/depth_surge_3d/rendering/stereo_projector.py`
- Modify: `src/depth_surge_3d/processing/frames/depth_processor.py`
- Modify: `tests/unit/test_depth_processor.py`
- Modify: `tests/integration/test_end_to_end.py`

**Interfaces:**
- Consumes: existing V2, V3, and See-Through factory functions and constants.
- Produces: foundational registry dataclasses plus `get_backend_spec`,
  `list_backend_specs`, `backend_availability`, `resolve_model_variant`, and
  `create_registered_depth_estimator`, together with registry-based projector
  construction.
- Deferred ownership: Task 7 implements `validate_backend_geometry_request`;
  Task 13 implements `TEMPORAL_STABILITY_WARNING` and
  `build_effective_depth_run_report`. The global **Shared Interfaces** section
  remains the eventual cross-task contract.

- [ ] **Step 1: Write failing registry tests for identity, variants, capabilities, and unknown IDs**

```python
@pytest.mark.parametrize("backend_id", ["v2", "v3", "see_through", "moge2"])
def test_registry_contains_every_supported_backend(backend_id: str) -> None:
    assert get_backend_spec(backend_id).backend_id == backend_id


def test_unknown_backend_is_not_a_v3_fallback() -> None:
    with pytest.raises(ValueError, match="Unknown depth backend: typo"):
        get_backend_spec("typo")


def test_moge_variants_and_pins_are_exact() -> None:
    spec = get_backend_spec("moge2")
    assert spec.default_model_size == "vitb"
    assert {
        key: (value.repo_id, value.revision, value.parameters_millions)
        for key, value in spec.variants.items()
    } == {
        "vits": (
            "Ruicheng/moge-2-vits-normal",
            "679230677b4d282c6f304189a93e98e14f085902",
            35,
        ),
        "vitb": (
            "Ruicheng/moge-2-vitb-normal",
            "54ad3a693e61907ea4633d13dec6ee682fa09419",
            104,
        ),
        "vitl": (
            "Ruicheng/moge-2-vitl",
            "39c4d5e957afe587e04eec59dc2bcc3be5ecd968",
            326,
        ),
    }
    assert spec.capabilities == BackendCapabilities(
        metric_depth=True,
        pinhole_fx=True,
        stereo_geometry_modes=frozenset({"relative", "metric_camera"}),
    )


def test_missing_moge_extra_reports_only_supported_command(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    availability = backend_availability("moge2")
    assert availability == BackendAvailability(
        available=False,
        reason="MoGe-2 optional dependency is not installed",
        install_command="uv sync --extra moge2",
    )
```

- [ ] **Step 2: Run the registry tests and verify the module is absent**

Run: `uv run pytest tests/unit/test_backend_registry.py -v`

Expected: FAIL during collection with `ModuleNotFoundError` for `backend_registry`.

- [ ] **Step 3: Implement immutable specs and explicit availability**

Use `MappingProxyType` for each `variants` mapping and preserve registration order `v2`, `v3`, `see_through`, `moge2`. Use these capability values:

```python
_RELATIVE_ONLY = frozenset({"relative"})

BackendCapabilities(metric_depth=True, pinhole_fx=False, stereo_geometry_modes=_RELATIVE_ONLY)  # v2
BackendCapabilities(metric_depth=True, pinhole_fx=False, stereo_geometry_modes=_RELATIVE_ONLY)  # v3
BackendCapabilities(metric_depth=False, pinhole_fx=False, stereo_geometry_modes=_RELATIVE_ONLY) # see_through
BackendCapabilities(
    metric_depth=True,
    pinhole_fx=True,
    stereo_geometry_modes=frozenset({"relative", "metric_camera"}),
)  # moge2
```

The MoGe availability probe must avoid importing the optional package:

```python
def _moge_availability() -> BackendAvailability:
    try:
        installed = importlib.util.find_spec("moge.model.v2") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        installed = False
    if installed:
        return BackendAvailability(available=True)
    return BackendAvailability(
        available=False,
        reason="MoGe-2 optional dependency is not installed",
        install_command="uv sync --extra moge2",
    )
```

- [ ] **Step 4: Implement existing-backend factories through the registry**

The private factories resolve `vits/vitb/vitl` before calling existing constructors:

```python
def _create_v2(request: EstimatorRequest) -> Any:
    variant = resolve_model_variant("v2", request.model_size)
    paths = MODEL_PATHS_METRIC if request.metric else MODEL_PATHS
    model_path = request.model_path or paths[variant.setting]
    return create_video_depth_estimator(
        model_path,
        request.device,
        request.metric,
        request.temporal_window_overlap,
    )


def _create_v3(request: EstimatorRequest) -> Any:
    variant = resolve_model_variant("v3", request.model_size)
    model_name = request.model_path or cast(str, variant.backend_value)
    return create_video_depth_estimator_da3(model_name, request.device, request.metric)


def _create_see_through(request: EstimatorRequest) -> Any:
    return create_see_through_depth_estimator(
        request.model_path or DEFAULT_SEE_THROUGH_REPO,
        request.device,
        False,
    )
```

Register V2 defaults as `vitl`, V3 defaults as `vitl`, See-Through as the single `vitl` presentation variant, and MoGe as `vitb`. The MoGe entry is a lazy adapter handoff so a default installation can import the registry; Task 4 owns `video_depth_estimator_moge2.py`, selected-factory construction, and its focused adapter test.

- [ ] **Step 5: Replace projector `if/elif/else` construction with one registry call**

Extend the constructor and factory with keyword-only `model_size` while preserving the four existing positional parameters:

```python
def __init__(
    self,
    model_path: str | None = None,
    device: str = "auto",
    metric: bool = False,
    depth_model_version: str = "v2",
    temporal_window_overlap: int = 10,
    stereo_renderer: StereoRenderer | None = None,
    *,
    model_size: str | None = None,
) -> None:
    self.backend_spec = get_backend_spec(depth_model_version)
    self.depth_estimator = create_registered_depth_estimator(
        depth_model_version,
        EstimatorRequest(
            model_path=model_path,
            model_size=model_size,
            device=device,
            metric=metric,
            temporal_window_overlap=temporal_window_overlap,
        ),
    )
```

Update integration mocks to patch `create_registered_depth_estimator`, then add:

```python
def test_projector_rejects_unknown_backend_without_constructing_an_estimator(monkeypatch) -> None:
    factory = Mock()
    monkeypatch.setattr(stereo_projector, "create_registered_depth_estimator", factory)
    with pytest.raises(ValueError, match="Unknown depth backend: typo"):
        create_stereo_projector(depth_model_version="typo")
    factory.assert_not_called()
```

- [ ] **Step 6: Run registry and projector tests**

Run: `uv run pytest tests/unit/test_depth_processor.py tests/unit/test_backend_registry.py tests/unit/test_stereo_projector.py tests/integration/test_end_to_end.py -v`

Expected: PASS without importing `moge` in tests that do not select it.

- [ ] **Step 7: Commit the registry slice**

```bash
git add docs/superpowers/plans/2026-08-16-moge2-flat-sbs-implementation.md src/depth_surge_3d/inference/depth/backend_registry.py src/depth_surge_3d/inference/depth/__init__.py src/depth_surge_3d/inference/__init__.py src/depth_surge_3d/rendering/stereo_projector.py src/depth_surge_3d/processing/frames/depth_processor.py tests/unit/test_backend_registry.py tests/unit/test_depth_processor.py tests/unit/test_stereo_projector.py tests/integration/test_end_to_end.py
git commit -m "refactor: centralize depth backend dispatch"
```

### Task 4: Add Typed Camera Output and the MoGe-2 Adapter

**Files:**
- Modify: `src/depth_surge_3d/inference/depth/types.py`
- Create: `src/depth_surge_3d/inference/depth/video_depth_estimator_moge2.py`
- Modify: `src/depth_surge_3d/inference/depth/backend_registry.py`
- Modify: `src/depth_surge_3d/inference/depth/__init__.py`
- Create: `tests/unit/test_video_depth_estimator_moge2.py`
- Modify: `tests/unit/test_depth_contract.py`

**Interfaces:**
- Consumes: `resolve_hf_snapshot(..., revision=...)`, registry `ModelVariantSpec`, and existing `DepthBatch` callers.
- Produces: `PinholeCameraBatch`, optional `DepthBatch.camera`, and `VideoDepthEstimatorMoGe2` with `max_batch_size=1`, `camera_model="pinhole_fx"`, `estimate_output_shape`, `load_model`, `estimate_depth_batch`, `get_model_size`, `get_model_info`, and `unload_model`.

- [ ] **Step 1: Write failing camera-contract tests**

```python
def test_existing_depth_batch_remains_camera_free() -> None:
    batch = DepthBatch(
        np.ones((2, 3, 4), dtype=np.float32),
        DepthRepresentation.RELATIVE_DEPTH,
    )
    assert batch.camera is None


def test_pinhole_camera_requires_positive_finite_float32_values() -> None:
    with pytest.raises(TypeError, match="float32"):
        PinholeCameraBatch(np.array([1.0], dtype=np.float64))
    with pytest.raises(ValueError, match="positive"):
        PinholeCameraBatch(np.array([0.0], dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        PinholeCameraBatch(np.array([np.nan], dtype=np.float32))


def test_depth_batch_camera_count_matches_depth_count() -> None:
    with pytest.raises(ValueError, match="batch length"):
        DepthBatch(
            np.ones((2, 3, 4), dtype=np.float32),
            DepthRepresentation.METRIC_DEPTH,
            camera=PinholeCameraBatch(np.array([0.8], dtype=np.float32)),
        )
```

- [ ] **Step 2: Run the contract tests and verify `PinholeCameraBatch` is missing**

Run: `uv run pytest tests/unit/test_depth_contract.py -v`

Expected: FAIL during import because `PinholeCameraBatch` does not exist.

- [ ] **Step 3: Implement strict camera validation and preserve old constructors**

```python
@dataclass(frozen=True)
class PinholeCameraBatch:
    focal_x_normalized: np.ndarray

    def __post_init__(self) -> None:
        values = self.focal_x_normalized
        if not isinstance(values, np.ndarray):
            raise TypeError("PinholeCameraBatch.focal_x_normalized must be a numpy array")
        if values.dtype != np.float32:
            raise TypeError("PinholeCameraBatch.focal_x_normalized must use float32")
        if values.ndim != 1:
            raise ValueError("PinholeCameraBatch.focal_x_normalized must have shape [N]")
        if not np.isfinite(values).all():
            raise ValueError("Pinhole focal values must be finite")
        if np.any(values <= 0.0):
            raise ValueError("Pinhole focal values must be positive")


@dataclass(frozen=True)
class DepthBatch:
    values: np.ndarray
    representation: DepthRepresentation
    camera: PinholeCameraBatch | None = None

    def __post_init__(self) -> None:
        # Keep every existing values/representation check unchanged.
        if self.camera is not None:
            if not isinstance(self.camera, PinholeCameraBatch):
                raise TypeError("DepthBatch.camera must be a PinholeCameraBatch or None")
            if len(self.camera.focal_x_normalized) != len(self.values):
                raise ValueError("Depth and camera batch lengths must match")
```

- [ ] **Step 4: Write failing adapter tests around a fake upstream model**

```python
class FakeMoGeModel:
    def __init__(self) -> None:
        self.calls = []

    def infer(self, image: torch.Tensor, **kwargs):
        self.calls.append((image.detach().cpu(), kwargs))
        height, width = image.shape[-2:]
        depth = torch.full((1, height, width), 2.0, dtype=torch.float32)
        mask = torch.ones((1, height, width), dtype=torch.bool)
        mask[:, 0, 0] = False
        intrinsics = torch.eye(3, dtype=torch.float32).unsqueeze(0)
        intrinsics[:, 0, 0] = 0.75
        return {"depth": depth, "mask": mask, "intrinsics": intrinsics}


def test_moge_preprocessing_is_rgb_area_no_upscale_and_forwards_fixed_options(monkeypatch):
    resize_calls = []
    real_resize = cv2.resize

    def recording_resize(image, size, *, interpolation):
        resize_calls.append((size, interpolation))
        return real_resize(image, size, interpolation=interpolation)

    monkeypatch.setattr(cv2, "resize", recording_resize)
    estimator = VideoDepthEstimatorMoGe2(model_size="vitb", device="cpu")
    estimator.model = FakeMoGeModel()
    bgr = np.zeros((1, 4, 8, 3), dtype=np.uint8)
    bgr[0, :, :, 0] = 255

    result = estimator.estimate_depth_batch(bgr, input_size=4, fp32=False)

    image, kwargs = estimator.model.calls[0]
    assert image.shape == (1, 3, 2, 4)
    assert image.dtype == torch.float32
    assert torch.all(image[:, 2] == 1.0)  # BGR blue becomes RGB blue.
    assert resize_calls == [((4, 2), cv2.INTER_AREA)]
    assert kwargs == {
        "force_projection": False,
        "apply_mask": True,
        "resolution_level": 9,
        "use_fp16": False,
    }
    assert result.representation is DepthRepresentation.METRIC_DEPTH
    assert np.isinf(result.values[0, 0, 0])
    np.testing.assert_array_equal(
        result.camera.focal_x_normalized,
        np.array([0.75], dtype=np.float32),
    )


def test_moge_never_upscales_and_accepts_only_one_frame() -> None:
    estimator = VideoDepthEstimatorMoGe2(model_size="vitb", device="cpu")
    estimator.model = FakeMoGeModel()
    estimator.estimate_depth_batch(np.zeros((1, 3, 5, 3), dtype=np.uint8), input_size=20)
    image, _kwargs = estimator.model.calls[0]
    assert image.shape[-2:] == (3, 5)
    with pytest.raises(ValueError, match="one frame"):
        estimator.estimate_depth_batch(np.zeros((2, 3, 5, 3), dtype=np.uint8))
```

Add these exact negative/precision cases; `FakeMoGeModel.output` defaults to the valid mapping above and `infer` returns it when assigned:

```python
@pytest.mark.parametrize("missing", ["depth", "mask", "intrinsics"])
def test_moge_requires_every_output_key(missing: str) -> None:
    estimator = loaded_fake_moge()
    estimator.model.output.pop(missing)
    with pytest.raises(ValueError, match=missing):
        estimator.estimate_depth_batch(one_frame())


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("depth", torch.ones((1, 4, 8), dtype=torch.float64), "depth.*float32"),
        ("depth", torch.ones((4, 8), dtype=torch.float32), "depth.*rank"),
        ("depth", torch.ones((2, 4, 8), dtype=torch.float32), "frame count"),
        ("mask", torch.ones((1, 4, 7), dtype=torch.bool), "spatial shape"),
        ("mask", torch.ones((4, 8), dtype=torch.bool), "mask.*rank"),
        ("intrinsics", torch.eye(3, dtype=torch.float32), "intrinsics.*rank"),
    ],
)
def test_moge_rejects_malformed_outputs(field, replacement, message) -> None:
    estimator = loaded_fake_moge()
    estimator.model.output[field] = replacement
    with pytest.raises((TypeError, ValueError), match=message):
        estimator.estimate_depth_batch(one_frame())


@pytest.mark.parametrize("focal", [0.0, float("nan")])
def test_moge_rejects_invalid_normalized_focal(focal: float) -> None:
    estimator = loaded_fake_moge()
    estimator.model.output["intrinsics"][:, 0, 0] = focal
    with pytest.raises(ValueError, match="focal"):
        estimator.estimate_depth_batch(one_frame())


@pytest.mark.parametrize(
    ("device", "expected_fp16", "expected_precision"),
    [("cpu", False, "float32"), ("cuda", True, "float16")],
)
def test_moge_precision_is_device_fixed(device, expected_fp16, expected_precision) -> None:
    estimator = loaded_fake_moge(device=device)
    estimator.estimate_depth_batch(one_frame())
    assert estimator.model.calls[0][1]["use_fp16"] is expected_fp16
    assert estimator.inference_precision == expected_precision


def test_moge_cuda_honors_explicit_fp32_without_changing_device() -> None:
    estimator = loaded_fake_moge(device="cuda")
    estimator.estimate_depth_batch(one_frame(), fp32=True)
    assert estimator.model.calls[0][1]["use_fp16"] is False
    assert estimator.device == "cuda"


def test_moge_cuda_oom_reports_fixed_inputs() -> None:
    estimator = loaded_fake_moge(device="cuda", model_size="vitl")
    estimator.model.infer = Mock(side_effect=torch.cuda.OutOfMemoryError("oom"))
    with pytest.raises(
        RuntimeError,
        match=r"model_size=vitl.*input=8x4.*resolution_level=9.*precision=float16.*device=cuda",
    ):
        estimator.estimate_depth_batch(one_frame())
```

- [ ] **Step 5: Run adapter tests and verify the module is absent**

Run: `uv run pytest tests/unit/test_video_depth_estimator_moge2.py -v`

Expected: FAIL during collection with `ModuleNotFoundError` for `video_depth_estimator_moge2`.

- [ ] **Step 6: Implement deterministic dimensions and custom artifact rules**

```python
MOGE_SOURCE_REVISION = "925b8ed835a7a9cdb7578ba15c658a0afc969030"
MOGE_RESOLUTION_LEVEL = 9
MOGE_PREPROCESSING_ALGORITHM = "moge2-rgb-area-max-edge-v1"
MOGE_INSTALL_COMMAND = "uv sync --extra moge2"


def _scaled_shape(width: int, height: int, max_edge: int) -> tuple[int, int]:
    if width < 1 or height < 1 or max_edge < 1:
        raise ValueError("MoGe dimensions and depth resolution must be positive")
    scale = min(1.0, max_edge / max(width, height))
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    if target_width > width or target_height > height or max(target_width, target_height) > max_edge:
        raise ValueError("MoGe preprocessing dimensions violate the single-scale contract")
    return target_height, target_width


def _split_remote_artifact(value: str) -> tuple[str, str]:
    repo_id, separator, revision = value.rpartition("@")
    if not separator or not repo_id or not revision:
        raise ValueError("Remote MoGe model must use repo_id@revision")
    return repo_id, revision
```

Local custom input may be either a `model.pt` file or a directory containing `model.pt`; any other file name fails. Remote custom input must use `repo_id@revision`. During construction, normalize these into `self.repo_id` and `self.revision`: defaults retain the registry pair, a local checkpoint uses its resolved `model.pt` path and `None`, and a custom remote uses the parsed pair. This makes startup reporting accurate without resolving weights. Hash local `model.pt` bytes with SHA-256. For defaults/remotes, call `resolve_hf_snapshot(repo_id, revision=revision)` and require `snapshot/model.pt`.

- [ ] **Step 7: Implement load and inference with strict upstream output validation**

The load path is exact:

```python
from moge.model.v2 import MoGeModel

self.model = MoGeModel.from_pretrained(str(checkpoint_path))
self.model = self.model.to(device=self.device)
self.model.eval()
```

Before loading, resolve and store the concrete checkpoint path plus `artifact_identity`. Report only classified stable identity fields:

```python
def get_model_info(self) -> dict[str, Any]:
    return {
        "family": "moge",
        "model_name": self.model_size,
        "model_version": "MoGe-2",
        "repository": self.repo_id,
        "revision": self.revision,
        "source_revision": MOGE_SOURCE_REVISION,
        "artifact_identity": self.artifact_identity,
        "metric": True,
        "device": self.device,
        "precision": self.inference_precision,
        "resolution_level": MOGE_RESOLUTION_LEVEL,
        "preprocessing_algorithm": MOGE_PREPROCESSING_ALGORITHM,
        "camera_model": self.camera_model,
        "inference_batch_size": self.max_batch_size,
    }
```

Inference sets `use_fp16 = self.device.startswith("cuda") and not fp32` and `precision = "float16" if use_fp16 else "float32"`, validates tensors before conversion, masks rejected pixels to `np.inf`, extracts only `intrinsics[:, 0, 0]`, and discards every point/normal/full-intrinsics object after returning. On `torch.cuda.OutOfMemoryError`, raise:

```python
raise RuntimeError(
    "MoGe-2 CUDA inference ran out of memory: "
    f"model_size={self.model_size}, input={width}x{height}, "
    f"resolution_level={MOGE_RESOLUTION_LEVEL}, "
    f"precision={precision}, device={self.device}"
) from error
```

`get_model_info()` reports `family`, `model_name`, `model_version`, `repository`, `revision`, `source_revision`, `artifact_identity`, `metric`, `device`, `precision`, `resolution_level`, `preprocessing_algorithm`, `camera_model`, and `inference_batch_size`. Add those identity keys to `MODEL_IDENTITY_INFO_KEYS` in Task 5 rather than allowing unclassified fields.

- [ ] **Step 8: Connect the registry factory to the selected immutable variant**

```python
def _create_moge2(request: EstimatorRequest) -> Any:
    availability = _moge_availability()
    if not availability.available:
        raise RuntimeError(
            f"{availability.reason}. Install with: {availability.install_command}"
        )
    variant = resolve_model_variant("moge2", request.model_size)
    from .video_depth_estimator_moge2 import create_video_depth_estimator_moge2

    return create_video_depth_estimator_moge2(
        model_size=variant.setting,
        model_path=request.model_path,
        repo_id=variant.repo_id,
        revision=variant.revision,
        device=request.device,
    )
```

- [ ] **Step 9: Run all inference-contract tests**

Run: `uv run pytest tests/unit/test_depth_contract.py tests/unit/test_video_depth_estimator.py tests/unit/test_video_depth_estimator_da3.py tests/unit/test_video_depth_estimator_see_through.py tests/unit/test_video_depth_estimator_moge2.py tests/unit/test_backend_registry.py -v`

Expected: PASS; existing estimators still construct camera-free `DepthBatch` values.

- [ ] **Step 10: Commit typed MoGe inference**

```bash
git add src/depth_surge_3d/inference/depth/types.py src/depth_surge_3d/inference/depth/video_depth_estimator_moge2.py src/depth_surge_3d/inference/depth/backend_registry.py src/depth_surge_3d/inference/depth/__init__.py tests/unit/test_depth_contract.py tests/unit/test_video_depth_estimator_moge2.py
git commit -m "feat: add typed MoGe-2 depth inference"
```

### Task 5: Upgrade Raw Depth to Schema v3 Without Breaking v2

**Files:**
- Modify: `src/depth_surge_3d/processing/frames/depth_storage.py`
- Modify: `src/depth_surge_3d/processing/frames/depth_processor.py`
- Modify: `src/depth_surge_3d/io/resume.py`
- Modify: `tests/unit/test_depth_storage.py`
- Modify: `tests/unit/test_depth_processor.py`
- Modify: `tests/unit/test_resume.py`

**Interfaces:**
- Consumes: `DepthBatch.camera` and existing schema-v2 `metadata.json`/NPZ layout.
- Produces: `RAW_DEPTH_SCHEMA_VERSION = 3`, `RAW_DEPTH_READABLE_SCHEMA_VERSIONS = frozenset({2, 3})`, `RawDepthStore.read_metadata(directory) -> dict[str, Any] | None`, `RawDepthStore.open_existing(...)`, `RawDepthStore.create(..., first_batch: DepthBatch)`, `write_batch(frame_names, batch: DepthBatch)`, and `load_batch(paths) -> DepthBatch`.

- [ ] **Step 1: Write failing schema compatibility tests**

```python
def test_new_depth_only_store_writes_v3_values_only(tmp_path):
    batch = DepthBatch(
        np.ones((1, 2, 3), dtype=np.float32),
        DepthRepresentation.RELATIVE_DEPTH,
    )
    store = RawDepthStore.create(
        tmp_path,
        frame_names=["frame_000001.png"],
        semantic_fingerprint={"camera_model": "none"},
        requested_dtype="float32",
        first_batch=batch,
    )
    store.write_batch(["frame_000001.png"], batch)
    assert store.metadata["schema_version"] == 3
    assert store.metadata["camera_model"] == "none"
    with zipfile.ZipFile(store.path_for("frame_000001.png")) as payload:
        assert payload.namelist() == ["values.npy"]


def test_v3_pinhole_payload_commits_depth_and_focal_together(tmp_path):
    batch = DepthBatch(
        np.ones((1, 2, 3), dtype=np.float32),
        DepthRepresentation.METRIC_DEPTH,
        camera=PinholeCameraBatch(np.array([0.8], dtype=np.float32)),
    )
    store = RawDepthStore.create(
        tmp_path,
        frame_names=["frame_000001.png"],
        semantic_fingerprint={"camera_model": "pinhole_fx"},
        requested_dtype="float32",
        first_batch=batch,
    )
    path = store.write_batch(["frame_000001.png"], batch)[0]
    with zipfile.ZipFile(path) as payload:
        assert payload.namelist() == ["values.npy", "focal_x_normalized.npy"]
    loaded = store.load_batch([path])
    assert loaded.camera is not None
    assert loaded.camera.focal_x_normalized.tolist() == pytest.approx([0.8])


def test_schema_v2_values_only_store_remains_reusable(v2_raw_store):
    before_metadata = v2_raw_store.metadata_path.read_bytes()
    before_payloads = [path.read_bytes() for path in v2_raw_store.complete_files]
    reopened = RawDepthStore.open_existing(
        v2_raw_store.directory,
        frame_names=["frame_000001.png"],
        semantic_fingerprint=v2_raw_store.metadata["semantic_fingerprint"],
        requested_dtype="float32",
    )
    assert reopened.metadata["schema_version"] == 2
    assert reopened.load_batch(reopened.complete_files).camera is None
    assert v2_raw_store.metadata_path.read_bytes() == before_metadata
    assert [path.read_bytes() for path in v2_raw_store.complete_files] == before_payloads
```

```python
@pytest.mark.parametrize(
    ("arrays", "message"),
    [
        ({"values": VALID_VALUES}, "focal_x_normalized"),
        (
            {
                "values": VALID_VALUES,
                "focal_x_normalized": VALID_FOCAL,
                "extra": np.array(1),
            },
            "exact members",
        ),
        (
            {"values": VALID_VALUES, "focal_x_normalized": np.array([0.8], np.float32)},
            "scalar",
        ),
        (
            {"values": VALID_VALUES, "focal_x_normalized": np.array(0.8, np.float64)},
            "float32",
        ),
        (
            {"values": VALID_VALUES, "focal_x_normalized": np.array(0.0, np.float32)},
            "positive",
        ),
        (
            {"values": VALID_VALUES, "focal_x_normalized": np.array(np.nan, np.float32)},
            "finite",
        ),
    ],
)
def test_corrupt_v3_pinhole_payload_is_rejected(v3_pinhole_store, arrays, message) -> None:
    overwrite_npz(v3_pinhole_store.complete_files[0], arrays)
    with pytest.raises(ValueError, match=message):
        v3_pinhole_store.validate_payloads()


def test_v3_camera_model_must_match_semantic_fingerprint(tmp_path) -> None:
    with pytest.raises(ValueError, match="camera_model"):
        RawDepthStore.create(
            tmp_path,
            frame_names=["frame_000001.png"],
            semantic_fingerprint={"camera_model": "none"},
            requested_dtype="float32",
            first_batch=metric_batch_with_focal(0.8),
        )


def test_interrupted_npz_temporary_is_not_a_committed_frame(v3_pinhole_store) -> None:
    temporary = v3_pinhole_store.directory / "frame_000002.npz.tmp"
    temporary.write_bytes(b"incomplete")
    assert v3_pinhole_store.validate_payloads() == 1
    assert not temporary.exists()
```

- [ ] **Step 2: Run storage tests and verify the new constructors are missing**

Run: `uv run pytest tests/unit/test_depth_storage.py -v`

Expected: FAIL because `RawDepthStore.create` and `open_existing` do not exist.

- [ ] **Step 3: Split open/create and make fingerprint keys schema-aware**

```python
RAW_DEPTH_SCHEMA_VERSION = 3
RAW_DEPTH_READABLE_SCHEMA_VERSIONS = frozenset({2, 3})


@staticmethod
def _fingerprint(metadata: dict[str, Any]) -> str:
    keys = [
        "schema_version",
        "representation",
        "frame_names",
        "native_shape",
        "requested_dtype",
        "selected_dtype",
        "storage_provenance",
        "compression",
        "semantic_fingerprint",
        "promoted_frame_count",
    ]
    if metadata.get("schema_version") == 3:
        keys.append("camera_model")
    return canonical_json_hash({key: metadata.get(key) for key in keys})
```

`open_existing` accepts only 2 or 3. It applies the old member rule to v2 and derives implicit `camera_model="none"` in memory without mutating metadata or payload bytes; dtype-promotion rewriting is disabled for schema v2. `create` always writes v3 and derives camera model from `first_batch.camera`; it verifies `semantic_fingerprint["camera_model"]` matches.

`read_metadata(directory)` parses only `metadata.json`, returns `None` for missing/malformed/non-mapping data, and never treats payload absence as metadata corruption. `open_existing` calls it and then performs the stricter schema/fingerprint checks.

- [ ] **Step 4: Write every frame through one exact-member atomic NPZ**

```python
def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
```

For each frame, build arrays in insertion order:

```python
arrays = {"values": frame_values.astype(dtype)}
if self.camera_model == "pinhole_fx":
    arrays["focal_x_normalized"] = np.asarray(focal_value, dtype=np.float32)
_atomic_save_npz(path, arrays)
```

The focal value is zero-dimensional. Float16 promotion rewrites `values` while reading and re-emitting the original scalar focal in the same NPZ transaction.

- [ ] **Step 5: Make model fingerprints classify camera and MoGe identity fields**

Add these exact keys to `MODEL_IDENTITY_INFO_KEYS`:

```python
"repository",
"source_revision",
"resolution_level",
"preprocessing_algorithm",
"camera_model",
```

Add `camera_model` to the semantic fingerprint before raw creation:

```python
camera_model = str(getattr(self.depth_estimator, "camera_model", "none"))
fingerprint["camera_model"] = camera_model
```

`depth_preprocessing_algorithm(settings)` returns `moge2-rgb-area-max-edge-v1` for `moge2`, the existing See-Through identity for `see_through`, and the existing default for V2/V3.

- [ ] **Step 6: Pass whole `DepthBatch` objects through the chunk writer**

Change `_infer_raw_chunk` to:

```python
if raw_store is None:
    raw_store = RawDepthStore.create(
        raw_dir,
        frame_names=frame_names,
        semantic_fingerprint=semantic_fingerprint,
        requested_dtype=requested_dtype,
        first_batch=result,
    )
else:
    raw_store.validate_batch_contract(result)
raw_store.write_batch(chunk_names, result)
```

Opening persisted data uses `RawDepthStore.open_existing`. Relative canonicalization continues to call `raw_store.load(path)` so existing downstream code sees the same float32 `[H,W]` values.

- [ ] **Step 7: Update resume validation to accept v2/v3 and require MoGe focal capability**

`_raw_mismatch_reason` accepts `RAW_DEPTH_READABLE_SCHEMA_VERSIONS`, calls schema-aware `RawDepthStore.validate_payloads()`, and rejects `moge2` metadata unless schema is 3 and `camera_model` is `pinhole_fx`. Existing backends may reuse schema 2 only when the semantic camera model is absent or `none`.

- [ ] **Step 8: Run storage, chunking, promotion, and resume tests**

Run: `uv run pytest tests/unit/test_depth_storage.py tests/unit/test_depth_processor.py tests/unit/test_resume.py -v`

Expected: PASS, including clean/chunked/resumed equality and schema-v2 reuse.

- [ ] **Step 9: Commit raw schema v3**

```bash
git add src/depth_surge_3d/processing/frames/depth_storage.py src/depth_surge_3d/processing/frames/depth_processor.py src/depth_surge_3d/io/resume.py tests/unit/test_depth_storage.py tests/unit/test_depth_processor.py tests/unit/test_resume.py
git commit -m "feat: persist typed camera data in raw depth v3"
```

### Task 6: Expose MoGe Selection in CLI and Web Without Metric Rendering Yet

**Files:**
- Modify: `depth_surge_3d.py`
- Modify: `app.py`
- Modify: `templates/index.html`
- Modify: `src/depth_surge_3d/rendering/stereo_projector.py`
- Create: `tests/unit/test_cli_moge2.py`
- Create: `tests/unit/test_web_moge2.py`

**Interfaces:**
- Consumes: registry availability/variant data and projector `model_size`.
- Produces: normalized `depth_model_version`, `model_size`, `model_path`, `depth_resolution`, and effective metric flag on both product surfaces. Geometry remains `relative` until Slice 2 lands.

- [ ] **Step 1: Write failing CLI parser and normalization tests**

```python
def test_cli_accepts_moge_and_normalizes_default_variant(cli_module) -> None:
    parser = cli_module.create_argument_parser()
    args = parser.parse_args(["clip.mp4", "--depth-model-version", "moge2"])
    settings = cli_module._build_processing_settings(args)
    assert settings["depth_model_version"] == "moge2"
    assert settings["model_size"] == "vitb"
    assert settings["model_path"] is None
    assert settings["use_metric_depth"] is True


def test_cli_model_override_and_size_are_mutually_exclusive(cli_module) -> None:
    parser = cli_module.create_argument_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "clip.mp4",
                "--depth-model-version",
                "moge2",
                "--model",
                "owner/repo@abc",
                "--model-size",
                "vits",
            ]
        )


def test_cli_missing_extra_fails_before_projector_creation(cli_module, monkeypatch) -> None:
    monkeypatch.setattr(
        cli_module,
        "backend_availability",
        lambda _backend: BackendAvailability(
            False,
            "MoGe-2 optional dependency is not installed",
            "uv sync --extra moge2",
        ),
    )
    args = cli_module.create_argument_parser().parse_args(
        ["clip.mp4", "--depth-model-version", "moge2"]
    )
    assert cli_module.validate_arguments(args) is None
```

- [ ] **Step 2: Run CLI tests and verify `moge2` is rejected by argparse**

Run: `uv run pytest tests/unit/test_cli_moge2.py -v`

Expected: FAIL because the current choices omit `moge2` and there is no `--model-size`.

- [ ] **Step 3: Add model arguments and normalize through the registry**

Use one argparse mutual-exclusion group:

```python
model_group = parser.add_mutually_exclusive_group()
model_group.add_argument("--model")
model_group.add_argument("--model-size", choices=["vits", "vitb", "vitl"], default=None)
parser.add_argument(
    "--depth-model-version",
    choices=[spec.backend_id for spec in list_backend_specs()],
    default="v2",
)
parser.add_argument("--depth-resolution", default="auto")
```

`_build_processing_settings` resolves the selected backend before calling `validate_settings`:

```python
spec = get_backend_spec(args.depth_model_version)
model_size = "custom" if args.model else args.model_size or spec.default_model_size
use_metric = True if args.depth_model_version == "moge2" else bool(args.metric)
```

Pass `model_size` to `create_stereo_projector` for new and resumed jobs. Check `backend_availability` in `validate_arguments` and print `Install with: uv sync --extra moge2` before constructing a projector.

- [ ] **Step 4: Write failing Web option tests**

```python
def test_index_marks_unavailable_moge_option_disabled(client, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.backend_availability",
        lambda backend_id: (
            BackendAvailability(False, "missing", "uv sync --extra moge2")
            if backend_id == "moge2"
            else BackendAvailability(True)
        ),
    )
    response = client.get("/")
    html = response.get_data(as_text=True)
    assert 'option value="moge2" disabled' in html
    assert "uv sync --extra moge2" in html


def test_moge_model_sizes_use_vits_vitb_vitl_values(client, monkeypatch) -> None:
    monkeypatch.setattr("app.backend_availability", lambda _backend: BackendAvailability(True))
    html = client.get("/").get_data(as_text=True)
    assert 'data-backend="moge2"' in html
    assert 'data-model-size="vits"' in html
    assert 'data-model-size="vitb"' in html
    assert 'data-model-size="vitl"' in html
```

- [ ] **Step 5: Render registry-backed backend data and remove hard-coded async dispatch**

Build a serializable list in `app.py` and pass it to the template:

```python
def _depth_backend_options() -> list[dict[str, Any]]:
    options = []
    for spec in list_backend_specs():
        availability = backend_availability(spec.backend_id)
        options.append(
            {
                "id": spec.backend_id,
                "label": spec.display_name,
                "default_model_size": spec.default_model_size,
                "variants": [asdict(value) for value in spec.variants.values()],
                "capabilities": {
                    "metric_depth": spec.capabilities.metric_depth,
                    "pinhole_fx": spec.capabilities.pinhole_fx,
                    "stereo_geometry_modes": sorted(spec.capabilities.stereo_geometry_modes),
                },
                "availability": asdict(availability),
            }
        )
    return options


@app.route("/")
def index():
    return render_template("index.html", depth_backends=_depth_backend_options())
```

Render backend options server-side so unavailable entries have the real HTML `disabled` attribute. Expose the same payload to JavaScript with Jinja `tojson` for size labels and capability-driven visibility. In `process_video_async`, delete the V2/See-Through/V3 `if/elif/else` block and call `create_stereo_projector(..., model_size=model_size)`.

- [ ] **Step 6: Make MoGe force metric inference while retaining relative stereo geometry**

When Web selects `moge2`, hide the depth-type control and send `use_metric_depth: true`. Hide V2 temporal controls. Do not show `metric_camera` yet; this task proves MoGe metric depth can flow through the existing relative canonicalization path.

- [ ] **Step 7: Run CLI, Web, projector, and settings tests**

Run: `uv run pytest tests/unit/test_cli_moge2.py tests/unit/test_web_moge2.py tests/unit/test_stereo_projector.py tests/unit/test_settings.py -v`

Expected: PASS with MoGe absent from the default environment; unavailable UI rendering does not import it.

- [ ] **Step 8: Commit Slice 1 product selection**

```bash
git add depth_surge_3d.py app.py templates/index.html src/depth_surge_3d/rendering/stereo_projector.py tests/unit/test_cli_moge2.py tests/unit/test_web_moge2.py tests/unit/test_stereo_projector.py tests/unit/test_settings.py
git commit -m "feat: expose optional MoGe-2 backend selection"
```

- [ ] **Step 9: Run the complete Slice 1 gate**

Run: `uv run pytest tests/unit/test_model_artifact.py tests/unit/test_backend_registry.py tests/unit/test_video_depth_estimator_moge2.py tests/unit/test_depth_storage.py tests/unit/test_depth_processor.py tests/unit/test_resume.py tests/unit/test_cli_moge2.py tests/unit/test_web_moge2.py -q`

Expected: PASS. `uv run python -c "import src.depth_surge_3d.inference.depth.backend_registry"` also succeeds without the `moge2` extra.

---

## Slice 2: Metric Geometry and Common Renderer Input

### Task 7: Add Metric Settings, Exact SAR Parsing, and Shared Crop Sizing

**Files:**
- Modify: `src/depth_surge_3d/core/constants.py`
- Modify: `src/depth_surge_3d/core/settings.py`
- Modify: `src/depth_surge_3d/io/operations.py`
- Modify: `src/depth_surge_3d/inference/depth/backend_registry.py`
- Modify: `src/depth_surge_3d/utils/imaging/image_processing.py`
- Modify: `tests/unit/test_settings.py`
- Modify: `tests/unit/test_io_operations.py`
- Modify: `tests/unit/test_image_processing.py`
- Modify: `tests/unit/test_backend_registry.py`

**Interfaces:**
- Consumes: backend capabilities and existing OpenCV video properties.
- Produces: settings schema v3, `parse_sample_aspect_ratio`, SAR-enriched video properties, `validate_backend_geometry_request`, and `calculate_center_crop_dimensions` used by both crop and metric projection.

- [ ] **Step 1: Write failing settings tests for exact defaults and numeric validation**

```python
def test_metric_geometry_defaults_are_explicit() -> None:
    settings = validate_settings({}, source="explicit")
    assert settings["stereo_geometry_mode"] == "relative"
    assert settings["virtual_baseline_mm"] == 63.0
    assert settings["metric_convergence_distance"] == "auto"
    assert settings["max_disparity_percent"] == 2.0


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("virtual_baseline_mm", -0.01),
        ("virtual_baseline_mm", 100.01),
        ("virtual_baseline_mm", float("nan")),
        ("metric_convergence_distance", 0.09),
        ("metric_convergence_distance", 1000.01),
        ("metric_convergence_distance", float("inf")),
        ("max_disparity_percent", -0.01),
        ("max_disparity_percent", 5.01),
    ],
)
def test_metric_settings_reject_out_of_contract_values(name, value) -> None:
    with pytest.raises(ValueError, match=name):
        validate_settings({name: value}, source="explicit")
```

- [ ] **Step 2: Write failing SAR grammar tests**

```python
@pytest.mark.parametrize("value", [None, "N/A"])
def test_missing_sar_normalizes_to_square(value) -> None:
    assert parse_sample_aspect_ratio(value) == (1, 1)


@pytest.mark.parametrize("value", ["1:1", "2:2", "2147483647:2147483647"])
def test_reducible_square_sar_is_canonicalized(value) -> None:
    assert parse_sample_aspect_ratio(value) == (1, 1)


@pytest.mark.parametrize(
    "value",
    ["0:0", "1:0", "0:1", "+1:1", "-1:1", "1:-1", "1:1x", " 1:1", "1:1 ",
     "2147483648:1", "1:2147483648", "1/1", "١:١"],
)
def test_invalid_explicit_sar_is_rejected(value) -> None:
    with pytest.raises(ValueError, match="sample_aspect_ratio"):
        parse_sample_aspect_ratio(value)


def test_non_square_explicit_sar_parses_but_metric_validation_rejects() -> None:
    assert parse_sample_aspect_ratio("4:3") == (4, 3)
    settings = validate_settings(
        {
            "depth_model_version": "moge2",
            "stereo_geometry_mode": "metric_camera",
            "vr_format": "side_by_side",
            "apply_distortion": False,
        },
        source="explicit",
    )
    with pytest.raises(ValueError, match="square-pixel"):
        validate_backend_geometry_request(
            settings,
            {"sample_aspect_ratio_numerator": 4, "sample_aspect_ratio_denominator": 3},
        )
```

- [ ] **Step 3: Run settings/SAR tests and verify the names are unknown**

Run: `uv run pytest tests/unit/test_settings.py tests/unit/test_io_operations.py tests/unit/test_backend_registry.py -v`

Expected: FAIL on unknown metric settings and missing `parse_sample_aspect_ratio`.

- [ ] **Step 4: Add schema-v3 settings with exact validators**

```python
PROCESSING_SETTINGS_SCHEMA_VERSION = 3

# constants.py DEFAULT_SETTINGS
"stereo_geometry_mode": "relative",
"virtual_baseline_mm": 63.0,
"metric_convergence_distance": "auto",
"max_disparity_percent": 2.0,
```

Use `_choice(..., {"relative", "metric_camera"})`, `_number(..., 0.0, 100.0)`, and `_number(..., 0.0, 5.0)`. Validate convergence with:

```python
def _validate_metric_convergence(value: object) -> str | float:
    if value == "auto":
        return "auto"
    return _number("metric_convergence_distance", value, 0.1, 1000.0)
```

Do not reinterpret `stereo_strength` or normalized `convergence` in metric mode.

- [ ] **Step 5: Implement the ASCII-only SAR parser and first-video-stream probe**

```python
_SAR_PATTERN = re.compile(r"([0-9]+):([0-9]+)", flags=re.ASCII)
_SAR_COMPONENT_MAX = 2_147_483_647


def parse_sample_aspect_ratio(value: object) -> tuple[int, int]:
    if value is None or value == "N/A":
        return 1, 1
    if not isinstance(value, str):
        raise ValueError("sample_aspect_ratio must be an unsigned numerator:denominator")
    match = _SAR_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError("sample_aspect_ratio must be an unsigned numerator:denominator")
    numerator, denominator = (int(part) for part in match.groups())
    if not 1 <= numerator <= _SAR_COMPONENT_MAX or not 1 <= denominator <= _SAR_COMPONENT_MAX:
        raise ValueError("sample_aspect_ratio components must be in 1..2147483647")
    divisor = math.gcd(numerator, denominator)
    return numerator // divisor, denominator // divisor
```

After OpenCV properties succeed, `get_video_properties` calls `get_video_info_ffprobe`, chooses the first stream whose `codec_type` is `video`, parses its `sample_aspect_ratio`, and stores:

```python
"sample_aspect_ratio_numerator": numerator,
"sample_aspect_ratio_denominator": denominator,
"sample_aspect_ratio": f"{numerator}:{denominator}",
```

An absent stream, absent field, `N/A`, or unavailable ffprobe becomes `1:1`. An explicit malformed field propagates `ValueError`; it is not swallowed as missing.

- [ ] **Step 6: Implement cross-field capability validation in one registry function**

```python
def validate_backend_geometry_request(settings, video_properties) -> None:
    backend_id = str(settings.get("depth_model_version", "v2"))
    spec = get_backend_spec(backend_id)
    mode = cast(StereoGeometryMode, settings.get("stereo_geometry_mode", "relative"))
    if mode not in spec.capabilities.stereo_geometry_modes:
        raise ValueError(f"{backend_id} does not support stereo geometry mode {mode}")
    if mode == "metric_camera":
        if not spec.capabilities.pinhole_fx:
            raise ValueError("metric_camera requires pinhole_fx camera output")
        if settings.get("vr_format") != "side_by_side":
            raise ValueError("metric_camera requires vr_format=side_by_side")
        if settings.get("apply_distortion") is not False:
            raise ValueError("metric_camera requires apply_distortion=false")
        numerator = video_properties.get("sample_aspect_ratio_numerator")
        denominator = video_properties.get("sample_aspect_ratio_denominator")
        if numerator is None or denominator is None:
            raise ValueError("metric_camera requires source sample-aspect-ratio metadata")
        if (numerator, denominator) != (1, 1):
            raise ValueError("metric_camera requires square-pixel source sample_aspect_ratio=1:1")
```

- [ ] **Step 7: Make crop sizing one integer algorithm used by projection and pixels**

```python
CENTER_CROP_ALGORITHM_VERSION = "integer-center-crop-v1"


def calculate_center_crop_dimensions(
    width: int, height: int, crop_factor: float
) -> tuple[int, int]:
    if width < 1 or height < 1:
        raise ValueError("Image dimensions must be positive")
    if crop_factor >= 1.0:
        return width, height
    return max(1, int(width * crop_factor)), max(1, int(height * crop_factor))
```

Refactor `apply_center_crop` to use this helper and retain its existing start-index rule `(size - crop_size) // 2`:

```python
def test_center_crop_pixels_and_metric_width_share_odd_integer_rounding() -> None:
    image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    crop_width, crop_height = calculate_center_crop_dimensions(7, 5, 0.5)
    cropped = apply_center_crop(image, 0.5)
    assert (crop_width, crop_height) == (3, 2)
    assert cropped.shape[:2] == (crop_height, crop_width)
    np.testing.assert_array_equal(cropped, image[1:3, 2:5])
```

- [ ] **Step 8: Run the focused contract suite**

Run: `uv run pytest tests/unit/test_settings.py tests/unit/test_io_operations.py tests/unit/test_image_processing.py tests/unit/test_backend_registry.py -v`

Expected: PASS.

- [ ] **Step 9: Commit settings, SAR, and crop contracts**

```bash
git add src/depth_surge_3d/core/constants.py src/depth_surge_3d/core/settings.py src/depth_surge_3d/io/operations.py src/depth_surge_3d/inference/depth/backend_registry.py src/depth_surge_3d/utils/imaging/image_processing.py tests/unit/test_settings.py tests/unit/test_io_operations.py tests/unit/test_image_processing.py tests/unit/test_backend_registry.py
git commit -m "feat: validate flat metric camera requests"
```

### Task 8: Generalize the Renderer Around `StereoGeometryFrame`

**Files:**
- Create: `src/depth_surge_3d/rendering/stereo_geometry.py`
- Modify: `src/depth_surge_3d/rendering/forward_splat.py`
- Modify: `src/depth_surge_3d/rendering/stereo_renderer.py`
- Create: `tests/unit/test_stereo_geometry.py`
- Modify: `tests/unit/test_forward_splat.py`
- Modify: `tests/unit/test_stereo_renderer.py`

**Interfaces:**
- Consumes: frozen relative hash from Task 1.
- Produces: `StereoGeometryFrame`, `build_relative_geometry`, backend-neutral `StereoRenderer.render_geometry`, and explicit `source_valid` masking in the packed z-buffer.

- [ ] **Step 1: Write failing common-geometry validation and relative formula tests**

```python
def test_geometry_frame_requires_exact_dtypes_and_shapes() -> None:
    with pytest.raises(TypeError, match="near_score.*float32"):
        StereoGeometryFrame(
            np.ones((2, 3), dtype=np.float64),
            np.zeros((2, 3), dtype=np.float64),
            np.ones((2, 3), dtype=np.bool_),
        )
    with pytest.raises(ValueError, match="same shape"):
        StereoGeometryFrame(
            np.ones((2, 3), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float64),
            np.ones((2, 3), dtype=np.bool_),
        )


def test_relative_builder_uses_existing_total_disparity_formula() -> None:
    geometry = build_relative_geometry(
        np.array([[0.0, 0.5, 1.0]], dtype=np.float32),
        (1, 3),
        stereo_strength=2.0,
        convergence=0.5,
    )
    np.testing.assert_allclose(
        geometry.total_disparity_fraction,
        np.array([[-0.01, 0.0, 0.01]], dtype=np.float64),
        rtol=0.0,
        atol=2e-18,
    )
    assert geometry.source_valid.all()
```

- [ ] **Step 2: Write failing source-validity and nonnegative-depth tests for splatting**

```python
def test_invalid_nearer_source_cannot_win_or_fill() -> None:
    image = torch.tensor([[[255.0, 0.0, 0.0], [0.0, 0.0, 255.0]]])
    near_score = torch.tensor([[10.0, 1.0]], dtype=torch.float32)
    valid = torch.tensor([[False, True]])
    result = forward_splat_band(
        image,
        near_score,
        _offsets([[HORIZONTAL_SUBPIXELS, 0]]),
        source_valid=valid,
    )
    torch.testing.assert_close(
        result.colour[0, HORIZONTAL_SUBPIXELS:],
        image[0, 1].expand(HORIZONTAL_SUBPIXELS, 3),
    )


def test_near_score_may_exceed_one_but_must_be_finite_and_nonnegative() -> None:
    image = torch.zeros((1, 1, 3))
    offsets = _offsets([[0]])
    assert forward_splat_band(image, torch.tensor([[12.0]]), offsets).valid.all()
    with pytest.raises(ValueError, match="nonnegative"):
        forward_splat_band(image, torch.tensor([[-0.1]]), offsets)
```

- [ ] **Step 3: Run new tests and verify the common type/signature is missing**

Run: `uv run pytest tests/unit/test_stereo_geometry.py tests/unit/test_forward_splat.py -v`

Expected: FAIL on missing `StereoGeometryFrame` and unsupported `source_valid`.

- [ ] **Step 4: Implement immutable geometry and the exact relative resize path**

`StereoGeometryFrame.__post_init__` requires 2D equal shapes, `float32` finite nonnegative near score, `float64` finite disparity fraction, and Boolean validity. `build_relative_geometry` uses the current Torch bilinear resize with `align_corners=False`, then computes float64 total disparity:

```python
resized = _resize_float32_bilinear(canonical, render_shape)
near64 = np.asarray(resized, dtype=np.float64)
total = near64 - np.float64(convergence)
total *= np.float64(stereo_strength)
total /= np.float64(100.0)
return StereoGeometryFrame(
    near_score=np.ascontiguousarray(resized, dtype=np.float32),
    total_disparity_fraction=np.ascontiguousarray(total, dtype=np.float64),
    source_valid=np.ones(render_shape, dtype=np.bool_),
)
```

- [ ] **Step 5: Generalize packed winners without changing tie order**

Rename internal `canonical` variables to `near_score`, remove the upper-bound-one check, and add optional compatibility input `source_valid: torch.Tensor | None = None`. If omitted, create all-true validity. Candidate validity is:

```python
candidate_valid = in_bounds & source_valid.unsqueeze(-1)
candidate_keys = candidate_keys.masked_fill(~candidate_valid, -1)
```

Keep positive-zero normalization, float32 bit packing, inverted full-frame source index, `scatter_reduce_(reduce="amax")`, and `SubpixelSplatResult.disparity` unchanged. The latter remains the near score used by background fill; retaining the field name avoids unrelated result-schema churn.

- [ ] **Step 6: Add a backend-neutral renderer entry point while keeping the relative wrapper**

```python
@dataclass(frozen=True)
class StereoSplatSettings:
    max_eye_shift_fraction: float
    occlusion_fill: Literal["none", "background"] = "background"


def calculate_geometry_eye_sample_offsets(
    total_disparity_fraction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    width = total_disparity_fraction.shape[1]
    fine_shift = np.asarray(total_disparity_fraction, dtype=np.float64)
    fine_shift = fine_shift * np.float64(width) * np.float64(0.5)
    fine_shift = fine_shift * np.float64(HORIZONTAL_SUBPIXELS)
    return (
        _narrow_sample_offsets(np.ceil(fine_shift - np.float64(0.5))),
        _narrow_sample_offsets(np.ceil(-fine_shift - np.float64(0.5))),
    )
```

`StereoRenderer.render_geometry(frame, geometry, StereoSplatSettings)` validates that geometry already matches the source raster, computes offsets once, and passes `geometry.near_score` plus `geometry.source_valid` to every band. Fill width is `ceil(width * max_eye_shift_fraction) + 2`.

Human-approved compatibility exception (2026-08-16): retain `StereoRenderer.render(frame, canonical, StereoRenderSettings)` as a relative-only compatibility wrapper that calls `build_relative_geometry`, computes eye offsets with the legacy `calculate_eye_sample_offsets(canonical, settings)` float64 operation order, and delegates those offsets to the same private splat core used by `render_geometry`, with `max_eye_shift_fraction=stereo_strength/200`. It must not delegate offset calculation through public `render_geometry`: the common formula's required operation order differs at legal half-lane boundaries. `render_geometry` continues to use `calculate_geometry_eye_sample_offsets` exactly as specified above; do not expose legacy arithmetic through common/metric geometry fields or the public geometry path.

- [ ] **Step 7: Run common renderer tests and the frozen hash**

Run: `uv run pytest tests/unit/test_stereo_geometry.py tests/unit/test_forward_splat.py tests/unit/test_stereo_renderer.py -v`

Expected: PASS, including `test_relative_cpu_regression_corpus_is_byte_frozen`. Do not update its hashes.

- [ ] **Step 8: Run independent scalar edge-reference tests**

Run: `uv run pytest tests/unit/test_stereo_edge_coverage.py tests/unit/test_verify_stereo_edge_fixture.py -v`

Expected: PASS with unchanged tie, fill, and rounding behavior.

- [ ] **Step 9: Commit the common renderer refactor**

```bash
git add src/depth_surge_3d/rendering/stereo_geometry.py src/depth_surge_3d/rendering/forward_splat.py src/depth_surge_3d/rendering/stereo_renderer.py tests/unit/test_stereo_geometry.py tests/unit/test_forward_splat.py tests/unit/test_stereo_renderer.py
git commit -m "refactor: feed stereo renderer common geometry"
```

### Task 9: Build Restartable Metric Geometry Storage and Auto Convergence

**Files:**
- Create: `src/depth_surge_3d/processing/frames/metric_geometry.py`
- Create: `tests/unit/test_metric_geometry.py`

**Interfaces:**
- Consumes: raw schema-v3 `RawDepthStore.load_batch`, candidate scene IDs, and source/raw fingerprints.
- Produces: all metric-store interfaces in **Shared Interfaces**, `METRIC_GEOMETRY_SCHEMA_VERSION = 1`, `METRIC_GEOMETRY_ALGORITHM_VERSION = "metric-inverse-depth-v1"`, and `METRIC_CONVERGENCE_ALGORITHM_VERSION = "clip-scene-grid-median-v1"`.

- [ ] **Step 1: Write failing metric-frame and reciprocal-validity tests**

```python
def test_metric_frame_derives_finite_inverse_depth_and_explicit_validity() -> None:
    smallest_subnormal = np.nextafter(
        np.float32(0.0), np.float32(1.0), dtype=np.float32
    )
    depth = np.array([[2.0, 0.0, np.inf], [4.0, -1.0, smallest_subnormal]], dtype=np.float32)
    frame = metric_frame_from_depth(depth, np.float32(0.8))
    assert frame.valid.tolist() == [[True, False, False], [True, False, False]]
    np.testing.assert_array_equal(
        frame.inverse_depth,
        np.array([[0.5, 0.0, 0.0], [0.25, 0.0, 0.0]], dtype=np.float32),
    )


def test_metric_geometry_frame_rejects_nonzero_invalid_scores() -> None:
    with pytest.raises(ValueError, match="invalid locations"):
        MetricGeometryFrame(
            inverse_depth=np.ones((1, 1), dtype=np.float32),
            valid=np.zeros((1, 1), dtype=np.bool_),
            focal_x_normalized=np.float32(0.8),
        )
```

- [ ] **Step 2: Write failing convergence-sampling tests**

```python
def test_clip_convergence_is_one_source_ordered_float32_median(raw_metric_store) -> None:
    result = sample_clip_convergence(
        raw_metric_store,
        raw_metric_store.complete_files,
        candidate_scene_ids=[0, 0, 1, 1],
    )
    assert result.distance_m.dtype == np.float32
    assert result.distance_m == np.float32(5.0)
    assert result.selected_frame_indexes == (0, 1, 2, 3)
    assert result.sample_count == 4


def test_convergence_selects_at_most_32_frames_per_candidate_scene() -> None:
    scene_ids = [0] * 100 + [1] * 100
    indexes = select_convergence_frame_indexes(scene_ids)
    assert len(indexes) == 64
    assert indexes == tuple(sorted(indexes))
    assert sum(index < 100 for index in indexes) == 32


def test_no_valid_metric_sample_is_a_hard_error(raw_store_with_only_invalid_depth) -> None:
    with pytest.raises(ValueError, match="No valid positive metric depth"):
        sample_clip_convergence(
            raw_store_with_only_invalid_depth,
            raw_store_with_only_invalid_depth.complete_files,
            candidate_scene_ids=[0],
        )


def test_metric_store_rejects_changed_candidate_scene_identity(complete_metric_store) -> None:
    with pytest.raises(ValueError, match="candidate scene fingerprint"):
        MetricGeometryStore.open_existing(
            complete_metric_store.directory,
            frame_names=complete_metric_store.frame_names,
            source_raw_fingerprint=complete_metric_store.source_raw_fingerprint,
            source_frame_fingerprint=complete_metric_store.source_frame_fingerprint,
            candidate_scene_fingerprint="changed",
        )
```

The `raw_metric_store` fixture uses four 1x1 depths `[2, 4, 6, 8]`, whose median is `5.0`.

- [ ] **Step 3: Write failing disk-bound and ENOSPC tests**

```python
def test_metric_disk_bound_uses_exact_uncompressed_formula() -> None:
    assert estimate_metric_geometry_disk_bytes(
        [(2, 3), (4, 5)], allocation_unit=4096
    ) == 16_781_600


def test_allocation_unit_falls_back_to_64_kib(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("os.statvfs", lambda _path: (_ for _ in ()).throw(OSError("no statvfs")), raising=False)
    monkeypatch.setattr(metric_geometry, "_windows_allocation_unit", lambda _path: None)
    assert filesystem_allocation_unit(tmp_path) == 65_536


def test_enospc_removes_only_current_temp_and_keeps_committed_frames(
    monkeypatch, partial_metric_store
) -> None:
    committed = partial_metric_store.path_for("frame_000001.png")
    failing = partial_metric_store.path_for("frame_000002.png")
    real_replace = Path.replace

    def fail_second_replace(path, destination):
        if Path(destination) == failing:
            raise OSError(errno.ENOSPC, "disk full")
        return real_replace(path, destination)

    monkeypatch.setattr(Path, "replace", fail_second_replace)
    with pytest.raises(MetricGeometryDiskFullError) as raised:
        partial_metric_store.write_frame("frame_000002.png", VALID_FRAME)
    assert committed.is_file()
    assert not failing.is_file()
    assert not list(partial_metric_store.directory.glob("*.tmp"))
    assert partial_metric_store.metadata.get("status") == "writing"
    assert "fingerprint" not in partial_metric_store.metadata
    assert raised.value.required_bytes > 0
    assert raised.value.failing_path == failing
```

- [ ] **Step 4: Run metric storage tests and verify the module is absent**

Run: `uv run pytest tests/unit/test_metric_geometry.py -v`

Expected: FAIL during collection with `ModuleNotFoundError` for `metric_geometry`.

- [ ] **Step 5: Implement deterministic sampling with no reciprocal overflow**

Compute validity in two stages so overflow cannot enter z-order:

```python
valid = np.isfinite(depth) & (depth > np.float32(0.0))
inverse = np.zeros(depth.shape, dtype=np.float32)
np.divide(np.float32(1.0), depth, out=inverse, where=valid)
valid &= np.isfinite(inverse)
inverse[~valid] = np.float32(0.0)
```

Select frames exactly:

```python
def select_convergence_frame_indexes(scene_ids: Sequence[int]) -> tuple[int, ...]:
    selected: list[int] = []
    for scene_id in dict.fromkeys(int(value) for value in scene_ids):
        source_indexes = np.flatnonzero(np.asarray(scene_ids) == scene_id)
        if len(source_indexes) > 32:
            positions = np.unique(
                np.rint(np.linspace(0, len(source_indexes) - 1, 32)).astype(np.int64)
            )
            source_indexes = source_indexes[positions]
        selected.extend(int(value) for value in source_indexes)
    return tuple(sorted(selected))
```

For each selected frame, choose row and column indexes with `np.rint(np.linspace(0, size - 1, min(size, 64))).astype(np.int64)`, use their Cartesian grid, retain finite positive metric depths in row-major order, concatenate frames in source order, and call `np.float32(np.median(samples.astype(np.float32, copy=False)))`.

- [ ] **Step 6: Implement exact allocation and disk arithmetic**

```python
def _ceil_five_quarters(value: int) -> int:
    return (5 * value + 3) // 4


def estimate_metric_geometry_disk_bytes(frame_shapes, *, allocation_unit):
    if not frame_shapes:
        return 0
    allocation = max(4096, int(allocation_unit))
    frame_payloads = [5 * height * width for height, width in frame_shapes]
    payload_bound = sum(frame_payloads)
    metadata_bound = max(16 * 1024 * 1024, allocation * len(frame_shapes))
    atomic_overlap = _ceil_five_quarters(max(frame_payloads)) + allocation
    return _ceil_five_quarters(payload_bound) + metadata_bound + atomic_overlap
```

Use `os.statvfs(...).f_frsize` on POSIX and `GetDiskFreeSpaceW` through `ctypes` on Windows. Return 65,536 only when the platform query is unavailable or invalid.

- [ ] **Step 7: Implement restartable exact-member storage**

Initial metadata has `status="writing"`, ordered frame names, native shape, source fingerprints, `candidate_scene_fingerprint`, `preflight_required_bytes`, representation, near direction, storage dtype, compression, and no completion fingerprint. `open_or_create` validates that a resumed call supplies the same scene identity and the same or a larger current preflight estimate, then updates the operational estimate atomically before new writes. Each NPZ contains exactly `inverse_depth.npy`, `valid.npy`, and `focal_x_normalized.npy`; header validation requires float32 2D, bool 2D, and float32 scalar.

`finalize(convergence)` first validates every payload, then atomically writes metadata containing:

```python
{
    "status": "complete",
    "convergence": {
        "algorithm_version": METRIC_CONVERGENCE_ALGORITHM_VERSION,
        "selected_frame_indexes": list(convergence.selected_frame_indexes),
        "sample_count": convergence.sample_count,
        "resolved_auto_distance_m": float(convergence.distance_m),
    },
    "fingerprint": canonical_json_hash(metadata_without_fingerprint),
}
```

The completion transaction removes the operational `preflight_required_bytes` field before hashing; it is needed only while writes can still fail and is not part of geometry identity.

`open_existing` accepts only complete metadata, validates the metadata fingerprint plus expected ordered frame names and source fingerprints, and exposes ordered `complete_files` only after every payload validates. `open_or_create` resumes writing metadata or returns the same validated complete store. Treat `error.errno == errno.ENOSPC` and Windows `error.winerror == 112` (`ERROR_DISK_FULL`) as the same condition. `MetricGeometryDiskFullError` reads `required_bytes` from the persisted current preflight field, queries current free bytes at the failure point, and carries both plus `failing_path` in fields and message. Every other `OSError` is re-raised unchanged.

- [ ] **Step 8: Run metric storage tests**

Run: `uv run pytest tests/unit/test_metric_geometry.py -v`

Expected: PASS, including exact 16,781,600-byte arithmetic and post-preflight ENOSPC recovery.

- [ ] **Step 9: Commit the independent metric stage implementation**

```bash
git add src/depth_surge_3d/processing/frames/metric_geometry.py tests/unit/test_metric_geometry.py
git commit -m "feat: add restartable metric geometry storage"
```

### Task 10: Derive Only the Selected Stage 3 in `DepthMapProcessor`

**Files:**
- Modify: `src/depth_surge_3d/core/constants.py`
- Modify: `src/depth_surge_3d/processing/frames/depth_storage.py`
- Modify: `src/depth_surge_3d/processing/frames/depth_processor.py`
- Modify: `src/depth_surge_3d/processing/orchestration/pipeline_orchestrator.py`
- Modify: `src/depth_surge_3d/processing/video/video_encoder.py`
- Modify: `tests/unit/test_depth_processor.py`
- Modify: `tests/unit/test_depth_storage.py`
- Modify: `tests/unit/test_pipeline_orchestrator.py`

**Interfaces:**
- Consumes: complete raw store, candidate scene manifest, `MetricGeometryStore`, and active `stereo_geometry_mode`.
- Produces: `generate_depth_map_files(...)` returning files from exactly one active stage: relative PNGs or metric NPZs.

- [ ] **Step 1: Write failing active-stage and retention tests**

```python
def test_metric_mode_builds_only_metric_geometry(
    tmp_path, frame_files, fake_moge_estimator, default_settings
) -> None:
    directories = make_depth_directories(tmp_path)
    settings = {
        **default_settings,
        "depth_model_version": "moge2",
        "stereo_geometry_mode": "metric_camera",
        "keep_intermediates": True,
    }
    files = DepthMapProcessor(fake_moge_estimator).generate_depth_map_files(
        frame_files, settings, directories, progress_tracker=None
    )
    assert files is not None
    assert all(path.parent == directories["metric_geometry"] for path in files)
    assert all(path.suffix == ".npz" for path in files)
    assert not list(directories["disparity_maps"].glob("*.png"))


def test_switching_to_relative_preserves_valid_metric_stage(
    tmp_path, frame_files, fake_moge_estimator, default_settings
) -> None:
    directories = make_depth_directories(tmp_path)
    processor = DepthMapProcessor(fake_moge_estimator)
    metric_settings = {
        **default_settings,
        "depth_model_version": "moge2",
        "stereo_geometry_mode": "metric_camera",
        "keep_intermediates": True,
    }
    metric_files = processor.generate_depth_map_files(
        frame_files, metric_settings, directories, None
    )
    metric_bytes = [path.read_bytes() for path in metric_files]
    relative_files = processor.generate_depth_map_files(
        frame_files,
        {**metric_settings, "stereo_geometry_mode": "relative"},
        directories,
        None,
    )
    assert all(path.parent == directories["disparity_maps"] for path in relative_files)
    assert [path.read_bytes() for path in metric_files] == metric_bytes


def test_metric_preflight_sums_raw_and_exact_selected_stage_before_inference(
    tmp_path, frame_files, fake_moge_estimator, default_settings, monkeypatch
) -> None:
    events: list[tuple[str, int | None]] = []
    monkeypatch.setattr(
        depth_processor,
        "require_disk_space",
        lambda _path, required: events.append(("preflight", required)),
    )
    fake_moge_estimator.on_inference = lambda: events.append(("inference", None))
    settings = {
        **default_settings,
        "depth_model_version": "moge2",
        "stereo_geometry_mode": "metric_camera",
        "keep_intermediates": True,
    }
    DepthMapProcessor(fake_moge_estimator).generate_depth_map_files(
        frame_files, settings, make_depth_directories(tmp_path), None
    )
    native_shape = fake_moge_estimator.estimate_output_shape(6, 8, 8)
    expected = estimate_raw_depth_only_bytes(
        frame_count=len(frame_files),
        native_width=native_shape[1],
        native_height=native_shape[0],
        storage_bytes=4,
        camera_bytes_per_frame=4,
    ) + estimate_metric_geometry_disk_bytes(
        [native_shape] * len(frame_files),
        allocation_unit=filesystem_allocation_unit(tmp_path),
    )
    assert events[0] == ("preflight", expected)
    assert next(index for index, event in enumerate(events) if event[0] == "inference") > 0
```

```python
@pytest.mark.parametrize(
    "change",
    [
        {"virtual_baseline_mm": 70.0},
        {"metric_convergence_distance": 3.0},
        {"max_disparity_percent": 1.0},
    ],
)
def test_metric_render_setting_changes_preserve_stage3_hashes(metric_job, change) -> None:
    before = hash_directory(metric_job.directories["metric_geometry"])
    files = metric_job.processor.generate_depth_map_files(
        metric_job.frame_files,
        {**metric_job.settings, **change},
        metric_job.directories,
        None,
    )
    assert files
    assert hash_directory(metric_job.directories["metric_geometry"]) == before
    assert metric_job.estimator.calls == metric_job.calls_after_first_run


def test_complete_metric_stage_reuse_makes_zero_estimator_calls(metric_job) -> None:
    metric_job.estimator.estimate_depth_batch = Mock(
        side_effect=AssertionError("inference must not run")
    )
    assert metric_job.processor.generate_depth_map_files(
        metric_job.frame_files,
        metric_job.settings,
        metric_job.directories,
        None,
    )


def test_no_retention_removes_raw_only_after_metric_finalize(
    fresh_metric_job, monkeypatch
) -> None:
    events: list[str] = []
    real_finalize = MetricGeometryStore.finalize
    real_remove = fresh_metric_job.processor._remove_raw_payloads

    def recording_finalize(store, convergence):
        result = real_finalize(store, convergence)
        events.append("finalized")
        return result

    def recording_remove(store):
        events.append("remove_raw")
        return real_remove(store)

    monkeypatch.setattr(MetricGeometryStore, "finalize", recording_finalize)
    monkeypatch.setattr(fresh_metric_job.processor, "_remove_raw_payloads", recording_remove)
    fresh_metric_job.run(keep_intermediates=False)
    assert events == ["finalized", "remove_raw"]
```

- [ ] **Step 2: Run the focused processor tests and verify the directory key is missing**

Run: `uv run pytest tests/unit/test_depth_processor.py -v`

Expected: FAIL because `metric_geometry` is absent from `INTERMEDIATE_DIRS` and the processor always returns canonical PNGs.

- [ ] **Step 3: Add the directory and make invalid upstream identities clear both derived formats**

```python
INTERMEDIATE_DIRS = {
    # existing entries
    "disparity_maps": "03_disparity_maps",
    "metric_geometry": "03_metric_geometry",
    # existing entries
}
```

Scene/source/raw semantic mismatch resets `02_depth_raw`, `03_disparity_maps`, and `03_metric_geometry`. A mere mode switch resets neither inactive stage.

- [ ] **Step 4: Extract one common raw barrier while leaving relative derivation byte-stable**

Probe the selected stage before requiring raw payloads:

```python
def _reusable_selected_geometry_files(
    self,
    *,
    settings: Mapping[str, Any],
    directories: Mapping[str, Path],
    frame_names: Sequence[str],
    source_frame_fingerprint: str,
    semantic_fingerprint: Mapping[str, Any],
    candidate_scene_fingerprint: str,
) -> list[Path] | None:
    if settings["stereo_geometry_mode"] != "metric_camera":
        return None
    raw_metadata = RawDepthStore.read_metadata(directories["depth_raw"])
    if raw_metadata is None or raw_metadata.get("storage_status") != "ready":
        return None
    persisted_semantic = raw_metadata.get("semantic_fingerprint")
    if not isinstance(persisted_semantic, dict):
        return None
    if canonical_json_hash(persisted_semantic) != canonical_json_hash(semantic_fingerprint):
        return None
    try:
        store = MetricGeometryStore.open_existing(
            directories["metric_geometry"],
            frame_names=list(frame_names),
            source_frame_fingerprint=source_frame_fingerprint,
            source_raw_fingerprint=str(raw_metadata["fingerprint"]),
            candidate_scene_fingerprint=candidate_scene_fingerprint,
        )
    except (OSError, KeyError, TypeError, ValueError):
        return None
    return list(store.complete_files)
```

This validation reads ready raw metadata but does not require raw NPZ payloads. `_remove_raw_payloads` deletes only frame NPZs and preserves ready raw metadata, so an already validated selected metric stage remains reusable after no-retention stage-3 cleanup. Relative mode continues through its existing local/global canonical reuse path. Only when the selected-mode reuse paths return `None` does the processor introduce this raw context:

```python
@dataclass(frozen=True)
class _RawStageContext:
    scene_dir: Path
    raw_dir: Path
    canonical_dir: Path
    metric_dir: Path
    manifest: dict[str, Any]
    raw_store: RawDepthStore
    raw_files: tuple[Path, ...]
    frame_names: tuple[str, ...]
    semantic_fingerprint: dict[str, Any]
    candidate_scene_fingerprint: str
```

`_prepare_raw_stage(...) -> _RawStageContext` opens or completes raw depth after scene analysis and does not derive either stage 3. It derives the scene identity once:

```python
candidate_manifest = self._candidate_manifest(manifest)
candidate_scene_fingerprint = canonical_json_hash(
    {
        "schema_version": candidate_manifest["schema_version"],
        "algorithm_version": candidate_manifest["algorithm_version"],
        "frame_names": candidate_manifest["frame_names"],
        "scene_ids": candidate_manifest["scene_ids"],
    }
)
```

Pass that value to both `_reusable_selected_geometry_files` and `MetricGeometryStore.open_or_create`. A candidate-scene assignment change therefore rebuilds metric geometry/convergence but not raw inference; a threshold edit that produces identical source-ordered scene IDs may reuse it. Keep the existing global canonical cache lookup and scene finalization inside the relative branch, so metric mode never restores or writes a relative global cache entry.

Extract the existing raw allowance without changing its `1.25` payload arithmetic:

```python
def estimate_raw_depth_only_bytes(
    *,
    frame_count: int,
    native_width: int,
    native_height: int,
    storage_bytes: int,
    camera_bytes_per_frame: int,
) -> int:
    if frame_count < 0 or native_width < 1 or native_height < 1 or storage_bytes < 1:
        raise ValueError("Raw frame count must be nonnegative and dimensions/storage positive")
    if camera_bytes_per_frame < 0:
        raise ValueError("Raw camera bytes must be nonnegative")
    payload = frame_count * (
        native_width * native_height * storage_bytes + camera_bytes_per_frame
    )
    return (5 * payload + 3) // 4
```

Relative mode continues to call `estimate_depth_disk_bytes` unchanged. Before the first metric estimator call, preflight one sum: prospective missing raw bytes plus `estimate_metric_geometry_disk_bytes` for the selected stage, using the estimated native output shape and target allocation unit. Repeat after the first committed raw batch if actual shape or selected dtype differs. Already committed raw or inactive-stage files consume real free space and are not added again. This prevents a job from passing raw preflight only to discover before stage 3 that the selected metric stage could never fit.

- [ ] **Step 5: Implement metric derivation from typed raw payloads**

```python
def _write_metric_geometry_stage(
    self,
    context: _RawStageContext,
    settings: dict[str, Any],
    progress_tracker,
) -> list[Path]:
    native_shape = tuple(int(value) for value in context.raw_store.metadata["native_shape"])
    required = estimate_metric_geometry_disk_bytes(
        [native_shape] * len(context.frame_names),
        allocation_unit=filesystem_allocation_unit(context.metric_dir),
    )
    require_metric_geometry_disk_space(context.metric_dir, required)
    store = MetricGeometryStore.open_or_create(
        context.metric_dir,
        frame_names=list(context.frame_names),
        native_shape=cast(tuple[int, int], native_shape),
        source_raw_fingerprint=str(context.raw_store.metadata["fingerprint"]),
        source_frame_fingerprint=str(
            context.semantic_fingerprint["source_frame_fingerprint"]
        ),
        candidate_scene_fingerprint=context.candidate_scene_fingerprint,
        preflight_required_bytes=required,
    )
    for index, (name, raw_path) in enumerate(zip(context.frame_names, context.raw_files)):
        output = store.path_for(name)
        if not output.is_file():
            batch = context.raw_store.load_batch([raw_path])
            if batch.representation is not DepthRepresentation.METRIC_DEPTH or batch.camera is None:
                raise ValueError("metric_camera requires metric raw depth with pinhole focal data")
            frame = metric_frame_from_depth(
                batch.values[0], batch.camera.focal_x_normalized[0]
            )
            store.write_frame(name, frame)
        self._report_metric_progress(progress_tracker, index, len(context.frame_names), output)
    convergence = sample_clip_convergence(
        context.raw_store,
        context.raw_files,
        self._candidate_manifest(context.manifest)["scene_ids"],
    )
    store.finalize(convergence)
    return [store.path_for(name) for name in context.frame_names]
```

Resolve and persist automatic convergence on every metric-stage build, including runs whose active render setting is explicit. The stage fingerprint excludes baseline, explicit convergence, cap, crop, source render width, and final output width.

- [ ] **Step 6: Keep metric depth previews visual-only and ordered before stereo**

When sampled by `PREVIEW_FRAME_SAMPLE_RATE`, load `inverse_depth` and form a uint8 preview using finite valid min/max normalization. Send it with `send_preview_frame_from_array(..., "depth_map", index + 1)`. Do not call the stereo builder and do not emit `stereo_left` from stage 3.

- [ ] **Step 7: Make no-retention cleanup transactional at the selected-stage barrier**

Only call `_remove_raw_payloads(raw_store)` after either canonical metadata or metric metadata passes its final validation. Do not delete the inactive stage during switching. The pipeline's successful final cleanup remains responsible for removing every intermediate directory.

- [ ] **Step 8: Update orchestration labels and downstream-progress detection**

Use `geometry_files` rather than `depth_files` locally. Print `Prepared N canonical disparity maps` for relative and `Prepared N metric geometry frames` for metric. Update `_has_downstream_progress` to count both `*.png` and `*.npz` stage payloads so a completed metric stage proves extraction advanced.

- [ ] **Step 9: Run processor and orchestration tests**

Run: `uv run pytest tests/unit/test_depth_processor.py tests/unit/test_pipeline_orchestrator.py tests/unit/test_video_encoder.py -v`

Expected: PASS; clean, chunked, and resumed metric derivation yields identical NPZ arrays and final metadata.

- [ ] **Step 10: Commit selected stage-3 derivation**

```bash
git add src/depth_surge_3d/core/constants.py src/depth_surge_3d/processing/frames/depth_storage.py src/depth_surge_3d/processing/frames/depth_processor.py src/depth_surge_3d/processing/orchestration/pipeline_orchestrator.py src/depth_surge_3d/processing/video/video_encoder.py tests/unit/test_depth_storage.py tests/unit/test_depth_processor.py tests/unit/test_pipeline_orchestrator.py tests/unit/test_video_encoder.py
git commit -m "feat: derive selected metric geometry stage"
```

### Task 11: Implement Crop-Aware Metric Projection and Render It Through the Common Pipeline

**Files:**
- Modify: `src/depth_surge_3d/rendering/stereo_geometry.py`
- Modify: `src/depth_surge_3d/processing/frames/stereo_generator.py`
- Modify: `src/depth_surge_3d/rendering/stereo_projector.py`
- Modify: `tests/unit/test_stereo_geometry.py`
- Modify: `tests/unit/test_stereo_generator.py`
- Modify: `tests/unit/test_stereo_projector.py`
- Modify: `tests/unit/test_depth_processor.py`
- Modify: `tests/unit/test_resolution.py`

**Interfaces:**
- Consumes: `MetricGeometryFrame`, completed metric metadata, common renderer, exact crop dimensions, and active metric settings.
- Produces: `build_metric_geometry`, mode-specific stage fingerprints, atomic per-frame clamp statistics, and a renderer pipeline with no backend branches.

- [ ] **Step 1: Write failing mask-aware resize tests**

```python
def test_metric_resize_does_not_bleed_invalid_zero_into_near_score() -> None:
    inverse = np.array([[4.0, 0.0], [4.0, 0.0]], dtype=np.float32)
    valid = np.array([[True, False], [True, False]], dtype=np.bool_)
    resized_inverse, resized_valid = resize_metric_geometry(inverse, valid, (2, 4))
    assert resized_valid[:, :2].all()
    assert not resized_valid[:, 2:].any()
    np.testing.assert_allclose(resized_inverse[:, :2], 4.0, rtol=0.0, atol=1e-6)
    assert np.count_nonzero(resized_inverse[:, 2:]) == 0
```

The implementation resizes `inverse*valid` and `valid.float32` with the same bilinear `align_corners=False` operation, divides where weight is nonzero, marks output valid at weight `>=0.5`, and zeroes every invalid output.

- [ ] **Step 2: Write failing projection formula, sign, cap, and z-order tests**

```python
def test_metric_projection_zero_near_far_sign_and_foreground_convention() -> None:
    geometry, stats = build_metric_geometry(
        inverse_depth=np.array([[1.0, 0.5, 0.25]], dtype=np.float32),
        valid=np.ones((1, 3), dtype=np.bool_),
        focal_x_normalized=np.float32(0.5),
        render_shape=(1, 3),
        virtual_baseline_mm=63.0,
        convergence_distance_m=2.0,
        max_disparity_percent=5.0,
        retained_crop_width=3,
    )
    assert geometry.total_disparity_fraction[0, 0] > 0.0
    assert geometry.total_disparity_fraction[0, 1] == 0.0
    assert geometry.total_disparity_fraction[0, 2] < 0.0
    # The three-pixel raster is below the frozen 1/16-pixel offset quantum.
    # Repeat the same geometry to test the foreground eye sign without
    # changing the projection formula or Task 8 rounding contract.
    expanded = np.tile(geometry.total_disparity_fraction, (1, 100))
    left, right = calculate_geometry_eye_sample_offsets(expanded)
    assert left[0, 0] > right[0, 0]
    assert stats.clamped_fraction == 0.0


def test_metric_projection_clamps_both_signs_in_final_output_coordinates() -> None:
    geometry, stats = build_metric_geometry(
        inverse_depth=np.array([[100.0, 0.001]], dtype=np.float32),
        valid=np.ones((1, 2), dtype=np.bool_),
        focal_x_normalized=np.float32(2.0),
        render_shape=(1, 2),
        virtual_baseline_mm=100.0,
        convergence_distance_m=2.0,
        max_disparity_percent=2.0,
        retained_crop_width=1,
    )
    np.testing.assert_allclose(
        geometry.total_disparity_fraction,
        np.array([[0.01, -0.01]], dtype=np.float64),
        rtol=0.0,
        atol=1e-15,
    )
    assert stats.clamped_pixel_count == 2
    assert stats.clamped_fraction == 1.0


def test_clamped_offsets_do_not_replace_unclamped_inverse_depth_z_order() -> None:
    # Both sources hit the same disparity cap; the q=10 source must still win.
    geometry, _stats = build_metric_geometry(
        inverse_depth=np.array([[10.0, 9.0]], dtype=np.float32),
        valid=np.ones((1, 2), dtype=np.bool_),
        focal_x_normalized=np.float32(10.0),
        render_shape=(1, 2),
        virtual_baseline_mm=100.0,
        convergence_distance_m=1000.0,
        max_disparity_percent=1.0,
        retained_crop_width=2,
    )
    assert geometry.near_score.tolist() == [[10.0, 9.0]]
    assert geometry.total_disparity_fraction[0, 0] == geometry.total_disparity_fraction[0, 1]
```

Add the remaining coordinate and validity checks explicitly:

```python
def test_metric_disparity_scales_linearly_with_focal_and_baseline() -> None:
    base, _ = project_one_valid(focal=0.5, baseline_mm=50.0)
    focal2, _ = project_one_valid(focal=1.0, baseline_mm=50.0)
    baseline2, _ = project_one_valid(focal=0.5, baseline_mm=100.0)
    assert focal2.total_disparity_fraction[0, 0] == pytest.approx(
        2.0 * base.total_disparity_fraction[0, 0]
    )
    assert baseline2.total_disparity_fraction[0, 0] == pytest.approx(
        2.0 * base.total_disparity_fraction[0, 0]
    )


def test_metric_invalid_and_zero_valid_pixels_do_not_count_or_shift() -> None:
    geometry, stats = build_metric_geometry(
        inverse_depth=np.array([[10.0, 0.0]], dtype=np.float32),
        valid=np.array([[False, False]], dtype=np.bool_),
        focal_x_normalized=np.float32(1.0),
        render_shape=(1, 2),
        virtual_baseline_mm=100.0,
        convergence_distance_m=2.0,
        max_disparity_percent=1.0,
        retained_crop_width=2,
    )
    assert not geometry.source_valid.any()
    assert np.count_nonzero(geometry.total_disparity_fraction) == 0
    assert stats == MetricProjectionStats(0, 0, 0.0)


def test_odd_crop_width_converts_final_cap_back_to_source_fraction() -> None:
    geometry, _ = build_metric_geometry(
        inverse_depth=np.full((1, 5), 100.0, dtype=np.float32),
        valid=np.ones((1, 5), dtype=np.bool_),
        focal_x_normalized=np.float32(2.0),
        render_shape=(1, 5),
        virtual_baseline_mm=100.0,
        convergence_distance_m=2.0,
        max_disparity_percent=2.0,
        retained_crop_width=3,
    )
    np.testing.assert_allclose(geometry.total_disparity_fraction, 0.012)


def test_metric_builder_is_independent_of_final_output_width() -> None:
    assert "per_eye_width" not in inspect.signature(build_metric_geometry).parameters
    assert "vr_output_width" not in inspect.signature(build_metric_geometry).parameters
    source_fraction = 0.012
    retained_width = 3
    for final_width in (7, 1920, 3840):
        final_disparity = source_fraction * 5 / retained_width * final_width
        assert final_disparity == pytest.approx(0.02 * final_width)
```

- [ ] **Step 3: Run geometry tests and verify the metric builder is missing**

Run: `uv run pytest tests/unit/test_stereo_geometry.py -v`

Expected: FAIL on missing `build_metric_geometry` and `resize_metric_geometry`.

- [ ] **Step 4: Implement the approved formula literally**

```python
source_height, source_width = render_shape
retained_fraction = np.float64(retained_crop_width) / np.float64(source_width)
baseline_m = np.float64(virtual_baseline_mm) / np.float64(1000.0)
limit = np.float64(max_disparity_percent) / np.float64(100.0)
raw_output_fraction = (
    np.float64(focal_x_normalized)
    / retained_fraction
    * baseline_m
    * (resized_inverse.astype(np.float64) - np.float64(1.0 / convergence_distance_m))
)
clamped_output_fraction = np.clip(raw_output_fraction, -limit, limit)
render_fraction = clamped_output_fraction * retained_fraction
render_fraction[~resized_valid] = 0.0
```

Clamp statistics count only resized valid pixels whose raw output fraction is outside `[-limit,+limit]`. `near_score` is the unclamped resized inverse depth. Validate retained crop width in `1..source_width` and every scalar as finite/in-range.

- [ ] **Step 5: Write failing stereo-generator mode and metadata tests**

```python
def test_metric_generator_uses_resolved_auto_convergence_and_common_renderer(
    metric_stage, frame_files, directories, recording_renderer, default_settings
) -> None:
    settings = {
        **default_settings,
        "stereo_geometry_mode": "metric_camera",
        "metric_convergence_distance": "auto",
        "virtual_baseline_mm": 63.0,
        "max_disparity_percent": 2.0,
        "crop_factor": 0.75,
        "apply_distortion": False,
        "vr_format": "side_by_side",
    }
    generator = StereoPairGenerator(renderer=recording_renderer)
    assert generator.create_stereo_pairs_from_files(
        frame_files, metric_stage.files, directories, settings
    )
    assert recording_renderer.geometry_calls
    assert not recording_renderer.legacy_relative_calls
    metadata = json.loads((directories["left_frames"] / "metadata.json").read_text())
    assert metadata["geometry_mode"] == "metric_camera"
    assert metadata["effective_convergence_distance_m"] == pytest.approx(
        metric_stage.resolved_auto_distance_m
    )
    assert "vr_output_width" not in json.dumps(metadata)
```

- [ ] **Step 6: Decode both stage formats outside the renderer**

Change `_DecodedMessage` from `canonical` to:

```python
@dataclass(frozen=True)
class _DecodedMessage:
    work: _FileWorkItem
    frame: np.ndarray | None = None
    geometry: StereoGeometryFrame | None = None
    projection_stats: MetricProjectionStats | None = None
    error: Exception | None = None
```

`StereoPairGenerator` chooses one decode closure at stage setup:

- Relative closure decodes uint16 PNG, calls `build_relative_geometry`, and returns no projection stats.
- Metric closure loads `MetricGeometryFrame`, resolves `Z0` from completed metadata for `auto` or uses the explicit float, calculates exact retained crop width with `calculate_center_crop_dimensions`, and calls `build_metric_geometry`.

The main render thread always calls:

```python
self.renderer.render_geometry(message.frame, message.geometry, self.splat_settings)
```

For relative, `max_eye_shift_fraction=stereo_strength/200`. For metric, it is `(max_disparity_percent/100) * (retained_crop_width/source_width) / 2`. There is no backend or geometry-mode condition inside `StereoRenderer` or `forward_splat_band`.

- [ ] **Step 7: Make stereo fingerprints mode-specific and crop-aware**

Static stereo metadata records `geometry_mode`, source stage fingerprint, renderer algorithm/device, frame names, render shape, occlusion fill, and only the active projection settings. Metric metadata additionally records projection algorithm identity, `source_width`, exact `retained_crop_width`, `CENTER_CROP_ALGORITHM_VERSION`, canonical SAR `1:1`, active baseline, requested convergence (`auto` or float), effective numeric convergence, and cap. It excludes `per_eye_width`, `vr_output_width`, and every inactive relative setting.

- [ ] **Step 8: Persist per-frame clamp stats transactionally with each metric pair**

Add `04_left_frames/clamp_stats/<frame>.json`. A metric `_WriteItem` contains `projection_stats`. `_write_pair` writes left image, right image, and the stat sidecar through temporaries; any failure deletes all three outputs for that frame. The sidecar is:

```python
{
    "schema_version": 1,
    "frame_name": item.work.frame_name,
    "valid_pixel_count": stats.valid_pixel_count,
    "clamped_pixel_count": stats.clamped_pixel_count,
    "clamped_fraction": stats.clamped_fraction,
}
```

Completed metric pairs are reusable only when their stat sidecar validates. After every pair validates, write `clamp_summary.json` atomically with ordered fractions, `affected_frame_count` for fractions greater than zero, mean across all frames, and maximum. Identify the lowest source index above `0.05` and print one warning. Store the summary on `StereoPairGenerator.last_metric_clamp_summary` for orchestration.

- [ ] **Step 9: Preserve the single-image relative path**

`StereoProjector.process_image` calls `build_relative_geometry` and `render_geometry`. If `stereo_geometry_mode=metric_camera` is supplied to image processing, reject it with `metric_camera is supported for video processing only`; do not invent single-image auto convergence.

- [ ] **Step 10: Run geometry, generator, renderer, and projector tests**

Run: `uv run pytest tests/unit/test_stereo_geometry.py tests/unit/test_stereo_generator.py tests/unit/test_stereo_renderer.py tests/unit/test_forward_splat.py tests/unit/test_stereo_projector.py -v`

Expected: PASS, including frozen relative hashes, invalid-source exclusion, both cap signs, common renderer use, sidecar repair, and one warning.

- [ ] **Step 11: Commit metric stereo rendering**

```bash
git add src/depth_surge_3d/rendering/stereo_geometry.py src/depth_surge_3d/processing/frames/stereo_generator.py src/depth_surge_3d/rendering/stereo_projector.py tests/unit/test_stereo_geometry.py tests/unit/test_stereo_generator.py tests/unit/test_stereo_projector.py
git commit -m "feat: render crop-aware metric SBS geometry"
```

### Task 12: Make Resume, Invalidation, and Cleanup Mode-Aware

**Files:**
- Modify: `src/depth_surge_3d/io/resume.py`
- Modify: `src/depth_surge_3d/io/operations.py`
- Modify: `src/depth_surge_3d/processing/orchestration/pipeline_orchestrator.py`
- Modify: `tests/unit/test_resume.py`
- Modify: `tests/unit/test_file_operations.py`
- Modify: `tests/unit/test_pipeline_orchestrator.py`

**Interfaces:**
- Consumes: both stage-3 metadata formats and mode-specific stereo metadata.
- Produces: independent `disparity_maps`/`metric_geometry` resume stages, selected upstream validation, legacy SAR re-probe, non-destructive switches, and successful all-intermediate cleanup.

- [ ] **Step 1: Write failing dual-stage resume and invalidation tests**

```python
def test_resume_reports_both_valid_stage3_directories_without_deleting_inactive_one(
    completed_dual_stage_job
) -> None:
    report = build_resume_report(
        completed_dual_stage_job.output_dir,
        completed_dual_stage_job.metric_settings,
        source_video=completed_dual_stage_job.source_video,
        model_fingerprint=completed_dual_stage_job.model_fingerprint,
    )
    assert report.stage("disparity_maps").disposition == "preserve"
    assert report.stage("metric_geometry").disposition == "preserve"
    assert report.stage("stereo").disposition == "preserve"


def test_metric_stereo_settings_invalidate_only_stereo(completed_metric_job) -> None:
    changed = {
        **completed_metric_job.settings,
        "virtual_baseline_mm": 70.0,
        "metric_convergence_distance": 3.0,
        "max_disparity_percent": 1.5,
    }
    report = build_resume_report(
        completed_metric_job.output_dir,
        changed,
        source_video=completed_metric_job.source_video,
        model_fingerprint=completed_metric_job.model_fingerprint,
    )
    assert report.stage("depth_raw").disposition in {"preserve", "resume"}
    assert report.stage("metric_geometry").disposition == "preserve"
    assert report.stage("stereo").disposition == "invalidate"


def test_crop_change_invalidates_metric_stereo_but_not_relative_stereo(
    completed_metric_job, completed_relative_job
) -> None:
    metric = build_report_with(completed_metric_job, crop_factor=0.8)
    relative = build_report_with(completed_relative_job, crop_factor=0.8)
    assert metric.stage("stereo").disposition == "invalidate"
    assert relative.stage("stereo").disposition == "preserve"
    assert relative.stage("crop").disposition == "invalidate"
```

```python
def test_mode_switch_preserves_both_valid_stage3_formats(completed_dual_stage_job) -> None:
    report = build_report_with(
        completed_dual_stage_job,
        stereo_geometry_mode="relative",
    )
    assert report.stage("disparity_maps").disposition == "preserve"
    assert report.stage("metric_geometry").disposition == "preserve"
    assert report.stage("stereo").disposition == "invalidate"


def test_final_output_width_change_reuses_metric_stereo(completed_metric_job) -> None:
    report = build_report_with(
        completed_metric_job,
        per_eye_width=2560,
        vr_output_width=5120,
    )
    assert report.stage("metric_geometry").disposition == "preserve"
    assert report.stage("stereo").disposition == "preserve"
    assert report.stage("crop").disposition == "invalidate"


def test_failed_job_retains_committed_stage3_checkpoints(failed_dual_stage_job) -> None:
    before = hash_stage3(failed_dual_stage_job.output_dir)
    failed_dual_stage_job.orchestrator._finalize_processing(
        False,
        failed_dual_stage_job.output_dir,
        str(failed_dual_stage_job.source_video),
        {**failed_dual_stage_job.settings, "keep_intermediates": False},
        failed_dual_stage_job.frame_count,
    )
    assert hash_stage3(failed_dual_stage_job.output_dir) == before


def test_no_retention_missing_mode_reports_reinference_without_deletion(
    metric_only_job_without_raw
) -> None:
    frames_before = hash_directory(metric_only_job_without_raw.frames_dir)
    metric_before = hash_directory(metric_only_job_without_raw.metric_dir)
    report = build_report_with(
        metric_only_job_without_raw,
        stereo_geometry_mode="relative",
    )
    assert "MoGe inference is required to build the selected geometry stage" in (
        report.stage("disparity_maps").reason
    )
    assert hash_directory(metric_only_job_without_raw.frames_dir) == frames_before
    assert hash_directory(metric_only_job_without_raw.metric_dir) == metric_before


def test_successful_cleanup_removes_both_stage3_directories(completed_dual_stage_job) -> None:
    cleanup_intermediate_files(completed_dual_stage_job.output_dir)
    assert not any(completed_dual_stage_job.disparity_dir.iterdir())
    assert not any(completed_dual_stage_job.metric_dir.iterdir())
```

- [ ] **Step 2: Write failing legacy SAR re-probe tests**

```python
def test_legacy_metric_resume_reprobes_sar_before_model_loading(
    legacy_job_without_sar, monkeypatch
) -> None:
    monkeypatch.setattr(
        resume,
        "get_video_properties",
        lambda _path: {
            "sample_aspect_ratio_numerator": 1,
            "sample_aspect_ratio_denominator": 1,
        },
    )
    report = build_resume_report(
        legacy_job_without_sar.output_dir,
        {**legacy_job_without_sar.settings, "stereo_geometry_mode": "metric_camera"},
        source_video=legacy_job_without_sar.source_video,
    )
    assert report.stage("frames").disposition == "preserve"


def test_legacy_metric_resume_without_source_fails_without_invalidating_relative_data(
    legacy_job_without_sar
) -> None:
    with pytest.raises(ValueError, match="re-probe.*sample aspect ratio"):
        build_resume_report(
            legacy_job_without_sar.output_dir,
            {**legacy_job_without_sar.settings, "stereo_geometry_mode": "metric_camera"},
            source_video=legacy_job_without_sar.output_dir / "missing.mp4",
        )
    relative = build_resume_report(
        legacy_job_without_sar.output_dir,
        {**legacy_job_without_sar.settings, "stereo_geometry_mode": "relative"},
        source_video=None,
    )
    assert relative.stage("disparity_maps").disposition == "preserve"
```

- [ ] **Step 3: Run resume tests and verify metric stage is not reported**

Run: `uv run pytest tests/unit/test_resume.py tests/unit/test_file_operations.py -v`

Expected: FAIL because `metric_geometry` is absent from the resume stage graph.

- [ ] **Step 4: Validate both stage-3 formats independently**

Add `_validate_metric_geometry_stage(...)` using `MetricGeometryStore.open_existing`, complete metadata fingerprint, ordered frame names, native shape, exact `source_raw_fingerprint`, source-frame fingerprint, and current candidate-scene fingerprint. Always append both `disparity_maps` and `metric_geometry` stages to the report. Inactive validity is reported independently and does not become a migration path merely because another mode is selected.

- [ ] **Step 5: Select one stage as stereo upstream and compare only active settings**

For relative, reuse the existing canonical metadata matcher and relative render settings. For metric, use metric metadata plus metric render settings and crop/SAR fingerprint. A mode change necessarily mismatches stereo's `geometry_mode`, but it never invalidates raw or either valid stage 3. Do not include final per-eye or packed width in metric stereo matching.

- [ ] **Step 6: Re-probe only the missing legacy SAR contract**

Read saved `video_properties`. If canonical SAR numerator/denominator are present, validate them. If absent and mode is metric, call `get_video_properties` on the existing original source before estimator construction; merge only the SAR fields into the in-memory properties. If the source is missing, raise the explicit re-probe error. Relative report construction skips this requirement and retains independently valid frame/relative stages.

- [ ] **Step 7: Preserve inactive data and clean only after validated success**

`cleanup_intermediate_files` already iterates `INTERMEDIATE_DIRS`; assert the new directory is included and both stage-3 directories are emptied only from the success branch in `_finalize_processing`. The failure branch writes failed status and performs no cleanup. A mode switch with missing selected stage and missing raw payload reports `MoGe inference is required to build the selected geometry stage` without deleting source frames or the inactive stage.

- [ ] **Step 8: Persist and report the metric clamp summary at job completion**

After stereo succeeds, copy `last_metric_clamp_summary` into the settings status `runtime_info` and print:

```text
Metric disparity clamp summary: affected_frames={count}, mean={mean:.4%}, max={max:.4%}
```

Relative jobs do not add this field. Interrupted metric jobs retain per-frame clamp sidecars for deterministic resume.

- [ ] **Step 9: Run resume, cleanup, and orchestration tests**

Run: `uv run pytest tests/unit/test_resume.py tests/unit/test_file_operations.py tests/unit/test_pipeline_orchestrator.py tests/unit/test_stereo_generator.py -v`

Expected: PASS for dual-stage preservation, mode-specific invalidation, legacy SAR behavior, successful cleanup, and failure retention.

- [ ] **Step 10: Commit Slice 2 cache and resume semantics**

```bash
git add src/depth_surge_3d/io/resume.py src/depth_surge_3d/io/operations.py src/depth_surge_3d/processing/orchestration/pipeline_orchestrator.py tests/unit/test_resume.py tests/unit/test_file_operations.py tests/unit/test_pipeline_orchestrator.py tests/unit/test_stereo_generator.py
git commit -m "feat: resume relative and metric geometry independently"
```

- [ ] **Step 11: Run the complete Slice 2 gate**

Run: `uv run pytest tests/unit/test_stereo_geometry.py tests/unit/test_forward_splat.py tests/unit/test_stereo_renderer.py tests/unit/test_metric_geometry.py tests/unit/test_depth_processor.py tests/unit/test_stereo_generator.py tests/unit/test_resume.py tests/unit/test_pipeline_orchestrator.py -q`

Expected: PASS, including the unchanged fixed relative hashes.

---

## Slice 3: Product Surface and Release Evidence

### Task 13: Complete Metric Startup Validation, CLI Reporting, and Web Controls

**Files:**
- Modify: `src/depth_surge_3d/inference/depth/backend_registry.py`
- Modify: `src/depth_surge_3d/inference/depth/video_depth_estimator.py`
- Modify: `src/depth_surge_3d/inference/depth/video_depth_estimator_da3.py`
- Modify: `src/depth_surge_3d/inference/depth/video_depth_estimator_see_through.py`
- Modify: `src/depth_surge_3d/rendering/stereo_projector.py`
- Modify: `depth_surge_3d.py`
- Modify: `app.py`
- Modify: `templates/index.html`
- Modify: `tests/unit/test_cli_moge2.py`
- Modify: `tests/unit/test_web_moge2.py`
- Modify: `tests/unit/test_stereo_projector.py`

**Interfaces:**
- Consumes: normalized settings, canonical video SAR, registry variant/capability data, and an estimator that has been constructed but not loaded.
- Produces: all four public metric settings, one canonical effective-run report, pre-load validation on CLI and Web, one exact temporal warning, and capability-driven controls.

- [ ] **Step 1: Write failing CLI geometry-option tests**

```python
def test_cli_normalizes_metric_geometry_options(cli_module) -> None:
    args = cli_module.create_argument_parser().parse_args(
        [
            "clip.mp4",
            "--depth-model-version",
            "moge2",
            "--model-size",
            "vits",
            "--stereo-geometry-mode",
            "metric_camera",
            "--virtual-baseline-mm",
            "64.0",
            "--metric-convergence-distance",
            "2.5",
            "--max-disparity-percent",
            "1.5",
            "--format",
            "side_by_side",
            "--no-distortion",
        ]
    )
    settings = cli_module._build_processing_settings(args)
    assert settings["stereo_geometry_mode"] == "metric_camera"
    assert settings["virtual_baseline_mm"] == 64.0
    assert settings["metric_convergence_distance"] == 2.5
    assert settings["max_disparity_percent"] == 1.5


def test_cli_accepts_auto_metric_convergence(cli_module) -> None:
    args = cli_module.create_argument_parser().parse_args(
        ["clip.mp4", "--metric-convergence-distance", "auto"]
    )
    assert cli_module._build_processing_settings(args)[
        "metric_convergence_distance"
    ] == "auto"


def test_cli_has_no_public_moge_resolution_level_flag(cli_module) -> None:
    help_text = cli_module.create_argument_parser().format_help()
    assert "moge-resolution" not in help_text
    assert "resolution-level" not in help_text
```

Parse metric convergence with `_parse_metric_convergence`: return the literal `"auto"`, otherwise parse one finite float and let `validate_settings` enforce `0.1..1000`.

- [ ] **Step 2: Write failing pre-load validation and report tests**

```python
TEMPORAL_STABILITY_WARNING = (
    "MoGe-2 performs per-frame depth and focal estimation. Temporal stability "
    "on video is not guaranteed; depth or focal drift may be visible across frames."
)


def test_metric_sar_is_rejected_before_model_load(projector, monkeypatch, tmp_path) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        stereo_projector,
        "get_video_properties",
        lambda _path: {
            "width": 8,
            "height": 4,
            "fps": 24.0,
            "frame_count": 2,
            "sample_aspect_ratio_numerator": 4,
            "sample_aspect_ratio_denominator": 3,
        },
    )
    projector.depth_estimator.load_model.side_effect = lambda: calls.append("load") or True
    ok = projector.process_video(
        str(tmp_path / "clip.mp4"),
        str(tmp_path / "out"),
        metric_settings(),
    )
    assert ok is False
    assert calls == []


def test_effective_moge_report_contains_every_required_identity(fake_moge_estimator) -> None:
    report = build_effective_depth_run_report(metric_settings(), fake_moge_estimator)
    assert report == {
        "backend": "moge2",
        "model_size": "vitb",
        "repository": "Ruicheng/moge-2-vitb-normal",
        "revision": "54ad3a693e61907ea4633d13dec6ee682fa09419",
        "device": "cpu",
        "precision": "float32",
        "depth_resolution": 1080,
        "adapter_resolution_level": 9,
        "camera_capability": "pinhole_fx",
        "geometry_mode": "metric_camera",
        "projection": {
            "virtual_baseline_mm": 63.0,
            "metric_convergence_distance": "auto",
            "max_disparity_percent": 2.0,
        },
    }
```

```python
def test_relative_report_contains_only_active_projection_settings(fake_estimator) -> None:
    report = build_effective_depth_run_report(relative_settings(), fake_estimator)
    assert report["projection"] == {
        "stereo_strength": 2.0,
        "convergence": 0.5,
        "occlusion_fill": "background",
    }


def test_custom_model_report_does_not_claim_a_registry_revision(fake_moge_estimator) -> None:
    fake_moge_estimator.repo_id = "D:/models/custom/model.pt"
    fake_moge_estimator.revision = None
    report = build_effective_depth_run_report(
        metric_settings(model_path="D:/models/custom/model.pt", model_size="custom"),
        fake_moge_estimator,
    )
    assert report["model_size"] == "custom"
    assert report["repository"] == "D:/models/custom/model.pt"
    assert report["revision"] is None
```

- [ ] **Step 3: Run CLI/projector tests and verify the flags and report are absent**

Run: `uv run pytest tests/unit/test_cli_moge2.py tests/unit/test_stereo_projector.py -v`

Expected: FAIL on missing CLI flags, report builder, and pre-load geometry validation.

- [ ] **Step 4: Add one canonical run-report builder**

First extract the existing automatic depth-size arithmetic from `DepthMapProcessor` into the shared resolution module without changing its thresholds:

```python
def resolve_depth_input_size(width: int, height: int, value: int | str) -> int:
    if width < 1 or height < 1:
        raise ValueError("Source dimensions must be positive")
    if value != "auto":
        resolved = int(value)
        if resolved < 1:
            raise ValueError("Depth resolution must be positive")
        return resolved
    megapixels = width * height / 1_000_000
    longest = max(width, height)
    if megapixels > MEGAPIXELS_4K:
        return min(longest, RESOLUTION_4K)
    if megapixels > MEGAPIXELS_1080P:
        return min(longest, RESOLUTION_1080P)
    if megapixels > MEGAPIXELS_720P:
        return min(longest, RESOLUTION_720P)
    return min(longest, RESOLUTION_SD)
```

```python
@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (640, 360, 640),
        (1280, 720, 640),
        (1920, 1080, 1080),
        (3840, 2160, 2160),
        (7680, 4320, 2160),
    ],
)
def test_shared_auto_depth_size_preserves_existing_thresholds(
    width: int, height: int, expected: int
) -> None:
    assert resolve_depth_input_size(width, height, "auto") == expected
```

`DepthMapProcessor._determine_chunk_params` calls this function, and CLI/Web replace `depth_resolution="auto"` with its numeric result immediately after video probing. Persist and report that numeric cap.

In `backend_registry.py`, define the warning once and add:

```python
def build_effective_depth_run_report(
    settings: Mapping[str, Any], estimator: Any
) -> dict[str, Any]:
    backend_id = str(settings["depth_model_version"])
    spec = get_backend_spec(backend_id)
    custom_path = cast(str | None, settings.get("model_path"))
    variant = None if custom_path else resolve_model_variant(
        backend_id, cast(str | None, settings.get("model_size"))
    )
    mode = cast(StereoGeometryMode, settings["stereo_geometry_mode"])
    if mode == "metric_camera":
        projection = {
            "virtual_baseline_mm": settings["virtual_baseline_mm"],
            "metric_convergence_distance": settings["metric_convergence_distance"],
            "max_disparity_percent": settings["max_disparity_percent"],
        }
    else:
        projection = {
            "stereo_strength": settings["stereo_strength"],
            "convergence": settings["convergence"],
            "occlusion_fill": settings["occlusion_fill"],
        }
    if custom_path:
        model_size = "custom"
        repository = str(getattr(estimator, "repo_id", custom_path))
        revision = getattr(estimator, "revision", None)
    else:
        assert variant is not None
        model_size = variant.setting
        repository = variant.repo_id or variant.backend_value
        revision = variant.revision
    return {
        "backend": backend_id,
        "model_size": model_size,
        "repository": repository,
        "revision": revision,
        "device": str(getattr(estimator, "device")),
        "precision": str(getattr(estimator, "inference_precision")),
        "depth_resolution": settings["depth_resolution"],
        "adapter_resolution_level": getattr(estimator, "resolution_level", None),
        "camera_capability": "pinhole_fx" if spec.capabilities.pinhole_fx else "none",
        "geometry_mode": mode,
        "projection": projection,
    }
```

Give every registered estimator an explicit `inference_precision` property and test its existing execution policy:

```python
# VideoDepthEstimator: DepthMapProcessor always calls fp32=False.
@property
def inference_precision(self) -> str:
    return "float16" if self.device == "cuda" else "float32"

# VideoDepthEstimatorDA3: this wrapper moves float32 weights without autocast.
@property
def inference_precision(self) -> str:
    return "float32"

# SeeThroughDepthEstimator: the loader already uses this resolved dtype.
@property
def inference_precision(self) -> str:
    return str(self._resolve_dtype()).removeprefix("torch.")
```

MoGe returns `float16` only for CUDA and `float32` for CPU. Add assertions for V2 CPU/CUDA, DA3 CUDA, and See-Through CPU plus mocked CUDA bf16/fp16 branches to their existing estimator tests. Format the report mapping in a stable field order for CLI output. The internal value `9` is reported but never accepted from settings or arguments.

- [ ] **Step 5: Move video/cross-field validation ahead of model loading**

`StereoProjector.process_video` must perform this order:

1. validate settings and paths;
2. call `get_video_properties`;
3. resolve automatic depth size and output dimensions without filesystem mutation;
4. call `validate_backend_geometry_request`;
5. build and print the effective run report;
6. print `TEMPORAL_STABILITY_WARNING` once when mode is `metric_camera`;
7. call `_ensure_model_loaded`;
8. start filesystem mutation.

In `app.py`, perform availability, source-video probing, and `validate_backend_geometry_request` inside `/process` before `socketio.start_background_task`. Pass the validated video-properties mapping to `process_video_async` so it is not re-probed. In the async function, construct the projector, emit `processing_configuration`, print the warning once, then call `load_model`. A rejected request starts no background task and constructs no estimator.

- [ ] **Step 6: Write failing Web payload, constraint, resume, and reporting tests**

```python
def test_web_metric_payload_uses_active_names(client, available_moge) -> None:
    html = client.get("/").get_data(as_text=True)
    for element_id in (
        "stereoGeometryMode",
        "relativeGeometrySettings",
        "metricGeometrySettings",
        "virtualBaselineMm",
        "metricConvergenceMode",
        "metricConvergenceDistance",
        "maxDisparityPercent",
        "metricExperimentalWarning",
        "effectiveProcessingConfig",
    ):
        assert f'id="{element_id}"' in html
    for setting_name in (
        "stereo_geometry_mode",
        "virtual_baseline_mm",
        "metric_convergence_distance",
        "max_disparity_percent",
    ):
        assert f"{setting_name}:" in html


def test_forged_metric_web_request_is_rejected_before_background_start(
    client, monkeypatch, uploaded_video
) -> None:
    start = Mock()
    monkeypatch.setattr(app.socketio, "start_background_task", start)
    response = client.post(
        "/process",
        json=process_payload(
            uploaded_video,
            depth_model_version="v3",
            stereo_geometry_mode="metric_camera",
            vr_format="side_by_side",
            apply_distortion=False,
        ),
    )
    assert response.status_code == 400
    assert "pinhole_fx" in response.get_json()["error"]
    start.assert_not_called()


def test_web_resume_restores_moge_metric_values(tmp_path) -> None:
    saved = metric_settings(
        model_size="vitl",
        virtual_baseline_mm=70.0,
        metric_convergence_distance=3.0,
        max_disparity_percent=1.25,
    )
    write_settings_file(tmp_path, saved)
    restored = app.detect_resume_settings(tmp_path)
    assert {key: restored[key] for key in saved} == saved
```

```python
@pytest.mark.parametrize(
    "settings, error",
    [
        ({"vr_format": "over_under", "apply_distortion": False}, "side_by_side"),
        ({"vr_format": "side_by_side", "apply_distortion": True}, "apply_distortion=false"),
    ],
)
def test_web_rejects_forged_metric_projection_constraints(
    client, uploaded_video, settings, error
) -> None:
    response = client.post(
        "/process",
        json=process_payload(
            uploaded_video,
            depth_model_version="moge2",
            stereo_geometry_mode="metric_camera",
            **settings,
        ),
    )
    assert response.status_code == 400
    assert error in response.get_json()["error"]


def test_web_rejects_unavailable_moge_even_when_json_is_forged(
    client, unavailable_moge, uploaded_video
) -> None:
    response = client.post(
        "/process", json=process_payload(uploaded_video, depth_model_version="moge2")
    )
    assert response.status_code == 400
    assert "uv sync --extra moge2" in response.get_json()["error"]


def test_web_rejects_non_square_sar_before_background_start(
    client, available_moge, uploaded_video, monkeypatch
) -> None:
    monkeypatch.setattr(app, "get_video_properties", lambda _path: video_props(sar=(4, 3)))
    start = Mock()
    monkeypatch.setattr(app.socketio, "start_background_task", start)
    response = client.post(
        "/process", json=process_payload(uploaded_video, **metric_settings())
    )
    assert response.status_code == 400
    start.assert_not_called()


def test_web_emits_configuration_before_model_load(web_async_harness) -> None:
    web_async_harness.run(**metric_settings())
    assert web_async_harness.events.index("processing_configuration") < (
        web_async_harness.events.index("load_model")
    )
```

- [ ] **Step 7: Render capability-driven metric controls**

Use the registry JSON created in Task 6. Add `relative` and `metric_camera` options only when the selected backend capability contains them. The metric group has numeric controls with exact bounds and steps, plus `auto`/`custom` convergence mode. On entering metric mode, set visible `vrFormat` to `side_by_side`, uncheck and disable visible `applyDistortion`, show the Experimental badge and exact warning, hide the relative group, and show the metric group. On leaving metric mode, re-enable packing/distortion and show only the relative group. Do not add `MOGE_RESOLUTION_LEVEL` to HTML or JavaScript.

Listen for `processing_configuration` and populate a hidden-until-used definition list via `textContent`; do not inject server strings as HTML. Keep the existing `frame_preview` event and payload unchanged.

- [ ] **Step 8: Run the complete product-surface unit gate**

Run: `uv run pytest tests/unit/test_cli_moge2.py tests/unit/test_web_moge2.py tests/unit/test_stereo_projector.py tests/unit/test_depth_processor.py tests/unit/test_resolution.py tests/unit/test_resume_template.py tests/unit/test_see_through_entrypoints.py tests/unit/test_video_depth_estimator.py tests/unit/test_video_depth_estimator_da3.py tests/unit/test_video_depth_estimator_see_through.py -v`

Expected: PASS with the optional extra absent and with mocked available MoGe. Existing V2, V3, and See-Through selections remain valid.

- [ ] **Step 9: Commit validated product controls**

```bash
git add src/depth_surge_3d/inference/depth/backend_registry.py src/depth_surge_3d/inference/depth/video_depth_estimator.py src/depth_surge_3d/inference/depth/video_depth_estimator_da3.py src/depth_surge_3d/inference/depth/video_depth_estimator_see_through.py src/depth_surge_3d/rendering/stereo_projector.py src/depth_surge_3d/processing/frames/depth_processor.py src/depth_surge_3d/utils/domain/resolution.py depth_surge_3d.py app.py templates/index.html tests/unit/test_cli_moge2.py tests/unit/test_web_moge2.py tests/unit/test_stereo_projector.py tests/unit/test_depth_processor.py tests/unit/test_resolution.py tests/unit/test_video_depth_estimator.py tests/unit/test_video_depth_estimator_da3.py tests/unit/test_video_depth_estimator_see_through.py
git commit -m "feat: expose validated metric SBS controls"
```

### Task 14: Prove Live-Preview Ordering and Mocked End-to-End Processing

**Files:**
- Modify: `src/depth_surge_3d/processing/frames/stereo_generator.py`
- Modify: `tests/unit/test_stereo_generator.py`
- Modify: `tests/unit/test_pipeline_orchestrator.py`
- Create: `tests/integration/test_moge2_pipeline.py`

**Interfaces:**
- Consumes: the public CLI/Web settings, a fake MoGe estimator that returns metric depth plus focal length, real raw/stage-3/stereo/crop/assembly code, and stubbed FFmpeg boundaries.
- Produces: an explicit completed-stage preview barrier and deterministic end-to-end evidence for both geometry modes, resume, active-setting changes, and no-retention failure recovery.

- [ ] **Step 1: Write a reusable fake MoGe integration estimator**

In `tests/integration/test_moge2_pipeline.py`, define `_FakeMoGeEstimator` with these stable attributes:

```python
class _FakeMoGeEstimator:
    max_batch_size = 1
    metric = True
    camera_model = "pinhole_fx"
    model_size = "vitb"
    repo_id = "Ruicheng/moge-2-vitb-normal"
    revision = "54ad3a693e61907ea4633d13dec6ee682fa09419"
    resolution_level = 9
    inference_precision = "float32"
    processing_resolution = 8
    device = "cpu"

    def __init__(self) -> None:
        self.calls = 0

    def estimate_output_shape(self, height: int, width: int, _input_size: int) -> tuple[int, int]:
        return height, width

    def estimate_depth_batch(self, frames, **_kwargs) -> DepthBatch:
        self.calls += len(frames)
        height, width = frames.shape[1:3]
        x = np.linspace(0.75, 4.0, width, dtype=np.float32)
        values = np.broadcast_to(x, (len(frames), height, width)).copy()
        values[:, 0, 0] = np.nan
        focal = np.full((len(frames),), 0.5, dtype=np.float32)
        return DepthBatch(
            values,
            DepthRepresentation.METRIC_DEPTH,
            PinholeCameraBatch(focal),
        )

    def get_model_size(self) -> str:
        return self.model_size

    def get_model_info(self) -> dict[str, object]:
        return {"model_size": self.model_size, "repo_id": self.repo_id}

    def unload_model(self) -> None:
        return None
```

The fixture writes three differently colored `8x6` source PNGs and calls `write_source_frame_manifest`. Patch only `VideoEncoder.extract_frames` to return those files and `VideoEncoder.create_video` to assert that three packed SBS PNGs exist, write the expected output filename, and return `True`. All stages between those two boundaries remain production code.

- [ ] **Step 2: Write failing live-preview barrier and parity tests**

```python
def test_metric_stereo_preview_follows_completed_clip_convergence(
    metric_pipeline_harness
) -> None:
    tracker = RecordingProgressTracker()
    result = metric_pipeline_harness.run(progress_tracker=tracker)
    assert result.success
    stereo_event = next(event for event in tracker.events if event.kind == "stereo_left")
    complete_event = next(event for event in tracker.events if event.kind == "metric_complete")
    assert complete_event.sequence < stereo_event.sequence
    metadata = json.loads(
        (result.output_dir / "03_metric_geometry" / "metadata.json").read_text()
    )
    assert metadata["status"] == "complete"
    assert metadata["convergence"]["resolved_auto_distance_m"] > 0.0
    assert stereo_event.png_bytes == stereo_event.path.read_bytes()


def test_explicit_metric_preview_and_final_use_the_same_convergence(
    metric_pipeline_harness
) -> None:
    result = metric_pipeline_harness.run(metric_convergence_distance=2.25)
    stereo_metadata = json.loads(
        (result.output_dir / "04_left_frames" / "metadata.json").read_text()
    )
    assert stereo_metadata["requested_convergence_distance_m"] == 2.25
    assert stereo_metadata["effective_convergence_distance_m"] == 2.25
    assert result.previewed_left_bytes == result.final_left_bytes
```

`RecordingProgressTracker` records existing `depth_map` and `stereo_left` callbacks. The harness records `metric_complete` immediately after `MetricGeometryStore.finalize` returns; it does not emit a Socket.IO event or create a new public preview state.

- [ ] **Step 3: Enforce the completed metric-stage barrier at stereo setup**

Before creating any metric decode work, `StereoPairGenerator` reads `03_metric_geometry/metadata.json` and requires `status="complete"`, a valid metadata fingerprint, a complete ordered frame manifest, and a persisted convergence block. It raises `ValueError("metric stereo requires completed clip-global convergence metadata")` before opening the renderer otherwise. Relative setup does not read this metadata. Continue previewing the atomically committed left PNG, never a second independently rendered preview array.

- [ ] **Step 4: Run preview tests**

Run: `uv run pytest tests/unit/test_stereo_generator.py tests/unit/test_pipeline_orchestrator.py -k "preview or convergence or metric" -v`

Expected: PASS; no metric `stereo_left` callback precedes finalized clip-global convergence and the preview bytes are production bytes.

- [ ] **Step 5: Write mocked clean, resume, and mode-change integration tests**

```python
@pytest.mark.parametrize("geometry_mode", ["relative", "metric_camera"])
def test_mocked_moge_runs_selected_geometry_through_sbs_assembly(
    integration_harness, geometry_mode
) -> None:
    result = integration_harness.run(geometry_mode=geometry_mode)
    assert result.success
    assert len(result.vr_frames) == 3
    assert result.final_video.is_file()
    assert result.metric_stage_exists is (geometry_mode == "metric_camera")
    assert result.relative_stage_exists is (geometry_mode == "relative")


def test_metric_projection_setting_change_reuses_raw_and_metric_stage(
    integration_harness
) -> None:
    first = integration_harness.run(geometry_mode="metric_camera")
    inference_calls = first.estimator.calls
    metric_hashes = hash_stage(first.metric_stage_files)
    second = integration_harness.resume(
        first,
        virtual_baseline_mm=70.0,
        metric_convergence_distance=3.0,
        max_disparity_percent=1.0,
    )
    assert second.success
    assert second.estimator.calls == inference_calls
    assert hash_stage(second.metric_stage_files) == metric_hashes
    assert second.stereo_hashes != first.stereo_hashes


def test_retained_mode_switch_builds_only_missing_selected_stage(
    integration_harness
) -> None:
    metric = integration_harness.run(geometry_mode="metric_camera", keep_intermediates=True)
    metric_hashes = hash_stage(metric.metric_stage_files)
    relative = integration_harness.resume(metric, geometry_mode="relative")
    assert relative.success
    assert hash_stage(relative.metric_stage_files) == metric_hashes
    assert relative.relative_stage_exists
```

For every run, assert raw schema v3 member names, selected stage member/header contracts, SBS dimensions, clamp-summary fields in metric mode, and the frozen relative CPU hashes at the renderer boundary.

- [ ] **Step 6: Write mocked CLI and Web dispatch integration tests**

Patch `create_registered_depth_estimator` to return `_FakeMoGeEstimator` and use the same video-I/O harness:

```python
def test_cli_dispatch_reaches_metric_pipeline(cli_module, integration_harness) -> None:
    exit_code = integration_harness.run_cli(
        [
            "clip.mp4",
            "--depth-model-version",
            "moge2",
            "--model-size",
            "vitb",
            "--stereo-geometry-mode",
            "metric_camera",
            "--format",
            "side_by_side",
            "--no-distortion",
        ]
    )
    assert exit_code == 0
    assert integration_harness.completed_metric_output()


def test_web_dispatch_reaches_relative_pipeline(app_client, integration_harness) -> None:
    response = integration_harness.run_web(
        depth_model_version="moge2",
        model_size="vits",
        stereo_geometry_mode="relative",
    )
    assert response.status_code == 200
    integration_harness.join_background_task()
    assert integration_harness.completed_relative_output()
```

The Web harness executes the queued callable synchronously rather than mocking it away. Socket emissions are recorded; no network server is started.

- [ ] **Step 7: Write the failed no-retention mode-switch integration test**

Run metric mode with `keep_intermediates=false` and inject stereo failure immediately after completed metric stage 3. Assert raw payload NPZs were removed only after metric metadata validation, source frames and completed metric payloads remain, status is failed, and a relative resume report says `MoGe inference is required to build the selected geometry stage` without deleting either retained stage.

- [ ] **Step 8: Run the full mocked integration gate**

Run: `uv run pytest tests/integration/test_moge2_pipeline.py -v -m integration`

Expected: PASS without importing or downloading the real `moge` package.

- [ ] **Step 9: Commit live-preview and integration coverage**

```bash
git add src/depth_surge_3d/processing/frames/stereo_generator.py tests/unit/test_stereo_generator.py tests/unit/test_pipeline_orchestrator.py tests/integration/test_moge2_pipeline.py
git commit -m "test: cover MoGe metric pipeline and previews"
```

### Task 15: Document the Experimental Contract and Third-Party Licenses

**Files:**
- Modify: `README.md`
- Modify: `docs/INSTALLATION.md`
- Modify: `docs/PARAMETERS.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/USAGE.md`
- Modify: `docs/TROUBLESHOOTING.md`
- Modify: `example_settings.json`
- Modify: `CHANGELOG.md`
- Create: `THIRD_PARTY_NOTICES.md`
- Create: `tests/unit/test_moge2_docs.py`

**Interfaces:**
- Consumes: the shipped commands, settings, pins, stage names, warnings, and failure behavior.
- Produces: documentation that is mechanically checked against those contracts and makes no physical-correctness or temporal-stability claim.

- [ ] **Step 1: Write failing documentation contract tests**

```python
WARNING = (
    "MoGe-2 performs per-frame depth and focal estimation. Temporal stability "
    "on video is not guaranteed; depth or focal drift may be visible across frames."
)


@pytest.mark.parametrize(
    "path",
    [Path("README.md"), Path("docs/PARAMETERS.md"), Path("docs/TROUBLESHOOTING.md")],
)
def test_temporal_warning_is_verbatim(path: Path) -> None:
    assert WARNING in path.read_text(encoding="utf-8")


def test_installation_documents_only_supported_optional_command() -> None:
    text = Path("docs/INSTALLATION.md").read_text(encoding="utf-8")
    assert "uv sync --extra moge2" in text
    assert "MOGE_RESOLUTION_LEVEL" not in text


def test_architecture_names_both_independent_stage3_formats() -> None:
    text = Path("docs/ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "03_disparity_maps" in text
    assert "03_metric_geometry" in text
    assert "raw schema v3" in text


def test_notices_retain_both_upstream_licenses() -> None:
    text = Path("THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "Microsoft MoGe" in text and "MIT License" in text
    assert "DINOv2" in text and "Apache License 2.0" in text
    assert "925b8ed835a7a9cdb7578ba15c658a0afc969030" in text
```

```python
def test_example_settings_use_safe_metric_defaults_without_internal_level() -> None:
    settings = json.loads(Path("example_settings.json").read_text(encoding="utf-8"))
    assert settings["stereo_geometry_mode"] == "relative"
    assert settings["virtual_baseline_mm"] == 63.0
    assert settings["metric_convergence_distance"] == "auto"
    assert settings["max_disparity_percent"] == 2.0
    assert "moge_resolution_level" not in settings


def test_documentation_lists_all_immutable_weight_pins() -> None:
    text = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in ("docs/INSTALLATION.md", "docs/PARAMETERS.md")
    )
    for repository, revision in (
        ("Ruicheng/moge-2-vits-normal", "679230677b4d282c6f304189a93e98e14f085902"),
        ("Ruicheng/moge-2-vitb-normal", "54ad3a693e61907ea4633d13dec6ee682fa09419"),
        ("Ruicheng/moge-2-vitl", "39c4d5e957afe587e04eec59dc2bcc3be5ecd968"),
    ):
        assert repository in text
        assert revision in text
```

- [ ] **Step 2: Run the docs tests and confirm the contracts are missing**

Run: `uv run pytest tests/unit/test_moge2_docs.py -v`

Expected: FAIL until the documentation and notice file are updated.

- [ ] **Step 3: Update installation and backend selection documentation**

Document the default install separately from the optional MoGe install. The only supported uv command is:

```bash
uv sync --extra moge2
```

List Small `vits` 35M, Base `vitb` 104M default, and Large `vitl` 326M with their exact repository/revision pairs. Explain immutable source/weight resolution, offline-cache behavior, the concrete `model.pt` requirement, CPU float32 versus CUDA float16, and explicit OOM failure without fallback. Retain pip-oriented `requirements-moge2.txt` as a separate manual-install path, not as an alternative uv command.

- [ ] **Step 4: Update parameters, usage, and example settings**

Document the six CLI flags from the design, all four geometry settings with exact defaults/bounds, and two complete commands:

```bash
uv run depth-surge-3d clip.mp4 --depth-model-version moge2 --model-size vitb

uv run depth-surge-3d clip.mp4 --depth-model-version moge2 --model-size vitb \
  --stereo-geometry-mode metric_camera --format side_by_side --no-distortion \
  --virtual-baseline-mm 63 --metric-convergence-distance auto \
  --max-disparity-percent 2
```

State that relative remains the default and existing flat SBS remains the main playback path. Explain `virtual_baseline_mm` as a virtual capture-camera baseline, not viewer IPD. Include the exact warning, Experimental label, square-SAR constraint, letterbox caveat, flat rectilinear SBS restriction, total-disparity cap, automatic clip-global convergence, and identical crop/resize rule. Do not claim calibrated physical scale, improved quality, comfort, or temporal stability.

- [ ] **Step 5: Update architecture, troubleshooting, changelog, and attribution**

Architecture documents registry ownership, typed camera data, raw schema v2 read/v3 write, both stage-3 directories, selected-only derivation, common `StereoGeometryFrame`, pre-crop/final disparity transform, clip-global convergence barrier, independent resume fingerprints, exact disk bound, and successful-only cleanup.

Troubleshooting adds exact remedies for missing extra, missing pinned snapshot or `model.pt`, invalid/explicit non-square SAR, no convergence samples, preflight exhaustion, post-preflight ENOSPC, OOM without fallback, high clamp fraction, temporal depth/focal drift, and letterbox bias. Changelog calls the mode Experimental and links the release checklist. `THIRD_PARTY_NOTICES.md` records the pinned MoGe source under MIT and its bundled DINOv2 code under Apache-2.0, with upstream source/license links and no implication that the project's MIT license replaces those notices.

- [ ] **Step 6: Run docs and JSON validation**

Run: `uv run pytest tests/unit/test_moge2_docs.py tests/unit/test_settings.py -v`

Run: `uv run python -m json.tool example_settings.json > $null`

Expected: PASS and valid JSON.

- [ ] **Step 7: Commit documentation and attribution**

```bash
git add README.md docs/INSTALLATION.md docs/PARAMETERS.md docs/ARCHITECTURE.md docs/USAGE.md docs/TROUBLESHOOTING.md example_settings.json CHANGELOG.md THIRD_PARTY_NOTICES.md tests/unit/test_moge2_docs.py
git commit -m "docs: document experimental MoGe metric SBS"
```

### Task 16: Add the Three-Variant Real-Model Release Runner

**Files:**
- Create: `scripts/verify_moge2_release.py`
- Create: `docs/release/moge2-release-checklist.md`
- Create: `tests/unit/test_moge2_release_script.py`

**Interfaces:**
- Consumes: one JSON corpus configuration, all three immutable registry variants, a CUDA-capable release machine, and the production image/video pipeline.
- Produces: source-hashed A/B media plus machine-readable and Markdown evidence; it is an explicit release command and is not collected by ordinary pytest.

- [ ] **Step 1: Define and test the corpus configuration contract**

The script accepts `--corpus-config`, `--output-dir`, `--device {cpu,cuda}`, and `--depth-resolution`. It does not accept a model subset or MoGe resolution level. The JSON file has this exact shape:

```json
{
  "fixed_image": {
    "path": "D:/moge2-corpus/fixed-image.png",
    "sha256": "64 lowercase hexadecimal characters"
  },
  "clips": [
    {
      "id": "indoor-near",
      "path": "D:/moge2-corpus/indoor-near.mp4",
      "sha256": "64 lowercase hexadecimal characters",
      "static_roi_xywh": [120, 80, 160, 160]
    },
    {
      "id": "outdoor-far",
      "path": "D:/moge2-corpus/outdoor-far.mp4",
      "sha256": "64 lowercase hexadecimal characters",
      "static_roi_xywh": [300, 120, 200, 180]
    },
    {
      "id": "scene-cut",
      "path": "D:/moge2-corpus/scene-cut.mp4",
      "sha256": "64 lowercase hexadecimal characters",
      "static_roi_xywh": [160, 90, 120, 120]
    }
  ]
}
```

The quoted hash strings above describe the schema; the release operator supplies real hashes. The loader rejects missing/extra top-level keys, duplicate or noncanonical clip IDs, absent files, checksum mismatch, non-positive/out-of-bounds ROIs, fewer or more than the three required clips, and a source with explicit non-square SAR.

```python
def test_release_config_requires_exact_named_corpus(tmp_path) -> None:
    config = write_release_config(tmp_path)
    loaded = load_corpus_config(config)
    assert [clip.clip_id for clip in loaded.clips] == [
        "indoor-near",
        "outdoor-far",
        "scene-cut",
    ]


def test_release_runner_has_no_variant_or_resolution_level_escape_hatch() -> None:
    parser = create_argument_parser()
    destinations = {action.dest for action in parser._actions}
    assert "variant" not in destinations
    assert "moge_resolution_level" not in destinations
```

- [ ] **Step 2: Write failing exact-variant and evidence-schema tests**

```python
def test_release_variants_are_registry_pins_in_required_order() -> None:
    assert release_variants() == (
        ("vits", "Ruicheng/moge-2-vits-normal", "679230677b4d282c6f304189a93e98e14f085902"),
        ("vitb", "Ruicheng/moge-2-vitb-normal", "54ad3a693e61907ea4633d13dec6ee682fa09419"),
        ("vitl", "Ruicheng/moge-2-vitl", "39c4d5e957afe587e04eec59dc2bcc3be5ecd968"),
    )


def test_fake_release_report_contains_every_required_measurement(
    fake_release_runner, release_config
) -> None:
    report = fake_release_runner.run(release_config)
    assert [item["model_size"] for item in report["variants"]] == [
        "vits", "vitb", "vitl"
    ]
    required_variant_fields = {
        "repository",
        "revision",
        "load_seconds",
        "peak_vram_bytes",
        "fixed_image_native_shape",
        "fixed_image_focal_x_normalized",
        "fixed_image_valid_metric_pixels",
        "clips",
    }
    required_clip_fields = {
        "inference_seconds_per_frame",
        "focal_min",
        "focal_max",
        "focal_stddev",
        "roi_metric_depth_mean_per_frame",
        "roi_metric_depth_stddev",
        "roi_output_disparity_mean_per_frame",
        "roi_output_disparity_stddev",
        "relative_hole_fraction",
        "metric_hole_fraction",
        "metric_clamped_fraction_per_frame",
        "relative_output",
        "metric_output",
    }
    for variant in report["variants"]:
        assert required_variant_fields <= variant.keys()
        for clip in variant["clips"]:
            assert required_clip_fields <= clip.keys()
```

- [ ] **Step 3: Run release-script unit tests and verify the module is missing**

Run: `uv run pytest tests/unit/test_moge2_release_script.py -v`

Expected: FAIL because the release runner does not exist.

- [ ] **Step 4: Implement deterministic measurement helpers**

Load each model once in `vits`, `vitb`, `vitl` order. Around CUDA measurements, call `torch.cuda.synchronize()`, `torch.cuda.reset_peak_memory_stats()`, and `torch.cuda.max_memory_allocated()`. Use `time.perf_counter()` for load and per-frame inference timing. Validate the fixed image result has a positive finite focal scalar and at least one finite positive metric-depth pixel.

For each clip, run inference once and reuse the same raw schema-v3 data to render `relative` and `metric_camera`; do not infer twice. Use default baseline, auto convergence, 2 percent cap, SBS packing, no distortion, and identical output/crop settings. Compute:

- focal min/max/population standard deviation across source frames;
- per-frame mean positive metric depth inside the configured static ROI and its population standard deviation;
- per-frame mean final-coordinate total disparity inside that ROI and its population standard deviation;
- invalid output pixels divided by total output-mask pixels for each mode;
- ordered metric per-frame clamped fractions from persisted sidecars.

No metric has a pass threshold except finite/nonempty structural checks. Store numeric JSON values, not formatted strings.

- [ ] **Step 5: Write atomic evidence and A/B outputs**

Use this layout under `--output-dir`:

```text
report.json
report.md
vits/
  fixed-image-depth.npz
  indoor-near-relative.mp4
  indoor-near-metric-camera.mp4
  outdoor-far-relative.mp4
  outdoor-far-metric-camera.mp4
  scene-cut-relative.mp4
  scene-cut-metric-camera.mp4
vitb/
  fixed-image-depth.npz
  indoor-near-relative.mp4
  indoor-near-metric-camera.mp4
  outdoor-far-relative.mp4
  outdoor-far-metric-camera.mp4
  scene-cut-relative.mp4
  scene-cut-metric-camera.mp4
vitl/
  fixed-image-depth.npz
  indoor-near-relative.mp4
  indoor-near-metric-camera.mp4
  outdoor-far-relative.mp4
  outdoor-far-metric-camera.mp4
  scene-cut-relative.mp4
  scene-cut-metric-camera.mp4
```

Write each media/report to a sibling temporary path and replace only after validation. `report.json` records tool schema version, UTC timestamp, project Git commit, dirty-tree boolean, OS/Python/PyTorch/CUDA/GPU identities, MoGe source commit, fixed adapter level `9`, depth resolution, every input SHA-256, active projection settings, every metric, output SHA-256, and per-run failures. `report.md` renders the same facts in tables plus unchecked human-inspection rows for edge tearing, foreground sign, scale pumping, focal breathing, convergence placement, and viewing discomfort. It explicitly states that observations are not portable thresholds and do not establish physical calibration, better quality, comfort, or temporal stability.

Each `fixed-image-depth.npz` contains exactly `depth.npy` float32 `[H,W]`, `valid.npy` Boolean `[H,W]`, and zero-dimensional float32 `focal_x_normalized.npy`; invalid depth is zeroed in this evidence artifact. Validate exact members and dtypes before recording its hash.

If any variant or clip fails, keep already committed evidence, record the failure, do not write a top-level `status="complete"`, and exit nonzero. A complete run sets `status="complete"` only after every listed file exists and matches its recorded hash.

- [ ] **Step 6: Document the non-CI release procedure**

`docs/release/moge2-release-checklist.md` gives corpus capture rules, checksum creation, required static ROI selection, free-disk/VRAM preparation, the exact install command, the runner command, artifact review, and sign-off. The execution commands are:

```powershell
uv sync --extra moge2
uv run --extra moge2 python scripts/verify_moge2_release.py `
  --corpus-config D:\moge2-corpus\corpus.json `
  --output-dir artifacts\moge2-release\2026-08-16 `
  --device cuda `
  --depth-resolution 1080
```

The checklist requires all 18 A/B clip videos (three clips, two modes, three variants), all three fixed-image depth files, complete JSON, and human sign-off. CI runs only the fake unit tests; it never downloads these weights or media.

- [ ] **Step 7: Run release-script unit tests and static help**

Run: `uv run pytest tests/unit/test_moge2_release_script.py -v`

Run: `uv run python scripts/verify_moge2_release.py --help`

Expected: PASS/help exits zero without importing `moge` or resolving weights.

- [ ] **Step 8: Commit the release runner**

```bash
git add scripts/verify_moge2_release.py docs/release/moge2-release-checklist.md tests/unit/test_moge2_release_script.py
git commit -m "test: add MoGe three-variant release evidence runner"
```

- [ ] **Step 9: Execute and inspect the real release gate on the NVIDIA release machine**

Run the checklist command against the approved checksummed corpus. Expected: `report.json` has `status="complete"`, all three immutable variants pass the image sanity checks, all 18 A/B videos validate, and `report.md` is manually completed. Record the report SHA-256 and artifact location in the release ticket; do not commit large media to Git.

### Task 17: Run the Final Compatibility and Acceptance Gate

**Files:**
- Verify only; modify a file only to correct a failure attributable to this feature.

**Interfaces:**
- Consumes: all three implementation slices and the unchanged legacy test corpus.
- Produces: one auditable pass/fail record separating default-install CI, optional-extra tests, CUDA renderer parity, and real-model release evidence.

- [ ] **Step 1: Verify default installation does not import MoGe**

Run:

```powershell
uv sync
uv run python -c "import importlib.util; import src.depth_surge_3d.inference.depth.backend_registry as r; assert importlib.util.find_spec('moge') is None; assert not r.backend_availability('moge2').available"
```

Expected: PASS and the registry remains importable without the optional package. If the developer machine already has an externally installed `moge`, run this assertion in a fresh uv environment instead of weakening it.

- [ ] **Step 2: Verify optional installation and immutable source pin**

Run:

```powershell
uv sync --extra moge2
uv run --extra moge2 python -c "from moge.model.v2 import MoGeModel; from src.depth_surge_3d.inference.depth.backend_registry import backend_availability; assert MoGeModel is not None; assert backend_availability('moge2').available"
uv lock --check
```

Inspect `uv.lock` and assert its MoGe Git source resolves to `925b8ed835a7a9cdb7578ba15c658a0afc969030`, not a branch or tag.

- [ ] **Step 3: Run formatting, lint, and scoped typing**

Run:

```powershell
$base = git merge-base HEAD main
$pythonFiles = @(git diff --name-only --diff-filter=ACMR "$base..HEAD" -- '*.py')
if ($pythonFiles.Count -eq 0) { throw "No changed Python files found" }
uv run black --check $pythonFiles
uv run flake8 $pythonFiles
uv run mypy src/depth_surge_3d/inference/depth/backend_registry.py src/depth_surge_3d/inference/depth/video_depth_estimator_moge2.py src/depth_surge_3d/processing/frames/metric_geometry.py src/depth_surge_3d/rendering/stereo_geometry.py
```

Expected: all commands exit zero. Apply Black only to files changed by this feature, then rerun tests if formatting changes code.

- [ ] **Step 4: Run the complete CPU unit and mocked integration suite**

Run:

```powershell
uv run pytest tests/unit -q
uv run pytest tests/integration -q -m integration
```

Expected: PASS. Ordinary tests do not resolve real MoGe weights or require the release corpus.

- [ ] **Step 5: Re-run the fixed relative byte gate explicitly**

Run:

```powershell
uv run pytest tests/unit/test_stereo_renderer.py tests/unit/test_forward_splat.py tests/unit/test_stereo_edge_coverage.py -q
```

Expected hashes:

```text
left_image       48f5e73497d9adea81c5a0dc1444dfa23776f71af291c4a8995470435052d0ce
right_image      5fbd0cb67a9a31f7f18957d597979a0c88db49d5df7b29e2ce731bfd7e3aae05
left_valid       2be1e207cb3c363ebec163e3de22b4b60a5357f10f7e3afc1e10a7e47c4dc03a
right_valid      2be1e207cb3c363ebec163e3de22b4b60a5357f10f7e3afc1e10a7e47c4dc03a
left_hole        57ffc9ca3beb6ee6226c28248ab9c77b2076ef6acffba839cec21fac28a8fd1f
right_hole       57ffc9ca3beb6ee6226c28248ab9c77b2076ef6acffba839cec21fac28a8fd1f
```

Any change is a release blocker, not a fixture-update opportunity.

- [ ] **Step 6: Run CUDA parity tests on the NVIDIA machine**

Run:

```powershell
uv run pytest tests/unit/test_forward_splat.py::test_cuda_matches_cpu_byte_exactly tests/unit/test_forward_splat.py::test_cuda_packed_collisions_are_repeatable tests/unit/test_stereo_renderer.py::test_renderer_cpu_cuda_parity_is_byte_exact tests/unit/test_stereo_edge_coverage.py::test_complete_fixed_uint8_corpus_is_repeatable_and_cpu_cuda_exact -v
```

Expected: all four execute rather than skip and pass byte-for-byte. Then execute Task 16's real three-variant release command.

- [ ] **Step 7: Audit public-surface and storage invariants**

Run:

```powershell
$publicDocs = @('README.md', 'docs/INSTALLATION.md', 'docs/PARAMETERS.md', 'docs/ARCHITECTURE.md', 'docs/USAGE.md', 'docs/TROUBLESHOOTING.md', 'docs/release/moge2-release-checklist.md')
rg -n "MOGE_RESOLUTION_LEVEL|resolution_level" depth_surge_3d.py app.py templates example_settings.json $publicDocs
rg -n "MoGe-2 performs per-frame depth and focal estimation" depth_surge_3d.py app.py templates $publicDocs
$base = git merge-base HEAD main
git diff --unified=0 "$base..HEAD" | rg "^\+.*(TBD|TODO|temporary fallback|fall back to.*(v2|v3|relative))"
```

Expected: the first search finds no public control or setting; documentation may describe that the fixed internal level is reported but not configurable. The exact warning exists on CLI, Web, and documentation surfaces. The final search finds no unfinished MoGe implementation or silent backend/geometry fallback.

- [ ] **Step 8: Inspect final history and worktree scope**

Run:

```powershell
git status --short
git log --oneline --decorate -20
$base = git merge-base HEAD main
git diff --check "$base..HEAD"
```

Expected: only planned files are changed, each task has a focused commit, no large release media is tracked, and `git diff --check` reports no whitespace errors. Preserve any unrelated user changes from the original workspace; never copy or commit them into the feature worktree.

---

## Completion Criteria

Implementation is complete only when Tasks 1-17 are checked, default and optional installation gates pass, the entire CPU suite passes, CUDA parity executes on the available NVIDIA system, and the release ticket contains a complete three-variant report plus reviewed A/B media. Unit and smoke tests prove contract correctness; they do not by themselves justify removing the Experimental label or claiming physical calibration, improved quality, temporal stability, or viewing comfort.
