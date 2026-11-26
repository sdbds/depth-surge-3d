"""Declared diagnostic names must not replace the final source-content audit."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from src.depth_surge_3d.io.canonical_json import CanonicalJsonError
from src.depth_surge_3d.io.durable_publish import publication_temporary_path
from src.depth_surge_3d.io.streamed_canonical_json import build_streamed_canonical_object
from src.depth_surge_3d.processing.frames import stereo_generator, stereo_input_manifest
from src.depth_surge_3d.rendering.quality import diagnostic_transaction as transaction
from src.depth_surge_3d.rendering.stereo_renderer import StereoRenderer
from tests.unit.test_quality_diagnostics import _publication, _streaming_context
from tests.unit.test_stereo_generator import (
    _make_manifest_inputs,
    _make_metric_manifest_inputs,
    _settings,
)


def _names_context(manifest, directories):
    provider = manifest.open_provider(
        left_dir=directories["left_frames"],
        right_dir=directories["right_frames"],
    )
    plan = stereo_generator.audit_stereo_provider(
        provider, lambda _item: stereo_generator.StereoAction.RENDER
    )
    context = stereo_generator.StereoPairGenerator._streaming_diagnostics_context(
        manifest=manifest,
        provider=provider,
        render_mode="fast",
        geometry_mode=manifest.geometry_mode,
        occlusion_fill="background",
        keep_masks=False,
        rgb_stage_fingerprint="a" * 64,
        quality_content=None,
        left_dir=directories["left_frames"],
        right_dir=directories["right_frames"],
    )
    return context, provider, plan


@pytest.mark.parametrize("geometry", ["relative", "metric"])
def test_diagnostics_name_replay_does_not_reopen_source_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, geometry: str
) -> None:
    fixture = _make_manifest_inputs if geometry == "relative" else _make_metric_manifest_inputs
    manifest, directories = fixture(tmp_path, count=2)
    context, provider, plan = _names_context(manifest, directories)
    opened = []
    original = stereo_input_manifest._actual_identity_matches

    def observe(*args, **kwargs):
        opened.append(args[1]["relative_path"])
        return original(*args, **kwargs)

    monkeypatch.setattr(stereo_input_manifest, "_actual_identity_matches", observe)
    expected = (
        ["frame_000001", "frame_000002"]
        if geometry == "relative"
        else ["frame_000000", "frame_000001"]
    )
    for _ in range(4):
        assert list(context.frame_names()) == expected
    assert opened == []

    stereo_generator.validate_stereo_provider_replay(provider, plan, mutation_started=True)
    assert len(opened) == 4
    changed = tmp_path / "00_original_frames" / f"{expected[-1]}.png"
    changed.write_bytes(changed.read_bytes() + b"changed")
    with pytest.raises(stereo_generator.StereoStageInputChangedError, match="payload changed"):
        stereo_generator.validate_stereo_provider_replay(provider, plan, mutation_started=True)


@pytest.mark.parametrize("mutation", ["identity", "order", "truncated", "extra"])
def test_diagnostics_names_reauthenticate_the_manifest_on_every_replay(
    tmp_path: Path, mutation: str
) -> None:
    manifest, directories = _make_manifest_inputs(tmp_path, count=2)
    context, _provider, _plan = _names_context(manifest, directories)
    assert list(context.frame_names()) == ["frame_000001", "frame_000002"]
    rows = [json.loads(line) for line in manifest.path.read_bytes().splitlines()]
    if mutation == "identity":
        rows[1]["guide"]["sha256"] = "e" * 64
    elif mutation == "order":
        rows[1], rows[2] = rows[2], rows[1]
    elif mutation == "truncated":
        rows.pop()
    else:
        rows.append(rows[-1])
    manifest.path.write_bytes(
        b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
            for row in rows
        )
    )
    with pytest.raises(stereo_generator.StereoInputIdentityMismatch):
        list(context.frame_names())


def test_diagnostics_names_reject_a_descriptor_count_mismatch(tmp_path: Path) -> None:
    manifest, directories = _make_manifest_inputs(tmp_path, count=2)
    context, _provider, _plan = _names_context(manifest, directories)
    context = replace(context, frame_count=3)
    with pytest.raises(ValueError, match="frame count"):
        list(transaction._iter_streaming_frame_names(context))


@pytest.mark.parametrize(
    ("geometry", "payload", "mutation"),
    [
        ("relative", "guide", "missing"),
        ("metric", "geometry", "corrupt"),
        ("relative", "manifest", "missing"),
        ("relative", "guide", "corrupt"),
    ],
)
def test_provider_replay_classifies_source_io_and_format_failures(
    tmp_path: Path, geometry: str, payload: str, mutation: str
) -> None:
    fixture = _make_manifest_inputs if geometry == "relative" else _make_metric_manifest_inputs
    manifest, directories = fixture(tmp_path, count=1)
    _context, provider, _plan = _names_context(manifest, directories)
    records = manifest.replay_declared_records()
    record = next(records)
    records.close()
    path = {
        "guide": record.guide_path,
        "geometry": record.geometry_path,
        "manifest": manifest.path,
    }[payload]
    if mutation == "missing":
        path.unlink()
    else:
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
    with pytest.raises(stereo_generator.StereoInputIdentityMismatch) as captured:
        list(provider.replay())
    assert captured.value.__cause__ is not None


@pytest.mark.parametrize(
    "error", [MemoryError("allocation denied"), stereo_generator.StereoControlBudgetError("budget")]
)
def test_provider_replay_does_not_reclassify_resource_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    manifest, directories = _make_manifest_inputs(tmp_path)
    _context, provider, _plan = _names_context(manifest, directories)

    def fail(_self):
        raise error

    monkeypatch.setattr(stereo_input_manifest._ManifestReplay, "_iter_items", fail)
    with pytest.raises(type(error)) as captured:
        list(provider.replay())
    assert captured.value is error


@pytest.mark.parametrize("transient", [False, True])
@pytest.mark.parametrize("reject", [False, True])
def test_diagnostics_final_audit_runs_after_metadata_sync_and_before_replace(
    tmp_path: Path, transient: bool, reject: bool
) -> None:
    context = _streaming_context(transaction, tmp_path, masks=False)
    publication = _publication(transaction, tmp_path, keep_masks=False, frame_name="frame_0000")
    transaction.begin_streaming_diagnostics_stage(context)
    prepared = transaction.prepare_quality_frame(publication)
    if not transient:
        transaction.publish_quality_frame(publication)
    root = tmp_path / "04_stereo_diagnostics"
    metadata = root / "metadata.json"
    temporary = publication_temporary_path(metadata)
    building = metadata.read_bytes()
    events = []
    audits = []

    def complete(**kwargs):
        if not transient:
            return transaction.complete_streaming_diagnostics_stage(context, **kwargs)
        spool = transaction.TransientDiagnosticsSpool(context)
        try:
            spool.append("frame_0000", prepared.stats_raw)
            return spool.complete(**kwargs)
        finally:
            spool.abort()

    def audit():
        assert events[-1] == "before_root_metadata_replace"
        assert metadata.read_bytes() == building
        assert temporary.is_file() and temporary.stat().st_size > 0
        assert (root / "stereo_coverage_summary.json").is_file()
        audits.append("audited")
        if reject:
            raise ValueError("source changed before complete publication")

    if reject:
        with pytest.raises(ValueError, match="source changed before complete publication"):
            complete(before_complete_publish=audit, fault=events.append)
        assert metadata.read_bytes() == building
        assert not temporary.exists()
        complete()
    else:
        complete(before_complete_publish=audit, fault=events.append)
    assert audits == ["audited"]
    assert json.loads(metadata.read_bytes())["status"] == "complete"
    assert not temporary.exists()


def test_invalid_canonical_replay_never_reaches_prepublication_audit(tmp_path: Path) -> None:
    values = ["first"]
    document = build_streamed_canonical_object(
        static_fields={"status": "complete"},
        array_key="names",
        items=lambda: iter(values),
    )
    destination = tmp_path / "metadata.json"
    destination.write_bytes(b"previous-generation")
    values.append("unexpected")
    audits = []
    with pytest.raises(CanonicalJsonError, match="replay changed"):
        document.publish(destination, before_replace=lambda: audits.append("unexpected audit"))
    assert audits == []
    assert destination.read_bytes() == b"previous-generation"
    assert not publication_temporary_path(destination).exists()


@pytest.mark.parametrize("geometry", ["relative", "metric"])
@pytest.mark.parametrize("render_mode", ["fast", "quality"])
@pytest.mark.parametrize("payload", ["guide", "geometry"])
@pytest.mark.parametrize("mutation", ["replacement", "missing"])
def test_noop_rechecks_source_content_after_aggregate_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    geometry: str,
    render_mode: str,
    payload: str,
    mutation: str,
) -> None:
    fixture = _make_manifest_inputs if geometry == "relative" else _make_metric_manifest_inputs
    manifest, directories = fixture(tmp_path, count=2)
    settings = _settings(
        stereo_strength=0.0,
        stereo_render_mode=render_mode,
        stereo_geometry_mode="relative" if geometry == "relative" else "metric_camera",
        metric_convergence_distance="auto",
    )
    generator = stereo_generator.StereoPairGenerator(renderer=StereoRenderer(device="cpu"))
    assert generator.create_stereo_pairs_from_manifest(manifest, directories, settings)
    retained = {
        path: path.read_bytes()
        for directory in ("04_left_frames", "04_right_frames", "04_stereo_diagnostics")
        for path in (tmp_path / directory).rglob("*")
        if path.is_file()
    }
    records = manifest.replay_declared_records()
    record = next(records)
    records.close()
    changed_path = record.guide_path if payload == "guide" else record.geometry_path
    original = stereo_generator.streaming_diagnostics_aggregates_match
    checks = []

    def change_during_final_aggregate(context):
        checks.append("aggregate")
        if len(checks) == 2:
            if mutation == "missing":
                changed_path.unlink()
            else:
                raw = bytearray(changed_path.read_bytes())
                raw[-1] ^= 1
                replacement = changed_path.with_suffix(".replacement")
                replacement.write_bytes(raw)
                replacement.replace(changed_path)
        return original(context)

    monkeypatch.setattr(
        stereo_generator, "streaming_diagnostics_aggregates_match", change_during_final_aggregate
    )
    with pytest.raises(stereo_generator.StereoProviderChangedError):
        generator.create_stereo_pairs_from_manifest(manifest, directories, settings)
    assert checks == ["aggregate", "aggregate"]
    assert generator.last_pipeline_stats is None
    assert all(path.read_bytes() == expected for path, expected in retained.items())


def _change_consolidation_input(tmp_path: Path, manifest, payload: str, geometry: str) -> None:
    if payload == "manifest":
        rows = [json.loads(line) for line in manifest.path.read_bytes().splitlines()]
        rows[-1]["guide"]["sha256"] = "f" * 64
        manifest.path.write_bytes(
            b"".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
                for row in rows
            )
        )
    else:
        if payload == "guide":
            directory, suffix = "00_original_frames", ".png"
        elif geometry == "relative":
            directory, suffix = "03_disparity_maps", ".png"
        else:
            directory, suffix = "03_metric_geometry", ".npz"
        frame_name = "frame_000002" if geometry == "relative" else "frame_000001"
        path = tmp_path / directory / f"{frame_name}{suffix}"
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        replacement = path.with_suffix(".replacement")
        replacement.write_bytes(raw)
        replacement.replace(path)


def _at_consolidation_boundary(monkeypatch, boundary, corrupt):
    def observe_publish(original):
        def publish(document, path, **kwargs):
            fault = kwargs.get("fault")

            def observe(stage):
                if fault is not None:
                    fault(stage)
                if stage == boundary:
                    corrupt()

            kwargs["fault"] = observe
            return original(document, path, **kwargs)

        return publish

    for document_type in (
        transaction.StreamedCanonicalObject,
        transaction.StreamedCanonicalValue,
    ):
        monkeypatch.setattr(document_type, "publish", observe_publish(document_type.publish))


@pytest.mark.parametrize(
    ("transient", "render_mode", "geometry"),
    [
        (False, "fast", "relative"),
        (True, "fast", "relative"),
        (False, "quality", "relative"),
        (True, "quality", "relative"),
        (False, "fast", "metric"),
        (False, "quality", "metric"),
    ],
)
@pytest.mark.parametrize("payload", ["guide", "geometry", "manifest"])
@pytest.mark.parametrize(
    "boundary", ["after_root_summary_replace", "after_root_metadata_temporary_sync"]
)
def test_manifest_pipeline_rejects_sources_changed_during_consolidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transient: bool,
    render_mode: str,
    geometry: str,
    payload: str,
    boundary: str,
) -> None:
    fixture = _make_manifest_inputs if geometry == "relative" else _make_metric_manifest_inputs
    manifest, directories = fixture(tmp_path, count=2)
    sentinel = tmp_path / "user-note.txt"
    sentinel.write_bytes(b"unrelated user file")
    changed = []

    def corrupt():
        _change_consolidation_input(tmp_path, manifest, payload, geometry)
        changed.append(boundary)

    _at_consolidation_boundary(monkeypatch, boundary, corrupt)

    class CountingGenerator(stereo_generator.StereoPairGenerator):
        attempts = 0

        def _create_stereo_pairs_from_manifest_once(self, *args, **kwargs):
            self.attempts += 1
            return super()._create_stereo_pairs_from_manifest_once(*args, **kwargs)

    generator = CountingGenerator(renderer=StereoRenderer(device="cpu"))
    consumed = []

    def consume(name, left, right):
        assert left.shape == right.shape == (8, 16, 3)
        consumed.append(name)

    with pytest.raises(stereo_generator.StereoStageInputChangedError):
        generator.create_stereo_pairs_from_manifest(
            manifest,
            directories,
            _settings(
                stereo_strength=0.0,
                stereo_render_mode=render_mode,
                stereo_geometry_mode="relative" if geometry == "relative" else "metric_camera",
                metric_convergence_distance="auto",
            ),
            frame_consumer=consume if transient else None,
        )
    assert changed == [boundary]
    assert generator.attempts == 1
    assert generator.last_pipeline_stats.live_threads_after_join == 0
    metadata = tmp_path / "04_stereo_diagnostics" / "metadata.json"
    assert not metadata.exists() or json.loads(metadata.read_bytes())["status"] == "building"
    assert not publication_temporary_path(metadata).exists()
    assert sentinel.read_bytes() == b"unrelated user file"
    if transient:
        assert consumed == ["frame_000001", "frame_000002"]
