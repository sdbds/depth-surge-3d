from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from src.depth_surge_3d.core.settings import validate_settings
from src.depth_surge_3d.rendering.quality.diagnostic_transaction import (
    canonical_json_bytes,
    project_diagnostics_metadata,
    self_fingerprint,
)
from src.depth_surge_3d.rendering.quality.cleanup_transaction import (
    build_payload_pruned_document,
    CleanupTransactionError,
    commit_payload_pruned,
    run_authorized_cleanup,
)
from src.depth_surge_3d.processing.video.final_publication import (
    EncodingInputPublicationSource,
    FinalMediaAuditDispositionV1,
    FINAL_ENCODING_RESERVATION_RELATIVE_PATH,
    FinalEncodingDiskBudgetError,
    FinalEncodingReservationCapabilityV1,
    FinalPublicationValidationError,
    FinalPublicationTransactionV4,
    InjectedFinalPublicationCrash,
    build_final_video_manifest_v4,
    parse_final_video_manifest_v4,
    prepare_final_encoding_reservation_v1,
    load_final_encoding_reservation_v1,
    recover_final_encoding_terminal_retirement_v1,
    audit_final_media_v1,
)
from src.depth_surge_3d.processing.video import (
    final_publication as final_publication_module,
)
from src.depth_surge_3d.io import reservation_protocol
from src.depth_surge_3d.io.file_identity import capture_file_identity
from src.depth_surge_3d.io.operations import (
    commit_cleanup_pending_status,
    recover_processing_control_prefix_v1,
)
from src.depth_surge_3d.io.reservation_protocol import ReservationAllocationEvidence
from tests.unit.test_settings_artifact_v5 import _artifact


def _input_manifest_value(
    frame_count: int = 2,
    *,
    mode: str = "assembled_vr",
    first_index: int = 0,
) -> dict:
    frame_names = [
        f"frame_{index:06d}" for index in range(first_index, first_index + frame_count)
    ]

    def identities(parent: str) -> list[dict]:
        return [
            {
                "relative_path": f"{parent}/{name}.png",
                "sha256": f"{position + 1:064x}",
                "byte_count": 67 + position,
                "png_header": {
                    "width": 1920,
                    "height": 1080,
                    "bit_depth": 8,
                    "color_type": 2,
                },
            }
            for position, name in enumerate(frame_names)
        ]

    base: dict = {
        "algorithm_version": "encoding-input-sequence-v2",
        "frame_names": frame_names,
        "frames": identities("99_vr_frames") if mode == "assembled_vr" else None,
        "left": identities("04_left_frames") if mode == "direct_stereo" else None,
        "mode": mode,
        "right": identities("05_right_frames") if mode == "direct_stereo" else None,
        "schema_version": 1,
    }
    return {**base, "fingerprint": self_fingerprint(base)}


def _encode_input_manifest(value: dict) -> tuple[bytes, str]:
    fingerprint = self_fingerprint(value)
    value = {**value, "fingerprint": fingerprint}
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii"),
        fingerprint,
    )


def _input_manifest(frame_count: int = 12) -> tuple[bytes, str]:
    return _encode_input_manifest(_input_manifest_value(frame_count))


def _assembled_arguments(fingerprint: str, relative_path: str) -> tuple[str, ...]:
    return (
        "ffmpeg",
        "-i",
        f"@image-sequence:sha256={fingerprint}:view=frames",
        "@output:path64=" + FinalPublicationTransactionV4.path64(relative_path),
    )


def test_stream_input_manifest_validates_schema_and_settings_cardinality(tmp_path):
    raw, fingerprint = _input_manifest(frame_count=2)
    path = tmp_path / "encoding_input_manifest.json"
    path.write_bytes(raw)

    assert final_publication_module._stream_input_manifest_identity(
        tmp_path,
        path,
        expected_frame_count=2,
    ) == (fingerprint, hashlib.sha256(raw).hexdigest(), len(raw))

    with pytest.raises(
        FinalPublicationValidationError,
        match="settings frame count",
    ):
        final_publication_module._stream_input_manifest_identity(
            tmp_path,
            path,
            expected_frame_count=1,
        )


def test_stream_input_manifest_rejects_name_replacement_after_handle_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.depth_surge_3d.io.nofollow_io import AcquiredRoot

    raw, _fingerprint = _input_manifest(frame_count=2)
    path = tmp_path / "encoding_input_manifest.json"
    path.write_bytes(raw)
    parked = tmp_path / "parked-input-manifest.json"
    original_identity = AcquiredRoot.persistent_regular_identity
    injected = False

    def replace_before_name_reattest(self, candidate, *, missing_ok=False):
        nonlocal injected
        if not injected and Path(candidate) == path:
            injected = True
            path.rename(parked)
            path.write_bytes(raw)
        return original_identity(self, candidate, missing_ok=missing_ok)

    monkeypatch.setattr(
        AcquiredRoot,
        "persistent_regular_identity",
        replace_before_name_reattest,
    )

    with pytest.raises(FinalPublicationValidationError, match="name changed"):
        final_publication_module._stream_input_manifest_identity(
            tmp_path,
            path,
            expected_frame_count=2,
        )

    assert injected
    assert parked.read_bytes() == raw
    assert path.read_bytes() == raw


def test_stream_input_manifest_rejects_symlink_before_open(tmp_path: Path) -> None:
    raw, _fingerprint = _input_manifest(frame_count=2)
    victim = tmp_path / "victim.json"
    victim.write_bytes(raw)
    path = tmp_path / "encoding_input_manifest.json"
    try:
        path.symlink_to(victim)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    with pytest.raises(FinalPublicationValidationError, match="no-follow"):
        final_publication_module._stream_input_manifest_identity(
            tmp_path,
            path,
            expected_frame_count=2,
        )

    assert victim.read_bytes() == raw


def test_capability_manifest_rejects_name_replacement_after_handle_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.depth_surge_3d.io.file_identity import (
        capture_directory_identity,
        capture_file_identity,
    )
    from src.depth_surge_3d.io.nofollow_io import AcquiredRoot

    raw, fingerprint = _input_manifest(frame_count=2)
    path = tmp_path / "encoding_input_manifest.json"
    path.write_bytes(raw)
    parked = tmp_path / "parked-input-manifest.json"
    source = EncodingInputPublicationSource(
        path=path,
        fingerprint=fingerprint,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        raw_byte_count=len(raw),
        job_root=tmp_path,
        job_root_identity=capture_directory_identity(tmp_path),
        source_parent_identity=capture_directory_identity(tmp_path),
        source_identity=capture_file_identity(path),
    )
    original_identity = AcquiredRoot.persistent_regular_identity
    identity_checks = 0

    def replace_after_read(self, candidate, *, missing_ok=False):
        nonlocal identity_checks
        if Path(candidate) == path:
            identity_checks += 1
            if identity_checks == 2:
                path.rename(parked)
                path.write_bytes(raw)
        return original_identity(self, candidate, missing_ok=missing_ok)

    monkeypatch.setattr(
        AcquiredRoot,
        "persistent_regular_identity",
        replace_after_read,
    )

    with pytest.raises(FinalPublicationValidationError, match="identity changed"):
        source.projection(expected_frame_count=2)

    assert identity_checks == 2
    assert parked.read_bytes() == raw
    assert path.read_bytes() == raw


@pytest.mark.parametrize(
    "case",
    [
        "unknown_root_key",
        "mode_shape",
        "frame_gap",
        "identity_filename",
        "identity_parent",
        "identity_unknown_key",
        "identity_hash",
        "identity_header",
        "array_cardinality",
        "image2_boundary",
    ],
)
def test_stream_input_manifest_rejects_rehashed_semantic_corruption(tmp_path, case):
    value = copy.deepcopy(_input_manifest_value(frame_count=2))
    if case == "unknown_root_key":
        value["unexpected"] = None
    elif case == "mode_shape":
        value["left"] = copy.deepcopy(value["frames"])
    elif case == "frame_gap":
        value["frame_names"][1] = "frame_000002"
        value["frames"][1]["relative_path"] = "99_vr_frames/frame_000002.png"
    elif case == "identity_filename":
        value["frames"][1]["relative_path"] = "99_vr_frames/not-the-frame.png"
    elif case == "identity_parent":
        value["frames"][1]["relative_path"] = "another/frame_000001.png"
    elif case == "identity_unknown_key":
        value["frames"][0]["unexpected"] = 1
    elif case == "identity_hash":
        value["frames"][0]["sha256"] = "A" * 64
    elif case == "identity_header":
        value["frames"][0]["png_header"]["bit_depth"] = 16
    elif case == "array_cardinality":
        value["frames"].pop()
    elif case == "image2_boundary":
        value = _input_manifest_value(frame_count=1, first_index=2_147_483_647)
    raw, _fingerprint = _encode_input_manifest(value)
    path = tmp_path / "encoding_input_manifest.json"
    path.write_bytes(raw)

    with pytest.raises(FinalPublicationValidationError):
        final_publication_module._stream_input_manifest_identity(
            tmp_path,
            path,
            expected_frame_count=2 if case != "image2_boundary" else 1,
        )


def _producer_artifact():
    return _artifact().with_producer_marker(
        generation="44" * 16,
        updated_at="2026-08-24T12:34:57.123456Z",
    )


def _pruning_artifact(frame_count: int = 1):
    video_properties = dict(_artifact().to_object()["video_properties"])
    video_properties["frame_count"] = frame_count
    return _artifact(
        video_properties=video_properties,
        processing_settings=validate_settings(
            {"keep_intermediates": False},
            source="explicit",
        ),
    )


def _complete_prune_diagnostics(root) -> None:
    diagnostics = root / "04_stereo_diagnostics"
    diagnostics.mkdir(parents=True)
    frames_jsonl = canonical_json_bytes({"frame_name": "frame_000000"}) + b"\n"
    summary = canonical_json_bytes({"schema_version": 1})
    (diagnostics / "stereo_coverage_frames.jsonl").write_bytes(frames_jsonl)
    (diagnostics / "stereo_coverage_summary.json").write_bytes(summary)
    metadata = {
        "schema_version": 1,
        "algorithm_version": "stereo-coverage-sidecar-v1",
        "status": "complete",
        "rgb_stage_fingerprint": "1" * 64,
        "source_guide_fingerprint": "2" * 64,
        "native_geometry_fingerprint": None,
        "quality_input_manifest_sha256": None,
        "frame_names": ["frame_000000"],
        "render_shape": [2, 3],
        "render_mode": "fast",
        "geometry_mode": "relative",
        "occlusion_fill": "background",
        "mask_payloads_enabled": False,
        "stats_schema_version": 1,
        "frame_manifest_schema_version": 1,
        "ordered_frame_manifest_fingerprint": "a" * 64,
        "frames_jsonl_sha256": hashlib.sha256(frames_jsonl).hexdigest(),
        "summary_sha256": hashlib.sha256(summary).hexdigest(),
        "metric_clamp_summary_sha256": None,
        "pruned_from": None,
        "final_video_manifest_sha256": None,
        "prune_entries": None,
    }
    metadata["fingerprint"] = self_fingerprint(metadata)
    (diagnostics / "metadata.json").write_bytes(canonical_json_bytes(metadata))
    for relative in (
        "00_original_frames",
        "04_left_frames",
        "04_right_frames",
        "04_stereo_diagnostics/frames",
    ):
        directory = root / relative
        directory.mkdir(parents=True)
        (directory / "payload.bin").write_bytes(relative.encode("ascii"))


def _upgrade_prune_diagnostics_to_quality(root) -> dict:
    diagnostics = root / "04_stereo_diagnostics"
    frame_name = "frame_000000"
    guide = {
        "byte_count": 1,
        "frame_name": frame_name,
        "png_header": {"bit_depth": 8, "color_type": 2, "height": 2, "width": 3},
        "relative_path": f"00_original_frames/{frame_name}.png",
        "sha256": "3" * 64,
    }
    geometry = {
        "byte_count": 1,
        "frame_name": frame_name,
        "payload_kind": "canonical_png",
        "png_header": {"bit_depth": 16, "color_type": 0, "height": 2, "width": 3},
        "relative_path": f"03_depth_canonical/{frame_name}.png",
        "sha256": "4" * 64,
    }
    scale_bits = "477fff00"
    source_fingerprint = hashlib.sha256(canonical_json_bytes([guide])).hexdigest()
    native_fingerprint = hashlib.sha256(
        canonical_json_bytes(
            {
                "geometry": [geometry],
                "geometry_mode": "relative",
                "relative_encoding_scale_float32_bits": scale_bits,
            }
        )
    ).hexdigest()
    content = {
        "algorithm_version": "quality-input-content-v4",
        "frame_names": [frame_name],
        "geometry": [geometry],
        "geometry_mode": "relative",
        "guides": [guide],
        "native_geometry_fingerprint": native_fingerprint,
        "relative_encoding_scale_float32_bits": scale_bits,
        "schema_version": 1,
        "source_guide_fingerprint": source_fingerprint,
    }
    content["fingerprint"] = self_fingerprint(content)
    content_raw = canonical_json_bytes(content)
    (diagnostics / "quality_input_manifest.json").write_bytes(content_raw)
    metadata_path = diagnostics / "metadata.json"
    metadata = json.loads(metadata_path.read_bytes())
    metadata.update(
        {
            "render_mode": "quality",
            "source_guide_fingerprint": source_fingerprint,
            "native_geometry_fingerprint": native_fingerprint,
            "quality_input_manifest_sha256": hashlib.sha256(content_raw).hexdigest(),
        }
    )
    metadata["fingerprint"] = self_fingerprint(metadata)
    metadata_path.write_bytes(canonical_json_bytes(metadata))
    return metadata


def _test_allocation(path, logical):
    return ReservationAllocationEvidence("test", logical, logical, f"test:{logical}")


def _test_allocation_counter(path):
    size = path.stat().st_size
    return ((size + 4095) // 4096) * 4096


def _install_schema5_job(root, artifact, *, initial=None) -> None:
    initial = initial or (artifact if artifact.revision == 0 else _artifact())
    reservation_protocol.bootstrap_settings_job_v1(
        job_root=root,
        artifact=initial,
        initial_settings_reservation_id="aa" * 16,
        allocation_adapter=_test_allocation,
        allocation_unit=4096,
        allocation_counter=_test_allocation_counter,
    )
    if artifact != initial:
        transition = reservation_protocol.SettingsArtifactTransactionV1.begin(
            job_root=root,
            current=initial,
            target=artifact,
            allocation_adapter=_test_allocation,
            allocation_unit=4096,
            allocation_counter=_test_allocation_counter,
            reservation_generation="bb" * 16,
            reservation_id="cc" * 16,
        )
        transition.commit()


def _published_pruning_reservation(root, *, diagnostic_frame_count: int = 1):
    artifact = _pruning_artifact(diagnostic_frame_count)
    _install_schema5_job(root, artifact)
    _complete_prune_diagnostics(root)
    if diagnostic_frame_count != 1:
        metadata_path = root / "04_stereo_diagnostics" / "metadata.json"
        metadata = json.loads(metadata_path.read_bytes())
        metadata["frame_names"] = [
            f"frame_{index:06d}" for index in range(diagnostic_frame_count)
        ]
        metadata["fingerprint"] = self_fingerprint(metadata)
        metadata_path.write_bytes(canonical_json_bytes(metadata))
    raw, fingerprint = _input_manifest(frame_count=diagnostic_frame_count)
    reservation = prepare_final_encoding_reservation_v1(
        job_root=root,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=len(raw),
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    producer = reservation.commit_producer_settings()
    assert reservation.input_manifest_transaction is not None
    reservation.input_manifest_transaction.stage_complete_full(raw)
    reservation.sibling_path.write_bytes(b"validated-pruning-recovery-video")
    FinalPublicationTransactionV4(
        job_root=root,
        settings=producer,
        sibling_relative_path=reservation.sibling_path.name,
        encoding_input_manifest_raw=raw,
        input_stage_fingerprint=fingerprint,
        resolved_encoding_arguments=_assembled_arguments(
            fingerprint,
            producer.to_object()["output_info"]["expected_final_relative_path"],
        ),
        container_validator=lambda _path: True,
        reservation=reservation,
        retire_reservation=False,
    ).publish()
    entries = reservation.commit_prune_markers()
    projection = project_diagnostics_metadata(
        root / "04_stereo_diagnostics" / "metadata.json"
    )
    final_raw = (root / "final_video_manifest.json").read_bytes()
    document = build_payload_pruned_document(
        projection,
        prune_entries=entries,
        final_video_manifest_sha256=hashlib.sha256(final_raw).hexdigest(),
    )
    transaction = reservation.diagnostics_payload_pruned_transaction
    assert transaction is not None
    return producer, reservation, transaction, document


def test_scalar_final_reservation_capability_authorizes_prune_and_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _producer, reservation, _transaction, _document = _published_pruning_reservation(
        tmp_path
    )
    capability = FinalEncodingReservationCapabilityV1.from_prepared(reservation)
    del reservation

    def materialize_with_test_allocation(self):
        settings = recover_processing_control_prefix_v1(
            self.job_root,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
        )
        assert settings is not None
        reopened = load_final_encoding_reservation_v1(
            self.job_root,
            settings,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
        )
        assert reopened.index.fingerprint == self.index_fingerprint
        return reopened

    monkeypatch.setattr(
        FinalEncodingReservationCapabilityV1,
        "materialize",
        materialize_with_test_allocation,
    )
    operations_module = __import__(
        "src.depth_surge_3d.io.operations",
        fromlist=["recover_processing_control_prefix_v1"],
    )
    monkeypatch.setattr(
        operations_module,
        "recover_processing_control_prefix_v1",
        lambda root: recover_processing_control_prefix_v1(
            root,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
        ),
    )
    monkeypatch.setattr(
        final_publication_module,
        "load_final_encoding_reservation_v1",
        lambda root, settings, **_kwargs: load_final_encoding_reservation_v1(
            root,
            settings,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
        ),
    )

    projection = commit_payload_pruned(tmp_path, reservation=capability)
    pending = commit_cleanup_pending_status(
        tmp_path / "processing-settings-v5.json",
        reservation=capability,
    )

    assert projection.status == "payload_pruned"
    assert pending.to_object()["metadata"]["cleanup_status"] == "pending"


def test_final_video_manifest_has_closed_canonical_self_fingerprint() -> None:
    manifest = build_final_video_manifest_v4(
        relative_path="result.mp4",
        video_sha256="12" * 32,
        video_byte_count=4217,
        input_stage_fingerprint="34" * 32,
        resolved_encoding_arguments=(
            "ffmpeg",
            "-i",
            "@image-sequence:sha256=" + "34" * 32 + ":view=frames",
            "@output:path64=cmVzdWx0Lm1wNA",
        ),
    )
    raw = manifest.encode()

    assert not raw.endswith(b"\n")
    assert parse_final_video_manifest_v4(raw) == manifest
    value = manifest.to_object()
    assert set(value) == {
        "schema_version",
        "algorithm_version",
        "relative_path",
        "sha256",
        "byte_count",
        "input_stage_fingerprint",
        "resolved_encoding_arguments",
        "encoding_settings_fingerprint",
        "fingerprint",
    }


@pytest.mark.parametrize(
    "arguments",
    [
        ("ffmpeg", "@output:path64=cmVzdWx0Lm1wNA"),
        (
            "ffmpeg",
            "@image-sequence:sha256=" + "34" * 32 + ":view=frames",
            "@output:path64=cmVzdWx0Lm1wNA",
        ),
        (
            "ffmpeg",
            "-i",
            "C:/foreign.png",
            "@output:path64=cmVzdWx0Lm1wNA",
        ),
        (
            "ffmpeg",
            "-i",
            "@image-sequence:sha256=" + "34" * 32 + ":view=frames",
            "@output:path64=cmVzdWx0Lm1wNA",
            "trailing.mp4",
        ),
    ],
)
def test_final_video_manifest_rejects_open_normalized_argv_grammar(arguments) -> None:
    with pytest.raises(FinalPublicationValidationError):
        build_final_video_manifest_v4(
            relative_path="result.mp4",
            video_sha256="12" * 32,
            video_byte_count=4217,
            input_stage_fingerprint="34" * 32,
            resolved_encoding_arguments=arguments,
        )


def test_final_media_audit_rejects_oversized_manifest_before_bulk_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _producer_artifact()
    expected = artifact.to_object()["output_info"]["expected_final_relative_path"]
    (tmp_path / expected).write_bytes(b"video")
    input_raw, _fingerprint = _input_manifest()
    (tmp_path / "encoding_input_manifest.json").write_bytes(input_raw)
    publication = tmp_path / "final_video_manifest.json"
    with publication.open("wb") as handle:
        handle.truncate(1024 * 1024 + 1)
    original_read_bytes = Path.read_bytes

    def reject_bulk_read(path: Path) -> bytes:
        if path == publication:
            raise AssertionError("oversized final manifest was bulk-read")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_bulk_read)

    assert audit_final_media_v1(
        tmp_path,
        artifact,
        container_validator=lambda _path: True,
    ) == FinalMediaAuditDispositionV1(
        final_video_present=True,
        final_video_authenticated=False,
        final_video_authentication_reason="publication_validation_failed",
    )


def test_successful_publication_orders_video_then_input_then_final_and_audits(
    tmp_path,
) -> None:
    artifact = _producer_artifact()
    expected = artifact.to_object()["output_info"]["expected_final_relative_path"]
    sibling = tmp_path / f".{expected}.encoding-v4.tmp"
    sibling.write_bytes(b"validated-mp4-fixture")
    input_raw, input_fingerprint = _input_manifest()
    events: list[str] = []
    transaction = FinalPublicationTransactionV4(
        job_root=tmp_path,
        settings=artifact,
        sibling_relative_path=sibling.name,
        encoding_input_manifest_raw=input_raw,
        input_stage_fingerprint=input_fingerprint,
        resolved_encoding_arguments=(
            "ffmpeg",
            "-i",
            "@image-sequence:sha256=" + input_fingerprint + ":view=frames",
            "@output:path64=" + FinalPublicationTransactionV4.path64(expected),
        ),
        container_validator=lambda path: path.read_bytes().startswith(b"validated"),
        event=events.append,
    )

    manifest = transaction.publish()

    assert events == [
        "video:published",
        "input-manifest:published",
        "final-manifest:published",
    ]
    assert manifest.sha256 == hashlib.sha256(b"validated-mp4-fixture").hexdigest()
    assert audit_final_media_v1(
        tmp_path,
        artifact,
        container_validator=lambda _path: True,
    ) == FinalMediaAuditDispositionV1(
        final_video_present=True,
        final_video_authenticated=True,
        final_video_authentication_reason=None,
    )
    assert audit_final_media_v1(tmp_path, artifact) == FinalMediaAuditDispositionV1(
        final_video_present=True,
        final_video_authenticated=False,
        final_video_authentication_reason="publication_validation_failed",
    )


def test_publication_rejects_sibling_swapped_after_container_validation(
    tmp_path,
) -> None:
    artifact = _producer_artifact()
    expected = artifact.to_object()["output_info"]["expected_final_relative_path"]
    sibling = tmp_path / f".{expected}.encoding-v4.tmp"
    sibling.write_bytes(b"validated-original-video")
    replacement = tmp_path / "replacement.mp4"
    replacement.write_bytes(b"different-structurally-valid-video")
    input_raw, input_fingerprint = _input_manifest()

    def swap_after_validation(path: Path) -> bool:
        os.replace(replacement, path)
        return True

    transaction = FinalPublicationTransactionV4(
        job_root=tmp_path,
        settings=artifact,
        sibling_relative_path=sibling.name,
        encoding_input_manifest_raw=input_raw,
        input_stage_fingerprint=input_fingerprint,
        resolved_encoding_arguments=(
            "ffmpeg",
            "-i",
            "@image-sequence:sha256=" + input_fingerprint + ":view=frames",
            "@output:path64=" + FinalPublicationTransactionV4.path64(expected),
        ),
        container_validator=swap_after_validation,
    )

    with pytest.raises(
        FinalPublicationValidationError,
        match="sibling video (identity changed|container validation failed)",
    ):
        transaction.publish()

    assert sibling.read_bytes() in {
        b"validated-original-video",
        b"different-structurally-valid-video",
    }
    assert not (tmp_path / expected).exists()
    assert not (tmp_path / "encoding_input_manifest.json").exists()
    assert not (tmp_path / "final_video_manifest.json").exists()


def test_publication_rejects_sibling_replaced_after_prior_full_validation(
    tmp_path,
) -> None:
    artifact = _producer_artifact()
    expected = artifact.to_object()["output_info"]["expected_final_relative_path"]
    sibling = tmp_path / f".{expected}.encoding-v4.tmp"
    sibling.write_bytes(b"validated-original-video")
    validated_identity = capture_file_identity(sibling)
    sibling.unlink()
    sibling.write_bytes(b"different-structurally-valid-video")
    input_raw, input_fingerprint = _input_manifest()
    transaction = FinalPublicationTransactionV4(
        job_root=tmp_path,
        settings=artifact,
        sibling_relative_path=sibling.name,
        encoding_input_manifest_raw=input_raw,
        input_stage_fingerprint=input_fingerprint,
        resolved_encoding_arguments=(
            "ffmpeg",
            "-i",
            "@image-sequence:sha256=" + input_fingerprint + ":view=frames",
            "@output:path64=" + FinalPublicationTransactionV4.path64(expected),
        ),
        container_validator=lambda _path: True,
        expected_sibling_identity=validated_identity,
    )

    with pytest.raises(
        FinalPublicationValidationError,
        match="identity changed since container validation",
    ):
        transaction.publish()

    assert sibling.read_bytes() == b"different-structurally-valid-video"
    assert not (tmp_path / expected).exists()


def test_publication_revalidates_same_inode_bytes_before_hash_and_replace(
    tmp_path,
) -> None:
    artifact = _producer_artifact()
    expected = artifact.to_object()["output_info"]["expected_final_relative_path"]
    sibling = tmp_path / f".{expected}.encoding-v4.tmp"
    sibling.write_bytes(b"validated-original-video")
    validated_identity = capture_file_identity(sibling)
    sibling.write_bytes(b"different-same-inode-video")
    assert capture_file_identity(sibling) == validated_identity
    input_raw, input_fingerprint = _input_manifest()
    transaction = FinalPublicationTransactionV4(
        job_root=tmp_path,
        settings=artifact,
        sibling_relative_path=sibling.name,
        encoding_input_manifest_raw=input_raw,
        input_stage_fingerprint=input_fingerprint,
        resolved_encoding_arguments=(
            "ffmpeg",
            "-i",
            "@image-sequence:sha256=" + input_fingerprint + ":view=frames",
            "@output:path64=" + FinalPublicationTransactionV4.path64(expected),
        ),
        container_validator=lambda path: path.read_bytes()
        == b"validated-original-video",
        expected_sibling_identity=validated_identity,
    )

    with pytest.raises(
        FinalPublicationValidationError,
        match="failed container validation",
    ):
        transaction.publish()

    assert sibling.read_bytes() == b"different-same-inode-video"
    assert not (tmp_path / expected).exists()


def test_post_video_crash_is_classified_from_actual_mixed_generation(tmp_path) -> None:
    artifact = _producer_artifact()
    expected = artifact.to_object()["output_info"]["expected_final_relative_path"]
    (tmp_path / expected).write_bytes(b"old-video")
    old_input, old_input_fingerprint = _input_manifest()
    (tmp_path / "encoding_input_manifest.json").write_bytes(old_input)
    old_manifest = build_final_video_manifest_v4(
        relative_path=expected,
        video_sha256=hashlib.sha256(b"old-video").hexdigest(),
        video_byte_count=len(b"old-video"),
        input_stage_fingerprint=old_input_fingerprint,
        resolved_encoding_arguments=_assembled_arguments(
            old_input_fingerprint,
            expected,
        ),
    )
    (tmp_path / "final_video_manifest.json").write_bytes(old_manifest.encode())
    sibling = tmp_path / f".{expected}.encoding-v4.tmp"
    sibling.write_bytes(b"validated-new-video")
    input_raw, input_fingerprint = _input_manifest()
    transaction = FinalPublicationTransactionV4(
        job_root=tmp_path,
        settings=artifact,
        sibling_relative_path=sibling.name,
        encoding_input_manifest_raw=input_raw,
        input_stage_fingerprint=input_fingerprint,
        resolved_encoding_arguments=(
            "ffmpeg",
            "-i",
            "@image-sequence:sha256=" + input_fingerprint + ":view=frames",
            "@output:path64=" + FinalPublicationTransactionV4.path64(expected),
        ),
        container_validator=lambda _path: True,
    )

    with pytest.raises(InjectedFinalPublicationCrash):
        transaction.publish(crash_at="after-video")

    assert (tmp_path / expected).read_bytes() == b"validated-new-video"
    disposition = audit_final_media_v1(tmp_path, artifact)
    assert disposition.final_video_present is True
    assert disposition.final_video_authenticated is False
    assert (
        disposition.final_video_authentication_reason == "publication_validation_failed"
    )


def test_publication_streams_staged_input_manifest_without_read_bytes(
    tmp_path, monkeypatch
) -> None:
    artifact = _producer_artifact()
    expected = artifact.to_object()["output_info"]["expected_final_relative_path"]
    sibling = tmp_path / f".{expected}.encoding-v4.tmp"
    sibling.write_bytes(b"validated-streamed-video")
    raw, fingerprint = _input_manifest()
    staged_path = tmp_path / ".encoding-input-manifest.staged"
    staged_path.write_bytes(raw)
    source = EncodingInputPublicationSource(
        path=staged_path,
        fingerprint=fingerprint,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        raw_byte_count=len(raw),
    )
    original_read_bytes = type(staged_path).read_bytes
    protected = {
        staged_path.resolve(),
        (tmp_path / "encoding_input_manifest.json").resolve(),
    }

    def reject_manifest_materialization(path):
        if path.resolve() in protected:
            raise AssertionError("encoding manifest must be streamed")
        return original_read_bytes(path)

    monkeypatch.setattr(
        type(staged_path), "read_bytes", reject_manifest_materialization
    )
    transaction = FinalPublicationTransactionV4(
        job_root=tmp_path,
        settings=artifact,
        sibling_relative_path=sibling.name,
        encoding_input_source=source,
        input_stage_fingerprint=fingerprint,
        resolved_encoding_arguments=(
            "ffmpeg",
            "-i",
            "@image-sequence:sha256=" + fingerprint + ":view=frames",
            "@output:path64=" + FinalPublicationTransactionV4.path64(expected),
        ),
        container_validator=lambda _path: True,
    )

    transaction.publish()

    assert not staged_path.exists()
    assert (
        audit_final_media_v1(
            tmp_path,
            artifact,
            container_validator=lambda _path: True,
        ).final_video_authenticated
        is True
    )


def test_final_encoding_reservation_publishes_index_before_owned_namespace(
    tmp_path,
) -> None:
    artifact = _producer_artifact()
    _install_schema5_job(tmp_path, artifact)
    events: list[str] = []

    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
        event=events.append,
    )

    expected = artifact.to_object()["output_info"]["expected_final_relative_path"]
    assert events[0:2] == ["index:temporary", "index:published"]
    assert events.index("index:published") < events.index("final-target:reserved")
    assert (tmp_path / expected).stat().st_size == 0
    assert reservation.sibling_path.name == f".depth-surge-final-v4-{'44' * 16}.tmp.mp4"
    assert reservation.index_path.is_file()
    assert reservation.target_descriptor_path.is_file()
    assert reservation.input_manifest_transaction.descriptor_path.is_file()
    assert reservation.final_manifest_transaction.descriptor_path.is_file()

    loaded = load_final_encoding_reservation_v1(
        tmp_path,
        artifact,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    recovery = loaded.observe()
    assert recovery.phase == "ready_pre_video"
    assert recovery.action == "encode"


def test_final_placeholder_never_writes_through_a_swapped_job_root(
    tmp_path: Path,
) -> None:
    artifact = _producer_artifact()
    _install_schema5_job(tmp_path, artifact)
    final_name = artifact.to_object()["output_info"]["expected_final_relative_path"]
    original_root = tmp_path.with_name(f"{tmp_path.name}-original")
    swapped_root = tmp_path.with_name(f"{tmp_path.name}-swapped")
    victim_payload = b"foreign final video"
    swapped = False

    def swap_after_index(stage: str) -> None:
        nonlocal swapped
        if stage != "final-target:parent-authenticated" or swapped:
            return
        swapped = True
        tmp_path.rename(original_root)
        tmp_path.mkdir()
        (tmp_path / final_name).write_bytes(victim_payload)

    try:
        with pytest.raises(
            reservation_protocol.ReservationConflictError,
            match="parent identity changed",
        ):
            prepare_final_encoding_reservation_v1(
                job_root=tmp_path,
                settings=artifact,
                publication_generation="44" * 16,
                input_manifest_raw_max=4096,
                final_manifest_raw_max=4096,
                allocation_unit=4096,
                allocation_adapter=_test_allocation,
                allocation_counter=_test_allocation_counter,
                reservation_generation="55" * 16,
                input_reservation_id="66" * 16,
                final_reservation_id="77" * 16,
                event=swap_after_index,
            )
        assert (tmp_path / final_name).read_bytes() == victim_payload
    finally:
        if swapped:
            tmp_path.rename(swapped_root)
            original_root.rename(tmp_path)

    assert (swapped_root / final_name).read_bytes() == victim_payload


def test_final_reservation_threads_cancellation_into_fixed_target_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _producer_artifact()
    _install_schema5_job(tmp_path, artifact)
    (tmp_path / "encoding_input_manifest.json").write_bytes(b"existing manifest")
    def cancelled() -> bool:
        return False
    scan_calls = 0

    def cancel_scan(_root, _path, **kwargs):
        nonlocal scan_calls
        scan_calls += 1
        assert kwargs["cancelled"] is cancelled
        raise InterruptedError("fixed-file scan cancelled")

    monkeypatch.setattr(
        final_publication_module,
        "_scan_fixed_file_v1",
        cancel_scan,
    )

    with pytest.raises(InterruptedError, match="cancelled"):
        prepare_final_encoding_reservation_v1(
            job_root=tmp_path,
            settings=artifact,
            publication_generation="44" * 16,
            input_manifest_raw_max=4096,
            final_manifest_raw_max=4096,
            allocation_unit=4096,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
            reservation_generation="55" * 16,
            input_reservation_id="66" * 16,
            final_reservation_id="77" * 16,
            cancelled=cancelled,
        )

    assert scan_calls == 1
    assert not (tmp_path / FINAL_ENCODING_RESERVATION_RELATIVE_PATH).exists()


def test_manifest_reset_never_writes_through_replaced_source_link(tmp_path) -> None:
    artifact = _artifact()
    _install_schema5_job(tmp_path, artifact)
    raw, _fingerprint = _input_manifest()
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=len(raw),
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    transaction = reservation.input_manifest_transaction
    assert transaction is not None
    sentinel = tmp_path.parent / f"{tmp_path.name}-outside-sentinel.bin"
    sentinel_raw = b"outside-must-not-change"
    sentinel.write_bytes(sentinel_raw)
    transaction.source_path.unlink()
    try:
        try:
            transaction.source_path.symlink_to(sentinel)
        except OSError as error:
            pytest.skip(f"file symlink creation is unavailable: {error}")
        with pytest.raises(reservation_protocol.ReservationConflictError):
            reservation.reset_incomplete_manifest_sources()
        assert sentinel.read_bytes() == sentinel_raw
    finally:
        if transaction.source_path.is_symlink():
            transaction.source_path.unlink()
        sentinel.unlink(missing_ok=True)


def test_final_encoding_disk_admission_is_read_only_and_exact_at_boundary(
    tmp_path,
) -> None:
    artifact = _producer_artifact()
    _install_schema5_job(tmp_path, artifact)
    before = tuple(
        sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    )
    arguments = {
        "job_root": tmp_path,
        "settings": artifact,
        "publication_generation": "44" * 16,
        "input_manifest_raw_max": 4096,
        "final_manifest_raw_max": 4096,
        "allocation_unit": 4096,
        "allocation_adapter": _test_allocation,
        "allocation_counter": _test_allocation_counter,
        "reservation_generation": "55" * 16,
        "input_reservation_id": "66" * 16,
        "final_reservation_id": "77" * 16,
    }

    with pytest.raises(FinalEncodingDiskBudgetError) as captured:
        prepare_final_encoding_reservation_v1(**arguments, free_bytes=lambda _path: 0)

    required = captured.value.required_bytes
    assert required > 0
    assert captured.value.free_bytes == 0
    assert (
        tuple(
            sorted(
                path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")
            )
        )
        == before
    )

    with pytest.raises(FinalEncodingDiskBudgetError):
        prepare_final_encoding_reservation_v1(
            **arguments,
            free_bytes=lambda _path: required - 1,
        )
    assert (
        tuple(
            sorted(
                path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")
            )
        )
        == before
    )

    prepared = prepare_final_encoding_reservation_v1(
        **arguments,
        free_bytes=lambda _path: required,
    )
    assert prepared.index_path.is_file()


def test_target_descriptor_one_byte_over_bound_fails_before_any_final_mutation(
    tmp_path,
    monkeypatch,
) -> None:
    from src.depth_surge_3d.processing.video import final_publication

    artifact = _producer_artifact()
    _install_schema5_job(tmp_path, artifact)
    before = tuple(
        sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    )
    original_bound = final_publication._final_video_target_descriptor_raw_bound_v1

    def one_byte_over(index):
        actual = original_bound(index)
        monkeypatch.setattr(
            final_publication,
            "FINAL_RESERVATION_DESCRIPTOR_RAW_BYTES",
            actual - 1,
        )
        return actual

    monkeypatch.setattr(
        final_publication,
        "_final_video_target_descriptor_raw_bound_v1",
        one_byte_over,
    )

    with pytest.raises(
        reservation_protocol.ReservationConflictError,
        match="target descriptor exceeds",
    ):
        prepare_final_encoding_reservation_v1(
            job_root=tmp_path,
            settings=artifact,
            publication_generation="44" * 16,
            input_manifest_raw_max=4096,
            final_manifest_raw_max=4096,
            allocation_unit=4096,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
            reservation_generation="55" * 16,
            input_reservation_id="66" * 16,
            final_reservation_id="77" * 16,
        )

    assert (
        tuple(
            sorted(
                path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")
            )
        )
        == before
    )


def test_pruning_final_reservation_preallocates_every_post_publication_extent(
    tmp_path,
) -> None:
    artifact = _pruning_artifact()
    _install_schema5_job(tmp_path, artifact)
    _complete_prune_diagnostics(tmp_path)

    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )

    assert [
        entry["reservation_kind"] for entry in reservation.index.extent_entries
    ] == [
        "producer_settings",
        "encoding_input_manifest",
        "final_video_manifest",
        "prune_marker",
        "prune_marker",
        "prune_marker",
        "prune_marker",
        "diagnostics_payload_pruned",
        "cleanup_pending_settings",
        "job_terminal_settings",
    ]
    assert reservation.producer_settings_transaction is not None
    assert len(reservation.prune_marker_transactions) == 4
    assert reservation.diagnostics_payload_pruned_transaction is not None
    assert reservation.cleanup_pending_settings_transaction is not None
    assert (
        reservation.index.diagnostics_metadata_raw_sha256_at_preflight
        == hashlib.sha256(
            (tmp_path / "04_stereo_diagnostics" / "metadata.json").read_bytes()
        ).hexdigest()
    )
    for _entry, transaction, _payload in reservation.prune_marker_transactions:
        assert transaction.target_path.stat().st_size == 0
        assert (
            transaction.source_path.stat().st_size
            == transaction.descriptor.logical_byte_count
        )


@pytest.mark.parametrize(
    "invalid_binding",
    ("crossed_hash", "pair_without_entry", "entry_without_pair"),
)
def test_final_encoding_index_rejects_crossed_diagnostics_binding(
    tmp_path: Path,
    invalid_binding: str,
) -> None:
    artifact = _pruning_artifact()
    _install_schema5_job(tmp_path, artifact)
    _complete_prune_diagnostics(tmp_path)
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    value = copy.deepcopy(reservation.index.to_object())
    entries = value["extent_entries"]
    diagnostics_entry = next(
        entry
        for entry in entries
        if entry["reservation_kind"] == "diagnostics_payload_pruned"
    )
    if invalid_binding == "crossed_hash":
        diagnostics_entry["existing_target_raw_sha256_at_preflight"] = "ab" * 32
    elif invalid_binding == "pair_without_entry":
        entries.remove(diagnostics_entry)
    else:
        value["diagnostics_metadata_raw_sha256_at_preflight"] = None
        value["diagnostics_metadata_fingerprint_at_preflight"] = None
    unsigned = {key: item for key, item in value.items() if key != "fingerprint"}
    value["fingerprint"] = reservation_protocol.sha256_canonical(unsigned)

    with pytest.raises(
        reservation_protocol.ReservationConflictError,
        match="diagnostics",
    ):
        type(reservation.index).from_object(value)


def test_pruning_preflight_streams_diagnostics_metadata_and_derives_extent_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _pruning_artifact()
    _install_schema5_job(tmp_path, artifact)
    _complete_prune_diagnostics(tmp_path)
    metadata_path = tmp_path / "04_stereo_diagnostics" / "metadata.json"
    original_read_bytes = Path.read_bytes

    def reject_metadata_read_bytes(path: Path) -> bytes:
        if path == metadata_path:
            raise AssertionError("final preflight must stream diagnostics metadata")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_metadata_read_bytes)

    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )

    entry = next(
        item
        for item in reservation.index.extent_entries
        if item["reservation_kind"] == "diagnostics_payload_pruned"
    )
    assert entry["payload_raw_max"] != 16 * 1024 * 1024
    assert entry["payload_raw_max"] > 0


def test_pruning_preflight_binds_diagnostics_count_to_schema5_settings(
    tmp_path: Path,
) -> None:
    artifact = _pruning_artifact()
    values = artifact.to_object()["video_properties"]
    mismatched_video = dict(values)
    mismatched_video["frame_count"] = 2
    artifact = _artifact(
        video_properties=mismatched_video,
        processing_settings=validate_settings(
            {"keep_intermediates": False},
            source="explicit",
        ),
    )
    _install_schema5_job(tmp_path, artifact)
    _complete_prune_diagnostics(tmp_path)
    before = tuple(
        sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    )

    with pytest.raises(FinalPublicationValidationError, match="frame count differs"):
        prepare_final_encoding_reservation_v1(
            job_root=tmp_path,
            settings=artifact,
            publication_generation="44" * 16,
            input_manifest_raw_max=4096,
            final_manifest_raw_max=4096,
            allocation_unit=4096,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
            reservation_generation="55" * 16,
            input_reservation_id="66" * 16,
            final_reservation_id="77" * 16,
        )

    assert (
        tuple(
            sorted(
                path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")
            )
        )
        == before
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stats_schema_version", 2),
        ("prune_entries", []),
        ("native_geometry_fingerprint", "Quality-only"),
        ("geometry_mode", "metric"),
        ("render_shape", ["2", 3]),
        ("mask_payloads_enabled", "false"),
    ],
)
def test_pruning_preflight_rejects_re_fingerprinted_invalid_state_before_index(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    artifact = _pruning_artifact()
    _install_schema5_job(tmp_path, artifact)
    _complete_prune_diagnostics(tmp_path)
    metadata_path = tmp_path / "04_stereo_diagnostics" / "metadata.json"
    metadata = json.loads(metadata_path.read_bytes())
    metadata[field] = value
    metadata["fingerprint"] = self_fingerprint(metadata)
    metadata_path.write_bytes(canonical_json_bytes(metadata))

    with pytest.raises(FinalPublicationValidationError, match="not canonical"):
        prepare_final_encoding_reservation_v1(
            job_root=tmp_path,
            settings=artifact,
            publication_generation="44" * 16,
            input_manifest_raw_max=4096,
            final_manifest_raw_max=4096,
            allocation_unit=4096,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
            reservation_generation="55" * 16,
            input_reservation_id="66" * 16,
            final_reservation_id="77" * 16,
        )

    assert not (tmp_path / FINAL_ENCODING_RESERVATION_RELATIVE_PATH).exists()


@pytest.mark.parametrize(
    "field",
    ["source_guide_fingerprint", "native_geometry_fingerprint"],
)
def test_pruning_preflight_rejects_quality_provenance_mismatch_before_index(
    tmp_path: Path,
    field: str,
) -> None:
    artifact = _pruning_artifact()
    _install_schema5_job(tmp_path, artifact)
    _complete_prune_diagnostics(tmp_path)
    metadata = _upgrade_prune_diagnostics_to_quality(tmp_path)
    metadata[field] = "f" * 64
    metadata["fingerprint"] = self_fingerprint(metadata)
    (tmp_path / "04_stereo_diagnostics" / "metadata.json").write_bytes(
        canonical_json_bytes(metadata)
    )

    with pytest.raises(FinalPublicationValidationError, match="not canonical"):
        prepare_final_encoding_reservation_v1(
            job_root=tmp_path,
            settings=artifact,
            publication_generation="44" * 16,
            input_manifest_raw_max=4096,
            final_manifest_raw_max=4096,
            allocation_unit=4096,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
            reservation_generation="55" * 16,
            input_reservation_id="66" * 16,
            final_reservation_id="77" * 16,
        )

    assert not (tmp_path / FINAL_ENCODING_RESERVATION_RELATIVE_PATH).exists()


def test_pruning_final_reservation_reopens_every_indexed_extent(tmp_path) -> None:
    artifact = _pruning_artifact()
    _install_schema5_job(tmp_path, artifact)
    _complete_prune_diagnostics(tmp_path)
    prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )

    reopened = load_final_encoding_reservation_v1(
        tmp_path,
        artifact,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )

    assert reopened.producer_settings_transaction is not None
    assert reopened.producer_settings_payload is None
    assert len(reopened.prune_marker_transactions) == 4
    assert all(
        transaction is not None
        for _, transaction, _ in reopened.prune_marker_transactions
    )
    assert reopened.diagnostics_payload_pruned_transaction is not None
    assert reopened.cleanup_pending_settings_transaction is not None
    assert reopened.cleanup_pending_settings_payload is None

    producer = reopened.commit_producer_settings()
    assert (
        producer.to_object()["metadata"]["final_media_publication_generation"]
        == "44" * 16
    )
    pending = reopened.commit_cleanup_pending_settings()
    assert pending.to_object()["metadata"]["cleanup_status"] == "pending"


def test_partial_producer_settings_source_is_reset_before_any_dependent_mutation(
    tmp_path,
) -> None:
    artifact = _pruning_artifact()
    _install_schema5_job(tmp_path, artifact)
    _complete_prune_diagnostics(tmp_path)
    prepared = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    assert prepared.producer_settings_transaction is not None
    with prepared.producer_settings_transaction.source_path.open(
        "r+b", buffering=0
    ) as handle:
        handle.write(b'{"content_fingerprint":"partial')
        handle.flush()
        os.fsync(handle.fileno())

    reopened = load_final_encoding_reservation_v1(
        tmp_path,
        artifact,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    committed = reopened.commit_producer_settings()

    assert (
        committed.to_object()["metadata"]["final_media_publication_generation"]
        == "44" * 16
    )


def test_pruning_failure_retirement_leaves_no_unindexed_reservation_objects(
    tmp_path,
) -> None:
    artifact = _pruning_artifact()
    _install_schema5_job(tmp_path, artifact)
    _complete_prune_diagnostics(tmp_path)
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    owned_paths = []
    for entry in reservation.index.extent_entries:
        if entry["reservation_kind"] == "job_terminal_settings":
            continue
        descriptor = tmp_path / entry["descriptor_relative_path"]
        source = tmp_path / entry["source_extent_relative_path"]
        owned_paths.extend((descriptor, source))

    failed = artifact.failed(
        terminal_at="2026-08-24T12:35:00.000000Z",
        diagnostic_code="FINAL_ENCODING_FAILED",
        diagnostic_message="fixture",
    )
    reservation_protocol.commit_terminal_settings_v1(
        job_root=tmp_path,
        current=artifact,
        target=failed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    reservation.retire_terminal_failure()

    assert not reservation.index_path.exists()
    assert not reservation.target_descriptor_path.exists()
    assert all(not path.exists() for path in owned_paths)


def test_pruning_success_consumes_all_extents_before_index_last_retirement(
    tmp_path,
) -> None:
    artifact = _pruning_artifact()
    _install_schema5_job(tmp_path, artifact)
    _complete_prune_diagnostics(tmp_path)
    raw, fingerprint = _input_manifest(frame_count=1)
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=len(raw),
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    producer = reservation.commit_producer_settings()
    assert reservation.input_manifest_transaction is not None
    reservation.input_manifest_transaction.stage_complete_full(raw)
    reservation.sibling_path.write_bytes(b"validated-pruning-video")
    FinalPublicationTransactionV4(
        job_root=tmp_path,
        settings=producer,
        sibling_relative_path=reservation.sibling_path.name,
        encoding_input_manifest_raw=raw,
        input_stage_fingerprint=fingerprint,
        resolved_encoding_arguments=_assembled_arguments(
            fingerprint,
            producer.to_object()["output_info"]["expected_final_relative_path"],
        ),
        container_validator=lambda _path: True,
        reservation=reservation,
        retire_reservation=False,
    ).publish()

    pruned = commit_payload_pruned(tmp_path, reservation=reservation)
    with pytest.raises(CleanupTransactionError, match="cleanup_status=pending"):
        run_authorized_cleanup(
            tmp_path,
            metadata=pruned,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
        )
    assert (tmp_path / "00_original_frames" / "payload.bin").is_file()
    pending = reservation.commit_cleanup_pending_settings()
    removed = run_authorized_cleanup(
        tmp_path,
        metadata=pruned,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    completed = pending.completed(
        terminal_at="2026-08-24T12:35:00.000000Z",
        frames_processed=1,
        cleanup_status="complete",
    )
    reservation_protocol.commit_terminal_settings_v1(
        job_root=tmp_path,
        current=pending,
        target=completed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    reservation.retire()

    assert pending.to_object()["metadata"]["cleanup_status"] == "pending"
    assert removed > 0
    assert not reservation.index_path.exists()
    assert not reservation.target_descriptor_path.exists()
    reservation_dir = tmp_path / ".depth-surge-reservations-v1"
    assert not list(reservation_dir.glob("d-r-55*"))


def test_pruning_crash_after_markers_reopens_the_remaining_ordered_prefix(
    tmp_path,
) -> None:
    artifact = _pruning_artifact()
    _install_schema5_job(tmp_path, artifact)
    _complete_prune_diagnostics(tmp_path)
    raw, fingerprint = _input_manifest(frame_count=1)
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=len(raw),
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    producer = reservation.commit_producer_settings()
    assert reservation.input_manifest_transaction is not None
    reservation.input_manifest_transaction.stage_complete_full(raw)
    reservation.sibling_path.write_bytes(b"validated-pruning-recovery-video")
    FinalPublicationTransactionV4(
        job_root=tmp_path,
        settings=producer,
        sibling_relative_path=reservation.sibling_path.name,
        encoding_input_manifest_raw=raw,
        input_stage_fingerprint=fingerprint,
        resolved_encoding_arguments=_assembled_arguments(
            fingerprint,
            producer.to_object()["output_info"]["expected_final_relative_path"],
        ),
        container_validator=lambda _path: True,
        reservation=reservation,
        retire_reservation=False,
    ).publish()

    def crash_after_markers(stage: str) -> None:
        if stage == "after_prune_markers":
            raise RuntimeError(stage)

    with pytest.raises(RuntimeError, match="after_prune_markers"):
        commit_payload_pruned(
            tmp_path,
            reservation=reservation,
            fault=crash_after_markers,
        )

    reopened = load_final_encoding_reservation_v1(
        tmp_path,
        producer,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    assert all(
        transaction is None for _, transaction, _ in reopened.prune_marker_transactions
    )
    assert reopened.diagnostics_payload_pruned_transaction is not None
    assert reopened.cleanup_pending_settings_transaction is not None

    pruned = commit_payload_pruned(tmp_path, reservation=reopened)
    pending = reopened.commit_cleanup_pending_settings()
    run_authorized_cleanup(
        tmp_path,
        metadata=pruned,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    completed = pending.completed(
        terminal_at="2026-08-24T12:35:00.000000Z",
        frames_processed=1,
        cleanup_status="complete",
    )
    reservation_protocol.commit_terminal_settings_v1(
        job_root=tmp_path,
        current=pending,
        target=completed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    reopened.retire()

    assert not reopened.index_path.exists()


@pytest.mark.parametrize(
    ("physical_state", "commits"),
    [
        ("zero", False),
        ("partial", False),
        ("complete-full", True),
        ("complete-short", True),
        ("committed-target", True),
    ],
)
def test_payload_pruned_recovery_classifies_streamed_physical_states(
    tmp_path: Path,
    physical_state: str,
    commits: bool,
) -> None:
    producer, _reservation, transaction, document = _published_pruning_reservation(
        tmp_path
    )
    if physical_state in {
        "partial",
        "complete-full",
        "complete-short",
        "committed-target",
    }:
        if physical_state == "partial":
            prefix = b"".join(document.chunks())[:31]
            with transaction.source_path.open("r+b", buffering=0) as handle:
                handle.write(prefix)
                handle.flush()
                os.fsync(handle.fileno())
        else:
            transaction.stage_complete_full_streamed(
                chunks=document.chunks,
                prefix_byte_count=document.raw_byte_count,
                prefix_sha256=document.raw_sha256,
            )
            if physical_state == "complete-short":
                with transaction.source_path.open("r+b", buffering=0) as handle:
                    handle.truncate(document.raw_byte_count)
                    handle.flush()
                    os.fsync(handle.fileno())
            elif physical_state == "committed-target":
                with pytest.raises(reservation_protocol.InjectedReservationCrash):
                    transaction.consume_streamed_prefix(
                        prefix_byte_count=document.raw_byte_count,
                        prefix_sha256=document.raw_sha256,
                        crash_at="after-rename-parent-sync",
                    )

    reopened = load_final_encoding_reservation_v1(
        tmp_path,
        producer,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    recovery = reopened.recover_publication_prefix()

    assert recovery.phase == "publication_authenticated"
    assert (reopened.diagnostics_payload_pruned_transaction is None) is commits
    projection = project_diagnostics_metadata(
        tmp_path / "04_stereo_diagnostics" / "metadata.json"
    )
    assert projection.status == ("payload_pruned" if commits else "complete")


def test_payload_pruned_recovery_rejects_a_foreign_streamed_prefix(
    tmp_path: Path,
) -> None:
    producer, _reservation, transaction, _document = _published_pruning_reservation(
        tmp_path
    )
    with transaction.source_path.open("r+b", buffering=0) as handle:
        handle.write(b"X")
        handle.flush()
        os.fsync(handle.fileno())

    reopened = load_final_encoding_reservation_v1(
        tmp_path,
        producer,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    with pytest.raises(reservation_protocol.ReservationConflictError, match="foreign"):
        reopened.recover_publication_prefix()


def test_payload_pruned_recovery_never_bulk_reads_or_decodes_large_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer, _reservation, transaction, document = _published_pruning_reservation(
        tmp_path,
        diagnostic_frame_count=90_000,
    )
    assert document.raw_byte_count > 1024 * 1024
    transaction.stage_complete_full_streamed(
        chunks=document.chunks,
        prefix_byte_count=document.raw_byte_count,
        prefix_sha256=document.raw_sha256,
    )
    original_read_bytes = Path.read_bytes
    original_json_loads = json.loads

    def bounded_read_bytes(path: Path) -> bytes:
        if path.stat().st_size > 1024 * 1024:
            raise AssertionError("large recovery artifact was read into memory")
        return original_read_bytes(path)

    def bounded_json_loads(value, *args, **kwargs):
        if len(value) > 1024 * 1024:
            raise AssertionError("large recovery JSON was decoded as one value")
        return original_json_loads(value, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", bounded_read_bytes)
    monkeypatch.setattr(json, "loads", bounded_json_loads)

    reopened = load_final_encoding_reservation_v1(
        tmp_path,
        producer,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    recovery = reopened.recover_publication_prefix()

    assert recovery.phase == "publication_authenticated"
    assert reopened.diagnostics_payload_pruned_transaction is None
    assert (
        project_diagnostics_metadata(
            tmp_path / "04_stereo_diagnostics" / "metadata.json"
        ).frame_count
        == 90_000
    )


def test_schema5_cleanup_rejects_missing_final_index_without_mutation(tmp_path) -> None:
    artifact = _pruning_artifact()
    _install_schema5_job(tmp_path, artifact)
    _complete_prune_diagnostics(tmp_path)
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    reservation.commit_producer_settings()
    pending = reservation.commit_cleanup_pending_settings()
    payload = tmp_path / "00_original_frames" / "payload.bin"
    payload_before = payload.read_bytes()
    settings_path = tmp_path / "processing-settings-v5.json"
    settings_before = settings_path.read_bytes()
    reservation.index_path.unlink()

    with pytest.raises(CleanupTransactionError, match="active final reservation"):
        commit_payload_pruned(tmp_path)
    with pytest.raises(CleanupTransactionError, match="active final reservation"):
        run_authorized_cleanup(
            tmp_path,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
        )

    assert pending.to_object()["metadata"]["cleanup_status"] == "pending"
    assert payload.read_bytes() == payload_before
    assert settings_path.read_bytes() == settings_before


@pytest.mark.parametrize("temporary_only", [False, True])
def test_global_recovery_rejects_an_unindexed_final_target_descriptor_pair(
    tmp_path,
    temporary_only,
) -> None:
    artifact = _producer_artifact()
    _install_schema5_job(tmp_path, artifact)
    reservation_dir = tmp_path / ".depth-surge-reservations-v1"
    descriptor = (
        reservation_dir / f"final-video-target-p-{'44' * 16}-r-{'55' * 16}.json"
    )
    if temporary_only:
        descriptor = descriptor.with_name(f".{descriptor.name}.create-new.tmp")
    descriptor.write_bytes(b'{"unindexed":true}')
    settings_before = (tmp_path / "processing-settings-v5.json").read_bytes()

    with pytest.raises(
        reservation_protocol.ReservationConflictError,
        match="unindexed reservation artifact",
    ):
        recover_processing_control_prefix_v1(
            tmp_path,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
        )

    assert descriptor.read_bytes() == b'{"unindexed":true}'
    assert (tmp_path / "processing-settings-v5.json").read_bytes() == settings_before


@pytest.mark.parametrize(
    ("relative_parent", "reserved_name"),
    [
        (
            "00_original_frames",
            f".depth-surge-r-{'55' * 16}-encoding_input_manifest-{'66' * 16}.reserved",
        ),
        (
            "04_stereo_diagnostics",
            f".depth-surge-r-{'55' * 16}-diagnostics_payload_pruned-{'66' * 16}.reserved",
        ),
        ("00_original_frames", ".depth-surge-prune-v2-foreign.json"),
    ],
)
def test_global_recovery_rejects_an_unindexed_nested_reserved_payload(
    tmp_path,
    relative_parent,
    reserved_name,
) -> None:
    artifact = _producer_artifact()
    _install_schema5_job(tmp_path, artifact)
    parent = tmp_path / relative_parent
    parent.mkdir(parents=True, exist_ok=True)
    reserved = parent / reserved_name
    reserved.write_bytes(b"foreign-reservation")
    settings_before = (tmp_path / "processing-settings-v5.json").read_bytes()

    with pytest.raises(
        reservation_protocol.ReservationConflictError,
        match="unindexed reserved payload",
    ):
        recover_processing_control_prefix_v1(
            tmp_path,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
        )

    assert reserved.read_bytes() == b"foreign-reservation"
    assert (tmp_path / "processing-settings-v5.json").read_bytes() == settings_before


@pytest.mark.parametrize(
    "stop_event",
    [
        "index:published",
        "final-target:reserved",
        "input-manifest:descriptor:published",
    ],
)
def test_final_reservation_construction_recovers_every_indexed_prefix(
    tmp_path, stop_event
) -> None:
    artifact = _producer_artifact()
    _install_schema5_job(tmp_path, artifact)

    def stop(value: str) -> None:
        if value == stop_event:
            raise RuntimeError(stop_event)

    with pytest.raises(RuntimeError, match=stop_event):
        prepare_final_encoding_reservation_v1(
            job_root=tmp_path,
            settings=artifact,
            publication_generation="44" * 16,
            input_manifest_raw_max=4096,
            final_manifest_raw_max=4096,
            allocation_unit=4096,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
            reservation_generation="55" * 16,
            input_reservation_id="66" * 16,
            final_reservation_id="77" * 16,
            event=stop,
        )

    recovered = load_final_encoding_reservation_v1(
        tmp_path,
        artifact,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )

    assert recovered.target_descriptor_path.is_file()
    assert recovered.input_manifest_transaction is not None
    assert recovered.final_manifest_transaction is not None
    assert recovered.observe().phase == "ready_pre_video"


@pytest.mark.parametrize("existing_final", [False, True])
def test_final_target_descriptor_rewrites_an_index_owned_partial_temporary(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    existing_final: bool,
) -> None:
    artifact = _producer_artifact()
    _install_schema5_job(tmp_path, artifact)
    if existing_final:
        expected = artifact.to_object()["output_info"]["expected_final_relative_path"]
        (tmp_path / expected).write_bytes(b"authenticated-old-final")

    def stop(value: str) -> None:
        if value == "index:published":
            raise RuntimeError(value)

    with pytest.raises(RuntimeError, match="index:published"):
        prepare_final_encoding_reservation_v1(
            job_root=tmp_path,
            settings=artifact,
            publication_generation="44" * 16,
            input_manifest_raw_max=4096,
            final_manifest_raw_max=4096,
            allocation_unit=4096,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
            reservation_generation="55" * 16,
            input_reservation_id="66" * 16,
            final_reservation_id="77" * 16,
            event=stop,
        )

    index = json.loads(
        (tmp_path / FINAL_ENCODING_RESERVATION_RELATIVE_PATH).read_text(
            encoding="ascii"
        )
    )
    descriptor = tmp_path / index["final_video_target_descriptor_relative_path"]
    temporary = descriptor.with_name(f".{descriptor.name}.create-new.tmp")
    original = reservation_protocol._write_control_temporary

    def interrupt_descriptor(
        path: Path,
        raw: bytes,
        *,
        event,
        expected_parent_identity=None,
    ):
        if path == temporary:
            with path.open("xb", buffering=0) as handle:
                prefix = raw[: max(1, min(37, len(raw) - 1))]
                assert handle.write(prefix) == len(prefix)
                handle.flush()
                os.fsync(handle.fileno())
            raise OSError("injected target descriptor short write")
        return original(
            path,
            raw,
            event=event,
            expected_parent_identity=expected_parent_identity,
        )

    monkeypatch.setattr(
        reservation_protocol,
        "_write_control_temporary",
        interrupt_descriptor,
    )
    with pytest.raises(OSError, match="short write"):
        load_final_encoding_reservation_v1(
            tmp_path,
            artifact,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
        )
    prefix = temporary.read_bytes()
    monkeypatch.setattr(reservation_protocol, "_write_control_temporary", original)

    recovered = load_final_encoding_reservation_v1(
        tmp_path,
        artifact,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )

    assert recovered.target_descriptor_path.is_file()
    assert recovered.target_descriptor_path.read_bytes().startswith(prefix)
    assert not temporary.exists()


def test_reserved_publication_retires_descriptors_then_index_last(tmp_path) -> None:
    initial = _artifact(
        processing_settings=validate_settings(
            {
                "vr_resolution": "custom:1920x1080",
                "per_eye_width": 960,
                "per_eye_height": 1080,
                "vr_output_width": 1920,
                "vr_output_height": 1080,
            },
            source="explicit",
        )
    )
    artifact = initial.with_producer_marker(
        generation="44" * 16,
        updated_at="2026-08-24T12:34:57.123456Z",
    )
    _install_schema5_job(tmp_path, artifact, initial=initial)
    raw, fingerprint = _input_manifest()
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=len(raw),
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    reservation.input_manifest_transaction.stage_complete_full(raw)
    input_descriptor = reservation.input_manifest_transaction.descriptor_path
    final_descriptor = reservation.final_manifest_transaction.descriptor_path
    reservation.sibling_path.write_bytes(b"validated-reserved-video")
    events: list[str] = []
    transaction = FinalPublicationTransactionV4(
        job_root=tmp_path,
        settings=artifact,
        sibling_relative_path=reservation.sibling_path.name,
        encoding_input_manifest_raw=raw,
        input_stage_fingerprint=fingerprint,
        resolved_encoding_arguments=(
            "ffmpeg",
            "-framerate",
            "24",
            "-start_number",
            "0",
            "-i",
            "@image-sequence:sha256=" + fingerprint + ":view=frames",
            "-frames:v",
            "12",
            "-c:v",
            "libx264",
            "@output:path64="
            + FinalPublicationTransactionV4.path64(
                artifact.to_object()["output_info"]["expected_final_relative_path"]
            ),
        ),
        container_validator=lambda _path: True,
        reservation=reservation,
        event=events.append,
    )

    transaction.publish()

    assert not input_descriptor.exists()
    assert not final_descriptor.exists()
    assert reservation.target_descriptor_path.is_file()
    assert reservation.index_path.is_file()
    reservation.validate_authenticated_publication(
        container_validator=lambda _path: True,
    )
    with pytest.raises(FinalPublicationValidationError, match="fixed index is active"):
        audit_final_media_v1(
            tmp_path,
            artifact,
            container_validator=lambda _path: True,
        )
    with pytest.raises(
        reservation_protocol.ReservationConflictError,
        match="exact completed settings",
    ):
        reservation.retire()
    assert reservation.target_descriptor_path.is_file()
    assert reservation.index_path.is_file()

    completed = artifact.completed(
        terminal_at="2026-08-24T12:35:00.000000Z",
        frames_processed=12,
        cleanup_status="not_requested",
    )
    reservation_protocol.commit_terminal_settings_v1(
        job_root=tmp_path,
        current=artifact,
        target=completed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    reservation.retire(event=events.append)

    assert not reservation.target_descriptor_path.exists()
    assert not reservation.index_path.exists()
    assert events[-2:] == [
        "reservation:target-descriptor-retired",
        "reservation:index-retired",
    ]
    assert (
        audit_final_media_v1(
            tmp_path,
            completed,
            container_validator=lambda _path: True,
        ).final_video_authenticated
        is True
    )


@pytest.mark.parametrize("producer_committed", [False, True])
def test_prelaunch_discard_retires_the_whole_invocation_but_keeps_job_authority(
    tmp_path: Path,
    producer_committed: bool,
) -> None:
    artifact = _artifact()
    _install_schema5_job(tmp_path, artifact)
    raw, _fingerprint = _input_manifest()
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=len(raw),
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    current = reservation.commit_producer_settings() if producer_committed else artifact
    assert reservation.input_manifest_transaction is not None
    reservation.input_manifest_transaction.stage_complete_full(raw)
    control = reservation_protocol.load_job_control(tmp_path)
    terminal_entry = next(
        entry
        for entry in control.bootstrap_extent_entries
        if entry["reservation_kind"] == "job_terminal_settings"
    )
    terminal_source = tmp_path / terminal_entry["source_extent_relative_path"]
    terminal_descriptor = tmp_path / terminal_entry["descriptor_relative_path"]

    reservation.discard_prelaunch_invocation()

    assert not reservation.index_path.exists()
    assert not reservation.target_descriptor_path.exists()
    assert not reservation.sibling_path.exists()
    assert terminal_source.is_file()
    assert terminal_descriptor.is_file()
    assert (
        recover_processing_control_prefix_v1(
            tmp_path,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
        )
        == current
    )
    replacement = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=current,
        publication_generation="44" * 16,
        input_manifest_raw_max=len(raw) + 4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="88" * 16,
        input_reservation_id="99" * 16,
        final_reservation_id="ab" * 16,
    )
    assert replacement.index_path.is_file()


def test_reserved_post_video_crash_keeps_index_as_recovery_authority(tmp_path) -> None:
    artifact = _producer_artifact()
    _install_schema5_job(tmp_path, artifact)
    raw, fingerprint = _input_manifest()
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=len(raw),
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    reservation.input_manifest_transaction.stage_complete_full(raw)
    reservation.sibling_path.write_bytes(b"validated-reserved-video")
    transaction = FinalPublicationTransactionV4(
        job_root=tmp_path,
        settings=artifact,
        sibling_relative_path=reservation.sibling_path.name,
        encoding_input_manifest_raw=raw,
        input_stage_fingerprint=fingerprint,
        resolved_encoding_arguments=_assembled_arguments(
            fingerprint,
            artifact.to_object()["output_info"]["expected_final_relative_path"],
        ),
        container_validator=lambda _path: True,
        reservation=reservation,
    )

    with pytest.raises(InjectedFinalPublicationCrash):
        transaction.publish(crash_at="after-video")

    assert reservation.index_path.is_file()
    assert reservation.target_descriptor_path.is_file()
    assert reservation.input_manifest_transaction.descriptor_path.is_file()
    assert reservation.final_manifest_transaction.descriptor_path.is_file()

    loaded = load_final_encoding_reservation_v1(
        tmp_path,
        artifact,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    recovery = loaded.observe()
    assert recovery.phase == "video_replaced_input_intent"
    assert recovery.action == "commit-input-manifest"


def test_reserved_input_manifest_refuses_a_replaced_preflight_target(tmp_path) -> None:
    artifact = _producer_artifact()
    _install_schema5_job(tmp_path, artifact)
    old_raw, _old_fingerprint = _input_manifest()
    input_target = tmp_path / "encoding_input_manifest.json"
    input_target.write_bytes(old_raw)
    new_raw, new_fingerprint = _input_manifest()
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=len(new_raw),
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    assert reservation.input_manifest_transaction is not None
    reservation.input_manifest_transaction.stage_complete_full(new_raw)
    replacement = tmp_path / "replacement-input.json"
    replacement.write_bytes(old_raw)
    os.replace(replacement, input_target)
    reservation.sibling_path.write_bytes(b"validated-predecessor-video")

    with pytest.raises(Exception, match="preflight target"):
        FinalPublicationTransactionV4(
            job_root=tmp_path,
            settings=artifact,
            sibling_relative_path=reservation.sibling_path.name,
            encoding_input_manifest_raw=new_raw,
            input_stage_fingerprint=new_fingerprint,
            resolved_encoding_arguments=_assembled_arguments(
                new_fingerprint,
                artifact.to_object()["output_info"]["expected_final_relative_path"],
            ),
            container_validator=lambda _path: True,
            reservation=reservation,
            retire_reservation=False,
        ).publish()

    assert input_target.read_bytes() == old_raw


@pytest.mark.skipif(
    os.name != "nt", reason="Windows replacement gaps are platform-specific"
)
def test_reserved_windows_video_replacement_gap_moves_indexed_sibling_forward(
    tmp_path,
) -> None:
    artifact = _producer_artifact()
    _install_schema5_job(tmp_path, artifact)
    raw, _fingerprint = _input_manifest()
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=len(raw),
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    assert reservation.input_manifest_transaction is not None
    reservation.input_manifest_transaction.stage_complete_full(raw)
    reservation.sibling_path.write_bytes(b"validated-replacement-gap-video")
    reservation.final_video_path.unlink()

    loaded = load_final_encoding_reservation_v1(
        tmp_path,
        artifact,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )

    assert loaded.observe().phase == "windows_video_replace_gap_before_input_commit"
    loaded.recover_video_replacement_gap()
    assert loaded.final_video_path.read_bytes() == b"validated-replacement-gap-video"
    assert not loaded.sibling_path.exists()
    assert loaded.observe().phase == "video_replaced_input_intent"


@pytest.mark.parametrize(
    ("crash_at", "phase", "action"),
    [
        (
            "after-input-manifest",
            "input_manifest_committed_no_final_intent",
            "revalidate-input-and-reencode",
        ),
        (
            "after-final-manifest",
            "publication_authenticated",
            "continue-finalization",
        ),
    ],
)
def test_reserved_manifest_crash_prefix_reopens_to_one_phase(
    tmp_path, crash_at, phase, action
) -> None:
    artifact = _producer_artifact()
    _install_schema5_job(tmp_path, artifact)
    raw, fingerprint = _input_manifest()
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=len(raw),
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    reservation.input_manifest_transaction.stage_complete_full(raw)
    reservation.sibling_path.write_bytes(b"validated-reserved-video")
    transaction = FinalPublicationTransactionV4(
        job_root=tmp_path,
        settings=artifact,
        sibling_relative_path=reservation.sibling_path.name,
        encoding_input_manifest_raw=raw,
        input_stage_fingerprint=fingerprint,
        resolved_encoding_arguments=_assembled_arguments(
            fingerprint,
            artifact.to_object()["output_info"]["expected_final_relative_path"],
        ),
        container_validator=lambda _path: True,
        reservation=reservation,
    )

    with pytest.raises(InjectedFinalPublicationCrash):
        transaction.publish(crash_at=crash_at)

    loaded = load_final_encoding_reservation_v1(
        tmp_path,
        artifact,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    recovery = loaded.observe()
    assert (recovery.phase, recovery.action) == (phase, action)
    if crash_at == "after-final-manifest":
        completed = artifact.completed(
            terminal_at="2026-08-24T12:35:00.000000Z",
            frames_processed=12,
            cleanup_status="not_requested",
        )
        reservation_protocol.commit_terminal_settings_v1(
            job_root=tmp_path,
            current=artifact,
            target=completed,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
        )
        loaded.retire()
        assert not loaded.index_path.exists()


def test_descriptorless_committed_manifests_require_three_way_binding(tmp_path) -> None:
    artifact = _producer_artifact()
    _install_schema5_job(tmp_path, artifact)
    raw, fingerprint = _input_manifest()
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    assert reservation.input_manifest_transaction is not None
    assert reservation.final_manifest_transaction is not None
    input_descriptor_path = reservation.input_manifest_transaction.descriptor_path
    final_descriptor_path = reservation.final_manifest_transaction.descriptor_path
    reservation.input_manifest_transaction.stage_complete_full(raw)
    reservation.sibling_path.write_bytes(b"validated-reserved-video")
    FinalPublicationTransactionV4(
        job_root=tmp_path,
        settings=artifact,
        sibling_relative_path=reservation.sibling_path.name,
        encoding_input_manifest_raw=raw,
        input_stage_fingerprint=fingerprint,
        resolved_encoding_arguments=_assembled_arguments(
            fingerprint,
            artifact.to_object()["output_info"]["expected_final_relative_path"],
        ),
        container_validator=lambda _path: True,
        reservation=reservation,
        retire_reservation=False,
    ).publish()
    assert not input_descriptor_path.exists()
    assert not final_descriptor_path.exists()
    wrong_raw, _wrong_fingerprint = _input_manifest(frame_count=3)
    (tmp_path / "encoding_input_manifest.json").write_bytes(wrong_raw)

    with pytest.raises(
        reservation_protocol.ReservationConflictError,
        match="committed evidence",
    ):
        load_final_encoding_reservation_v1(
            tmp_path,
            artifact,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
        )


@pytest.mark.parametrize(
    "crash_point",
    [
        "after-target-descriptor-unlink-before-sync",
        "after-target-descriptor-retirement",
        "after-index-unlink-before-sync",
    ],
)
def test_terminal_retirement_recovers_each_descriptor_index_sync_boundary(
    tmp_path,
    crash_point,
) -> None:
    artifact = _producer_artifact()
    _install_schema5_job(tmp_path, artifact)
    raw, fingerprint = _input_manifest()
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=len(raw),
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    reservation.input_manifest_transaction.stage_complete_full(raw)
    reservation.sibling_path.write_bytes(b"validated-reserved-video")
    FinalPublicationTransactionV4(
        job_root=tmp_path,
        settings=artifact,
        sibling_relative_path=reservation.sibling_path.name,
        encoding_input_manifest_raw=raw,
        input_stage_fingerprint=fingerprint,
        resolved_encoding_arguments=_assembled_arguments(
            fingerprint,
            artifact.to_object()["output_info"]["expected_final_relative_path"],
        ),
        container_validator=lambda _path: True,
        reservation=reservation,
        retire_reservation=False,
    ).publish()
    completed = artifact.completed(
        terminal_at="2026-08-24T12:35:00.000000Z",
        frames_processed=12,
        cleanup_status="not_requested",
    )
    reservation_protocol.commit_terminal_settings_v1(
        job_root=tmp_path,
        current=artifact,
        target=completed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    index_temporary = reservation.index_path.with_name(
        f".{reservation.index_path.name}.create-new.tmp"
    )
    index_temporary.write_bytes(b"stale-index-temporary")

    with pytest.raises(InjectedFinalPublicationCrash):
        reservation.retire(crash_at=crash_point)

    assert not index_temporary.exists()
    recovered = recover_final_encoding_terminal_retirement_v1(
        tmp_path,
        completed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    assert recovered is (crash_point != "after-index-unlink-before-sync")
    assert not reservation.index_path.exists()


def test_completed_retirement_rejects_revision_outside_indexed_prefix(tmp_path) -> None:
    artifact = _producer_artifact()
    _install_schema5_job(tmp_path, artifact)
    raw, fingerprint = _input_manifest()
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=len(raw),
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    reservation.input_manifest_transaction.stage_complete_full(raw)
    reservation.sibling_path.write_bytes(b"validated-reserved-video")
    FinalPublicationTransactionV4(
        job_root=tmp_path,
        settings=artifact,
        sibling_relative_path=reservation.sibling_path.name,
        encoding_input_manifest_raw=raw,
        input_stage_fingerprint=fingerprint,
        resolved_encoding_arguments=_assembled_arguments(
            fingerprint,
            artifact.to_object()["output_info"]["expected_final_relative_path"],
        ),
        container_validator=lambda _path: True,
        reservation=reservation,
        retire_reservation=False,
    ).publish()
    completed = artifact.completed(
        terminal_at="2026-08-24T12:35:00.000000Z",
        frames_processed=12,
        cleanup_status="not_requested",
    )
    reservation_protocol.commit_terminal_settings_v1(
        job_root=tmp_path,
        current=artifact,
        target=completed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    value = completed.to_object()
    value["metadata"]["settings_revision"] += 1
    value["content_fingerprint"] = hashlib.sha256(
        canonical_json_bytes(
            {key: item for key, item in value.items() if key != "content_fingerprint"}
        )
    ).hexdigest()
    wrong_revision = type(completed).from_object(value)
    (tmp_path / "processing-settings-v5.json").write_bytes(wrong_revision.encode())

    with pytest.raises(
        reservation_protocol.ReservationConflictError,
        match="revision.*indexed prefix",
    ):
        recover_final_encoding_terminal_retirement_v1(
            tmp_path,
            wrong_revision,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
        )

    assert reservation.index_path.is_file()
    assert reservation.target_descriptor_path.is_file()


def test_terminal_failure_retirement_removes_only_unused_placeholders(tmp_path) -> None:
    artifact = _producer_artifact()
    _install_schema5_job(tmp_path, artifact)
    expected = artifact.to_object()["output_info"]["expected_final_relative_path"]
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    reservation.sibling_path.write_bytes(b"partial-video")
    failed = artifact.failed(
        terminal_at="2026-08-24T12:35:00.000000Z",
        diagnostic_code="FINAL_ENCODING_FAILED",
        diagnostic_message="fixture",
    )
    reservation_protocol.commit_terminal_settings_v1(
        job_root=tmp_path,
        current=artifact,
        target=failed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )

    assert recover_final_encoding_terminal_retirement_v1(
        tmp_path,
        failed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )

    assert not (tmp_path / expected).exists()
    assert not (tmp_path / "encoding_input_manifest.json").exists()
    assert not (tmp_path / "final_video_manifest.json").exists()
    assert not reservation.index_path.exists()


@pytest.mark.parametrize(
    "crash_point",
    [
        "failure-retirement:0:descriptor",
        "failure-retirement:descriptors-unlinked-before-sync",
        "failure-retirement:index-unlinked-before-sync",
    ],
)
def test_terminal_failure_retirement_resumes_without_recreating_cleaned_extents(
    tmp_path,
    crash_point,
) -> None:
    artifact = _producer_artifact()
    _install_schema5_job(tmp_path, artifact)
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    failed = artifact.failed(
        terminal_at="2026-08-24T12:35:00.000000Z",
        diagnostic_code="FINAL_ENCODING_FAILED",
        diagnostic_message="fixture",
    )
    reservation_protocol.commit_terminal_settings_v1(
        job_root=tmp_path,
        current=artifact,
        target=failed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    first_entry = reservation.index.extent_entries[0]
    first_descriptor = tmp_path / first_entry["descriptor_relative_path"]
    first_source = tmp_path / first_entry["source_extent_relative_path"]

    with pytest.raises(InjectedFinalPublicationCrash):
        recover_final_encoding_terminal_retirement_v1(
            tmp_path,
            failed,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
            crash_at=crash_point,
        )

    assert not first_descriptor.exists()
    assert not first_source.exists()
    recovered = recover_final_encoding_terminal_retirement_v1(
        tmp_path,
        failed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    assert recovered is (crash_point != "failure-retirement:index-unlinked-before-sync")
    assert not first_descriptor.exists()
    assert not first_source.exists()
    assert not reservation.index_path.exists()


def test_terminal_failure_discards_descriptor_temporaries_and_zero_placeholders(
    tmp_path,
) -> None:
    artifact = _producer_artifact()
    _install_schema5_job(tmp_path, artifact)
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    entry = next(
        item
        for item in reservation.index.extent_entries
        if item["reservation_kind"] == "encoding_input_manifest"
    )
    descriptor = tmp_path / entry["descriptor_relative_path"]
    source = tmp_path / entry["source_extent_relative_path"]
    descriptor_raw = descriptor.read_bytes()
    descriptor.unlink()
    source.unlink()
    (tmp_path / entry["target_relative_path"]).unlink()
    descriptor_temporary = descriptor.with_name(f".{descriptor.name}.create-new.tmp")
    descriptor_temporary.write_bytes(descriptor_raw)
    target_descriptor = reservation.target_descriptor_path
    target_descriptor_raw = target_descriptor.read_bytes()
    target_descriptor.unlink()
    reservation.final_video_path.unlink()
    target_descriptor_temporary = target_descriptor.with_name(
        f".{target_descriptor.name}.create-new.tmp"
    )
    target_descriptor_temporary.write_bytes(target_descriptor_raw)
    failed = artifact.failed(
        terminal_at="2026-08-24T12:35:00.000000Z",
        diagnostic_code="FINAL_ENCODING_FAILED",
        diagnostic_message="fixture",
    )
    reservation_protocol.commit_terminal_settings_v1(
        job_root=tmp_path,
        current=artifact,
        target=failed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )

    assert recover_final_encoding_terminal_retirement_v1(
        tmp_path,
        failed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )

    assert not descriptor_temporary.exists()
    assert not target_descriptor_temporary.exists()
    assert not (tmp_path / entry["target_relative_path"]).exists()
    assert not reservation.final_video_path.exists()
    assert not reservation.index_path.exists()


def test_terminal_failure_preserves_descriptorless_zero_placeholder(tmp_path) -> None:
    artifact = _producer_artifact()
    _install_schema5_job(tmp_path, artifact)
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    entry = next(
        item
        for item in reservation.index.extent_entries
        if item["reservation_kind"] == "encoding_input_manifest"
    )
    descriptor = tmp_path / entry["descriptor_relative_path"]
    (tmp_path / entry["source_extent_relative_path"]).unlink()
    descriptor.unlink()
    placeholder = tmp_path / entry["target_relative_path"]
    failed = artifact.failed(
        terminal_at="2026-08-24T12:35:00.000000Z",
        diagnostic_code="FINAL_ENCODING_FAILED",
        diagnostic_message="fixture",
    )
    reservation_protocol.commit_terminal_settings_v1(
        job_root=tmp_path,
        current=artifact,
        target=failed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )

    with pytest.raises(
        reservation_protocol.ReservationConflictError,
        match="descriptorless extent target",
    ):
        recover_final_encoding_terminal_retirement_v1(
            tmp_path,
            failed,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
        )

    assert placeholder.is_file()
    assert placeholder.stat().st_size == 0
    assert reservation.index_path.is_file()


def test_terminal_failure_never_deletes_a_durable_producer_settings_intent(
    tmp_path,
) -> None:
    artifact = _pruning_artifact()
    _install_schema5_job(tmp_path, artifact)
    _complete_prune_diagnostics(tmp_path)
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    transaction = reservation.producer_settings_transaction
    payload = reservation.producer_settings_payload
    assert transaction is not None and payload is not None
    transaction.stage_complete_full(payload)
    failed = artifact.failed(
        terminal_at="2026-08-24T12:35:00.000000Z",
        diagnostic_code="FINAL_ENCODING_FAILED",
        diagnostic_message="fixture",
    )
    reservation_protocol.commit_terminal_settings_v1(
        job_root=tmp_path,
        current=artifact,
        target=failed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )

    with pytest.raises(reservation_protocol.ReservationConflictError):
        recover_final_encoding_terminal_retirement_v1(
            tmp_path,
            failed,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
        )

    assert transaction.source_path.is_file()
    assert transaction.descriptor_path.is_file()
    assert reservation.index_path.is_file()


def test_terminal_update_forwards_durable_producer_intent_before_failed_commit(
    tmp_path,
) -> None:
    artifact = _pruning_artifact()
    _install_schema5_job(tmp_path, artifact)
    _complete_prune_diagnostics(tmp_path)
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    transaction = reservation.producer_settings_transaction
    payload = reservation.producer_settings_payload
    assert transaction is not None and payload is not None
    transaction.stage_complete_full(payload)
    forwarded = recover_processing_control_prefix_v1(
        tmp_path,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    assert forwarded is not None
    assert (
        forwarded.to_object()["metadata"]["final_media_publication_generation"]
        == "44" * 16
    )
    failed = forwarded.failed(
        terminal_at="2026-08-24T12:35:00.000000Z",
        diagnostic_code="FINAL_ENCODING_FAILED",
        diagnostic_message="fixture",
    )
    reservation_protocol.commit_terminal_settings_v1(
        job_root=tmp_path,
        current=forwarded,
        target=failed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    assert recover_final_encoding_terminal_retirement_v1(
        tmp_path,
        failed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )

    persisted = recover_processing_control_prefix_v1(
        tmp_path,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    assert persisted is not None
    assert persisted.status == "failed"
    assert not reservation.index_path.exists()


def test_terminal_failure_preserves_final_video_replacement_gap(tmp_path) -> None:
    artifact = _artifact()
    _install_schema5_job(tmp_path, artifact)
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    reservation.sibling_path.write_bytes(b"new-video")
    reservation.final_video_path.unlink()
    failed = artifact.failed(
        terminal_at="2026-08-24T12:35:00.000000Z",
        diagnostic_code="FINAL_ENCODING_FAILED",
        diagnostic_message="fixture",
    )
    reservation_protocol.commit_terminal_settings_v1(
        job_root=tmp_path,
        current=artifact,
        target=failed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )

    with pytest.raises(
        reservation_protocol.ReservationConflictError,
        match="replacement gap",
    ):
        recover_final_encoding_terminal_retirement_v1(
            tmp_path,
            failed,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
        )

    assert reservation.sibling_path.read_bytes() == b"new-video"
    assert reservation.index_path.is_file()


def test_failed_retirement_validates_target_descriptor_before_deleting_owned_paths(
    tmp_path,
) -> None:
    artifact = _artifact()
    _install_schema5_job(tmp_path, artifact)
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    reservation.sibling_path.write_bytes(b"partial-video")
    input_source = reservation.input_manifest_transaction.source_path
    input_descriptor = reservation.input_manifest_transaction.descriptor_path
    source_before = input_source.read_bytes()
    descriptor_before = input_descriptor.read_bytes()
    reservation.target_descriptor_path.write_bytes(b'{"damaged":true}')
    failed = artifact.failed(
        terminal_at="2026-08-24T12:35:00.000000Z",
        diagnostic_code="FINAL_ENCODING_FAILED",
        diagnostic_message="fixture",
    )
    reservation_protocol.commit_terminal_settings_v1(
        job_root=tmp_path,
        current=artifact,
        target=failed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )

    with pytest.raises(reservation_protocol.ReservationConflictError):
        recover_final_encoding_terminal_retirement_v1(
            tmp_path,
            failed,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
        )

    assert reservation.sibling_path.read_bytes() == b"partial-video"
    assert input_source.read_bytes() == source_before
    assert input_descriptor.read_bytes() == descriptor_before
    assert reservation.index_path.is_file()


@pytest.mark.parametrize("descriptor_kind", ["extent", "final-target"])
def test_failed_retirement_rejects_a_foreign_both_present_descriptor_temporary(
    tmp_path,
    descriptor_kind,
) -> None:
    artifact = _artifact()
    _install_schema5_job(tmp_path, artifact)
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    transaction = reservation.input_manifest_transaction
    assert transaction is not None
    descriptor = (
        transaction.descriptor_path
        if descriptor_kind == "extent"
        else reservation.target_descriptor_path
    )
    temporary = descriptor.with_name(f".{descriptor.name}.create-new.tmp")
    foreign = canonical_json_bytes({"algorithm_version": "foreign-descriptor-v1"})
    temporary.write_bytes(foreign)
    reservation.sibling_path.write_bytes(b"partial-video")
    failed = artifact.failed(
        terminal_at="2026-08-24T12:35:00.000000Z",
        diagnostic_code="FINAL_ENCODING_FAILED",
        diagnostic_message="fixture",
    )
    reservation_protocol.commit_terminal_settings_v1(
        job_root=tmp_path,
        current=artifact,
        target=failed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    source_before = transaction.source_path.read_bytes()

    with pytest.raises(reservation_protocol.ReservationConflictError):
        recover_final_encoding_terminal_retirement_v1(
            tmp_path,
            failed,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
        )

    assert temporary.read_bytes() == foreign
    assert descriptor.is_file()
    assert transaction.source_path.read_bytes() == source_before
    assert reservation.sibling_path.read_bytes() == b"partial-video"
    assert reservation.index_path.is_file()


@pytest.mark.parametrize("descriptor_kind", ["extent", "final-target"])
def test_failed_retirement_preserves_a_foreign_temp_only_descriptor(
    tmp_path,
    descriptor_kind,
) -> None:
    artifact = _artifact()
    _install_schema5_job(tmp_path, artifact)
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    transaction = reservation.input_manifest_transaction
    assert transaction is not None
    if descriptor_kind == "extent":
        descriptor = transaction.descriptor_path
        transaction.source_path.unlink()
        transaction.target_path.unlink()
    else:
        descriptor = reservation.target_descriptor_path
        reservation.final_video_path.unlink()
    descriptor.unlink()
    temporary = descriptor.with_name(f".{descriptor.name}.create-new.tmp")
    foreign = b'{"foreign":true}'
    temporary.write_bytes(foreign)
    failed = artifact.failed(
        terminal_at="2026-08-24T12:35:00.000000Z",
        diagnostic_code="FINAL_ENCODING_FAILED",
        diagnostic_message="fixture",
    )
    reservation_protocol.commit_terminal_settings_v1(
        job_root=tmp_path,
        current=artifact,
        target=failed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    final_transaction = reservation.final_manifest_transaction
    assert final_transaction is not None
    unrelated_source_before = final_transaction.source_path.read_bytes()

    with pytest.raises(reservation_protocol.ReservationConflictError):
        recover_final_encoding_terminal_retirement_v1(
            tmp_path,
            failed,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
        )

    assert temporary.read_bytes() == foreign
    assert reservation.index_path.is_file()
    assert final_transaction.source_path.read_bytes() == unrelated_source_before


def test_failed_retirement_keeps_index_temporary_until_full_preflight_passes(
    tmp_path,
) -> None:
    artifact = _artifact()
    _install_schema5_job(tmp_path, artifact)
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    failed = artifact.failed(
        terminal_at="2026-08-24T12:35:00.000000Z",
        diagnostic_code="FINAL_ENCODING_FAILED",
        diagnostic_message="fixture",
    )
    reservation_protocol.commit_terminal_settings_v1(
        job_root=tmp_path,
        current=artifact,
        target=failed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    index_temporary = reservation.index_path.with_name(
        f".{reservation.index_path.name}.create-new.tmp"
    )
    index_temporary.write_bytes(b"stale-index-temporary")
    reservation.sibling_path.write_bytes(b"partial-video")
    reservation.target_descriptor_path.write_bytes(b'{"damaged":true}')
    source = reservation.input_manifest_transaction.source_path
    source_before = source.read_bytes()

    with pytest.raises(reservation_protocol.ReservationConflictError):
        recover_processing_control_prefix_v1(
            tmp_path,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
        )

    assert index_temporary.read_bytes() == b"stale-index-temporary"
    assert reservation.sibling_path.read_bytes() == b"partial-video"
    assert source.read_bytes() == source_before
    assert reservation.index_path.is_file()


def test_failed_retirement_rejects_swapped_zero_placeholder_before_deletion(
    tmp_path,
) -> None:
    artifact = _artifact()
    _install_schema5_job(tmp_path, artifact)
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    reservation.sibling_path.write_bytes(b"partial-video")
    input_source = reservation.input_manifest_transaction.source_path
    source_before = input_source.read_bytes()
    original_identity = reservation_protocol._identity(reservation.final_video_path)
    reservation.final_video_path.unlink()
    reservation.final_video_path.write_bytes(b"")
    assert (
        reservation_protocol._identity(reservation.final_video_path)
        != original_identity
    )
    failed = artifact.failed(
        terminal_at="2026-08-24T12:35:00.000000Z",
        diagnostic_code="FINAL_ENCODING_FAILED",
        diagnostic_message="fixture",
    )
    reservation_protocol.commit_terminal_settings_v1(
        job_root=tmp_path,
        current=artifact,
        target=failed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )

    with pytest.raises(
        reservation_protocol.ReservationConflictError,
        match="placeholder identity",
    ):
        recover_final_encoding_terminal_retirement_v1(
            tmp_path,
            failed,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
        )

    assert reservation.sibling_path.read_bytes() == b"partial-video"
    assert input_source.read_bytes() == source_before
    assert reservation.index_path.is_file()


def test_reserved_publication_rejects_nonempty_owned_placeholder_before_replacement(
    tmp_path,
) -> None:
    artifact = _producer_artifact()
    _install_schema5_job(tmp_path, artifact)
    raw, fingerprint = _input_manifest()
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=len(raw),
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    input_transaction = reservation.input_manifest_transaction
    assert input_transaction is not None
    input_transaction.stage_complete_full(raw)
    reservation.sibling_path.write_bytes(b"validated-video")
    reservation.final_video_path.write_bytes(b"unauthorized-placeholder-bytes")
    source_before = input_transaction.source_path.read_bytes()
    descriptor_before = input_transaction.descriptor_path.read_bytes()

    with pytest.raises(
        reservation_protocol.ReservationConflictError,
        match="placeholder.*nonempty",
    ):
        FinalPublicationTransactionV4(
            job_root=tmp_path,
            settings=artifact,
            sibling_relative_path=reservation.sibling_path.name,
            encoding_input_manifest_raw=raw,
            input_stage_fingerprint=fingerprint,
            resolved_encoding_arguments=_assembled_arguments(
                fingerprint,
                artifact.to_object()["output_info"]["expected_final_relative_path"],
            ),
            container_validator=lambda _path: True,
            reservation=reservation,
            retire_reservation=False,
        ).publish()

    assert (
        reservation.final_video_path.read_bytes() == b"unauthorized-placeholder-bytes"
    )
    assert reservation.sibling_path.read_bytes() == b"validated-video"
    assert input_transaction.source_path.read_bytes() == source_before
    assert input_transaction.descriptor_path.read_bytes() == descriptor_before
    assert reservation.index_path.is_file()


def test_failed_retirement_preflights_committed_manifest_before_any_mutation(
    tmp_path,
) -> None:
    artifact = _artifact()
    _install_schema5_job(tmp_path, artifact)
    raw, _fingerprint = _input_manifest()
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=len(raw),
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    transaction = reservation.input_manifest_transaction
    assert transaction is not None
    transaction.stage_complete_full(raw)
    with pytest.raises(reservation_protocol.InjectedReservationCrash):
        transaction.consume_streamed_prefix(
            prefix_byte_count=len(raw),
            prefix_sha256=hashlib.sha256(raw).hexdigest(),
            crash_at="after-rename-parent-sync",
        )
    assert transaction.descriptor_path.is_file()
    assert not transaction.source_path.exists()
    reservation.sibling_path.write_bytes(b"partial-video")
    reservation.target_descriptor_path.write_bytes(b'{"damaged":true}')
    failed = artifact.failed(
        terminal_at="2026-08-24T12:35:00.000000Z",
        diagnostic_code="FINAL_ENCODING_FAILED",
        diagnostic_message="fixture",
    )
    reservation_protocol.commit_terminal_settings_v1(
        job_root=tmp_path,
        current=artifact,
        target=failed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    descriptor_before = transaction.descriptor_path.read_bytes()
    target_before = transaction.target_path.read_bytes()

    with pytest.raises(reservation_protocol.ReservationConflictError):
        recover_final_encoding_terminal_retirement_v1(
            tmp_path,
            failed,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
        )

    assert transaction.descriptor_path.read_bytes() == descriptor_before
    assert transaction.target_path.read_bytes() == target_before
    assert reservation.sibling_path.read_bytes() == b"partial-video"
    assert reservation.index_path.is_file()


def test_failed_retirement_rejects_nonempty_owned_placeholder_before_any_mutation(
    tmp_path,
) -> None:
    artifact = _artifact()
    _install_schema5_job(tmp_path, artifact)
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    input_transaction = reservation.input_manifest_transaction
    assert input_transaction is not None
    reservation.sibling_path.write_bytes(b"partial-video")
    reservation.final_video_path.write_bytes(b"unauthorized-placeholder-bytes")
    failed = artifact.failed(
        terminal_at="2026-08-24T12:35:00.000000Z",
        diagnostic_code="FINAL_ENCODING_FAILED",
        diagnostic_message="fixture",
    )
    reservation_protocol.commit_terminal_settings_v1(
        job_root=tmp_path,
        current=artifact,
        target=failed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    source_before = input_transaction.source_path.read_bytes()

    with pytest.raises(
        reservation_protocol.ReservationConflictError,
        match="placeholder.*nonempty",
    ):
        recover_final_encoding_terminal_retirement_v1(
            tmp_path,
            failed,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
        )

    assert (
        reservation.final_video_path.read_bytes() == b"unauthorized-placeholder-bytes"
    )
    assert reservation.sibling_path.read_bytes() == b"partial-video"
    assert input_transaction.source_path.read_bytes() == source_before
    assert reservation.index_path.is_file()


def test_failed_retirement_normalizes_descriptor_prefix_before_source_retirement(
    tmp_path,
) -> None:
    artifact = _producer_artifact()
    _install_schema5_job(tmp_path, artifact)
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    transaction = reservation.input_manifest_transaction
    assert transaction is not None
    descriptor_temporary = transaction.descriptor_path.with_name(
        f".{transaction.descriptor_path.name}.create-new.tmp"
    )
    descriptor_raw = transaction.descriptor_path.read_bytes()
    transaction.descriptor_path.unlink()
    descriptor_temporary.write_bytes(descriptor_raw[: len(descriptor_raw) // 2])
    reservation.sibling_path.write_bytes(b"partial-video")
    failed = artifact.failed(
        terminal_at="2026-08-24T12:35:00.000000Z",
        diagnostic_code="FINAL_ENCODING_FAILED",
        diagnostic_message="fixture",
    )
    reservation_protocol.commit_terminal_settings_v1(
        job_root=tmp_path,
        current=artifact,
        target=failed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    entry_position = reservation.index.extent_entries.index(
        next(
            entry
            for entry in reservation.index.extent_entries
            if entry["reservation_kind"] == "encoding_input_manifest"
        )
    )

    with pytest.raises(
        InjectedFinalPublicationCrash,
        match=f"failure-retirement:{entry_position}:source",
    ):
        recover_final_encoding_terminal_retirement_v1(
            tmp_path,
            failed,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
            crash_at=f"failure-retirement:{entry_position}:source",
        )

    assert transaction.descriptor_path.read_bytes() == descriptor_raw
    assert not descriptor_temporary.exists()
    assert not transaction.source_path.exists()
    assert not reservation.sibling_path.exists()
    assert reservation.index_path.is_file()

    assert recover_final_encoding_terminal_retirement_v1(
        tmp_path,
        failed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    assert not transaction.descriptor_path.exists()
    assert not reservation.index_path.exists()


def test_failed_retirement_readonly_preflight_never_repairs_an_early_short_source(
    tmp_path,
) -> None:
    artifact = _producer_artifact()
    _install_schema5_job(tmp_path, artifact)
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    transaction = reservation.input_manifest_transaction
    assert transaction is not None
    descriptor_raw = transaction.descriptor_path.read_bytes()
    descriptor_temporary = transaction.descriptor_path.with_name(
        f".{transaction.descriptor_path.name}.create-new.tmp"
    )
    transaction.descriptor_path.unlink()
    descriptor_temporary.write_bytes(descriptor_raw[:37])
    transaction.source_path.write_bytes(b"\x00")
    final_transaction = reservation.final_manifest_transaction
    assert final_transaction is not None
    final_transaction.target_path.write_bytes(b"foreign")
    failed = artifact.failed(
        terminal_at="2026-08-24T12:35:00.000000Z",
        diagnostic_code="FINAL_ENCODING_FAILED",
        diagnostic_message="fixture",
    )
    reservation_protocol.commit_terminal_settings_v1(
        job_root=tmp_path,
        current=artifact,
        target=failed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    prefix_before = descriptor_temporary.read_bytes()
    later_target_before = final_transaction.target_path.read_bytes()

    with pytest.raises(
        reservation_protocol.ReservationIncompleteError,
        match="read-only validation",
    ):
        recover_final_encoding_terminal_retirement_v1(
            tmp_path,
            failed,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
        )

    assert descriptor_temporary.read_bytes() == prefix_before
    assert transaction.source_path.read_bytes() == b"\x00"
    assert final_transaction.target_path.read_bytes() == later_target_before
    assert reservation.index_path.is_file()


@pytest.mark.parametrize(
    ("stop_after_input", "retry_sibling", "authenticated"),
    [
        (True, False, False),
        (True, True, False),
        (False, False, True),
    ],
)
def test_failed_retirement_preserves_each_legal_post_boundary_publication_prefix(
    tmp_path,
    stop_after_input,
    retry_sibling,
    authenticated,
) -> None:
    artifact = _producer_artifact()
    _install_schema5_job(tmp_path, artifact)
    raw, fingerprint = _input_manifest()
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=len(raw),
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    input_transaction = reservation.input_manifest_transaction
    assert input_transaction is not None
    input_transaction.stage_complete_full(raw)
    reservation.sibling_path.write_bytes(b"published-post-boundary-video")
    publication = FinalPublicationTransactionV4(
        job_root=tmp_path,
        settings=artifact,
        sibling_relative_path=reservation.sibling_path.name,
        encoding_input_manifest_raw=raw,
        input_stage_fingerprint=fingerprint,
        resolved_encoding_arguments=_assembled_arguments(
            fingerprint,
            artifact.to_object()["output_info"]["expected_final_relative_path"],
        ),
        container_validator=lambda _path: True,
        reservation=reservation,
        retire_reservation=False,
    )
    if stop_after_input:
        with pytest.raises(InjectedFinalPublicationCrash):
            publication.publish(crash_at="after-input-manifest")
    else:
        publication.publish()
    if retry_sibling:
        reservation.sibling_path.write_bytes(b"unpublished-retry-video")

    failed = artifact.failed(
        terminal_at="2026-08-24T12:35:00.000000Z",
        diagnostic_code="FINAL_ENCODING_FAILED",
        diagnostic_message="fixture",
    )
    reservation_protocol.commit_terminal_settings_v1(
        job_root=tmp_path,
        current=artifact,
        target=failed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    assert recover_final_encoding_terminal_retirement_v1(
        tmp_path,
        failed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )

    assert reservation.final_video_path.read_bytes() == b"published-post-boundary-video"
    assert (tmp_path / "encoding_input_manifest.json").read_bytes() == raw
    assert not reservation.sibling_path.exists()
    assert not reservation.index_path.exists()
    disposition = audit_final_media_v1(
        tmp_path,
        failed,
        container_validator=lambda _path: True,
    )
    assert disposition.final_video_authenticated is authenticated
    assert disposition.final_video_authentication_reason == (
        None if authenticated else "incomplete_publication_evidence"
    )


@pytest.mark.skipif(
    os.name != "nt", reason="Windows replacement gaps are platform-specific"
)
def test_root_control_recovery_restores_final_index_owned_missing_settings(
    tmp_path,
) -> None:
    artifact = _artifact()
    _install_schema5_job(tmp_path, artifact)
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    transaction = reservation.producer_settings_transaction
    payload = reservation.producer_settings_payload
    assert transaction is not None and payload is not None
    transaction.stage_complete_full(payload)
    transaction.truncate_to_payload(payload)
    transaction._short_release_recovery(payload)
    (tmp_path / "processing-settings-v5.json").unlink()

    recovered = recover_processing_control_prefix_v1(
        tmp_path,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )

    assert recovered is not None
    assert (
        recovered.to_object()["metadata"]["final_media_publication_generation"]
        == "44" * 16
    )
    assert (tmp_path / "processing-settings-v5.json").read_bytes() == payload
