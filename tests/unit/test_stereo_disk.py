"""Per-volume stereo-stage disk admission contracts."""

from pathlib import Path

import pytest

from src.depth_surge_3d.processing.frames.stereo_disk import (
    StereoStageDiskBudgetError,
    StereoStageDiskPlan,
    StereoStageDiskPlanBuilder,
    StereoVolumeRequirement,
    admit_stereo_stage_disk,
)


def test_disk_admission_rejects_one_byte_below_the_per_volume_requirement(
    tmp_path: Path,
) -> None:
    plan = StereoStageDiskPlan((StereoVolumeRequirement("volume", tmp_path, 4096, 100),))

    with pytest.raises(StereoStageDiskBudgetError) as captured:
        admit_stereo_stage_disk(plan, free_bytes=lambda _path: 99)

    assert captured.value.required_bytes == 100
    assert captured.value.free_bytes == 99


def test_disk_admission_accepts_exactly_the_per_volume_requirement(tmp_path: Path) -> None:
    plan = StereoStageDiskPlan((StereoVolumeRequirement("volume", tmp_path, 4096, 100),))

    admit_stereo_stage_disk(plan, free_bytes=lambda _path: 100)


def test_disk_admission_checks_first_mutation_without_reclaim_credit(tmp_path: Path) -> None:
    plan = StereoStageDiskPlan(
        (
            StereoVolumeRequirement(
                "volume",
                tmp_path,
                4096,
                required_bytes=100,
                first_mutation_reserve=50,
            ),
        )
    )

    with pytest.raises(StereoStageDiskBudgetError) as captured:
        admit_stereo_stage_disk(plan, free_bytes=lambda _path: 49)

    assert captured.value.first_mutation_reserve == 50
    assert captured.value.reclaimable_bytes == 0


def test_disk_admission_never_credits_ambient_residue(tmp_path: Path) -> None:
    plan = StereoStageDiskPlan(
        (
            StereoVolumeRequirement(
                "volume",
                tmp_path,
                4096,
                required_bytes=111,
                first_mutation_reserve=50,
            ),
        )
    )

    with pytest.raises(StereoStageDiskBudgetError):
        admit_stereo_stage_disk(plan, free_bytes=lambda _path: 50)

    with pytest.raises(StereoStageDiskBudgetError):
        admit_stereo_stage_disk(plan, free_bytes=lambda _path: 110)
    admit_stereo_stage_disk(plan, free_bytes=lambda _path: 111)


def test_transaction_overlap_is_folded_to_a_scalar_maximum(tmp_path: Path) -> None:
    builder = StereoStageDiskPlanBuilder(
        volume_key=lambda _path: "volume",
        allocation_unit=lambda _path: 4096,
    )
    payload = tmp_path / "payload.bin"
    for _index in range(10_000):
        builder.add_transaction(((payload, 1), (payload, 1)))

    plan = builder.build()

    assert len(plan.volumes) == 1
    assert plan.volumes[0].required_bytes == 4 * 4096
    assert not hasattr(builder, "_overlap")


def test_plan_builder_resolves_one_directory_target_once(tmp_path: Path) -> None:
    volume_queries = 0
    allocation_queries = 0

    def volume_key(_path: Path) -> str:
        nonlocal volume_queries
        volume_queries += 1
        return "volume"

    def allocation_unit(_path: Path) -> int:
        nonlocal allocation_queries
        allocation_queries += 1
        return 4096

    builder = StereoStageDiskPlanBuilder(
        volume_key=volume_key,
        allocation_unit=allocation_unit,
    )
    directory = tmp_path / "frames" / "frame_000001"
    for index in range(10_000):
        builder.add_payload(directory / f"payload_{index}.bin", 1)

    plan = builder.build()

    assert volume_queries == 1
    assert allocation_queries == 1
    assert plan.required_bytes == 10_000 * 4096


def test_plan_builder_still_checks_units_across_directories(tmp_path: Path) -> None:
    units = iter((4096, 8192))
    builder = StereoStageDiskPlanBuilder(
        volume_key=lambda _path: "volume",
        allocation_unit=lambda _path: next(units),
    )
    builder.add_payload(tmp_path / "left" / "payload.bin", 1)

    with pytest.raises(ValueError, match="inconsistent allocation units"):
        builder.add_payload(tmp_path / "right" / "payload.bin", 1)


def test_stage_plan_exposes_no_ambient_reclaim_credit_api(
    tmp_path: Path,
) -> None:
    builder = StereoStageDiskPlanBuilder(
        volume_key=lambda _path: "volume",
        allocation_unit=lambda _path: 4096,
    )
    builder.add_payload(tmp_path / "new.bin", 1)
    volume = builder.build().volumes[0]

    assert not hasattr(builder, "add_reclaimable")
    assert not hasattr(volume, "reclaimable_bytes")
