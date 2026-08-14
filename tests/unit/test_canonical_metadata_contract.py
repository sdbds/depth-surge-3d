"""One authoritative contract governs canonical depth metadata."""

from src.depth_surge_3d.core.depth_contract import (
    CANONICAL_DEPTH_ALGORITHM_VERSION,
    CANONICAL_DEPTH_SCHEMA_VERSION,
    CANONICAL_METADATA_REQUIRED_FIELDS,
    canonical_json_hash,
)


def test_canonical_contract_contains_identity_and_payload_fields() -> None:
    assert CANONICAL_DEPTH_SCHEMA_VERSION == 1
    assert CANONICAL_DEPTH_ALGORITHM_VERSION == "scene-percentile-v1"
    assert {
        "schema_version",
        "algorithm_version",
        "native_shape",
        "source_raw_fingerprint",
        "fingerprint",
    } <= CANONICAL_METADATA_REQUIRED_FIELDS


def test_canonical_hash_is_stable_across_dictionary_order() -> None:
    assert canonical_json_hash({"b": 2, "a": 1}) == canonical_json_hash({"a": 1, "b": 2})
