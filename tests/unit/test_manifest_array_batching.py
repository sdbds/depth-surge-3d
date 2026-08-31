"""Batch bounded arrays without retaining evidence or changing failure boundaries."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math

import pytest

from src.depth_surge_3d.processing.video import final_publication as publication
from tests.unit.test_encoding_manifest_reader import _CountedDigest, _FragmentedInput


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


def _values(kind, count=48, *, first=1, parent="frames"):
    names = [f"frame_{index:06d}" for index in range(first, first + count)]
    if kind == "names":
        return names
    return [
        {
            "byte_count": 60 if kind == "raw" else 67 + index,
            "raw_frame" if kind == "raw" else "png_header": (
                {"height": 2, "pixel_format": "rgb24", "width": 10}
                if kind == "raw"
                else {"height": 2, "width": 10, "bit_depth": 8, "color_type": 2}
            ),
            "relative_path": f"{parent}/{name}.{'rgb' if kind == 'raw' else 'png'}",
            "sha256": f"{index + 1:064x}",
        }
        for index, name in enumerate(names)
    ]


def _parse(kind, reader, count=48, *, first=1):
    if kind == "names":
        return publication._parse_frame_names_v1(reader, expected_frame_count=count)
    return publication._parse_identity_array_v1(
        reader, first_index=first, frame_count=count, raw_rgb24=kind == "raw"
    )


@pytest.mark.parametrize("kind", ["names", "png", "raw"])
@pytest.mark.parametrize("self_hash", [False, True])
def test_complete_buffered_arrays_decode_and_hash_as_one_batch(monkeypatch, kind, self_hash):
    raw = _encoded(_values(kind))
    prefix, suffix = b"prefix:", b",unconsumed"
    reader = publication._EncodingManifestReader(
        io.BytesIO(prefix + raw + suffix), raw_byte_count=None
    )
    reader.raw_digest = _CountedDigest()
    reader.self_digest = _CountedDigest()
    reader.expect(prefix)
    reader.include_in_self_digest = self_hash
    loads = []
    original = publication.strict_canonical_json_loads

    def observe(value, **kwargs):
        loads.append(len(value))
        return original(value, **kwargs)

    monkeypatch.setattr(publication, "strict_canonical_json_loads", observe)
    result = _parse(kind, reader)

    assert result == ((1, 48) if kind == "names" else (48, "frames", (10, 2)))
    assert reader.consumed == len(prefix + raw) and reader.peek_byte() == ord(",")
    assert reader.raw_digest.hexdigest() == hashlib.sha256(prefix + raw).hexdigest()
    assert reader.self_digest.hexdigest() == hashlib.sha256(raw if self_hash else b"").hexdigest()
    assert loads == [len(raw)]
    assert reader.raw_digest.updates == 2
    assert reader.self_digest.updates == int(self_hash)


@pytest.mark.parametrize("kind", ["names", "png", "raw"])
@pytest.mark.parametrize("fragment", [1, 7, 64, 65536])
def test_array_batching_preserves_fragmented_and_limited_reads(kind, fragment):
    raw = _encoded(_values(kind, 3))
    handle = _FragmentedInput(raw + b"outside-limit", fragment)
    reader = publication._EncodingManifestReader(handle, raw_byte_count=len(raw))
    reader.include_in_self_digest = True

    assert _parse(kind, reader, 3) == ((1, 3) if kind == "names" else (3, "frames", (10, 2)))
    assert reader.consumed == handle.tell() == len(raw)
    assert reader.peek_byte() is None
    assert (
        reader.raw_digest.hexdigest()
        == reader.self_digest.hexdigest()
        == hashlib.sha256(raw).hexdigest()
    )


@pytest.mark.parametrize("kind", ["png", "raw"])
@pytest.mark.parametrize(
    "parent", ["frames]suffix", "\u4e2d\U0001f642", 'frames"quoted', "frames/nested"]
)
def test_array_shortcut_does_not_restrict_valid_identity_paths(kind, parent):
    raw = _encoded(_values(kind, 3, parent=parent))
    reader = publication._EncodingManifestReader(io.BytesIO(raw + b","), raw_byte_count=None)

    assert _parse(kind, reader, 3) == (3, parent, (10, 2))
    assert reader.consumed == len(raw) and reader.peek_byte() == ord(",")
    assert reader.raw_digest.hexdigest() == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("kind", ["names", "png", "raw"])
def test_array_requires_its_closing_bracket_inside_the_raw_limit(kind):
    raw = _encoded(_values(kind, 2))
    handle = io.BytesIO(raw + b"suffix")
    reader = publication._EncodingManifestReader(handle, raw_byte_count=len(raw) - 1)

    with pytest.raises(publication.FinalPublicationValidationError, match="ended unexpectedly"):
        _parse(kind, reader, 2)
    assert reader.consumed == handle.tell() == len(raw) - 1
    assert reader.raw_digest.hexdigest() == hashlib.sha256(raw[:-1]).hexdigest()


@pytest.mark.parametrize(
    ("raw", "expected_count", "message", "consumed"),
    [
        (b'["bad","frame_000001"]', 2, "frame name is noncanonical", 6),
        (b'["frame_000001","frame_000003"]', 2, "not gap-free", 30),
        (b'["frame_000001","frame_000002"]', 3, "settings frame count", 31),
        (b'[1,"frame_000001"]', 2, "expected a JSON string", 1),
        (b'["frame_000001", "frame_000002"]', 2, "expected a JSON string", 16),
        (b'["bad","\\ud800"]', 2, "frame name is noncanonical", 6),
        (b"[]", 1, "frame list is empty", 2),
    ],
)
def test_failed_name_batch_preserves_first_error_and_consumed_hash(
    raw, expected_count, message, consumed
):
    reader = publication._EncodingManifestReader(io.BytesIO(raw), raw_byte_count=None)
    reader.include_in_self_digest = True

    with pytest.raises(publication.FinalPublicationValidationError, match=message):
        _parse("names", reader, expected_count)
    assert reader.consumed == consumed
    assert (
        reader.raw_digest.hexdigest()
        == reader.self_digest.hexdigest()
        == hashlib.sha256(raw[:consumed]).hexdigest()
    )


@pytest.mark.parametrize("kind", ["png", "raw"])
@pytest.mark.parametrize("bad_tail", [b'{"sha256":"a","sha256":"b"}', b'{"bad":NaN}', b"17", b"[]"])
def test_failed_identity_batch_does_not_hide_an_earlier_semantic_error(kind, bad_tail):
    first = _values(kind, 1)[0]
    first["byte_count"] = 0
    first_raw = _encoded(first)
    raw = b"[" + first_raw + b"," + bad_tail + b"]"
    reader = publication._EncodingManifestReader(io.BytesIO(raw), raw_byte_count=None)

    with pytest.raises(publication.FinalPublicationValidationError, match="byte count"):
        _parse(kind, reader, 2)
    assert reader.consumed == 1 + len(first_raw)
    assert reader.raw_digest.hexdigest() == hashlib.sha256(b"[" + first_raw).hexdigest()


@pytest.mark.parametrize("kind", ["names", "png", "raw"])
@pytest.mark.parametrize("batch", ["whole", "items"])
def test_recursive_later_item_preserves_the_earlier_error_and_hash_boundary(kind, batch):
    first = "bad" if kind == "names" else _values(kind, 1)[0]
    if kind != "names":
        first["byte_count"] = 0
    first_raw = _encoded(first)
    recursive = b'{"a":' * 1100 + b"0" + b"}" * 1100
    raw = b"[" + first_raw + b"," + recursive
    count = 2
    if batch == "items":
        tail = _values(kind, 2, first=3, parent="x" * 65536)
        if kind == "names":
            tail[1] = "x" * 65536
        else:
            tail[0] = _values(kind, 1, first=3)[0]
        raw += b"," + _encoded(tail)[1:-1]
        count = 4
    raw += b"]"
    prefix = b"prefix:"
    reader = publication._EncodingManifestReader(
        io.BytesIO(prefix + raw + b",unconsumed"), raw_byte_count=None
    )
    reader.expect(prefix)
    reader.include_in_self_digest = True
    if batch == "items":
        assert b"]" not in reader.buffer
    else:
        assert reader.buffer.endswith(b",unconsumed")

    message = "frame name is noncanonical" if kind == "names" else "byte count"
    with pytest.raises(publication.FinalPublicationValidationError, match=message):
        _parse(kind, reader, count)
    consumed_array = b"[" + first_raw
    assert reader.consumed == len(prefix + consumed_array)
    assert reader.raw_digest.hexdigest() == hashlib.sha256(prefix + consumed_array).hexdigest()
    assert reader.self_digest.hexdigest() == hashlib.sha256(consumed_array).hexdigest()


@pytest.mark.parametrize("kind", ["png", "raw"])
@pytest.mark.parametrize("batch", ["whole", "items"])
def test_recursive_first_identity_still_raises_after_consuming_its_token(kind, batch):
    recursive = b'{"a":' * 1100 + b"0" + b"}" * 1100
    tail = _values(kind, 2, first=2, parent="x" * 65536 if batch == "items" else "frames")
    tail[0] = _values(kind, 1, first=2)[0]
    raw = b"[" + recursive + b"," + _encoded(tail)[1:-1] + b"]"
    reader = publication._EncodingManifestReader(io.BytesIO(raw), raw_byte_count=None)
    reader.include_in_self_digest = True

    with pytest.raises(RecursionError):
        _parse(kind, reader, 3)
    consumed = b"[" + recursive
    assert reader.consumed == len(consumed)
    assert (
        reader.raw_digest.hexdigest()
        == reader.self_digest.hexdigest()
        == hashlib.sha256(consumed).hexdigest()
    )


@pytest.mark.parametrize("kind", ["png", "raw"])
@pytest.mark.parametrize(
    "mutation", ["extra_key", "shape", "parent", "stem", "hash", "cardinality"]
)
def test_buffered_identity_arrays_retain_schema_and_sequence_checks(kind, mutation):
    values = _values(kind, 3)
    messages = {
        "extra_key": "schema is not closed",
        "shape": "shape changes within",
        "parent": "parent changes within",
        "stem": "filename differs",
        "hash": "hash is invalid",
        "cardinality": "cardinality differs",
    }
    if mutation == "extra_key":
        values[1]["unexpected"] = True
    elif mutation == "shape":
        values[1]["raw_frame" if kind == "raw" else "png_header"]["width"] = 11
        if kind == "raw":
            values[1]["byte_count"] = 66
    elif mutation == "parent":
        values[1]["relative_path"] = values[1]["relative_path"].replace("frames/", "changed/")
    elif mutation == "stem":
        values[1]["relative_path"] = values[1]["relative_path"].replace("000002", "000009")
    elif mutation == "hash":
        values[1]["sha256"] = "f" * 63
    elif mutation == "cardinality":
        values.pop()
    reader = publication._EncodingManifestReader(io.BytesIO(_encoded(values)), raw_byte_count=None)

    with pytest.raises(publication.FinalPublicationValidationError, match=messages[mutation]):
        _parse(kind, reader, 3)


@pytest.mark.parametrize("kind", ["names", "png", "raw"])
def test_batching_never_exceeds_the_existing_value_cap(monkeypatch, kind):
    raw = _encoded(_values(kind, 1))
    cap = len(raw) - 3
    monkeypatch.setattr(publication, "_ENCODING_MANIFEST_VALUE_MAX_BYTES", cap)
    reader = publication._EncodingManifestReader(io.BytesIO(raw), raw_byte_count=None)

    with pytest.raises(publication.FinalPublicationValidationError, match="exceeds its bound"):
        _parse(kind, reader, 1)
    assert reader.consumed == cap + 1
    assert reader.raw_digest.hexdigest() == hashlib.sha256(raw[: cap + 1]).hexdigest()


@pytest.mark.parametrize("kind", ["names", "png", "raw"])
def test_array_results_and_hashes_are_not_reused_across_calls(kind):
    def parse(first):
        raw = _encoded(_values(kind, 4, first=first))
        reader = publication._EncodingManifestReader(io.BytesIO(raw), raw_byte_count=None)
        result = _parse(kind, reader, 4, first=first)
        assert result == ((first, 4) if kind == "names" else (4, "frames", (10, 2)))
        assert reader.raw_digest.hexdigest() == hashlib.sha256(raw).hexdigest()
        return reader.raw_digest.hexdigest()

    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes = list(pool.map(parse, range(1, 9)))
    assert len(set(hashes)) == 8


@pytest.mark.parametrize(
    ("kind", "count"),
    [("names", 9000), ("png", 360), ("png", 9000), ("raw", 360), ("raw", 9000)],
)
def test_arrays_larger_than_one_buffer_still_batch_without_bulk_decoding(monkeypatch, kind, count):
    raw = _encoded(_values(kind, count))
    assert len(raw) > publication._ENCODING_MANIFEST_STREAM_BYTES
    reader = publication._EncodingManifestReader(io.BytesIO(b":" + raw + b","), raw_byte_count=None)
    reader.raw_digest = _CountedDigest()
    reader.self_digest = _CountedDigest()
    reader.expect(b":")
    reader.include_in_self_digest = True
    loads = []
    original = publication.strict_canonical_json_loads

    def observe(value, **kwargs):
        loads.append(len(value))
        assert len(value) <= publication._ENCODING_MANIFEST_STREAM_BYTES
        return original(value, **kwargs)

    monkeypatch.setattr(publication, "strict_canonical_json_loads", observe)

    assert _parse(kind, reader, count) == (
        (1, count) if kind == "names" else (count, "frames", (10, 2))
    )
    blocks = math.ceil((len(raw) + 1) / publication._ENCODING_MANIFEST_STREAM_BYTES)
    assert len(loads) <= 3 * blocks + 2
    assert reader.raw_digest.updates <= 8 * blocks + 4
    assert reader.consumed == len(raw) + 1 and reader.peek_byte() == ord(",")
    assert reader.raw_digest.hexdigest() == hashlib.sha256(b":" + raw).hexdigest()
    assert reader.self_digest.hexdigest() == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("kind", ["png", "raw"])
@pytest.mark.parametrize(
    "parent",
    ['frames]},{"quoted' * 8, "x" * 65536 + ']},{"quoted'],
    ids=["quoted_delimiters", "long_quoted_path"],
)
def test_unusual_multibuffer_paths_attempt_batches_at_most_once_per_buffer(
    monkeypatch, kind, parent
):
    count = 360 if len(parent) < 65536 else 4
    raw = _encoded(_values(kind, count, parent=parent))
    reader = publication._EncodingManifestReader(io.BytesIO(raw + b","), raw_byte_count=None)
    attempts = []
    original = publication._buffered_array_items_v1

    def observe(reader, **kwargs):
        attempts.append(reader.loaded)
        result = original(reader, **kwargs)
        if result is not None:
            assert result[1] + 2 <= publication._ENCODING_MANIFEST_STREAM_BYTES
        return result

    monkeypatch.setattr(publication, "_buffered_array_items_v1", observe)
    assert _parse(kind, reader, count) == (count, parent, (10, 2))
    assert len(attempts) == len(set(attempts))
    assert len(attempts) <= math.ceil(len(raw) / publication._ENCODING_MANIFEST_STREAM_BYTES)
    assert reader.consumed == len(raw) and reader.peek_byte() == ord(",")
    assert reader.raw_digest.hexdigest() == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("kind", ["png", "raw"])
@pytest.mark.parametrize("mutation", ["byte_count", "shape", "parent"])
def test_later_array_batches_preserve_global_checks_and_failure_boundary(kind, mutation):
    values = _values(kind, 500)
    bad_index = 400
    messages = {
        "byte_count": "byte count",
        "shape": "shape changes within",
        "parent": "parent changes within",
    }
    if mutation == "byte_count":
        values[bad_index]["byte_count"] = 0
    elif mutation == "shape":
        values[bad_index]["raw_frame" if kind == "raw" else "png_header"]["width"] = 11
        if kind == "raw":
            values[bad_index]["byte_count"] = 66
    else:
        values[bad_index]["relative_path"] = values[bad_index]["relative_path"].replace(
            "frames/", "changed/"
        )
    raw = _encoded(values)
    bad_raw = _encoded(values[bad_index])
    consumed = raw.index(bad_raw) + len(bad_raw)
    reader = publication._EncodingManifestReader(io.BytesIO(raw), raw_byte_count=None)
    reader.include_in_self_digest = True

    with pytest.raises(publication.FinalPublicationValidationError, match=messages[mutation]):
        _parse(kind, reader, 500)
    assert reader.consumed == consumed
    assert (
        reader.raw_digest.hexdigest()
        == reader.self_digest.hexdigest()
        == hashlib.sha256(raw[:consumed]).hexdigest()
    )


def test_name_batches_preserve_global_control_cap_and_failure_boundary(monkeypatch):
    values = _values("names", 9000)
    raw = _encoded(values)
    monkeypatch.setattr(publication, "_ENCODING_CONTROL_FRAME_CAP", 5000)
    bad_raw = _encoded(values[5000])
    consumed = raw.index(bad_raw) + len(bad_raw)
    reader = publication._EncodingManifestReader(io.BytesIO(raw), raw_byte_count=None)

    with pytest.raises(
        publication.FinalPublicationValidationError, match="exceeds the control cap"
    ):
        _parse("names", reader, 9000)
    assert reader.consumed == consumed
    assert reader.raw_digest.hexdigest() == hashlib.sha256(raw[:consumed]).hexdigest()
