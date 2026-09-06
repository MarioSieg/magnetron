# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

import gc
import json
import struct
from pathlib import Path

import pytest

from magnetron import Tensor, dtype
from magnetron.snapshot import SnapshotWriter, _FileManifest, _TBLOB_ALIGN, deserialize

_HDR_SIZE: int = 4 + 4 + 4 + 8 + 8 + 8 + 8  # magic, version, aux, meta range, blob range


def _sample(dt: dtype.DType, *shape: int) -> Tensor:
    numel: int = 1
    for dim in shape:
        numel *= dim
    return Tensor.arange(numel, dtype=dtype.float32).reshape(*shape).cast(dt)


def _write(path: Path, tensors: dict[str, Tensor], metadata: dict | None = None) -> Path:
    with SnapshotWriter(path, metadata) as writer:
        for name, tensor in tensors.items():
            writer.declare(name, tensor.shape, tensor.dtype)
        for name, tensor in tensors.items():
            writer.write(name, tensor)
    return path


def _read_manifest(path: Path) -> tuple[_FileManifest, int]:
    raw = path.read_bytes()
    _magic, _ver, _aux, meta_off, meta_len, blob_off, blob_len = struct.unpack('<IIIQQQQ', raw[:_HDR_SIZE])
    return _FileManifest.deserialize(raw[meta_off : meta_off + meta_len].decode()), blob_len


@pytest.mark.parametrize('dt', sorted(dtype.all, key=lambda d: d.ordinal), ids=lambda d: d.name)
def test_roundtrip_preserves_values_for_every_dtype(tmp_path: Path, dt: dtype.DType) -> None:
    src = {'mat': _sample(dt, 2, 3), 'vec': _sample(dt, 5), 'cube': _sample(dt, 2, 2, 2)}
    got, _ = deserialize(_write(tmp_path / 'dtypes.mag', src))
    assert set(got) == set(src)
    for name, want in src.items():
        assert tuple(got[name].shape) == tuple(want.shape)
        assert got[name].dtype == want.dtype
        assert got[name].tolist() == want.tolist()


def test_roundtrip_preserves_metadata(tmp_path: Path) -> None:
    meta = {'arch': 'qwen3', 'layers': [1, 2, 3], 'cfg': {'tied': True, 'rope': 1e6}, 'none': None}
    _, got = deserialize(_write(tmp_path / 'meta.mag', {'t': _sample(dtype.float32, 4)}, meta))
    assert got == meta


def test_roundtrip_without_metadata(tmp_path: Path) -> None:
    _, got = deserialize(_write(tmp_path / 'nometa.mag', {'t': _sample(dtype.float32, 4)}))
    assert got == {}


def test_every_tensor_starts_on_a_page(tmp_path: Path) -> None:
    src = {f't{i}': _sample(dtype.float32, i + 1) for i in range(6)}  # Sizes that never land on a page by luck
    path = _write(tmp_path / 'aligned.mag', src)
    manifest, blob_len = _read_manifest(path)
    manifest.validate(blob_len)
    for name, meta in manifest.tensor_map.items():
        assert meta.offset % _TBLOB_ALIGN == 0, f'{name} at {meta.offset}'
    raw = path.read_bytes()
    _, _, _, _, _, blob_off, _ = struct.unpack('<IIIQQQQ', raw[:_HDR_SIZE])
    assert blob_off % _TBLOB_ALIGN == 0


@pytest.mark.parametrize('position', ['leading', 'interior', 'trailing'])
def test_zero_element_tensors(tmp_path: Path, position: str) -> None:
    empty = Tensor.empty(0, dtype=dtype.float32)
    order = {
        'leading': [('e', empty), ('a', _sample(dtype.float32, 3))],
        'interior': [('a', _sample(dtype.float32, 3)), ('e', empty), ('b', _sample(dtype.float32, 2))],
        'trailing': [('a', _sample(dtype.float32, 3)), ('e', empty)],
    }[position]
    got, _ = deserialize(_write(tmp_path / f'{position}.mag', dict(order)))
    assert tuple(got['e'].shape) == (0,)
    assert got['a'].tolist() == order[0 if position != 'leading' else 1][1].tolist()


def test_mmap_load_shares_the_file_and_copy_does_not(tmp_path: Path) -> None:
    path = _write(tmp_path / 'share.mag', {'t': _sample(dtype.float32, 4)})
    owned, _ = deserialize(path, mmap=False)
    owned['t'].fill_(7.0)
    assert owned['t'].tolist() == [7.0] * 4
    fresh, _ = deserialize(path, mmap=False)
    assert fresh['t'].tolist() == [0.0, 1.0, 2.0, 3.0], 'the copy must not write through to the file'


def test_large_text_metadata_survives_byte_identical(tmp_path: Path) -> None:
    tricky = 'quote " backslash \\ newline \n accent \u00e9 kanji \u6f22'
    blob = json.dumps({'vocab': {f'tok{i}': i for i in range(70_000)}, 'tricky': tricky})
    assert len(blob) > (1 << 20)
    _, got = deserialize(_write(tmp_path / 'bigmeta.mag', {'t': _sample(dtype.float32, 4)}, {'tokenizer_json': blob}))
    assert got['tokenizer_json'] == blob
    assert json.loads(got['tokenizer_json'])['tricky'] == tricky


def test_borrowed_tensor_outlives_the_reader(tmp_path: Path) -> None:
    path = _write(tmp_path / 'life.mag', {'a': _sample(dtype.float32, 4), 'b': _sample(dtype.float32, 4)})
    tensors, _ = deserialize(path)
    survivor = tensors['b']
    del tensors
    gc.collect()
    assert survivor.tolist() == [0.0, 1.0, 2.0, 3.0]


def test_writes_must_follow_declaration_order(tmp_path: Path) -> None:
    path = tmp_path / 'order.mag'
    with pytest.raises(RuntimeError, match='declared order'):
        with SnapshotWriter(path) as writer:
            writer.declare('a', (2,), dtype.float32)
            writer.declare('b', (2,), dtype.float32)
            writer.write('b', _sample(dtype.float32, 2))
    assert not path.exists(), 'a failed write must not leave a snapshot behind'


def test_duplicate_declaration_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        with SnapshotWriter(tmp_path / 'dup.mag') as writer:
            writer.declare('a', (2,), dtype.float32)
            writer.declare('a', (2,), dtype.float32)


def test_undelivered_tensor_leaves_no_file(tmp_path: Path) -> None:
    path = tmp_path / 'short.mag'
    with pytest.raises(RuntimeError, match='never written'):
        with SnapshotWriter(path) as writer:
            writer.declare('a', (2,), dtype.float32)
            writer.declare('b', (2,), dtype.float32)
            writer.write('a', _sample(dtype.float32, 2))
    assert not path.exists()


@pytest.mark.parametrize('mismatch', ['shape', 'dtype'])
def test_payload_must_match_its_declaration(tmp_path: Path, mismatch: str) -> None:
    payload = _sample(dtype.float32, 3) if mismatch == 'shape' else _sample(dtype.int32, 2)
    with pytest.raises(RuntimeError, match='declared'):
        with SnapshotWriter(tmp_path / 'mismatch.mag') as writer:
            writer.declare('a', (2,), dtype.float32)
            writer.write('a', payload)


def _corrupt(raw: bytes, what: str) -> bytes:
    return {
        'magic': b'NOPE' + raw[4:],
        'version': raw[:4] + struct.pack('<I', 99_99_99) + raw[8:],
        'endianness': raw[:8] + struct.pack('<I', 1) + raw[12:],
        'reserved_aux': raw[:8] + struct.pack('<I', 1 << 7) + raw[12:],
        'truncated': raw[: len(raw) // 2],
        'stub': raw[:16],
        'blob_past_eof': raw[:28] + struct.pack('<Q', 1 << 40) + raw[36:],
    }[what]


@pytest.mark.parametrize('what', ['magic', 'version', 'endianness', 'reserved_aux', 'truncated', 'stub', 'blob_past_eof'])
def test_corrupt_files_are_rejected(tmp_path: Path, what: str) -> None:
    good = _write(tmp_path / 'good.mag', {'t': _sample(dtype.float32, 8)})
    bad = tmp_path / 'bad.mag'
    bad.write_bytes(_corrupt(good.read_bytes(), what))
    with pytest.raises(Exception):
        deserialize(bad)


def test_manifest_that_does_not_tile_the_blob_is_rejected(tmp_path: Path) -> None:
    manifest, blob_len = _read_manifest(_write(tmp_path / 'tile.mag', {'a': _sample(dtype.float32, 4), 'b': _sample(dtype.float32, 4)}))
    manifest.validate(blob_len)
    manifest.tensor_map['b'].offset = manifest.tensor_map['a'].offset  # Overlap
    with pytest.raises(ValueError, match='overlaps'):
        manifest.validate(blob_len)


def test_writer_stats_describe_the_file(tmp_path: Path) -> None:
    src = {'a': _sample(dtype.float32, 3), 'b': _sample(dtype.int64, 4)}
    path = tmp_path / 'stats.mag'
    with SnapshotWriter(path, {'k': 'v'}) as writer:
        for name, tensor in src.items():
            writer.declare(name, tensor.shape, tensor.dtype)
        assert writer.tensor_count == 2
        assert writer.payload_numbytes == 3 * 4 + 4 * 8
        assert writer.blob_numbytes >= writer.payload_numbytes
        for name, tensor in src.items():
            writer.write(name, tensor)
    assert path.stat().st_size == _align_up(_HDR_SIZE + writer.metadata_numbytes) + writer.blob_numbytes
    assert json.loads(_read_manifest(path)[0].serialize())['usr_metadata'] == {'k': 'v'}


def _align_up(x: int, a: int = 4096) -> int:
    return (x + a - 1) & ~(a - 1)
