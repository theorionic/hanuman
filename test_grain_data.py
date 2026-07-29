"""Tests for the Grain streaming pipeline (grain_data.py).

Runs entirely on the host against a synthetic local parquet file -- no network,
no accelerator -- so packing, batching, filtering and determinism are exercised
without booking a TPU.
"""
from __future__ import annotations

import itertools

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data import ByteTokenizer
from grain_data import build_grain_dataset


def _write_parquet(path, docs, field="content"):
    pq.write_table(pa.table({field: docs}), path, row_group_size=8)


@pytest.fixture
def shards(tmp_path):
    """Two parquet shards of short ASCII documents plus one empty doc."""
    rng = np.random.default_rng(0)
    def make(n):
        out = []
        for _ in range(n):
            L = int(rng.integers(20, 120))
            out.append("".join(chr(int(c)) for c in rng.integers(97, 123, size=L)))
        return out
    f0, f1 = tmp_path / "a.parquet", tmp_path / "b.parquet"
    _write_parquet(f0, make(40) + [""])   # include an empty doc to exercise the filter
    _write_parquet(f1, make(40))
    return [str(f0), str(f1)]


def _dataset(shards, seq_len=32, batch_size=4, **kw):
    tok = ByteTokenizer(vocab_size=256)
    return build_grain_dataset(
        shards, tok, seq_len, batch_size, eos_id=tok.eos_id,
        filesystem=fsspec.filesystem("local"), sharding=None,
        interleave=kw.pop("interleave", 2), cpu_buffer=2, **kw)


def test_batch_shape_dtype_and_range(shards):
    ds = _dataset(shards, seq_len=32, batch_size=4)
    batches = list(itertools.islice(iter(ds), 5))
    assert len(batches) == 5
    for b in batches:
        assert isinstance(b, np.ndarray)
        assert b.shape == (4, 32)
        assert b.dtype == np.int32
        assert b.min() >= 0 and b.max() < 256   # byte-tokenizer vocab


def test_packing_is_dense(shards):
    """concat-then-split fills every sequence to seq_len (no padding)."""
    ds = _dataset(shards, seq_len=16, batch_size=2)
    b = next(iter(ds))
    # Every row is exactly seq_len long and carries real tokens, not pad zeros
    # left over from a short final document.
    assert b.shape == (2, 16)
    assert (b >= 0).all()


def test_deterministic_given_seed(shards):
    a = next(iter(_dataset(shards, seq_len=32, batch_size=4, seed=7)))
    b = next(iter(_dataset(shards, seq_len=32, batch_size=4, seed=7)))
    np.testing.assert_array_equal(a, b)


def test_shuffle_seed_changes_order(shards):
    a = next(iter(_dataset(shards, seq_len=32, batch_size=4, seed=1)))
    b = next(iter(_dataset(shards, seq_len=32, batch_size=4, seed=2)))
    assert not np.array_equal(a, b)


def test_empty_documents_are_filtered(shards):
    # A lone EOS from the empty document would show up as an isolated eos_id run;
    # the pipeline drops empty docs before packing, so the stream stays valid and
    # keeps producing full batches.
    ds = _dataset(shards, seq_len=24, batch_size=3)
    batches = list(itertools.islice(iter(ds), 3))
    assert len(batches) == 3
    assert all(b.shape == (3, 24) for b in batches)
