"""Grain streaming data pipeline for TPU pretraining.

Reads Ultra-FineWeb-L3 parquet shards straight off the HF hub (pyarrow+fsspec),
tokenizes and concat-then-split packs to a fixed ``seq_len``, batches, and
prefetches onto the TPU -- every stage on a background thread, so the training
step consumes a batch already resident in HBM instead of blocking on the host.

Why Grain rather than the hand-rolled thread in data.py:
  - ``device_put`` runs the whole read -> tokenize -> pack -> batch pipeline on a
    CPU prefetch thread AND double-buffers finished batches on the device. A read
    of one parquet row group is a ~3 s network burst; the CPU buffer drains
    during it, so the accelerator never stalls.
  - ``ParquetIterDataset`` is checkpointable (row-group + row index), so the
    source can be resumed instead of re-streaming from the first file.
  - ``ConcatThenSplitIterDataset`` packs documents with no padding waste -- the
    same semantics as data.pack_sequences (EOS between docs, long docs split
    across sequences).

Deliberately single-process. Measured single-thread tokenization is ~0.75M
tok/s per document (batched: ~5M tok/s), 10-150x faster than a 7B step consumes
(32768 tok/step at a few steps/s). The bottleneck is network I/O, which the CPU
prefetch thread already hides; worker processes would add picklability
constraints and startup cost for no throughput.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence

import grain
import numpy as np
from grain.experimental import (
    ConcatThenSplitIterDataset,
    InterleaveIterDataset,
    ThreadPrefetchIterDataset,
    device_put,
)

from config import Config
from data import _TEXT_FIELDS, ByteTokenizer


def _list_files(config: Config) -> list[str]:
    """Resolve the parquet shard paths for the configured dataset.

    Prefers the explicit ``dataset_files`` glob (an ``hf://`` pattern); the
    fsspec HF filesystem expands it in ~0.5 s, versus minutes to resolve a named
    config against the repo's file table.
    """
    import fsspec

    glob = getattr(config, "dataset_files", None)
    if not glob:
        raise ValueError(
            "grain pipeline needs config.dataset_files (an hf:// parquet glob); "
            f"got {glob!r}")
    fs = fsspec.filesystem("hf")
    files = sorted(fs.glob(glob))
    if not files:
        raise ValueError(f"no parquet files matched {glob!r}")
    return files


class _OpenParquet:
    """Map a file path to a (lazy) ParquetIterDataset over that file."""

    def __init__(self, filesystem):
        self.fs = filesystem

    def __call__(self, path: str) -> grain.IterDataset:
        return grain.experimental.ParquetIterDataset(path, filesystem=self.fs)


class _Tokenize:
    """Record dict -> {'input_ids': int32[L]} with a trailing EOS.

    Empty documents yield a length-0 array and are dropped by the filter that
    follows, so no stray lone-EOS sequences leak into the packer.
    """

    def __init__(self, tokenizer, eos_id: int, text_fields: Sequence[str]):
        self.tok = tokenizer
        self.eos = int(eos_id)
        self.fields = tuple(text_fields)

    def __call__(self, record: dict) -> dict:
        text = next((record[f] for f in self.fields if record.get(f)), None)
        if not text:
            return {"input_ids": np.empty((0,), dtype=np.int32)}
        if isinstance(self.tok, ByteTokenizer):
            ids = self.tok.encode(text)
        else:
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
        ids = np.asarray(ids, dtype=np.int32)
        return {"input_ids": np.append(ids, np.int32(self.eos))}


def _eos_id(config: Config, tokenizer) -> int:
    eos = getattr(tokenizer, "eos_id", None)
    if eos is None:
        eos = getattr(tokenizer, "eos_token_id", None)  # HF tokenizers
    if eos is None:
        eos = config.vocab_size - 1
    return int(eos)


def build_grain_dataset(
    files: Sequence[str],
    tokenizer,
    seq_len: int,
    batch_size: int,
    eos_id: int,
    *,
    filesystem,
    sharding=None,
    interleave: int = 4,
    shuffle_files: bool = True,
    seed: int = 0,
    cpu_buffer: int = 8,
    device_buffer: int = 2,
    text_fields: Sequence[str] = _TEXT_FIELDS,
) -> grain.IterDataset:
    """Assemble the Grain pipeline.

    Rooting the pipeline in a ``MapDataset`` of file paths (shuffled, repeated)
    keeps the read source checkpointable and lets ``InterleaveIterDataset`` pull
    from ``interleave`` files concurrently, smoothing the per-row-group latency.

    ``sharding`` None yields host numpy batches (behind a CPU prefetch thread) --
    used by the tests and CPU dev. A jax sharding turns on ``device_put``, which
    prefetches the whole pipeline on a thread and double-buffers batches in HBM.
    """
    file_ds = grain.MapDataset.source(list(files))
    if shuffle_files:
        file_ds = file_ds.shuffle(seed=seed)
    file_ds = file_ds.repeat()                       # infinite epochs
    parquet_ds = file_ds.map(_OpenParquet(filesystem))

    ds = InterleaveIterDataset(parquet_ds, cycle_length=interleave)
    ds = ds.map(_Tokenize(tokenizer, eos_id, text_fields))
    ds = ds.filter(lambda r: r["input_ids"].size > 0)
    ds = ConcatThenSplitIterDataset(ds, length_struct={"input_ids": seq_len})
    ds = ds.map(lambda r: np.asarray(r["input_ids"], dtype=np.int32))
    ds = ds.batch(batch_size)

    if sharding is not None:
        ds = device_put(ds, sharding, cpu_buffer_size=cpu_buffer,
                        device_buffer_size=device_buffer)
    else:
        ds = ThreadPrefetchIterDataset(ds, prefetch_buffer_size=cpu_buffer)
    return ds


def make_grain_batches(
    config: Config,
    batch_size: int,
    seq_len: int,
    tokenizer,
    sharding=None,
) -> Iterator[np.ndarray]:
    """Yield packed batches for training.

    With ``sharding`` set each item is an on-device, correctly-sharded
    ``jax.Array`` of shape ``[batch_size, seq_len]``; the training loop consumes
    it without a further ``device_put``. Without it, host numpy of the same shape.
    """
    import fsspec

    files = _list_files(config)
    fs = fsspec.filesystem("hf")
    eos = _eos_id(config, tokenizer)
    ds = build_grain_dataset(
        files, tokenizer, seq_len, batch_size, eos,
        filesystem=fs,
        sharding=sharding,
        interleave=getattr(config, "grain_interleave", 4),
        shuffle_files=getattr(config, "grain_shuffle_files", True),
        seed=getattr(config, "grain_seed", 0),
        cpu_buffer=getattr(config, "grain_cpu_buffer", 8),
        device_buffer=getattr(config, "grain_device_buffer", 2),
    )
    return iter(ds)
