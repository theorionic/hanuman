"""Data loading: Ultra-FineWeb-L3 streaming, tokenization, packing to seq_len.

Everything up to the final `device_put` stays in numpy on the host. Batches are
built on a background thread so tokenization overlaps the TPU step instead of
blocking it -- a synchronous pipeline left the accelerator idle for minutes
before the first step of a real-data run.
"""
from __future__ import annotations

import itertools
import os
import queue
import threading
from typing import Iterator, Optional

import numpy as np

from config import Config

# The fast (Rust) tokenizers parallelize across a batch of documents; without
# this they warn and fall back to one thread when used from a worker thread.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

_DOC_CHUNK = 256   # documents handed to the tokenizer at once
_PREFETCH = 8      # batches kept ready ahead of the training loop


class ByteTokenizer:
    """Trivial byte-level tokenizer for the smoke test (vocab=256).

    Maps each character/byte to its byte value (0-255). EOS = 255 unused-ish;
    we use a dedicated EOS id = 256 if vocab>256, else 255.
    """

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.eos_id = vocab_size - 1

    def encode(self, text: str) -> list[int]:
        return [b for b in text.encode("utf-8")][: self.vocab_size - 1]

    def decode(self, ids: list[int]) -> str:
        return bytes([i for i in ids if 0 <= i < 256]).decode("utf-8", errors="replace")


def get_tokenizer(config: Config):
    if config.tokenizer == "byte":
        return ByteTokenizer(vocab_size=config.vocab_size)
    elif config.tokenizer == "llama3":
        # meta-llama is a gated repo; without HF credentials it 401s. Fall back
        # to an ungated mirror of the 32000-entry Llama tokenizer rather than to
        # bytes -- a byte tokenizer under a 32000 vocab leaves 99% of the
        # embedding table permanently untrained, which silently wrecks the run.
        from transformers import AutoTokenizer
        for name in ("meta-llama/Llama-3.2-1B", "hf-internal-testing/llama-tokenizer"):
            try:
                tok = AutoTokenizer.from_pretrained(name)
                print(f"[data] Tokenizer: {name} (vocab {tok.vocab_size})")
                return tok
            except Exception as e:
                print(f"[data] Could not load {name} ({str(e)[:80]})")
        print("[data] Falling back to byte tokenizer")
        return ByteTokenizer(vocab_size=config.vocab_size)
    else:
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(config.tokenizer)
        except Exception as e:
            print(f"[data] Could not load tokenizer {config.tokenizer!r} ({e}); falling back to byte")
            return ByteTokenizer(vocab_size=config.vocab_size)


def encode_documents(tokenizer, texts: list[str]) -> list[list[int]]:
    """Tokenize many documents in one call.

    HF fast tokenizers release the GIL and parallelize over the batch, so one
    batched call is several times faster than a Python loop over `encode`.
    """
    if isinstance(tokenizer, ByteTokenizer):
        return [tokenizer.encode(t) for t in texts]
    return tokenizer(texts, add_special_tokens=False)["input_ids"]


# Field holding the document body, most specific first. Ultra-FineWeb-L3 uses
# 'content'; most other corpora use 'text'. Guessing wrong is silent and fatal:
# every record reads as empty, the packer never fills a sequence, and the loop
# spins forever without producing a batch.
_TEXT_FIELDS = ("content", "text", "raw_content", "document")


def _open_stream(config: Config):
    """Open the dataset as a streaming iterable.

    Prefers an explicit parquet glob. Resolving a named config on a repo this
    size means listing and matching all 1771 files through the datasets config
    machinery, which takes minutes; pointing at the parquet files directly takes
    ~2 s for the same data.
    """
    from datasets import load_dataset

    files = getattr(config, "dataset_files", None)
    if files:
        return load_dataset("parquet", data_files=files, split="train", streaming=True)
    return load_dataset(config.dataset, getattr(config, "dataset_config", None),
                        split="train", streaming=True)


def stream_dataset(config: Config) -> Iterator[str]:
    """Stream text from Ultra-FineWeb-L3 (or fall back to synthetic data)."""
    try:
        ds = _open_stream(config)
        field = None
        for ex in ds:
            if field is None:
                field = next((f for f in _TEXT_FIELDS if ex.get(f)), None)
                if field is None:
                    raise KeyError(f"no text field in record; keys={list(ex)}")
                print(f"[data] Streaming field {field!r}")
            text = ex.get(field, "")
            if text:
                yield text
    except Exception as e:
        print(f"[data] Could not stream {config.dataset} ({e}); using synthetic text")
        # Synthetic fallback: repeating lorem-ish text
        base = ("The quick brown fox jumps over the lazy dog. "
                "In a world of tokens and transformers, the model learns to predict "
                "the next word from context. Mixture of experts routes tokens. ")
        i = 0
        while True:
            yield base * (1 + (i % 3))
            i += 1


def pack_sequences(token_iter: Iterator[list[int]], seq_len: int, eos_id: int) -> Iterator[np.ndarray]:
    """Pack token lists into fixed-length sequences with EOS between docs."""
    buf: list[int] = []
    for tokens in token_iter:
        buf.extend(tokens)
        buf.append(eos_id)
        while len(buf) >= seq_len:
            chunk = buf[:seq_len]
            del buf[:seq_len]
            yield np.asarray(chunk, dtype=np.int32)


def prefetch(it: Iterator, depth: int = _PREFETCH) -> Iterator:
    """Run `it` on a background thread, buffering up to `depth` items.

    Tokenization is host work with nothing to do with the accelerator, so
    overlapping it with the step is free throughput. The queue is bounded, so a
    fast consumer applies backpressure rather than letting the producer run away
    with host memory.
    """
    q: queue.Queue = queue.Queue(maxsize=depth)
    sentinel = object()

    def worker():
        try:
            for item in it:
                q.put(item)
        except Exception as e:                # surface producer errors downstream
            q.put(e)
        finally:
            q.put(sentinel)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is sentinel:
            return
        if isinstance(item, Exception):
            raise item
        yield item


def make_batches(config: Config, batch_size: int, seq_len: int, tokenizer=None) -> Iterator[np.ndarray]:
    """Yield batches of shape [batch_size, seq_len] as host numpy arrays."""
    if tokenizer is None:
        tokenizer = get_tokenizer(config)
    eos = getattr(tokenizer, "eos_id", None)
    if eos is None:
        eos = getattr(tokenizer, "eos_token_id", None)  # HF tokenizers
    if eos is None:
        eos = config.vocab_size - 1

    def token_iter():
        docs = stream_dataset(config)
        while True:
            chunk = list(itertools.islice(docs, _DOC_CHUNK))
            if not chunk:
                return
            yield from encode_documents(tokenizer, chunk)

    def batches():
        packed = pack_sequences(token_iter(), seq_len, eos)
        while True:
            batch = list(itertools.islice(packed, batch_size))
            if len(batch) < batch_size:
                return
            yield np.stack(batch)

    return prefetch(batches())


def random_batches(config: Config, batch_size: int, seq_len: int, seed: int = 0) -> Iterator[np.ndarray]:
    """Yield random token batches (for smoke test / no-data mode)."""
    rng = np.random.default_rng(seed)
    while True:
        yield rng.integers(0, config.vocab_size, size=(batch_size, seq_len), dtype=np.int32)
