"""Data loading: Ultra-FineWeb-L3 streaming, tokenization, packing to seq_len."""
from __future__ import annotations

import itertools
from typing import Iterator, Optional

import jax
import jax.numpy as jnp
import numpy as np

from config import Config


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


def stream_dataset(config: Config) -> Iterator[str]:
    """Stream text from Ultra-FineWeb-L3 (or fall back to synthetic data)."""
    try:
        from datasets import load_dataset
        # Ultra-FineWeb-L3 has no default config; a name must be given.
        ds = load_dataset(config.dataset, getattr(config, "dataset_config", None),
                          split="train", streaming=True)
        for ex in ds:
            # Ultra-FineWeb-L3 has a 'text' field
            text = ex.get("text", "")
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


def pack_sequences(token_iter: Iterator[list[int]], seq_len: int, eos_id: int) -> Iterator[jnp.ndarray]:
    """Pack token lists into fixed-length sequences with EOS between docs."""
    buf = []
    for tokens in token_iter:
        buf.extend(tokens)
        buf.append(eos_id)
        while len(buf) >= seq_len:
            chunk = buf[:seq_len]
            buf = buf[seq_len:]
            yield jnp.array(chunk, dtype=jnp.int32)


def make_batches(config: Config, batch_size: int, seq_len: int, tokenizer=None) -> Iterator[jnp.ndarray]:
    """Yield batches of shape [batch_size, seq_len]."""
    if tokenizer is None:
        tokenizer = get_tokenizer(config)
    eos = getattr(tokenizer, "eos_id", None)
    if eos is None:
        eos = getattr(tokenizer, "eos_token_id", None)  # HF tokenizers
    if eos is None:
        eos = config.vocab_size - 1

    def token_iter():
        for text in stream_dataset(config):
            yield tokenizer.encode(text)

    packed = pack_sequences(token_iter(), seq_len, eos)
    while True:
        batch = list(itertools.islice(packed, batch_size))
        if len(batch) < batch_size:
            break
        yield jnp.stack(batch)


def random_batches(config: Config, batch_size: int, seq_len: int, seed: int = 0) -> Iterator[jnp.ndarray]:
    """Yield random token batches (for smoke test / no-data mode)."""
    rng = np.random.default_rng(seed)
    while True:
        yield jnp.array(rng.integers(0, config.vocab_size, size=(batch_size, seq_len)), dtype=jnp.int32)