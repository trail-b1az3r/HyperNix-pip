"""The model list opens every GGUF on disk each time it is asked, and the
runner's picker asks every couple of seconds -- so reading a file must
be quick, and reading an unchanged one again must not read it at all."""
from __future__ import annotations

import os

from hypernix.quant.gguf import GGUFFile, GGUFValueType, GGUFWriter
from hypernix.t1api import modelindex

ARRAY = int(GGUFValueType.ARRAY)


def _write(path, *, context=4096, vocab=64):
    w = GGUFWriter(path)
    w.set_metadata("general.architecture", "qwen3")
    w.set_metadata("qwen3.context_length", context)
    w.set_metadata("tokenizer.ggml.tokens", [f"tok{i}" for i in range(vocab)])
    w.set_metadata("tokenizer.ggml.token_type", list(range(vocab)),
                   type_hint=(ARRAY, int(GGUFValueType.INT32)))
    w.set_metadata("tokenizer.ggml.scores", [i / 2 for i in range(vocab)],
                   type_hint=(ARRAY, int(GGUFValueType.FLOAT32)))
    w.set_metadata("x.flags", [True, False, True], type_hint=(ARRAY, int(GGUFValueType.BOOL)))
    w.add_tensor("token_embd.weight", (vocab, 8), 0)
    w.write(lambda t: b"\0" * t.nbytes)


def test_arrays_read_in_bulk_are_the_arrays_written(tmp_path):
    path = tmp_path / "m.gguf"
    _write(path, vocab=300)
    meta = GGUFFile.read(path).metadata
    assert meta["tokenizer.ggml.tokens"] == [f"tok{i}" for i in range(300)]
    assert meta["tokenizer.ggml.token_type"] == list(range(300))
    assert meta["tokenizer.ggml.scores"] == [i / 2 for i in range(300)]
    assert meta["x.flags"] == [True, False, True]


def test_an_unchanged_file_is_not_read_again(tmp_path, monkeypatch):
    path = tmp_path / "m.gguf"
    _write(path)
    first = modelindex.inspect(path)
    assert first.context_limit == 4096 and first.vocab_size == 64

    reads = []
    real = modelindex._read_gguf
    monkeypatch.setattr(modelindex, "_read_gguf", lambda p: reads.append(p) or real(p))
    again = modelindex.inspect(path)
    assert reads == []
    assert again == first
    again.assumed.append("changed by a caller")
    assert "changed by a caller" not in modelindex.inspect(path).assumed


def test_a_changed_file_is_read_again(tmp_path):
    path = tmp_path / "m.gguf"
    _write(path, context=4096)
    assert modelindex.inspect(path).context_limit == 4096
    _write(path, context=8192, vocab=80)
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert modelindex.inspect(path).context_limit == 8192
