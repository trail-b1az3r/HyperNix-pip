"""strife.tokenizer — the tokenizer stored in the GGUF, nothing downloaded.

Two families, which between them cover the architectures Strife runs:

* **gpt2** -- byte-level BPE (Qwen, Llama 3, Mistral Nemo): text is split
  by the pre-tokenizer regex named in ``tokenizer.ggml.pre``, each piece
  is mapped byte-to-unicode, and merges apply in rank order.
* **llama** -- SentencePiece BPE (Llama 2, Mistral 7B): spaces become
  ``▁``, adjacent symbols merge by highest score, and anything left
  outside the vocabulary falls back to ``<0xNN>`` byte tokens.

Control and user-defined tokens (``<|im_start|>`` and the like) are
matched literally first when ``special=True``, so a chat template's
markers become single tokens instead of being spelled out.

:class:`StreamDecoder` turns token ids into text as they arrive without
splitting a multi-byte character across two chunks.
"""
from __future__ import annotations

import heapq
import re
from functools import lru_cache
from typing import Any

from .errors import StrifeError

__all__ = ["Tokenizer", "StreamDecoder"]

TOKEN_NORMAL, TOKEN_UNKNOWN, TOKEN_CONTROL, TOKEN_USER, TOKEN_UNUSED, TOKEN_BYTE = 1, 2, 3, 4, 5, 6

# The pre-tokenizer patterns llama.cpp uses, by ``tokenizer.ggml.pre``.
_GPT2 = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_LLAMA3 = (r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|"""
           r""" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""")
_QWEN2 = (r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"""
          r""" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""")
_PRE = {"default": _GPT2, "gpt2": _GPT2, "llama-bpe": _LLAMA3, "llama3": _LLAMA3,
        "qwen2": _QWEN2, "deepseek-llm": _QWEN2, "smaug-bpe": _LLAMA3}


_LETTER, _NUMBER = "a-zA-Z\u00aa\u00b5\u00ba\u00c0-\u00d6\u00d8-\u00f6\u00f8-\uffff", "0-9"


def _compile(pattern: str) -> Any:
    try:
        import regex

        return regex.compile(pattern)
    except ImportError:
        pass
    # Without the regex module: \p{L} and \p{N} become explicit ranges,
    # inside a character class as they are, outside one in brackets.
    def inside(match: re.Match[str]) -> str:
        return match.group(0).replace(r"\p{L}", _LETTER).replace(r"\p{N}", _NUMBER)

    translated = re.sub(r"\[[^\]]*\]", inside, pattern)
    translated = translated.replace(r"\p{L}", f"[{_LETTER}]").replace(r"\p{N}", f"[{_NUMBER}]")
    return re.compile(translated)


@lru_cache(maxsize=1)
def _byte_to_unicode() -> dict[int, str]:
    keep = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1)))
    mapping, extra = {}, 0
    for b in range(256):
        if b in keep:
            mapping[b] = chr(b)
        else:
            mapping[b] = chr(256 + extra)
            extra += 1
    return mapping


class Tokenizer:
    def __init__(self, metadata: dict[str, Any]) -> None:
        self.model = str(metadata.get("tokenizer.ggml.model") or "")
        tokens = metadata.get("tokenizer.ggml.tokens")
        if self.model not in ("gpt2", "llama") or not isinstance(tokens, list):
            raise StrifeError("strife.tokenizer.unsupported",
                              f"tokenizer {self.model!r} is not one Strife reads")
        self.tokens: list[str] = [str(t) for t in tokens]
        self.vocab: dict[str, int] = {}
        for i, tok in enumerate(self.tokens):
            self.vocab.setdefault(tok, i)
        types = metadata.get("tokenizer.ggml.token_type") or [TOKEN_NORMAL] * len(self.tokens)
        self.types: list[int] = [int(t) for t in types][: len(self.tokens)]
        self.types += [TOKEN_NORMAL] * (len(self.tokens) - len(self.types))
        self.scores: list[float] = [float(s) for s in (metadata.get("tokenizer.ggml.scores") or [])]
        self.bos_id = _opt_int(metadata.get("tokenizer.ggml.bos_token_id"))
        self.eos_id = _opt_int(metadata.get("tokenizer.ggml.eos_token_id"))
        self.eot_id = _opt_int(metadata.get("tokenizer.ggml.eot_token_id"))
        self.unk_id = _opt_int(metadata.get("tokenizer.ggml.unknown_token_id"))
        default_bos = self.model == "llama"
        self.add_bos = bool(metadata.get("tokenizer.ggml.add_bos_token", default_bos))
        self.add_space_prefix = bool(metadata.get("tokenizer.ggml.add_space_prefix",
                                                  self.model == "llama"))
        self.chat_template = str(metadata.get("tokenizer.chat_template") or "")
        self.pre = str(metadata.get("tokenizer.ggml.pre") or "default")
        specials = [t for t, ty in zip(self.tokens, self.types, strict=False)
                    if ty in (TOKEN_CONTROL, TOKEN_USER) and t]
        self._special_re = (re.compile("|".join(re.escape(s) for s in
                                                 sorted(specials, key=len, reverse=True)))
                            if specials else None)
        self.stop_ids: set[int] = {i for i in (self.eos_id, self.eot_id) if i is not None}
        for marker in ("<|im_end|>", "<|eot_id|>", "<|endoftext|>", "<end_of_turn>"):
            if marker in self.vocab and self.types[self.vocab[marker]] == TOKEN_CONTROL:
                self.stop_ids.add(self.vocab[marker])
        if self.model == "gpt2":
            merges = metadata.get("tokenizer.ggml.merges") or []
            self.ranks: dict[tuple[str, str], int] = {}
            for rank, merge in enumerate(merges):
                left, _, right = str(merge).partition(" ")
                self.ranks.setdefault((left, right), rank)
            self._split = _compile(_PRE.get(self.pre, _GPT2))
            self._b2u = _byte_to_unicode()
            self._u2b = {v: k for k, v in self._b2u.items()}
        self._bpe_cache: dict[str, list[int]] = {}

    @property
    def n_vocab(self) -> int:
        return len(self.tokens)

    # -- encoding --------------------------------------------------------

    def encode(self, text: str, *, add_bos: bool | None = None, special: bool = True) -> list[int]:
        ids: list[int] = []
        if (self.add_bos if add_bos is None else add_bos) and self.bos_id is not None:
            ids.append(self.bos_id)
        pieces: list[tuple[str, bool]] = []
        if special and self._special_re is not None:
            last = 0
            for match in self._special_re.finditer(text):
                if match.start() > last:
                    pieces.append((text[last:match.start()], False))
                pieces.append((match.group(0), True))
                last = match.end()
            if last < len(text):
                pieces.append((text[last:], False))
        else:
            pieces = [(text, False)]
        first_text = True
        for piece, is_special in pieces:
            if is_special:
                ids.append(self.vocab[piece])
                continue
            if self.model == "gpt2":
                ids.extend(self._encode_gpt2(piece))
            else:
                ids.extend(self._encode_spm(piece, prefix=first_text and self.add_space_prefix))
            first_text = False
        return ids

    def _encode_gpt2(self, text: str) -> list[int]:
        out: list[int] = []
        for word in self._split.findall(text):
            cached = self._bpe_cache.get(word)
            if cached is None:
                cached = self._bpe(word)
                if len(self._bpe_cache) < 65536:
                    self._bpe_cache[word] = cached
            out.extend(cached)
        return out

    def _bpe(self, word: str) -> list[int]:
        symbols = [self._b2u[b] for b in word.encode("utf-8")]
        while len(symbols) > 1:
            best, best_rank = -1, None
            for i in range(len(symbols) - 1):
                rank = self.ranks.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best, best_rank = i, rank
            if best < 0:
                break
            symbols[best:best + 2] = [symbols[best] + symbols[best + 1]]
        out: list[int] = []
        for sym in symbols:
            tid = self.vocab.get(sym)
            if tid is None:  # a merge produced something outside the vocabulary
                out.extend(self.vocab[self._b2u[b]] for b in
                           bytes(self._u2b[c] for c in sym) if self._b2u[b] in self.vocab)
            else:
                out.append(tid)
        return out

    def _encode_spm(self, text: str, *, prefix: bool) -> list[int]:
        if not text:
            return []
        text = text.replace(" ", "▁")
        if prefix:
            text = "▁" + text
        symbols = list(text)
        prev = list(range(-1, len(symbols) - 1))
        nxt = list(range(1, len(symbols) + 1))
        nxt[-1] = -1
        alive = [True] * len(symbols)
        heap: list[tuple[float, int, int, str]] = []

        def push(i: int) -> None:
            j = nxt[i]
            if j < 0:
                return
            merged = symbols[i] + symbols[j]
            tid = self.vocab.get(merged)
            if tid is not None:
                score = self.scores[tid] if tid < len(self.scores) else 0.0
                heapq.heappush(heap, (-score, i, j, merged))

        for i in range(len(symbols) - 1):
            push(i)
        while heap:
            _neg, i, j, merged = heapq.heappop(heap)
            if not (alive[i] and alive[j]) or nxt[i] != j or symbols[i] + symbols[j] != merged:
                continue
            symbols[i] = merged
            alive[j] = False
            nxt[i] = nxt[j]
            if nxt[j] >= 0:
                prev[nxt[j]] = i
            if prev[i] >= 0:
                push(prev[i])
            push(i)
        out: list[int] = []
        i = 0
        while i >= 0 and i < len(symbols):
            if alive[i]:
                tid = self.vocab.get(symbols[i])
                if tid is not None:
                    out.append(tid)
                else:
                    for b in symbols[i].encode("utf-8"):
                        byte_tok = self.vocab.get(f"<0x{b:02X}>")
                        if byte_tok is not None:
                            out.append(byte_tok)
                        elif self.unk_id is not None:
                            out.append(self.unk_id)
            i = nxt[i] if alive[i] else i + 1
        return out

    # -- decoding --------------------------------------------------------

    def token_bytes(self, tid: int) -> bytes:
        if not 0 <= tid < len(self.tokens):
            return b""
        text, kind = self.tokens[tid], self.types[tid]
        if kind == TOKEN_CONTROL:
            return b""
        if self.model == "gpt2":
            if kind == TOKEN_USER:
                return text.encode("utf-8")
            try:
                return bytes(self._u2b[c] for c in text)
            except KeyError:
                return text.encode("utf-8")
        if kind == TOKEN_BYTE or (text.startswith("<0x") and text.endswith(">") and len(text) == 6):
            try:
                return bytes([int(text[3:5], 16)])
            except ValueError:
                return b""
        return text.replace("▁", " ").encode("utf-8")

    def decode(self, ids: list[int]) -> str:
        data = b"".join(self.token_bytes(i) for i in ids)
        text = data.decode("utf-8", errors="replace")
        if self.model == "llama" and self.add_space_prefix and text.startswith(" "):
            text = text[1:]  # the prefix encode added
        return text

    # -- chat ------------------------------------------------------------

    def apply_chat_template(self, messages: list[dict[str, str]], *,
                            add_generation_prompt: bool = True) -> str:
        """The model's own template (Jinja, sandboxed); ChatML without one."""
        if self.chat_template:
            # Jinja sandbox renders forbidden dunder lookups as Undefined in
            # some versions instead of raising. Reject the attempted escape
            # explicitly so a hostile template is never silently accepted.
            if re.search(r"\.\s*__[A-Za-z0-9_]+__|\b__[A-Za-z0-9_]+__\s*\(", self.chat_template):
                raise StrifeError(
                    "strife.chat.template",
                    "chat template: access to Python dunder attributes is forbidden",
                )
            try:
                from jinja2.sandbox import ImmutableSandboxedEnvironment

                env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

                def raise_exception(message: str) -> None:
                    raise StrifeError("strife.chat.template", f"chat template: {message}")

                env.globals["raise_exception"] = raise_exception
                bos = self.tokens[self.bos_id] if self.bos_id is not None else ""
                eos = self.tokens[self.eos_id] if self.eos_id is not None else ""
                return env.from_string(self.chat_template).render(
                    messages=messages, add_generation_prompt=add_generation_prompt,
                    bos_token=bos, eos_token=eos)
            except ImportError:
                pass
            except StrifeError:
                raise
            except Exception as exc:  # a template this engine cannot render
                raise StrifeError("strife.chat.template",
                                  f"the model's chat template did not render: {exc}",
                                  remedy="Pass a plain prompt instead of messages.") from exc
        parts = [f"<|im_start|>{m.get('role', 'user')}\n{m.get('content', '')}<|im_end|>\n"
                 for m in messages]
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def templated_has_bos(self, text: str) -> bool:
        return self.bos_id is not None and text.startswith(self.tokens[self.bos_id])


def _opt_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


class StreamDecoder:
    """Token ids in, text out, never splitting a UTF-8 character.

    An incomplete character at the end is held until its last byte comes;
    bytes that can never form one are replaced with U+FFFD at once, so
    what streams is exactly what the whole decode would have produced.
    """

    def __init__(self, tokenizer: Tokenizer, *, strip_leading_space: bool = False) -> None:
        import codecs

        self.tokenizer = tokenizer
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._strip = strip_leading_space

    def push(self, tid: int) -> str:
        text = self._decoder.decode(self.tokenizer.token_bytes(tid))
        if self._strip and text:
            text = text[1:] if text.startswith(" ") else text
            self._strip = False
        return text

    def flush(self) -> str:
        return self._decoder.decode(b"", final=True)
