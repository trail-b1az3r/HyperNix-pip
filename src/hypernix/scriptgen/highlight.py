"""scriptgen.highlight — colour spans for the live preview (0.72.8rc1).

No GUI toolkit here: :func:`spans` returns ``(tag, (line, col), (line,
col))`` triples, Tk's own text indices, and the GUI applies them. Python
goes through :mod:`tokenize`, so a string spanning lines or a ``#``
inside a string is coloured as what it is; the config formats (JSON,
YAML, TOML, env, flags) are simple enough for a few patterns.
"""
from __future__ import annotations

import io
import keyword
import re
import tokenize

__all__ = ["spans", "TAGS"]

#: Every tag :func:`spans` can return.
TAGS = ("keyword", "string", "number", "comment", "name", "builtin", "marker")

_BUILTINS = frozenset({
    "print", "len", "range", "dict", "list", "tuple", "set", "int", "float", "str", "bool",
    "isinstance", "getattr", "setattr", "hasattr", "min", "max", "sum", "sorted", "open",
    "enumerate", "zip", "type", "vars", "next", "any", "all", "round", "super", "self",
})
_SOFT = frozenset({"True", "False", "None"})

Span = tuple[str, tuple[int, int], tuple[int, int]]


def spans(text: str, fmt: str = "python") -> list[Span]:
    """Colour spans for *text*: Python source, or a config in *fmt*."""
    if fmt == "python":
        return _python(text)
    return _config(text)


def _python(text: str) -> list[Span]:
    out: list[Span] = []
    previous = ""
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            kind, value = token.type, token.string
            tag = ""
            if kind == tokenize.COMMENT:
                tag = "marker" if value.startswith("# --- hypernix scriptgen") else "comment"
            elif kind == tokenize.STRING and value[:3] in ('"""', "'''") and previous in ("", ":"):
                tag = "comment"  # a docstring reads as prose, not as data
            elif kind == tokenize.STRING or kind == getattr(tokenize, "FSTRING_START", -1):
                tag = "string"
            elif kind in (getattr(tokenize, "FSTRING_MIDDLE", -1), getattr(tokenize, "FSTRING_END", -1)):
                tag = "string"
            elif kind == tokenize.NUMBER:
                tag = "number"
            elif kind == tokenize.NAME:
                if previous in ("def", "class"):
                    tag = "name"
                elif value in _SOFT:
                    tag = "number"
                elif keyword.iskeyword(value):
                    tag = "keyword"
                elif value in _BUILTINS:
                    tag = "builtin"
            if tag:
                out.append((tag, token.start, token.end))
            if kind not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
                previous = value
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Half-typed code in an injected script: what was coloured stands.
        pass
    return out


_KEY = re.compile(r'^(\s*)("?[A-Za-z_][\w.\-]*"?)(\s*[:=])')
_FLAG = re.compile(r"(--[\w\-]+)")
_STRING = re.compile(r'"(?:[^"\\]|\\.)*"|\'[^\']*\'')
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?![\w.])")
_WORD = re.compile(r"\b(true|false|null|True|False|None)\b")


def _config(text: str) -> list[Span]:
    out: list[Span] = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            out.append(("comment", (number, 0), (number, len(line))))
            continue
        found: list[tuple[str, int, int]] = []
        key = _KEY.match(line)
        if key:
            found.append(("name", key.start(2), key.end(2)))
        found += [("name", m.start(), m.end()) for m in _FLAG.finditer(line)]
        found += [("string", m.start(), m.end()) for m in _STRING.finditer(line)]
        found += [("number", m.start(), m.end()) for m in _NUMBER.finditer(line)]
        found += [("keyword", m.start(), m.end()) for m in _WORD.finditer(line)]
        # First come, first served: a number inside a string stays a string.
        taken: list[tuple[int, int]] = []
        for tag, start, end in found:
            if any(start < b and end > a for a, b in taken):
                continue
            taken.append((start, end))
            out.append((tag, (number, start), (number, end)))
    return out
