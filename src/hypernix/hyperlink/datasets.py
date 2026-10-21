"""hyperlink.datasets — parts of a dataset file, as text a model reads.

A person attaches a dataset to a HyperLink chat (0.72.8a3) — JSONL,
JSON, Parquet, CSV/TSV, Arrow, or plain text and Markdown — and chooses
which part goes to the model: a range of rows, some columns, the rows
that match a search, or a random sample. :func:`excerpt` reads exactly
that part and writes it as compact text with a header saying what it is
(and how much of the file it is not), so the model never mistakes a
sample for the whole.

Reading is streaming where the format allows (JSONL, CSV and text are
read line by line; Parquet by row group), and every excerpt is capped in
characters, so a 20 GB file costs what the excerpt costs.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import random
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["DATASET_SUFFIXES", "DatasetError", "Excerpt", "Selection", "excerpt", "format_of", "is_dataset"]

#: What counts as a dataset, by suffix (``.gz`` is looked through).
DATASET_SUFFIXES = {
    ".jsonl": "jsonl", ".ndjson": "jsonl", ".json": "json", ".parquet": "parquet", ".pq": "parquet",
    ".csv": "csv", ".tsv": "tsv", ".arrow": "arrow", ".feather": "arrow",
    ".txt": "text", ".md": "markdown", ".markdown": "markdown", ".rst": "text", ".log": "text",
}
DEFAULT_ROWS = 20
MAX_CHARS = 12_000
MAX_CELL = 1_500


class DatasetError(ValueError):
    """A file or a selection that cannot be read, and why."""


@dataclass
class Selection:
    """Which part of the file. Everything optional: the default is the first rows."""

    rows: str = ""              # "0-20", "100-", "5", "-10" (the last 10)
    columns: list[str] = field(default_factory=list)
    query: str = ""             # rows/lines containing this (case-insensitive); "re:..." is a regex
    sample: int = 0             # a random sample of this many rows instead of a range
    seed: int = 0
    style: str = "auto"         # auto | jsonl | table | text
    max_chars: int = MAX_CHARS

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Selection:
        data = dict(data or {})
        if "as" in data and "style" not in data:
            data["style"] = data.pop("as")
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def window(self) -> tuple[int | None, int | None]:
        """``(start, stop)``; a negative start means "the last N"."""
        text = str(self.rows or "").replace(" ", "")
        if not text:
            return 0, DEFAULT_ROWS
        if re.fullmatch(r"\d+", text):
            return int(text), int(text) + 1
        if re.fullmatch(r"-\d+", text):
            return -int(text[1:]), None
        m = re.fullmatch(r"(\d*)[-:](\d*)", text)
        if not m:
            raise DatasetError(f"rows is like 0-20, 100-, 5 or -10, not {self.rows!r}")
        start = int(m.group(1) or 0)
        stop = int(m.group(2)) if m.group(2) else None
        if stop is not None and stop < start:
            raise DatasetError(f"rows {self.rows!r} ends before it starts")
        return start, stop


@dataclass
class Excerpt:
    name: str
    format: str
    text: str
    shown: int
    total: int | None
    columns: list[str]
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "format": self.format, "text": self.text, "rows_shown": self.shown,
                "rows_total": self.total, "columns": self.columns, "truncated": self.truncated}


def format_of(name: str) -> str:
    lowered = name.lower()
    if lowered.endswith(".gz"):
        lowered = lowered[:-3]
    for suffix, kind in DATASET_SUFFIXES.items():
        if lowered.endswith(suffix):
            return kind
    return ""


def is_dataset(name: str) -> bool:
    return format_of(name) in ("jsonl", "json", "parquet", "csv", "tsv", "arrow")


# ---------------------------------------------------------------------------
# Reading rows
# ---------------------------------------------------------------------------

def _open_text(path: Path) -> io.TextIOBase:
    if path.name.lower().endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with _open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                value = {"_unparsed": line}
            yield value if isinstance(value, dict) else {"value": value}


def _json(path: Path) -> Iterator[dict[str, Any]]:
    with _open_text(path) as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        for key in ("data", "rows", "records", "items", "train", "examples"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]
    if not isinstance(data, list):
        raise DatasetError("a JSON dataset is a list of records (or {\"data\": [...]})")
    for item in data:
        yield item if isinstance(item, dict) else {"value": item}


def _csv(path: Path, delimiter: str) -> Iterator[dict[str, Any]]:
    with _open_text(path) as fh:
        yield from csv.DictReader(fh, delimiter=delimiter)


def _parquet(path: Path, columns: list[str]) -> tuple[Iterator[dict[str, Any]], int, list[str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DatasetError("reading Parquet needs pyarrow: pip install pyarrow") from exc
    reader = pq.ParquetFile(path)
    names = reader.schema_arrow.names
    wanted = [c for c in columns if c in names] or None

    def rows() -> Iterator[dict[str, Any]]:
        for batch in reader.iter_batches(batch_size=512, columns=wanted):
            yield from batch.to_pylist()

    return rows(), reader.metadata.num_rows, names


def _arrow(path: Path) -> tuple[Iterator[dict[str, Any]], int, list[str]]:
    try:
        import pyarrow.feather as feather
    except ImportError as exc:
        raise DatasetError("reading Arrow needs pyarrow: pip install pyarrow") from exc
    table = feather.read_table(path)
    return iter(table.to_pylist()), table.num_rows, table.column_names


def _records(path: Path, kind: str, columns: list[str]) -> tuple[Iterable[dict[str, Any]], int | None, list[str]]:
    if kind == "parquet":
        return _parquet(path, columns)
    if kind == "arrow":
        return _arrow(path)
    if kind == "jsonl":
        return _jsonl(path), None, []
    if kind == "json":
        rows = list(_json(path))
        return rows, len(rows), []
    if kind in ("csv", "tsv"):
        return _csv(path, "\t" if kind == "tsv" else ","), None, []
    raise DatasetError(f"{path.name} is not a dataset format this reads")


def _matches(query: str) -> Any:
    if not query:
        return None
    if query.startswith("re:"):
        try:
            pattern = re.compile(query[3:], re.IGNORECASE)
        except re.error as exc:
            raise DatasetError(f"the search is not a valid regular expression: {exc}") from exc
        return pattern.search
    lowered = query.lower()
    return lambda text: lowered in text.lower()


def _cell(value: Any) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(value)
    return text if len(text) <= MAX_CELL else text[:MAX_CELL] + f"… [{len(text) - MAX_CELL} more characters]"


def _select(rows: Iterable[Any], selection: Selection, matcher: Any, text_of: Any) -> tuple[list[tuple[int, Any]], int | None]:
    """The chosen ``(index, row)`` pairs, and the count read (all of them, when the whole file was read)."""
    start, stop = selection.window()
    if selection.sample > 0:
        rng = random.Random(selection.seed)
        chosen: list[tuple[int, Any]] = []
        seen = 0
        for index, row in enumerate(rows):
            if matcher and not matcher(text_of(row)):
                continue
            seen += 1
            if len(chosen) < selection.sample:
                chosen.append((index, row))
            else:
                j = rng.randrange(seen)
                if j < selection.sample:
                    chosen[j] = (index, row)
        return sorted(chosen, key=lambda pair: pair[0]), None
    if start is not None and start < 0:              # the last N
        tail: list[tuple[int, Any]] = []
        count = 0
        for index, row in enumerate(rows):
            count += 1
            if matcher and not matcher(text_of(row)):
                continue
            tail.append((index, row))
            if len(tail) > -start:
                tail.pop(0)
        return tail, count
    out = []
    position = 0
    for index, row in enumerate(rows):
        if matcher and not matcher(text_of(row)):
            continue
        if position >= start and (stop is None or position < stop):
            out.append((index, row))
        position += 1
        if stop is not None and position >= stop:
            break
    return out, None


def _lines(path: Path) -> Iterator[str]:
    with _open_text(path) as fh:
        for line in fh:
            yield line.rstrip("\n")


# ---------------------------------------------------------------------------
# Writing the excerpt
# ---------------------------------------------------------------------------

def _render_rows(rows: list[tuple[int, dict[str, Any]]], columns: list[str], style: str) -> str:
    if style == "auto":
        style = "table" if len(columns) <= 4 and all(len(_cell(r.get(c, ""))) <= 80 for _, r in rows for c in columns) else "jsonl"
    if style == "table":
        def esc(text: str) -> str:
            return text.replace("|", "\\|").replace("\n", " ")
        head = "| # | " + " | ".join(esc(c) for c in columns) + " |\n|---|" + "---|" * len(columns)
        body = "\n".join(f"| {i} | " + " | ".join(esc(_cell(r.get(c, ""))) for c in columns) + " |" for i, r in rows)
        return head + ("\n" + body if body else "")
    if style == "text":
        return "\n\n".join(f"[{i}]\n" + "\n".join(f"{c}: {_cell(r.get(c, ''))}" for c in columns) for i, r in rows)
    return "\n".join(json.dumps({"_row": i, **{c: r.get(c) for c in columns}}, ensure_ascii=False, default=str)
                     for i, r in rows)


def _cap(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    cut = text.rfind("\n", 0, limit)
    return text[:cut if cut > limit // 2 else limit], True


def excerpt(path: str | Path, selection: Selection | dict[str, Any] | None = None, *, name: str = "") -> Excerpt:
    """The chosen part of *path*, as text for the model."""
    path = Path(path)
    name = name or path.name
    selection = selection if isinstance(selection, Selection) else Selection.from_dict(selection)
    limit = max(500, min(int(selection.max_chars or MAX_CHARS), 100_000))
    kind = format_of(name) or format_of(path.name)
    if not kind:
        raise DatasetError(f"{name}: not a format this reads ({', '.join(sorted(set(DATASET_SUFFIXES.values())))})")
    matcher = _matches(selection.query)

    if kind in ("text", "markdown"):
        if kind == "markdown" and selection.query and not selection.query.startswith("re:") and not selection.rows:
            return _markdown_section(path, name, selection.query, limit)
        lines = list(_lines(path)) if selection.rows.startswith("-") else _lines(path)
        chosen, total = _select(lines, selection, matcher, lambda line: line)
        body = "\n".join(line for _, line in chosen)
        text, truncated = _cap(body, limit)
        span = f"lines {chosen[0][0] + 1}-{chosen[-1][0] + 1}" if chosen else "no lines"
        what = "matching " + repr(selection.query) + ", " if selection.query else ""
        head = f"--- {name} ({kind}; {what}{span}{f' of {total}' if total else ''}) ---"
        return Excerpt(name, kind, f"{head}\n{text}" + ("\n[cut: the excerpt reached its size limit]" if truncated else ""),
                       len(chosen), total, [], truncated)

    rows, total, names = _records(path, kind, selection.columns)
    rows_list: list[tuple[int, dict[str, Any]]]

    def text_of(row: dict[str, Any]) -> str:
        return " ".join(_cell(v) for v in row.values())

    rows_list, counted = _select(rows, selection, matcher, text_of)
    total = total if total is not None else counted
    seen = names or list(dict.fromkeys(k for _, r in rows_list for k in r))
    columns = [c for c in selection.columns if c in seen] if selection.columns else seen
    missing = [c for c in selection.columns if c not in seen]
    if selection.columns and not columns:
        raise DatasetError(f"{name} has no column {', '.join(selection.columns)}; it has {', '.join(seen[:30])}")
    body = _render_rows(rows_list, columns, selection.style) if rows_list else "(no rows match)"
    text, truncated = _cap(body, limit)
    if rows_list:
        span = (f"a random sample of {len(rows_list)} rows" if selection.sample else
                f"rows {rows_list[0][0]}-{rows_list[-1][0]}")
    else:
        span = "no rows"
    what = f"matching {selection.query!r}, " if selection.query else ""
    of = f" of {total:,}" if total else ""
    note = f"; no column {', '.join(missing)}" if missing else ""
    head = f"--- {name} ({kind}; {what}{span}{of}; columns: {', '.join(columns)}{note}) ---"
    tail = "\n[cut: the excerpt reached its size limit; ask for fewer rows or columns]" if truncated else ""
    return Excerpt(name, kind, f"{head}\n{text}{tail}", len(rows_list), total, columns, truncated)


def _markdown_section(path: Path, name: str, heading: str, limit: int) -> Excerpt:
    """A Markdown file's section under the heading containing *heading*."""
    lines = list(_lines(path))
    wanted = heading.lower().lstrip("# ").strip()
    start, level = None, 0
    for index, line in enumerate(lines):
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if not m:
            continue
        if start is None and wanted in m.group(2).lower():
            start, level = index, len(m.group(1))
        elif start is not None and len(m.group(1)) <= level:
            body = "\n".join(lines[start:index])
            break
    else:
        body = "\n".join(lines[start:]) if start is not None else ""
    if start is None:
        raise DatasetError(f"{name} has no heading containing {heading!r}")
    text, truncated = _cap(body, limit)
    return Excerpt(name, "markdown", f"--- {name} (markdown; the section {lines[start].strip()!r}) ---\n{text}",
                   text.count("\n") + 1, len(lines), [], truncated)
