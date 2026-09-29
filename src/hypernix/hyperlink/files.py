"""hyperlink.files — the attachment store behind "send a photo" and "upload".

Content-addressed: a file's identity is the SHA-256 of its bytes. That
one decision buys most of the properties this needs.

* **Re-sending the same screenshot costs nothing.** The phone uploads,
  the hash matches, the store returns the existing id. Over cellular
  that is the difference between instant and forty seconds.
* **The id cannot be guessed or enumerated.** A 64-hex-character content
  hash is not a sequence number.
* **Nothing is ever overwritten.** Two files with the same hash *are*
  the same file; two different files cannot collide on one id.

Ownership is tracked separately from content, in the metadata table:
several devices can reference one blob, and revoking a device's access
does not delete bytes another device still points at. Deletion is
reference-counted for the same reason (:meth:`AttachmentStore.delete`).

Limits are enforced at write time, not by the web layer, because the
web layer is not the only writer — the CLI and the LM Studio bridge
both put files here — and a limit only one caller checks is not a limit.

The store is deliberately dumb about *content*. It sniffs enough of a
magic number to label a file as an image (which decides whether the chat
layer may send it to a vision model) and otherwise treats everything as
opaque bytes. It does not transcode, resize, or parse. A store that
parses attachments is a store that can be attacked with one.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..t1api.db import SQLiteBackend
from ..t1api.errors import T1APIError, T1ErrorCode

__all__ = [
    "Attachment",
    "AttachmentStore",
    "sniff_content_type",
    "DEFAULT_MAX_BYTES",
    "IMAGE_TYPES",
]

DEFAULT_MAX_BYTES = 64 * 1024 * 1024        # 64 MiB — a phone photo is ~5
DEFAULT_MAX_IMAGE_BYTES = 24 * 1024 * 1024

IMAGE_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/gif", "image/webp", "image/heic", "image/bmp"}
)

#: (offset, magic bytes, content type). Ordered most-specific first.
_MAGIC: tuple[tuple[int, bytes, str], ...] = (
    (0, b"\x89PNG\r\n\x1a\n", "image/png"),
    (0, b"\xff\xd8\xff", "image/jpeg"),
    (0, b"GIF87a", "image/gif"),
    (0, b"GIF89a", "image/gif"),
    (0, b"BM", "image/bmp"),
    (0, b"%PDF-", "application/pdf"),
    (0, b"PK\x03\x04", "application/zip"),
    (0, b"GGUF", "application/x-gguf"),
    (4, b"ftypheic", "image/heic"),
    (4, b"ftypheix", "image/heic"),
    (4, b"ftypmif1", "image/heic"),
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS hyperlink_files (
    file_id TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    filename TEXT NOT NULL,
    content_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    owner TEXT NOT NULL,
    device_id TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_hyperlink_files_owner ON hyperlink_files (owner);
CREATE INDEX IF NOT EXISTS idx_hyperlink_files_sha ON hyperlink_files (sha256);
CREATE INDEX IF NOT EXISTS idx_hyperlink_files_session ON hyperlink_files (session_id);
"""


def sniff_content_type(data: bytes, *, filename: str = "", declared: str = "") -> str:
    """Decide a content type from the bytes, then the name, then the claim.

    In that order, and the order is the point: a client can declare
    anything, and the declaration is what a naive store would use to
    decide "this is an image, send it to the vision model". Magic bytes
    are checked first so a ``.png`` that is really a zip is labelled as
    a zip.
    """
    for offset, magic, ctype in _MAGIC:
        if data[offset : offset + len(magic)] == magic:
            return ctype
    suffix = Path(filename).suffix.lower()
    by_suffix = {
        ".txt": "text/plain", ".md": "text/markdown", ".json": "application/json",
        ".py": "text/x-python", ".swift": "text/x-swift", ".js": "text/javascript",
        ".ts": "text/typescript", ".c": "text/x-c", ".h": "text/x-c",
        ".cpp": "text/x-c++", ".rs": "text/x-rust", ".go": "text/x-go",
        ".java": "text/x-java", ".sh": "text/x-shellscript", ".yml": "text/yaml",
        ".yaml": "text/yaml", ".toml": "text/toml", ".csv": "text/csv",
        ".html": "text/html", ".css": "text/css", ".sql": "text/x-sql",
        ".gguf": "application/x-gguf", ".pdf": "application/pdf",
    }
    if suffix in by_suffix:
        return by_suffix[suffix]
    if declared and "/" in declared:
        return declared.split(";", 1)[0].strip().lower()
    # Printable-ASCII-ish content with no magic number is almost always
    # source or notes, and treating it as text is what makes "upload
    # this file and explain it" work without a special case per language.
    sample = data[:2048]
    if sample and b"\x00" not in sample:
        try:
            sample.decode("utf-8")
            return "text/plain"
        except UnicodeDecodeError:
            pass
    return "application/octet-stream"


@dataclass
class Attachment:
    file_id: str
    sha256: str
    filename: str
    content_type: str
    size_bytes: int
    owner: str
    created_at: float
    device_id: str = ""
    session_id: str = ""
    metadata: dict[str, Any] | None = None

    @property
    def is_image(self) -> bool:
        return self.content_type in IMAGE_TYPES

    @property
    def is_text(self) -> bool:
        return self.content_type.startswith("text/") or self.content_type in (
            "application/json",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_id": self.file_id,
            "sha256": self.sha256,
            "filename": self.filename,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "owner": self.owner,
            "device_id": self.device_id,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "is_image": self.is_image,
            "is_text": self.is_text,
            "metadata": self.metadata or {},
        }


class AttachmentStore:
    """Blobs on disk, metadata in SQL."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        backend: SQLiteBackend | None = None,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    ) -> None:
        self.root = Path(root or Path.home() / ".hypernix" / "hyperlink" / "files")
        self.root.mkdir(parents=True, exist_ok=True)
        self.backend = backend or SQLiteBackend()
        self.max_bytes = int(max_bytes)
        self.max_image_bytes = int(max_image_bytes)
        self._lock = threading.Lock()
        self.backend.executescript(_SCHEMA)

    # -- paths --------------------------------------------------------

    def _blob_path(self, digest: str) -> Path:
        """``ab/cd/abcdef…`` — two levels of fan-out.

        A single flat directory with fifty thousand files in it is slow
        to list on every filesystem and pathological on some. Two hex
        bytes of prefix keeps any one directory to a few hundred entries
        at realistic volumes.
        """
        return self.root / digest[:2] / digest[2:4] / digest

    # -- write --------------------------------------------------------

    def put(
        self,
        data: bytes,
        *,
        filename: str,
        owner: str,
        device_id: str = "",
        session_id: str = "",
        declared_type: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Attachment:
        """Store bytes and return the attachment record.

        Re-uploading identical bytes returns a *new record* pointing at
        the *same blob*: the second upload may belong to a different
        session or device, and collapsing them would make one device's
        deletion silently remove another's attachment.
        """
        if not data:
            raise T1APIError(T1ErrorCode.VALIDATION_ERROR, "Refusing to store an empty file")
        digest = hashlib.sha256(data).hexdigest()
        content_type = sniff_content_type(data, filename=filename, declared=declared_type)
        limit = self.max_image_bytes if content_type in IMAGE_TYPES else self.max_bytes
        if len(data) > limit:
            raise T1APIError(
                T1ErrorCode.VALIDATION_ERROR,
                f"{filename or 'file'} is {len(data)} bytes; the limit for {content_type} "
                f"is {limit} bytes",
            )

        path = self._blob_path(digest)
        with self._lock:
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                # Write to a temp name in the same directory and rename:
                # a crash mid-write must not leave a truncated blob that
                # a later upload would find by hash and trust.
                tmp = path.with_suffix(".part")
                tmp.write_bytes(data)
                os.replace(tmp, path)

            record = Attachment(
                file_id="file_" + hashlib.sha256(
                    f"{digest}:{owner}:{session_id}:{time.time_ns()}".encode()
                ).hexdigest()[:24],
                sha256=digest,
                filename=_safe_filename(filename),
                content_type=content_type,
                size_bytes=len(data),
                owner=owner,
                created_at=time.time(),
                device_id=device_id,
                session_id=session_id,
                metadata=metadata or {},
            )
            with self.backend.connect() as conn:
                conn.execute(
                    """INSERT INTO hyperlink_files
                       (file_id, sha256, filename, content_type, size_bytes, owner,
                        device_id, session_id, created_at, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        record.file_id, record.sha256, record.filename, record.content_type,
                        record.size_bytes, record.owner, record.device_id, record.session_id,
                        record.created_at, json.dumps(record.metadata),
                    ),
                )
        return record

    # -- read ---------------------------------------------------------

    def get(self, file_id: str, *, owner: str | None = None) -> Attachment:
        with self.backend.connect() as conn:
            row = conn.execute(
                "SELECT * FROM hyperlink_files WHERE file_id = ?", (file_id,)
            ).fetchone()
        if row is None:
            raise T1APIError(T1ErrorCode.NOT_FOUND, f"No attachment {file_id!r}")
        record = _from_row(row)
        # A 404 rather than a 403 for someone else's file: confirming
        # that an id exists is itself information.
        if owner is not None and record.owner != owner:
            raise T1APIError(T1ErrorCode.NOT_FOUND, f"No attachment {file_id!r}")
        return record

    def read(self, file_id: str, *, owner: str | None = None) -> bytes:
        record = self.get(file_id, owner=owner)
        path = self._blob_path(record.sha256)
        if not path.exists():
            raise T1APIError(
                T1ErrorCode.NOT_FOUND,
                f"Attachment {file_id} is recorded but its blob is missing from {self.root}",
            )
        return path.read_bytes()

    def path_for(self, file_id: str, *, owner: str | None = None) -> Path:
        """On-disk path, for callers that stream rather than read."""
        return self._blob_path(self.get(file_id, owner=owner).sha256)

    def list_files(
        self, *, owner: str | None = None, session_id: str | None = None, limit: int = 100
    ) -> list[Attachment]:
        clauses, params = [], []
        if owner is not None:
            clauses.append("owner = ?")
            params.append(owner)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))
        with self.backend.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM hyperlink_files {where} ORDER BY created_at DESC LIMIT ?",  # nosec B608 - values are bound parameters; interpolated SQL is fixed or allowlisted
                tuple(params),
            ).fetchall()
        return [_from_row(r) for r in rows]

    def data_url(self, file_id: str, *, owner: str | None = None) -> str:
        """``data:image/png;base64,…`` — the form vision models take.

        Only for images. A 20 MB PDF as a data URL inside a JSON request
        body is how a model server runs out of memory parsing the
        request, so anything else is refused here rather than at the far
        end.
        """
        import base64

        record = self.get(file_id, owner=owner)
        if not record.is_image:
            raise T1APIError(
                T1ErrorCode.VALIDATION_ERROR,
                f"{record.filename} is {record.content_type}, not an image; "
                "only images can be inlined for a vision model",
            )
        payload = base64.b64encode(self.read(file_id, owner=owner)).decode("ascii")
        return f"data:{record.content_type};base64,{payload}"

    def text_of(self, file_id: str, *, owner: str | None = None, max_chars: int = 200_000) -> str:
        """Decoded text of a text attachment, truncated with a marker.

        Truncation is visible in the returned string rather than silent,
        because the alternative is a model confidently answering about
        the half of a file it was given.
        """
        record = self.get(file_id, owner=owner)
        if not record.is_text:
            raise T1APIError(
                T1ErrorCode.VALIDATION_ERROR,
                f"{record.filename} is {record.content_type}; it has no text to extract",
            )
        text = self.read(file_id, owner=owner).decode("utf-8", "replace")
        if len(text) > max_chars:
            return text[:max_chars] + f"\n\n…[truncated: {len(text) - max_chars} more characters]"
        return text

    # -- delete -------------------------------------------------------

    def delete(self, file_id: str, *, owner: str | None = None) -> bool:
        """Drop the record, and the blob when no record still needs it."""
        record = self.get(file_id, owner=owner)
        with self._lock, self.backend.connect() as conn:
            conn.execute("DELETE FROM hyperlink_files WHERE file_id = ?", (file_id,))
            remaining = conn.execute(
                "SELECT COUNT(*) AS n FROM hyperlink_files WHERE sha256 = ?", (record.sha256,)
            ).fetchone()
            still_referenced = int(remaining["n"]) if remaining is not None else 0
        if not still_referenced:
            path = self._blob_path(record.sha256)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                # A blob we cannot delete is wasted disk, not a failed
                # deletion: the record is gone and the bytes are
                # unreachable through the API.
                pass
        return True

    def usage_bytes(self, *, owner: str | None = None) -> int:
        """Total distinct bytes stored, deduplicated by hash.

        Summing ``size_bytes`` over records would double-count the
        screenshot someone sent to three sessions, and a quota that
        double-counts is a quota people work around.
        """
        where = "WHERE owner = ?" if owner is not None else ""
        params = (owner,) if owner is not None else ()
        with self.backend.connect() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT sha256, size_bytes FROM hyperlink_files {where}", params  # nosec B608 - values are bound parameters; interpolated SQL is fixed or allowlisted
            ).fetchall()
        return sum(int(r["size_bytes"]) for r in rows)


def _safe_filename(name: str) -> str:
    """Keep the basename, drop anything that could escape a directory.

    The store never uses this for a path — blobs are named by hash — but
    it *is* echoed back in ``Content-Disposition`` and shown in the app,
    so a name containing ``../`` or a newline is worth neutralising at
    the boundary rather than at each use.
    """
    base = Path(name or "file").name.replace("\x00", "")
    base = "".join(ch for ch in base if ch.isprintable() and ch not in '\\/:*?"<>|')
    return (base.strip() or "file")[:200]


def _from_row(row: Any) -> Attachment:
    return Attachment(
        file_id=row["file_id"],
        sha256=row["sha256"],
        filename=row["filename"],
        content_type=row["content_type"],
        size_bytes=int(row["size_bytes"]),
        owner=row["owner"],
        created_at=row["created_at"],
        device_id=row["device_id"] or "",
        session_id=row["session_id"] or "",
        metadata=json.loads(row["metadata"] or "{}"),
    )
