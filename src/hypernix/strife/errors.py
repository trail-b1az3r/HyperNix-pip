"""strife.errors — every refusal has a code, a reason and a way forward."""
from __future__ import annotations

from typing import Any

__all__ = ["StrifeError"]


class StrifeError(RuntimeError):
    """Strife cannot do what was asked.

    ``code`` is stable and machine-readable (``strife.<area>.<reason>``);
    ``remedy`` says what to do about it. ``details`` carries the specifics
    (the tensor, the type id, the architecture) for ``--json`` output.
    """

    def __init__(self, code: str, message: str, *, remedy: str = "",
                 details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.remedy = remedy
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), "remedy": self.remedy,
                "details": self.details}

    def __str__(self) -> str:  # pragma: no cover - trivial
        return super().__str__()
