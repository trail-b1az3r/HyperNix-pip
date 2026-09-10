"""tvtop — live training dashboard (compatibility shim).

This module exists solely for backwards-compatibility.  All
functionality lives in :mod:`hypernix.tv`; this file just
re-exports everything so that ``import hypernix.tvtop`` continues
to work after the v0.70.0 tvtop rewrite.

Quick use::

    from hypernix.tvtop import cli_main  # or just run `tvtop` CLI

The console script ``tvtop`` is still registered in pyproject.toml
and points at :func:`hypernix.tv.cli_main`, which is also imported
here for convenience.
"""
from __future__ import annotations

from hypernix.system.deprecation import deprecated_module

# This module said it existed "solely for backwards-compatibility" and
# then said nothing to anyone importing it. A shim that never announces
# itself keeps its callers on the shim.
deprecated_module(
    "hypernix.monitoring.tvtop",
    instead="hypernix.monitoring.tv",
    since="0.70.0",
    extra="The `tvtop` console script now runs hypernix.monitoring.cctvtop.",
)

# Re-export everything from the real implementation module.
from hypernix.monitoring.tv import Frame, LogTail, TVTop, cli_main  # noqa: F401

__all__ = [
    "cli_main",
    "TVTop",
    "Frame",
    "LogTail",
]
