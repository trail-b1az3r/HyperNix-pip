"""Whose a HyperLink chat is (0.72.8).

A chat, its attachments, memories, preferences and sync history belong
to one *owner*. Before 0.72.8 that was the key a request came in on --
and a request with no key at all, from a trusted network, was owned by
``trusted-lan`` or ``trusted-tailnet``: one shared pool that every
keyless device on the network read and wrote. Chats were global.

Now:

* **Accounts on** (web accounts enabled, and the key belongs to one):
  the owner is the account, ``account:<id>``. Every key and device of
  that account sees the same chats; a new key minted for it does not
  start from nothing.
* **Otherwise** the owner is the key: a T1, T2 or T2.1 key's id, or for a
  paired device the key that paired it.
* **No key** (trusted-network access): no chats. Keyless access still
  reads the server's status, models and hardware; a conversation needs
  someone to belong to. ``T1_HYPERLINK_SHARED_KEYLESS_CHATS=1`` brings
  back the old shared pool for a deployment that wants it.

When a key is first seen under its account, what it owned is moved to
the account (:func:`adopt`), so turning accounts on loses nothing.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["ACCOUNT_PREFIX", "KeylessChatsRefused", "adopt", "keyless_chats_shared", "owner_for_key"]

ACCOUNT_PREFIX = "account:"

#: app.state attribute -> [(table, owner is the primary key)]
STORES: dict[str, list[tuple[str, bool]]] = {
    "t1_session_store": [("hyperlink_sessions", False)],
    "t1_attachment_store": [("hyperlink_files", False)],
    "t1_memory_store": [("hyperlink_memories", False), ("hyperlink_memory_log", False),
                        ("hyperlink_memory_floor", True)],
    "t1_notification_store": [("hyperlink_push_registrations", False), ("hyperlink_push_queue", False)],
    "t1_preference_store": [("hyperlink_preferences", True)],
    "t1_sync_store": [("hyperlink_change_log", False), ("hyperlink_claims", False)],
}

_adopted: set[tuple[str, str]] = set()
_adopt_lock = threading.Lock()


class KeylessChatsRefused(PermissionError):
    """A request with no key asked for something that belongs to someone."""


def keyless_chats_shared() -> bool:
    return os.environ.get("T1_HYPERLINK_SHARED_KEYLESS_CHATS", "").strip().lower() in ("1", "true", "yes", "on")


def _account_for_key(accounts: Any, key_id: str) -> str:
    """The id of the account *key_id* belongs to, or ""."""
    finder = getattr(accounts, "account_for_key", None)
    if finder is None:
        return ""
    try:
        account = finder(key_id)
    except Exception:  # noqa: BLE001 - an accounts store that fails must not lock people out
        logger.warning("hyperlink: could not look up the account for a key", exc_info=True)
        return ""
    return getattr(account, "account_id", "") if account is not None else ""


def owner_for_key(state: Any, key_id: str) -> str:
    """The owner of what *key_id* does in HyperLink: its account when
    accounts are on and it has one, otherwise the key itself."""
    settings = getattr(state, "t1_webauth", None)
    accounts = getattr(state, "t1_accounts", None)
    if key_id and settings is not None and getattr(settings, "enabled", False) and accounts is not None:
        account_id = _account_for_key(accounts, key_id)
        if account_id:
            owner = ACCOUNT_PREFIX + account_id
            adopt(state, key_id, owner)
            return owner
    return key_id


def adopt(state: Any, old: str, new: str) -> int:
    """Move everything *old* owns to *new*, once per process. Returns rows moved.

    Where the owner is a table's key (preferences, a memory floor) and
    *new* already has a row, *new*'s is kept and *old*'s dropped: the
    account's own settings win over one key's.
    """
    if not old or not new or old == new:
        return 0
    with _adopt_lock:
        if (old, new) in _adopted:
            return 0
        _adopted.add((old, new))
    moved = 0
    for attr, tables in STORES.items():
        store = getattr(state, attr, None)
        backend = getattr(store, "backend", None)
        if backend is None:
            continue
        lock = getattr(store, "_lock", None) or threading.Lock()
        try:
            with lock, backend.connect() as conn:
                for table, keyed in tables:
                    if keyed:
                        exists = conn.execute(f"SELECT 1 FROM {table} WHERE owner = ?", (new,)).fetchone()  # nosec B608 - table names are the constants above
                        if exists:
                            conn.execute(f"DELETE FROM {table} WHERE owner = ?", (old,))  # nosec B608 - table names are the constants above
                            continue
                    cursor = conn.execute(f"UPDATE {table} SET owner = ? WHERE owner = ?", (new, old))  # nosec B608 - table names are the constants above
                    moved += max(0, getattr(cursor, "rowcount", 0) or 0)
        except Exception:  # noqa: BLE001 - a store without one of these tables is skipped
            logger.warning("hyperlink: could not move %s's data in %s to the account", old[:8], attr, exc_info=True)
    if moved:
        logger.info("hyperlink: moved %d rows from key %s… to %s", moved, old[:8], new)
    return moved
