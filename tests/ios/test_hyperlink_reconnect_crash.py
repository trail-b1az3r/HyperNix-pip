"""HyperLink crashed on connecting to a server -- reported over Tailscale.

Checked on the Swift source, without a Mac, like the other files here.

``HyperLinkClient`` is an actor, and an actor is re-entrant at every
``await``. ``send`` worked out its failover order from ``endpoints``,
awaited the network, and then indexed ``endpoints`` again. When
``configure`` replaced the list during that wait -- connecting to a
server, pairing, signing in, switching machines -- the old indices ran
past the end of the new, shorter list: index out of range, which in
Swift is a crash. Away from home on a tailnet it is the usual case, not
a rare one: the LAN address at the front of the list takes its whole
timeout to fail, and that is plenty of time to connect somewhere else.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

IOS = Path(__file__).resolve().parents[2] / "ios" / "HyperLink" / "Sources"

if not IOS.is_dir():  # pragma: no cover - the app is not in every checkout
    pytest.skip("the iOS sources are not here", allow_module_level=True)


def _send() -> str:
    text = (IOS / "Networking" / "HyperLinkClient.swift").read_text(encoding="utf-8")
    start = text.index("    private func send(")
    return text[start:text.index("\n    }\n", start)]


def test_send_iterates_a_copy_of_the_endpoints():
    body = _send()
    assert "let bases = endpoints" in body
    # Nothing in the loop reads the live list.
    assert not re.search(r"\bendpoints\[", body)
    assert "endpoints.count" not in body


def test_a_request_overtaken_by_configure_is_cancelled_not_used():
    text = (IOS / "Networking" / "HyperLinkClient.swift").read_text(encoding="utf-8")
    configure = text[text.index("func configure(endpoints:"):]
    configure = configure[:configure.index("\n    }\n")]
    assert "generation &+= 1" in configure
    body = _send()
    # Checked after the await, before the answer or its index is kept.
    after = body[body.index("try await session.data(for: request)"):]
    assert after.index("guard generation == setup") < after.index("preferredIndex = index")
    # And the cancellation is not mistaken for a dead address.
    assert "catch is CancellationError" in body


def test_the_app_does_not_report_a_cancelled_request_as_an_error():
    state = (IOS / "Store" / "AppState.swift").read_text(encoding="utf-8")
    handle = state[state.index("private func handle(_ error: Error)"):]
    assert handle.index("is CancellationError") < handle.index("lastError =")


def test_a_missing_credential_does_not_unpair():
    """The other half of the report. A keyless connection sends no
    credential, and the server answered its first chat-list refresh with
    401 AUTH_MISSING_CREDENTIALS ("chats need an account or a key"). The
    app read every 401 AUTH_* as a revoked device and unpaired, so a phone
    that connected over Tailscale was signed straight back out."""
    text = (IOS / "Networking" / "HyperLinkClient.swift").read_text(encoding="utf-8")
    rule = text[text.index("var requiresRepairing: Bool {"):]
    rule = rule[:rule.index("\n    }\n")]
    assert 'code != "AUTH_MISSING_CREDENTIALS"' in rule
