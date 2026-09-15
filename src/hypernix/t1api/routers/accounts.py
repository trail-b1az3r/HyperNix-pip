"""t1api.routers.accounts — the sign-up pages and the session API.

The endpoints behind :mod:`hypernix.t1api.accounts`. Two surfaces over
one store: HTML pages a person uses in a browser, and JSON endpoints the
pages (and a script) call.

What these routes can do
------------------------
Nothing the account's own T1 keys could not already do. The privileged
operations here are "mint a key for the signed-in account" and "revoke
one of that account's keys" — no route reads another account's keys, and
none touches any other part of the API. Every other T1 route still
authenticates with a T1 key exactly as before.

Off by default
--------------
``T1_ACCOUNTS_ENABLED`` defaults to off, and every route here answers 404
when it is. Not 403: a deployment that does not have accounts should not
advertise that it could, and "this endpoint does not exist here" is the
true statement.

CSRF
----
The session is a cookie, so the browser attaches it to any request to
this origin including one another site caused. Three things stop that,
and all three are on because the thing being protected is a page that
mints API keys:

1. ``SameSite`` on the cookie (``strict`` on loopback, ``lax`` elsewhere).
2. An ``Origin`` check against the configured public URL.
3. A per-session CSRF token in a custom header, which a cross-site form
   cannot set and a cross-site fetch cannot send without a preflight
   this server does not answer.
"""
from __future__ import annotations

import html
import logging
from typing import Any

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse

from ..accounts import Account, AccountError, AccountStore, Session
from ..errors import T1APIError, T1ErrorCode
from ..webauth import (
    CSRF_HEADER,
    SESSION_COOKIE,
    WebAuthSettings,
    client_ip,
    cookie_kwargs,
    is_origin_allowed,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/accounts", tags=["accounts"])


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def get_settings(request: Request) -> WebAuthSettings:
    settings = getattr(request.app.state, "t1_webauth", None)
    if settings is None or not settings.enabled:
        # 404, not 403. A deployment without accounts should not
        # advertise that it could have them.
        raise T1APIError(
            T1ErrorCode.NOT_FOUND,
            "This server does not have web accounts enabled.",
        )
    return settings


def get_store(request: Request) -> AccountStore:
    store = getattr(request.app.state, "t1_accounts", None)
    if store is None:
        raise T1APIError(
            T1ErrorCode.NOT_FOUND,
            "This server does not have web accounts enabled.",
        )
    return store


def _peer_ip(request: Request, settings: WebAuthSettings) -> str:
    peer = request.client.host if request.client else ""
    return client_ip(request.headers, peer, settings.mode)


def current_session(
    request: Request,
    store: AccountStore = Depends(get_store),
    settings: WebAuthSettings = Depends(get_settings),
) -> tuple[Session, Account]:
    """The signed-in session, or 401."""
    token = request.cookies.get(SESSION_COOKIE, "")
    found = store.resolve_session(token, client_ip=_peer_ip(request, settings))
    if found is None:
        raise T1APIError(
            T1ErrorCode.AUTH_MISSING_CREDENTIALS, "Sign in to do that.",
        )
    return found


def require_write(
    request: Request,
    pair: tuple[Session, Account] = Depends(current_session),
    settings: WebAuthSettings = Depends(get_settings),
    store: AccountStore = Depends(get_store),
) -> tuple[Session, Account]:
    """A signed-in session that has also cleared the CSRF checks."""
    session, account = pair
    origin = request.headers.get("origin", "")
    if not is_origin_allowed(origin, settings):
        logger.warning(
            "t1api.accounts: refused a request from origin %r", origin[:128]
        )
        raise T1APIError(
            T1ErrorCode.AUTH_INSUFFICIENT_SCOPE,
            "That request came from an origin this server does not serve.",
        )
    if not store.check_csrf(session, request.headers.get(CSRF_HEADER, "")):
        raise T1APIError(
            T1ErrorCode.AUTH_INSUFFICIENT_SCOPE,
            f"Missing or wrong {CSRF_HEADER}. Read it from GET /accounts/me "
            "and send it with every state-changing request.",
        )
    return session, account


def _account_error(exc: AccountError) -> T1APIError:
    """Map an account failure onto the T1 error taxonomy."""
    mapping = {
        "bad_credentials": T1ErrorCode.AUTH_INVALID_KEY,
        "account_locked": T1ErrorCode.AUTH_INVALID_KEY,
        "account_disabled": T1ErrorCode.AUTH_REVOKED_KEY,
        "registration_closed": T1ErrorCode.AUTH_INSUFFICIENT_SCOPE,
        "not_found": T1ErrorCode.NOT_FOUND,
    }
    return T1APIError(
        mapping.get(exc.code, T1ErrorCode.VALIDATION_ERROR), exc.message
    )


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


@router.get("/config")
def account_config(settings: WebAuthSettings = Depends(get_settings)) -> dict[str, Any]:
    """What the sign-up page needs to render itself. No secrets."""
    return settings.to_dict()


@router.post("/register")
async def register(
    request: Request,
    response: Response,
    store: AccountStore = Depends(get_store),
    settings: WebAuthSettings = Depends(get_settings),
) -> JSONResponse:
    """Create an account and sign it in.

    No T1 key required — this is the whole point of the module. What a
    new account can do is mint keys for itself, which is the smallest
    useful thing and not a way into anyone else's.
    """
    body = await request.json()
    try:
        account = store.create(
            username=body.get("username", ""),
            password=body.get("password", ""),
            display_name=body.get("display_name", ""),
            email=body.get("email", ""),
            invite_code=body.get("invite_code", ""),
        )
    except AccountError as exc:
        raise _account_error(exc) from exc

    session, token = store.start_session(
        account,
        client_ip=_peer_ip(request, settings),
        user_agent=request.headers.get("user-agent", ""),
    )
    payload = JSONResponse({
        "account": account.to_dict(),
        "csrf_token": session.csrf_token,
        "expires_at": session.expires_at,
    })
    payload.set_cookie(
        value=token, **cookie_kwargs(settings.mode, max_age=store.session_ttl)
    )
    logger.info("t1api.accounts: registered %s", account.username)
    return payload


@router.post("/login")
async def login(
    request: Request,
    store: AccountStore = Depends(get_store),
    settings: WebAuthSettings = Depends(get_settings),
) -> JSONResponse:
    body = await request.json()
    try:
        account = store.authenticate(
            body.get("username", ""), body.get("password", "")
        )
    except AccountError as exc:
        raise _account_error(exc) from exc

    session, token = store.start_session(
        account,
        client_ip=_peer_ip(request, settings),
        user_agent=request.headers.get("user-agent", ""),
    )
    payload = JSONResponse({
        "account": account.to_dict(),
        "csrf_token": session.csrf_token,
        "expires_at": session.expires_at,
    })
    payload.set_cookie(
        value=token, **cookie_kwargs(settings.mode, max_age=store.session_ttl)
    )
    return payload


@router.post("/logout")
def logout(
    pair: tuple[Session, Account] = Depends(current_session),
    store: AccountStore = Depends(get_store),
    settings: WebAuthSettings = Depends(get_settings),
) -> JSONResponse:
    """End this session.

    Deliberately not behind the CSRF check. A forged logout is a denial
    of service against one browser tab and nothing else, and refusing a
    logout is the one failure that leaves somebody *more* signed in than
    they wanted to be.
    """
    session, _account = pair
    store.end_session(session.session_id)
    payload = JSONResponse({"ok": True})
    payload.delete_cookie(SESSION_COOKIE, path="/")
    return payload


@router.get("/me")
def whoami(
    pair: tuple[Session, Account] = Depends(current_session),
) -> dict[str, Any]:
    """The signed-in account, and the CSRF token for the next write."""
    session, account = pair
    return {
        "account": account.to_dict(include_private=True),
        "csrf_token": session.csrf_token,
        "session": {
            "created_at": session.created_at,
            "expires_at": session.expires_at,
        },
    }


@router.post("/password")
async def change_password(
    request: Request,
    pair: tuple[Session, Account] = Depends(require_write),
    store: AccountStore = Depends(get_store),
) -> JSONResponse:
    """Change this account's password, checking the old one first."""
    _session, account = pair
    body = await request.json()
    try:
        store.change_password(
            account.account_id,
            body.get("current_password", ""),
            body.get("new_password", ""),
        )
    except AccountError as exc:
        raise _account_error(exc) from exc
    # Every session, including this one: a password change that left the
    # current browser signed in would leave a stolen one signed in too.
    payload = JSONResponse({"ok": True, "signed_out": True})
    payload.delete_cookie(SESSION_COOKIE, path="/")
    return payload


@router.get("/keys")
def list_keys(
    request: Request,
    pair: tuple[Session, Account] = Depends(current_session),
) -> dict[str, Any]:
    """This account's keys. Metadata only — never the key material.

    A key is shown once, when it is minted. If it is lost, mint another
    and revoke the old one; there is nowhere here it could be read back
    from, by design.
    """
    _session, account = pair
    keymaster = getattr(request.app.state, "t1_keymaster", None)
    keys = []
    for key_id in account.key_ids:
        meta = keymaster.get(key_id) if keymaster else None
        if meta is None:
            continue
        keys.append({
            "key_id": meta.key_id,
            "key_type": meta.key_type.value,
            "active": meta.active,
            "created_at": meta.created_at,
            "expires_at": meta.expires_at,
            "note": meta.note,
        })
    return {"keys": keys, "count": len(keys)}


@router.post("/keys")
async def mint_key(
    request: Request,
    pair: tuple[Session, Account] = Depends(require_write),
    store: AccountStore = Depends(get_store),
) -> dict[str, Any]:
    """Mint a T1 key for this account. The one thing an account is for.

    The key is returned once and never again — it is stored encrypted in
    the Keymaster and nothing here can read it back.
    """
    from hypernix.security.keymaster import KeyScope, KeyType

    _session, account = pair
    keymaster = getattr(request.app.state, "t1_keymaster", None)
    if keymaster is None:
        raise T1APIError(
            T1ErrorCode.INTERNAL_ERROR,
            "This server has no Keymaster, so it cannot mint keys.",
        )
    body = await request.json() if await request.body() else {}

    # A self-service key gets read and write, never admin. An account can
    # be created by anyone the registration policy lets in, and "anyone
    # who can sign up can mint an admin key" would make the policy the
    # only thing between a stranger and the whole API.
    scopes = {KeyScope.READ, KeyScope.WRITE}
    meta = keymaster.create(
        key_type=KeyType.USER,
        scopes=scopes,
        note=str(body.get("note", ""))[:200] or f"web account {account.username}",
    )
    store.remember_key(account.account_id, meta.key_id)
    logger.info(
        "t1api.accounts: %s minted key %s", account.username, meta.key_id[:8]
    )
    return {
        "key_id": meta.key_id,
        # Once. There is no endpoint that returns this again.
        "key": meta.key,
        "key_type": meta.key_type.value,
        "scopes": sorted(s.value for s in scopes),
        "warning": (
            "This is the only time this key is shown. Store it now; the "
            "server keeps it encrypted and cannot read it back."
        ),
    }


@router.delete("/keys/{key_id}")
def revoke_key(
    key_id: str,
    request: Request,
    pair: tuple[Session, Account] = Depends(require_write),
    store: AccountStore = Depends(get_store),
) -> dict[str, Any]:
    """Revoke one of this account's keys."""
    _session, account = pair
    if key_id not in account.key_ids:
        # Not "no such key": whether a key id exists elsewhere on this
        # server is not this account's business.
        raise T1APIError(
            T1ErrorCode.NOT_FOUND, "That key does not belong to this account."
        )
    keymaster = getattr(request.app.state, "t1_keymaster", None)
    if keymaster is not None:
        keymaster.revoke(key_id, reason=f"revoked by {account.username}")
    store.forget_key(account.account_id, key_id)
    return {"ok": True, "key_id": key_id}


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>HyperNix T1 &mdash; {heading}</title>
<style>
  :root {{ color-scheme: light dark; --fg:#16181d; --bg:#fbfbfd; --mut:#666c78;
           --line:#d8dce3; --acc:#2f6f4f; --bad:#a3342c; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --fg:#e7e9ee; --bg:#14161a; --mut:#9aa2b1; --line:#2c313a;
             --acc:#7fd1a4; --bad:#e08b83; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.55
          ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }}
  main {{ max-width: 27rem; margin: 0 auto; padding: 3rem 1rem 4rem; }}
  h1 {{ font-size: 1.35rem; margin: 0 0 .25rem; letter-spacing: -.01em; }}
  p.sub {{ color: var(--mut); margin: 0 0 2rem; }}
  label {{ display:block; font-size:.82rem; color:var(--mut); margin:1rem 0 .3rem; }}
  input {{ width:100%; padding:.6rem .7rem; font:inherit; color:inherit;
           background:transparent; border:1px solid var(--line); border-radius:7px; }}
  input:focus {{ outline:2px solid var(--acc); outline-offset:1px; border-color:transparent; }}
  button {{ width:100%; margin-top:1.5rem; padding:.65rem; font:inherit;
            font-weight:600; color:var(--bg); background:var(--fg);
            border:0; border-radius:7px; cursor:pointer; }}
  button:disabled {{ opacity:.5; cursor:default; }}
  .msg {{ margin-top:1rem; padding:.7rem .8rem; border-radius:7px;
          border:1px solid var(--line); font-size:.9rem; display:none; }}
  .msg.show {{ display:block; }}
  .msg.bad {{ border-color:var(--bad); color:var(--bad); }}
  code {{ font:13px ui-monospace, "SF Mono", Menlo, monospace;
          background:rgba(128,128,128,.14); padding:.15em .35em; border-radius:4px;
          overflow-wrap:anywhere; }}
  .alt {{ margin-top:2rem; font-size:.87rem; color:var(--mut); }}
  a {{ color:var(--acc); }}
  .note {{ margin-top:2.5rem; padding-top:1.25rem; border-top:1px solid var(--line);
           font-size:.82rem; color:var(--mut); }}
</style>
<main>
  <h1>{heading}</h1>
  <p class="sub">{subtitle}</p>
  <form id="f" autocomplete="on">{fields}
    <button type="submit">{action}</button>
  </form>
  <div class="msg" id="m"></div>
  <p class="alt">{alt}</p>
  <p class="note">{note}</p>
</main>
<script>
const base = {base!r};
const form = document.getElementById("f");
const box  = document.getElementById("m");
function say(text, bad) {{
  box.className = "msg show" + (bad ? " bad" : "");
  box.innerHTML = text;
}}
form.addEventListener("submit", async (event) => {{
  event.preventDefault();
  const button = form.querySelector("button");
  button.disabled = true;
  const body = Object.fromEntries(new FormData(form).entries());
  try {{
    const reply = await fetch(base + {endpoint!r}, {{
      method: "POST",
      headers: {{"Content-Type": "application/json"}},
      credentials: "same-origin",
      body: JSON.stringify(body),
    }});
    const data = await reply.json();
    if (!reply.ok) {{
      say((data.error && data.error.message) || "That did not work.", true);
      button.disabled = false;
      return;
    }}
    say("Signed in. Taking you to your account\\u2026", false);
    location.href = base + "/accounts/";
  }} catch (err) {{
    say("Could not reach the server: " + err.message, true);
    button.disabled = false;
  }}
}});
</script>
"""

_HOME = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>HyperNix T1 &mdash; your account</title>
<style>
  :root {{ color-scheme: light dark; --fg:#16181d; --bg:#fbfbfd; --mut:#666c78;
           --line:#d8dce3; --acc:#2f6f4f; --bad:#a3342c; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --fg:#e7e9ee; --bg:#14161a; --mut:#9aa2b1; --line:#2c313a;
             --acc:#7fd1a4; --bad:#e08b83; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.55
          ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }}
  main {{ max-width: 42rem; margin:0 auto; padding: 3rem 1rem 4rem; }}
  h1 {{ font-size:1.35rem; margin:0 0 .2rem; }}
  h2 {{ font-size:1rem; margin:2.5rem 0 .6rem; }}
  p.sub {{ color:var(--mut); margin:0; }}
  button {{ padding:.5rem .9rem; font:inherit; font-weight:600; color:var(--bg);
            background:var(--fg); border:0; border-radius:7px; cursor:pointer; }}
  button.ghost {{ color:var(--fg); background:transparent;
                  border:1px solid var(--line); font-weight:400; }}
  table {{ width:100%; border-collapse:collapse; font-size:.88rem; }}
  th, td {{ text-align:left; padding:.5rem .4rem; border-bottom:1px solid var(--line); }}
  th {{ color:var(--mut); font-weight:500; font-size:.78rem; }}
  .wrap {{ overflow-x:auto; }}
  code {{ font:13px ui-monospace, Menlo, monospace; overflow-wrap:anywhere; }}
  .fresh {{ margin-top:1rem; padding:.9rem; border:1px solid var(--acc);
            border-radius:7px; font-size:.9rem; }}
  .empty {{ color:var(--mut); font-size:.9rem; }}
  a {{ color:var(--acc); }}
</style>
<main>
  <h1>Your T1 account</h1>
  <p class="sub" id="who">&hellip;</p>

  <h2>API keys</h2>
  <div class="wrap"><table>
    <thead><tr><th>Key ID</th><th>Type</th><th>Status</th><th></th></tr></thead>
    <tbody id="keys"><tr><td colspan="4" class="empty">Loading&hellip;</td></tr></tbody>
  </table></div>
  <p><button id="mint">Mint a new key</button>
     <button id="out" class="ghost">Sign out</button></p>
  <div id="fresh"></div>

  <p style="margin-top:3rem;color:var(--mut);font-size:.82rem">
    A key is shown once, when it is minted. The server stores it encrypted and
    cannot read it back &mdash; if you lose one, mint another and revoke the old.
  </p>
</main>
<script>
const base = {base!r};
let csrf = "";
const api = (path, options) => fetch(base + "/accounts" + path, Object.assign(
  {{credentials: "same-origin", headers: {{"Content-Type": "application/json",
    {csrf_header!r}: csrf}}}}, options || {{}}));

async function load() {{
  const reply = await api("/me");
  if (!reply.ok) {{ location.href = base + "/accounts/login"; return; }}
  const data = await reply.json();
  csrf = data.csrf_token;
  document.getElementById("who").textContent =
    data.account.username + (data.account.is_admin ? " \\u00b7 admin" : "");
  const keys = await (await api("/keys")).json();
  const body = document.getElementById("keys");
  body.innerHTML = "";
  if (!keys.keys.length) {{
    body.innerHTML = '<tr><td colspan="4" class="empty">No keys yet.</td></tr>';
    return;
  }}
  for (const key of keys.keys) {{
    const row = document.createElement("tr");
    row.innerHTML = "<td><code>" + key.key_id.slice(0, 8) + "\\u2026</code></td><td>" +
      key.key_type + "</td><td>" + (key.active ? "active" : "revoked") + "</td><td></td>";
    if (key.active) {{
      const kill = document.createElement("button");
      kill.className = "ghost"; kill.textContent = "Revoke";
      kill.onclick = async () => {{
        if (!confirm("Revoke this key? Anything using it stops working.")) return;
        await api("/keys/" + key.key_id, {{method: "DELETE"}});
        load();
      }};
      row.lastElementChild.appendChild(kill);
    }}
    body.appendChild(row);
  }}
}}

document.getElementById("mint").onclick = async () => {{
  const reply = await api("/keys", {{method: "POST", body: "{{}}"}});
  const data = await reply.json();
  if (!reply.ok) {{ alert((data.error && data.error.message) || "Could not mint a key."); return; }}
  const box = document.getElementById("fresh");
  box.className = "fresh";
  box.textContent = "";
  const label = document.createElement("div");
  label.textContent = "Your new key \\u2014 copy it now, it is not shown again:";
  const value = document.createElement("code");
  // textContent, not innerHTML: the key is server-generated but it goes
  // through a DOM either way and there is no reason for it to be parsed.
  value.textContent = data.key;
  box.appendChild(label); box.appendChild(value);
  load();
}};

document.getElementById("out").onclick = async () => {{
  await api("/logout", {{method: "POST"}});
  location.href = base + "/accounts/login";
}};

load();
</script>
"""


def _base_path(request: Request) -> str:
    """The mount prefix, so the pages work under ``/t1`` as well as ``/``."""
    root = request.scope.get("root_path", "") or ""
    return root.rstrip("/")


def _page(
    request: Request, *, heading: str, subtitle: str, fields: str,
    action: str, endpoint: str, alt: str, note: str,
) -> HTMLResponse:
    return HTMLResponse(_PAGE.format(
        heading=html.escape(heading),
        subtitle=subtitle,
        fields=fields,
        action=html.escape(action),
        endpoint=endpoint,
        alt=alt,
        note=note,
        base=_base_path(request),
    ))


@router.get("/login", response_class=HTMLResponse)
def login_page(
    request: Request, settings: WebAuthSettings = Depends(get_settings)
) -> HTMLResponse:
    can_register = settings.registration != "closed"
    base = _base_path(request)
    return _page(
        request,
        heading="Sign in",
        subtitle="to your HyperNix T1 server.",
        fields=(
            '<label for="u">Username</label>'
            '<input id="u" name="username" autocomplete="username" autofocus required>'
            '<label for="p">Password</label>'
            '<input id="p" name="password" type="password" '
            'autocomplete="current-password" required>'
        ),
        action="Sign in",
        endpoint="/accounts/login",
        alt=(
            f'No account? <a href="{base}/accounts/register">Create one</a>.'
            if can_register else
            "This server is not accepting new accounts."
        ),
        note=(
            "Signing in gets you a page that mints API keys for your own "
            "account. It is not itself API access."
        ),
    )


@router.get("/register", response_class=HTMLResponse)
def register_page(
    request: Request,
    store: AccountStore = Depends(get_store),
    settings: WebAuthSettings = Depends(get_settings),
) -> HTMLResponse:
    base = _base_path(request)
    allowed, reason = store.can_register()
    if not allowed and settings.registration != "invite":
        return _page(
            request,
            heading="Registration is closed",
            subtitle=html.escape(reason),
            fields="",
            action="Sign in instead",
            endpoint="/accounts/login",
            alt=f'<a href="{base}/accounts/login">Sign in</a>',
            note="",
        )
    invite = (
        '<label for="i">Invite code</label>'
        '<input id="i" name="invite_code" required>'
        if settings.registration == "invite" else ""
    )
    from ..accounts import PASSWORD_MIN_LENGTH

    return _page(
        request,
        heading="Create an account",
        subtitle="No API key needed. You will mint your own.",
        fields=(
            '<label for="u">Username</label>'
            '<input id="u" name="username" autocomplete="username" '
            'pattern="[a-z0-9][a-z0-9._\\-]{1,31}" autofocus required>'
            '<label for="p">Password</label>'
            f'<input id="p" name="password" type="password" '
            f'autocomplete="new-password" minlength="{PASSWORD_MIN_LENGTH}" required>'
            + invite
        ),
        action="Create account",
        endpoint="/accounts/register",
        alt=f'Already have one? <a href="{base}/accounts/login">Sign in</a>.',
        note=(
            f"At least {PASSWORD_MIN_LENGTH} characters. There is no rule here "
            "about symbols — length is the property that matters, and a rule "
            "demanding a symbol produces \"Password1!\" on every system that "
            "has one."
        ),
    )


@router.get("/", response_class=HTMLResponse)
@router.get("", response_class=HTMLResponse)
def account_home(
    request: Request, _settings: WebAuthSettings = Depends(get_settings)
) -> HTMLResponse:
    """The account page. Its own script redirects to login when not signed in."""
    return HTMLResponse(_HOME.format(
        base=_base_path(request), csrf_header=CSRF_HEADER,
    ))
