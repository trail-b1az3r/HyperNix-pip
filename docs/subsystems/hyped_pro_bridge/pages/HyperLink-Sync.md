# HyperLink: sync, notifications and search

New in **0.72.4.post9**. Three subsystems that exist because a phone is
not a desktop client, and the difference is not cosmetic.

- [`hypernix.hyperlink.sync`](#sync) — catching up, and not sending twice
- [`hypernix.hyperlink.notify`](#notifications) — telling a phone something happened while it slept
- [`hypernix.hyperlink.search`](#search) — finding the conversation you half remember

All three share the T1 API's existing backend, so a HyperLink deployment
is still one SQLite file (or one PostgreSQL URL) and nothing extra to
back up.

---

## Sync

### The two problems

**A retry sends the message twice.** The phone POSTs a turn, the
connection drops before the response arrives, and it cannot tell *"the
server never saw it"* from *"the server saw it and the reply was lost"*.
Retrying is the only safe-looking option and it produces two identical
user messages and two model replies — one of which cost real tokens for
nothing.

**A phone that was away does not know what it missed.** Polling
`GET /hyperlink/sessions` returns the current state of everything, which
on a cellular connection means downloading conversations it already has,
and still does not reveal that a session was **deleted**. An absence is
invisible when you are diffing against a list you no longer trust.

### Idempotency keys

The client mints a `client_msg_id` *before its first attempt* and reuses
it on every retry.

```bash
curl -X POST https://host/hyperlink/sync/claim \
     -H "Authorization: Bearer $DEVICE_TOKEN" \
     -d '{"client_msg_id": "cmsg_9f2a…"}'
```

```jsonc
{"fresh": true,  "client_msg_id": "cmsg_9f2a…", "settled": false, "result": {}}
```

`fresh` is the only field to branch on:

| `fresh` | `settled` | what it means | what to do |
|---|---|---|---|
| `true` | `false` | you won the claim | do the work |
| `false` | `true` | an earlier attempt finished | use `result`, do **not** repeat it |
| `false` | `false` | an earlier attempt is still running | wait and ask again |

Keys are scoped **per device**, so two phones cannot collide, and expire
after 24 hours — an idempotency table that only grows is a slow leak.

### The change feed

```bash
curl "https://host/hyperlink/sync?cursor=412&limit=100" \
     -H "Authorization: Bearer $DEVICE_TOKEN"
```

```jsonc
{
  "changes": [
    {"seq": 413, "kind": "created", "entity": "message",
     "entity_id": "msg_…", "session_id": "chat_…", "created_at": 1.7e9},
    {"seq": 414, "kind": "deleted", "entity": "session",
     "entity_id": "chat_…", "created_at": 1.7e9}
  ],
  "cursor": 414, "more": false, "resync_required": false, "head": 414
}
```

Loop until `more` is `false`, carrying `cursor` forward each time. Do
**not** advance the cursor by the page size: a filtered feed skips rows,
and guessing would skip real changes with them.

- **Deletions are rows**, not absences, so the phone can drop a session
  it still holds.
- **`head`** lets a brand-new device skip the history entirely: fetch
  current state, then start the feed at `head`. Starting at `0` replays
  every change ever made to rebuild a state it already has.
- **`resync_required`** means the cursor predates the oldest surviving
  row. Tombstones expire after 30 days, so the deletions that device
  needs to hear about are gone and replaying what is left would leave it
  holding a session the server has forgotten. Refetch state, restart at
  `head`.

### Sequence numbers

Allocated from a counter row read and written **inside the same
transaction as the change it labels**. Not `MAX(seq) + 1`: two writers
reading the same maximum pick the same number, and the second insert
either fails on the primary key or — on a backend without one —
succeeds, producing two rows a client can only ever see one of. Verified
under eight concurrent writers.

### What it deliberately does not do

It does not merge. There is exactly one writer of record for a
conversation — the machine — and the phone is a client of it, so
"resolve a conflict between two divergent histories" is a problem this
shape does not have. Message content is append-only; session metadata is
last-write-wins.

### Memories (0.72.6)

Memories have their own feed, `GET /memory/sync`, because they belong to
a person rather than a session and the model writes them in the middle
of a chat. The app used to refetch `/memory/list` whole: the first 200,
when the Memories screen appeared or a reply ended, with any error
swallowed. A fact the model wrote reached the phone only if somebody
happened to be looking, the 201st never did, and offline the screen was
empty.

Every memory write appends to a log in the same transaction: created,
updated or deleted, including what the auto-memory budget evicts. The
log keeps one row per memory, its latest, so a fact edited forty times
costs one row and a page never names a memory twice. Numbers come from a
counter row, like the change feed's, so they appear in commit order.

```text
GET /memory/sync?cursor=0          -> {"full": true,  "memories": [...], "cursor": 41}
GET /memory/sync?cursor=41         -> {"full": false, "memories": [], "deleted": [], "cursor": 41}
... the model remembers something ...
GET /memory/sync?cursor=41         -> {"full": false, "memories": [{...}], "deleted": [], "cursor": 44}
```

- **A delta** (`full: false`): upsert `memories` by `memory_id`, drop
  the ids in `deleted`, keep `cursor`. Ask again while `more` is true;
  a page holds at most 500.
- **The whole set** (`full: true`) replaces the copy. It is sent for a
  first sync (`reason: "first"`), a cursor older than the tombstones,
  which are kept for 30 days (`"expired"`), and a cursor this server
  never issued, as when the phone was paired to a server that has since
  been restored from a backup or replaced (`"unknown_cursor"`).
- **A chat turn** that remembers or forgets something sends
  `{"type": "memory", "cursor": N}` after `done` on `/chat/stream`, so
  the app syncs while the reply is still on screen.

The app keeps each server's copy on disk (Application Support, excluded
from backups), shows it at launch and offline, and syncs on launch, on
every return to the foreground, on the `memory` frame, and after each
edit. When the server cannot be reached, the Memories screen says so and
how old its copy is. Against a server without `/memory/sync`, it falls
back to the whole list.

---

## Notifications

Every interesting event on a HyperNix machine happens on a timescale a
phone is not awake for. A fine-tune runs for six hours. A 70B download
takes forty minutes. A chat turn against a large local model takes long
enough for iOS to suspend the app.

### What ships and what does not

Everything up to the HTTP/2 POST: which devices want what, a durable
queue, retry with backoff, collapse handling, and the exact payload APNs
expects. **Delivery itself is an operator-supplied `Transport`**, because
sending an APNs push needs an Apple team key, a signing key and a route
to `api.push.apple.com`, none of which ships with an open-source package.

`RecordingTransport` makes the whole path testable without Apple, and is
a perfectly good production choice for someone who wants notifications
in a log rather than on a lock screen.

### Device tokens are credentials

An APNs token lets whoever holds it push to that device. It is stored
because delivery needs it and **never** returned by an API, written to a
log, or included in a `repr`. Everything outward-facing carries an
eight-character `fingerprint` instead — enough to tell two of your own
phones apart, useless for sending anything.

```bash
curl -X POST https://host/hyperlink/push \
     -H "Authorization: Bearer $DEVICE_TOKEN" \
     -d '{"token": "<64 hex chars>", "bundle_id": "org.hypernix.hyperlink"}'
```

```jsonc
{"registration": {"registration_id": "push_…", "fingerprint": "ffe054fe",
                  "events": ["chat.reply", "download.done", …]}}
```

Re-registering the same token **updates** rather than duplicates: iOS
hands the app a token on every launch, and a row per launch would send
every notification once per day the app had been opened.

### Event kinds

`GET /hyperlink/push/events` serves the list rather than the app
hard-coding it, so a server that gains a kind does not need a client
update to offer it.

| kind | default |
|---|---|
| `chat.reply` | on |
| `training.done` / `training.failed` | on |
| `download.done` / `download.failed` | on |
| `quantize.done` | on |
| `thermal.alert` | on |
| `job.done`, `server.offline`, `device.paired` | off |

Failures are on by default: silence is the worst outcome there. An event
nobody subscribed to is dropped **at enqueue time**, so a queue length of
zero means "nothing to say" rather than "plenty to say, all unwanted".

### Collapse ids

Download progress at 40%, 60% and 80% is one notification the phone sees
once, not three buzzes for one fact. Passing a `collapse_id` supersedes
any undelivered notification with the same id.

### The 4 KB wall

APNs rejects a payload over 4096 bytes, and a model reply is frequently
longer on its own. `build_apns_payload()` trims the body by **measuring
the encoded payload**, not by estimating:

- `json.dumps` defaults to `ensure_ascii=True`, which escapes each
  non-ASCII character to a six-byte `\uXXXX` where UTF-8 needs three.
  The first version measured UTF-8 and shipped Japanese replies at
  double the intended size. `encode_payload()` is the single canonical
  wire form, and a transport must use it.
- Subtracting the overflow over-corrects: a body of 8000 double quotes
  escapes to two bytes each, so the first overflow is about as large as
  the budget, and the subtraction drove it to zero. A reply containing
  code arrived with **no body at all**. Binary search finds the real
  maximum instead.

---

## Search

A phone accumulates months of conversations and offers a date-sorted
list to find them in. What people want is *"the one where I worked out
the CUDA thing"*, and scrolling is the only tool for it.

```bash
curl "https://host/hyperlink/search?q=cuda+vram&limit=25" \
     -H "Authorization: Bearer $DEVICE_TOKEN"
```

### Why not FTS5

The T1 API runs on SQLite **or** PostgreSQL, and FTS5 has no PostgreSQL
counterpart sharing its syntax. Writing to an FTS5 virtual table would
make search a SQLite-only feature and the schema unportable.

So SQL narrows — by owner, archive state and date, all indexed — and
Python matches. That buys three things `LIKE` cannot give:

**Unicode-correct case folding.** SQLite's `LIKE` is case-insensitive
for ASCII only. `LIKE '%STRASSE%'` does not match "straße" and
`'%İSTANBUL%'` does not match "istanbul". `str.casefold()` handles both,
and NFKC normalisation first means a phone keyboard and a desktop
keyboard produce comparable text. Accents are **not** stripped — folding
is not transliteration, and "resume" should not match "résumé".

**No wildcard injection.** A query containing `%` is a search for a
percent sign, not a request to match everything.

**Ranking that can see the whole match.** Term proximity and field
weighting need the positions, which a boolean `LIKE` has discarded by the
time it answers.

The cost is a bounded scan — 20,000 rows — and the result says `capped`
when it hit the bound, so the client can say "showing the most recent"
rather than implying it searched everything. A silent partial answer is
what makes someone conclude the conversation is gone.

### Ranking

Deliberately simple and explainable, because a chat search that returns a
surprising order is worse than one that returns an obvious one: title
beats body, all-terms beats some, exact phrase beats scattered words,
more occurrences beats fewer (damped by a square root, or the longest
document wins every search), and recency breaks ties with a 30-day half
life.

Terms are ANDed for *ranking* but not for *inclusion* — requiring every
term turns a slightly misremembered query into no results, which reads as
"it is gone".

### Snippets

Offsets, not markup. A server that returns HTML has decided how a
SwiftUI view should highlight a match, which it cannot use.

```jsonc
{"text": "gradient checkpointing trades compute for vram.",
 "ranges": [[42, 46]], "truncated_start": false, "truncated_end": false}
```

---

## Endpoints

| method | path | |
|---|---|---|
| `GET` | `/hyperlink/sync` | changes since a cursor, plus `head` |
| `POST` | `/hyperlink/sync/claim` | claim an idempotency key |
| `POST` | `/hyperlink/push` | register an APNs token |
| `GET` | `/hyperlink/push` | this owner's registrations (never tokens) |
| `GET` | `/hyperlink/push/events` | what may be subscribed to |
| `PATCH` | `/hyperlink/push/{id}` | narrow the event set |
| `DELETE` | `/hyperlink/push/{id}` | unregister, and drop its queue |
| `GET` | `/hyperlink/search` | search sessions and messages |
| `GET` | `/memory/sync` | memories changed since a cursor, or the whole set |

A registration id is not a secret, so it is never the authority: every
mutation checks the registration belongs to the caller, and answers
**404** rather than 403 when it does not — confirming an id exists tells
an unauthorised caller something they should not learn.

See also: [CLI](CLI.md) · [T1 API](T1-API.md) · [Home](Home.md)
