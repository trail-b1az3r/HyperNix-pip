# HyperNix Studio

A Linux desktop client for a HyperNix server: switch models, chat, work
on a folder of code, and let the model edit files in it — with your
agreement, every time.

Qt 6 with QML, C++17.

```bash
sudo apt install qt6-base-dev qt6-declarative-dev cmake g++   # Debian/Ubuntu
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/hypernix-studio
```

## What it does

**Model switching.** The server's registry, with a GPU-offloading slider
that defaults to *let the server decide* — because the server is the
machine that knows how much VRAM is actually free, and a layer count
chosen on a client is a guess about somebody else's GPU.

**Chatting.** Through `/inference/chat`, the governed surface: the
registry, the routing cascade and the quota all apply. Not
`/bridge/lmstudio`, which applies none of them.

**Coding, remotely.** Choose a folder; the model gets file tools scoped
to it. The server can be this machine or one across a tailnet, and
Studio does not care which — there is one protocol, not a local path and
a remote path.

**Tool calling.** `read_file`, `list_directory`, `write_file`,
`create_file`, `delete_file`, `web_search`.

**Hugging Face.** Paste a model page or file link; the *server* resolves
and downloads it, because the server is the machine that will hold the
weights and the one that has the token.

**GPU / CPU / RAM.** From `/training/resources`, the server's own GPU
abstraction — NVIDIA, AMD and CPU-only all arrive in the same shape.

## What it will not do

**It cannot run a command.** There is no shell tool, no `exec`, no "run
the tests" button that shells out. Not disabled — absent. A model can ask
for a file to be written and a person can agree; there is no path by
which a model runs code. That is the one guarantee here that does not
depend on a check being correct, and the only way to keep it is not to
write the feature.

**It does not administer anything.** Studio authenticates with a **T2S
key**, limited by construction to reading and non-admin writing. No code
path needs more. Key management, training controls and pairing belong to
`waiter` and `hypernix-t1` on the machine itself.

**It does not run a model.** The server has the GPU, the registry, the
quota and the cascade. Re-implementing any of that here would mean two
things to keep in agreement.

## The approval flow

This is the part worth reading before changing anything.

1. The model asks for a tool. **Nothing has happened.**
2. `ToolPolicy` decides. A **Deny** never reaches the user — there is
   nothing to approve about reading a private key, and a prompt for one
   is a prompt people learn to click through, which would then be there
   for the request that mattered.
3. An **Allow** — a read or a listing, inside the workspace, not a
   credentials file — runs immediately. That is the only case that does.
4. A **NeedsApproval** becomes a dialog showing the *resolved* path and
   the full contents. Nothing runs until it is answered.

What the dialog deliberately lacks:

| not present | because |
|---|---|
| "Approve all" | every call is a separate decision |
| "Remember this" | a remembered yes is a yes nobody read |
| a timeout | nothing defaults to approval |
| click-outside-to-dismiss | dismissing something unread would be an answer |
| default focus on the affirmative | a dialog answered by reflex should answer "no" |

Escape declines. The decline button holds focus.

### The two boundary checks

`ToolPolicy::Resolve` is **lexical**: it collapses `.` and `..` and
requires the result to be under the workspace, with no filesystem access
at all. That catches `../../etc/passwd` without a syscall, and it is
pure, so every escape is testable.

`ToolRunner::IsTrulyInside` is the **filesystem** check, run again
immediately before each operation. It catches what the lexical one
cannot: a symlink *inside* the workspace pointing out of it, which
passes every string test there is. Running it at the moment of the write
rather than at the moment of the decision also closes the gap between
the two — a path approved a second ago can be a symlink now.

Neither is redundant. Both have their own tests.

## Layout

```
src/
  ToolPolicy.h/.cpp      what a model may do. No Qt. Pure.
  ToolRunner.h/.cpp      doing it. No Qt. std::filesystem.
  HyperLinkClient.h/.cpp the T1 API, over Qt Network.
  StudioBridge.h/.cpp    the one object QML talks to.
  main.cpp
qml/
  Theme.qml              every colour, size and duration, once.
  Main.qml               window, sidebar, stack, identity block.
  ChatView, CodeView, ModelsView, MachineView
  ToolApproval.qml       the dialog described above.
tests/
  tool_policy_test.cpp   69 checks. No Qt, no filesystem.
  tool_runner_test.cpp   41 checks. No Qt, real filesystem.
```

The security-critical half has no Qt dependency on purpose, so it can be
built and checked with a compiler and nothing else:

```bash
cmake -S . -B build -DSTUDIO_CORE_ONLY=ON
cmake --build build && ctest --test-dir build
```

That is how the path escapes and the symlink escapes were found rather
than assumed, and it is what lets CI check them on a runner with no Qt.

## Server identity

`/hyperlink/endpoints` reports a fingerprint; Studio pins it on first
connect and compares it afterwards. A mismatch blocks the whole window —
not a toast — and the key is never sent. The reason is the same one
HyperLink has: the address Studio reaches can end up pointing at a
different machine, and the server *name* authenticates nothing, since
anything on a LAN or a tailnet can claim one.

The pinned fingerprint is shown in the sidebar, grouped in eights,
matching what `hypernix-t1 status` prints on the machine itself — so the
two can be compared by eye.

## Sub-bit models

Studio loads whatever the server loads, including `IQ0.9_L` and the
other HyperNix sub-bit tiers, because it is the server doing the
loading. If you want *llama.cpp* to read those files — for LM Studio, or
for `llama-cli` — see [`../native/ggml-hnx`](../native/ggml-hnx).

## State of it

The Qt half has not been compiled: there is no Qt in the environment this
was written in, so `HyperLinkClient`, `StudioBridge` and every `.qml`
file are unbuilt. They are written against the Qt 6.5 APIs and reviewed,
not verified. Expect to fix compile errors on a first build.

The core half — `ToolPolicy` and `ToolRunner`, which is where being wrong
costs someone their files — is built, tested and passing: 110 checks
across the two suites, run by `ctest` and by `tests/test_studio_core.py`
in the Python suite.
