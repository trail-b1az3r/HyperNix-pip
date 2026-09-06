# HyperLink for iOS

Chat with the models on your own PC, from your phone — at home over
Wi-Fi, and anywhere else over Tailscale. Send photos, upload files and
code, switch models, and turn any Hugging Face GGUF link into a download
plan your PC can actually run.

The app is a client for the HyperNix **T1 API** (`hypernix.t1api`,
T1 v1.0.26.8.0.1). It has no model of its own and never will: inference
happens on the machine with the GPU, which is the whole point.

---

## What it does

| | |
|---|---|
| **Chat** | Streaming replies, token by token. Conversations live on the PC, so one thread continues across your phone, iPad and desktop. |
| **Photos** | Send a picture to a vision model. The server inlines it as a data URL; the projector a VLM needs is handled for you. |
| **Files and code** | Attach a source file and the server puts it in the model's context as a fenced block, filename and all. Replies render code as code, with a copy button. |
| **Models** | See what is loaded in LM Studio on the PC and switch between them per conversation. |
| **Hugging Face** | Paste a model page, a direct download link, or both. Split GGUFs are expanded to the full set of parts; vision projectors are included; a mismatch between the two links is reported rather than guessed at. |
| **Away from home** | The app tries every address the server advertises, Tailscale first, and keeps the one that answers. Nothing to switch when you leave the house. |

---

## Getting it on your phone

### 1. On the PC

```bash
pip install "hypernix[t1api]"
export T1_LMSTUDIO_URL=http://localhost:1234     # where LM Studio is
uvicorn "hypernix.t1api:create_app" --factory --host 0.0.0.0 --port 8000
```

In another terminal:

```bash
waiter hyperlink pair --label "my iPhone"
```

That prints a six-character code and the addresses the PC answers on.
The code is single-use and lasts ten minutes.

Check LM Studio is actually visible to the server first if you want to
save a step:

```bash
waiter lmstudio status
```

### 2. On the phone

Install the IPA (below), open it, and type in the address and the code.
That is the whole setup — the app asks the server for its other
addresses during pairing, so you never type a Tailscale name.

### Signing in without a pairing code

Pairing needs someone at the PC, because minting a code is an admin
operation. When the PC is not to hand — or a code expired mid-setup —
switch the pairing screen to **T2S key** and paste one instead:

```bash
gkey create -v v2short --scopes read,write
```

A T2S key is 26 typeable characters plus a prefix, deliberately limited
to reading and non-admin writing, which is what makes it safe to type
into a phone. Paste it exactly as printed: it is case-sensitive and
contains punctuation, and the app does not "clean it up" the way it
normalises a pairing code.

Two differences from pairing. The key is the credential, so there is no
device record on the server and the phone will not appear in `waiter
hyperlink devices`. And signing out just forgets the key — to cut off a
lost phone, revoke the key on the PC with `gkey revoke <key-id>`, which
kills it in both its T2S and T1 spellings.

A T2S key can never mint pairing codes for other phones. That is a
property of the format, not a permission you can grant it.

### 3. Away from home

Install [Tailscale](https://tailscale.com) on both the PC and the phone
and sign both into the same tailnet. Nothing in the app changes: the
Tailscale address is already in its list and it will be used
automatically the moment the LAN address stops answering.

---

## Installing the IPA

Every release attaches `HyperLink-<version>.ipa`
([Releases](https://github.com/trail-b1az3r/hypernix-pip/releases)),
and every CI run uploads one as a build artifact.

The IPA is **unsigned** unless the release was built with signing
secrets. Three ways to install it:

* **Sideloadly** or **AltStore** — sign with your own Apple ID and
  install over USB or Wi-Fi. A free Apple ID works; the app expires
  after seven days and is refreshed by re-signing.
* **Xcode** — open `ios/HyperLink.xcodeproj` (after generating it, see
  below), pick your own team under Signing & Capabilities, and run on a
  connected device.
* **An Apple Developer account** — set the repository secrets listed
  below and CI produces a properly signed, distributable build.

---

## Building it yourself

Requirements: macOS, Xcode 16 or newer (developed against **Xcode 27
beta 5** and the iOS 27 SDK), and [XcodeGen](https://github.com/yonaskolb/XcodeGen).

```bash
brew install xcodegen
cd ios
xcodegen generate            # writes HyperLink.xcodeproj
open HyperLink.xcodeproj
```

The `.xcodeproj` is generated rather than committed: a `pbxproj` is a
merge-conflict machine, and CI has to build from a clean checkout
without anyone having opened Xcode. `ios/project.yml` is the source of
truth — edit that, not the generated project.

Command line:

```bash
cd ios
xcodegen generate
xcodebuild test -project HyperLink.xcodeproj -scheme HyperLink \
  -destination 'platform=iOS Simulator,name=iPhone 16' CODE_SIGNING_ALLOWED=NO
```

### Deployment target

iOS **18.0**, and the app is developed and tested against the iOS 27
SDK. Nothing in it needs an API newer than 18, and a lower floor means
the same IPA installs on a phone that has not taken the beta —
"works on iOS 27 and newer" is satisfied by a target that also works on
18, and the reverse is not true. Raise
`options.deploymentTarget.iOS` in `project.yml` if you want to use
something newer.

### CI signing secrets

Set all four and `.github/workflows/ios.yml` produces a signed IPA
instead of an unsigned one. Leave them unset and the build still works.

| Secret | What it is |
|---|---|
| `IOS_DIST_CERTIFICATE_P12` | Base64 of your `.p12` distribution certificate |
| `IOS_DIST_CERTIFICATE_PASSWORD` | Its password |
| `IOS_PROVISIONING_PROFILE` | Base64 of the `.mobileprovision` |
| `IOS_TEAM_ID` | Your ten-character Apple team ID |
| `IOS_EXPORT_METHOD` | Optional — `ad-hoc` (default), `development`, `app-store`, `enterprise` |

---

## How it is put together

```
Sources/
  HyperLinkApp.swift        the App entry point
  Models/APITypes.swift     the wire types, mirroring t1api/schemas.py
  Networking/
    HyperLinkClient.swift   one actor; endpoint failover lives here
    SSEStream.swift         the streaming chat frames
    TokenStore.swift        the device token, in the Keychain
  Store/AppState.swift      one @Observable object, on the main actor
  Views/                    SwiftUI, one file per screen
```

Three decisions worth knowing before changing anything:

**Endpoint failover, not network detection.** The phone cannot reliably
know which network it is on, and asking iOS is racy. So the client keeps
a *list* of addresses in the order the server ranked them, tries them in
turn, and remembers the one that worked. A 404 stops failover (that
address is clearly a server); only a transport failure moves on.

**The streaming task belongs to the state, not the view.** A reply
arrives over several seconds and has to survive the user scrolling,
rotating, or switching tabs — none of which a view's lifetime does.
`AppState` owns the `Task`, which is also what makes cancellation
possible.

**Partial answers are the server's problem, and it handles them.** When
the connection drops mid-reply the server persists what it streamed, so
the app reloads history instead of keeping a half-message that exists
only on the phone and would vanish on the next refresh.

---

## Privacy

The app talks to one server: yours. There is no HyperNix cloud, no
telemetry, and no third party in the path. Conversations, images and
files are stored on your own machine, under `~/.hypernix/`.

The device token is a bearer credential for your PC. It is kept in the
Keychain with `kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly`, so it
never rides an iCloud backup to another device — a token identifies one
physical phone to the server, and revoking it should revoke exactly one
phone. Unpair from **Server → Unpair this device**, or from the PC with
`waiter hyperlink unpair <device_id>`.
