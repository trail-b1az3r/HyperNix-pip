# Error codes

Every error HyperNix raises carries a code of the shape `L#-NNNNN.kS` — for
example `I2-02250.a3`. Search for the code; it lands here. Generated from
`hypernix.system.errorcatalogue` by `scripts/generate_error_codes.py`; do not
edit by hand.

```bash
hypernix errors explain I2-02250.a3   # what it means and what to do
hypernix errors explain HPC-CFG-001   # an old name works too
hypernix errors list I                # one domain
hypernix errors aliases               # every old name and its code
```

## Reading a code

| Part | Means |
|---|---|
| `L` | the domain: `M` models, `T` the t1 api, `D` data, `S` system, `I` interfaces, `Q` quantisation, `E` elements, `R` the runtime and runner, `X` training |
| `#` | the tier, 0–9: `1` is a surface people use, `9` a kernel or an internal fault |
| `NNNNN` | five digits, unique in the domain; the thousands say what kind of trouble (below) |
| `k` | whose problem it is: `a` the caller's — what was asked cannot be done; `b` configuration or environment; `c` not enough of something; `d` something outside HyperNix; `e` data that is present and wrong; `f` HyperNix's own fault — a bug |
| `S` | severity, 1–5: `1` notice, `2` warning, `3` error, `4` critical, `5` fatal |

The thousands: `0xxxx` loading and finding, `1xxxx` memory, disk, cores and quota,
`2xxxx` configuration and arguments, `3xxxx` data and formats, `4xxxx` the network
and other processes, `5xxxx` permission and identity, `9xxxx` internal faults.

There are **190** codes and **72** older names that still resolve.

## Contents

- [M — Models](#m--models)
- [T — The T1 API](#t--the-t1-api)
- [D — Data](#d--data)
- [S — System](#s--system)
- [I — Interfaces (hyped-pro, the TUIs)](#i--interfaces-hyped-pro-the-tuis)
- [Q — Quantisation](#q--quantisation)
- [E — Elements](#e--elements)
- [R — The runtime and runner](#r--the-runtime-and-runner)
- [X — Training](#x--training)
- [Older names](#older-names)

## M — Models

| Code | Severity | Meaning | What to do |
|---|---|---|---|
| `M2-00015.a3` | error | no model at that path or name | `hypernix devices` lists what is on disk; check ~/.hypernix/models |
| `M2-00030.a2` | warning | the snapshot has no tokenizer beside it | pass --tokenizer-source, or copy tokenizer.json into the snapshot directory |
| `M2-00045.a3` | error | the repo has no file for that quant | the message lists the quants it has |
| `M2-01015.c3` | error | the model does not fit in the memory available | quantise it, or lower --gpu-layers to keep part of it in RAM |
| `M3-02020.a3` | error | no architecture preset by that name | `hypernix oven --list-arch` prints the presets |
| `M3-02060.a3` | error | that model already sees images; hnx0xv2image fuses with text-only models |  |
| `M3-02061.a3` | error | hnx0xv2image does not fuse with that model family | it fuses with LFM2/LFM2.5, Qwen3.5 and HyperNix models |
| `M2-03010.e3` | error | the model file is present but cannot be read as a model | a truncated download is the usual cause — check the file size against the repo |
| `M3-03025.e4` | critical | the checkpoint's architecture is not the one the config declares | the config's model_type decides which class loads the weights; they have to agree |
| `M2-04040.a3` | error | no Hugging Face repo by that id | check the spelling; `hnx download` takes owner/name |
| `M2-05040.a3` | error | the repo is gated or private | accept its terms on huggingface.co and set HF_TOKEN |
| `M9-99000.f4` | critical | internal fault in the model layer |  |

## T — The T1 API

| Code | Severity | Meaning | What to do |
|---|---|---|---|
| `T2-00010.a3` | error | that model is not served here |  |
| `T1-00040.a3` | error | nothing by that id |  |
| `T2-00060.a3` | error | no module by that name |  |
| `T2-00070.a3` | error | no registered server by that id |  |
| `T2-00080.a3` | error | no job by that id |  |
| `T2-00220.a3` | error | that folder of the sandbox is not a git repository | init or clone one first |
| `T2-00221.b3` | error | git is not installed on this server |  |
| `T2-00222.a3` | error | no code sandbox by that id, or not one of yours |  |
| `T2-00223.b3` | error | that language's interpreter is not installed on this server | GET /code lists the installed languages |
| `T2-01010.c3` | error | the quota for that model is used up |  |
| `T1-01020.c2` | warning | too many requests; Retry-After says when |  |
| `T1-01021.c3` | error | the account's quota is used up |  |
| `T2-01100.c3` | error | this needs a paid balance |  |
| `T2-01101.c3` | error | the balance does not cover this |  |
| `T1-01160.c3` | error | the request body is over the limit |  |
| `T2-01220.c3` | error | the sandbox's time limit ran out | s3 sets the limit, up to 600 seconds |
| `T2-01221.c3` | error | the sandbox's disk budget is used up | delete files, or an admin raises s5 |
| `T2-01222.c3` | error | you already hold as many sandboxes as one caller may | DELETE /code/sandbox/<id> one you no longer need |
| `T1-02030.b2` | warning | the endpoint exists and is turned off on this server |  |
| `T1-02050.a3` | error | the request does not parse, or asks for something impossible |  |
| `T2-02080.a2` | warning | the job is past the point where it can be cancelled |  |
| `T1-02170.a2` | warning | this is destructive and needs explicit confirmation |  |
| `T1-02180.b3` | error | the configuration does not parse or is inconsistent |  |
| `T2-02220.a3` | error | only https:// repositories are cloned |  |
| `T2-02221.a3` | error | that is not a branch or commit name |  |
| `T2-02222.b3` | error | this machine cannot keep a run off the network, and the network is meant to be off | an admin can accept running with the network on: /code/create/sandbox/perms/s2?:=on |
| `T2-02223.a3` | error | the code sandbox refused the request |  |
| `T2-02240.a2` | warning | the counter refused a heartbeat that was not in the expected shape |  |
| `T1-03050.e3` | error | the request conflicts with what is already stored |  |
| `T2-03060.e3` | error | a module by that name is already stored |  |
| `T2-03061.e3` | error | the uploaded module failed validation |  |
| `T2-03100.e3` | error | that payment token was already redeemed |  |
| `T2-03150.e4` | critical | the data does not match its checksum |  |
| `T2-03200.e3` | error | a remote MCP server answered with something that is not MCP | check the URL ends at the server's MCP endpoint (often /mcp) |
| `T2-03210.a3` | error | a Messages API request could not be turned into a chat request | the detail names the part; Claude Code sends nothing this refuses |
| `T2-03230.e3` | error | the dataset file could not be read as its format | the detail names the line or column; JSONL wants one JSON object per line |
| `T2-03231.a1` | notice | no rows match the part of the dataset that was chosen |  |
| `T2-04010.d3` | error | the model is registered but its backend is not answering |  |
| `T2-04110.d4` | critical | every backend in the routing cascade failed |  |
| `T1-04140.d3` | error | something the server depends on did not answer |  |
| `T2-04200.d3` | error | a remote MCP server did not answer | GET /mcp/remote shows each server's last error; it is retried after a short wait |
| `T1-05010.a3` | error | no credential on the request |  |
| `T1-05011.a3` | error | the key is not one this server issued |  |
| `T1-05012.a3` | error | the key has expired |  |
| `T1-05013.a3` | error | the key was revoked |  |
| `T1-05014.a3` | error | the token does not verify |  |
| `T1-05015.a2` | warning | the token has expired; exchange the key for a new one |  |
| `T1-05020.a3` | error | the credential is valid and lacks the scope this needs |  |
| `T1-05021.a3` | error | this needs an admin credential |  |
| `T2-05070.a4` | critical | that server is registered and not trusted |  |
| `T1-05090.a4` | critical | a path tried to leave the directory it was confined to |  |
| `T1-05091.a4` | critical | a URL pointed somewhere the server may not reach |  |
| `T2-05100.a3` | error | the payment token does not verify |  |
| `T2-05101.a3` | error | billing refused this key |  |
| `T1-05120.a4` | critical | the caller's address is blocklisted |  |
| `T1-05121.a3` | error | the caller's address is not on the allowlist |  |
| `T1-05130.a3` | error | this server requires a client certificate |  |
| `T1-05131.a4` | critical | the client certificate does not verify |  |
| `T2-05200.a3` | error | a remote MCP server refused the credential it was given | set the server's token, or its token_env variable, in mcp-servers.json |
| `T2-05220.a3` | error | a path tried to leave the code sandbox |  |
| `T2-05221.b2` | warning | this needs the network, which code sandboxes do not have | an admin turns it on with /code/create/sandbox/perms/s2?:=on |
| `T2-05222.b3` | error | running code is off for sandboxes | an admin turns it on with /code/create/sandbox/perms/s1?:=on |
| `T2-05223.b3` | error | that language is not allowed in sandboxes | an admin adds it to s4 |
| `T2-05250.b3` | error | running programs on other machines is off on this server | T1_NET_REMOTE_RUN=1 turns it on |
| `T9-99000.f4` | critical | internal fault in the T1 API |  |

## D — Data

| Code | Severity | Meaning | What to do |
|---|---|---|---|
| `D2-01020.c1` | notice | the crawl stopped on its page or time budget rather than because it finished | `stopped_because` on the result says which; the data is partial either way |
| `D2-02030.a3` | error | no writer for that output format |  |
| `D2-04010.d3` | error | the site or dataset could not be reached |  |

## S — System

| Code | Severity | Meaning | What to do |
|---|---|---|---|
| `S1-00020.d1` | notice | the process was gone before anything could be done to it | not an error worth acting on — it exited on its own |
| `S1-00040.d2` | warning | the working directory no longer exists | the shell was left in a directory that has since been deleted — cd somewhere real |
| `S1-00060.b3` | error | an optional Python package this needs is not installed | pip install the package, or the hypernix extra, named in the message |
| `S1-00070.d2` | warning | the job ended without recording how | it was killed outright; its log shows how far it got |
| `S1-01050.c4` | critical | out of disk space |  |
| `S1-02070.a3` | error | launch-script could not start that: a missing script, folder or interpreter |  |
| `S1-02090.b4` | critical | no Python 3.10 to 3.13 was found to run the T1 server with | install one from your package manager, then run `hnx-t1 setup` again |
| `S1-02091.a2` | warning | that answer is not one the setup can use | the question says what it accepts; Enter takes the default |
| `S1-04070.d3` | error | systemd refused to start the job | the detail has systemd-run's words; `--supervisor setsid` avoids systemd |
| `S1-04080.d3` | error | the screen-lock settings could not be put back | `hnx prot restore` tries again |
| `S1-04090.d4` | critical | installing HyperNix into the T1 server's own environment failed | pip's last lines are in the detail; the network or a missing compiler is the usual cause |
| `S1-05010.a2` | warning | that process is on the never-touch list and was left alone | terminals, shells, Python and HyperNix's own processes are never limited |
| `S1-05030.a3` | error | the operating system refused; this needs more privilege than the process has |  |
| `S9-99000.f4` | critical | internal fault in the system layer |  |

## I — Interfaces (hyped-pro, the TUIs)

| Code | Severity | Meaning | What to do |
|---|---|---|---|
| `I1-00010.b3` | error | there is nothing to talk to — no server and no local model | `hypernix-t1 start`, or put a .gguf in ~/.hypernix/models |
| `I2-00220.d3` | error | the local model is not on disk and downloading it failed (was `HPC-LOCAL-001`) | check the network and the repo id; `hnx download <repo>` shows the real error |
| `I2-00221.d3` | error | local inference failed: out of memory, a dtype, or an architecture the runtime cannot run (was `HPC-LOCAL-002`) |  |
| `I1-00280.d4` | critical | the Python bridge is not running (was `HPO-BRIDGE-001`) | restart hyped-pro; its log says why the bridge stopped |
| `I1-00281.b4` | critical | the Python bridge could not be started (was `HPT-BRIDGE-001`) | `pip install hypernix` for that Python, or set HYPED_PRO_PYTHON to one that has it |
| `I1-00282.d4` | critical | the model and provider catalog could not be loaded at startup (was `HPT-CATALOG-001`) |  |
| `I1-00283.d3` | error | the hyped-pro GUI could not be launched (was `HPT-GUI-001`) |  |
| `I1-00290.d3` | error | Qt6 failed to start under the chosen platform plugin (was `HPG-QT-002`) |  |
| `I1-00291.d3` | error | GTK4 failed to start (was `HPG-GTK-002`) |  |
| `I2-00300.a2` | warning | the file to edit does not exist; create it instead (was `TOOL-EDIT-001`) |  |
| `I2-00301.a2` | warning | the file to read does not exist (was `TOOL-READ-001`) |  |
| `I2-00302.a2` | warning | the file to delete does not exist (was `TOOL-DELETE-001`) |  |
| `I2-00303.a2` | warning | the file to move does not exist (was `TOOL-MOVE-001`) |  |
| `I2-00304.a2` | warning | the folder to list does not exist (was `TOOL-LIST-001`) |  |
| `I2-00305.a2` | warning | the path to search does not exist (was `TOOL-SEARCH-001`) |  |
| `I2-00310.b3` | error | git is not installed or not on PATH (was `TOOL-GIT-001`) |  |
| `I2-00311.a3` | error | the workspace is not inside a git repository (was `TOOL-GIT-003`) |  |
| `I2-01210.c3` | error | a cloud model kept calling tools past the round limit without answering (was `HPC-CLOUD-004`) | ask more narrowly, or raise the limit with /rounds |
| `I2-01211.c3` | error | a local model kept calling tools past the round limit without answering (was `HPC-LOCAL-003`) | ask more narrowly, or raise the limit with /rounds |
| `I1-01281.c3` | error | the bridge did not answer in time (was `HPT-BRIDGE-004`) | it may still be working; /quit and restart if it stays stuck |
| `I2-01300.c2` | warning | the file is over the read limit; read a range of lines (was `TOOL-READ-003`) |  |
| `I2-01310.c3` | error | git took too long and was stopped (was `TOOL-GIT-002`) |  |
| `I1-02020.b2` | warning | a dots config file raised while it was being read | the file is executed, so a syntax error in it is an error here |
| `I2-02210.b3` | error | no API key is set for this cloud provider (was `HPC-CLOUD-001`) | `/key <vendor> <key>` in hyped-pro, or the vendor's own environment variable |
| `I2-02230.b3` | error | a package hyped-pro needs is not installed for the Python actually running it (was `HPC-DEPS-001`) | several Pythons on one machine is the usual cause; HYPED_PRO_PYTHON picks one |
| `I2-02250.a3` | error | no provider or model by that name (was `HPC-CFG-001`) | `/models` lists them; a Hugging Face repo id or a .gguf path also works |
| `I1-02290.b4` | critical | no usable display session was found (was `HPG-PLATFORM-001`) | run it inside a desktop session, or use hyped-pro in the terminal |
| `I1-02291.b2` | warning | PySide6 is not installed; trying GTK4 (was `HPG-QT-001`) | pip install 'hypernix[gui-qt]' |
| `I1-02292.b2` | warning | PyGObject / GTK4 is not installed (was `HPG-GTK-001`) | pip install 'hypernix[gui-gtk]' |
| `I1-02293.b4` | critical | neither Qt6 nor GTK4 is available, so there is no GUI to start (was `HPG-BACKEND-001`) | pip install 'hypernix[gui-qt]' (or [gui-gtk]) |
| `I2-02300.a2` | warning | the path to edit is a folder, not a file (was `TOOL-EDIT-002`) |  |
| `I2-02301.a2` | warning | the path to read is a folder; list it instead (was `TOOL-READ-002`) |  |
| `I2-02302.a2` | warning | the path to write is a folder, not a file (was `TOOL-WRITE-001`) |  |
| `I2-02303.a2` | warning | delete_file removes files, and that path is a folder (was `TOOL-DELETE-002`) |  |
| `I2-02304.a2` | warning | the path to list is a file, not a folder (was `TOOL-LIST-002`) |  |
| `I2-02305.a2` | warning | search mode is content or filename (was `TOOL-SEARCH-002`) |  |
| `I2-02306.a2` | warning | the search pattern is not a valid regular expression (was `TOOL-SEARCH-003`) |  |
| `I2-02307.a2` | warning | the model asked for a tool hyped-pro does not have (was `TOOL-CFG-001`) |  |
| `I2-02308.a2` | warning | the tool's arguments are not an object of the right fields (was `TOOL-CFG-002`) |  |
| `I2-02310.a3` | error | that is not a branch, remote or revision hyped-pro passes to git (was `TOOL-GIT-005`) |  |
| `I2-02311.a2` | warning | the git command needs at least one path (was `TOOL-GIT-006`) |  |
| `I2-02312.a2` | warning | a commit needs a message (was `TOOL-GIT-007`) |  |
| `I2-02313.a2` | warning | nothing is staged to commit (was `TOOL-GIT-008`) |  |
| `I2-02314.a3` | error | that is not a git operation hyped-pro knows (was `TOOL-GIT-009`) |  |
| `I2-03270.a3` | error | the bridge was sent a line that is not a command it knows, or not JSON (was `HPB-PROTO-001`) |  |
| `I1-03281.e3` | error | the bridge wrote a line that is not JSON (was `HPT-BRIDGE-003`) |  |
| `I1-03282.e4` | critical | the model and provider catalog was malformed (was `HPT-CATALOG-002`) |  |
| `I2-03300.a2` | warning | the file to create already exists; edit it instead (was `TOOL-CREATE-001`) |  |
| `I2-03301.a2` | warning | the text to replace is not in the file as given (was `TOOL-EDIT-003`) |  |
| `I2-03302.a2` | warning | the text to replace occurs more than once; it must be unique (was `TOOL-EDIT-004`) |  |
| `I2-03303.e2` | warning | the file is not UTF-8 text (was `TOOL-READ-004`) |  |
| `I2-03304.e2` | warning | the file changed on disk since it was read; read it again (was `TOOL-WRITE-002`) |  |
| `I2-03305.a2` | warning | moving would overwrite a file that is already there (was `TOOL-MOVE-002`) |  |
| `I2-04210.d3` | error | the cloud provider's API call failed: the network, or a non-2xx answer (was `HPC-CLOUD-002`) | the detail has the status; a 401 is the key, a 429 the provider's rate limit |
| `I2-04211.d3` | error | the cloud provider answered with something that could not be read as a reply (was `HPC-CLOUD-003`) |  |
| `I2-04240.d3` | error | the T1 API could not be reached, or refused the call (was `HPC-T1API-002`, `HPT-T1API-001`) | `hypernix-t1 status` on the server says whether it is up and where |
| `I2-04241.d3` | error | the T1 API routed to a model this client cannot run locally (was `HPC-T1API-003`) |  |
| `I2-04260.d3` | error | the Noodle swarm could not be started (was `HPB-NOODLE-001`) |  |
| `I2-04261.d3` | error | a Noodle request failed (was `HPB-NOODLE-002`, `HPT-NOODLE-001`) |  |
| `I1-04280.d3` | error | the bridge reported a failure without saying which (was `HPO-BRIDGE-002`) |  |
| `I1-04281.d4` | critical | the Python bridge exited while requests were waiting (was `HPT-BRIDGE-002`) |  |
| `I1-04290.d3` | error | a chat request from the GUI failed; the detail has the code underneath (was `HPG-CHAT-001`) |  |
| `I1-04291.d3` | error | a model download from the GUI failed; the detail has the code underneath (was `HPG-DOWNLOAD-001`) |  |
| `I2-04300.d3` | error | the tool raised while it ran (was `TOOL-EXEC-001`) |  |
| `I2-04310.d3` | error | git ran and failed; its own words are in the detail (was `TOOL-GIT-004`) |  |
| `I2-05240.a3` | error | the T1 key is missing, or the local Gatekeeper refused it (was `HPC-T1-001`) | `/key t1 <key>`; `gkey verify <key>` says why a key is refused |
| `I2-05241.b3` | error | no T1 API key is configured (was `HPC-T1API-001`) | `/key t1api <key>`, or set T1_API_KEY |
| `I1-05290.d3` | error | a provider key could not be saved or read (was `HPG-KEY-001`) |  |
| `I2-05300.a3` | error | a tool's path resolves outside the workspace (was `TOOL-PATH-001`) |  |
| `I2-05301.a3` | error | tools never write inside .git (was `TOOL-PATH-002`) |  |
| `I2-05302.a2` | warning | the person declined, so the tool did not run (was `TOOL-CONSENT-001`) |  |
| `I9-99000.f4` | critical | internal fault in an interface (was `HPB-INTERNAL-001`) |  |

## Q — Quantisation

| Code | Severity | Meaning | What to do |
|---|---|---|---|
| `Q3-00010.a3` | error | that quantisation format is not one this build can write |  |
| `Q3-00040.d3` | error | converting the checkpoint to GGUF failed | the converter's last lines are in the detail; an architecture it does not know is the usual cause |
| `Q3-02050.b3` | error | this quant type needs llama-quantize, which is not installed | `hypernix runtime build`, or put llama-quantize on PATH |
| `Q3-03020.e2` | warning | a layer could not be quantised and was left at full precision | reported rather than fatal — the file is still usable, just larger |
| `Q3-04030.d3` | error | llama.cpp's converter could not be fetched from GitHub or PyPI | set HNX_LLAMA_CPP to a llama.cpp checkout, or try again when the network is up |

## E — Elements

| Code | Severity | Meaning | What to do |
|---|---|---|---|
| `E2-00010.a3` | error | no element by that symbol or name | `hypernix elements list` prints what is installed |
| `E2-00020.d3` | error | the element was found and raised while loading | an addon's import error is the addon's fault, not the oven's |
| `E2-00040.b3` | error | the element was built against a different hydrogen generation | elements declare the generation they need; this one is not this one |
| `E2-02030.b1` | notice | this element is experimental and is not enabled by default | rows 6 and 7 are experimental; pass --experimental or set allow_experimental |
| `E2-02035.b3` | error | two elements claim the same symbol | a user element cannot take a symbol a built-in already has |
| `E1-05050.a3` | error | the element asked for something it is not permitted to do | elements declare their permissions up front and are held to them |
| `E9-99000.f4` | critical | internal fault in an element |  |

## R — The runtime and runner

| Code | Severity | Meaning | What to do |
|---|---|---|---|
| `R4-00010.b4` | critical | there is no built llama.cpp for this machine | `hypernix runtime build`, or point HYPERNIX_LLAMA_BIN at one |
| `R3-00020.a3` | error | no model is loaded | `hypernix-t1 built-in-runner start`, or POST /runner/load |
| `R3-00025.d4` | critical | the model server was started and did not become ready | its last output is in the detail; a CUDA allocation failure is the usual cause |
| `R3-00040.d4` | critical | the model server exited while it was serving | out of VRAM mid-request is the common cause; the exit code is in the detail |
| `R3-01035.c2` | warning | every instance is answering something and the queue is full | try again shortly, or raise T1_HYPERCHAT_INSTANCES if the machine has the cores |
| `R2-02050.a2` | warning | the model asked for a tool that does not exist | it is a model mistake, not a caller one — the refusal names the tools that do |
| `R2-02055.a2` | warning | the model's tool arguments do not match the tool's schema | the detail names the field; the model is told, so it can correct itself |
| `R3-04030.d3` | error | the port the runner wants is already in use | something else is on it — `hypernix-t1 status` says whether it is ours |
| `R2-05060.a3` | error | the model asked for a tool this caller may not use | scopes decide; a tool the caller cannot call is hidden rather than offered |
| `R9-99000.f4` | critical | internal fault in the runtime |  |

## X — Training

| Code | Severity | Meaning | What to do |
|---|---|---|---|
| `X4-00020.a3` | error | the dataset is empty after filtering |  |
| `X4-03010.e4` | critical | the loss stopped being a number | lower the learning rate, or check the data for a batch of NaNs |

## Older names

hyped-pro had its own names until 0.72.8a4. Each is an alias of a registered code
now: `explain` answers for both, and hyped-pro's bridge sends the old name beside
the new one as `legacy_code`.

| Old name | Code |
|---|---|
| `HPB-INTERNAL-001` | `I9-99000.f4` |
| `HPB-NOODLE-001` | `I2-04260.d3` |
| `HPB-NOODLE-002` | `I2-04261.d3` |
| `HPB-PROTO-001` | `I2-03270.a3` |
| `HPC-CFG-001` | `I2-02250.a3` |
| `HPC-CLOUD-001` | `I2-02210.b3` |
| `HPC-CLOUD-002` | `I2-04210.d3` |
| `HPC-CLOUD-003` | `I2-04211.d3` |
| `HPC-CLOUD-004` | `I2-01210.c3` |
| `HPC-DEPS-001` | `I2-02230.b3` |
| `HPC-LOCAL-001` | `I2-00220.d3` |
| `HPC-LOCAL-002` | `I2-00221.d3` |
| `HPC-LOCAL-003` | `I2-01211.c3` |
| `HPC-T1-001` | `I2-05240.a3` |
| `HPC-T1API-001` | `I2-05241.b3` |
| `HPC-T1API-002` | `I2-04240.d3` |
| `HPC-T1API-003` | `I2-04241.d3` |
| `HPG-BACKEND-001` | `I1-02293.b4` |
| `HPG-CHAT-001` | `I1-04290.d3` |
| `HPG-DOWNLOAD-001` | `I1-04291.d3` |
| `HPG-GTK-001` | `I1-02292.b2` |
| `HPG-GTK-002` | `I1-00291.d3` |
| `HPG-KEY-001` | `I1-05290.d3` |
| `HPG-PLATFORM-001` | `I1-02290.b4` |
| `HPG-QT-001` | `I1-02291.b2` |
| `HPG-QT-002` | `I1-00290.d3` |
| `HPO-BRIDGE-001` | `I1-00280.d4` |
| `HPO-BRIDGE-002` | `I1-04280.d3` |
| `HPT-BRIDGE-001` | `I1-00281.b4` |
| `HPT-BRIDGE-002` | `I1-04281.d4` |
| `HPT-BRIDGE-003` | `I1-03281.e3` |
| `HPT-BRIDGE-004` | `I1-01281.c3` |
| `HPT-CATALOG-001` | `I1-00282.d4` |
| `HPT-CATALOG-002` | `I1-03282.e4` |
| `HPT-GUI-001` | `I1-00283.d3` |
| `HPT-NOODLE-001` | `I2-04261.d3` |
| `HPT-T1API-001` | `I2-04240.d3` |
| `TOOL-CFG-001` | `I2-02307.a2` |
| `TOOL-CFG-002` | `I2-02308.a2` |
| `TOOL-CONSENT-001` | `I2-05302.a2` |
| `TOOL-CREATE-001` | `I2-03300.a2` |
| `TOOL-DELETE-001` | `I2-00302.a2` |
| `TOOL-DELETE-002` | `I2-02303.a2` |
| `TOOL-EDIT-001` | `I2-00300.a2` |
| `TOOL-EDIT-002` | `I2-02300.a2` |
| `TOOL-EDIT-003` | `I2-03301.a2` |
| `TOOL-EDIT-004` | `I2-03302.a2` |
| `TOOL-EXEC-001` | `I2-04300.d3` |
| `TOOL-GIT-001` | `I2-00310.b3` |
| `TOOL-GIT-002` | `I2-01310.c3` |
| `TOOL-GIT-003` | `I2-00311.a3` |
| `TOOL-GIT-004` | `I2-04310.d3` |
| `TOOL-GIT-005` | `I2-02310.a3` |
| `TOOL-GIT-006` | `I2-02311.a2` |
| `TOOL-GIT-007` | `I2-02312.a2` |
| `TOOL-GIT-008` | `I2-02313.a2` |
| `TOOL-GIT-009` | `I2-02314.a3` |
| `TOOL-LIST-001` | `I2-00304.a2` |
| `TOOL-LIST-002` | `I2-02304.a2` |
| `TOOL-MOVE-001` | `I2-00303.a2` |
| `TOOL-MOVE-002` | `I2-03305.a2` |
| `TOOL-PATH-001` | `I2-05300.a3` |
| `TOOL-PATH-002` | `I2-05301.a3` |
| `TOOL-READ-001` | `I2-00301.a2` |
| `TOOL-READ-002` | `I2-02301.a2` |
| `TOOL-READ-003` | `I2-01300.c2` |
| `TOOL-READ-004` | `I2-03303.e2` |
| `TOOL-SEARCH-001` | `I2-00305.a2` |
| `TOOL-SEARCH-002` | `I2-02305.a2` |
| `TOOL-SEARCH-003` | `I2-02306.a2` |
| `TOOL-WRITE-001` | `I2-02302.a2` |
| `TOOL-WRITE-002` | `I2-03304.e2` |
