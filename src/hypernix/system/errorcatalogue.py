"""hypernix.system.errorcatalogue — every code this package can raise.

One file, on purpose. The value of a code is that you can search it and
land somewhere that says what it means, and that only holds while there
is one place to land. Modules import the names they need from here
rather than calling ``register`` themselves, so adding a code is a diff
against this list and a collision is a failure at import.

Numbering
---------
Numbers are grouped by thousand inside each domain, so a code's
neighbourhood is readable without a lookup:

* ``0xxxx`` — loading, starting, finding things
* ``1xxxx`` — memory, disk, cores, quota
* ``2xxxx`` — configuration and arguments
* ``3xxxx`` — data, files, formats
* ``4xxxx`` — the network and other processes
* ``5xxxx`` — permission and identity
* ``9xxxx`` — internal faults, the ones that are bugs

Deprecated modules are deliberately absent. Codes are a promise to keep
a meaning stable, and a module on its way out is the wrong place to make
one — ``pressure_cooker`` v1–v3 raise what they always raised.
"""
from __future__ import annotations

from .errorcodes import ErrorCode, alias, register

__all__ = ["ALL", "by_name", "coded", "HYPED_PRO_LEGACY", "HYPED_TOOL_CODES", "SANDBOX_CODES"]


def _r(domain, tier, number, kind, severity, explanation, remedy="") -> ErrorCode:
    return register(domain, tier, number, kind, severity, explanation, remedy)


# ---------------------------------------------------------------------------
# M — models: ovens, presets, snapshots, GGUF
# ---------------------------------------------------------------------------

MODEL_NOT_FOUND = _r(
    "M", 2, 15, "a", 3,
    "no model at that path or name",
    "`hypernix devices` lists what is on disk; check ~/.hypernix/models",
)
MODEL_UNREADABLE = _r(
    "M", 2, 3_010, "e", 3,
    "the model file is present but cannot be read as a model",
    "a truncated download is the usual cause — check the file size against the repo",
)
MODEL_TOO_LARGE = _r(
    "M", 2, 1_015, "c", 3,
    "the model does not fit in the memory available",
    "quantise it, or lower --gpu-layers to keep part of it in RAM",
)
ARCH_UNKNOWN = _r(
    "M", 3, 2_020, "a", 3,
    "no architecture preset by that name",
    "`hypernix oven --list-arch` prints the presets",
)
ARCH_MISMATCH = _r(
    "M", 3, 3_025, "e", 4,
    "the checkpoint's architecture is not the one the config declares",
    "the config's model_type decides which class loads the weights; they have to agree",
)
TOKENIZER_MISSING = _r(
    "M", 2, 30, "a", 2,
    "the snapshot has no tokenizer beside it",
    "pass --tokenizer-source, or copy tokenizer.json into the snapshot directory",
)

# ---------------------------------------------------------------------------
# R — the runtime and the runner
# ---------------------------------------------------------------------------

RUNTIME_NOT_BUILT = _r(
    "R", 4, 10, "b", 4,
    "there is no built llama.cpp for this machine",
    "`hypernix runtime build`, or point HYPERNIX_LLAMA_BIN at one",
)
RUNNER_NOT_LOADED = _r(
    "R", 3, 20, "a", 3,
    "no model is loaded",
    "`hypernix-t1 built-in-runner start`, or POST /runner/load",
)
RUNNER_START_FAILED = _r(
    "R", 3, 25, "d", 4,
    "the model server was started and did not become ready",
    "its last output is in the detail; a CUDA allocation failure is the usual cause",
)
RUNNER_PORT_TAKEN = _r(
    "R", 3, 4_030, "d", 3,
    "the port the runner wants is already in use",
    "something else is on it — `hypernix-t1 status` says whether it is ours",
)
RUNNER_BUSY = _r(
    "R", 3, 1_035, "c", 2,
    "every instance is answering something and the queue is full",
    "try again shortly, or raise T1_HYPERCHAT_INSTANCES if the machine has the cores",
)
RUNNER_DIED = _r(
    "R", 3, 40, "d", 4,
    "the model server exited while it was serving",
    "out of VRAM mid-request is the common cause; the exit code is in the detail",
)
TOOLCALL_UNKNOWN = _r(
    "R", 2, 2_050, "a", 2,
    "the model asked for a tool that does not exist",
    "it is a model mistake, not a caller one — the refusal names the tools that do",
)
TOOLCALL_BAD_ARGUMENTS = _r(
    "R", 2, 2_055, "a", 2,
    "the model's tool arguments do not match the tool's schema",
    "the detail names the field; the model is told, so it can correct itself",
)
TOOLCALL_REFUSED = _r(
    "R", 2, 5_060, "a", 3,
    "the model asked for a tool this caller may not use",
    "scopes decide; a tool the caller cannot call is hidden rather than offered",
)

# ---------------------------------------------------------------------------
# E — elements: hydrogen, natural gas, and the element modules
# ---------------------------------------------------------------------------

ELEMENT_NOT_FOUND = _r(
    "E", 2, 10, "a", 3,
    "no element by that symbol or name",
    "`hypernix elements list` prints what is installed",
)
ELEMENT_LOAD_FAILED = _r(
    "E", 2, 20, "d", 3,
    "the element was found and raised while loading",
    "an addon's import error is the addon's fault, not the oven's",
)
ELEMENT_EXPERIMENTAL = _r(
    "E", 2, 2_030, "b", 1,
    "this element is experimental and is not enabled by default",
    "rows 6 and 7 are experimental; pass --experimental or set allow_experimental",
)
ELEMENT_CONFLICT = _r(
    "E", 2, 2_035, "b", 3,
    "two elements claim the same symbol",
    "a user element cannot take a symbol a built-in already has",
)
ELEMENT_API_MISMATCH = _r(
    "E", 2, 40, "b", 3,
    "the element was built against a different hydrogen generation",
    "elements declare the generation they need; this one is not this one",
)
ELEMENT_PERMISSION = _r(
    "E", 1, 5_050, "a", 3,
    "the element asked for something it is not permitted to do",
    "elements declare their permissions up front and are held to them",
)

# ---------------------------------------------------------------------------
# S — the system: processes, hardware, the machine
# ---------------------------------------------------------------------------

PROCESS_PROTECTED = _r(
    "S", 1, 5_010, "a", 2,
    "that process is on the never-touch list and was left alone",
    "terminals, shells, Python and HyperNix's own processes are never limited",
)
PROCESS_GONE = _r(
    "S", 1, 20, "d", 1,
    "the process was gone before anything could be done to it",
    "not an error worth acting on — it exited on its own",
)
NO_PERMISSION = _r(
    "S", 1, 5_030, "a", 3,
    "the operating system refused; this needs more privilege than the process has",
)
CWD_GONE = _r(
    "S", 1, 40, "d", 2,
    "the working directory no longer exists",
    "the shell was left in a directory that has since been deleted — cd somewhere real",
)
NO_DISK_SPACE = _r(
    "S", 1, 1_050, "c", 4,
    "out of disk space",
)
PACKAGE_MISSING = _r(
    "S", 1, 60, "b", 3,
    "an optional Python package this needs is not installed",
    "pip install the package, or the hypernix extra, named in the message",
)

# ---------------------------------------------------------------------------
# T — the T1 API. One code per `T1ErrorCode` member, 1:1, so a search on
# the new code lands on exactly the meaning the old name had. The old
# names stay in the envelope as `code`; these travel beside them as
# `hx_code`. See `t1api.errors.HX_CODE_FOR`.
# ---------------------------------------------------------------------------

#: name -> (tier, number, kind, severity, explanation)
_T1_TABLE: dict[str, tuple[int, int, str, int, str]] = {
    "MODEL_NOT_SUPPORTED":        (2,    10, "a", 3, "that model is not served here"),
    "MODEL_QUOTA_EXHAUSTED":      (2,  1010, "c", 3, "the quota for that model is used up"),
    "MODEL_UNAVAILABLE":          (2,  4010, "d", 3, "the model is registered but its backend is not answering"),
    "AUTH_MISSING_CREDENTIALS":   (1,  5010, "a", 3, "no credential on the request"),
    "AUTH_INVALID_KEY":           (1,  5011, "a", 3, "the key is not one this server issued"),
    "AUTH_EXPIRED_KEY":           (1,  5012, "a", 3, "the key has expired"),
    "AUTH_REVOKED_KEY":           (1,  5013, "a", 3, "the key was revoked"),
    "AUTH_INVALID_TOKEN":         (1,  5014, "a", 3, "the token does not verify"),
    "AUTH_EXPIRED_TOKEN":         (1,  5015, "a", 2, "the token has expired; exchange the key for a new one"),
    "AUTH_INSUFFICIENT_SCOPE":    (1,  5020, "a", 3, "the credential is valid and lacks the scope this needs"),
    "AUTH_ADMIN_REQUIRED":        (1,  5021, "a", 3, "this needs an admin credential"),
    "RATE_LIMITED":               (1,  1020, "c", 2, "too many requests; Retry-After says when"),
    "QUOTA_EXCEEDED":             (1,  1021, "c", 3, "the account's quota is used up"),
    "NOT_SUPPORTED":              (1,  2030, "b", 2, "the endpoint exists and is turned off on this server"),
    "NOT_FOUND":                  (1,    40, "a", 3, "nothing by that id"),
    "VALIDATION_ERROR":           (1,  2050, "a", 3, "the request does not parse, or asks for something impossible"),
    "CONFLICT":                   (1,  3050, "e", 3, "the request conflicts with what is already stored"),
    "INTERNAL_ERROR":             (9, 99000, "f", 4, "internal fault in the T1 API"),
    "MODULE_NOT_FOUND":           (2,    60, "a", 3, "no module by that name"),
    "MODULE_ALREADY_EXISTS":      (2,  3060, "e", 3, "a module by that name is already stored"),
    "MODULE_UPLOAD_REJECTED":     (2,  3061, "e", 3, "the uploaded module failed validation"),
    "SERVER_NOT_FOUND":           (2,    70, "a", 3, "no registered server by that id"),
    "SERVER_UNTRUSTED":           (2,  5070, "a", 4, "that server is registered and not trusted"),
    "JOB_NOT_FOUND":              (2,    80, "a", 3, "no job by that id"),
    "JOB_NOT_CANCELLABLE":        (2,  2080, "a", 2, "the job is past the point where it can be cancelled"),
    "PATH_TRAVERSAL_REJECTED":    (1,  5090, "a", 4, "a path tried to leave the directory it was confined to"),
    "SSRF_BLOCKED":               (1,  5091, "a", 4, "a URL pointed somewhere the server may not reach"),
    "PAYMENT_TOKEN_INVALID":      (2,  5100, "a", 3, "the payment token does not verify"),
    "BILLING_PAYMENT_REQUIRED":   (2,  1100, "c", 3, "this needs a paid balance"),
    "BILLING_KEY_REFUSED":        (2,  5101, "a", 3, "billing refused this key"),
    "PAYMENT_TOKEN_ALREADY_REDEEMED": (2, 3100, "e", 3, "that payment token was already redeemed"),
    "INSUFFICIENT_BALANCE":       (2,  1101, "c", 3, "the balance does not cover this"),
    "ROUTING_EXHAUSTED":          (2,  4110, "d", 4, "every backend in the routing cascade failed"),
    "IP_BLOCKED":                 (1,  5120, "a", 4, "the caller's address is blocklisted"),
    "IP_NOT_ALLOWLISTED":         (1,  5121, "a", 3, "the caller's address is not on the allowlist"),
    "MTLS_REQUIRED":              (1,  5130, "a", 3, "this server requires a client certificate"),
    "MTLS_INVALID":               (1,  5131, "a", 4, "the client certificate does not verify"),
    "TRANSPORT_FAILED":           (1,  4140, "d", 3, "something the server depends on did not answer"),
    "CHECKSUM_MISMATCH":          (2,  3150, "e", 4, "the data does not match its checksum"),
    "PAYLOAD_TOO_LARGE":          (1,  1160, "c", 3, "the request body is over the limit"),
    "CONFIRMATION_REQUIRED":      (1,  2170, "a", 2, "this is destructive and needs explicit confirmation"),
    "CONFIG_INVALID":             (1,  2180, "b", 3, "the configuration does not parse or is inconsistent"),
}

T1_CODES: dict[str, ErrorCode] = {
    name: _r("T", tier, number, kind, severity, explanation)
    for name, (tier, number, kind, severity, explanation) in _T1_TABLE.items()
}

# ---------------------------------------------------------------------------
# I — interfaces: the CLIs and TUIs
# ---------------------------------------------------------------------------

UI_NO_BACKEND = _r(
    "I", 1, 10, "b", 3,
    "there is nothing to talk to — no server and no local model",
    "`hypernix-t1 start`, or put a .gguf in ~/.hypernix/models",
)
UI_CONFIG_INVALID = _r(
    "I", 1, 2_020, "b", 2,
    "a dots config file raised while it was being read",
    "the file is executed, so a syntax error in it is an error here",
)

# ---------------------------------------------------------------------------
# I — hyped-pro (0.72.8a4). Its own HP{B,C,G,O,T}-AREA-NNN names, moved in:
# each is a registered code now, and the old name is an alias of it, so a
# log line from before still explains (`hypernix errors explain HPC-CFG-001`).
# Tier 2 is the shared core and its bridge; tier 1 the TUIs and the GUI.
# ---------------------------------------------------------------------------

HYPED_CLOUD_NO_KEY = _r(
    "I", 2, 2_210, "b", 3,
    "no API key is set for this cloud provider",
    "`/key <vendor> <key>` in hyped-pro, or the vendor's own environment variable",
)
HYPED_CLOUD_HTTP = _r(
    "I", 2, 4_210, "d", 3,
    "the cloud provider's API call failed: the network, or a non-2xx answer",
    "the detail has the status; a 401 is the key, a 429 the provider's rate limit",
)
HYPED_CLOUD_BAD_REPLY = _r(
    "I", 2, 4_211, "d", 3,
    "the cloud provider answered with something that could not be read as a reply",
)
HYPED_CLOUD_TOOL_ROUNDS = _r(
    "I", 2, 1_210, "c", 3,
    "a cloud model kept calling tools past the round limit without answering",
    "ask more narrowly, or raise the limit with /rounds",
)
HYPED_LOCAL_MISSING = _r(
    "I", 2, 220, "d", 3,
    "the local model is not on disk and downloading it failed",
    "check the network and the repo id; `hnx download <repo>` shows the real error",
)
HYPED_LOCAL_INFERENCE = _r(
    "I", 2, 221, "d", 3,
    "local inference failed: out of memory, a dtype, or an architecture the runtime cannot run",
)
HYPED_LOCAL_TOOL_ROUNDS = _r(
    "I", 2, 1_211, "c", 3,
    "a local model kept calling tools past the round limit without answering",
    "ask more narrowly, or raise the limit with /rounds",
)
HYPED_DEPS_MISSING = _r(
    "I", 2, 2_230, "b", 3,
    "a package hyped-pro needs is not installed for the Python actually running it",
    "several Pythons on one machine is the usual cause; HYPED_PRO_PYTHON picks one",
)
HYPED_T1_KEY = _r(
    "I", 2, 5_240, "a", 3,
    "the T1 key is missing, or the local Gatekeeper refused it",
    "`/key t1 <key>`; `gkey verify <key>` says why a key is refused",
)
HYPED_T1API_NO_KEY = _r(
    "I", 2, 5_241, "b", 3,
    "no T1 API key is configured",
    "`/key t1api <key>`, or set T1_API_KEY",
)
HYPED_T1API_UNREACHABLE = _r(
    "I", 2, 4_240, "d", 3,
    "the T1 API could not be reached, or refused the call",
    "`hypernix-t1 status` on the server says whether it is up and where",
)
HYPED_T1API_UNRUNNABLE = _r(
    "I", 2, 4_241, "d", 3,
    "the T1 API routed to a model this client cannot run locally",
)
HYPED_UNKNOWN_MODEL = _r(
    "I", 2, 2_250, "a", 3,
    "no provider or model by that name",
    "`/models` lists them; a Hugging Face repo id or a .gguf path also works",
)
HYPED_NOODLE_START = _r(
    "I", 2, 4_260, "d", 3,
    "the Noodle swarm could not be started",
)
HYPED_NOODLE_CALL = _r(
    "I", 2, 4_261, "d", 3,
    "a Noodle request failed",
)
HYPED_BRIDGE_PROTOCOL = _r(
    "I", 2, 3_270, "a", 3,
    "the bridge was sent a line that is not a command it knows, or not JSON",
)
HYPED_BRIDGE_DOWN = _r(
    "I", 1, 280, "d", 4,
    "the Python bridge is not running",
    "restart hyped-pro; its log says why the bridge stopped",
)
HYPED_BRIDGE_UNCODED = _r(
    "I", 1, 4_280, "d", 3,
    "the bridge reported a failure without saying which",
)
HYPED_BRIDGE_SPAWN = _r(
    "I", 1, 281, "b", 4,
    "the Python bridge could not be started",
    "`pip install hypernix` for that Python, or set HYPED_PRO_PYTHON to one that has it",
)
HYPED_BRIDGE_EXITED = _r(
    "I", 1, 4_281, "d", 4,
    "the Python bridge exited while requests were waiting",
)
HYPED_BRIDGE_BAD_JSON = _r(
    "I", 1, 3_281, "e", 3,
    "the bridge wrote a line that is not JSON",
)
HYPED_BRIDGE_TIMEOUT = _r(
    "I", 1, 1_281, "c", 3,
    "the bridge did not answer in time",
    "it may still be working; /quit and restart if it stays stuck",
)
HYPED_CATALOG_LOAD = _r(
    "I", 1, 282, "d", 4,
    "the model and provider catalog could not be loaded at startup",
)
HYPED_CATALOG_BAD = _r(
    "I", 1, 3_282, "e", 4,
    "the model and provider catalog was malformed",
)
HYPED_GUI_LAUNCH = _r(
    "I", 1, 283, "d", 3,
    "the hyped-pro GUI could not be launched",
)
HYPED_GUI_NO_DISPLAY = _r(
    "I", 1, 2_290, "b", 4,
    "no usable display session was found",
    "run it inside a desktop session, or use hyped-pro in the terminal",
)
HYPED_GUI_NO_QT = _r(
    "I", 1, 2_291, "b", 2,
    "PySide6 is not installed; trying GTK4",
    "pip install 'hypernix[gui-qt]'",
)
HYPED_GUI_QT_INIT = _r(
    "I", 1, 290, "d", 3,
    "Qt6 failed to start under the chosen platform plugin",
)
HYPED_GUI_NO_GTK = _r(
    "I", 1, 2_292, "b", 2,
    "PyGObject / GTK4 is not installed",
    "pip install 'hypernix[gui-gtk]'",
)
HYPED_GUI_GTK_INIT = _r(
    "I", 1, 291, "d", 3,
    "GTK4 failed to start",
)
HYPED_GUI_NO_BACKEND = _r(
    "I", 1, 2_293, "b", 4,
    "neither Qt6 nor GTK4 is available, so there is no GUI to start",
    "pip install 'hypernix[gui-qt]' (or [gui-gtk])",
)
HYPED_GUI_CHAT = _r(
    "I", 1, 4_290, "d", 3,
    "a chat request from the GUI failed; the detail has the code underneath",
)
HYPED_GUI_DOWNLOAD = _r(
    "I", 1, 4_291, "d", 3,
    "a model download from the GUI failed; the detail has the code underneath",
)
HYPED_GUI_KEY = _r(
    "I", 1, 5_290, "d", 3,
    "a provider key could not be saved or read",
)

# hyped-pro's tools (files and git), which the model calls. Kind `a`
# throughout bar the few that are not the caller's: the model asked for
# something it can correct, and the refusal is fed back to it.
#: name -> (legacy name, number, kind, severity, explanation)
_HYPED_TOOL_TABLE: dict[str, tuple[str, int, str, int, str]] = {
    "HYPED_TOOL_OUTSIDE_WORKSPACE": ("TOOL-PATH-001", 5_300, "a", 3, "a tool's path resolves outside the workspace"),
    "HYPED_TOOL_IN_GIT_DIR":        ("TOOL-PATH-002", 5_301, "a", 3, "tools never write inside .git"),
    "HYPED_TOOL_CONSENT":           ("TOOL-CONSENT-001", 5_302, "a", 2, "the person declined, so the tool did not run"),
    "HYPED_TOOL_EXISTS":            ("TOOL-CREATE-001", 3_300, "a", 2, "the file to create already exists; edit it instead"),
    "HYPED_TOOL_EDIT_MISSING":      ("TOOL-EDIT-001", 300, "a", 2, "the file to edit does not exist; create it instead"),
    "HYPED_TOOL_EDIT_DIR":          ("TOOL-EDIT-002", 2_300, "a", 2, "the path to edit is a folder, not a file"),
    "HYPED_TOOL_EDIT_NO_MATCH":     ("TOOL-EDIT-003", 3_301, "a", 2, "the text to replace is not in the file as given"),
    "HYPED_TOOL_EDIT_AMBIGUOUS":    ("TOOL-EDIT-004", 3_302, "a", 2, "the text to replace occurs more than once; it must be unique"),
    "HYPED_TOOL_READ_MISSING":      ("TOOL-READ-001", 301, "a", 2, "the file to read does not exist"),
    "HYPED_TOOL_READ_DIR":          ("TOOL-READ-002", 2_301, "a", 2, "the path to read is a folder; list it instead"),
    "HYPED_TOOL_READ_TOO_LARGE":    ("TOOL-READ-003", 1_300, "c", 2, "the file is over the read limit; read a range of lines"),
    "HYPED_TOOL_READ_NOT_TEXT":     ("TOOL-READ-004", 3_303, "e", 2, "the file is not UTF-8 text"),
    "HYPED_TOOL_WRITE_DIR":         ("TOOL-WRITE-001", 2_302, "a", 2, "the path to write is a folder, not a file"),
    "HYPED_TOOL_WRITE_STALE":       ("TOOL-WRITE-002", 3_304, "e", 2, "the file changed on disk since it was read; read it again"),
    "HYPED_TOOL_DELETE_MISSING":    ("TOOL-DELETE-001", 302, "a", 2, "the file to delete does not exist"),
    "HYPED_TOOL_DELETE_DIR":        ("TOOL-DELETE-002", 2_303, "a", 2, "delete_file removes files, and that path is a folder"),
    "HYPED_TOOL_MOVE_MISSING":      ("TOOL-MOVE-001", 303, "a", 2, "the file to move does not exist"),
    "HYPED_TOOL_MOVE_OVERWRITE":    ("TOOL-MOVE-002", 3_305, "a", 2, "moving would overwrite a file that is already there"),
    "HYPED_TOOL_LIST_MISSING":      ("TOOL-LIST-001", 304, "a", 2, "the folder to list does not exist"),
    "HYPED_TOOL_LIST_FILE":         ("TOOL-LIST-002", 2_304, "a", 2, "the path to list is a file, not a folder"),
    "HYPED_TOOL_SEARCH_MISSING":    ("TOOL-SEARCH-001", 305, "a", 2, "the path to search does not exist"),
    "HYPED_TOOL_SEARCH_MODE":       ("TOOL-SEARCH-002", 2_305, "a", 2, "search mode is content or filename"),
    "HYPED_TOOL_SEARCH_REGEX":      ("TOOL-SEARCH-003", 2_306, "a", 2, "the search pattern is not a valid regular expression"),
    "HYPED_TOOL_UNKNOWN":           ("TOOL-CFG-001", 2_307, "a", 2, "the model asked for a tool hyped-pro does not have"),
    "HYPED_TOOL_BAD_ARGS":          ("TOOL-CFG-002", 2_308, "a", 2, "the tool's arguments are not an object of the right fields"),
    "HYPED_TOOL_FAILED":            ("TOOL-EXEC-001", 4_300, "d", 3, "the tool raised while it ran"),
    "HYPED_GIT_MISSING":            ("TOOL-GIT-001", 310, "b", 3, "git is not installed or not on PATH"),
    "HYPED_GIT_TIMEOUT":            ("TOOL-GIT-002", 1_310, "c", 3, "git took too long and was stopped"),
    "HYPED_GIT_NOT_A_REPO":         ("TOOL-GIT-003", 311, "a", 3, "the workspace is not inside a git repository"),
    "HYPED_GIT_FAILED":             ("TOOL-GIT-004", 4_310, "d", 3, "git ran and failed; its own words are in the detail"),
    "HYPED_GIT_BAD_NAME":           ("TOOL-GIT-005", 2_310, "a", 3, "that is not a branch, remote or revision hyped-pro passes to git"),
    "HYPED_GIT_NO_PATHS":           ("TOOL-GIT-006", 2_311, "a", 2, "the git command needs at least one path"),
    "HYPED_GIT_NO_MESSAGE":         ("TOOL-GIT-007", 2_312, "a", 2, "a commit needs a message"),
    "HYPED_GIT_NOTHING_STAGED":     ("TOOL-GIT-008", 2_313, "a", 2, "nothing is staged to commit"),
    "HYPED_GIT_UNKNOWN_OP":         ("TOOL-GIT-009", 2_314, "a", 3, "that is not a git operation hyped-pro knows"),
}

HYPED_TOOL_CODES: dict[str, ErrorCode] = {
    name: _r("I", 2, number, kind, severity, explanation)
    for name, (_legacy, number, kind, severity, explanation) in _HYPED_TOOL_TABLE.items()
}
globals().update(HYPED_TOOL_CODES)

#: Every hyped-pro name from before the registry -> its code now.
HYPED_PRO_LEGACY: dict[str, ErrorCode] = {
    "HPC-CLOUD-001": HYPED_CLOUD_NO_KEY,
    "HPC-CLOUD-002": HYPED_CLOUD_HTTP,
    "HPC-CLOUD-003": HYPED_CLOUD_BAD_REPLY,
    "HPC-CLOUD-004": HYPED_CLOUD_TOOL_ROUNDS,
    "HPC-LOCAL-001": HYPED_LOCAL_MISSING,
    "HPC-LOCAL-002": HYPED_LOCAL_INFERENCE,
    "HPC-LOCAL-003": HYPED_LOCAL_TOOL_ROUNDS,
    "HPC-DEPS-001": HYPED_DEPS_MISSING,
    "HPC-T1-001": HYPED_T1_KEY,
    "HPC-T1API-001": HYPED_T1API_NO_KEY,
    "HPC-T1API-002": HYPED_T1API_UNREACHABLE,
    "HPC-T1API-003": HYPED_T1API_UNRUNNABLE,
    "HPC-CFG-001": HYPED_UNKNOWN_MODEL,
    "HPB-NOODLE-001": HYPED_NOODLE_START,
    "HPB-NOODLE-002": HYPED_NOODLE_CALL,
    "HPB-PROTO-001": HYPED_BRIDGE_PROTOCOL,
    "HPO-BRIDGE-001": HYPED_BRIDGE_DOWN,
    "HPO-BRIDGE-002": HYPED_BRIDGE_UNCODED,
    "HPT-BRIDGE-001": HYPED_BRIDGE_SPAWN,
    "HPT-BRIDGE-002": HYPED_BRIDGE_EXITED,
    "HPT-BRIDGE-003": HYPED_BRIDGE_BAD_JSON,
    "HPT-BRIDGE-004": HYPED_BRIDGE_TIMEOUT,
    "HPT-CATALOG-001": HYPED_CATALOG_LOAD,
    "HPT-CATALOG-002": HYPED_CATALOG_BAD,
    "HPT-GUI-001": HYPED_GUI_LAUNCH,
    "HPT-T1API-001": HYPED_T1API_UNREACHABLE,
    "HPT-NOODLE-001": HYPED_NOODLE_CALL,
    "HPG-PLATFORM-001": HYPED_GUI_NO_DISPLAY,
    "HPG-QT-001": HYPED_GUI_NO_QT,
    "HPG-QT-002": HYPED_GUI_QT_INIT,
    "HPG-GTK-001": HYPED_GUI_NO_GTK,
    "HPG-GTK-002": HYPED_GUI_GTK_INIT,
    "HPG-BACKEND-001": HYPED_GUI_NO_BACKEND,
    "HPG-CHAT-001": HYPED_GUI_CHAT,
    "HPG-DOWNLOAD-001": HYPED_GUI_DOWNLOAD,
    "HPG-KEY-001": HYPED_GUI_KEY,
    **{legacy: HYPED_TOOL_CODES[name] for name, (legacy, *_rest) in _HYPED_TOOL_TABLE.items()},
}
# HPB-INTERNAL-001 is INTERNAL_INTERFACES, below.

# ---------------------------------------------------------------------------
# T — the T1 API beyond T1ErrorCode (0.72.8a4): remote MCP, the Messages
# API, /code sandboxes, datasets, /net, the counter. These travel in an
# error's `details.hx_code`; the envelope's own code stays T1ErrorCode's.
# ---------------------------------------------------------------------------

MCP_REMOTE_UNREACHABLE = _r(
    "T", 2, 4_200, "d", 3,
    "a remote MCP server did not answer",
    "GET /mcp/remote shows each server's last error; it is retried after a short wait",
)
MCP_REMOTE_PROTOCOL = _r(
    "T", 2, 3_200, "e", 3,
    "a remote MCP server answered with something that is not MCP",
    "check the URL ends at the server's MCP endpoint (often /mcp)",
)
MCP_REMOTE_AUTH = _r(
    "T", 2, 5_200, "a", 3,
    "a remote MCP server refused the credential it was given",
    "set the server's token, or its token_env variable, in mcp-servers.json",
)
MESSAGES_UNTRANSLATABLE = _r(
    "T", 2, 3_210, "a", 3,
    "a Messages API request could not be turned into a chat request",
    "the detail names the part; Claude Code sends nothing this refuses",
)
SANDBOX_BAD_PATH = _r(
    "T", 2, 5_220, "a", 3,
    "a path tried to leave the code sandbox",
)
SANDBOX_NETWORK_OFF = _r(
    "T", 2, 5_221, "b", 2,
    "this needs the network, which code sandboxes do not have",
    "an admin turns it on with /code/create/sandbox/perms/s2?:=on",
)
SANDBOX_NOT_A_REPO = _r(
    "T", 2, 220, "a", 3,
    "that folder of the sandbox is not a git repository",
    "init or clone one first",
)
SANDBOX_NO_GIT = _r(
    "T", 2, 221, "b", 3,
    "git is not installed on this server",
)
SANDBOX_TIMEOUT = _r(
    "T", 2, 1_220, "c", 3,
    "the sandbox's time limit ran out",
    "s3 sets the limit, up to 600 seconds",
)
SANDBOX_BAD_URL = _r(
    "T", 2, 2_220, "a", 3,
    "only https:// repositories are cloned",
)
SANDBOX_BAD_REF = _r(
    "T", 2, 2_221, "a", 3,
    "that is not a branch or commit name",
)
DATASET_UNREADABLE = _r(
    "T", 2, 3_230, "e", 3,
    "the dataset file could not be read as its format",
    "the detail names the line or column; JSONL wants one JSON object per line",
)
DATASET_NO_ROWS = _r(
    "T", 2, 3_231, "a", 1,
    "no rows match the part of the dataset that was chosen",
)
NET_REMOTE_RUN_OFF = _r(
    "T", 2, 5_250, "b", 3,
    "running programs on other machines is off on this server",
    "T1_NET_REMOTE_RUN=1 turns it on",
)
COUNTER_HEARTBEAT_REFUSED = _r(
    "T", 2, 2_240, "a", 2,
    "the counter refused a heartbeat that was not in the expected shape",
)

SANDBOX_NOT_FOUND = _r(
    "T", 2, 222, "a", 3,
    "no code sandbox by that id, or not one of yours",
)
SANDBOX_INTERPRETER_MISSING = _r(
    "T", 2, 223, "b", 3,
    "that language's interpreter is not installed on this server",
    "GET /code lists the installed languages",
)
SANDBOX_DISK_BUDGET = _r(
    "T", 2, 1_221, "c", 3,
    "the sandbox's disk budget is used up",
    "delete files, or an admin raises s5",
)
SANDBOX_TOO_MANY = _r(
    "T", 2, 1_222, "c", 3,
    "you already hold as many sandboxes as one caller may",
    "DELETE /code/sandbox/<id> one you no longer need",
)
SANDBOX_EXECUTE_OFF = _r(
    "T", 2, 5_222, "b", 3,
    "running code is off for sandboxes",
    "an admin turns it on with /code/create/sandbox/perms/s1?:=on",
)
SANDBOX_LANGUAGE_OFF = _r(
    "T", 2, 5_223, "b", 3,
    "that language is not allowed in sandboxes",
    "an admin adds it to s4",
)
SANDBOX_NETWORK_UNENFORCEABLE = _r(
    "T", 2, 2_222, "b", 3,
    "this machine cannot keep a run off the network, and the network is meant to be off",
    "an admin can accept running with the network on: /code/create/sandbox/perms/s2?:=on",
)
SANDBOX_REFUSED = _r(
    "T", 2, 2_223, "a", 3,
    "the code sandbox refused the request",
)

#: SandboxError's own names -> their codes.
SANDBOX_CODES: dict[str, ErrorCode] = {
    "invalid": SANDBOX_REFUSED,
    "not_found": SANDBOX_NOT_FOUND,
    "not_installed": SANDBOX_INTERPRETER_MISSING,
    "disk_budget": SANDBOX_DISK_BUDGET,
    "too_many": SANDBOX_TOO_MANY,
    "execute_off": SANDBOX_EXECUTE_OFF,
    "language_off": SANDBOX_LANGUAGE_OFF,
    "network_unenforceable": SANDBOX_NETWORK_UNENFORCEABLE,
    "bad_path": SANDBOX_BAD_PATH,
    "network_off": SANDBOX_NETWORK_OFF,
    "not_a_repo": SANDBOX_NOT_A_REPO,
    "no_git": SANDBOX_NO_GIT,
    "timeout": SANDBOX_TIMEOUT,
    "bad_url": SANDBOX_BAD_URL,
    "bad_ref": SANDBOX_BAD_REF,
}

# ---------------------------------------------------------------------------
# D — data: gather, scavenger, corpora
# ---------------------------------------------------------------------------

DATA_SOURCE_UNREACHABLE = _r(
    "D", 2, 4_010, "d", 3,
    "the site or dataset could not be reached",
)
DATA_BUDGET_SPENT = _r(
    "D", 2, 1_020, "c", 1,
    "the crawl stopped on its page or time budget rather than because it finished",
    "`stopped_because` on the result says which; the data is partial either way",
)
DATA_FORMAT_UNKNOWN = _r(
    "D", 2, 2_030, "a", 3,
    "no writer for that output format",
)

# ---------------------------------------------------------------------------
# Q — quantisation
# ---------------------------------------------------------------------------

QUANT_UNSUPPORTED = _r(
    "Q", 3, 10, "a", 3,
    "that quantisation format is not one this build can write",
)
QUANT_LAYER_REFUSED = _r(
    "Q", 3, 3_020, "e", 2,
    "a layer could not be quantised and was left at full precision",
    "reported rather than fatal — the file is still usable, just larger",
)

QUANT_CONVERTER_UNAVAILABLE = _r(
    "Q", 3, 4_030, "d", 3,
    "llama.cpp's converter could not be fetched from GitHub or PyPI",
    "set HNX_LLAMA_CPP to a llama.cpp checkout, or try again when the network is up",
)
QUANT_CONVERT_FAILED = _r(
    "Q", 3, 40, "d", 3,
    "converting the checkpoint to GGUF failed",
    "the converter's last lines are in the detail; an architecture it does not know is the usual cause",
)
QUANT_NEEDS_LLAMA_QUANTIZE = _r(
    "Q", 3, 2_050, "b", 3,
    "this quant type needs llama-quantize, which is not installed",
    "`hypernix runtime build`, or put llama-quantize on PATH",
)

# ---------------------------------------------------------------------------
# M — downloads and hnx0xv2image (0.72.8a4)
# ---------------------------------------------------------------------------

REPO_NOT_FOUND = _r(
    "M", 2, 4_040, "a", 3,
    "no Hugging Face repo by that id",
    "check the spelling; `hnx download` takes owner/name",
)
REPO_GATED = _r(
    "M", 2, 5_040, "a", 3,
    "the repo is gated or private",
    "accept its terms on huggingface.co and set HF_TOKEN",
)
REPO_NO_SUCH_QUANT = _r(
    "M", 2, 45, "a", 3,
    "the repo has no file for that quant",
    "the message lists the quants it has",
)
IMAGE_FUSE_VL_REFUSED = _r(
    "M", 3, 2_060, "a", 3,
    "that model already sees images; hnx0xv2image fuses with text-only models",
)
IMAGE_FUSE_FAMILY = _r(
    "M", 3, 2_061, "a", 3,
    "hnx0xv2image does not fuse with that model family",
    "it fuses with LFM2/LFM2.5, Qwen3.5 and HyperNix models",
)

# ---------------------------------------------------------------------------
# S — launch-script, prot and setup (0.72.8a4)
# ---------------------------------------------------------------------------

LAUNCH_REFUSED = _r(
    "S", 1, 2_070, "a", 3,
    "launch-script could not start that: a missing script, folder or interpreter",
)
LAUNCH_SUPERVISOR_FAILED = _r(
    "S", 1, 4_070, "d", 3,
    "systemd refused to start the job",
    "the detail has systemd-run's words; `--supervisor setsid` avoids systemd",
)
LAUNCH_OUTCOME_UNKNOWN = _r(
    "S", 1, 70, "d", 2,
    "the job ended without recording how",
    "it was killed outright; its log shows how far it got",
)
PROTECT_LOCK_NOT_RESTORED = _r(
    "S", 1, 4_080, "d", 3,
    "the screen-lock settings could not be put back",
    "`hnx prot restore` tries again",
)
SETUP_NO_PYTHON = _r(
    "S", 1, 2_090, "b", 4,
    "no Python 3.10 to 3.13 was found to run the T1 server with",
    "install one from your package manager, then run `hnx-t1 setup` again",
)
SETUP_INSTALL_FAILED = _r(
    "S", 1, 4_090, "d", 4,
    "installing HyperNix into the T1 server's own environment failed",
    "pip's last lines are in the detail; the network or a missing compiler is the usual cause",
)
SETUP_ANSWER_REFUSED = _r(
    "S", 1, 2_091, "a", 2,
    "that answer is not one the setup can use",
    "the question says what it accepts; Enter takes the default",
)

# ---------------------------------------------------------------------------
# X — training
# ---------------------------------------------------------------------------

TRAIN_DIVERGED = _r(
    "X", 4, 3_010, "e", 4,
    "the loss stopped being a number",
    "lower the learning rate, or check the data for a batch of NaNs",
)
TRAIN_NO_DATA = _r(
    "X", 4, 20, "a", 3,
    "the dataset is empty after filtering",
)

# ---------------------------------------------------------------------------
# Internal faults. Ours, in every domain that has one.
# ---------------------------------------------------------------------------

INTERNAL_MODELS = _r("M", 9, 99_000, "f", 4, "internal fault in the model layer")
INTERNAL_RUNTIME = _r("R", 9, 99_000, "f", 4, "internal fault in the runtime")
INTERNAL_ELEMENTS = _r("E", 9, 99_000, "f", 4, "internal fault in an element")
INTERNAL_SYSTEM = _r("S", 9, 99_000, "f", 4, "internal fault in the system layer")
INTERNAL_INTERFACES = _r("I", 9, 99_000, "f", 4, "internal fault in an interface")

HYPED_PRO_LEGACY["HPB-INTERNAL-001"] = INTERNAL_INTERFACES
for _old, _code in HYPED_PRO_LEGACY.items():
    alias(_old, _code)
del _old, _code


ALL: dict[str, ErrorCode] = {
    name: value for name, value in list(globals().items())
    if isinstance(value, ErrorCode)
}


def coded(code: ErrorCode | str) -> tuple[ErrorCode, str]:
    """``(code, legacy name)`` for a catalogue code or a name from before
    the registry, for the classes that still carry both (hyped-pro's)."""
    from .errorcodes import ALIASES, resolve

    found = code if isinstance(code, ErrorCode) else resolve(str(code))
    if isinstance(code, str) and code.strip() in ALIASES:
        return found, code.strip()
    legacy = next((old for old, new in ALIASES.items() if new == found.code), "")
    return found, legacy


def by_name(name: str) -> ErrorCode:
    """A code by its Python name, for a caller holding a string."""
    try:
        return ALL[name.upper()]
    except KeyError:
        raise KeyError(
            f"no error code named {name!r}. There are {len(ALL)}; "
            f"`codes_for(domain)` lists a domain's."
        ) from None
