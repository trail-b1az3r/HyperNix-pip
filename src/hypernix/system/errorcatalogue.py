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

from .errorcodes import ErrorCode, register

__all__ = ["ALL", "by_name"]


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


ALL: dict[str, ErrorCode] = {
    name: value for name, value in list(globals().items())
    if isinstance(value, ErrorCode)
}


def by_name(name: str) -> ErrorCode:
    """A code by its Python name, for a caller holding a string."""
    try:
        return ALL[name.upper()]
    except KeyError:
        raise KeyError(
            f"no error code named {name!r}. There are {len(ALL)}; "
            f"`codes_for(domain)` lists a domain's."
        ) from None
