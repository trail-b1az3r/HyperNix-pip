"""hypernix.config — Configuration management for HyperNix.

Provides the ``hnx config`` subcommand for setting persistent per-user
configuration such as the default model, API keys, and download preferences.

Config is stored at ``~/.hypernix/config.json``.

Subcommands
-----------
hnx config dmodel <model>            Set the default download/chat model.
                                     Accepts:
                                       * Any HuggingFace model ID   (org/name)
                                       * Any HuggingFace URL
                                       * OpenAI API key             (sk-...)
                                       * Anthropic API key          (sk-ant-...)
                                       * Google Gemini API key      (AIza...)

hnx config dmodel-experimental <m>  Set the default model to any local or
                                     remote model, bypassing the supported-
                                     model check. Accepts:
                                       * Local PyTorch folder (has config.json)
                                       * Local GGUF file    (.gguf extension)
                                       * Local MLX file     (.mlx extension)
                                       * HuggingFace model ID or URL

hnx config get <key>                 Print the current value of a config key.
hnx config list                      Print all current config values.
hnx config reset [<key>]             Reset one key (or all keys) to defaults.
hnx config path                      Print the path to the config file.
"""
from __future__ import annotations

import json
import os
import re
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Config file location + defaults
# ---------------------------------------------------------------------------

_CONFIG_DIR  = Path.home() / ".hypernix"
_CONFIG_FILE = _CONFIG_DIR / "config.json"

_DEFAULTS: dict[str, Any] = {
    "default_model":       None,
    "default_model_type":  None,   # "hf" | "openai" | "anthropic" | "gemini" | "local" | "experimental"
    "download_dir":        str(Path.home() / ".hypernix" / "models"),
    "hf_token":            None,
    "preferred_quant":     None,
    "auto_update":         True,
    "telemetry":           False,
    # v0.71.4b6: per-vendor cloud API keys, used by hyped_pro's real
    # provider dispatch (anthropic / openai / moonshot / dashscope) and by
    # the hyped-pro-gui settings dialog. Keyed by vendor id (see
    # ``hypernix.hyped_pro_core.PROVIDERS``). Never logged in full.
    "provider_keys":       {},
    # v0.71.5rc2: which HyperNix T1 API server hyped-pro's "t1-routed"
    # model talks to, and how that server's model_ids map onto local
    # catalog short names. Empty map = match by name only.
    "t1_api_url":          None,
    "t1_api_model_map":    {},
}

# Vendor -> env var name, used by /key, hyped_pro_core.py, and the GUI
# settings dialog so they never disagree about which env var wins.
PROVIDER_ENV_VARS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai":    "OPENAI_API_KEY",
    "moonshot":  "MOONSHOT_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
    "t1":        "HNX_T1_KEY",
    # v0.71.5rc2: distinct from "t1" — that key is checked by the local
    # Gatekeeper, this one is presented to a real T1 API server over HTTP.
    "t1api":     "HNX_T1_API_KEY",
}

# ---------------------------------------------------------------------------
# API key patterns
# ---------------------------------------------------------------------------

_OPENAI_RE    = re.compile(r"^sk-[A-Za-z0-9\-_]{20,}$")
_ANTHROPIC_RE = re.compile(r"^sk-ant-[A-Za-z0-9\-_]{20,}$")
_GEMINI_RE    = re.compile(r"^AIza[A-Za-z0-9\-_]{30,}$")
_HF_TOKEN_RE  = re.compile(r"^hf_[A-Za-z0-9]{20,}$")
_HF_URL_RE    = re.compile(r"https?://huggingface\.co/([^/]+/[^/?#]+)")
_HF_ID_RE     = re.compile(r"^[A-Za-z0-9_\-\.]+/[A-Za-z0-9_\-\.]+$")

# ---------------------------------------------------------------------------
# Supported model families (for dmodel validation)
# ---------------------------------------------------------------------------

_SUPPORTED_FAMILIES: tuple[str, ...] = (
    "nix", "hyper-nix", "hypernix", "qwen", "llama", "mistral",
    "gemma", "phi", "falcon", "mpt", "gpt", "deepseek", "internlm",
    "baichuan", "yi", "bloom", "opt", "pythia", "gptj", "neox",
    "flan", "t5", "bart", "bert", "roberta", "kimi", "fable",
)


def _is_supported_hf_model(model_id: str) -> bool:
    """Return True if the model ID looks like a known/supported HF model."""
    lower = model_id.lower()
    return any(f in lower for f in _SUPPORTED_FAMILIES)


# ---------------------------------------------------------------------------
# Config I/O
# ---------------------------------------------------------------------------

def _load_config() -> dict[str, Any]:
    """Load config from disk, returning defaults for missing keys."""
    if not _CONFIG_FILE.exists():
        return dict(_DEFAULTS)
    _tighten_permissions()
    try:
        with open(_CONFIG_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        # Merge with defaults so new keys always exist
        cfg = dict(_DEFAULTS)
        cfg.update(data)
        return cfg
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[config] Warning: could not read {_CONFIG_FILE}: {exc}", file=sys.stderr)
        return dict(_DEFAULTS)


#: Permissions for the config file and its directory.
#:
#: This file holds ``provider_keys`` — the user's Anthropic, OpenAI,
#: Moonshot and DashScope API keys — and ``hf_token``. It was written
#: with the default umask, which on every mainstream distribution is
#: 0644: world-readable, in a 0755 directory. Any other account on the
#: machine, and anything running as nobody, could read every API key the
#: user had set.
#:
#: Every other secret store in this package already did this right —
#: keymaster.py, t2keys.py, waiter/local_config.py and
#: hyperlink/identity.py all write 0600. This one was the exception.
_CONFIG_MODE = 0o600
_CONFIG_DIR_MODE = 0o700


def _save_config(cfg: dict[str, Any]) -> None:
    """Persist config to disk, readable only by its owner.

    Written to a temporary file and renamed, so an interrupted write
    leaves the previous config intact rather than a truncated one — the
    file holds the only copy of keys the user typed once, and half of a
    JSON object is not recoverable.

    The temporary file is created 0600 by ``os.open`` rather than chmod-ed
    afterwards: between an ``open()`` and a ``chmod()`` there is a window
    where the secret exists at the umask's permissions, and on a shared
    machine that window is the whole vulnerability.
    """
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(_CONFIG_DIR, _CONFIG_DIR_MODE)
    except OSError:
        # Windows has no POSIX modes, and a directory owned by someone
        # else is their business. Neither is a reason not to write.
        pass

    payload = json.dumps(cfg, indent=2, ensure_ascii=False)
    temporary = _CONFIG_FILE.with_name(_CONFIG_FILE.name + f".{os.getpid()}.tmp")
    try:
        handle = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _CONFIG_MODE
        )
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, _CONFIG_FILE)
    except OSError:
        with suppress(OSError):
            temporary.unlink()
        raise
    # os.replace keeps the *temporary* file's mode, which is already
    # 0600. Re-applying it covers the one case that is not: a
    # pre-existing config whose mode the rename did not change because
    # the platform's replace semantics differ.
    with suppress(OSError):
        os.chmod(_CONFIG_FILE, _CONFIG_MODE)


def _tighten_permissions() -> None:
    """Fix a config file that an older version wrote world-readable.

    Called on read rather than only on write. Somebody who set their keys
    six months ago and has not changed one since would otherwise keep the
    0644 file forever, and the upgrade that fixed this would not have
    fixed anything for them.
    """
    try:
        current = os.stat(_CONFIG_FILE).st_mode
    except OSError:
        return
    if current & 0o077:
        with suppress(OSError):
            os.chmod(_CONFIG_FILE, _CONFIG_MODE)
    with suppress(OSError):
        if os.stat(_CONFIG_DIR).st_mode & 0o077:
            os.chmod(_CONFIG_DIR, _CONFIG_DIR_MODE)


def get_config_value(key: str) -> Any:
    """Public API: get a single config value by key."""
    cfg = _load_config()
    if key not in cfg:
        raise KeyError(f"Unknown config key: {key!r}")
    return cfg[key]


def set_config_value(key: str, value: Any) -> None:
    """Public API: set a single config value by key."""
    cfg = _load_config()
    cfg[key] = value
    _save_config(cfg)


def get_default_model() -> str | None:
    """Public API: return the currently configured default model (or None)."""
    return _load_config().get("default_model")


def get_models_dir() -> Path:
    """Public API: return the unified models directory used across all modules."""
    import os
    env_dir = os.getenv("HYPERNIX_MODELS_DIR")
    if env_dir:
        p = Path(env_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
    cfg = _load_config()
    d_dir = cfg.get("download_dir") or str(Path.home() / ".hypernix" / "models")
    p = Path(d_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_hf_token() -> str | None:
    """Public API: return the HuggingFace token from environment or config."""
    import os
    env_tok = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if env_tok:
        return env_tok
    return _load_config().get("hf_token")


def get_provider_key(vendor: str) -> str | None:
    """Public API: return the API key for a cloud vendor.

    Resolution order: the vendor's environment variable (see
    ``PROVIDER_ENV_VARS``) first, then the persisted ``provider_keys`` entry
    in ``~/.hypernix/config.json``. Returns ``None`` if neither is set —
    callers must not fabricate a key or pretend a call succeeded.
    """
    import os
    env_var = PROVIDER_ENV_VARS.get(vendor)
    if env_var:
        env_val = os.getenv(env_var)
        if env_val:
            return env_val
    keys = _load_config().get("provider_keys") or {}
    return keys.get(vendor) or None


def set_provider_key(vendor: str, key: str) -> None:
    """Public API: persist an API key for a cloud vendor to disk.

    Stored under ``provider_keys.<vendor>`` in ``~/.hypernix/config.json``.
    Does not touch the process environment — callers that want the key
    usable immediately in-process should also set ``os.environ`` themselves.
    """
    cfg = _load_config()
    keys = dict(cfg.get("provider_keys") or {})
    keys[vendor] = key
    cfg["provider_keys"] = keys
    _save_config(cfg)


def clear_provider_key(vendor: str) -> None:
    """Public API: remove a persisted provider key (env var is untouched)."""
    cfg = _load_config()
    keys = dict(cfg.get("provider_keys") or {})
    keys.pop(vendor, None)
    cfg["provider_keys"] = keys
    _save_config(cfg)



# ---------------------------------------------------------------------------
# Model type detection
# ---------------------------------------------------------------------------

def _classify_model_input(arg: str) -> tuple[str, str, str]:
    """Classify a model argument and return (model_value, model_type, display_type).

    Possible types:
    * "openai"    — OpenAI API key (sk-...)
    * "anthropic" — Anthropic API key (sk-ant-...)
    * "gemini"    — Google Gemini API key (AIza...)
    * "hf_url"    — HuggingFace URL → extracted to model ID
    * "hf"        — HuggingFace model ID (org/name)
    * "unknown"   — doesn't match any known pattern
    """
    # API keys
    if _ANTHROPIC_RE.match(arg):
        return arg, "anthropic", "Anthropic API key"
    if _OPENAI_RE.match(arg):
        return arg, "openai", "OpenAI API key"
    if _GEMINI_RE.match(arg):
        return arg, "gemini", "Google Gemini API key"

    # HuggingFace URL → extract model ID
    m = _HF_URL_RE.match(arg)
    if m:
        model_id = m.group(1)
        return model_id, "hf", f"HuggingFace model ({model_id})"

    # HuggingFace model ID (org/name)
    if _HF_ID_RE.match(arg):
        return arg, "hf", f"HuggingFace model ({arg})"

    return arg, "unknown", f"unknown ({arg!r})"


def _classify_experimental_input(arg: str) -> tuple[str, str, str]:
    """Classify an experimental model argument.

    Experimental accepts anything including local paths, GGUF/MLX files, or
    any HF ID even if not in the supported families list.

    Returns (model_value, model_type, display_type).
    """
    p = Path(arg)
    if p.exists():
        if p.is_dir():
            # Check if it looks like a HF-style model folder
            has_config = (p / "config.json").exists()
            label = "HF snapshot directory" if has_config else "local directory"
            return str(p.resolve()), "local", label
        if p.suffix.lower() == ".gguf":
            return str(p.resolve()), "local_gguf", "local GGUF file"
        if p.suffix.lower() == ".mlx":
            return str(p.resolve()), "local_mlx", "local MLX file"
        return str(p.resolve()), "local", "local file"

    # Non-local: treat as HF model (experimental, no family check)
    m = _HF_URL_RE.match(arg)
    if m:
        model_id = m.group(1)
        return model_id, "hf_experimental", f"HuggingFace model (experimental, {model_id})"
    if _HF_ID_RE.match(arg):
        return arg, "hf_experimental", f"HuggingFace model (experimental, {arg})"

    return arg, "experimental", f"experimental ({arg!r})"


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _cmd_dmodel(args: list[str]) -> int:
    """Handle: hnx config dmodel <model>"""
    if not args:
        print("Usage: hnx config dmodel <hf-id|hf-url|openai-key|anthropic-key|gemini-key>",
              file=sys.stderr)
        return 1

    arg = args[0]
    model_value, model_type, display = _classify_model_input(arg)

    if model_type == "unknown":
        # Not an API key and not a recognised HF ID — give a helpful error
        print(
            f"[config] Error: {arg!r} does not look like a supported model.\n"
            f"  Accepted formats:\n"
            f"    HuggingFace model ID  : org/model-name\n"
            f"    HuggingFace URL       : https://huggingface.co/org/model-name\n"
            f"    OpenAI API key        : sk-...\n"
            f"    Anthropic API key     : sk-ant-...\n"
            f"    Google Gemini API key : AIza...\n\n"
            f"  For unsupported / local models use:\n"
            f"    hnx config dmodel-experimental <path-or-id>",
            file=sys.stderr,
        )
        return 1

    if model_type == "hf" and not _is_supported_hf_model(model_value):
        # HF ID found, but doesn't match any known family
        print(
            f"[config] Warning: {model_value!r} doesn't match any known HyperNix-supported\n"
            f"  model family.  Setting it anyway, but some features may not work.\n"
            f"  To bypass this check entirely use: hnx config dmodel-experimental",
            file=sys.stderr,
        )

    cfg = _load_config()
    cfg["default_model"]      = model_value
    cfg["default_model_type"] = model_type
    _save_config(cfg)

    print(f"[config] Default model set: {display}")
    print(f"         Value: {model_value}")
    print(f"         Type:  {model_type}")
    return 0


def _cmd_dmodel_experimental(args: list[str]) -> int:
    """Handle: hnx config dmodel-experimental <model>"""
    if not args:
        print(
            "Usage: hnx config dmodel-experimental "
            "<local-folder|local.gguf|local.mlx|hf-id|hf-url>",
            file=sys.stderr,
        )
        return 1

    arg = args[0]
    model_value, model_type, display = _classify_experimental_input(arg)

    cfg = _load_config()
    cfg["default_model"]      = model_value
    cfg["default_model_type"] = model_type
    _save_config(cfg)

    print(f"[config] ⚠  Experimental default model set: {display}")
    print(f"         Value: {model_value}")
    print(f"         Type:  {model_type}")
    print(
        "         Note: experimental models may not work with all HyperNix features."
    )
    return 0


def _cmd_get(args: list[str]) -> int:
    """Handle: hnx config get <key>"""
    if not args:
        print("Usage: hnx config get <key>", file=sys.stderr)
        return 1
    key = args[0]
    try:
        value = get_config_value(key)
    except KeyError:
        print(f"[config] Unknown key: {key!r}", file=sys.stderr)
        print(f"  Known keys: {', '.join(sorted(_DEFAULTS.keys()))}", file=sys.stderr)
        return 1
    print(f"{key} = {json.dumps(value)}")
    return 0


def _cmd_list(_args: list[str]) -> int:
    """Handle: hnx config list"""
    cfg = _load_config()
    max_key = max(len(k) for k in cfg)
    print(f"[config] Config file: {_CONFIG_FILE}")
    print()
    for k, v in sorted(cfg.items()):
        print(f"  {k:<{max_key}}  =  {json.dumps(v)}")
    return 0


def _cmd_reset(args: list[str]) -> int:
    """Handle: hnx config reset [<key>]"""
    cfg = _load_config()
    if args:
        key = args[0]
        if key not in _DEFAULTS:
            print(f"[config] Unknown key: {key!r}", file=sys.stderr)
            return 1
        cfg[key] = _DEFAULTS[key]
        _save_config(cfg)
        print(f"[config] Reset {key!r} to default ({json.dumps(_DEFAULTS[key])})")
    else:
        _save_config(dict(_DEFAULTS))
        print("[config] All settings reset to defaults.")
    return 0


def _cmd_path(_args: list[str]) -> int:
    """Handle: hnx config path"""
    print(str(_CONFIG_FILE))
    return 0


# ---------------------------------------------------------------------------
# Usage / help
# ---------------------------------------------------------------------------

_USAGE = """\
Usage: hnx config <subcommand> [arguments]

Subcommands:
  dmodel <model>              Set the default model.
                              Accepts HF model IDs, HF URLs, or API keys.
  dmodel-experimental <m>     Set the default model (no validation).
                              Accepts local folders, .gguf, .mlx, or any HF ID.
  get <key>                   Print a single config value.
  list                        Print all config values.
  reset [<key>]               Reset one key (or all) to defaults.
  path                        Print the config file path.

Examples:
  hnx config dmodel Qwen/Qwen3-14B
  hnx config dmodel https://huggingface.co/meta-llama/Llama-3.1-8B
  hnx config dmodel sk-abc123...
  hnx config dmodel AIzaSy...
  hnx config dmodel-experimental /home/user/my-llm-folder
  hnx config dmodel-experimental ~/models/llama.gguf
  hnx config get default_model
  hnx config list
  hnx config reset default_model
"""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def cli_main(raw: list[str]) -> int:
    """Entry point for ``hnx config`` / ``hypernix config``."""
    if not raw or raw[0] in ("-h", "--help", "help"):
        print(_USAGE)
        return 0

    subcmd, *rest = raw

    if subcmd == "dmodel":
        return _cmd_dmodel(rest)
    if subcmd == "dmodel-experimental":
        return _cmd_dmodel_experimental(rest)
    if subcmd == "get":
        return _cmd_get(rest)
    if subcmd == "list":
        return _cmd_list(rest)
    if subcmd == "reset":
        return _cmd_reset(rest)
    if subcmd == "path":
        return _cmd_path(rest)

    print(f"[config] Unknown subcommand: {subcmd!r}", file=sys.stderr)
    print(_USAGE, file=sys.stderr)
    return 1
