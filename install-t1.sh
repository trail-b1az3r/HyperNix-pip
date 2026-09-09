#!/usr/bin/env bash
#
# install-t1.sh — interactive installer and setup for the HyperNix T1 API.
#
# Installs the server, asks what kind of deployment you want, and writes a
# configuration that matches the answers. Nothing is guessed silently: every
# question states what the choice does and what the safe default is, and the
# summary at the end lists what was written and where.
#
#   curl -fsSL .../install-t1.sh | bash          # interactive
#   ./install-t1.sh --non-interactive            # defaults, no prompts
#   ./install-t1.sh --dry-run                    # show, write nothing
#
# Two properties this script is built around:
#
#   * Secrets are never echoed and never land in your shell history. The
#     token secret is generated here and printed once; the T2 admin password
#     is read with the terminal echo off. Both files are written 0600 before
#     anything is put in them, not after.
#
#   * It is re-runnable. An existing .env is backed up with a timestamp
#     rather than clobbered, and every generated file says at the top that
#     it was generated and by what.
#
# Requires: bash 3.2+ (the macOS system bash), python3 3.10+, and pip.
# Deliberately avoids bash 4 features (associative arrays, ${x,,}) so it
# runs on a stock macOS without anyone installing a newer bash first.

set -euo pipefail

VERSION="0.72.2.post5"
T1_API_VERSION="1.0.26.8.1.1"

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

# The HyperNix palette, as close as a terminal gets: red accent on neutrals.
# Disabled when stdout is not a TTY, when NO_COLOR is set (the informal
# standard), or on request — a log file full of escape codes is worse than
# a plain one.
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  C_RED=$'\033[38;5;160m'; C_DIM=$'\033[38;5;245m'; C_TEXT=$'\033[38;5;253m'
  C_OK=$'\033[38;5;71m';  C_WARN=$'\033[38;5;179m'; C_BOLD=$'\033[1m'; C_OFF=$'\033[0m'
else
  C_RED=''; C_DIM=''; C_TEXT=''; C_OK=''; C_WARN=''; C_BOLD=''; C_OFF=''
fi

say()  { printf '%s\n' "${C_TEXT}$*${C_OFF}"; }
dim()  { printf '%s\n' "${C_DIM}$*${C_OFF}"; }
ok()   { printf '%s\n' "${C_OK}  ✓${C_OFF} ${C_TEXT}$*${C_OFF}"; }
warn() { printf '%s\n' "${C_WARN}  !${C_OFF} ${C_TEXT}$*${C_OFF}" >&2; }
err()  { printf '%s\n' "${C_RED}  ✗${C_OFF} ${C_TEXT}$*${C_OFF}" >&2; }
head2() { printf '\n%s\n' "${C_BOLD}${C_RED}$*${C_OFF}"; }

die() { err "$*"; exit 1; }

banner() {
  printf '\n'
  printf '%s\n' "${C_RED}${C_BOLD}  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓${C_OFF}"
  printf '%s\n' "${C_RED}${C_BOLD}  ┃${C_OFF}  ${C_TEXT}HyperNix T1 API — installer${C_OFF}                 ${C_RED}${C_BOLD}┃${C_OFF}"
  printf '%s\n' "${C_RED}${C_BOLD}  ┃${C_OFF}  ${C_DIM}hypernix ${VERSION} · t1 v${T1_API_VERSION}${C_OFF}          ${C_RED}${C_BOLD}┃${C_OFF}"
  printf '%s\n' "${C_RED}${C_BOLD}  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛${C_OFF}"
}

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

INTERACTIVE=1
DRY_RUN=0
CONFIG_DIR="${T1_CONFIG_DIR:-$HOME/.hypernix/t1api}"
INSTALL_MODE=""          # venv | user | system | skip
ASSUME_YES=0
PYTHON_OVERRIDE="${HYPERNIX_PYTHON:-}"

usage() {
  cat <<'USAGE'
install-t1.sh — interactive installer and setup for the HyperNix T1 API.

  --non-interactive     Accept every default; ask nothing. For CI and images.
  --dry-run             Print what would be written; write nothing.
  --config-dir DIR      Where to put .env and the generated files
                        (default: ~/.hypernix/t1api)
  --install venv|user|system|skip
                        How to install the package. "skip" configures an
                        existing installation.
  --python PATH         Use this interpreter instead of searching. Needed
                        when several are installed and the one that has
                        (or should have) hypernix is not the newest.
  --yes, -y             Answer yes to confirmation prompts; open questions
                        (port, allowlist, prices) are still asked, unless
                        --non-interactive is also given. Does not start the
                        server at the end — that stays an explicit choice.
  --host ADDR           Bind address, answering the "Bind to" question up
                        front. 127.0.0.1 keeps it on loopback; 0.0.0.0
                        exposes it to every network this machine is on.
  --port N              Port, answering the "Port" question up front.
  --force               Overwrite an existing .env instead of stopping.
  --help                This.

`hypernix-t1 create` forwards its own --host/--port/--force here, so the
two commands accept the same flags whether or not this script is present.

Environment:
  T1_CONFIG_DIR         Same as --config-dir.
  HYPERNIX_PYTHON       Same as --python.
  NO_COLOR              Disable colour.

Everything the installer writes is listed in the summary at the end, and
every generated file records that it was generated.
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --non-interactive) INTERACTIVE=0; ASSUME_YES=1 ;;
    --dry-run)         DRY_RUN=1 ;;
    --yes|-y)          ASSUME_YES=1 ;;
    --host)            shift; [ $# -gt 0 ] || die "--host needs an address"; FORCED_HOST="$1" ;;
    --port)            shift; [ $# -gt 0 ] || die "--port needs a number"; FORCED_PORT="$1" ;;
    --force)           FORCE_OVERWRITE=1 ;;
    --trusted-network) TRUSTED_NETWORK=1; TRUSTED_NETWORK_SET=1 ;;
    --no-trusted-network) TRUSTED_NETWORK=0; TRUSTED_NETWORK_SET=1 ;;
    --trusted-network-partial-admin)
                       TRUSTED_NETWORK=1; TRUSTED_NETWORK_SET=1
                       TRUSTED_NETWORK_PARTIAL_ADMIN=1 ;;
    --config-dir)      shift; [ $# -gt 0 ] || die "--config-dir needs a path"; CONFIG_DIR="$1" ;;
    --install)         shift; [ $# -gt 0 ] || die "--install needs a mode"; INSTALL_MODE="$1" ;;
    --python)          shift; [ $# -gt 0 ] || die "--python needs a path"; PYTHON_OVERRIDE="$1" ;;
    --help|-h)         usage; exit 0 ;;
    *)                 die "Unknown option: $1 (try --help)" ;;
  esac
  shift
done

# Working out where answers come from when stdin is not a terminal.
#
# Two cases look identical from inside and want opposite handling:
#
#   curl ... | bash          the *script* arrived on stdin, so there is
#                            nothing to read answers from — reach for the
#                            controlling terminal.
#   answers | ./install.sh   stdin *is* the answers — read them.
#
# The distinguishing fact is whether "$0" is a real file: when bash reads a
# script from stdin, it is not. Getting this backwards silently discards
# scripted answers, which looks like the prompts were ignored.
#
# `[ -r /dev/tty ]` is not the right test for "there is a controlling
# terminal" either: the device node exists and is readable in a container
# or under cron, and the redirect still fails. Attempting it in a subshell
# is the only reliable check.
if [ "$INTERACTIVE" = "1" ] && [ ! -t 0 ]; then
  if [ ! -f "$0" ]; then
    if (exec < /dev/tty) 2>/dev/null; then
      exec < /dev/tty
    else
      warn "No terminal to ask questions on — using defaults for everything."
      warn "Download the script and run it directly to answer them."
      INTERACTIVE=0; ASSUME_YES=1
    fi
  fi
  # Otherwise stdin is piped answers to a script file: leave it alone.
fi

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ANSWER=""

# ask <prompt> <default> [hint]
ask() {
  local prompt="$1" default="${2:-}" hint="${3:-}"
  if [ -n "$hint" ]; then dim "     $hint"; fi
  if [ "$INTERACTIVE" = "0" ]; then ANSWER="$default"; return; fi
  local shown=""
  [ -n "$default" ] && shown=" ${C_DIM}[$default]${C_OFF}"
  printf '%s' "${C_RED}  ?${C_OFF} ${C_TEXT}${prompt}${C_OFF}${shown} "
  local reply=""
  IFS= read -r reply || reply=""
  ANSWER="${reply:-$default}"
}

# ask_secret <prompt> — never echoed, never defaulted.
ask_secret() {
  local prompt="$1" reply=""
  if [ "$INTERACTIVE" = "0" ]; then ANSWER=""; return; fi
  printf '%s' "${C_RED}  ?${C_OFF} ${C_TEXT}${prompt}${C_OFF} "
  # `read -s` is bash-only but present in 3.2; stty is the fallback for a
  # shell that lacks it. Either way the terminal is restored on the way out.
  if read -rs reply 2>/dev/null; then
    printf '\n'
  else
    local saved; saved="$(stty -g 2>/dev/null || true)"
    [ -n "$saved" ] && stty -echo
    IFS= read -r reply || reply=""
    [ -n "$saved" ] && stty "$saved"
    printf '\n'
  fi
  ANSWER="$reply"
}

# ask_yes_no <prompt> <default y|n>
ask_yes_no() {
  local prompt="$1" default="${2:-n}"
  if [ "$INTERACTIVE" = "0" ]; then
    [ "$default" = "y" ] && return 0 || return 1
  fi
  # --yes answers the confirmations and leaves the open questions alone,
  # which is the difference between it and --non-interactive: you still
  # get asked for your port and your allowlist, just not "are you sure".
  if [ "$ASSUME_YES" = "1" ]; then
    dim "     $prompt — yes (--yes)"
    return 0
  fi
  local hint="y/N"; [ "$default" = "y" ] && hint="Y/n"
  while :; do
    printf '%s' "${C_RED}  ?${C_OFF} ${C_TEXT}${prompt}${C_OFF} ${C_DIM}[$hint]${C_OFF} "
    local reply=""; IFS= read -r reply || reply=""
    reply="$(printf '%s' "${reply:-$default}" | tr '[:upper:]' '[:lower:]')"
    case "$reply" in
      y|yes) return 0 ;;
      n|no)  return 1 ;;
      *)     warn "Please answer y or n." ;;
    esac
  done
}

# ask_choice <prompt> <default-index> <label1> <label2> ...
# Sets ANSWER to the chosen index (1-based).
ask_choice() {
  local prompt="$1" default="$2"; shift 2
  local count=$#
  local i=1
  for label in "$@"; do
    local marker="  "
    [ "$i" = "$default" ] && marker="${C_RED}>${C_OFF} "
    printf '%s\n' "     ${marker}${C_BOLD}${i})${C_OFF} ${C_TEXT}${label}${C_OFF}"
    i=$((i + 1))
  done
  if [ "$INTERACTIVE" = "0" ]; then ANSWER="$default"; return; fi
  while :; do
    printf '%s' "${C_RED}  ?${C_OFF} ${C_TEXT}${prompt}${C_OFF} ${C_DIM}[$default]${C_OFF} "
    local reply=""; IFS= read -r reply || reply=""
    reply="${reply:-$default}"
    case "$reply" in
      ''|*[!0-9]*) warn "Enter a number between 1 and $count." ;;
      *) if [ "$reply" -ge 1 ] && [ "$reply" -le "$count" ]; then ANSWER="$reply"; return; fi
         warn "Enter a number between 1 and $count." ;;
    esac
  done
}

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

PYTHON=""

preflight() {
  head2 "Checking this machine"

  # `python3` comes first because it is the interpreter the operator is
  # actually using — a venv's, or the one a version manager put on PATH.
  # The versioned names are the fallback for a machine where `python3` is
  # too old, newest first.
  local search="python3 python3.14 python3.13 python3.12 python3.11 python"
  local candidate

  if [ -n "$PYTHON_OVERRIDE" ]; then
    command -v "$PYTHON_OVERRIDE" >/dev/null 2>&1 \
      || die "--python $PYTHON_OVERRIDE: not found."
    "$PYTHON_OVERRIDE" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null \
      || die "--python $PYTHON_OVERRIDE: needs python >= 3.10."
    PYTHON="$(command -v "$PYTHON_OVERRIDE")"
  fi

  # First pass: an interpreter that already has hypernix. That is the
  # right answer for --install skip, whose whole premise is that the
  # package is installed somewhere — picking the newest interpreter
  # instead would verify an install that was never meant to be there and
  # fail on a machine where everything is fine. It is also right for the
  # other modes: installing into the interpreter that already has it
  # upgrades in place rather than leaving two copies on the machine.
  if [ -z "$PYTHON" ]; then
    for candidate in $search; do
      command -v "$candidate" >/dev/null 2>&1 || continue
      "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null || continue
      if "$candidate" -c 'import hypernix' >/dev/null 2>&1; then
        PYTHON="$(command -v "$candidate")"
        break
      fi
    done
  fi

  # Second pass: any interpreter new enough.
  if [ -z "$PYTHON" ]; then
    for candidate in $search; do
      command -v "$candidate" >/dev/null 2>&1 || continue
      if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
        PYTHON="$(command -v "$candidate")"
        break
      fi
    done
  fi

  [ -n "$PYTHON" ] || die "No python3 >= 3.10 found. Install one and re-run."

  # Command substitution captures stdout only, so anything the interpreter
  # writes to stderr at startup lands on the terminal verbatim. A stale
  # .pth file in a system site-packages produces a full traceback there,
  # which under a heading reading "Checking this machine" looks like the
  # installer itself has failed. Keep the information, drop the alarm.
  local py_version=""
  local py_noise=""
  py_version="$("$PYTHON" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null)"
  py_noise="$("$PYTHON" -c 'pass' 2>&1 >/dev/null | head -c 400)"
  ok "python: $PYTHON ($py_version)"
  if [ -n "$py_noise" ]; then
    warn "This interpreter prints a warning on every start:"
    dim "     $(printf '%s' "$py_noise" | head -1)"
    dim "     Harmless for the install, but worth fixing separately."
  fi

  "$PYTHON" -m pip --version >/dev/null 2>&1 \
    || die "pip is not available for $PYTHON. Install it (python3 -m ensurepip) and re-run."
  ok "pip: present"

  case "$(uname -s 2>/dev/null || echo unknown)" in
    Linux)  OS_KIND="linux" ;;
    Darwin) OS_KIND="macos" ;;
    *)      OS_KIND="other" ;;
  esac
  ok "os: $OS_KIND"

  if [ "$DRY_RUN" = "1" ]; then
    warn "Dry run — nothing will be installed or written."
  fi
}

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

VENV_DIR=""
PIP_TARGET_DESC=""

choose_install_mode() {
  if [ -n "$INSTALL_MODE" ]; then return; fi
  head2 "How should the package be installed?"
  dim "     A virtual environment is the safe default: it cannot disturb"
  dim "     anything else on the machine, and removing it removes the install."
  ask_choice "Install mode" 1 \
    "Virtual environment at $CONFIG_DIR/venv  (recommended)" \
    "User install (pip --user)" \
    "System-wide (needs sudo, and can conflict with your package manager)" \
    "Skip — hypernix is already installed"
  case "$ANSWER" in
    1) INSTALL_MODE="venv" ;;
    2) INSTALL_MODE="user" ;;
    3) INSTALL_MODE="system" ;;
    4) INSTALL_MODE="skip" ;;
  esac
}

install_package() {
  head2 "Installing hypernix[t1api]"
  local extras="t1api"

  if [ "$WANT_TUI" = "1" ]; then extras="$extras,dev"; fi
  # security brings cryptography, which is what encrypts the auth-undo
  # payloads and the Keymaster store at rest. Without it both fall back to
  # plain JSON with a warning, which is a worse default than an extra
  # dependency.
  extras="$extras,security"

  local spec="hypernix[$extras]"
  case "$INSTALL_MODE" in
    venv)
      VENV_DIR="$CONFIG_DIR/venv"
      PIP_TARGET_DESC="$VENV_DIR"
      if [ "$DRY_RUN" = "1" ]; then
        dim "     would create venv at $VENV_DIR and install $spec"
        return
      fi
      mkdir -p "$CONFIG_DIR"
      "$PYTHON" -m venv "$VENV_DIR" || die "Could not create a virtual environment at $VENV_DIR"
      PYTHON="$VENV_DIR/bin/python"
      "$PYTHON" -m pip install --quiet --upgrade pip >/dev/null 2>&1 || true
      "$PYTHON" -m pip install --quiet "$spec" || die "pip install $spec failed"
      ;;
    user)
      PIP_TARGET_DESC="user site-packages"
      if [ "$DRY_RUN" = "1" ]; then dim "     would run: $PYTHON -m pip install --user $spec"; return; fi
      "$PYTHON" -m pip install --quiet --user "$spec" || die "pip install --user $spec failed"
      ;;
    system)
      PIP_TARGET_DESC="system site-packages"
      if [ "$DRY_RUN" = "1" ]; then dim "     would run: sudo $PYTHON -m pip install $spec"; return; fi
      sudo "$PYTHON" -m pip install --quiet "$spec" || die "pip install $spec failed"
      ;;
    skip)
      PIP_TARGET_DESC="existing installation"
      ;;
    *) die "Unknown install mode: $INSTALL_MODE" ;;
  esac

  if [ "$DRY_RUN" = "0" ]; then
    if ! "$PYTHON" -c 'import hypernix' 2>/dev/null; then
      # "skip" means the operator says it is already installed. If it is
      # not importable here, the likely cause is that it lives in a
      # different interpreter than the one this search picked — which is
      # a different problem from a failed install, and needs a different
      # remedy.
      if [ "$INSTALL_MODE" = "skip" ]; then
        err "hypernix is not importable with $PYTHON."
        dim "     --install skip configures an existing installation, and this"
        dim "     interpreter does not have one. If it is installed elsewhere:"
        dim ""
        dim "       ./install-t1.sh --install skip --python /path/to/python"
        dim ""
        dim "     Or drop --install skip and let the installer install it."
        exit 1
      fi
      die "hypernix is not importable with $PYTHON after installing. Check the pip output above."
    fi
    local installed
    installed="$("$PYTHON" -c 'import hypernix; print(hypernix.__version__)' 2>/dev/null || echo unknown)"
    ok "hypernix $installed installed into $PIP_TARGET_DESC"
  fi
}

# Run a snippet against the installed hypernix. Used so the shell and the
# Python side agree about validation rather than reimplementing rules here
# and drifting from them.
py() { "$PYTHON" -c "$1" 2>/dev/null; }

py_available() { "$PYTHON" -c 'import hypernix' >/dev/null 2>&1; }

# validate_cidrs <comma-separated list> — echoes "OK" or the first problem.
#
# Uses the standard library rather than the shipped parser because this runs
# before the package is installed. The two agree on what a CIDR is: both go
# through ipaddress.ip_network(strict=False), so a value accepted here is
# accepted by the server. Catching a typo at the prompt matters more than it
# looks — a bad entry aborts the seeding partway through, which leaves the
# whitelist on and half-populated.
validate_cidrs() {
  T1_CIDRS="$1" "$PYTHON" - <<'CIDREOF' 2>/dev/null || echo "could not be checked"
import ipaddress
import os

raw = [c.strip() for c in os.environ.get("T1_CIDRS", "").split(",") if c.strip()]
for cidr in raw:
    try:
        ipaddress.ip_network(cidr, strict=False)
    except ValueError:
        print("%r is not a valid address or CIDR range" % cidr)
        break
else:
    print("OK")
CIDREOF
}

# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------

SERVER_NAME=""
BIND_HOST=""
BIND_PORT=""
#: Set by --host/--port, which answer the bind questions up front. Empty
#: means "ask", which is what an interactive run wants.
FORCED_HOST="${FORCED_HOST:-}"
FORCED_PORT="${FORCED_PORT:-}"
FORCE_OVERWRITE="${FORCE_OVERWRITE:-0}"
PUBLIC_URL=""
ENVIRONMENT="development"
KEY_POLICY="both"          # t1 | t2 | both
ADMIN_PASSWORD=""
REQUIRE_WHITELIST=0
# Defaults assigned *after* argument parsing, so they must not clobber
# a flag: --trusted-network is parsed above and would be reset to 0.
TRUSTED_NETWORK=${TRUSTED_NETWORK:-0}
TRUSTED_NETWORK_SET=${TRUSTED_NETWORK_SET:-0}
TRUSTED_NETWORK_PARTIAL_ADMIN=${TRUSTED_NETWORK_PARTIAL_ADMIN:-0}
ALLOWED_CIDRS=""
RATE_PRESET="standard"
BILLING_MODE="free"        # free | metered
INPUT_PRICE="0.0"
OUTPUT_PRICE="0.0"
CURRENCY="USD"
DEFAULT_PLAN="free"
MODEL_SOURCE="lmstudio"    # lmstudio | examples | custom | none
LMSTUDIO_URL=""
WANT_HYPERLINK=1
WANT_TUI=1
WANT_TLS=0
TLS_CERT=""; TLS_KEY=""
WANT_SYSTEMD=0
TOKEN_SECRET=""

detect_addresses() {
  DETECTED_LAN="$(py 'import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("8.8.8.8", 80)); print(s.getsockname()[0])
except OSError:
    pass
finally:
    s.close()' || true)"
  DETECTED_TS="$(py 'import json, shutil, subprocess
exe = shutil.which("tailscale") or "/Applications/Tailscale.app/Contents/MacOS/Tailscale"
try:
    p = subprocess.run([exe, "status", "--json"], capture_output=True, text=True, timeout=4)
    self_ = json.loads(p.stdout).get("Self") or {}
    print((self_.get("DNSName") or "").rstrip(".") or (self_.get("TailscaleIPs") or [""])[0])
except Exception:
    pass' || true)"
}

q_identity() {
  head2 "Identity"
  ask "What should this server be called?" "$(hostname 2>/dev/null || echo hypernix)" \
      "Reported by GET /status, and what 'waiter -F <name>' matches on."
  SERVER_NAME="$ANSWER"
}

q_network() {
  head2 "Network"
  detect_addresses

  local lan_label="LAN address"
  [ -n "${DETECTED_LAN:-}" ] && lan_label="LAN address (${DETECTED_LAN})"
  local ts_label="Tailscale only"
  [ -n "${DETECTED_TS:-}" ] && ts_label="Tailscale (${DETECTED_TS})"

  # --host and --port answer these two before they are asked. The
  # prompts are skipped rather than pre-filled, because a prompt that
  # shows an answer and then accepts a different one is how an
  # unattended run ends up bound to an address nobody chose.
  if [ -n "$FORCED_HOST" ]; then
    BIND_HOST="$FORCED_HOST"
    dim "     Bind to — $BIND_HOST (--host)"
  else
  dim "     Which addresses should the server accept connections on?"
  ask_choice "Bind to" 1 \
    "Loopback only — 127.0.0.1 (safest; reach it over an SSH tunnel)" \
    "$lan_label — anything on your network can reach it" \
    "All interfaces — 0.0.0.0 (only behind a firewall or a tailnet)" \
    "$ts_label"
  case "$ANSWER" in
    1) BIND_HOST="127.0.0.1" ;;
    2) BIND_HOST="0.0.0.0"; PUBLIC_URL="http://${DETECTED_LAN:-127.0.0.1}" ;;
    3) BIND_HOST="0.0.0.0" ;;
    4) BIND_HOST="0.0.0.0"; [ -n "${DETECTED_TS:-}" ] && PUBLIC_URL="http://${DETECTED_TS}" ;;
  esac
  fi

  if [ -n "$FORCED_PORT" ]; then
    BIND_PORT="$FORCED_PORT"
    dim "     Port — $BIND_PORT (--port)"
  else
    ask "Port" "8000" "The T1 API listens here. HyperLink advertises this port to clients."
    BIND_PORT="$ANSWER"
  fi
  case "$BIND_PORT" in
    ''|*[!0-9]*) warn "Port must be a number; using 8000."; BIND_PORT=8000 ;;
  esac
  [ -n "$PUBLIC_URL" ] && PUBLIC_URL="${PUBLIC_URL}:${BIND_PORT}"

  if [ "$BIND_HOST" = "0.0.0.0" ]; then
    warn "Binding to 0.0.0.0 exposes this API to every network this machine is on."
    dim "     The whitelist question below is how you narrow that back down."
  fi

  ask "Public URL clients should use (blank = work it out from the interfaces)" "$PUBLIC_URL" \
      "Set this when there is a reverse proxy or a tunnel in front."
  PUBLIC_URL="$ANSWER"
}

q_environment() {
  head2 "Deployment kind"
  dim "     Production is not a label: it makes the configuration validated"
  dim "     rather than assumed, and the server refuses to start if something"
  dim "     unsafe is set. Development skips those checks."
  ask_choice "This deployment is" 1 \
    "Development — a machine you are sitting at" \
    "Staging — real config, not real users" \
    "Production — validated on start; refuses unsafe settings"
  case "$ANSWER" in
    1) ENVIRONMENT="development" ;;
    2) ENVIRONMENT="staging" ;;
    3) ENVIRONMENT="production" ;;
  esac
}

q_keys() {
  head2 "Key policy"
  dim "     T1 keys are the long-standing format. T2 adds an access level"
  dim "     (1-9), an optional admin password, and an SSPKID so one server"
  dim "     can hold several individually-addressable keys."
  dim "     A T2 key converts to a valid T1 key, so 'both' costs nothing."
  ask_choice "Which key formats may connect?" 3 \
    "T1 only — refuse T2 keys (for a strict migration window)" \
    "T2 only — T1 keys are still valid but not accepted here" \
    "Both — T2 keys accepted alongside T1 (recommended)"
  case "$ANSWER" in
    1) KEY_POLICY="t1" ;;
    2) KEY_POLICY="t2" ;;
    3) KEY_POLICY="both" ;;
  esac

  if [ "$KEY_POLICY" = "t1" ]; then
    dim "     Skipping the T2 admin password — this server will not accept T2 keys."
    return
  fi

  head2 "T2 admin password"
  dim "     An admin T2 key carries a 7-13 character password in its prefix,"
  dim "     from A-Z, a-z and 1-9. It is what stops an admin key being forged"
  dim "     by pattern alone, since the key format itself is public."
  dim "     It is stored in the config file with 0600 permissions and is"
  dim "     never echoed here."

  if [ "$INTERACTIVE" = "0" ]; then
    ADMIN_PASSWORD="$(generate_admin_password)"
    ok "Generated a T2 admin password (shown in the summary once)."
    return
  fi

  if ask_yes_no "Generate a strong one for me?" "y"; then
    ADMIN_PASSWORD="$(generate_admin_password)"
    ok "Generated."
  else
    while :; do
      ask_secret "T2 admin password (7-13 chars, A-Z a-z 1-9):"
      local candidate="$ANSWER"
      local verdict
      verdict="$(validate_admin_password "$candidate")"
      if [ "$verdict" = "OK" ]; then
        ADMIN_PASSWORD="$candidate"
        ok "Accepted."
        break
      fi
      err "$verdict"
      if ask_yes_no "Generate one instead?" "n"; then
        ADMIN_PASSWORD="$(generate_admin_password)"; ok "Generated."; break
      fi
    done
  fi
}

generate_admin_password() {
  if py_available; then
    py 'from hypernix.security.t2keys import generate_admin_password
print(generate_admin_password())'
    return
  fi
  # Pre-install fallback. Same alphabet; the Python generator additionally
  # rejects predictable sequences, and the value is re-validated after the
  # install so the two cannot disagree silently.
  #
  # `head -c` *first*, then filter. The obvious spelling —
  # `tr -dc ... < /dev/urandom | head -c 12` — makes head close the pipe
  # while tr is still writing, and under `set -o pipefail` that SIGPIPE
  # (141) takes the whole installer down. Bounding the input instead lets
  # tr reach EOF and exit cleanly.
  head -c 4096 /dev/urandom | LC_ALL=C tr -dc 'A-Za-z1-9' | cut -c1-12
}

# Echoes "OK" or the reason it was rejected. Uses the shipped validator so
# the installer and the server agree about what a valid password is.
validate_admin_password() {
  local candidate="$1"
  if py_available; then
    T1_CANDIDATE="$candidate" "$PYTHON" -c 'import os
from hypernix.security.t2keys import validate_admin_password
ok, reason = validate_admin_password(os.environ.get("T1_CANDIDATE", ""))
print("OK" if ok else reason)' 2>/dev/null && return
  fi
  local length=${#candidate}
  if [ "$length" -lt 7 ] || [ "$length" -gt 13 ]; then
    printf 'Must be 7-13 characters; that one is %s.\n' "$length"; return
  fi
  case "$candidate" in
    *[!A-Za-z1-9]*) printf 'Only A-Z, a-z and 1-9 are allowed (no 0, no symbols).\n'; return ;;
  esac
  printf 'OK\n'
}

q_whitelist() {
  head2 "Who may connect"
  dim "     With the whitelist on, an address must be explicitly allowed"
  dim "     before it can reach any endpoint — the check runs before"
  dim "     authentication, so a blocked address never gets to present a key."
  local default_answer="n"
  [ "$BIND_HOST" = "0.0.0.0" ] && default_answer="y"
  if ask_yes_no "Require an allowlist to connect?" "$default_answer"; then
    REQUIRE_WHITELIST=1
    local suggestion="127.0.0.1/32"
    [ -n "${DETECTED_LAN:-}" ] && suggestion="127.0.0.1/32,$(printf '%s' "$DETECTED_LAN" | sed 's/\.[0-9]*$/.0\/24/')"
    [ -n "${DETECTED_TS:-}" ] && suggestion="$suggestion,100.64.0.0/10"
    local cidr_verdict=""
    local cidr_tries=0
    while :; do
      ask "Addresses to allow (comma-separated CIDRs)" "$suggestion" \
          "100.64.0.0/10 is the whole Tailscale range. Add more later with 'waiter security --allow'."
      ALLOWED_CIDRS="$ANSWER"
      [ -n "$ALLOWED_CIDRS" ] || break
      cidr_verdict="$(validate_cidrs "$ALLOWED_CIDRS")"
      [ "$cidr_verdict" = "OK" ] && break
      cidr_tries=$((cidr_tries + 1))
      err "$cidr_verdict"
      # Don't loop forever against a non-interactive stdin or a stubborn
      # typo; fall back to loopback, which is always reachable from here.
      if [ "$INTERACTIVE" = "0" ] || [ "$cidr_tries" -ge 3 ]; then
        warn "Using 127.0.0.1/32 instead. Add the rest with 'waiter security --allow'."
        ALLOWED_CIDRS="127.0.0.1/32"
        break
      fi
    done
    if [ -z "$ALLOWED_CIDRS" ]; then
      warn "An empty allowlist with the whitelist on locks everyone out, including you."
      warn "Adding 127.0.0.1/32 so the server stays reachable from this machine."
      ALLOWED_CIDRS="127.0.0.1/32"
    fi
  else
    REQUIRE_WHITELIST=0
    if [ "$BIND_HOST" = "0.0.0.0" ]; then
      warn "Bound to all interfaces with no allowlist: anything that can route"
      warn "  to this machine can reach the API and try keys against it."
    fi
  fi
}

q_trusted_network() {
  head2 "Keyless access from your own network"
  dim "     Normally every request needs a key. Trusted-network mode lets"
  dim "     connections from this machine, your LAN, or your Tailscale"
  dim "     tailnet skip that — so the phone app pairs without a key and"
  dim "     'waiter' works from your desk without one."
  dim ""
  dim "     A public address can never get this, whatever you answer. The"
  dim "     check is on the address the connection actually came from, so"
  dim "     knowing the endpoint is not enough to get in."
  # --yes must not turn this on. ask_yes_no answers every confirmation
  # with yes, which is right for "are you sure" and wrong for the one
  # question that lowers an authentication requirement: an unattended
  # install would come up serving the LAN without a key, and nobody
  # would have chosen that. It takes its own flag, or a person.
  if [ "$TRUSTED_NETWORK_SET" = "1" ]; then
    if [ "$TRUSTED_NETWORK" = "1" ]; then
      dim "     Enabled by --trusted-network."
    else
      dim "     Disabled by --no-trusted-network."
      return 0
    fi
  elif [ "$ASSUME_YES" = "1" ] && [ "$INTERACTIVE" = "1" ]; then
    dim "     Left off: --yes does not enable keyless access. Pass"
    dim "     --trusted-network to turn it on deliberately."
    TRUSTED_NETWORK=0
    return 0
  elif ! ask_yes_no "Allow keyless access from this machine, the LAN and your tailnet?" "n"; then
    TRUSTED_NETWORK=0
    return 0
  fi
  TRUSTED_NETWORK=1
  warn "Keyless access is on. Anything that can reach this server from your"
  warn "  LAN or tailnet can use the API without presenting a key — including"
  warn "  a guest on your wifi, and any device someone else added to the"
  warn "  tailnet. This lowers the authentication requirement on purpose."
  if [ "$BIND_HOST" = "0.0.0.0" ] && [ "$REQUIRE_WHITELIST" = "0" ]; then
    warn ""
    warn "  You are also bound to all interfaces with no allowlist. Keyless"
    warn "  access still refuses public addresses, but the allowlist is what"
    warn "  stops them reaching the server at all — consider turning it on."
  fi
  dim ""
  dim "     Partial admin additionally lets those connections change things"
  dim "     (write operations). It never grants full admin: a keyless caller"
  dim "     cannot do what an admin key exists to gate."
  if [ "$ASSUME_YES" = "1" ] && [ "$TRUSTED_NETWORK_SET" != "1" ]; then
    dim "     Partial admin left off (--yes does not enable it)."
  elif ask_yes_no "Also allow partial admin from trusted connections?" "n"; then
    TRUSTED_NETWORK_PARTIAL_ADMIN=1
    warn "Partial admin is on for LAN and tailnet connections."
  fi
  if [ -n "${REVERSE_PROXY:-}" ] || [ -n "${PUBLIC_URL:-}" ]; then
    dim ""
    dim "     You mentioned a proxy or a public URL. If a reverse proxy sits"
    dim "     in front of this server, set T1_TRUSTED_PROXIES to its address"
    dim "     in the generated .env — otherwise every request appears to come"
    dim "     from the proxy, and keyless access is refused for all of them."
  fi
}

q_requests() {
  head2 "Request limits"
  dim "     Rate limits run before authentication and before any model work,"
  dim "     so an over-limit caller costs nothing. These are per-key buckets:"
  dim "     a burst size and a sustained rate."
  ask_choice "Limits" 2 \
    "Relaxed — 600 burst, 10/s sustained (a machine you trust)" \
    "Standard — 120 burst, 2/s sustained (recommended)" \
    "Strict — 30 burst, 0.5/s sustained (a shared or public deployment)" \
    "Off — no rate limiting (not advisable on anything reachable)"
  case "$ANSWER" in
    1) RATE_PRESET="relaxed" ;;
    2) RATE_PRESET="standard" ;;
    3) RATE_PRESET="strict" ;;
    4) RATE_PRESET="off"
       warn "Rate limiting off means one client can occupy the whole server." ;;
  esac
}

q_cost() {
  head2 "Cost and plans"
  dim "     The T1 API meters usage per key and can price it. Pricing is"
  dim "     recorded per model; this sets the default for the models you"
  dim "     select next. Nothing here charges anyone — it is accounting,"
  dim "     surfaced through 'waiter cost'."
  ask_choice "Usage accounting" 1 \
    "Free — count tokens, price everything at zero" \
    "Metered — count tokens and apply a per-1k price"
  case "$ANSWER" in
    1) BILLING_MODE="free"; INPUT_PRICE="0.0"; OUTPUT_PRICE="0.0" ;;
    2) BILLING_MODE="metered"
       ask "Input price per 1k tokens" "0.50"
       INPUT_PRICE="$ANSWER"
       ask "Output price per 1k tokens" "1.50"
       OUTPUT_PRICE="$ANSWER"
       ask "Currency" "USD"
       CURRENCY="$ANSWER"
       ;;
  esac

  ask "Default plan for new keys" "free" \
      "Plans drive the routing cascade and which models a key may reach."
  DEFAULT_PLAN="$ANSWER"
}

q_models() {
  head2 "Models"
  dim "     What should this server serve? The LM Studio bridge is the"
  dim "     quickest start: it borrows whatever model LM Studio already has"
  dim "     loaded, so there is nothing to download."
  ask_choice "Model source" 1 \
    "LM Studio bridge — borrow a model already loaded in LM Studio" \
    "A registry file I will supply" \
    "The bundled example entries (placeholders; not real models)" \
    "None for now — configure models later"
  case "$ANSWER" in
    1) MODEL_SOURCE="lmstudio"
       local guess="http://localhost:1234"
       ask "LM Studio address" "$guess" \
           "Reachable from *this* machine, not from your clients — the bridge relays."
       LMSTUDIO_URL="$ANSWER"
       ;;
    2) MODEL_SOURCE="custom"
       ask "Path to a model registry JSON" "$CONFIG_DIR/models.json" \
           "A template is written there if the file does not exist."
       MODEL_REGISTRY_PATH="$ANSWER"
       ;;
    3) MODEL_SOURCE="examples"
       warn "The example entries are placeholders. Routing against them will fail."
       ;;
    4) MODEL_SOURCE="none" ;;
  esac
}

q_features() {
  head2 "Features"

  if ask_yes_no "Enable HyperLink (the phone app's surface: pairing, chat, files)?" "y"; then
    WANT_HYPERLINK=1
  else
    WANT_HYPERLINK=0
  fi

  dim "     The waiter manager TUI is a full-screen curses dashboard over"
  dim "     this server: models, keys, usage, events and live logs, with"
  dim "     graphical controls instead of one subcommand at a time."
  if ask_yes_no "Install the waiter manager TUI for graphical controls?" "y"; then
    WANT_TUI=1
  else
    WANT_TUI=0
  fi

  if [ "$ENVIRONMENT" = "production" ] || [ "$BIND_HOST" = "0.0.0.0" ]; then
    dim "     Without TLS, keys cross the network in plaintext. If a reverse"
    dim "     proxy already terminates TLS, answer no and set"
    dim "     T1_MTLS_BEHIND_PROXY=1 in the generated .env."
    if ask_yes_no "Configure TLS certificates now?" "n"; then
      WANT_TLS=1
      ask "Path to the certificate (.pem/.crt)" ""
      TLS_CERT="$ANSWER"
      ask "Path to the private key" ""
      TLS_KEY="$ANSWER"
      if [ ! -f "$TLS_CERT" ] || [ ! -f "$TLS_KEY" ]; then
        warn "One of those paths does not exist yet; writing them anyway."
        warn "  The server will refuse to start until both are readable."
      fi
    fi
  fi

  if [ "$OS_KIND" = "linux" ] && [ "$ENVIRONMENT" != "development" ]; then
    if ask_yes_no "Write a systemd unit so it starts on boot?" "n"; then
      WANT_SYSTEMD=1
    fi
  fi
}

# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

WRITTEN=""

note_written() { WRITTEN="$WRITTEN
  $1"; }

# write_file <path> <mode> — content on stdin.
write_file() {
  local path="$1" mode="${2:-0644}"
  if [ "$DRY_RUN" = "1" ]; then
    dim "     would write $path (mode $mode)"
    cat > /dev/null
    return
  fi
  mkdir -p "$(dirname "$path")"
  if [ -f "$path" ]; then
    local backup
    backup="${path}.$(date +%Y%m%d-%H%M%S).bak"
    cp "$path" "$backup"
    warn "$(basename "$path") existed — kept a copy at $(basename "$backup")"
  fi
  # Create empty at the target mode *before* writing, so a secret is never
  # briefly world-readable between creation and chmod.
  : > "$path"
  chmod "$mode" "$path"
  cat > "$path"
  note_written "$path"
}

rate_rules_json() {
  case "$RATE_PRESET" in
    relaxed)  printf '[{"name":"per-key","capacity":600,"refill_per_second":10,"applies_to":["key"]},{"name":"per-ip","capacity":1200,"refill_per_second":20,"applies_to":["ip"]}]' ;;
    standard) printf '[{"name":"per-key","capacity":120,"refill_per_second":2,"applies_to":["key"]},{"name":"per-ip","capacity":240,"refill_per_second":4,"applies_to":["ip"]}]' ;;
    strict)   printf '[{"name":"per-key","capacity":30,"refill_per_second":0.5,"applies_to":["key"]},{"name":"per-ip","capacity":60,"refill_per_second":1,"applies_to":["ip"]}]' ;;
    off)      printf '' ;;
  esac
}

write_env() {
  # An existing .env is somebody's working server. Overwriting it
  # regenerates the token secret, which invalidates every key already
  # minted against it -- a failure that shows up later as "the server
  # rejects my keys" rather than here as "the file was replaced".
  # `hypernix-t1 create` refuses the same way, and forwards --force here
  # so the two commands agree.
  if [ -f "$CONFIG_DIR/.env" ] && [ "$FORCE_OVERWRITE" != "1" ]; then
    die "$CONFIG_DIR/.env already exists. Edit it with \`hypernix-t1 configure\`, or pass --force to replace it (which invalidates existing keys)."
  fi
  head2 "Writing the configuration"

  if [ "$DRY_RUN" = "0" ]; then
    # Same SIGPIPE care as generate_admin_password, and note the
    # parentheses: `a || b | c` parses as `a || (b | c)`, so the fallback
    # pipeline is one unit rather than the `||` binding to `b` alone.
    TOKEN_SECRET="$(py 'import secrets; print(secrets.token_hex(32))')"
    if [ -z "$TOKEN_SECRET" ]; then
      TOKEN_SECRET="$(head -c 4096 /dev/urandom | LC_ALL=C tr -dc 'a-f0-9' | cut -c1-64)"
    fi
  else
    TOKEN_SECRET="<generated at write time>"
  fi

  # Both directions of the migration switch. "t2" is a real narrowing of
  # what the server accepts, not a label: without T1_ACCEPT_T1_KEYS=0 the
  # server would go on accepting the T1 spelling and the choice would mean
  # nothing. Both off is refused by the server at startup.
  local accept_t2=1
  local accept_t1=1
  case "$KEY_POLICY" in
    t1) accept_t2=0 ;;
    t2) accept_t1=0 ;;
  esac

  local rules; rules="$(rate_rules_json)"
  local rate_enabled=1
  [ "$RATE_PRESET" = "off" ] && rate_enabled=0

  local allow_unlisted=1
  [ "$REQUIRE_WHITELIST" = "1" ] && allow_unlisted=0

  write_file "$CONFIG_DIR/.env" 0600 <<ENVEOF
# HyperNix T1 API configuration
# Generated by install-t1.sh (hypernix $VERSION, t1 v$T1_API_VERSION)
# on $(date -u '+%Y-%m-%d %H:%M:%S UTC').
#
# This file contains secrets and is mode 0600. Re-running the installer
# keeps a timestamped copy rather than overwriting it.
#
# Every variable here maps 1:1 to a field on T1APIConfig. See
# wiki/T1-API.md#configuration for the full list, including the ones this
# installer does not ask about.

# --- Identity --------------------------------------------------------------
T1_SERVER_NAME=$SERVER_NAME
T1_ENVIRONMENT=$ENVIRONMENT

# --- Secrets ---------------------------------------------------------------
# Signs scoped tokens. Changing it invalidates every outstanding token,
# which is the intended way to revoke them all at once.
T1_TOKEN_SECRET=$TOKEN_SECRET
$(if [ -n "$ADMIN_PASSWORD" ]; then
cat <<INNER
# The T2 admin password component. Keys minted with 'gkey create --type admin'
# and presented as T2 carry this; it is what distinguishes an admin key from
# a user key that merely has a high access level.
T1_T2_ADMIN_PASSWORD=$ADMIN_PASSWORD
INNER
fi)

# --- Keys ------------------------------------------------------------------
# Policy chosen at install: $KEY_POLICY
# T1 keys always work. This switch controls whether T2 keys are also
# accepted; a T2 key converts to a valid T1 key either way.
T1_ACCEPT_T2_KEYS=$accept_t2
T1_ACCEPT_T1_KEYS=$accept_t1
# The store the admin key below was minted into. Without this the
# server reads ~/.hypernix/keymaster instead and would not recognise it.
T1_KEYMASTER_DIR=$CONFIG_DIR/keymaster
T1_DEFAULT_PLAN=$DEFAULT_PLAN

# --- Network ---------------------------------------------------------------
# Where uvicorn binds. start-t1.sh passes these on the command line and
# 'hypernix-t1 start' reads them from here, so both start the server in
# the same place. They used to live only in start-t1.sh, which meant
# 'hypernix-t1 start' fell back to its own 127.0.0.1:8000 default and
# every later status, logs, key and test pointed at an address nothing
# was listening on.
#
# Single quotes, not backticks: this heredoc is unquoted so that
# $BIND_HOST expands, which means a backtick here is command
# substitution and would run the command while writing this file.
T1_HOST=$BIND_HOST
T1_PORT=$BIND_PORT

# The port advertised to HyperLink clients. Keep it matching whatever
# uvicorn is actually bound to — it cannot be inferred behind a proxy.
T1_HYPERLINK_PORT=$BIND_PORT
$(if [ -n "$PUBLIC_URL" ]; then printf 'T1_HYPERLINK_PUBLIC_URL=%s\n' "$PUBLIC_URL"; else printf '# T1_HYPERLINK_PUBLIC_URL=https://t1.example.com\n'; fi)

# Whether an address must be explicitly allowed before it can reach any
# endpoint. The check runs before authentication.
T1_NETWORK_POLICY_ENABLED=1
T1_ALLOW_UNLISTED_CLIENTS=$allow_unlisted

# Keyless access from this machine, the LAN and the tailnet. Off unless
# it was asked for at install. A public address never gets it whatever
# this says, and the decision is made on the address the connection
# actually arrived from, so knowing the endpoint is not enough.
T1_TRUSTED_NETWORK=$TRUSTED_NETWORK
T1_TRUSTED_NETWORK_PARTIAL_ADMIN=$TRUSTED_NETWORK_PARTIAL_ADMIN

# Reverse proxies whose X-Forwarded-For may be believed. Leave empty if
# there is no proxy. If there IS one and this is empty, every request
# looks like it came from the proxy, so keyless access is refused for
# all of them rather than granted to all of them.
# T1_TRUSTED_PROXIES=127.0.0.1/32

# Never "*" on an authenticated API: a wildcard origin lets any site drive
# it with a user's credentials.
T1_CORS_ALLOW_ORIGINS=

# --- Request limits --------------------------------------------------------
# Preset: $RATE_PRESET
T1_RATE_LIMIT_ENABLED=$rate_enabled
$(if [ -n "$rules" ]; then printf "T1_RATE_LIMIT_RULES='%s'\n" "$rules"; else printf '# T1_RATE_LIMIT_RULES=\n'; fi)

# --- Audit and backups -----------------------------------------------------
T1_AUDIT_ENABLED=1
T1_BACKUP_DIR=$CONFIG_DIR/backups
T1_BACKUP_MAX_COUNT=20

# --- Storage ---------------------------------------------------------------
T1_DB_PATH=$CONFIG_DIR/t1api.sqlite3
T1_MODULE_STORAGE_DIR=$CONFIG_DIR/modules
T1_HYPERLINK_FILES_DIR=$CONFIG_DIR/files
T1_HF_DOWNLOAD_DIR=$CONFIG_DIR/models
# PostgreSQL is the documented production backend. SQLite needs nothing.
# T1_DATABASE_URL=postgresql://user:pass@host/hypernix

# --- Models ----------------------------------------------------------------
# Source chosen at install: $MODEL_SOURCE
$(case "$MODEL_SOURCE" in
  lmstudio) printf 'T1_LMSTUDIO_ENABLED=1\nT1_LMSTUDIO_URL=%s\nT1_ENABLE_EXAMPLE_MODELS=0\n' "$LMSTUDIO_URL" ;;
  custom)   printf 'T1_LMSTUDIO_ENABLED=1\nT1_MODEL_REGISTRY_PATH=%s\nT1_ENABLE_EXAMPLE_MODELS=0\n' "${MODEL_REGISTRY_PATH:-$CONFIG_DIR/models.json}" ;;
  examples) printf 'T1_LMSTUDIO_ENABLED=1\nT1_ENABLE_EXAMPLE_MODELS=1\n' ;;
  none)     printf 'T1_LMSTUDIO_ENABLED=0\nT1_ENABLE_EXAMPLE_MODELS=0\n' ;;
esac)

# --- HyperLink -------------------------------------------------------------
T1_HYPERLINK_ENABLED=$WANT_HYPERLINK
T1_HF_DOWNLOADS_ENABLED=1
# A Hugging Face token, for gated repositories.
# T1_HF_TOKEN=

# --- TLS -------------------------------------------------------------------
$(if [ "$WANT_TLS" = "1" ]; then
printf 'T1_TLS_CERTFILE=%s\nT1_TLS_KEYFILE=%s\n' "$TLS_CERT" "$TLS_KEY"
else
printf '# No TLS configured. If a reverse proxy terminates it, say so:\n# T1_MTLS_BEHIND_PROXY=1\n# T1_TRUSTED_PROXIES=127.0.0.1\n'
fi)
ENVEOF

  ok "Configuration written to $CONFIG_DIR/.env"
}

write_registry_template() {
  [ "$MODEL_SOURCE" = "custom" ] || return 0
  local path="${MODEL_REGISTRY_PATH:-$CONFIG_DIR/models.json}"
  [ -f "$path" ] && { dim "     $path already exists — left alone."; return 0; }

  write_file "$path" 0644 <<REGEOF
[
  {
    "_comment": "Generated by install-t1.sh. Replace model_id and the limits with a real model, then restart. 'status': 'available' is what makes it routable.",
    "model_id": "local-model",
    "display_name": "Local model",
    "version": "1.0",
    "total_parameters": 7.0,
    "active_parameters": null,
    "architecture": "dense",
    "supported_tasks": ["chat", "completion"],
    "availability": "public",
    "minimum_plan": "$DEFAULT_PLAN",
    "free_tier_available": true,
    "api_available": true,
    "local_available": true,
    "remote_available": false,
    "context_limit": 8192,
    "input_token_limit": 8192,
    "output_token_limit": 2048,
    "tool_call_limit": 8,
    "pricing": {
      "input_price_per_1k": $INPUT_PRICE,
      "output_price_per_1k": $OUTPUT_PRICE,
      "currency": "$CURRENCY"
    },
    "routing_priority": 10,
    "fallback_model": null,
    "license": "unspecified",
    "status": "available",
    "notes": "Edit before serving traffic."
  }
]
REGEOF
  ok "Model registry template at $path"
}

write_start_script() {
  local runner="$PYTHON"
  [ "$INSTALL_MODE" = "venv" ] && runner="$CONFIG_DIR/venv/bin/python"

  write_file "$CONFIG_DIR/start-t1.sh" 0755 <<STARTEOF
#!/usr/bin/env bash
# Start the HyperNix T1 API. Generated by install-t1.sh.
set -euo pipefail
cd "\$(dirname "\$0")"

# The server reads .env itself; exporting here as well means an operator
# running an ad-hoc 'waiter' or 'gkey' in this shell sees the same settings.
set -a
# shellcheck disable=SC1091
. ./.env
set +a

exec "$runner" -m uvicorn "hypernix.t1api:create_app" --factory \\
  --host "$BIND_HOST" --port "$BIND_PORT" "\$@"
STARTEOF
  ok "Start script at $CONFIG_DIR/start-t1.sh"
}

write_systemd_unit() {
  [ "$WANT_SYSTEMD" = "1" ] || return 0
  write_file "$CONFIG_DIR/hypernix-t1api.service" 0644 <<UNITEOF
# Generated by install-t1.sh. Install with:
#   sudo cp $CONFIG_DIR/hypernix-t1api.service /etc/systemd/system/
#   sudo systemctl daemon-reload && sudo systemctl enable --now hypernix-t1api
[Unit]
Description=HyperNix T1 API
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
User=$(id -un)
WorkingDirectory=$CONFIG_DIR
ExecStart=$CONFIG_DIR/start-t1.sh
Restart=on-failure
RestartSec=5

# The config file holds secrets; the service does not need anything else.
EnvironmentFile=$CONFIG_DIR/.env
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=$CONFIG_DIR

[Install]
WantedBy=multi-user.target
UNITEOF
  ok "systemd unit at $CONFIG_DIR/hypernix-t1api.service"
}

# ---------------------------------------------------------------------------
# Post-install
# ---------------------------------------------------------------------------

ADMIN_KEY=""
ADMIN_KEY_LABEL="Admin key"

mint_admin_key() {
  [ "$DRY_RUN" = "0" ] || return 0
  head2 "Creating an admin key"
  dim "     Printed once, here. It is not written to disk in plaintext —"
  dim "     the key store keeps only what it needs to verify it."

  ADMIN_KEY="$(T1_KEYMASTER_DIR="$CONFIG_DIR/keymaster" "$PYTHON" - <<'MINTEOF' 2>/dev/null || true
import os
from hypernix.security.keymaster import Keymaster, KeyScope, KeyType

km = Keymaster(store_dir=os.environ["T1_KEYMASTER_DIR"], auto_rotate=False)
meta = km.create(
    key_type=KeyType.ADMIN,
    scopes={KeyScope.ADMIN, KeyScope.READ, KeyScope.WRITE},
    prefix="installer",
    note="Created by install-t1.sh",
)
print(meta.key)
MINTEOF
)"

  if [ -z "$ADMIN_KEY" ]; then
    warn "Could not mint an admin key automatically. Create one with:"
    warn "  gkey create --type admin --scopes admin,read,write"
    return
  fi

  # Under a T2-only policy the key just minted is in the T1 spelling, which
  # is exactly what this server has been told to refuse. Printing it as the
  # admin credential would hand the operator a key their own server rejects
  # on the first request — the lockout is total, because there is no other
  # admin key to fix the setting with.
  #
  # The T1 key stays valid in the store; only the spelling changes. Admin
  # authority comes from the store (key type + scopes), not from the T2
  # password component, so the wrapped key is a working admin credential.
  if [ "$KEY_POLICY" = "t2" ]; then
    local wrapped=""
    wrapped="$(T1_WRAP_KEY="$ADMIN_KEY" "$PYTHON" - <<'WRAPEOF' 2>/dev/null || true
import os

from hypernix.security.t2keys import T2KeyGenerator

print(T2KeyGenerator.from_t1(os.environ["T1_WRAP_KEY"], access_level=9).raw)
WRAPEOF
)"
    if [ -n "$wrapped" ]; then
      ADMIN_KEY="$wrapped"
      ADMIN_KEY_LABEL="Admin key (T2)"
      ok "Admin key created and presented as T2 — this server refuses the T1 spelling."
    else
      # Don't leave the operator with a credential that cannot work.
      warn "Could not present the admin key in its T2 form."
      warn "  This server is set to refuse T1 keys, so the key below will not"
      warn "  authenticate. Setting T1_ACCEPT_T1_KEYS=1 in the .env so you can"
      warn "  get in; narrow it again once you have a working T2 key."
      KEY_POLICY="both"
      if [ -f "$CONFIG_DIR/.env" ]; then
        sed 's/^T1_ACCEPT_T1_KEYS=0$/T1_ACCEPT_T1_KEYS=1/' "$CONFIG_DIR/.env" > "$CONFIG_DIR/.env.tmp" \
          && chmod 0600 "$CONFIG_DIR/.env.tmp" && mv "$CONFIG_DIR/.env.tmp" "$CONFIG_DIR/.env"
      fi
      ok "Admin key created."
    fi
  else
    ok "Admin key created."
  fi
}

seed_allowlist() {
  [ "$REQUIRE_WHITELIST" = "1" ] || return 0
  [ "$DRY_RUN" = "0" ] || return 0
  [ -n "$ALLOWED_CIDRS" ] || return 0
  # Seeded into the database directly rather than over HTTP: the server is
  # not running yet, and an allowlist that only takes effect after the first
  # successful connection is not an allowlist.
  #
  # The result is checked rather than assumed. Reporting success here when
  # the seeding failed is the worst outcome this script can produce: the
  # operator turns on the whitelist, is told the addresses are allowed, and
  # is then locked out of their own server with no way in short of editing
  # the database. So stderr is captured and shown, and the verification is
  # a separate read-back rather than the exit code of the writer.
  #
  # stderr is folded into the capture so a traceback survives to the error
  # message, which means the interpreter's own noise (deprecation warnings,
  # broken .pth files in a system site-packages) lands there too. The
  # verified list is therefore tagged and extracted rather than being
  # assumed to be the whole capture.
  local seed_output=""
  local seed_status=0
  seed_output="$(T1_SEED_CIDRS="$ALLOWED_CIDRS" T1_SEED_DB="$CONFIG_DIR/t1api.sqlite3" \
    "$PYTHON" - 2>&1 <<'SEEDEOF'
import os
import sys

from hypernix.t1api.db import SQLiteBackend
from hypernix.t1api.netpolicy import NetworkPolicy

policy = NetworkPolicy(SQLiteBackend(os.environ["T1_SEED_DB"]), allow_unlisted=False)
wanted = [c.strip() for c in os.environ.get("T1_SEED_CIDRS", "").split(",") if c.strip()]
for cidr in wanted:
    policy.allow(cidr, reason="seeded by install-t1.sh", created_by="installer")

# Read back rather than trusting the writes. An allowlist that silently
# did not persist locks the operator out of their own server.
present = {entry.cidr for entry in policy.list_entries()}
missing = [c for c in wanted if c not in present]
if missing:
    sys.exit("did not persist: " + ", ".join(missing))
print("T1SEEDOK:" + ",".join(sorted(present)))
SEEDEOF
)" || seed_status=$?

  if [ "$seed_status" -ne 0 ]; then
    err "Could not seed the allowlist — and the whitelist is ON."
    err "  As written, this server will refuse every connection including yours."
    dim "     $seed_output"
    dim "     Fix it before starting, with either:"
    dim "       waiter security --allow 127.0.0.1/32"
    dim "       or set T1_ALLOW_UNLISTED_CLIENTS=1 in $CONFIG_DIR/.env"
    return
  fi
  case "$seed_output" in
    *T1SEEDOK:*)
      ok "Allowlist seeded and verified: ${seed_output##*T1SEEDOK:}"
      ;;
    *)
      # Exit status 0 without the marker should be impossible; treat it as a
      # failure anyway rather than claiming a verification that did not run.
      err "Allowlist seeding produced no verification marker — treat the"
      err "  whitelist as unseeded and check it before starting the server."
      dim "     $seed_output"
      ;;
  esac
}

verify_config() {
  [ "$DRY_RUN" = "0" ] || return 0
  head2 "Checking the configuration"
  local problems
  problems="$(T1_ENV_FILE="$CONFIG_DIR/.env" "$PYTHON" - <<'CHECKEOF' 2>/dev/null || true
import os

from hypernix.t1api.config import T1APIConfig

config = T1APIConfig.from_env(dotenv_path=os.environ["T1_ENV_FILE"])
for problem in config.production_problems():
    print(problem)
CHECKEOF
)"
  if [ -z "$problems" ]; then
    ok "No configuration warnings."
    return
  fi
  if [ "$ENVIRONMENT" = "production" ]; then
    err "This configuration will not start in production mode:"
  else
    warn "Would block a production deployment (fine for $ENVIRONMENT):"
  fi
  printf '%s\n' "$problems" | while IFS= read -r line; do
    [ -n "$line" ] && dim "     • $line"
  done
}

summary() {
  head2 "Done"
  say ""
  say "  ${C_BOLD}Server${C_OFF}      $SERVER_NAME  (${ENVIRONMENT})"
  say "  ${C_BOLD}Listening${C_OFF}   http://${BIND_HOST}:${BIND_PORT}"
  [ -n "$PUBLIC_URL" ] && say "  ${C_BOLD}Clients use${C_OFF} $PUBLIC_URL"
  say "  ${C_BOLD}Keys${C_OFF}        $KEY_POLICY"
  say "  ${C_BOLD}Allowlist${C_OFF}   $([ "$REQUIRE_WHITELIST" = "1" ] && printf 'required (%s)' "$ALLOWED_CIDRS" || printf 'not required')"
  say "  ${C_BOLD}Limits${C_OFF}      $RATE_PRESET"
  say "  ${C_BOLD}Accounting${C_OFF}  $BILLING_MODE$([ "$BILLING_MODE" = "metered" ] && printf ' (%s/%s per 1k %s)' "$INPUT_PRICE" "$OUTPUT_PRICE" "$CURRENCY")"
  say "  ${C_BOLD}Models${C_OFF}      $MODEL_SOURCE$([ -n "$LMSTUDIO_URL" ] && printf ' — %s' "$LMSTUDIO_URL")"

  say ""
  say "  ${C_BOLD}Written${C_OFF}${WRITTEN}"

  if [ -n "$ADMIN_KEY" ] || [ -n "$ADMIN_PASSWORD" ]; then
    say ""
    printf '%s\n' "${C_RED}${C_BOLD}  Shown once — copy these now${C_OFF}"
    [ -n "$ADMIN_KEY" ] && say "  ${C_BOLD}${ADMIN_KEY_LABEL}${C_OFF}       $ADMIN_KEY"
    [ -n "$ADMIN_PASSWORD" ] && say "  ${C_BOLD}T2 password${C_OFF}     $ADMIN_PASSWORD"
    dim "     The password is also in $CONFIG_DIR/.env (mode 0600)."
    dim "     The key is not stored in plaintext anywhere."
  fi

  say ""
  say "  ${C_BOLD}Next${C_OFF}"
  say "    $CONFIG_DIR/start-t1.sh"
  if [ -n "$ADMIN_KEY" ]; then
    say "    waiter serv -A -I http://${BIND_HOST}:${BIND_PORT} -K '<admin key>' -E"
  fi
  [ "$WANT_TUI" = "1" ] && say "    waiter tui              ${C_DIM}# the manager dashboard${C_OFF}"
  [ "$MODEL_SOURCE" = "lmstudio" ] && say "    waiter lmstudio status  ${C_DIM}# is a model loaded?${C_OFF}"
  [ "$WANT_HYPERLINK" = "1" ] && say "    waiter hyperlink pair   ${C_DIM}# connect the phone app${C_OFF}"
  [ "$WANT_SYSTEMD" = "1" ] && say "    sudo cp $CONFIG_DIR/hypernix-t1api.service /etc/systemd/system/"
  say ""
  dim "  Docs: wiki/T1-API.md    Security checklist: wiki/T1-API-Security-Checklist.md"
  say ""
}

offer_launch() {
  [ "$DRY_RUN" = "0" ] || return 0
  [ "$INTERACTIVE" = "1" ] || return 0
  # --yes answers confirmations; it does not decide to run a server. Both
  # branches below end in exec, so treating this as a confirmation would
  # hand a foreground process to someone who asked not to be interrupted.
  # Setting up and starting are separate decisions, and this is the second.
  if [ "$ASSUME_YES" = "1" ]; then
    dim "     Not starting the server (--yes answers confirmations, not this)."
    dim "     Start it with: $CONFIG_DIR/start-t1.sh"
    return 0
  fi
  if [ "$WANT_TUI" = "1" ] && [ -n "$ADMIN_KEY" ]; then
    if ask_yes_no "Start the server and open the waiter manager TUI now?" "y"; then
      say ""
      dim "     Starting the server in the background…"
      ( "$CONFIG_DIR/start-t1.sh" >"$CONFIG_DIR/server.log" 2>&1 & echo $! > "$CONFIG_DIR/server.pid" )
      sleep 3
      if kill -0 "$(cat "$CONFIG_DIR/server.pid" 2>/dev/null)" 2>/dev/null; then
        ok "Server running (log: $CONFIG_DIR/server.log)"
        local waiter="waiter"
        [ "$INSTALL_MODE" = "venv" ] && waiter="$CONFIG_DIR/venv/bin/waiter"
        "$waiter" serv -A -I "http://127.0.0.1:${BIND_PORT}" -K "$ADMIN_KEY" -E >/dev/null 2>&1 || true
        exec "$waiter" tui
      else
        err "The server did not start. See $CONFIG_DIR/server.log"
      fi
    fi
  elif ask_yes_no "Start the server now?" "n"; then
    exec "$CONFIG_DIR/start-t1.sh"
  fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
  banner
  preflight
  choose_install_mode

  # Features are asked before the install because the answers decide which
  # extras get pulled in — installing twice to add one is a slow surprise.
  q_identity
  q_network
  q_environment
  q_keys
  q_whitelist
  q_trusted_network
  q_requests
  q_cost
  q_models
  q_features

  install_package

  # Re-validate a hand-typed password now that the shipped validator is
  # available. The pre-install fallback checks length and alphabet but not
  # the sequence rules, and the two must not disagree.
  if [ -n "$ADMIN_PASSWORD" ] && [ "$DRY_RUN" = "0" ] && py_available; then
    verdict="$(validate_admin_password "$ADMIN_PASSWORD")"
    if [ "$verdict" != "OK" ]; then
      warn "The password you entered does not pass the full check: $verdict"
      warn "Replacing it with a generated one."
      ADMIN_PASSWORD="$(generate_admin_password)"
    fi
  fi

  write_env
  write_registry_template
  write_start_script
  write_systemd_unit
  mint_admin_key
  seed_allowlist
  verify_config
  summary
  offer_launch
}

main "$@"
