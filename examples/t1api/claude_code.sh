#!/usr/bin/env bash
# Claude Code against a model running in HyperNix (0.72.8a3).
#
# T1 speaks the Anthropic Messages API (/v1/messages), so Claude Code can
# use whatever model this server is serving: the runner's, LM Studio's, or
# a registered one. `hnx claude` sets the variables and starts `claude`:
#
#   ./examples/t1api/claude_code.sh                      # this machine's server
#   T1=http://100.101.102.103:8000 ./examples/t1api/claude_code.sh
#   ./examples/t1api/claude_code.sh -- -p "explain this repo"   # args after -- go to claude
#
# What it sets, if you would rather do it yourself:
#
#   ANTHROPIC_BASE_URL=$T1              # no /v1: Claude Code adds it
#   ANTHROPIC_AUTH_TOKEN=$T1_API_KEY    # or t1-keyless on a trusted network
#   ANTHROPIC_MODEL=<model id>          # and ANTHROPIC_SMALL_FAST_MODEL
#
# A model name Claude Code asks for by itself (claude-*) is answered by
# T1_CLAUDE_CODE_MODEL on the server, or by the model it has loaded.
set -euo pipefail

T1="${T1:-http://127.0.0.1:8000}"
command -v hnx >/dev/null || { echo "hnx is not on PATH: pip install 'hypernix[t1api]'" >&2; exit 1; }

if ! command -v claude >/dev/null; then
  echo "Claude Code is not installed (npm install -g @anthropic-ai/claude-code). The variables:" >&2
  exec hnx claude --server "$T1" ${T1_API_KEY:+--key "$T1_API_KEY"} ${MODEL:+--model "$MODEL"} --print-env
fi
exec hnx claude --server "$T1" ${T1_API_KEY:+--key "$T1_API_KEY"} ${MODEL:+--model "$MODEL"} "$@"
