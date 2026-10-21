# HyperNix Browser Agent (Firefox)

A sidebar in Firefox where you give a model a task — "find the cheapest
train to Leeds tomorrow", "fill in this form from my notes" — and it
searches the web, reads pages, clicks, types and scrolls to do it, asking
you before it acts.

The model can run on:

| Where | Address | Tools |
|---|---|---|
| **HyperNix T1 API** | your server, e.g. `http://127.0.0.1:8000` | text protocol, through `/inference/chat`, so the server's registry, quota and prices apply |
| **Your own machine** | any OpenAI-compatible server: llama-server, LM Studio (`http://127.0.0.1:1234/v1`), Ollama (`http://127.0.0.1:11434/v1`) | native tool calling, or the text protocol for models without it |
| **OpenAI** | `https://api.openai.com/v1`, your API key | native tool calling |
| **Anthropic** | `https://api.anthropic.com/v1`, your API key | native tool use |

No model is built in: pick one in the settings page, where **Load models
from the server** lists what the server offers.

## Install

From a checkout:

1. `about:debugging` → **This Firefox** → **Load Temporary Add-on…** →
   pick `browser/firefox/manifest.json`. (Temporary: gone when Firefox
   restarts.)
2. Or build a package: `python scripts/build_firefox_extension.py` writes
   `dist/hypernix-browser-agent-<version>.xpi`. It is unsigned; Firefox
   Developer Edition and Nightly install it with
   `xpinstall.signatures.required` set to `false`, and release Firefox
   needs it signed on addons.mozilla.org.

Then open the settings (⚙ in the sidebar), choose where the model runs,
enter the key and model, and **Save**. Firefox asks once for access to
websites; without it the agent cannot read or use pages. Open the sidebar
with the toolbar button or **Alt+Shift+H**.

## What it can do

`search_web`, `open_url`, `list_tabs`, `switch_tab`, `read_page`,
`find_in_page`, `click`, `type_text`, `press_key`, `select_option`,
`scroll`, `go_back`, `wait`, `screenshot`.

`read_page` gives the model the page's text and a numbered list of what
can be clicked or typed into; the numbers are what `click` and
`type_text` take, and they change every time the page is read.

Search goes through DuckDuckGo by default (no key), or through your T1
server's `/web/v1/search`.

## Safety

* **It asks first.** By default every action that changes something —
  opening a page, clicking, typing, pressing a key, choosing an option —
  waits for **Allow**, **Allow for this task** or **Decline** in the
  sidebar. The settings can narrow that to typing only, or turn it off.
* **Pages are data, not instructions.** Page text reaches the model
  fenced as `<untrusted_page_text>`, and the system prompt tells it not
  to follow anything a page says — the usual way a page tries to hijack
  an agent.
* **Passwords.** Password fields are never read. Typing into one is off
  unless you turn it on.
* **Only web pages.** `open_url` refuses `javascript:`, `data:`, `file:`
  and `about:` addresses; `about:` and add-on pages cannot be scripted
  anyway.
* **Keys stay here.** API keys are kept in this browser's extension
  storage (`storage.local`, not synced) and sent only to the server they
  belong to. Anthropic requires the
  `anthropic-dangerous-direct-browser-access` header for calls from a
  browser; the extension sends it because the key is your own.
* **A step limit** (25 by default) stops a task that is going in circles;
  **Stop** ends one at once.

## Development

```
cd browser/firefox
bun test                      # tools, provider formats, the agent loop, page formatting
pytest tests/browser          # the manifest, and content.js in a real page (needs Playwright)
```

`lib/` is plain script loaded by the background page (it sets
`globalThis.HNX`) and by the tests with `require()`. `content.js` is
injected into a page only when the model asks to read or use it.
