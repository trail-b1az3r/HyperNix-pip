# HyperNix Browser Agent (Firefox)

`browser/firefox` (0.72.8) is a Firefox sidebar extension: give a model a
task, and it searches the web, reads pages, clicks, types and scrolls to
do it — asking you before it acts.

The model runs where you choose:

| Where | Notes |
|---|---|
| HyperNix T1 API | through `/inference/chat`, so the server's registry, quota and prices apply; tools by a text protocol |
| your own machine | any OpenAI-compatible server — llama-server, LM Studio, Ollama — with native tool calling, or the text protocol for models without it |
| OpenAI | your API key |
| Anthropic | your API key |

No model is built in; the settings page lists what the server offers.

## Tools

`search_web` (DuckDuckGo, or T1's web search), `open_url`, `list_tabs`,
`switch_tab`, `read_page` (the text and a numbered list of controls),
`find_in_page`, `click`, `type_text`, `press_key`, `select_option`,
`scroll`, `go_back`, `wait`, `screenshot`.

## Safety

- Every action that changes something waits for **Allow**, **Allow for
  this task** or **Decline**, unless the settings say otherwise
  (typing only, or never).
- Page text is fenced as untrusted, and the model is told not to follow
  instructions found in pages.
- Password fields are never read, and not typed into unless allowed.
- Only `http(s)` addresses open.
- API keys stay in this browser's extension storage and go only to their
  own server.

## Install

`about:debugging` → This Firefox → Load Temporary Add-on → pick
`browser/firefox/manifest.json`; or `python
scripts/build_firefox_extension.py` for an `.xpi` (unsigned). The
extension's [README](../browser/firefox/README.md) has the details.
