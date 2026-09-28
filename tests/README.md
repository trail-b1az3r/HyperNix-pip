# Tests

One folder per part of HyperNix, named like the package it tests, so the
tests for `src/hypernix/t1api/` are in `tests/t1api/`, and so on:

| Folder | Covers |
| --- | --- |
| `audio/`, `chat/`, `data/`, `dilute/`, `elements/`, `evaluation/`, `hyperlink/`, `interfaces/`, `models/`, `monitoring/`, `neuron/`, `optimizers/`, `quant/`, `runtime/`, `scriptgen/`, `security/`, `system/`, `t1api/`, `t1sdk/`, `timing/`, `training/`, `waiter/` | `src/hypernix/<same name>/` |
| `ios/` | the HyperLink iPhone and iPad app in `ios/` |
| `desktop/` | the desktop studio in `desktop/` |
| `docs/` | the changelog, the wiki and the documentation site |
| `repo/` | CI, the release workflow, the installers, and checks that span the whole repository |

A test that covers several packages lives with the one it mostly
exercises. `conftest.py` and the shared helpers (`sdk_bridge.py`,
`shell_support.py`) stay at the top, and every folder is on the import
path, so a test can import a helper from a test in another folder.

Run everything with `pytest`, one area with `pytest tests/t1api`, or one
file with `pytest tests/t1api/test_mcp.py`.
