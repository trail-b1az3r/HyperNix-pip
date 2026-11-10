### non web and non python package and non hyperlink changelog here (mainly for security issues and breaking changes)
### empty as of public release of 0.72.7.post1

## 0.73.0.dev2

<details>
<summary>0.73.0.dev2 — pre-alpha 2: security</summary>

- **Security:** Strife chat templates refuse underscore attributes however they are spelled.
- **Security:** the Strife build registry (`~/.hypernix/strife/builds.json`) is written owner-only and refused if others can write it. The T1 API selects builds by registered name only.

</details>

## 0.73.0.dev1

<details>
<summary>0.73.0.dev1 — pre-alpha 1: breaking changes and security</summary>

- **Breaking:** T1 `/web/v1` is removed and answers 410 with its `/web/v2` successor. The T1 API is now generation 1.2 (v1.2.26.11.0.0), and 1.1 clients are told to upgrade.
- **Security:** ATC is the outermost layer of the T1 API. It jails addresses that probe, guess keys or map the API, limits bodies and in-flight requests, and offers `hypernix-atc/1` end-to-end encryption (`/atc/handshake`, `/atc/tunnel`).
- **Security:** a T2 key's stated access level is held to the level it was granted.
- `hnx sandbox` confines a command to one directory: no web, no writes outside it, hidden home folder.
- Dependencies reviewed across the repository: Python (pyproject.toml, setup.cfg), npm (docs/), Bun (hyped-pro, tvtop-max). No version changes; `setup.cfg` no longer repeats the dependency list.

</details>

## 0.72.8.post2

<details>
<summary>0.72.8.post2 — In-development bug-fix log</summary>

- Dependencies reviewed across the repository: Python (pyproject.toml, setup.cfg), npm (docs/), Bun (hyped-pro, tvtop-max). Unchanged.

</details>

## 0.72.8.post1

<details>
<summary>0.72.8.post1 — In-development bug-fix log</summary>

- Dependencies reviewed across the repository: Python (pyproject.toml, setup.cfg), npm (docs/), Bun (hyped-pro, tvtop-max). Unchanged.

</details>

## 0.72.8

<details>
<summary>0.72.8 — In-development bug-fix log</summary>

- **FIXED DURING DEVELOPMENT — introduced by this update:** `pip install .` in the 3.15 test rewrote the tracked `src/hypernix.egg-info` files; they were restored, not committed.
- **FIXED DURING DEVELOPMENT — introduced by this update:** the torch-pin test matched only an unmarked `"torch>=…,<…"`; it accepts the 3.15 marker now and still checks the floor and ceiling.
- Dependencies reviewed across the repository: Python (pyproject.toml, setup.cfg), npm (docs/), Bun (hyped-pro, tvtop-max). Changed: torch and sentencepiece carry `python_version < '3.15'`.

</details>

## 0.72.8rc1

<details>
<summary>0.72.8rc1 — In-development bug-fix log</summary>

- **FIXED DURING DEVELOPMENT — introduced by this update:** the runner sent the default model (HyperNix.3-mini, now also a known release) down the new release download path, which changed how it is fetched; the existing test caught it.
- **FIXED DURING DEVELOPMENT — introduced by this update:** every learning-rate sweep trial saved a full model; trials keep none.
- **FIXED DURING DEVELOPMENT — introduced by this update:** scriptgen's sidebar badge was squeezed by long section titles, and the log slider drew its value over the entry beside it.
- **FIXED DURING DEVELOPMENT — introduced by this update:** the security page's Pages build wrote docs/public/v1, so the docs-conflict test counted it as a data writer; writers are now the workflows that push.
- Dependencies reviewed across the repository: Python (pyproject.toml, setup.cfg), npm (docs/), Bun (hyped-pro, tvtop-max). Raised: cryptography (to 50).

</details>

## 0.72.8a5

<details>
<summary>0.72.8a5 — In-development bug-fix log</summary>

- **FIXED DURING DEVELOPMENT — introduced by this update:** `hnx0xv2image_gguf.load_gguf` decoded plain-string metadata (the text family) as JSON and failed on every load; caught by the new tests before release.
- **FIXED DURING DEVELOPMENT — introduced by this update:** the new T1 audit test first used high-entropy fake secrets that gitleaks would have reported; they are runs of one letter.
- Dependencies reviewed across the repository: Python (pyproject.toml, setup.cfg), npm (docs/, src/hypernix/), Bun (hyped-pro, tvtop-max). Raised: Pillow, cryptography, python-multipart, transformers, pillow-heif, pydantic, python-dotenv, tqdm, pytest.

</details>

