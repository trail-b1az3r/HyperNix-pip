### non web and non python package and non hyperlink changelog here (mainly for security issues and breaking changes)
### empty as of public release of 0.72.7.post1

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

