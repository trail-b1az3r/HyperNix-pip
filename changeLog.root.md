### non web and non python package and non hyperlink changelog here (mainly for security issues and breaking changes)
### empty as of public release of 0.72.7.post1

## 0.72.8a5

<details>
<summary>0.72.8a5 — In-development bug-fix log</summary>

- **FIXED DURING DEVELOPMENT — introduced by this update:** `hnx0xv2image_gguf.load_gguf` decoded plain-string metadata (the text family) as JSON and failed on every load; caught by the new tests before release.
- **FIXED DURING DEVELOPMENT — introduced by this update:** the new T1 audit test first used high-entropy fake secrets that gitleaks would have reported; they are runs of one letter.
- Dependencies reviewed across the repository: Python (pyproject.toml, setup.cfg), npm (docs/, src/hypernix/), Bun (hyped-pro, tvtop-max). Raised: Pillow, cryptography, python-multipart, transformers, pillow-heif, pydantic, python-dotenv, tqdm, pytest.

</details>

