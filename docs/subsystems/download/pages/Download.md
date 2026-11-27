# Download — `hypernix.download`

Fetches a full HuggingFace-style model snapshot — weights, configs,
tokenizer files, sharded-weight index manifests — into a local
directory, and verifies the result is actually usable by
[`hypernix convert`](Quantization.md).

## Short-name resolution

```python
from hypernix.download import resolve_repo_id, resolve_model_info

resolve_repo_id("hyper-nix.2")   # -> "ray0rf1re/hyper-Nix.2"
resolve_repo_id("org/my-model")  # unchanged (already has a "/")
```

`KNOWN_MODELS: dict[str, ModelInfo]` is the checked-in registry of short names and aliases → repository IDs, keyed case-insensitively. It is intentionally not reproduced as a static list here because it evolves with the source. `ModelInfo(repo_id, arch, notes="", family="")` is a frozen dataclass. `arch` is a dispatch/metadata tag indicating the intended path; it is not a test result or universal compatibility guarantee:

| `arch` | Loaded via |
|---|---|
| `"hypernix"` | `HyperNixModel`, interleaved RoPE, no q/k/v bias (HyperNix-native Llama-shape). |
| `"llama"` / `"qwen2"` / `"mistral"` | `HyperNixModel`, HF-shaped state dict with `model.` prefix. |
| `"nano-nano"` | Custom `NanoNanoModel` (toy architecture). |
| `"auto"` | Intended generic Transformers `AutoModelForCausalLM` path for compatible causal language models; actual loadability depends on installed Transformers version, model code/config, dependencies, and access permissions. |

`resolve_repo_id(name_or_repo_id)` — anything already containing `/` is
returned unchanged; otherwise looked up in `KNOWN_MODELS` (falls back to
returning the input unchanged if not found, so unrecognized names still
pass through to `snapshot_download` as-is).

`resolve_model_info(name_or_repo_id)` — returns the matching `ModelInfo`
by short name or by matching a full `repo_id` against the registry
values; `None` if nothing matches.

## Compatibility and truth in the registry

The registry contains names, aliases, and architecture/family labels used by the current code. It is **not** a verified catalog of reachable Hub repositories, licensed downloads, architecture-specific inference support, or quality-tested models. Some tags such as `"asr"` identify a modality/category and should not be read as proof that the ordinary causal-language-model loading path supports that model. Always validate the downloaded snapshot and test the intended loader.

## Fallback chains

```python
FALLBACK_CHAINS: dict[str, tuple[str, ...]] = {
    "nix": ("Nix-ai/Nix-2.7a", "Nix-ai/Nix2.6-mm", "ray0rf1re/Nix2.5"),
}
```

For short names with a fallback chain, `download_model()` tries each
repo in order until one succeeds — later entries are mirrors/older
versions used only if the preferred repo 404s, is gated, or hits a
network error. `resolve_repo_id()` itself always returns just the first
entry; the chain is only consulted inside `download_model()`.

## `download_model()`

```python
from hypernix.download import download_model

path = download_model("hyper-nix.2", quiet=False)
```

| Arg | Type | Default | Notes |
|---|---|---|---|
| `repo_id` | `str` | `"ray0rf1re/hyper-nix.1"` | Short name or full repo id. |
| `revision` | `str \| None` | `None` | Git revision/branch/tag. |
| `cache_dir` | `str \| None` | `None` | Overrides the HF cache dir. |
| `local_dir` | `str \| None` | `None` | If set, downloads directly here instead of the blob-store cache. |
| `model_dir_name` | `str \| None` | `None` | Name for the model dir under `$HOME/.cache/hypernix/models`; defaults to the repo's last path segment. |
| `token` | `str \| None` | `None` | HF token, or reads `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`. |
| `quiet` | `bool` | `False` | Suppresses per-file logging (still emits nothing on stdout — logs go to stderr). |
| `verify` | `bool` | `True` | After download, calls `verify_snapshot()` and raises if incomplete. |

Behavior:
1. Emits a warning via `hypernix.utils.warn_hyper_nix_2` if the resolved
   repo is hyper-Nix.2 (best-effort — any exception here is swallowed).
2. Computes the target directory (`local_dir` if given, else
   `~/.cache/hypernix/models/{model_dir_name or repo-name}`).
3. Walks the fallback chain (or single resolved repo) calling
   `snapshot_download(..., allow_patterns=REQUIRED_PATTERNS)` until one
   succeeds; logs a warning per failed candidate. Raises `RuntimeError`
   if every candidate in the chain fails.
4. Safety net: if `config.json` still isn't present after the snapshot
   download (some repos gate it behind a non-default branch or embed it
   oddly), attempts a single-file `hf_hub_download` for just that file.
5. If `verify=True`, calls `verify_snapshot(path)`; else just lists
   present files.
6. Logs the final file list (to stderr, unless `quiet=True`) and
   returns the snapshot `Path`.

`REQUIRED_PATTERNS` is the explicit allow-list passed to
`snapshot_download` — every JSON in the repo root, every tokenizer
flavor (`tokenizer.*`, `vocab.*`, `merges.*`, SentencePiece files,
`chat_template.*`, `*.tiktoken`), every weight format (`*.safetensors`
+ its index, `pytorch_model*.bin` + its index, `*.pt`, `*.pth`), plus
`*.txt`/`*.md`/`*.model`/`LICENSE*`/`README*`.

## `verify_snapshot(model_dir)`

```python
from hypernix.download import verify_snapshot
files = verify_snapshot("~/.cache/hypernix/models/hyper-Nix.2")
```

Raises `FileNotFoundError` if the directory doesn't exist, if
`config.json` is missing, or if no file matches any of
`_WEIGHT_GLOBS = ("*.safetensors", "pytorch_model*.bin", "*.pt", "*.pth", "*.bin")`
— the error message includes up to 20 of the actually-present filenames
for debugging. On success, returns the sorted list of present filenames.

### Required modules

- `huggingface_hub` (`hf_hub_download`, `snapshot_download`) — hard dependency
- `hypernix.utils.warn_hyper_nix_2` (internal, best-effort, exceptions swallowed)
- Standard library: `sys`, `dataclasses`, `pathlib`


> **Canonical source:** `src/hypernix/models/download.py`. The `hypernix.download` import spelling used in examples is the compatibility-facing API; the path above is the implementation to check when verifying behavior. The short-name registry changes with the checked-in source. A registry entry or parser label is metadata, not a guarantee that the upstream repository is available or that every runtime can load it.

---

## See also

- [Ovens](Ovens.md) — `preheat()`, which calls `download_model()` under the hood when given a repo id rather than a local dir
- [Quantization](Quantization.md) — `hypernix convert` / `hypernix quantize`, the next pipeline stage after a verified download
- `hypernix.fetcher` — downloads the `llama-quantize` binary itself, a separate concern from model snapshots
