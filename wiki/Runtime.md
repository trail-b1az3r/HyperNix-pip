# `runtime` — the HyperNix llama.cpp, from other applications

`native/ggml-hnx` builds a llama.cpp that reads the sub-bit types. This
is how everything else gets to use it.

```bash
hnx runtime status                # what is built, detected, installed
hnx runtime serve model.gguf      # start the patched server
hnx runtime path                  # the build's bin directory
hnx runtime install --yes         # put it inside LM Studio
hnx runtime restore --yes         # put LM Studio back
```

## Two routes, very different risk

**`serve` is the one to reach for.** It starts the patched
`llama-server`, which speaks the OpenAI-compatible API that LM Studio,
Jan, Open WebUI, Continue, Zed, Cursor and most of the rest already
know. Nothing on the machine is modified — it is a process on a port,
and closing it puts everything back. It works with applications this has
never heard of, and it does not care what version of them you have.

**`install` copies the patched libraries over the ones LM Studio
bundles**, so LM Studio loads a sub-bit model natively. Nicer when it
works, and it is surgery on somebody else's application. Read the
guarantees below before using it.

## Serving

```bash
hnx runtime serve ~/models/Qwen3.8-2B-IQ0.5_XXXL.gguf --port 8080
```

It prints where to point things, on **stderr** — so `--print-only`
leaves stdout as a bare command line you can run or script:

```
  base URL   http://127.0.0.1:8080/v1
  api key    any non-empty string (it is not checked)

  point one of these at it:
    LM Studio            Developer ▸ add a Remote/OpenAI-compatible provider
    Jan                  Settings ▸ Model Providers ▸ add an OpenAI-compatible provider
    Open WebUI           Settings ▸ Connections ▸ OpenAI API
    Continue (VS Code)   config.json: "provider": "openai", "apiBase"
    Zed                  assistant settings ▸ openai ▸ api_url
    Cursor               Settings ▸ Models ▸ Override OpenAI Base URL
    AnythingLLM          LLM Preference ▸ Local AI / Generic OpenAI
```

The menu paths move between versions — the base URL is the part that
matters, and any client with a "custom OpenAI endpoint" box takes it.

| Flag | Meaning |
| --- | --- |
| `--host`, `--port` | Where to listen. Default `127.0.0.1:8080`. |
| `-ngl`, `--gpu-layers` | Layers on the GPU. 0 (default) is CPU only. |
| `-c`, `--context` | Context length. 0 takes the model's own. |
| `--alias` | The name the model answers to over the API. |
| `--print-only` | Print the command instead of running it. |

`-ngl 0` and leaving `-ngl` out are not the same thing, so the flags are
only passed when you ask for them.

## Installing into LM Studio

```bash
hnx runtime install              # reports what it would do, changes nothing
hnx runtime install --yes        # does it
hnx runtime restore --yes        # undoes it
```

What it guarantees:

- **Nothing without `--yes`.** Bare `install` prints the plan and exits
  `2`, so a script can tell "you did not confirm" from "done".
- **Everything replaced is backed up first**, into
  `~/.hypernix/runtime-bridge/backup/<timestamp>`, with a manifest —
  so `restore` works after the installing process is long gone.
- **It refuses a directory that is not a runtime directory.** If nothing
  in it is named like `libllama` or `libggml`, it stops rather than
  scattering shared objects into somebody's Documents folder.
- **It refuses an unpatched build.** Installing one would replace a
  runtime that cannot read sub-bit models with another that cannot,
  while looking like a fix. Whether a build is patched is read out of
  the binary, not guessed from its path.
- **`restore` removes what it added**, not just what it replaced — a
  library LM Studio never shipped does not get left behind.

What it does not promise: **LM Studio does not support this and will not
help if it goes wrong.** Its updates will overwrite the libraries, and
the ABI it expects can change between versions. The symptom is LM Studio
failing to start a model; `hnx runtime restore --yes` is the fix.

The layout is *detected*, not assumed — `hnx runtime status` shows what
was found. Set `LMSTUDIO_HOME` if yours is somewhere unusual, or pass
`--target` to name the directory outright.

## Wiring it in by hand

```bash
export LD_LIBRARY_PATH="$(hnx runtime path)"
```

`path` prints the bin directory bare, for exactly this. Anything that
dlopens a llama.cpp can be pointed at it.

## Which build it uses

In order: `--build`, then `$HNX_LLAMA_BUILD`, then the one
`native/ggml-hnx/build.sh` makes, then `~/llama.cpp/build`. `status`
says which it found and whether it carries the decoder:

```
  build      /home/me/HyperNix-pip/native/ggml-hnx/llama.cpp/build/bin  (patched)
  libraries  libggml, libggml-base, libggml-cpu, libllama
```

`libggml-base` and `libggml-cpu` both matter: llama.cpp splits ggml into
a format half and a per-backend half, and the sub-bit decode lands in
both. Replacing one and not the other gives a loader that knows the type
exists and cannot compute with it.

## See also

- [HyprSlug](HyprSlug.md) — making the sub-bit files.
- [Studio](../desktop/README.md) — the desktop app, which runs them
  in-process without any of this.
