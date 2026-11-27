# hyprslug-headers — the three ways out of "this model will not load"

```bash
hypernix hyprslug-headers install
hypernix hyprslug-headers serve model.iq09.gguf --port 1234
hypernix hyprslug-headers wrap  model.iq09.gguf -o compat.gguf --to Q4_K_M
hypernix hyprslug-headers stamp model.iq09.gguf
```

## The error this exists for

```
llama_model_loader: failed to load model from
  ~/.lmstudio/models/HyperNix/Qwen3.8-2B-IQ0.9_L/Qwen3.8-2B-IQ0.9_L.gguf
```

That error is **correct**, and it is worth being blunt about that before
describing a command with "headers" in the name.

LM Studio's bundled `llama.cpp` read the tensor table, found GGML type
200, and stopped. The type id is how it noticed. The reason it stopped is
that there is no dequantisation kernel for 200 in that binary — the
arithmetic to turn seven stored signs into eight weights does not exist
there, and no metadata adds it.

So the thing that sounds like it should work does not: rewriting the
header to claim type 12 makes `llama.cpp` read a 0.9-bit tensor as `Q4_K`
and generate confident nonsense. **That is worse than the error**, because
the error is honest and the nonsense is not.

What a header *can* do is make the file explain itself, and there are two
other mechanisms that get a model running. Three things, and the help
text leads with which is which.

| | keeps the tier | opens in LM Studio | costs |
|---|---|---|---|
| `serve` | yes | yes, over HTTP | a running process |
| `wrap` | no | yes, as a file | 2–8× the size |
| `stamp` | yes | no | nothing |

## serve — keep the tier, move the boundary

The model stays sub-bit inside [HnxRun](HnxRun.md) and is reached over an
OpenAI-compatible endpoint. LM Studio, Bionic, and anything else that
speaks `/v1/chat/completions` can talk to it.

```bash
hypernix hyprslug-headers serve model.iq09.gguf --port 1234 --cache-bytes 2G
curl -s localhost:1234/v1/models | jq '.data[0].hypernix'
```

```json
{
  "tier": "IQ0.9_L",
  "family": "sign-and-scale",
  "bits_per_weight_on_disk": 0.9375,
  "resident_bits_per_weight": 0.9651,
  "runtime": "hypernix.models.hnxrun"
}
```

`GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`,
`GET /health`. Non-streaming: `hnxrun` produces a token at a time
already, but an SSE stream that stalls mid-reply is a worse failure than
a slow complete one, and at these bitrates the whole reply arrives in
about the time a stream's first token would. A `stream: true` request is
refused with that sentence rather than ignored.

`--device` runs it on an accelerator: `auto`, `cpu`, `cuda`, `cuda:1`,
`mps`, `xpu`. On a GPU the *packed* bytes are uploaded once and decoded
there, so a 7B at `IQ0.9_L` puts about 800 MB on the card rather than the
28 GB a host-side decode would push across PCIe every token. See
[Devices](Devices.md) — especially if the card is a GTX 10-series, where
`torch.cuda.is_available()` says True and the first kernel launch fails.

Standard-library `http.server`, not FastAPI. Requiring a web framework in
order to find out why a model will not load — on the machine short enough
of memory that someone quantised to 0.9 bits — is the wrong ask.

Loopback by default. `--host 0.0.0.0` publishes an unauthenticated
inference endpoint to every network the machine is on, and the command
says so on stderr rather than leaving it to be noticed.

## wrap — make it a file that opens anywhere

```bash
hypernix hyprslug-headers wrap model.iq09.gguf -o compat.gguf --to Q4_K_M
```

```
model.iq09.gguf
  -> compat.gguf
  IQ0.9_L -> Q4_K_M, 181.9 KB -> 942.3 KB (5.18x)

  This is a Q4_K_M copy of a IQ0.9_L model, not a IQ0.9_L model that
  llama.cpp can now read. It is 5.2x larger and it carries the error of
  both quantisations. To keep the tier, serve the original through
  hnxrun instead.
```

That paragraph is the deliverable as much as the file is. A compatibility
export presented as the original is the exact class of claim this package
keeps being rebuilt to stop making.

Defaults per tier — the narrowest upstream type that does not throw away
more than the extension type already has:

| from | to |
|---|---|
| `IQ0.25_UXL`, `IQ0.5_XXXL`, `IQ0.75_M`, `IQ0.9_L`, `INT1` | `Q2_K` |
| `FP2` | `Q3_K` |
| `INT4` | `Q4_K` |

The output is **verified**, not assumed: `wrap` re-reads the file it
wrote and deletes it if any tensor still carries an extension type. The
whole promise is one sentence — "stock llama.cpp can open the result" —
so it is checked. See [the bug this caught](#two-bugs-this-found).

## stamp — make the file self-describing

```bash
hypernix hyprslug-headers stamp model.iq09.gguf
hypernix hyprslug-headers show  model.iq09.gguf
```

```
IQ0.9_L: 7 of every 8 signs kept, 256 weights per 30-byte block
         (0.938 bpw), GGML type 200
```

Writes a versioned `hyprslug.header.*` block into the GGUF's own
metadata:

| key | example | why |
|---|---|---|
| `family` | `sign-and-scale` | which of the two decoders applies |
| `block_elements` | `256` | weights per block |
| `block_bytes` | `30` | bytes per block |
| `kept` / `group` | `7` / `8` | signs stored, and per how many weights |
| `levels` | `[-2,-1,1,2]` | fixed-codebook types instead |
| `code_bits` | `2` | bits per code |
| `scale_offset` / `scale_dtype` | `0` / `f16` | where the block scale sits |
| `bits_per_weight` | `0.9375` | the real rate |
| `fallback` | `Q2_K` | what `wrap` would target |
| `runtime` | `hypernix.models.hnxrun` | what can execute it as-is |

Deliberately **arithmetic rather than symbolic**. `packing` names the
codec for anything that has this package installed; everything else is
enough for a reader that does not, which is the entire point of writing
it into the file. A loader that has never heard of type 203 can still
learn that it is looking at 256-weight blocks of 8 bytes with three signs
kept of every sixteen.

The values are read out of the packing tables at write time, never
written down twice — a header that disagreed with the packer would be
worse than no header, because it would be believed.

Unknown keys are skipped on read, so a file stamped by a newer HyperNix
stays readable here. A file with *no* stamp still works: the header is
derived from the tensor types, because every GGUF this package has ever
written carries those and only the recent ones carry the stamp.

## install-model — put a loadable copy where LM Studio looks

```bash
hypernix hyprslug-headers install-model model.iq09.gguf \
    --to Q4_K_M --publisher HyperNix --name Qwen3.8-2B
```

Writes `<root>/<publisher>/<name>/<name>.gguf`, which is the layout LM
Studio and Bionic both scan. What lands there is a `wrap` of the model,
because a sub-bit GGUF is not something their llama.cpp can open — and
the command says so, because "installed into LM Studio" is exactly the
phrase under which someone would otherwise assume the 0.9-bit file itself
now works there.

An already-upstream GGUF is **copied unchanged** rather than
re-quantised: it already opens, and re-encoding it would lose a
generation of quality for nothing.

Set `LMSTUDIO_HOME` or pass `--root` if the store is somewhere unusual.
The original is never touched, and the report points at `serve` for
anyone who wanted the tier kept.

## Finding what applies to what

```bash
hypernix hyprslug-headers install
```

```
runtime installed to ~/.hypernix/hyprslug-headers
  config   : ~/.hypernix/hyprslug-headers/runtime.json
  endpoint : http://127.0.0.1:1234/v1
  LM Studio: ~/.lmstudio/models
  scanned 14 model(s); 2 need this runtime
    IQ0.9_L      Qwen3.8-2B-IQ0.9_L.gguf
    IQ0.5_XXXL   nano-mini-IQ0.5_XXXL.gguf

  Those will not open in LM Studio directly. Either:
    hypernix hyprslug-headers serve ...
    hypernix hyprslug-headers wrap  ... -o compat.gguf
```

LM Studio and Bionic share a model store, so one scan covers both. Set
`LMSTUDIO_HOME` if it is somewhere unusual. `scan` takes any directory;
`status --json` reports what is installed and which types it decodes.

An unreadable file in the tree is *reported*, not raised — the point of
scanning a directory of models is to find out which one is the problem,
and one corrupt file must not end the walk.

## Two bugs this found

Both were invisible from outside, and both have tests.

**`wrap` reported success on a file it had not converted.** `hyprslug`'s
`_readable()` did not list the extension types, so `_should_quantize`
declined every tensor with *"source type 200 is one hyprslug cannot
read"* and copied it verbatim. The output was a `Q2_K`-labelled file
still full of type-200 tensors — refused by exactly the loader the
command exists to satisfy, with the same error the user started with.
`hyprslug` now reads the extension types as a source, which also makes
plain requantisation *from* a sub-bit model work, and `wrap` verifies its
own output.

**`stamp` corrupted `general.alignment`.** Copying metadata key by key
with `set_metadata` re-infers a GGUF type per value, and nothing about
the number `32` says UINT32 rather than INT32. The reference reader
rejected the result with *"Bad type for general.alignment field"* — a
file this package could still read and nothing else could, which is the
worst of both. `copy_metadata_from` preserves the types, and a test now
asserts every type survives a stamp.

## See also

- [HnxRun](HnxRun.md) — the runtime `serve` puts behind the endpoint
- [Devices](Devices.md) — `--device` for `serve`, and the Vulkan answer
- [HyprSlug](HyprSlug.md) — the quantiser that writes these files
- [LowBit](LowBit.md) — the types at 200 and above
