# HyperNix0xV3 (experimental)

HyperNix0xV3 is the architecture after hyperNix0x-v2. It is in 0.73.0 as an
**experimental** option. `brew new` still builds V2 unless you pass `-E`:

```bash
brew new --preset cpu-nano -E --name my-v3   # HyperNix0xV3
brew new --preset cpu-nano --name my-v2      # hyperNix0x-v2, as before
brew new --preset xl -E                      # xl exists only for V3
```

`brew new -E` prints what the model will cost: parameters, the attention
window and chunk, the state slots, and how much smaller its cache is than
full attention at the model's context length.

## The layer

Every block has two token mixers and a feed-forward, each behind a learned
gate:

- **Tidepool attention.** Each token attends, in one softmax, to:
  - the tokens in its local window (`window`, causal);
  - one summary for every earlier chunk of `chunk` tokens. A summary is
    the chunk's keys and values pooled by a learned salience score, so the
    tokens that matter most weigh most.

  Only finished chunks are summarised, so a token never sees its own
  chunk's future. Distant text stays reachable at a cost that grows with
  the number of chunks, not the number of tokens.
- **Keel state.** A small set of memory slots (`n_slots` × `slot_dim`).
  Every token reads them by attention. At the end of each chunk the slots
  are updated by a gated write from that chunk, so information can travel
  through the state without being looked up again.
- **SwiGLU feed-forward**, or a mixture of experts (`n_experts`,
  `experts_active` routed per token, with a balance loss) for `xl`.

RMSNorm before each part, rotary positions (NeoX style), grouped-query
attention, and a tied output head by default.

## Presets

| Preset | d_model | Layers | Context | Window / chunk | Slots | Parameters |
|---|---|---|---|---|---|---|
| `cpu-nano` | 128 | 4 | 1,024 | 128 / 32 | 8 | 2.0 M |
| `cpu-tiny` | 256 | 6 | 2,048 | 128 / 32 | 8 | 7.4 M |
| `cpu-small` | 384 | 8 | 4,096 | 256 / 64 | 12 | 21.0 M |
| `33m`, `micro` | 512 | 6 | 8,192 | 256 / 64 | 16 | 36.8 M |
| `small` | 1024 | 9 | 32,768 | 512 / 64 | 24 | 153 M |
| `medium` | 1280 | 18 | 65,536 | 1024 / 128 | 32 | 422 M |
| `large` | 2048 | 36 | 131,072 | 2048 / 128 | 48 | 2.0 B |
| `xl` | 4096 | 48 | 262,144 | 4096 / 256 | 64 | 106.6 B total, 16.3 B active (16 experts) |

The counts are exact (the model is built on the `meta` device to count).

## What works

- Brewer builds, trains (`brew train`; the expert balance loss is added
  for `xl`), saves (`config.json` + `model.safetensors`), reloads and
  exports to `.pt`.
- The brewer adapter and NeoOven load a V3 folder, so the runner's PyTorch
  server and HyperLink serve it. The HyperLink catalogue labels it
  `hyperNix0x-v3`.
- `config.json` carries `"arch": "hypernix0x-v3"`. That tag is how every
  loader tells V3 from V2.

## What does not, yet

- **No GGUF.** llama.cpp does not know Tidepool or Keel, so
  `brew export --format gguf` refuses V3 with that reason, and the runner
  never tries to convert a V3 folder. Strife support is the planned route.
- Generation recomputes the whole sequence for each token. A cache comes
  with Strife support.
- It is experimental: presets and the checkpoint layout may change before
  0.73.0.
