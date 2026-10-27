# hnx0xv2image — an image-to-text model you fuse onto a text model

`hypernix.models.hnx0xv2image` (0.72.8a3) is a HyperNix image model in
two halves that are trained, saved and swapped on their own:

| Part | What it is |
|---|---|
| **The image model** (`model_type: hnx0xv2image`) | A vision transformer in the hyperNix0x-v2 style — patches, RMSNorm, bidirectional attention, SwiGLU, no biases — and a projector that pools the patch grid 2x2 and maps it into a text model's embedding space. |
| **A text model** | Any **LFM2.5** that is not already a vision model (`lfm2`, `lfm2_moe`), any **Qwen3.5** (`qwen3_5`, `qwen3_5_text`, the MoE ones — their text part), or any **HyperNix** model: a hyperNix0x-v2 Brewer folder (with or without `config.json` — a folder uploaded straight from a training run, like HyperNix.3.1-mini's beta, reads its config from the safetensors header or the checkpoint beside it, and its tokenizer from `tokenizer/`), a Brewer `.pt` checkpoint, or a HyperNix snapshot transformers can load. |

Fusing does not touch the text model. The prompt is split at each
`<image>` and the image's tokens are spliced in at the embedding level,
so the vocabulary and every weight of the text model are exactly the
ones you had (training `stage="projector"` is tested to leave them
bit-for-bit). A `-VL` model is refused: it has a vision tower already.

## Presets

| Preset | Image | Tokens an image | Parameters |
|---|---|---|---|
| `tiny` | 64 px, patch 8 | 16 | ~0.2 M (tests, a CPU) |
| `small` | 384 px, patch 16 | 144 | ~27 M |
| `base` | 512 px, patch 16 | 256 | ~100 M |

Images are letterboxed to the square (not stretched or cropped), so a
caption is of the whole picture in its own proportions. PNG, JPEG, WebP
and anything Pillow opens; a path, bytes, or a `data:` URL.

## Training, one part at a time

```python
from hypernix.models import hnx0xv2image as hi

image = hi.ImageModel(hi.preset("small"))
fused = hi.fuse(image, "LiquidAI/LFM2.5-1.2B-Instruct")      # a folder or a repo id
hi.train(fused, "captions.jsonl", stage="projector", steps=2000, lr=1e-3)
hi.train(fused, "captions.jsonl", stage="image", steps=2000, lr=2e-4)
hi.train(fused, "chats.jsonl", stage="all", steps=500, lr=2e-5)
fused.save("./lfm2.5-eye")
print(fused.generate("photo.jpg", "What is in this picture?"))
```

| `stage` | Trains | Frozen |
|---|---|---|
| `projector` | the projector | the vision encoder and the text model |
| `image` | the vision encoder and projector | the text model |
| `text` | the text model | the image model |
| `all` | everything | nothing |

`optimizer="c4"` trains with the [c4](C4.md) hybrid AdamW / Pressure
Cooker V5S, which checks each part's gradient before it updates.

A JSONL row is `{"image": PATH, "text": CAPTION}`, `{"image": PATH,
"prompt": ..., "response": ...}`, or `{"images": [...], "messages":
[...]}`; a relative path is read from beside the file. A prompt with
several `<image>`s takes several images, in order; a prompt without one
gets the image first.

## Fusing the same image model elsewhere

`fuse()` checks the projector's width against the text model's. When it
differs (a projector trained for LFM2.5's 2048 on a 4096-wide Qwen3.5) it
is replaced with a new one, and the vision encoder is kept: train
`stage="projector"` again before trusting a caption. `hnx image split`
writes the image model out of a fused one, ready to fuse again.

## Saved layout

```
lfm2.5-eye/
  config.json        model_type hnx0xv2image-fused, the family, tokens per image
  image/             the image model (config.json + model.safetensors)
  text/              the text model, as it was (save_pretrained, or a Brewer folder)
```

`fused.save(folder, text=False)` (`hnx image fuse --link`) writes a
reference to the text model instead of a copy, for a text model you
already have on disk.

## Command line

```bash
hnx image new -o ./eye --preset small
hnx image fuse ./eye LiquidAI/LFM2.5-1.2B-Instruct -o ./lfm-eye
hnx image fuse new Qwen/Qwen3.5-2B -o ./qwen-eye --preset base
hnx image train ./lfm-eye --data captions.jsonl --stage projector --steps 2000
hnx image caption ./lfm-eye photo.jpg -p "Describe the image."
hnx image split ./lfm-eye -o ./eye-for-lfm
hnx image quant ./lfm-eye Q4_K_M -o ./lfm-eye.q4km
hnx image info ./lfm-eye
```

## Quantising with hyprslug

`hnx image quant` (or `hyprslug ./lfm-eye Q4_K_M -o ./lfm-eye.q4km`, which
sends a fused folder here) puts the whole model — image tower, projector
and text model, Brewer or transformers — into one GGUF
(`general.architecture = hnx0xv2image-fused`, both configs in its
metadata) and packs it at any hyprslug tier, sub-bit ones included:

```
lfm-eye.q4km/
  config.json   model_type hnx0xv2image-fused, "format": "gguf", the tier
  model.gguf    every weight, at the tier
  tokenizer/    the text model's tokenizer (none for a character vocabulary)
```

`hnx image caption`, `split`, `train` and `hi.load()` read that folder
(or the `.gguf` itself): each tensor is dequantised as it is loaded. As
everywhere in hyprslug, a tensor whose rows do not divide into the
tier's block (256 for the k-quants and sub-bit tiers, 32 for Q8_0) is
kept at source precision, and the report counts them.

🛈 This is the model at the tier's size on disk, for storing and moving.
llama.cpp does not run it — its multimodal path wants an `mmproj` for a
vision tower it already knows — and in PyTorch it is float again once
loaded.

## Limits

* It runs in PyTorch (`hnx image caption`, the Python API). The GGUF that
  `hnx image quant` writes is for HyperNix to load, not llama.cpp:
  llama.cpp's multimodal path expects its own `mmproj` layout for known
  vision towers.
* Generation is a plain loop without a KV cache — correct for every
  family, slow for long answers.
* A text model from another family is refused unless `allow_other=True`
  (`--allow-other`); it works for any causal LM whose embeddings can be
  given directly, but only the three families are tested.
