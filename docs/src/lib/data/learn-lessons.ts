// Content for the 17-lesson interactive Learn page.

export const LEARN_LESSONS = [
  {
    id: 'download-basics',
    title: 'Downloading Models',
    track: 'Getting Started',
    trackColor: '#4a9eff',
    concept: `HyperNix resolves "short names" to full HuggingFace repo IDs automatically — you never need to remember the full path for supported models. The download is cached locally after the first call.`,
    example: `from hypernix.download import download_snapshot

# Short name resolves automatically — no full path needed
path = download_snapshot("nix2.5")
print("Saved to:", path)

# Long-form still works
path2 = download_snapshot("ray0rf1re/hyper-nix.1")`,
    exercise: `Use hypernix.download to download the "llama-3.2-1b" model and store the local path in a variable called \`local_path\`, then print it.`,
    starter: `from hypernix.download import ___\n\nlocal_path = ___("___")\nprint("Model at:", local_path)`,
    solution: `from hypernix.download import download_snapshot\n\nlocal_path = download_snapshot("llama-3.2-1b")\nprint("Model at:", local_path)`,
    hints: [
      'The function you need is in hypernix.download — look at the API reference.',
      'The function is called download_snapshot. Pass the short name as the first argument.',
    ],
  },
  {
    id: 'gated-repos',
    title: 'Gated Repos & Tokens',
    track: 'Getting Started',
    trackColor: '#4a9eff',
    concept: `Some models on HuggingFace require you to agree to a license before downloading (called "gated" repos). You pass your HF token to download_snapshot so it can authenticate. You can also set the HF_TOKEN environment variable instead.`,
    example: `from hypernix.download import download_snapshot

# Pass token directly
path = download_snapshot(
    "meta-llama/Llama-3.1-8B-Instruct",
    token="hf_YOUR_TOKEN_HERE"
)

# Or set env var before running:
# export HF_TOKEN=hf_YOUR_TOKEN_HERE`,
    exercise: `Write code to download a gated model called "my-org/private-model" using a token stored in a variable called \`my_token\`. Store the result in \`path\`.`,
    starter: `from hypernix.download import download_snapshot\n\nmy_token = "hf_abc123"\n\n# Download the gated model using my_token\npath = ___(___,\n    token=___\n)\nprint(path)`,
    solution: `from hypernix.download import download_snapshot\n\nmy_token = "hf_abc123"\n\npath = download_snapshot("my-org/private-model",\n    token=my_token\n)\nprint(path)`,
    hints: [
      'download_snapshot accepts a keyword argument called token=',
      'Pass your variable my_token as the value: token=my_token',
    ],
  },
  {
    id: 'preheat-chat',
    title: 'Loading & Chatting',
    track: 'Inference',
    trackColor: '#e8960a',
    concept: `preheat() loads a model into memory and returns a CodeOven object. You can then call .complete() for single-turn text, or .chat() for multi-turn conversations. The chat() method accepts OpenAI-style message dicts.`,
    example: `from hypernix.old_oven import preheat

oven = preheat("nix2.5")

# Single-turn completion
result = oven.complete("The capital of France is")
print(result)

# Multi-turn chat
messages = [
    {"role": "user", "content": "What is PyTorch?"}
]
reply = oven.chat(messages)
print(reply)`,
    exercise: `Load the model "nix2.5" into a variable called \`oven\`, then call .chat() with the message "Explain gradient descent in one sentence." and print the reply.`,
    starter: `from hypernix.old_oven import ___\n\noven = ___("nix2.5")\n\nmessages = [\n    {"role": "___", "content": "Explain gradient descent in one sentence."}\n]\n\nreply = oven.___(messages)\nprint(reply)`,
    solution: `from hypernix.old_oven import preheat\n\noven = preheat("nix2.5")\n\nmessages = [\n    {"role": "user", "content": "Explain gradient descent in one sentence."}\n]\n\nreply = oven.chat(messages)\nprint(reply)`,
    hints: [
      'The function to load a model is called preheat — import it from hypernix.old_oven.',
      'The role key in the message dict should be "user" for a user message.',
      'The method to run multi-turn chat is .chat(), not .complete().',
    ],
  },
  {
    id: 'fim-completion',
    title: 'Fill-in-the-Middle',
    track: 'Inference',
    trackColor: '#e8960a',
    concept: `CodeOven.fill() does fill-in-the-middle (FIM) completion — you provide a prefix and suffix and the model fills in the gap. This is ideal for code completion tasks. The model must be trained for FIM (most code models are).`,
    example: `from hypernix.old_oven import preheat

oven = preheat("nix2.5")

# Fill in the blank between prefix and suffix
result = oven.fill(
    prefix="def multiply(a, b):\\n    result = ",
    suffix="\\n    return result",
    max_tokens=16
)
print(result)  # should output: a * b`,
    exercise: `Use .fill() to complete a function. The prefix should be "def greet(name):\\n    message = " and the suffix should be "\\n    return message". Store the result in \`filled\` and print it.`,
    starter: `from hypernix.old_oven import preheat\n\noven = preheat("nix2.5")\n\nfilled = oven.___(___=\n    "def greet(name):\\n    message = ",\n    ___="\\n    return message"\n)\nprint(filled)`,
    solution: `from hypernix.old_oven import preheat\n\noven = preheat("nix2.5")\n\nfilled = oven.fill(\n    prefix="def greet(name):\\n    message = ",\n    suffix="\\n    return message"\n)\nprint(filled)`,
    hints: [
      'The method is .fill() — not .complete() or .chat().',
      'The two keyword arguments are called prefix= and suffix=',
    ],
  },
  {
    id: 'frying-pan',
    title: 'Preprocessing: FryingPan',
    track: 'Data Pipeline',
    trackColor: '#8866dd',
    concept: `FryingPan is Tier 1 of the preprocessing pipeline. It handles basic tokenisation and deduplication. Pass it a list of text strings (your raw dataset) and call .process() to get the cleaned output. Each higher tier wraps the output of the one below it.`,
    example: `from hypernix.pans import FryingPan

raw = [
    "Hello world",
    "Hello world",   # duplicate — will be removed
    "Gradient descent minimizes loss.",
]

pan = FryingPan(raw)
clean = pan.process()
print(f"Before: {len(raw)}, After: {len(clean)}")`,
    exercise: `Create a FryingPan with the list ["Train models fast.", "Train models fast.", "HyperNix is great."], process it, and print the number of items after deduplication.`,
    starter: `from hypernix.pans import ___\n\ndata = [\n    "Train models fast.",\n    "Train models fast.",\n    "HyperNix is great.",\n]\n\npan = ___(data)\ncleaned = pan.___() \nprint("Items after dedup:", len(cleaned))`,
    solution: `from hypernix.pans import FryingPan\n\ndata = [\n    "Train models fast.",\n    "Train models fast.",\n    "HyperNix is great.",\n]\n\npan = FryingPan(data)\ncleaned = pan.process()\nprint("Items after dedup:", len(cleaned))`,
    hints: [
      'Import FryingPan from hypernix.pans',
      'The method to run processing is .process() — it returns the cleaned dataset.',
    ],
  },
  {
    id: 'blender-weights',
    title: 'Mixing Datasets',
    track: 'Data Pipeline',
    trackColor: '#8866dd',
    concept: `PersonalBlender lets you mix multiple datasets with explicit weights. For example, if you have 80% code data and 20% chat data but want a 50/50 mix in training, you set weights=[0.5, 0.5]. Weights are normalized automatically and don't need to sum to 1.`,
    example: `from hypernix.blender import PersonalBlender

code_data  = ["for i in range(10): print(i)", "def hello(): return 'hi'"]
chat_data  = ["How are you?", "What is 2+2?"]
instruct   = ["Write a poem.", "Summarize this text."]

# 50% code, 30% chat, 20% instruct
blender = PersonalBlender(
    sources=[code_data, chat_data, instruct],
    weights=[0.5, 0.3, 0.2]
)
for sample in blender:
    train(sample)`,
    exercise: `Create a PersonalBlender that mixes two datasets: \`science\` and \`history\`, with 70% science and 30% history. Store it in a variable called \`blender\`.`,
    starter: `from hypernix.blender import ___\n\nscience = ["E=mc^2", "Cells divide by mitosis."]\nhistory = ["Rome fell in 476 AD.", "WW2 ended in 1945."]\n\nblender = ___(\n    sources=[___, ___],\n    weights=[___, ___]\n)`,
    solution: `from hypernix.blender import PersonalBlender\n\nscience = ["E=mc^2", "Cells divide by mitosis."]\nhistory = ["Rome fell in 476 AD.", "WW2 ended in 1945."]\n\nblender = PersonalBlender(\n    sources=[science, history],\n    weights=[0.7, 0.3]\n)`,
    hints: [
      'Import PersonalBlender from hypernix.blender',
      'Weights should be proportional — 0.7 and 0.3 sum to 1.0 for 70/30 split.',
    ],
  },
  {
    id: 'hypernixconfig',
    title: 'Training Config',
    track: 'Training',
    trackColor: '#c8192e',
    concept: `HyperNixConfig is a dataclass that holds all training hyperparameters. The key fields are n_layers (transformer depth), n_heads (attention heads), n_embd (embedding dimension), vocab_size (tokenizer vocab), and block_size (context window). Larger n_embd and n_layers = bigger model.`,
    example: `from hypernix.train import HyperNixConfig

# ~50M parameter model
cfg = HyperNixConfig(
    n_layers=8,
    n_heads=8,
    n_embd=512,
    vocab_size=50257,
    block_size=1024
)
print(cfg)`,
    exercise: `Create a HyperNixConfig for a small 6-layer model with 6 attention heads, 384-dim embeddings, vocab size 50257, and a context window of 512 tokens. Store it in \`cfg\`.`,
    starter: `from hypernix.train import ___\n\ncfg = ___(\n    n_layers=___,\n    n_heads=___,\n    n_embd=___,\n    vocab_size=___,\n    block_size=___\n)\nprint(cfg)`,
    solution: `from hypernix.train import HyperNixConfig\n\ncfg = HyperNixConfig(\n    n_layers=6,\n    n_heads=6,\n    n_embd=384,\n    vocab_size=50257,\n    block_size=512\n)\nprint(cfg)`,
    hints: [
      'Import HyperNixConfig from hypernix.train',
      'The field names exactly match what you see in the API reference: n_layers, n_heads, n_embd, vocab_size, block_size.',
    ],
  },
  {
    id: 'init-train',
    title: 'Initializing & Training',
    track: 'Training',
    trackColor: '#c8192e',
    concept: `init_from_scratch(cfg) creates a randomly-initialised model from your config. Then train() runs the causal-LM loop. The out_dir is where checkpoints are written. This is a single-device, no-sharding loop — perfect for a GTX 1080 or similar.`,
    example: `from hypernix.train import HyperNixConfig, init_from_scratch, train

cfg = HyperNixConfig(n_layers=6, n_heads=6, n_embd=384,
                     vocab_size=50257, block_size=512)

model = init_from_scratch(cfg)
param_count = sum(p.numel() for p in model.parameters())
print(f"Model has {param_count/1e6:.1f}M parameters")

train(model, my_dataset, out_dir="./checkpoints", steps=1000)`,
    exercise: `Given \`cfg\` already defined, write code to: (1) create a model with init_from_scratch, (2) count its parameters and print them in millions, (3) call train() for 500 steps saving to "./run1".`,
    starter: `from hypernix.train import HyperNixConfig, ___, ___\n\ncfg = HyperNixConfig(n_layers=6, n_heads=6, n_embd=384,\n                     vocab_size=50257, block_size=512)\n\n# 1. Create the model\nmodel = ___(cfg)\n\n# 2. Count parameters\nparams = sum(p.numel() for p in model.parameters())\nprint(f"Parameters: {params/___:.1f}M")\n\n# 3. Train for 500 steps\n___(model, my_dataset, out_dir=___, steps=___)`,
    solution: `from hypernix.train import HyperNixConfig, init_from_scratch, train\n\ncfg = HyperNixConfig(n_layers=6, n_heads=6, n_embd=384,\n                     vocab_size=50257, block_size=512)\n\nmodel = init_from_scratch(cfg)\n\nparams = sum(p.numel() for p in model.parameters())\nprint(f"Parameters: {params/1e6:.1f}M")\n\ntrain(model, my_dataset, out_dir="./run1", steps=500)`,
    hints: [
      'init_from_scratch takes one argument: your config object.',
      'To convert to millions divide by 1e6.',
      'train() signature: train(model, dataset, out_dir, steps)',
    ],
  },
  {
    id: 'pressure-cooker',
    title: 'Custom Optimizer',
    track: 'Training',
    trackColor: '#c8192e',
    concept: `PressureCooker wraps AdamW with a built-in LR schedule: linear warmup → steady → cosine cooldown. You call opt.step(loss) instead of doing loss.backward() + opt.step() separately — it handles everything including lookahead. Save/load state for checkpointing.`,
    example: `from hypernix.pressure_cooker import PressureCooker

opt = PressureCooker(model, lr=3e-4, warmup_steps=200)

for batch in dataloader:
    loss = model(batch)
    opt.step(loss)  # backward + gradient step + LR schedule in one call

# Checkpoint the optimizer
import json
json.dump(opt.state_dict(), open("opt_state.json", "w"))

# Resume
opt.load_state_dict(json.load(open("opt_state.json")))`,
    exercise: `Create a PressureCooker for \`model\` with a learning rate of 1e-4 and 100 warmup steps. Store it in \`opt\`. Then write a training step that computes \`loss\` from \`batch\` and calls opt.step() correctly.`,
    starter: `from hypernix.pressure_cooker import ___\n\nopt = ___(\n    model,\n    lr=___,\n    warmup_steps=___\n)\n\n# Single training step\nbatch = next(iter(dataloader))\nloss = model(batch)\n___.step(___)`,
    solution: `from hypernix.pressure_cooker import PressureCooker\n\nopt = PressureCooker(\n    model,\n    lr=1e-4,\n    warmup_steps=100\n)\n\nbatch = next(iter(dataloader))\nloss = model(batch)\nopt.step(loss)`,
    hints: [
      'Import PressureCooker from hypernix.pressure_cooker',
      'lr=1e-4 means 0.0001 — scientific notation is fine in Python.',
      'opt.step(loss) takes the loss tensor — no need to call loss.backward() first.',
    ],
  },
  {
    id: 'freezer-vram',
    title: 'VRAM Management',
    track: 'Training',
    trackColor: '#c8192e',
    concept: `The freezer module manages GPU memory for consumer GPUs. OldFreezer targets 8-10 GB cards (like the GTX 1080), NewFreezer targets 11+ GB, and FlashFreezer retries automatically on OOM. pascal_safe_dtype() returns the right dtype for Pascal-class GPUs.`,
    example: `from hypernix.freezer import auto_freezer, pascal_safe_dtype

# Auto-detect GPU and pick the right freezer
freezer = auto_freezer()
freezer.prepare(model)

# For GTX 1080 / Pascal explicitly
from hypernix.freezer import OldFreezer
dtype = pascal_safe_dtype()  # float16 on sm_61
freezer = OldFreezer(model, target_vram_gb=7.5)`,
    exercise: `Import FlashFreezer and wrap \`model\` with it, storing the result in \`freezer\`. FlashFreezer takes the model as its only required argument.`,
    starter: `from hypernix.freezer import ___\n\n# Wrap model in OOM-safe freezer\nfreezer = ___(model)\n\n# This will now auto-retry on CUDA OOM\nprint(type(freezer))`,
    solution: `from hypernix.freezer import FlashFreezer\n\nfreezer = FlashFreezer(model)\n\nprint(type(freezer))`,
    hints: [
      'Import FlashFreezer from hypernix.freezer',
      'FlashFreezer(model) — just pass the model, no other required args.',
    ],
  },
  {
    id: 'espresso-eval',
    title: 'Evaluating Your Model',
    track: 'Evaluation',
    trackColor: '#e05555',
    concept: `espresso_maker provides 4 tiers of evaluation. Ristretto is a quick 5-prompt spot-check. Lungo is the full multi-run averaged eval with variance. Start with Ristretto during development and move to Lungo before releasing.`,
    example: `from hypernix.espresso_maker import Ristretto, Lungo
from hypernix.old_oven import preheat

oven = preheat("nix2.5")

# Quick sanity check (Tier 1)
results = Ristretto(oven, [
    "What is 2 + 2?",
    "Name the capital of Japan.",
])
print(results)

# Full eval with 3 repeated runs (Tier 4)
results = Lungo(oven, prompts, rubric,
                judge_model=judge_oven, n_runs=3)
print(results.summary())`,
    exercise: `Use Ristretto to spot-check \`my_oven\` (already loaded) with two prompts: "What year did WW2 end?" and "What is the boiling point of water in Celsius?". Store the results in \`spot_check\`.`,
    starter: `from hypernix.espresso_maker import ___\n\n# my_oven is already loaded\nspot_check = ___(\n    my_oven,\n    [\n        "___",\n        "___",\n    ]\n)\nprint(spot_check)`,
    solution: `from hypernix.espresso_maker import Ristretto\n\nspot_check = Ristretto(\n    my_oven,\n    [\n        "What year did WW2 end?",\n        "What is the boiling point of water in Celsius?",\n    ]\n)\nprint(spot_check)`,
    hints: [
      'Import Ristretto from hypernix.espresso_maker',
      'Ristretto(model, list_of_prompts) — the second argument is a plain Python list.',
    ],
  },
  {
    id: 'convert-quantize',
    title: 'Convert & Quantize',
    track: 'Ship',
    trackColor: '#00b5ad',
    concept: `To ship a model for llama.cpp inference you: (1) convert safetensors → GGUF with hypernix.convert, (2) quantize the GGUF with hypernix.quantize. Q4_K_M is the best quality/size balance. Q8_0 keeps near-full quality. fp16 is lossless.`,
    example: `from hypernix.convert import convert_to_gguf_fp16
from hypernix.quantize import quantize

# Step 1: convert to fp16 GGUF
convert_to_gguf_fp16("./my-model", "./my-model-f16.gguf")

# Step 2: quantize to Q4_K_M
quantize("./my-model-f16.gguf", quant_type="q4_k_m")

# Or do both from CLI in one line:
# hypernix convert --repo-id ./my-model --quants q4_k_m`,
    exercise: `Write code to convert "./trained-model" to a fp16 GGUF at "./trained-model-f16.gguf", then quantize it to Q8_0. Use the Python API (not CLI).`,
    starter: `from hypernix.convert import ___\nfrom hypernix.quantize import ___\n\n# Step 1: convert to fp16 GGUF\n___(___,\n    "___"\n)\n\n# Step 2: quantize to q8_0\n___("./trained-model-f16.gguf",\n    quant_type=___\n)`,
    solution: `from hypernix.convert import convert_to_gguf_fp16\nfrom hypernix.quantize import quantize\n\nconvert_to_gguf_fp16("./trained-model",\n    "./trained-model-f16.gguf"\n)\n\nquantize("./trained-model-f16.gguf",\n    quant_type="q8_0"\n)`,
    hints: [
      'Step 1 uses convert_to_gguf_fp16(source_dir, output_path)',
      'Step 2 uses quantize(gguf_path, quant_type="q8_0")',
    ],
  },
  {
    id: 'upload-hub',
    title: 'Uploading to HuggingFace',
    track: 'Ship',
    trackColor: '#00b5ad',
    concept: `push_to_hub() uploads your model directory to the HuggingFace Hub with multipart support for large files. create_model_card() auto-generates a README.md. You can also use the CLI: hypernix upload --repo-id your/model --path ./dir`,
    example: `from hypernix.upload import push_to_hub, create_model_card

push_to_hub(
    "./my-model",
    "my-username/my-cool-model",
    token="hf_YOUR_TOKEN"
)

create_model_card(
    "my-username/my-cool-model",
    license="apache-2.0",
    language="en",
    tags=["text-generation", "pytorch"]
)`,
    exercise: `Upload the directory "./fine-tuned-model" to the repo "ray0rf1re/my-fine-tune" using the token stored in \`HF_TOKEN\`, then generate a model card with license "mit" and tags ["text-generation"].`,
    starter: `from hypernix.upload import ___, ___\n\nHF_TOKEN = "hf_abc123"\n\n# Upload the model directory\n___(\n    "___",\n    "ray0rf1re/my-fine-tune",\n    token=___\n)\n\n# Generate the model card\n___(\n    "ray0rf1re/my-fine-tune",\n    license=___,\n    tags=___\n)`,
    solution: `from hypernix.upload import push_to_hub, create_model_card\n\nHF_TOKEN = "hf_abc123"\n\npush_to_hub(\n    "./fine-tuned-model",\n    "ray0rf1re/my-fine-tune",\n    token=HF_TOKEN\n)\n\ncreate_model_card(\n    "ray0rf1re/my-fine-tune",\n    license="mit",\n    tags=["text-generation"]\n)`,
    hints: [
      'You need two imports: push_to_hub and create_model_card',
      'push_to_hub(local_path, repo_id, token=...)',
      'create_model_card(repo_id, license=..., tags=[...])',
    ],
  },

  // ── hypernix.old_fridge ───────────────────────────────────────────────────
  {
    id: 'freeze-layers',
    title: 'Freezing Layers',
    track: 'Memory Management',
    trackColor: '#8866dd',
    concept: `freeze() locks parameter groups so their weights aren't updated during training. Pass name patterns to target specific layers (e.g. "embed" freezes all embedding layers). Call parameter_stats() to confirm what's frozen vs trainable.`,
    example: `from hypernix.old_fridge import freeze, parameter_stats

# Freeze all embedding layers
freeze(model, patterns=("embed",))

stats = parameter_stats(model)
print(stats)
# → {'frozen': 12582912, 'trainable': 62914560, 'total': 75497472}`,
    exercise: `Freeze only the layers matching "attn" in \`model\`, then print the parameter stats using parameter_stats().`,
    starter: `from hypernix.old_fridge import ___, ___\n\n# Freeze attention layers only\n___(model, patterns=(___,))\n\nstats = ___(model)\nprint(stats)`,
    solution: `from hypernix.old_fridge import freeze, parameter_stats\n\nfreeze(model, patterns=("attn",))\n\nstats = parameter_stats(model)\nprint(stats)`,
    hints: ['Import both freeze and parameter_stats from hypernix.old_fridge', 'patterns= takes a tuple of strings — use ("attn",) with a trailing comma'],
  },
  {
    id: 'unfreeze-offload',
    title: 'Unfreeze & VRAM Cleanup',
    track: 'Memory Management',
    trackColor: '#8866dd',
    concept: `unfreeze() reverses freeze(). offload_to_cpu() moves all tensors off GPU to free VRAM — useful before loading a second model. chill_cache() calls torch.cuda.empty_cache() and gc.collect() to reclaim fragmented VRAM.`,
    example: `from hypernix.old_fridge import freeze, unfreeze, offload_to_cpu, chill_cache

freeze(model)          # lock everything
# ... do something ...
unfreeze(model)        # unlock everything again

# Free VRAM between jobs
offload_to_cpu(model)
chill_cache()`,
    exercise: `Call offload_to_cpu() on \`model_a\` to free VRAM, then call chill_cache() to empty the CUDA cache. Import both from hypernix.old_fridge.`,
    starter: `from hypernix.old_fridge import ___, ___\n\n# Move model off GPU\n___(model_a)\n\n# Empty CUDA cache\n___()`,
    solution: `from hypernix.old_fridge import offload_to_cpu, chill_cache\n\noffload_to_cpu(model_a)\n\nchill_cache()`,
    hints: ['Import offload_to_cpu and chill_cache from hypernix.old_fridge', 'chill_cache() takes no arguments'],
  },

  // ── hypernix.mediocre_fridge ──────────────────────────────────────────────
  {
    id: 'judge-corpus',
    title: 'Generating Judge Data',
    track: 'Data Pipeline',
    trackColor: '#8866dd',
    concept: `synthesize_judge_corpus() generates synthetic preference pairs — (good answer, bad answer) tuples — for training a judge model. You control how many pairs (n) and where to write them. collect_responses_from() runs an existing model over prompts and scores them with a rubric.`,
    example: `from hypernix.mediocre_fridge import synthesize_judge_corpus

corpus = synthesize_judge_corpus(n=512, out_path="judge_data.txt")
print(f"Generated {len(corpus)} pairs")`,
    exercise: `Generate 256 judge preference pairs and save them to "my_judge.txt". Store the result in \`corpus\` and print its length.`,
    starter: `from hypernix.mediocre_fridge import ___\n\ncorpus = ___(n=___, out_path=___)\nprint(len(corpus))`,
    solution: `from hypernix.mediocre_fridge import synthesize_judge_corpus\n\ncorpus = synthesize_judge_corpus(n=256, out_path="my_judge.txt")\nprint(len(corpus))`,
    hints: ['Function is synthesize_judge_corpus from hypernix.mediocre_fridge', 'n=256, out_path="my_judge.txt"'],
  },

  // ── hypernix.new_fridge ───────────────────────────────────────────────────
  {
    id: 'parse-log',
    title: 'Parsing Training Logs',
    track: 'Visualization',
    trackColor: '#4a9eff',
    concept: `parse_training_log() extracts step/loss/lr records from raw training log text. plot_loss_curve() saves a PNG chart from those records. These two are always used together to visualize a training run.`,
    example: `from hypernix.new_fridge import parse_training_log, plot_loss_curve

log_text = open("train.log").read()
records = parse_training_log(log_text)
print(records[:3])
# [{'step': 1, 'loss': 4.23, 'lr': 3e-4}, ...]

plot_loss_curve(records, output_path="loss.png")`,
    exercise: `Read "my_run.log" into a string called \`log_text\`, parse it into \`records\`, then save a loss curve PNG to "charts/loss.png".`,
    starter: `from hypernix.new_fridge import ___, ___\n\nlog_text = open("my_run.log").read()\nrecords = ___(log_text)\n\n___(records, output_path=___)`,
    solution: `from hypernix.new_fridge import parse_training_log, plot_loss_curve\n\nlog_text = open("my_run.log").read()\nrecords = parse_training_log(log_text)\n\nplot_loss_curve(records, output_path="charts/loss.png")`,
    hints: ['Import parse_training_log and plot_loss_curve', 'plot_loss_curve(records, output_path="charts/loss.png")'],
  },

  // ── hypernix.new_range ────────────────────────────────────────────────────
  {
    id: 'new-range-score',
    title: 'First-Fail Rubric',
    track: 'Evaluation',
    trackColor: '#e05555',
    concept: `new_range.score() is the lightest evaluation rubric — zero dependencies. It applies a series of checks and returns True on the first pass (all pass) or False on first fail. Use it in tight training loops where you want cheap, fast quality checks.`,
    example: `from hypernix import new_range

response = "The capital of France is Paris."
passed = new_range.score(response)
print(passed)  # True or False`,
    exercise: `Import new_range from hypernix and score the string "Gradient descent minimizes the loss function." Store the boolean result in \`result\` and print it.`,
    starter: `from hypernix import ___\n\nresponse = "Gradient descent minimizes the loss function."\nresult = ___.score(response)\nprint(result)`,
    solution: `from hypernix import new_range\n\nresponse = "Gradient descent minimizes the loss function."\nresult = new_range.score(response)\nprint(result)`,
    hints: ['Import new_range from hypernix (not hypernix.new_range)', 'Call new_range.score(response)'],
  },
  {
    id: 'industrial-range',
    title: 'LLM-as-Judge Scoring',
    track: 'Evaluation',
    trackColor: '#e05555',
    concept: `industrial_range.score() uses another model as a judge. You pass the response and a judge_model (a CodeOven). It returns a dict with per-criterion scores and explanations — the highest-quality rubric, but slower because it calls a model.`,
    example: `from hypernix import industrial_range
from hypernix.old_oven import preheat

judge = preheat("nix2.5")
result = industrial_range.score(
    "The moon orbits Earth every 27 days.",
    judge_model=judge
)
print(result['overall_score'])`,
    exercise: `Load "nix2.5" as \`judge_oven\`, then use industrial_range.score() to evaluate the string "Neural networks learn via backpropagation." Pass the judge and store the full result dict in \`score_result\`.`,
    starter: `from hypernix import ___\nfrom hypernix.old_oven import preheat\n\njudge_oven = preheat("nix2.5")\n\nscore_result = ___.score(\n    "Neural networks learn via backpropagation.",\n    judge_model=___\n)\nprint(score_result)`,
    solution: `from hypernix import industrial_range\nfrom hypernix.old_oven import preheat\n\njudge_oven = preheat("nix2.5")\n\nscore_result = industrial_range.score(\n    "Neural networks learn via backpropagation.",\n    judge_model=judge_oven\n)\nprint(score_result)`,
    hints: ['Import industrial_range from hypernix', 'judge_model=judge_oven — pass your loaded CodeOven'],
  },

  // ── hypernix.smoke_alarm ──────────────────────────────────────────────────
  {
    id: 'gas-alarm',
    title: 'Training ETA with GasAlarm',
    track: 'Training',
    trackColor: '#c8192e',
    concept: `GasAlarm estimates training time using GPU-specific preset timing tables. Give it total steps, batch size, and the device string. It provides an ETA before training starts so you know how long to wait.`,
    example: `from hypernix.smoke_alarm import GasAlarm

alarm = GasAlarm(steps=5000, batch_size=4, device="cuda")
print(alarm.eta())   # "~2h 14m"
print(alarm.plan())  # step-by-step schedule dict`,
    exercise: `Create a GasAlarm for 2000 steps with batch_size=2 on "cuda". Store it in \`alarm\` and print its eta().`,
    starter: `from hypernix.smoke_alarm import ___\n\nalarm = ___(steps=___, batch_size=___, device=___)\nprint(alarm.eta())`,
    solution: `from hypernix.smoke_alarm import GasAlarm\n\nalarm = GasAlarm(steps=2000, batch_size=2, device="cuda")\nprint(alarm.eta())`,
    hints: ['Import GasAlarm from hypernix.smoke_alarm', 'Constructor: GasAlarm(steps, batch_size, device)'],
  },
  {
    id: 'smoke-check',
    title: 'Mid-Run NaN Detection',
    track: 'Training',
    trackColor: '#c8192e',
    concept: `smoke_alarm.check() inspects a model and loss value for NaN/Inf and raises RuntimeError immediately if found. Call it every N steps inside your training loop to catch silent corruption early — before it propagates across hundreds of steps.`,
    example: `from hypernix.smoke_alarm import check

for step, batch in enumerate(dataloader):
    loss = model(batch)
    
    if step % 50 == 0:
        check(model, loss)   # raises RuntimeError on NaN/Inf
    
    optimizer.step()`,
    exercise: `Write a training loop snippet that calls check(model, loss) every 100 steps. Use \`step\` as the loop counter and assume \`dataloader\` and \`optimizer\` exist.`,
    starter: `from hypernix.smoke_alarm import ___\n\nfor step, batch in enumerate(dataloader):\n    loss = model(batch)\n    \n    if step % ___ == 0:\n        ___(model, ___)\n    \n    optimizer.step()`,
    solution: `from hypernix.smoke_alarm import check\n\nfor step, batch in enumerate(dataloader):\n    loss = model(batch)\n    \n    if step % 100 == 0:\n        check(model, loss)\n    \n    optimizer.step()`,
    hints: ['Import check from hypernix.smoke_alarm', 'check(model, loss) — call it inside the if block'],
  },
  {
    id: 'storage-warning',
    title: 'Disk Space Guard',
    track: 'Training',
    trackColor: '#c8192e',
    concept: `storage_warning() warns if free disk space at a path falls below a threshold. Call it before long training runs — running out of disk mid-training kills the job and corrupts checkpoints. The default threshold is 5.0 GB.`,
    example: `from hypernix.smoke_alarm import storage_warning

# Warn if less than 10 GB free in the output dir
storage_warning("./checkpoints", min_gb=10.0)

# Now safe to start training
train(model, dataset, out_dir="./checkpoints")`,
    exercise: `Before training, check that "./output" has at least 8 GB free using storage_warning(), then call train() with 500 steps.`,
    starter: `from hypernix.smoke_alarm import ___\nfrom hypernix.train import train\n\n___(___,  min_gb=___)\n\ntrain(model, dataset, out_dir="./output", steps=___)`,
    solution: `from hypernix.smoke_alarm import storage_warning\nfrom hypernix.train import train\n\nstorage_warning("./output", min_gb=8.0)\n\ntrain(model, dataset, out_dir="./output", steps=500)`,
    hints: ['storage_warning(path, min_gb=8.0)', 'Then call train(model, dataset, out_dir="./output", steps=500)'],
  },

  // ── hypernix.microwave ────────────────────────────────────────────────────
  {
    id: 'microwave-tiers',
    title: 'Throwaway Inference Tiers',
    track: 'Inference',
    trackColor: '#e8960a',
    concept: `microwave provides 5 inference tiers. defrost() loads a model. low_zap() is greedy/fastest. zap() adds sampling. high_zap() uses nucleus sampling. chat_zap() handles multi-turn. Pick the tier that matches your quality vs speed tradeoff.`,
    example: `from hypernix.microwave import defrost, zap, high_zap

model = defrost("./my-model")

# Standard sampling
result = zap(model, "Tell me about Paris", max_tokens=128)

# Higher quality
result = high_zap(model, "Tell me about Paris",
                  max_tokens=256, top_p=0.95)`,
    exercise: `Load "./trained" with defrost(), then run high_zap() on it with the prompt "Explain transformers" using max_tokens=200 and top_p=0.9. Store in \`out\`.`,
    starter: `from hypernix.microwave import ___, ___\n\nmodel = ___("./trained")\n\nout = ___(\n    model,\n    "Explain transformers",\n    max_tokens=___,\n    top_p=___\n)\nprint(out)`,
    solution: `from hypernix.microwave import defrost, high_zap\n\nmodel = defrost("./trained")\n\nout = high_zap(\n    model,\n    "Explain transformers",\n    max_tokens=200,\n    top_p=0.9\n)\nprint(out)`,
    hints: ['Import defrost and high_zap', 'high_zap(model, prompt, max_tokens, top_p)'],
  },
  {
    id: 'microwave-reheat',
    title: 'Continuing a Generation',
    track: 'Inference',
    trackColor: '#e8960a',
    concept: `reheat() continues from a prior generation without reloading the model. You pass the model, the prior output, and a continuation prompt. This avoids the KV cache warmup cost of starting fresh and is ideal for streaming long documents in chunks.`,
    example: `from hypernix.microwave import defrost, zap, reheat

model = defrost("./my-model")
part1 = zap(model, "Chapter 1: The beginning")

# Continue seamlessly
part2 = reheat(model, prior_output=part1,
               continuation_prompt="Continue the story.")
print(part1 + part2)`,
    exercise: `After getting \`first_half\` from zap(), use reheat() to get \`second_half\` that continues it. The continuation_prompt should be "Keep going, more detail."`,
    starter: `from hypernix.microwave import defrost, zap, ___\n\nmodel = defrost("./my-model")\nfirst_half = zap(model, "The history of computing")\n\nsecond_half = ___(\n    model,\n    prior_output=___,\n    continuation_prompt=___\n)\nprint(first_half + second_half)`,
    solution: `from hypernix.microwave import defrost, zap, reheat\n\nmodel = defrost("./my-model")\nfirst_half = zap(model, "The history of computing")\n\nsecond_half = reheat(\n    model,\n    prior_output=first_half,\n    continuation_prompt="Keep going, more detail."\n)\nprint(first_half + second_half)`,
    hints: ['Import reheat from hypernix.microwave', 'reheat(model, prior_output=first_half, continuation_prompt="...")'],
  },

  // ── hypernix.table ────────────────────────────────────────────────────────
  {
    id: 'table-inspect',
    title: 'Inspecting Training Logs',
    track: 'Visualization',
    trackColor: '#4a9eff',
    concept: `Table is a lightweight tabular viewer. from_training_log() parses a log string into a Table. You can then chain filter() and select() to drill into specific rows and columns, and call show() to print a formatted view.`,
    example: `from hypernix.table import from_training_log

log = open("train.log").read()
t = from_training_log(log)

# Show only steps where loss > 2.0
t.filter(lambda r: r['loss'] > 2.0).show(n=10)`,
    exercise: `Parse "run.log" into a Table called \`t\`, then filter to rows where step > 500, select only the "step" and "loss" columns, and show 5 rows.`,
    starter: `from hypernix.table import ___\n\nt = ___(open("run.log").read())\n\nt.filter(lambda r: r[___] > 500)\\\n .select(___, ___)\\\n .show(n=___)`,
    solution: `from hypernix.table import from_training_log\n\nt = from_training_log(open("run.log").read())\n\nt.filter(lambda r: r['step'] > 500)\\\n .select('step', 'loss')\\\n .show(n=5)`,
    hints: ['from_training_log(log_text) creates the Table', 'Chain .filter().select("step","loss").show(n=5)'],
  },

  // ── hypernix.sink ─────────────────────────────────────────────────────────
  {
    id: 'sink-basic',
    title: 'Append-Only File Sink',
    track: 'Data Pipeline',
    trackColor: '#8866dd',
    concept: `Sink is an append-only file writer — ideal for streaming training outputs or scraping results without loading everything into memory. Call pour() to write records and flush() to force a buffer flush. Set rotate_mb to rotate on size.`,
    example: `from hypernix.sink import Sink

s = Sink("outputs.jsonl", rotate_mb=100, dedupe=True)

for record in my_records:
    s.pour([record])

s.flush()`,
    exercise: `Create a Sink writing to "scraped.jsonl" with 50 MB rotation and deduplication enabled. Pour the list \`my_data\` into it, then flush.`,
    starter: `from hypernix.sink import ___\n\ns = ___(___,\n    rotate_mb=___,\n    dedupe=___\n)\n\ns.pour(my_data)\ns.___()`,
    solution: `from hypernix.sink import Sink\n\ns = Sink("scraped.jsonl",\n    rotate_mb=50,\n    dedupe=True\n)\n\ns.pour(my_data)\ns.flush()`,
    hints: ['Sink("path", rotate_mb=50, dedupe=True)', 's.flush() forces a buffer flush'],
  },

  // ── hypernix.instant_pot ──────────────────────────────────────────────────
  {
    id: 'instant-pot-brew',
    title: 'One-Shot Pipeline with brew()',
    track: 'Training',
    trackColor: '#c8192e',
    concept: `instant_pot.brew() runs an entire end-to-end pipeline from a single recipe dict. The recipe describes what to download, how to preprocess, training config, and where to upload. It's the fastest path from data to deployed model.`,
    example: `from hypernix.instant_pot import brew

recipe = {
    "model": "nix2.5",
    "dataset": "ray0rf1re/Su",
    "steps": 1000,
    "out_dir": "./run",
    "upload_to": "ray0rf1re/my-fine-tune"
}

brew(recipe)`,
    exercise: `Write a brew() call using a recipe dict that fine-tunes "llama-3.2-1b" on "my-org/dataset" for 500 steps, saving to "./out", and uploading to "my-org/fine-tuned".`,
    starter: `from hypernix.instant_pot import ___\n\nrecipe = {\n    "model": ___,\n    "dataset": ___,\n    "steps": ___,\n    "out_dir": ___,\n    "upload_to": ___\n}\n\n___(recipe)`,
    solution: `from hypernix.instant_pot import brew\n\nrecipe = {\n    "model": "llama-3.2-1b",\n    "dataset": "my-org/dataset",\n    "steps": 500,\n    "out_dir": "./out",\n    "upload_to": "my-org/fine-tuned"\n}\n\nbrew(recipe)`,
    hints: ['Import brew from hypernix.instant_pot', 'recipe is a plain Python dict, then call brew(recipe)'],
  },

  // ── hypernix.toaster ──────────────────────────────────────────────────────
  {
    id: 'toaster-basic',
    title: 'Formatting Lines with Toaster',
    track: 'Data Pipeline',
    trackColor: '#8866dd',
    concept: `Toaster formats raw text lines into model-ready strings. TwoSliceToaster does basic strip+normalize. FourSliceToaster also applies a chat template. ConveyorToaster processes in batches for large datasets. ToasterOven is fully configurable.`,
    example: `from hypernix.toaster import FourSliceToaster

lines = [
    "explain gradient descent",
    "what is a transformer",
]

formatted = FourSliceToaster(lines)
for f in formatted:
    print(f)`,
    exercise: `Use ConveyorToaster to format \`raw_lines\` in batches of 32. Store the result in \`formatted\`.`,
    starter: `from hypernix.toaster import ___\n\nraw_lines = ["hello world", "train a model", "what is loss?"]\n\nformatted = ___(raw_lines, batch_size=___)`,
    solution: `from hypernix.toaster import ConveyorToaster\n\nraw_lines = ["hello world", "train a model", "what is loss?"]\n\nformatted = ConveyorToaster(raw_lines, batch_size=32)`,
    hints: ['Import ConveyorToaster from hypernix.toaster', 'ConveyorToaster(lines, batch_size=32)'],
  },
  {
    id: 'toaster-oven',
    title: 'Custom Template Formatting',
    track: 'Data Pipeline',
    trackColor: '#8866dd',
    concept: `ToasterOven is Tier 4 — fully configurable per-line formatting. You provide a template string with {text} as the placeholder. This is used when you need non-standard formatting like adding prefix/suffix tokens, special separators, or BOS/EOS wrapping.`,
    example: `from hypernix.toaster import ToasterOven

lines = ["What is PyTorch?", "Explain Adam optimizer."]
template = "<|user|>{text}<|end|><|assistant|>"

formatted = ToasterOven(lines, template=template)`,
    exercise: `Use ToasterOven to format \`lines\` with a Llama-3 style template: "\\<|user|\\>\\n{text}\\n\\<|eot_id|\\>\\n\\<|assistant|\\>". Store the result in \`out\`.`,
    starter: `from hypernix.toaster import ___\n\nlines = ["Hello!", "Explain backprop."]\ntemplate = "<|user|>\\n{text}\\n<|eot_id|>\\n<|assistant|>"\n\nout = ___(lines, template=___)`,
    solution: `from hypernix.toaster import ToasterOven\n\nlines = ["Hello!", "Explain backprop."]\ntemplate = "<|user|>\\n{text}\\n<|eot_id|>\\n<|assistant|>"\n\nout = ToasterOven(lines, template=template)`,
    hints: ['Import ToasterOven from hypernix.toaster', 'ToasterOven(lines, template=template)'],
  },

  // ── hypernix.smoker ───────────────────────────────────────────────────────
  {
    id: 'useable-smoker',
    title: 'Smoke-Testing Training',
    track: 'Evaluation',
    trackColor: '#e05555',
    concept: `UseableSmoker runs a minimal training smoke-test to verify the model, optimizer, and dataset work together without crashing. It's the first thing to run after any architecture change. If this passes, you know the forward/backward pass is intact.`,
    example: `from hypernix.smoker import UseableSmoker

smoker = UseableSmoker(model, dataset)
smoker.run(steps=10)   # Fast: just check it doesn't crash
print("Smoke test passed!")`,
    exercise: `Run a UseableSmoker on \`model\` and \`train_dataset\` for 20 steps and print "All clear" if it completes without exception.`,
    starter: `from hypernix.smoker import ___\n\nsmoker = ___(model, train_dataset)\nsmoker.run(steps=___)\nprint("All clear")`,
    solution: `from hypernix.smoker import UseableSmoker\n\nsmoker = UseableSmoker(model, train_dataset)\nsmoker.run(steps=20)\nprint("All clear")`,
    hints: ['UseableSmoker(model, dataset)', 'smoker.run(steps=20)'],
  },
  {
    id: 'commercial-smoker',
    title: 'Quality-Gated Training',
    track: 'Evaluation',
    trackColor: '#e05555',
    concept: `CommercialSmoker adds quality thresholds — training halts automatically if your eval suite misses a threshold. Use it when training must meet minimum quality before a checkpoint is saved. HighQualitySmoker does the same but across multiple seeds for reproducibility.`,
    example: `from hypernix.smoker import CommercialSmoker

smoker = CommercialSmoker(
    model, dataset,
    evals=my_eval_suite,
    thresholds={"accuracy": 0.75, "perplexity": 12.0}
)
smoker.run(steps=2000)`,
    exercise: `Create a CommercialSmoker for \`model\` and \`dataset\` with thresholds accuracy=0.7 and perplexity=15.0, then run it for 1000 steps.`,
    starter: `from hypernix.smoker import ___\n\nsmoker = ___(\n    model, dataset,\n    evals=my_evals,\n    thresholds={___: 0.7, ___: 15.0}\n)\nsmoker.run(steps=___)`,
    solution: `from hypernix.smoker import CommercialSmoker\n\nsmoker = CommercialSmoker(\n    model, dataset,\n    evals=my_evals,\n    thresholds={"accuracy": 0.7, "perplexity": 15.0}\n)\nsmoker.run(steps=1000)`,
    hints: ['CommercialSmoker(model, dataset, evals, thresholds)', 'thresholds is a dict: {"accuracy": 0.7, "perplexity": 15.0}'],
  },

  // ── hypernix.deep_fryer ───────────────────────────────────────────────────
  {
    id: 'light-fry',
    title: 'Weight Perturbation',
    track: 'Advanced Training',
    trackColor: '#ff6b35',
    concept: `deep_fryer perturbs model weights. LightFry adds small Gaussian noise (good as a regulariser). HeavyFry applies severe perturbation to generate "bad model" negatives for contrastive training. Always snapshot() before frying so you can restore() after.`,
    example: `from hypernix.deep_fryer import LightFry, snapshot, restore

snap = snapshot(model)           # save clean weights
LightFry(model, noise_scale=0.01)  # add small noise

# ... use perturbed model ...

restore(model, snap)             # back to original`,
    exercise: `Snapshot \`model\` into \`snap\`, apply LightFry with noise_scale=0.005, use the model, then restore it from \`snap\`.`,
    starter: `from hypernix.deep_fryer import ___, ___, ___\n\nsnap = ___(model)\n___(model, noise_scale=___)\n\n# ... use perturbed model ...\n\n___(model, snap)`,
    solution: `from hypernix.deep_fryer import LightFry, snapshot, restore\n\nsnap = snapshot(model)\nLightFry(model, noise_scale=0.005)\n\n# ... use perturbed model ...\n\nrestore(model, snap)`,
    hints: ['Import LightFry, snapshot, restore from hypernix.deep_fryer', 'snapshot → LightFry → restore'],
  },

  // ── hypernix.cake_pan ─────────────────────────────────────────────────────
  {
    id: 'bakeoff-context',
    title: 'Safe Training with BakeOff',
    track: 'Advanced Training',
    trackColor: '#ff6b35',
    concept: `BakeOff is a context manager that wraps training with automatic NaN/Inf detection, memory-pressure offload, and wall-time watchdog. On NaN or OOM it rolls back to the pinned pre-run state. Use it for long runs where silent corruption is risky.`,
    example: `from hypernix.cake_pan import BakeOff

with BakeOff(model, dataset, max_hours=4) as run:
    run.train(steps=10000)
    # On NaN/OOM: auto-rollback to start of this block
    # On wall-time limit: clean checkpoint + exit`,
    exercise: `Wrap a 5000-step training run in a BakeOff context manager with a 2-hour wall-time limit. Use \`model\` and \`dataset\`.`,
    starter: `from hypernix.cake_pan import ___\n\nwith ___(model, dataset, max_hours=___) as run:\n    run.train(steps=___)`,
    solution: `from hypernix.cake_pan import BakeOff\n\nwith BakeOff(model, dataset, max_hours=2) as run:\n    run.train(steps=5000)`,
    hints: ['BakeOff is a context manager — use the with statement', 'BakeOff(model, dataset, max_hours=2)'],
  },

  // ── hypernix.salt_shaker ──────────────────────────────────────────────────
  {
    id: 'salt-augment',
    title: 'Data Augmentation (Salt)',
    track: 'Data Pipeline',
    trackColor: '#8866dd',
    concept: `salt_shaker provides gentle augmentation. FromTheBag does random synonym substitution (lightest). HandCrusher paraphrases. PoshSaltDish does full style-transfer. These are applied to training data to reduce overfitting on exact phrasing.`,
    example: `from hypernix.salt_shaker import FromTheBag, HandCrusher

texts = ["The optimizer converges quickly.", "Batch normalization helps."]

# Light synonym substitution
augmented = FromTheBag(texts, p=0.1)

# Controlled paraphrase
paraphrased = HandCrusher(texts, p=0.15)`,
    exercise: `Use HandCrusher to augment the list \`training_texts\` with p=0.12. Store the result in \`augmented\`.`,
    starter: `from hypernix.salt_shaker import ___\n\ntraining_texts = [\n    "Attention is all you need.",\n    "Transformers dominate NLP.",\n]\n\naugmented = ___(training_texts, p=___)`,
    solution: `from hypernix.salt_shaker import HandCrusher\n\ntraining_texts = [\n    "Attention is all you need.",\n    "Transformers dominate NLP.",\n]\n\naugmented = HandCrusher(training_texts, p=0.12)`,
    hints: ['HandCrusher from hypernix.salt_shaker', 'HandCrusher(texts, p=0.12)'],
  },

  // ── hypernix.pepper_shaker ────────────────────────────────────────────────
  {
    id: 'pepper-perturb',
    title: 'Adversarial Perturbations',
    track: 'Data Pipeline',
    trackColor: '#8866dd',
    concept: `pepper_shaker creates adversarial perturbations. SmallShaker masks tokens (MLM-style). Dish injects keyboard typos. TallHandmade adds semantic negations. These create hard negatives and robustness training data.`,
    example: `from hypernix.pepper_shaker import SmallShaker, TallHandmade

texts = ["Paris is the capital of France."]

# Mask 15% of tokens
masked = SmallShaker(texts, mask_prob=0.15)

# Inject negations
negated = TallHandmade(texts, negation_rate=0.2)`,
    exercise: `Use Dish to inject typos into \`clean_texts\` at a typo_rate of 0.05. Store the result in \`noisy\`.`,
    starter: `from hypernix.pepper_shaker import ___\n\nclean_texts = [\n    "The loss decreases during training.",\n    "Dropout prevents overfitting.",\n]\n\nnoisy = ___(clean_texts, typo_rate=___)`,
    solution: `from hypernix.pepper_shaker import Dish\n\nclean_texts = [\n    "The loss decreases during training.",\n    "Dropout prevents overfitting.",\n]\n\nnoisy = Dish(clean_texts, typo_rate=0.05)`,
    hints: ['Import Dish from hypernix.pepper_shaker', 'Dish(texts, typo_rate=0.05)'],
  },

  // ── hypernix.torch_compat ─────────────────────────────────────────────────
  {
    id: 'torch-compat-shim',
    title: 'Legacy Torch Compatibility',
    track: 'Advanced Training',
    trackColor: '#ff6b35',
    concept: `torch_compat provides shims for running HyperNix on older torch builds (macOS Intel, old servers). rms_norm() replaces torch.nn.RMSNorm which didn't exist before torch 2.0. sdpa() replaces F.scaled_dot_product_attention. is_legacy_torch() detects torch < 2.0.`,
    example: `from hypernix.torch_compat import rms_norm, sdpa, is_legacy_torch

if is_legacy_torch():
    print("Using compat shims for torch < 2.0")

# Drop-in replacements
x_normed = rms_norm(x, weight=norm_weight, eps=1e-6)
out = sdpa(q, k, v)`,
    exercise: `Check if we're on a legacy torch version with is_legacy_torch(). If True, apply rms_norm to tensor \`hidden\` using \`w\` as the weight. If False, use torch.nn.functional.rms_norm.`,
    starter: `from hypernix.torch_compat import ___, ___\nimport torch.nn.functional as F\n\nif ___():\n    normed = ___(hidden, weight=w)\nelse:\n    normed = F.rms_norm(hidden, hidden.shape[-1:], weight=w)`,
    solution: `from hypernix.torch_compat import rms_norm, is_legacy_torch\nimport torch.nn.functional as F\n\nif is_legacy_torch():\n    normed = rms_norm(hidden, weight=w)\nelse:\n    normed = F.rms_norm(hidden, hidden.shape[-1:], weight=w)`,
    hints: ['Import rms_norm and is_legacy_torch from hypernix.torch_compat', 'is_legacy_torch() returns bool; rms_norm(tensor, weight=w)'],
  },

  // ── hypernix.whisk ────────────────────────────────────────────────────────
  {
    id: 'checkpoint-averaging',
    title: 'Averaging Checkpoints',
    track: 'Advanced Training',
    trackColor: '#ff6b35',
    concept: `Averaging multiple checkpoints from the same training run often beats any single checkpoint — especially for generalization. average_checkpoints() takes a list of paths and merges the weights arithmetically. The result is saved to output_path.`,
    example: `from hypernix.whisk import average_checkpoints

checkpoints = [
    "./run/ckpt-1000.pt",
    "./run/ckpt-2000.pt",
    "./run/ckpt-3000.pt",
]

average_checkpoints(checkpoints, output_path="./run/averaged.pt")`,
    exercise: `Average three checkpoints at "ckpt-500.pt", "ckpt-1000.pt", and "ckpt-1500.pt", saving to "final.pt".`,
    starter: `from hypernix.whisk import ___\n\n___([\n    "___",\n    "___",\n    "___",\n], output_path=___)`,
    solution: `from hypernix.whisk import average_checkpoints\n\naverage_checkpoints([\n    "ckpt-500.pt",\n    "ckpt-1000.pt",\n    "ckpt-1500.pt",\n], output_path="final.pt")`,
    hints: ['average_checkpoints from hypernix.whisk', 'average_checkpoints(list_of_paths, output_path="final.pt")'],
  },
  {
    id: 'ema-tracking',
    title: 'Exponential Moving Average',
    track: 'Advanced Training',
    trackColor: '#ff6b35',
    concept: `EMA tracks a shadow copy of model weights that moves slowly (controlled by decay). EMA weights are smoother than live weights and often generalize better at eval. Call update() every step, then apply_shadow() to copy EMA weights into the model for inference.`,
    example: `from hypernix.whisk import EMA

ema = EMA(model, decay=0.999)

for batch in dataloader:
    loss = model(batch)
    optimizer.step()
    ema.update(model)    # update shadow weights

# Inference with EMA weights
ema.apply_shadow(model)
result = model.generate(prompt)`,
    exercise: `Create an EMA with decay=0.9999 on \`model\`. In a loop over \`dataloader\`, call optimizer.step() then ema.update(model). After the loop, apply the shadow weights for inference.`,
    starter: `from hypernix.whisk import ___\n\nema = ___(model, decay=___)\n\nfor batch in dataloader:\n    loss = model(batch)\n    optimizer.step()\n    ___.update(___)\n\n___.apply_shadow(model)`,
    solution: `from hypernix.whisk import EMA\n\nema = EMA(model, decay=0.9999)\n\nfor batch in dataloader:\n    loss = model(batch)\n    optimizer.step()\n    ema.update(model)\n\nema.apply_shadow(model)`,
    hints: ['EMA(model, decay=0.9999)', 'ema.update(model) each step; ema.apply_shadow(model) after'],
  },

  // ── hypernix.cutting_board ────────────────────────────────────────────────
  {
    id: 'dataset-split',
    title: 'Train/Val/Test Splitting',
    track: 'Data Pipeline',
    trackColor: '#8866dd',
    concept: `cutting_board.split() divides a dataset into train/val/test subsets. Proportions must sum to 1.0. Always set a seed for reproducibility — the same seed always gives the same split, which is critical for fair evaluation.`,
    example: `from hypernix.cutting_board import split

train, val, test = split(
    my_dataset,
    train=0.8, val=0.1, test=0.1,
    seed=42
)
print(len(train), len(val), len(test))`,
    exercise: `Split \`full_dataset\` into 70% train, 15% val, 15% test with seed=0. Unpack into \`train\`, \`val\`, \`test\` and print their lengths.`,
    starter: `from hypernix.cutting_board import ___\n\ntrain, val, test = ___(\n    full_dataset,\n    train=___, val=___, test=___,\n    seed=___\n)\nprint(len(train), len(val), len(test))`,
    solution: `from hypernix.cutting_board import split\n\ntrain, val, test = split(\n    full_dataset,\n    train=0.7, val=0.15, test=0.15,\n    seed=0\n)\nprint(len(train), len(val), len(test))`,
    hints: ['split from hypernix.cutting_board', 'train=0.7, val=0.15, test=0.15 — they must sum to 1.0'],
  },
  {
    id: 'kfold-split',
    title: 'K-Fold Cross Validation',
    track: 'Data Pipeline',
    trackColor: '#8866dd',
    concept: `kfold() generates k non-overlapping folds of the dataset. Each call to the iterator yields a (train, val) tuple. Use k-fold when your dataset is small and you want a more robust evaluation than a single train/val split.`,
    example: `from hypernix.cutting_board import kfold

for fold_idx, (train, val) in enumerate(kfold(dataset, k=5)):
    model = train_model(train)
    score = evaluate(model, val)
    print(f"Fold {fold_idx}: {score:.3f}")`,
    exercise: `Run 3-fold cross validation on \`small_dataset\` with seed=99. Print the fold index and the length of each val set.`,
    starter: `from hypernix.cutting_board import ___\n\nfor i, (train, val) in enumerate(___(\n    small_dataset, k=___, seed=___\n)):\n    print(f"Fold {i}: val size = {len(val)}")`,
    solution: `from hypernix.cutting_board import kfold\n\nfor i, (train, val) in enumerate(kfold(\n    small_dataset, k=3, seed=99\n)):\n    print(f"Fold {i}: val size = {len(val)}")`,
    hints: ['kfold from hypernix.cutting_board', 'kfold(dataset, k=3, seed=99) — iterate with enumerate'],
  },

  // ── hypernix.countertop / bell / flour ────────────────────────────────────
  {
    id: 'countertop-session',
    title: 'Chat Sessions with Countertop',
    track: 'Inference',
    trackColor: '#e8960a',
    concept: `Countertop creates a stateful Session object that tracks conversation history. Session.add() appends messages. Session.render() returns the full history as a list of {role, content} dicts — ready to pass to any model's chat() method.`,
    example: `from hypernix.countertop import Countertop

session = Countertop(system_prompt="You are a helpful assistant.")
session.add("user", "What is HyperNix?")
session.add("assistant", "HyperNix is a PyTorch LLM toolkit.")
session.add("user", "What modules does it have?")

messages = session.render()
print(messages)`,
    exercise: `Create a Countertop session with system prompt "You are a code tutor.". Add a user message "What is a tensor?" and render the result into \`msgs\`.`,
    starter: `from hypernix.countertop import ___\n\nsession = ___(system_prompt=___)\nsession.add(___, "What is a tensor?")\n\nmsgs = session.render()\nprint(msgs)`,
    solution: `from hypernix.countertop import Countertop\n\nsession = Countertop(system_prompt="You are a code tutor.")\nsession.add("user", "What is a tensor?")\n\nmsgs = session.render()\nprint(msgs)`,
    hints: ['Countertop from hypernix.countertop', 'session.add("user", "...") — role is "user" or "assistant"'],
  },
  {
    id: 'bell-flour',
    title: 'Ringing the Bell & Flour Templates',
    track: 'Inference',
    trackColor: '#e8960a',
    concept: `bell() sends a Session to a model and returns the response string. flour() applies a named chat template to raw text — converting it to the format the model expects (ChatML, Llama-3, etc.). Use flour() before bell() when building from raw text instead of Session objects.`,
    example: `from hypernix.countertop import Countertop
from hypernix.bell import bell
from hypernix.flour import flour

# Apply template to raw text
formatted = flour("Explain attention.", template="chatml")

# Build session and ring model
session = Countertop()
session.add("user", formatted)
response = bell(session, model=my_oven)
print(response)`,
    exercise: `Apply flour() with template "llama3" to "Summarize this text: ...", then create a Countertop session, add the result as a user message, and call bell() with \`my_oven\` to get the response.`,
    starter: `from hypernix.countertop import Countertop\nfrom hypernix.bell import ___\nfrom hypernix.flour import ___\n\nformatted = ___("Summarize this text: ...", template=___)\n\nsession = Countertop()\nsession.add("user", formatted)\nresponse = ___(session, model=my_oven)\nprint(response)`,
    solution: `from hypernix.countertop import Countertop\nfrom hypernix.bell import bell\nfrom hypernix.flour import flour\n\nformatted = flour("Summarize this text: ...", template="llama3")\n\nsession = Countertop()\nsession.add("user", formatted)\nresponse = bell(session, model=my_oven)\nprint(response)`,
    hints: ['flour(text, template="llama3") — then session.add("user", formatted)', 'bell(session, model=my_oven) returns the response string'],
  },

  // ── hypernix.convert advanced ─────────────────────────────────────────────
  {
    id: 'infer-arch',
    title: 'Architecture Detection',
    track: 'Ship',
    trackColor: '#00b5ad',
    concept: `infer_arch() inspects a model directory's tensor names and returns the architecture string (e.g. "llama", "mistral", "qwen"). This is needed when convert scripts need to know the model family. It's architecture-agnostic and works on any safetensors checkpoint.`,
    example: `from hypernix.convert import infer_arch, convert_to_gguf_fp16

arch = infer_arch("./my-model")
print(f"Detected: {arch}")   # "llama", "qwen", etc.

# Use arch in logging/validation before convert
convert_to_gguf_fp16("./my-model", f"./my-model-{arch}-f16.gguf")`,
    exercise: `Detect the architecture of "./downloaded-model", print it, then convert to fp16 GGUF using the detected arch name in the output filename.`,
    starter: `from hypernix.convert import ___, ___\n\narch = ___("./downloaded-model")\nprint(f"Arch: {arch}")\n\n___(./downloaded-model",\n    f"./downloaded-model-{arch}-f16.gguf")`,
    solution: `from hypernix.convert import infer_arch, convert_to_gguf_fp16\n\narch = infer_arch("./downloaded-model")\nprint(f"Arch: {arch}")\n\nconvert_to_gguf_fp16("./downloaded-model",\n    f"./downloaded-model-{arch}-f16.gguf")`,
    hints: ['infer_arch and convert_to_gguf_fp16 from hypernix.convert', 'infer_arch("./path") returns a string like "llama"'],
  },

  // ── hypernix.quantize advanced ────────────────────────────────────────────
  {
    id: 'list-quant-types',
    title: 'Available Quant Types',
    track: 'Ship',
    trackColor: '#00b5ad',
    concept: `list_quant_types() returns all 30 supported quantisation aliases. Use it to discover what types your llama-quantize binary supports before deciding which to produce. Q4_K_M is the default best quality/size. Q8_0 is near-lossless. IQ types are importance-matrix quantised.`,
    example: `from hypernix.quantize import list_quant_types, quantize

# Discover all options
types = list_quant_types()
print(types)
# ['q4_k_m', 'q8_0', 'q6_k', 'q5_k_m', 'q4_0', 'iq4_xs', ...]

# Produce multiple quants
for qt in ["q4_k_m", "q8_0", "q6_k"]:
    quantize("model.gguf", quant_type=qt)`,
    exercise: `Print all available quant types, then quantize "big-model-f16.gguf" to both "q4_k_m" and "q5_k_m" in a loop.`,
    starter: `from hypernix.quantize import ___, ___\n\nfor qt in ___():\n    print(qt)\n\nfor qt in [___, ___]:\n    ___("big-model-f16.gguf", quant_type=qt)`,
    solution: `from hypernix.quantize import list_quant_types, quantize\n\nfor qt in list_quant_types():\n    print(qt)\n\nfor qt in ["q4_k_m", "q5_k_m"]:\n    quantize("big-model-f16.gguf", quant_type=qt)`,
    hints: ['list_quant_types() and quantize from hypernix.quantize', 'Loop: for qt in ["q4_k_m", "q5_k_m"]: quantize(path, quant_type=qt)'],
  },

  // ── hypernix.espresso advanced ────────────────────────────────────────────
  {
    id: 'double-shot-judge',
    title: 'Dual-Scored Evaluation',
    track: 'Evaluation',
    trackColor: '#e05555',
    concept: `DoubleShot adds a second LLM judge to cross-check scores. The judge model independently scores each response. Agreement between rubric and judge gives higher confidence. Use DoubleShot when a single rubric might be gamed by the model you're evaluating.`,
    example: `from hypernix.espresso_maker import DoubleShot
from hypernix.old_oven import preheat
from hypernix import old_range

oven = preheat("my-model")
judge = preheat("nix2.5")   # use a trusted judge

results = DoubleShot(
    oven, my_prompts,
    rubric=old_range,
    judge=judge
)
print(results.agreement_rate())`,
    exercise: `Run a DoubleShot evaluation on \`candidate_oven\` over \`eval_prompts\` using old_range as rubric and \`trusted_judge\` as judge. Store in \`results\` and print agreement_rate().`,
    starter: `from hypernix.espresso_maker import ___\nfrom hypernix import old_range\n\nresults = ___(\n    candidate_oven,\n    eval_prompts,\n    rubric=___,\n    judge=___\n)\nprint(results.agreement_rate())`,
    solution: `from hypernix.espresso_maker import DoubleShot\nfrom hypernix import old_range\n\nresults = DoubleShot(\n    candidate_oven,\n    eval_prompts,\n    rubric=old_range,\n    judge=trusted_judge\n)\nprint(results.agreement_rate())`,
    hints: ['DoubleShot from hypernix.espresso_maker', 'DoubleShot(model, prompts, rubric=old_range, judge=trusted_judge)'],
  },

  // ── Full pipeline exercises (advanced) ────────────────────────────────────
  {
    id: 'full-data-pipeline',
    title: 'Full Preprocessing Pipeline',
    track: 'Advanced Pipelines',
    trackColor: '#ff6b35',
    concept: `A production data pipeline chains FryingPan → Skillet → Wok, then dumps to a Sink. FryingPan deduplicates, Skillet applies the chat template, Wok packs sequences into Arrow format. Sink writes the Arrow rows to disk without holding everything in memory.`,
    example: `from hypernix.pans import FryingPan, Skillet, Wok
from hypernix.sink import Sink

raw = load_raw_data("corpus.jsonl")
cleaned  = FryingPan(raw).process()
templated = Skillet(cleaned).process(template="chatml")
packed   = Wok(templated).process()

sink = Sink("dataset.arrow")
sink.pour(packed)
sink.flush()`,
    exercise: `Chain FryingPan → Skillet (template="llama3") → Wok on \`raw_data\`. Write the packed output to a Sink at "final.arrow" and flush it.`,
    starter: `from hypernix.pans import ___, ___, ___\nfrom hypernix.sink import ___\n\ncleaned   = ___(raw_data).process()\ntemplated = ___(cleaned).process(template=___)\npacked    = ___(templated).process()\n\nsink = ___("final.arrow")\nsink.pour(packed)\nsink.flush()`,
    solution: `from hypernix.pans import FryingPan, Skillet, Wok\nfrom hypernix.sink import Sink\n\ncleaned   = FryingPan(raw_data).process()\ntemplated = Skillet(cleaned).process(template="llama3")\npacked    = Wok(templated).process()\n\nsink = Sink("final.arrow")\nsink.pour(packed)\nsink.flush()`,
    hints: ['Chain: FryingPan → Skillet → Wok, all .process()', 'Sink("final.arrow"); sink.pour(packed); sink.flush()'],
  },
  {
    id: 'full-ship-pipeline',
    title: 'Full Ship Pipeline',
    track: 'Advanced Pipelines',
    trackColor: '#ff6b35',
    concept: `The full ship pipeline is: (1) convert safetensors → fp16 GGUF, (2) quantize to Q4_K_M, (3) push both to HuggingFace Hub, (4) generate a model card. This is the exact sequence used to release a HyperNix-trained model publicly.`,
    example: `from hypernix.convert import convert_to_gguf_fp16
from hypernix.quantize import quantize
from hypernix.upload import push_to_hub, create_model_card

convert_to_gguf_fp16("./model", "./model-f16.gguf")
quantize("./model-f16.gguf", quant_type="q4_k_m")
push_to_hub("./", "user/my-model", token=TOKEN)
create_model_card("user/my-model", license="apache-2.0")`,
    exercise: `Execute the full ship pipeline for "./my-trained-model" → "ray0rf1re/shipped-model". Convert to fp16, quantize to q8_0, push to hub with \`MY_TOKEN\`, then create a model card with license "mit".`,
    starter: `from hypernix.convert import ___\nfrom hypernix.quantize import ___\nfrom hypernix.upload import ___, ___\n\n___("./my-trained-model", "./model-f16.gguf")\n___("./model-f16.gguf", quant_type=___)\n___("./", "ray0rf1re/shipped-model", token=MY_TOKEN)\n___("ray0rf1re/shipped-model", license=___)`,
    solution: `from hypernix.convert import convert_to_gguf_fp16\nfrom hypernix.quantize import quantize\nfrom hypernix.upload import push_to_hub, create_model_card\n\nconvert_to_gguf_fp16("./my-trained-model", "./model-f16.gguf")\nquantize("./model-f16.gguf", quant_type="q8_0")\npush_to_hub("./", "ray0rf1re/shipped-model", token=MY_TOKEN)\ncreate_model_card("ray0rf1re/shipped-model", license="mit")`,
    hints: ['Four imports: convert_to_gguf_fp16, quantize, push_to_hub, create_model_card', 'quant_type="q8_0", license="mit"'],
  },
  {
    id: 'augment-blend-pipeline',
    title: 'Augment Then Blend',
    track: 'Advanced Pipelines',
    trackColor: '#ff6b35',
    concept: `A common pattern: augment each dataset independently (salt/pepper), then blend them with weights. This gives you control over both the augmentation level per dataset and the final mixing ratio without the datasets contaminating each other's augmentation style.`,
    example: `from hypernix.salt_shaker import HandCrusher
from hypernix.pepper_shaker import SmallShaker
from hypernix.blender import PersonalBlender

code_aug   = HandCrusher(code_data, p=0.05)
chat_aug   = SmallShaker(chat_data, mask_prob=0.1)

blended = PersonalBlender(
    sources=[code_aug, chat_aug],
    weights=[0.6, 0.4]
)`,
    exercise: `Apply FromTheBag (p=0.08) to \`science_data\` and Dish (typo_rate=0.04) to \`news_data\`, then blend 55% science / 45% news using PersonalBlender. Store in \`training_stream\`.`,
    starter: `from hypernix.salt_shaker import ___\nfrom hypernix.pepper_shaker import ___\nfrom hypernix.blender import ___\n\nsci_aug  = ___(science_data, p=___)\nnews_aug = ___(news_data, typo_rate=___)\n\ntraining_stream = ___(\n    sources=[sci_aug, news_aug],\n    weights=[___, ___]\n)`,
    solution: `from hypernix.salt_shaker import FromTheBag\nfrom hypernix.pepper_shaker import Dish\nfrom hypernix.blender import PersonalBlender\n\nsci_aug  = FromTheBag(science_data, p=0.08)\nnews_aug = Dish(news_data, typo_rate=0.04)\n\ntraining_stream = PersonalBlender(\n    sources=[sci_aug, news_aug],\n    weights=[0.55, 0.45]\n)`,
    hints: ['FromTheBag, Dish, PersonalBlender — one from each module', 'weights=[0.55, 0.45] — must sum to 1.0'],
  },
  {
    id: 'safe-long-run',
    title: 'Resilient Multi-Day Training',
    track: 'Advanced Pipelines',
    trackColor: '#ff6b35',
    concept: `For multi-day runs combine: BakeOff (NaN/OOM safety), FlashFreezer (VRAM management), PressureCooker (optimizer with LR schedule), and storage_warning (disk space guard). This four-way combination is standard for unattended GPU training.`,
    example: `from hypernix.cake_pan import BakeOff
from hypernix.freezer import FlashFreezer
from hypernix.pressure_cooker import PressureCooker
from hypernix.smoke_alarm import storage_warning

storage_warning("./runs", min_gb=20.0)
FlashFreezer(model)
opt = PressureCooker(model, lr=2e-4, warmup_steps=500)

with BakeOff(model, dataset, max_hours=48) as run:
    run.train(steps=100000)`,
    exercise: `Set up a resilient long run: warn if "./runs" has < 15 GB, wrap model in FlashFreezer, create a PressureCooker with lr=1e-4 and warmup_steps=300, then BakeOff for 48 hours / 50000 steps.`,
    starter: `from hypernix.cake_pan import ___\nfrom hypernix.freezer import ___\nfrom hypernix.pressure_cooker import ___\nfrom hypernix.smoke_alarm import ___\n\n___(___,  min_gb=___)\n___(model)\nopt = ___(model, lr=___, warmup_steps=___)\n\nwith ___(model, dataset, max_hours=___) as run:\n    run.train(steps=___)`,
    solution: `from hypernix.cake_pan import BakeOff\nfrom hypernix.freezer import FlashFreezer\nfrom hypernix.pressure_cooker import PressureCooker\nfrom hypernix.smoke_alarm import storage_warning\n\nstorage_warning("./runs", min_gb=15.0)\nFlashFreezer(model)\nopt = PressureCooker(model, lr=1e-4, warmup_steps=300)\n\nwith BakeOff(model, dataset, max_hours=48) as run:\n    run.train(steps=50000)`,
    hints: ['Four imports: BakeOff, FlashFreezer, PressureCooker, storage_warning', 'BakeOff goes last — it wraps the actual training call'],
  },
  {
    id: 'ema-plus-averaging',
    title: 'EMA + Checkpoint Averaging',
    track: 'Advanced Training',
    trackColor: '#ff6b35',
    concept: `Combining EMA tracking with post-run checkpoint averaging gives three different model variants from one training run: the live model, the EMA model, and the averaged checkpoint model. Run all three through espresso evaluation to pick the best one before shipping.`,
    example: `from hypernix.whisk import EMA, average_checkpoints
from hypernix.espresso_maker import SingleShot
from hypernix import new_range

ema = EMA(model, decay=0.999)
checkpoints = []

for step, batch in enumerate(dataloader):
    loss = model(batch); optimizer.step()
    ema.update(model)
    if step % 1000 == 0:
        save_checkpoint(model, f"ckpt-{step}.pt")
        checkpoints.append(f"ckpt-{step}.pt")

ema.apply_shadow(model)
average_checkpoints(checkpoints[-3:], "averaged.pt")`,
    exercise: `After training, apply EMA shadow to \`model\`, average the last 3 checkpoints in \`ckpt_paths\` to "best.pt", then run SingleShot evaluation on \`model\` with \`prompts\` and new_range rubric.`,
    starter: `from hypernix.whisk import ___, ___\nfrom hypernix.espresso_maker import ___\nfrom hypernix import new_range\n\n___.apply_shadow(model)\n___(ckpt_paths[-3:], output_path=___)\n\nresults = ___(model, prompts, rubric=___)\nprint(results.mean_score())`,
    solution: `from hypernix.whisk import EMA, average_checkpoints\nfrom hypernix.espresso_maker import SingleShot\nfrom hypernix import new_range\n\nema.apply_shadow(model)\naverage_checkpoints(ckpt_paths[-3:], output_path="best.pt")\n\nresults = SingleShot(model, prompts, rubric=new_range)\nprint(results.mean_score())`,
    hints: ['ema.apply_shadow(model) — assumes ema already exists', 'average_checkpoints(list, output_path="best.pt"), then SingleShot(model, prompts, rubric=new_range)'],
  },
  {
    id: 'curriculum-full',
    title: 'Curriculum Learning Pipeline',
    track: 'Advanced Pipelines',
    trackColor: '#ff6b35',
    concept: `Curriculum learning trains on easier data first, then harder data. HighPowerBlender supports staged weight schedules — step 0 might be 100% simple data, step 2000 shifts to 50/50, step 5000 goes 80% hard. Combine with CommercialSmoker to halt if quality drops during the curriculum transition.`,
    example: `from hypernix.blender import HighPowerBlender
from hypernix.smoker import CommercialSmoker

curriculum = [
    {0:    [1.0, 0.0]},   # warm-up: easy only
    {1000: [0.6, 0.4]},   # mid: mix in hard
    {3000: [0.2, 0.8]},   # late: mostly hard
]

blended = HighPowerBlender(
    [easy_data, hard_data],
    weights=[1.0, 0.0],
    seed=42,
    curriculum=curriculum
)

smoker = CommercialSmoker(
    model, blended,
    evals=my_suite,
    thresholds={"accuracy": 0.65}
)
smoker.run(steps=5000)`,
    exercise: `Build a HighPowerBlender with 3 stages: 0→all basic, 1500→60/40, 4000→30/70 on \`[basic_data, advanced_data]\`. Wrap in CommercialSmoker with threshold accuracy=0.6 for 3000 steps.`,
    starter: `from hypernix.blender import ___\nfrom hypernix.smoker import ___\n\ncurriculum = [\n    {0:    [___, ___]},\n    {1500: [___, ___]},\n    {4000: [___, ___]},\n]\n\nblended = ___(\n    [basic_data, advanced_data],\n    weights=[1.0, 0.0], seed=42,\n    curriculum=___\n)\n\nsmoker = ___(\n    model, blended,\n    evals=my_suite,\n    thresholds={___: ___}\n)\nsmoker.run(steps=___)`,
    solution: `from hypernix.blender import HighPowerBlender\nfrom hypernix.smoker import CommercialSmoker\n\ncurriculum = [\n    {0:    [1.0, 0.0]},\n    {1500: [0.6, 0.4]},\n    {4000: [0.3, 0.7]},\n]\n\nblended = HighPowerBlender(\n    [basic_data, advanced_data],\n    weights=[1.0, 0.0], seed=42,\n    curriculum=curriculum\n)\n\nsmoker = CommercialSmoker(\n    model, blended,\n    evals=my_suite,\n    thresholds={"accuracy": 0.6}\n)\nsmoker.run(steps=3000)`,
    hints: ['HighPowerBlender(sources, weights, seed, curriculum) — curriculum is a list of {step: [w1, w2]} dicts', 'CommercialSmoker thresholds={"accuracy": 0.6}'],
  },
  {
    id: 'evaluate-and-perturb',
    title: 'Evaluate, Perturb, Re-evaluate',
    track: 'Advanced Pipelines',
    trackColor: '#ff6b35',
    concept: `A common robustness test: evaluate baseline, apply HeavyFry perturbation, evaluate again. A large quality drop means the model relies on exact weight values and may not generalise. Small drop = robust. Always snapshot before HeavyFry and restore after.`,
    example: `from hypernix.deep_fryer import HeavyFry, snapshot, restore
from hypernix.espresso_maker import Ristretto

snap = snapshot(model)
baseline = Ristretto(model, prompts)

HeavyFry(model, noise_scale=0.1)
perturbed = Ristretto(model, prompts)

restore(model, snap)

print("Baseline:", baseline)
print("Perturbed:", perturbed)`,
    exercise: `Snapshot \`model\`, run Ristretto on \`prompts\` for baseline, apply HeavyFry(noise_scale=0.05), run Ristretto again for perturbed, restore from snapshot. Print both results.`,
    starter: `from hypernix.deep_fryer import ___, ___, ___\nfrom hypernix.espresso_maker import ___\n\nsnap = ___(model)\nbaseline = ___(model, prompts)\n\n___(model, noise_scale=___)\nperturbed = ___(model, prompts)\n\n___(model, snap)\nprint("Baseline:", baseline)\nprint("Perturbed:", perturbed)`,
    solution: `from hypernix.deep_fryer import HeavyFry, snapshot, restore\nfrom hypernix.espresso_maker import Ristretto\n\nsnap = snapshot(model)\nbaseline = Ristretto(model, prompts)\n\nHeavyFry(model, noise_scale=0.05)\nperturbed = Ristretto(model, prompts)\n\nrestore(model, snap)\nprint("Baseline:", baseline)\nprint("Perturbed:", perturbed)`,
    hints: ['snapshot → Ristretto → HeavyFry → Ristretto → restore', 'All three deep_fryer imports: HeavyFry, snapshot, restore'],
  },
  {
    id: 'camo-rlaf',
    title: 'Camouflage (RLHF/RLAF)',
    track: 'Advanced Alignment',
    trackColor: '#c8192e',
    concept: `Camouflage (hypernix.camouflage) provides built-in RLHF and RLAF model alignment loops. By specifying a reward model (-Ai with -M), you can train a local model (-Lmodel) through automated feedback. It uses a REINFORCE loop to optimize the local policy.`,
    example: `from hypernix.camouflage import run_rlhf

run_rlhf(
    local_model="meta-llama/Llama-3.2-1B",
    steps=100,
    use_ai=True,
    evaluator_path="meta-llama/Llama-3.1-8B-Instruct",
    sys_prompt="Rate this response from 1 to 10."
)`,
    exercise: `Run RLHF for 200 steps on "my-local-model", using AI evaluator "my-reward-model" with prompt "Score it!".`,
    starter: `from hypernix.camouflage import run_rlhf\n\n___(\n    local_model=___,\n    steps=___,\n    use_ai=___,\n    evaluator_path=___,\n    sys_prompt=___\n)`,
    solution: `from hypernix.camouflage import run_rlhf\n\nrun_rlhf(\n    local_model="my-local-model",\n    steps=200,\n    use_ai=True,\n    evaluator_path="my-reward-model",\n    sys_prompt="Score it!"\n)`,
    hints: ['run_rlhf with local_model="my-local-model"', 'use_ai=True and evaluator_path="my-reward-model"'],
  },
  {
    id: 'hyper-log',
    title: 'Hyper-Log Dashboard',
    track: 'Dashboards',
    trackColor: '#34c759',
    concept: `Hyper-Log provides a premium TUI for training metrics with support for grad norm, ETA, Epoch, and Hardware Telemetry. It is fully compatible with tvtop and can be manually paused or stopped via TUI interactions.`,
    example: `from hypernix.hyper_log import HyperLogger

logger = HyperLogger(total_steps=5000)
logger.start()

# Inside loop:
# logger.update(step, loss, grad_norm, lr, epoch)

logger.stop()`,
    exercise: `Initialize HyperLogger for 1000 steps, start it, update with step=10, loss=0.5, grad_norm=0.1, lr=1e-4, epoch=0.1, and then stop it.`,
    starter: `from hypernix.hyper_log import ___\n\nlogger = ___(total_steps=___)\nlogger.___()\n\nlogger.update(___, ___, ___, ___, ___)\n\nlogger.___()`,
    solution: `from hypernix.hyper_log import HyperLogger\n\nlogger = HyperLogger(total_steps=1000)\nlogger.start()\n\nlogger.update(10, 0.5, 0.1, 1e-4, 0.1)\n\nlogger.stop()`,
    hints: ['HyperLogger(total_steps=1000) then start()', 'logger.update(10, 0.5, 0.1, 1e-4, 0.1)'],
  },
  {
    id: 'v5s-optimizer',
    title: 'Pressure Cooker V5S',
    track: 'Core Optimizers',
    trackColor: '#007aff',
    concept: `Pressure Cooker V5S is the oscillation resistant cosin 3d, pressure diffusion low power optimizer. It offers 2.1x speedup over AdamW while using less RAM.`,
    example: `from hypernix.pressure_cooker_v5s import PressureCookerV5S\n\nopt = PressureCookerV5S(model.parameters(), lr=1e-4)\nopt.step()`,
    exercise: `Create a PressureCookerV5S for \`model\` with lr=1e-3.`,
    starter: `from hypernix.pressure_cooker_v5s import ___\n\nopt = ___(model.parameters(), lr=___)`,
    solution: `from hypernix.pressure_cooker_v5s import PressureCookerV5S\n\nopt = PressureCookerV5S(model.parameters(), lr=1e-3)`,
    hints: ['PressureCookerV5S(model.parameters(), lr=1e-3)'],
  },
  {
    id: 'vera-assistant',
    title: 'Vera AI Assistant',
    track: 'CLI Tools',
    trackColor: '#af52de',
    concept: `Vera is the built-in AI assistant for HyperNix. Access her interactively from the CLI using \`hnx vera\`.`,
    example: `# CLI Usage:\n# hnx vera`,
    exercise: `You can chat with Vera from the CLI.`,
    starter: `# hnx vera`,
    solution: `# hnx vera`,
    hints: ['Use hnx vera in your terminal'],
  },
  {
    id: 'scavenger-data',
    title: 'Scavenger Data Tools',
    track: 'CLI Tools',
    trackColor: '#af52de',
    concept: `Scavenger is a toolset for data filtering and collection. Access it interactively via \`hnx scavenger\`.`,
    example: `# CLI Usage:\n# hnx scavenger`,
    exercise: `You can access scavenger from the CLI.`,
    starter: `# hnx scavenger`,
    solution: `# hnx scavenger`,
    hints: ['Use hnx scavenger in your terminal'],
  },
]

