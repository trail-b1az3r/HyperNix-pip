// Reference data for the subsystem browser, homepage feature grid,
// supported-model list, and quickstart steps. Kept separate from UI
// components so it can be grepped/updated without touching markup.

export const SUBSYSTEMS = [
  { name: 'hypernix.download', desc: 'Hub snapshot downloads with short-name resolution, gated repo support, and offline cache' },
  { name: 'hypernix.train', desc: 'HyperNixConfig, HyperNixModel, init_from_scratch, expand_checkpoint, and full training loop' },
  { name: 'hypernix.brewer', desc: 'hyperNix0x-v2 architecture presets: 33m/micro/small/medium/large (GPU) + cpu-nano/cpu-tiny/cpu-small (CPU-only). custom_arch() for a bespoke config' },
  { name: 'hypernix.old_oven', desc: 'CodeOven wrapper with .complete() / .chat() / .fill() / .save_pt(). new_oven() for parametric models' },
  { name: 'hypernix.old_fridge', desc: 'Memory housekeeping: freeze, unfreeze, parameter_stats, offload_to_cpu, chill_cache' },
  { name: 'hypernix.mediocre_fridge', desc: 'Judge-training dataset generation: synthesize_judge_corpus, collect_responses_from' },
  { name: 'hypernix.new_fridge', desc: 'Training-curve graphing: parse_training_log, plot_loss_curve, plot_score_distribution' },
  { name: 'hypernix.new_range', desc: 'Zero-dep first-fail labeling rubric — lightest scoring tier' },
  { name: 'hypernix.old_range', desc: 'Scored rubric with per-criterion scores and explainability strings' },
  { name: 'hypernix.industrial_range', desc: 'LLM-as-judge wrapper routing through a CodeOven for scoring' },
  { name: 'hypernix.freezer', desc: 'OldFreezer (8-10 GB) / NewFreezer (11 GB+) / FlashFreezer (OOM-safe retry). 20 GPU presets + 60 CPU presets (Intel + AMD Ryzen)' },
  { name: 'hypernix.smoke_alarm', desc: 'Training-step planner & monitor: RadsAlarm / GasAlarm / ModernAlarm / AutoAlarm + NaN check' },
  { name: 'hypernix.pans', desc: '5-tier data preprocessing: FryingPan → SaucePan → Skillet → GrillPan → Wok' },
  { name: 'hypernix.microwave', desc: '5-tier throwaway inference: defrost → low_zap → zap → high_zap → chat_zap, plus reheat' },
  { name: 'hypernix.table', desc: 'Dead-simple tabular viewer: from_training_log, from_judge_corpus, filter, select, show' },
  { name: 'hypernix.sink', desc: 'Append-only file sink with optional rotation and deduplication' },
  { name: 'hypernix.instant_pot', desc: 'brew(recipe) — one-shot end-to-end pipeline. Also available as CLI: hypernix brew recipe.json' },
  { name: 'hypernix.coffee_maker', desc: '3 training tiers: drip / french_press / percolator + cold_brew for long checkpointed runs' },
  { name: 'hypernix.espresso_maker', desc: '4-tier evaluation: Ristretto / SingleShot / DoubleShot / Lungo — prompt battery + scoring' },
  { name: 'hypernix.blender', desc: '4-tier multi-source mixing: HandBlender / PersonalBlender / CountertopBlender / HighPowerBlender' },
  { name: 'hypernix.toaster', desc: '4-tier per-line formatting: TwoSliceToaster / FourSliceToaster / ConveyorToaster / ToasterOven' },
  { name: 'hypernix.food_processor', desc: '4-tier bulk chunking: ChopBlade / SliceBlade / ShredBlade / PureeBlade' },
  { name: 'hypernix.smoker', desc: '4-tier training quality: UseableSmoker / GoodSmoker / CommercialSmoker / HighQualitySmoker' },
  { name: 'hypernix.deep_fryer', desc: '2-tier model-weight perturbation: LightFry (regulariser) / HeavyFry (severe). Reversible via snapshot' },
  { name: 'hypernix.cake_pan', desc: 'Hybrid CPU+GPU training guard: NaN/Inf detection, wall-time watchdog, memory-pressure rollback via BakeOff' },
  { name: 'hypernix.salt_shaker', desc: '3-tier gentle data augmentation: FromTheBag / HandCrusher / PoshSaltDish' },
  { name: 'hypernix.pepper_shaker', desc: '3-tier sharp perturbations: SmallShaker (MLM-mask) / Dish (typos) / TallHandmade (negation)' },
  { name: 'hypernix.pressure_cooker', desc: 'Custom optimizer: AdamW + warmup / plateau detection / cosine cooldown + lookahead' },
  { name: 'hypernix.torch_compat', desc: 'Portability shim (RMSNorm + SDPA) for running on old Intel Macs with torch 1.13' },
  { name: 'hypernix.convert', desc: 'Safetensors → GGUF at fp32/fp16. Architecture-agnostic tensor naming' },
  { name: 'hypernix.quantize', desc: 'llama-quantize driver for Q8_0, Q6_K, Q4_K_M, Q5_K_M — 30 quant types total' },
  { name: 'hypernix.upload', desc: 'Push artifacts to HuggingFace Hub with multipart + model card generation' },
  { name: 'hypernix.whisk', desc: 'Checkpoint averaging and EMA (Exponential Moving Average) utilities' },
  { name: 'hypernix.cutting_board', desc: 'Train / val / test dataset splitting with stratified and k-fold support' },
  { name: 'hypernix.countertop', desc: 'Stateful chat session management — Countertop creates sessions, bell rings the model, flour applies templates' },
  { name: 'hypernix.hyped_pro_core', desc: 'hyped-pro/hyped+ TUI\'s provider+model registry and real dispatch: cloud APIs (Anthropic/OpenAI-compatible), local safetensors, GGUF via multilama, and an agentic tool-calling loop' },
  { name: 'hypernix.multilama', desc: 'One interface over several llama.cpp variants (vanilla, ik_llama.cpp, PrismML fork, KoboldCpp, Nanbeige fork) for GGUF models a single backend can\'t all load' },
  { name: 'hypernix.hyped_pro_tools', desc: 'Real, workspace-scoped file create/edit/read/search tools for hyped-pro\'s agentic chat loop' },
  { name: 'hypernix.hyped_pro_bridge', desc: 'Line-delimited JSON stdio worker the Node hyped-pro TUI shells out to for every real operation' },
  { name: 'hypernix.hyped_pro_gui', desc: 'hyped-pro desktop GUI: Qt6 (X11 + Wayland) via PySide6, GTK4 fallback' },
]


export const FEATURES = [
  { icon: 'download', title: 'Download', color: '#4a9eff', desc: 'Pull snapshots from the Hub with short-name resolution, gated repo support, and offline cache.' },
  { icon: 'train', title: 'Train', color: '#c8192e', desc: 'HyperNixConfig, HyperNixModel, init_from_scratch, expand_checkpoint, and full training loops.' },
  { icon: 'chat', title: 'Chat & Complete', color: '#e8960a', desc: 'CodeOven wrapper: .complete(), .chat(), .fill(). Chat templates for all major model families.' },
  { icon: 'quantize', title: 'Quantize', color: '#d4c800', desc: '30 quantization types from fp32/fp16 to IQ-quants. llama-quantize integration + auto-caching.' },
  { icon: 'vram', title: 'VRAM Management', color: '#34c759', desc: 'OldFreezer (8-10 GB), NewFreezer (11 GB+), FlashFreezer (OOM-safe retry). 20 GPU presets.' },
  { icon: 'evaluate', title: 'Evaluate', color: '#e05555', desc: '4-tier evaluation: Ristretto → Lungo. Run prompt batteries, score results, generate reports.' },
  { icon: 'preprocess', title: 'Preprocess', color: '#8866dd', desc: '5-tier data pipeline: FryingPan → SaucePan → Skillet → GrillPan → Wok.' },
  { icon: 'ship', title: 'Ship', color: '#00b5ad', desc: 'Push to HuggingFace Hub. GGUF conversion, upload utilities, dataset packaging.' },
]


export const MODELS = [
  { family: 'HyperNix / Nix', models: ['hyper-nix.1', 'nix2.5', 'nix2.6-m', 'nix2.6-mm', 'nix-2.7a', 'nix2.7', 'nano-nano-v4'] },
  { family: 'Llama 3.x / 4', models: ['llama-3.1-8b', 'llama-3.2-1b', 'llama-3.2-3b', 'llama-3.3-70b-instruct', 'llama4'] },
  { family: 'Qwen 2.5 / 3 / 3.5 / 3.6', models: ['qwen2.5-*', 'qwen3-0.6b', 'qwen3-8b', 'qwen3.5-4b', 'qwen3.5-35b-a3b', 'qwen3.6-35b-a3b'] },
  { family: 'Gemma 2 / 3 / 4', models: ['gemma-2-2b', 'gemma-2-9b', 'gemma-3-1b', 'gemma-3-4b', 'gemma-4-e4b', 'gemma-4-26b-a4b'] },
  { family: 'Phi 3 / 3.5 / 4', models: ['phi-3-mini', 'phi-3.5-mini', 'phi-4'] },
  { family: 'DeepSeek', models: ['deepseek-r1-distill-llama-8b', 'deepseek-r1-distill-qwen-7b', 'deepseek-v2-lite', 'deepseek-v3'] },
  { family: 'GLM 4 / 5 / 5.1', models: ['glm-4-9b-chat', 'glm-4.1v', 'glm-5', 'glm-5.1', 'glm-5.1-fp8'] },
  { family: 'Mistral / Mixtral', models: ['mistral-7b-instruct', 'mixtral-8x7b-instruct', 'mistral-nemo-12b'] },
  { family: 'NVIDIA', models: ['nemotron-4-15b', 'llama-3.1-nemotron-70b-instruct'] },
  { family: 'OpenAI gpt-oss', models: ['gpt-oss-20b', 'gpt-oss-120b'] },
]


export const QUICKSTART = [
  { step: '01', title: 'Install', code: 'pip install "hypernix[llama-cpp]"', desc: 'Core toolkit + llama-cpp-python bundled' },
  { step: '02', title: 'Chat', code: 'hypernix chat --repo-id nix2.5 --message "hello"', desc: 'Chat with any supported model using short names' },
  { step: '03', title: 'Convert', code: 'hypernix --repo-id ray0rf1re/hyper-nix.1 --quants fp32 fp16 q4_k_m', desc: 'Convert to GGUF with k-quants in one command' },
  { step: '04', title: 'Train', code: 'python examples/train_hypernix_1_5_gtx1080.py', desc: 'Train on consumer GPUs with automatic optimization' },
]


export const SUBSYSTEM_GROUPS = {
  'Data': ['pans','blender','toaster','food_processor','cutting_board','salt_shaker',
    'pepper_shaker','sink','table','mediocre_fridge'],
  'Training': ['train','brewer','instant_pot','coffee_maker','smoker','cake_pan',
    'smoke_alarm','pressure_cooker','deep_fryer','whisk','new_fridge'],
  'Inference & chat': ['old_oven','microwave','countertop','multilama','hyped_pro_core',
    'hyped_pro_tools','hyped_pro_bridge','hyped_pro_gui'],
  'Evaluation': ['new_range','old_range','industrial_range','espresso_maker'],
  'Runtime & memory': ['freezer','old_fridge','torch_compat'],
  'Ship & convert': ['download','convert','quantize','upload'],
}

export const GROUP_ORDER = Object.keys(SUBSYSTEM_GROUPS)

// name -> group, built once. Anything not listed above lands in "Other" rather
// than disappearing from the browser.
export const GROUP_OF = (() => {
  const out = {}
  for (const [group, names] of Object.entries(SUBSYSTEM_GROUPS)) {
    for (const n of names) out[n] = group
  }
  return out
})()

export function subsystemGroup(fullName) {
  return GROUP_OF[fullName.replace(/^hypernix\./, '')] || 'Other'
}

// Scripted terminal session shown in the homepage hero.
export const HERO_SESSION = [
  { kind:'cmd',  text:'pip install "hypernix[llama-cpp]"' },
  { kind:'ok',   text:'installed hypernix' },
  { kind:'gap',  text:'' },
  { kind:'cmd',  text:'hypernix chat --repo-id nix2.5 --message "hello"' },
  { kind:'dim',  text:'  resolving nix2.5 -> ray0rf1re/nix2.5' },
  { kind:'dim',  text:'  loaded 1.4B params  ·  q4_k_m  ·  cuda:0' },
  { kind:'out',  text:'  hey — what are we building today?' },
  { kind:'gap',  text:'' },
  { kind:'cmd',  text:'hypernix --repo-id ray0rf1re/hyper-nix.1 --quants q4_k_m' },
  { kind:'ok',   text:'hyper-nix.1.q4_k_m.gguf' },
]
