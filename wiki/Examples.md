# Examples — HyperNix Module Encyclopedia

> A source-oriented guide to what the modules in `src/hypernix/` do. This page is an index and explanation of the package, not a replacement for API reference pages or runnable scripts in `examples/`. Public module names are identified separately from internal implementation modules.

**Accuracy rule:** implementation and exported symbols are authoritative. A file's presence does not mean its feature is stable, enabled by default, supported on every platform, or part of the public API. Hardware/provider capabilities are conditional on the actual backend and dependencies. Follow the linked topic guide for detailed usage; check the source for exact signatures.

## Table of contents

- [Using this page](#using-this-page)
- [Public module registry](#public-module-registry)
  - [Chat](#chat)
  - [Data](#data)
  - [Evaluation](#evaluation)
  - [Interfaces](#interfaces)
  - [Models](#models)
  - [Monitoring](#monitoring)
  - [Optimizers](#optimizers)
  - [Quantization](#quantization)
  - [Security](#security)
  - [System](#system)
  - [Elements](#elements)
  - [Runtime](#runtime)
  - [Timing](#timing)
  - [Training](#training)
- [Additional package namespaces](#additional-package-namespaces)
  - [Audio](#audio)
  - [Bridge](#bridge)
  - [C4 model analysis](#c4-model-analysis)
  - [Dilute](#dilute)
  - [HyperLink](#hyperlink)
  - [Noodle agents](#noodle-agents)
  - [Neuron](#neuron)
  - [Script generator](#script-generator)
  - [T1 API server](#t1-api-server)
  - [T1 SDK](#t1-sdk)
  - [Waiter client](#waiter-client)
- [How modules fit together](#how-modules-fit-together)
- [Runnable examples and deeper references](#runnable-examples-and-deeper-references)
- [Keeping this page accurate](#keeping-this-page-accurate)

## Using this page

The canonical public grouping is `hypernix.MODULE_CATEGORIES` in [`src/hypernix/__init__.py`](../src/hypernix/__init__.py). Import paths shown below use the current category layout. The legacy flat import aliases are retained for compatibility where the package provides them; new code should prefer the category path.

```python
import hypernix

# Inspect the public module categories exposed by the package.
for category, modules in hypernix.MODULE_CATEGORIES.items():
    print(category, ", ".join(modules))

# Import a module directly.
from hypernix.chat.countertop import Countertop
```

The registry is not the entire source tree. The additional namespaces below document important packages and their internal components too. Internal components are implementation boundaries, not a promise that every file is a supported public import.

## Public module registry

### Chat

Chat modules format conversations, manage prompts and sessions, and process generated output.

| Module | What it does | Related guide |
|---|---|---|
| `hypernix.chat.cookbook` | Selects and applies model-appropriate chat templates, including ChatML, Llama, Alpaca, Vicuna, and HyperNix formats. | [Cookbook](Cookbook.md) |
| `hypernix.chat.countertop` | Maintains multi-turn conversation history around an oven/model; supports sending turns, resetting, saving/loading, trimming history, and optional streaming/output processing. | [Countertop](Countertop.md) |
| `hypernix.chat.flour` | Applies generation-time output controls such as repetition/frequency/presence penalties, no-repeat n-grams, bad-word handling, and role-leak suppression. | [Flour](Flour.md) |
| `hypernix.chat.injection` | Provides token/prompt injection helpers for controlled thinking, testing, system overrides, and custom injected content. | [Camouflage](Camouflage.md) |
| `hypernix.chat.menu` | Stores named system-prompt/persona presets and resolves a requested persona for a chat session. | [Menu](Menu.md) |

### Data

Data modules discover, collect, transform, split, augment, package, and persist datasets.

| Module | What it does | Related guide |
|---|---|---|
| `hypernix.data.blender` | Mixes/interleaves multiple data sources using selectable processing tiers. | [Blender](Blender.md) |
| `hypernix.data.cardboard_box` | Provides a structured container/packing abstraction for dataset or training artifacts; inspect its API before assuming a storage format. | [Tupperware](Tupperware.md) |
| `hypernix.data.cutting_board` | Splits records into train/validation/test sets, including deterministic and stratified splitting. | [CuttingBoard](CuttingBoard.md) |
| `hypernix.data.food_processor` | Chunks or segments large text/data records using selectable chopping, slicing, shredding, and purée strategies. | [FoodProcessor](FoodProcessor.md) |
| `hypernix.data.gather` | Collects and retrieves datasets/samples from configured sources; handles discovery and collection workflow rather than model training. | [Scavenger](Scavenger.md) |
| `hypernix.data.gather_cli` | Command-line entry point for the data-gathering workflow. | [CLI](CLI.md) |
| `hypernix.data.lunchbox` | Packages datasets into consistent schemas for evaluation or Hub upload, catching column mismatches before publishing. | [Lunchbox](Lunchbox.md) |
| `hypernix.data.mediocre_fridge` | Builds judge-training corpora and collects model responses for later scoring/training. | [Fridges](Fridges.md) |
| `hypernix.data.pans` | Applies staged text/data preprocessing through the FryingPan-to-Wok processing tiers. | [Kitchen](Kitchen.md) |
| `hypernix.data.pepper_shaker` | Applies stronger/specific text perturbations, including masking, typo, and negation-style variants. | [Shakers](Shakers.md) |
| `hypernix.data.qa` | Converts structured question/answer or text records into causal-language-model training text. | [Kitchen](Kitchen.md) |
| `hypernix.data.salt_shaker` | Applies gentler dataset augmentation/variation strategies. | [Shakers](Shakers.md) |
| `hypernix.data.scavenger` | Searches Hugging Face datasets and filters candidates by configured relevance, quality, recency, and storage constraints. | [Scavenger](Scavenger.md) |
| `hypernix.data.sink` | Writes output records through an append-oriented sink with optional rotation and deduplication. | [Kitchen](Kitchen.md) |
| `hypernix.data.strainer` | Filters dataset records, including near-duplicate detection, before training/evaluation. | [Kitchen](Kitchen.md) |
| `hypernix.data.toaster` | Formats or transforms records line by line using selectable output-processing tiers. | [Toaster](Toaster.md) |
| `hypernix.data.tupperware` | Splits/organizes data into rounds or staged subsets, with optional training/evaluation support. | [Tupperware](Tupperware.md) |

### Evaluation

Evaluation modules score outputs, define rubrics, visualize results, and verify module quality.

| Module | What it does | Related guide |
|---|---|---|
| `hypernix.evaluation.espresso_maker` | Runs prompt batteries at selectable evaluation tiers and returns scored evaluation results. | [EspressoMaker](EspressoMaker.md) |
| `hypernix.evaluation.industrial_range` | Uses a model-compatible judge to evaluate outputs, including pairwise preference comparisons. | [Ranges](Ranges.md) |
| `hypernix.evaluation.new_fridge` | Parses training/evaluation logs and produces plots such as loss curves and score distributions. | [Fridges](Fridges.md) |
| `hypernix.evaluation.new_range` | Provides a lightweight first-fail labeling rubric for evaluating responses. | [Ranges](Ranges.md) |
| `hypernix.evaluation.old_range` | Provides weighted, explainable rule-based scoring for response evaluation. | [Ranges](Ranges.md) |
| `hypernix.evaluation.vera` | Checks modules for syntax, documentation/type coverage, and configured smoke-test expectations. | [Vera](Vera.md) |

### Interfaces

Interface modules expose HyperNix functionality through commands, terminal/desktop UIs, and application bridges.

| Module | What it does | Related guide |
|---|---|---|
| `hypernix.interfaces.assistant` | Implements the assistant-facing interaction layer and its command integration. | [CLI](CLI.md) |
| `hypernix.interfaces.cli` | Implements the main HyperNix command-line interface and subcommand dispatch. | [CLI](CLI.md) |
| `hypernix.interfaces.hyped` | Implements the basic local-AI terminal chat experience. | [CLI](CLI.md) |
| `hypernix.interfaces.hyped_pro` | Implements the expanded Hyped Pro terminal interface and its application workflow. | [CLI](CLI.md) |
| `hypernix.interfaces.hyped_pro_bridge` | Connects the Hyped Pro front end to its Python/backend operations. | [CLI](CLI.md) |
| `hypernix.interfaces.hyped_pro_core` | Contains the shared model/provider dispatch and core Hyped Pro behavior. | [CLI](CLI.md) |
| `hypernix.interfaces.hyped_pro_git` | Provides Git operations used by the Hyped Pro workspace. | [CLI](CLI.md) |
| `hypernix.interfaces.hyped_pro_gui` | Launches/integrates the desktop GUI for Hyped Pro. | [CLI](CLI.md) |
| `hypernix.interfaces.hyped_pro_otui` | Implements the OpenTUI-based terminal UI. | [CLI](CLI.md) |
| `hypernix.interfaces.hyped_pro_tools` | Defines and dispatches tools exposed by Hyped Pro. | [CLI](CLI.md) |
| `hypernix.interfaces.version_launcher` | Selects/launches the installed HyperNix version or entry point. | [CLI](CLI.md) |
| `hypernix.interfaces.websearch` | Exposes the interactive web-search interface/command. | [CLI](CLI.md) |
| `hypernix.interfaces.wiki_cli` | Builds and browses module documentation from source documentation. | [Vera](Vera.md) |

### Models

Model modules define architectures, load snapshots, generate text, and connect supported model families.

| Module | What it does | Related guide |
|---|---|---|
| `hypernix.models.arch` | Defines architecture presets and model-family configuration used when creating/loading models. | [Architectures](Architectures.md) |
| `hypernix.models.brewer_adapter` | Adapts the HyperNix Brewer model/configuration to supported external model/runtime interfaces. | [Architectures](Architectures.md) |
| `hypernix.models.download` | Downloads model snapshots and resolves short names, gated repositories, and cached/offline copies. | [Download](Download.md) |
| `hypernix.models.generate` | Provides text-generation helpers over loaded models. | [Ovens](Ovens.md) |
| `hypernix.models.microwave` | Runs short-lived/one-shot inference using a selected model and generation tier. | [Microwave](Microwave.md) |
| `hypernix.models.multilama` | Dispatches GGUF inference across supported llama.cpp-family backends, with capability-aware selection/fallback. | [Runtime](Runtime.md) |
| `hypernix.models.nano_nano` | Implements the Nano-Nano model family and its model-specific configuration. | [Architectures](Architectures.md) |
| `hypernix.models.neo_oven` | Implements the Neo Oven model/training interface and its architecture-aware workflow. | [Ovens](Ovens.md) |
| `hypernix.models.old_oven` | Provides the CodeOven snapshot wrapper for completion, chat, fill, training, and snapshot-related operations. | [Ovens](Ovens.md) |
| `hypernix.models.stml` | Provides context-length/VRAM estimation and sequence-segmentation helpers for training. | [STML](STML.md) |
| `hypernix.models.whisk` | Averages model checkpoints using supported averaging strategies and writes a merged snapshot. | [Whisk](Whisk.md) |
| `hypernix.models.workshop` | Supplies reusable model-framework templates and task pipelines for LLM, vision, TTS, and ASR workloads. | [Workshop](Workshop.md) |

### Monitoring

Monitoring modules collect runtime/training telemetry, display dashboards, and inspect runs.

| Module | What it does | Related guide |
|---|---|---|
| `hypernix.monitoring.cctvtop` | Runs the enhanced terminal dashboard combining training and host monitoring views. | [Dashboards](Dashboards.md) |
| `hypernix.monitoring.hyper_log` | Provides structured training/runtime logging helpers. | [HyperLog](HyperLog.md) |
| `hypernix.monitoring.map` | Maps/displays hardware and runtime information for monitoring components. | [Dashboards](Dashboards.md) |
| `hypernix.monitoring.plasma` | Runs a short hardware calibration workload and returns a factor usable by training-time estimates. | [Alarms](Alarms.md) |
| `hypernix.monitoring.table` | Displays tabular data, including parsed training logs and judge corpora. | [Dashboards](Dashboards.md) |
| `hypernix.monitoring.thermometer` | Samples supported CPU/GPU temperature sensors. | [Devices](Devices.md) |
| `hypernix.monitoring.tv` | Provides the original lightweight terminal training monitor. | [Dashboards](Dashboards.md) |
| `hypernix.monitoring.tvtop` | Preserves the legacy tvtop import/command surface and compatibility behavior. | [Dashboards](Dashboards.md) |
| `hypernix.monitoring.tvtop_plus_plus` | Implements the richer terminal training dashboard and its display components. | [Dashboards](Dashboards.md) |

### Optimizers

Optimizer modules implement the Pressure Cooker generations and their common dispatch/compatibility layer.

| Module | What it does | Related guide |
|---|---|---|
| `hypernix.optimizers.deprecation` | Centralizes optimizer deprecation warnings and compatibility notices. | [Optimizers](Optimizers.md) |
| `hypernix.optimizers.optimizer_framework` | Provides shared optimizer selection/configuration and common optimizer plumbing. | [Optimizers](Optimizers.md) |
| `hypernix.optimizers.pressure_cooker` | Implements the original tiered AdamW-style Pressure Cooker optimizer family. | [Optimizers](Optimizers.md) |
| `hypernix.optimizers.pressure_cooker_v3` | Implements the V3 optimizer with quantized-state and ZeRO-style sharding features. | [Pressure Cooker v3](Pressure-Cooker-V3.md) |
| `hypernix.optimizers.pressure_cooker_v4` | Implements the V4 optimizer generation and its quantization-aware training integration. | [Pressure Cooker v4](Pressure-Cooker-V4.md) |
| `hypernix.optimizers.pressure_cooker_v5` | Implements the V5/V5+ optimizer family, including compressed momentum/state, QAT, and EMA mechanisms. | [Pressure Cooker v5](Pressure-Cooker-V5.md) |
| `hypernix.optimizers.pressure_cooker_v5s` | Implements the V5S optimizer variant and its oscillation/stability mechanisms. | [Pressure Cooker v5](Pressure-Cooker-V5.md) |
| `hypernix.optimizers.pressure_cooker_v6` | Implements the speed-oriented V6 optimizer with multi-tensor operations and reduced synchronization. | [Pressure Cooker v6](Pressure-Cooker-V6.md) |
| `hypernix.optimizers.pressure_cooker_v6v` | Adds the V6V execution path, including optional compile/CUDA-graph acceleration where supported. | [Pressure Cooker v6](Pressure-Cooker-V6.md) |

### Quantization

Quant modules convert, inspect, quantize, fetch required tooling, and upload model artifacts.

| Module | What it does | Related guide |
|---|---|---|
| `hypernix.quant.convert` | Converts supported model checkpoints to GGUF. | [Convert](Convert.md) |
| `hypernix.quant.fetcher` | Finds/fetches the external quantization tooling required by selected workflows. | [Pipeline mechanics](PipelineMechanics.md) |
| `hypernix.quant.quantize` | Drives llama.cpp's `llama-quantize` utility for supported GGUF quant types. | [Quantization](Quantization.md) |
| `hypernix.quant.upload` | Uploads generated model artifacts to a Hugging Face repository. | [Quantization](Quantization.md) |

### Security

Security modules manage API credentials, key lifecycle, and request authorization.

| Module | What it does | Related guide |
|---|---|---|
| `hypernix.security.gatekeeper` | Validates credentials and enforces configured request/key access rules. | [T1 API](T1-API.md) |
| `hypernix.security.gkey_cli` | Provides command-line operations for the key-management surface. | [T1 API](T1-API.md) |
| `hypernix.security.keymaster` | Creates, stores, validates, rotates, and revokes HyperNix API keys and their metadata. | [T1 API](T1-API.md) |

### System

System modules detect/configure the environment, manage hardware/resources, and provide housekeeping and diagnostics.

| Module | What it does | Related guide |
|---|---|---|
| `hypernix.system.compactor` | Archives older checkpoints/artifacts according to retention settings. | [Training](Training.md) |
| `hypernix.system.config` | Loads and validates shared HyperNix configuration. | [CLI](CLI.md) |
| `hypernix.system.deps` | Detects and manages optional runtime dependencies. | [Devices](Devices.md) |
| `hypernix.system.dishwasher` | Cleans selected stale logs, checkpoints, and build artifacts. | [Training](Training.md) |
| `hypernix.system.doctor` | Diagnoses installation, dependencies, hardware, and runtime readiness. | [CLI](CLI.md) |
| `hypernix.system.ethanol` | Provides guarded GPU clock/overclock controls through available vendor utilities; requires explicit confirmation. | [Devices](Devices.md) |
| `hypernix.system.freezer` | Selects memory/VRAM management strategies and OOM-aware retry behavior. | [Freezer](Freezer.md) |
| `hypernix.system.gpus` | Detects and describes GPU devices and their available backend capabilities. | [Devices](Devices.md) |
| `hypernix.system.fusebox` | Provides GPU thermal/power-aware training control and safety integration. | [Devices](Devices.md) |
| `hypernix.system.errorcatalogue` | Defines the structured error catalogue and human-readable explanations. | [Error codes](Error-Codes.md) |
| `hypernix.system.errorcodes` | Defines stable error-code identifiers and metadata. | [Error codes](Error-Codes.md) |
| `hypernix.system.errors_cli` | Exposes error lookup/explanation through the CLI. | [Error codes](Error-Codes.md) |
| `hypernix.system.hubcompat` | Handles compatibility boundaries for Hugging Face Hub APIs/versions. | [Download](Download.md) |
| `hypernix.system.net` | Provides network discovery and connectivity helpers. | [T1 API](T1-API.md) |
| `hypernix.system.nettrust` | Evaluates configured network trust/allowlist policy. | [T1 API](T1-API.md) |
| `hypernix.system.old_fridge` | Provides memory housekeeping, parameter freezing/unfreezing, and CPU offload helpers. | [Fridges](Fridges.md) |
| `hypernix.system.outage` | Temporarily blanks/suppresses supported displays during long-running work and restores them on exit. | [Training](Training.md) |
| `hypernix.system.pathfix` | Applies or undoes the console-script PATH setup. | [CLI](CLI.md) |
| `hypernix.system.protect` | Provides guarded system/display protection controls. | [CLI](CLI.md) |
| `hypernix.system.release` | Reads and reports HyperNix release/version information. | [CLI](CLI.md) |
| `hypernix.system.torch_compat` | Supplies PyTorch compatibility helpers for supported older runtimes/platforms. | [macOS legacy](macOS-legacy.md) |
| `hypernix.system.ups` | Adds an uninterruptible-power/outage-aware training safety mode and checkpoint callback behavior. | [Training](Training.md) |
| `hypernix.system.utils` | Collects shared utilities such as health checks, model listing, and environment diagnostics. | [CLI](CLI.md) |
| `hypernix.system.vram` | Provides opt-in allocator, activation-checkpointing, optimizer-state, and peak-memory helpers. | [VRAM](VRAM.md) |

### Elements

The elements namespace is a small add-on framework: Hydrogen defines the extension model, Natural Gas attaches add-ons to an oven, and individual elements provide capabilities.

| Module | What it does |
|---|---|
| `hypernix.elements.builtins` | Registers the built-in element implementations. |
| `hypernix.elements.carbon` | Provides the Carbon element implementation. |
| `hypernix.elements.elements_cli` | Exposes element discovery/management through the CLI. |
| `hypernix.elements.hydrogen` | Defines the base element/add-on framework and lifecycle. |
| `hypernix.elements.magnesium` | Provides the Magnesium element implementation. |
| `hypernix.elements.natural_gas` | Attaches and manages element add-ons on an oven. |

### Runtime

Runtime modules provide inference helpers and model/tool integration primitives.

| Module | What it does | Related guide |
|---|---|---|
| `hypernix.runtime.hub_kernels` | Integrates supported Hugging Face Hub kernels for selected model layers/operations. | [Llama Easy](Llama-Easy.md) |
| `hypernix.runtime.relay` | Splits long prompts into context-sized hops and coordinates sequential inference across smaller-context model instances. | [Context relay](Context-Relay.md) |
| `hypernix.runtime.t1tools` | Exposes permitted T1 API operations as tools for a model/agent runtime. | [T1 API](T1-API.md) |
| `hypernix.runtime.toolcalls` | Parses and normalizes tool-call requests from model output before dispatch. | [T1 API](T1-API.md) |

### Timing

Timing modules handle timers, scheduled training/evaluation, progress indicators, and step planning.

| Module | What it does | Related guide |
|---|---|---|
| `hypernix.timing.bell` | Streams generated tokens and invokes completion notifications/callbacks. | [Bell](Bell.md) |
| `hypernix.timing.coffee_maker` | Schedules training/evaluation runs at configured times or intervals. | [CoffeeMaker](CoffeeMaker.md) |
| `hypernix.timing.smoke_alarm` | Plans/recommends training steps and monitors expected duration using hardware presets or calibration. | [Alarms](Alarms.md) |
| `hypernix.timing.spinner` | Provides terminal progress/spinner animation primitives. | [Dashboards](Dashboards.md) |
| `hypernix.timing.timer` | Provides reusable countdown, interval, and Pomodoro-style timers. | [CLI](CLI.md) |

### Training

Training modules construct models, coordinate training, configure schedules, and apply optional training-time transformations.

| Module | What it does | Related guide |
|---|---|---|
| `hypernix.training.abbicus` | Adjusts training context/sequence length as a curriculum based on model size, step, and configured limits. | [Abbicus](Abbicus.md) |
| `hypernix.training.monitor` | Connects training runs to monitoring/telemetry and reports training progress. | [Dashboards](Dashboards.md) |
| `hypernix.training.apron` | Snapshots and restores RNG states, optionally seeding Python, NumPy, and PyTorch for reproducible work. | [Apron](Apron.md) |
| `hypernix.training.brewer` | Defines the HyperNix0x-v2 Brewer architecture presets and model construction helpers. | [Architectures](Architectures.md) |
| `hypernix.training.cake_pan` | Guards training with non-finite checks, time/memory safeguards, and rollback support. | [CakePan](CakePan.md) |
| `hypernix.training.camouflage` | Provides alignment/RLHF/RLAIF-style training scaffolding and optional evaluator integration. | [Camouflage](Camouflage.md) |
| `hypernix.training.compute_framework` | Abstracts supported compute devices and distributed training setup/gradient synchronization. | [Frameworks](Frameworks.md) |
| `hypernix.training.deep_fryer` | Applies reversible model-weight perturbations for regularization or stress/negative examples. | [DeepFryer](DeepFryer.md) |
| `hypernix.training.fizzle` | Provides training-time failure/stop handling helpers; consult source docs for its exact hooks and policy. | [Training](Training.md) |
| `hypernix.training.instant_pot` | Runs a configured end-to-end training recipe/pipeline. | [Kitchen](Kitchen.md) |
| `hypernix.training.lazy_suzan` | Coordinates supported multi-device/multi-GPU linking for distributed workloads. | [Frameworks](Frameworks.md) |
| `hypernix.training.mtp` | Implements Multi-Token Prediction heads/training support for predicting multiple future tokens. | [MTP](MTP.md) |
| `hypernix.training.recipe_book` | Stores named training/evaluation recipes and resolves them with per-run overrides. | [RecipeBook](RecipeBook.md) |
| `hypernix.training.smoker` | Provides tiered training-quality checks/assessment helpers. | [Smoker](Smoker.md) |
| `hypernix.training.train` | Exposes the core training entry points, including initialization, checkpoint expansion, and model training. | [Training](Training.md) |

## Additional package namespaces

These packages are part of the source tree but are not all entries in `MODULE_CATEGORIES`. Their modules are included here so the registry is not mistaken for the complete source inventory.

### Audio

The audio package reads files, processes signals, extracts features, and trains/runs wake-word detection. Its package documentation explicitly describes these boundaries.

| Module | What it does |
|---|---|
| `hypernix.audio.audiofile` | Reads audio input files and normalizes access to the incoming audio data. |
| `hypernix.audio.features` | Converts audio into feature representations such as log-mel frames. |
| `hypernix.audio.processor` | Provides audio signal-processing operations such as resampling, filtering, level control, silence splitting, and pipeline composition. |
| `hypernix.audio.wakeup` | Implements wake-word training and streaming detection using the audio pipeline. |
| `hypernix.audio.wakeup_cli` | Command-line interface for the wake-word workflow. |

### Bridge

A bridge connects to an inference server owned by another process. It does not manage that server's model lifecycle.

| Module | What it does |
|---|---|
| `hypernix.bridge.lmstudio` | Discovers and calls LM Studio's OpenAI-compatible local/LAN/Tailscale server. |

### C4 model analysis

C4 decomposes a model into analyzable parts, measures their contribution, and can assemble a derived model. It is a separate model-analysis workflow, not ordinary dataset preprocessing.

| Module | What it does |
|---|---|
| `hypernix.c4.dial` | Scores parameter groups/dials, identifies low-importance parts, and builds training-priority reports. |
| `hypernix.c4.explode` | Loads a model and decomposes its layers/weights into C4 analysis units. |
| `hypernix.c4.model` | Defines/loads the model representation used by C4. |
| `hypernix.c4.optim` | Implements C4-specific optimization and comparison strategies. |
| `hypernix.c4.probe` | Runs calibration/evaluation inputs against a model to measure behavior. |
| `hypernix.c4.reap` | Selects and extracts chosen layers, weights, or channels from an exploded model. |
| `hypernix.c4.restore` | Restores snapshots and supports controlled temporary ablation/restore operations. |
| `hypernix.c4.sow` | Assembles selected components into a new model and validates/heals the result. |
| `hypernix.c4.stem` | Reports and tests layer/stem-level importance and behavior. |
| `hypernix.c4.cli` | Command-line entry point for C4 analysis operations. |

### Dilute

Dilute generates multiple candidate completions, scores them, retains stronger samples, and can stream accepted samples to a sink. The evaluators can be model-based or ordinary Python functions.

| Module | What it does |
|---|---|
| `hypernix.dilute.backends` | Selects/adapts the model-generation backend used by the sampling workflow. |
| `hypernix.dilute.cli` | Command-line entry point for Dilute. |
| `hypernix.dilute.core` | Implements candidate generation, temperature ladders, ranking/tie-breaking, and streaming distillation. |
| `hypernix.dilute.evaluators` | Implements evaluator adapters, including model- and function-based scoring. |

### HyperLink

HyperLink is the phone/web client integration layer: pairing, sessions, attachments, model links, and communication with the T1 server. The HTTP routes themselves live under `hypernix.t1api.routers`.

| Module | What it does |
|---|---|
| `hypernix.hyperlink.brewed` | Handles the app's locally bundled/brewed client-side assets or data representation. |
| `hypernix.hyperlink.brewed_server` | Serves the bundled HyperLink web client. |
| `hypernix.hyperlink.capabilities` | Describes client/server capabilities used to negotiate supported behavior. |
| `hypernix.hyperlink.catalogue` | Builds or serves the HyperLink model/catalogue view. |
| `hypernix.hyperlink.codetools` | Provides code-oriented helpers exposed to the HyperLink client. |
| `hypernix.hyperlink.compaction` | Compacts conversation/context content to fit model limits. |
| `hypernix.hyperlink.datasets` | Imports or exposes dataset context for chat/workflows. |
| `hypernix.hyperlink.default_prompt` | Defines the default HyperLink system prompt. |
| `hypernix.hyperlink.defaultmodel` | Resolves the default model selection. |
| `hypernix.hyperlink.discovery` | Discovers and ranks local/network endpoints at which the server can be reached. |
| `hypernix.hyperlink.files` | Stores and retrieves content-addressed attachments such as images, documents, and code. |
| `hypernix.hyperlink.generation` | Coordinates generation requests from HyperLink. |
| `hypernix.hyperlink.handover` | Handles handover/continuation of a conversation or task between app/server contexts. |
| `hypernix.hyperlink.hfdownload` | Downloads model files through the HyperLink Hugging Face workflow. |
| `hypernix.hyperlink.hfmerge` | Resolves Hugging Face model pages and direct links into a complete GGUF download plan. |
| `hypernix.hyperlink.hyperchat` | Implements the HyperLink chat orchestration layer. |
| `hypernix.hyperlink.hypercompress` | Provides the HyperLink compression/compaction integration. |
| `hypernix.hyperlink.identity` | Represents and resolves the identity associated with a HyperLink client. |
| `hypernix.hyperlink.imagecodec` | Encodes/decodes image payloads for supported message transport. |
| `hypernix.hyperlink.inference` | Connects HyperLink requests to the selected inference path. |
| `hypernix.hyperlink.latest` | Resolves current/latest release information used by the client. |
| `hypernix.hyperlink.managed` | Coordinates managed model/runtime behavior exposed to HyperLink. |
| `hypernix.hyperlink.memory` | Implements HyperLink memory storage/access behavior subject to configured privacy controls. |
| `hypernix.hyperlink.modellinks` | Parses and resolves model links for the client. |
| `hypernix.hyperlink.notify` | Handles notification delivery/events for HyperLink. |
| `hypernix.hyperlink.offload` | Coordinates supported offload behavior for client/model workloads. |
| `hypernix.hyperlink.ondevice` | Integrates on-device model execution. |
| `hypernix.hyperlink.ownership` | Enforces/represents ownership boundaries for user data and sessions. |
| `hypernix.hyperlink.pairing` | Turns short pairing codes into per-device credentials and device records. |
| `hypernix.hyperlink.peers` | Tracks peer/server connections used by the client. |
| `hypernix.hyperlink.preferences` | Loads and persists HyperLink preferences. |
| `hypernix.hyperlink.ring` | Reports context/usage ring state for the client UI. |
| `hypernix.hyperlink.runner_history` | Stores or retrieves inference-run history. |
| `hypernix.hyperlink.search` | Implements the client-facing search integration. |
| `hypernix.hyperlink.sessions` | Stores server-side chat sessions and messages so conversations can continue across devices. |
| `hypernix.hyperlink.shell` | Implements the HyperLink shell/tool interaction boundary. |
| `hypernix.hyperlink.signing` | Signs and verifies requests/session-related payloads. |
| `hypernix.hyperlink.sync` | Synchronizes supported client/server state. |
| `hypernix.hyperlink.titles` | Creates or updates chat/session titles. |
| `hypernix.hyperlink.toolloop` | Coordinates the model/tool execution loop for HyperLink. |

### Noodle agents

Noodle is the multi-provider agent executor used by Hyped Pro. It normalizes provider calls, runs tools in a workspace, and verifies results. Tool execution and memory are subject to explicit configuration and approval.

| Module | What it does |
|---|---|
| `hypernix.interfaces.noodle.agent` | Runs an agent turn and records its events/results. |
| `hypernix.interfaces.noodle.cli` | Exposes Noodle operations through a command-line interface. |
| `hypernix.interfaces.noodle.hpo` | Implements hyperparameter/search-space optimization helpers used by Noodle workflows. |
| `hypernix.interfaces.noodle.hyped` | Connects Noodle sessions/events to Hyped Pro. |
| `hypernix.interfaces.noodle.providers` | Defines provider adapters and normalizes their chat/tool-call formats. |
| `hypernix.interfaces.noodle.swarm` | Coordinates multiple agents/tasks into a swarm execution. |
| `hypernix.interfaces.noodle.tools` | Defines workspace-scoped tools and dispatches approved tool calls. |
| `hypernix.interfaces.noodle.validate` | Provides syntax/command/result verifiers for agent outputs. |

### Neuron

Neuron trains compact networks for action and perception tasks. It supports supervised learning, imitation learning, reinforcement learning, and execution-based evaluation.

| Module | What it does |
|---|---|
| `hypernix.neuron.cli` | Command-line entry point for Neuron workflows. |
| `hypernix.neuron.data` | Prepares training/evaluation data for compact networks. |
| `hypernix.neuron.envs` | Defines the environment protocol and deterministic test environments. |
| `hypernix.neuron.evaluate` | Evaluates trained networks, including task execution/return metrics. |
| `hypernix.neuron.imitation` | Trains policies from recorded demonstrations. |
| `hypernix.neuron.nets` | Defines the compact network architectures used by Neuron. |
| `hypernix.neuron.rl` | Implements reinforcement-learning training. |
| `hypernix.neuron.supervised` | Implements supervised training for labeled examples. |

### Script generator

ScriptGen builds training scripts through a terminal wizard or GUI. Parameter definitions are shared between validation, widgets, and generated code.

| Module | What it does |
|---|---|
| `hypernix.scriptgen.app` | Holds form state, validation, preview, and GUI launch behavior. |
| `hypernix.scriptgen.cli` | Implements the ScriptGen command-line entry point. |
| `hypernix.scriptgen.highlight` | Provides syntax highlighting for generated script previews. |
| `hypernix.scriptgen.params` | Defines parameter metadata, groups, defaults, validation, and search/suggestions. |
| `hypernix.scriptgen.recipes` | Provides reusable parameter recipes/presets. |
| `hypernix.scriptgen.templates` | Generates complete scripts or injects configuration into existing scripts. |
| `hypernix.scriptgen.theme` | Defines palettes/fonts and checks palette contrast. |
| `hypernix.scriptgen.widgets` | Implements custom form controls used by the GUI. |

### T1 API server

The T1 API is the optional HTTP server. Its core auth, registry, routing, storage, and policy modules are separated from FastAPI routers so core behavior can be tested without the HTTP extra. See [T1 API](T1-API.md) for endpoint contracts.

| Module | What it does |
|---|---|
| `hypernix.t1api.accounts` | Local account records and account lifecycle operations. |
| `hypernix.t1api.accounts_cli` | Account-management CLI. |
| `hypernix.t1api.anthropic` | Anthropic-compatible API/provider integration. |
| `hypernix.t1api.app` | Assembles the FastAPI application and middleware/routes. |
| `hypernix.t1api.audit` | Durable audit events for security- and administration-sensitive actions. |
| `hypernix.t1api.auth` | Authenticates requests and resolves caller credentials/scopes. |
| `hypernix.t1api.authhistory` | Tracks authentication/session history. |
| `hypernix.t1api.backup` | Backs up and restores supported server data. |
| `hypernix.t1api.billing` | Tracks balances, transactions, and billing ledger state. |
| `hypernix.t1api.billingkeys` | Associates payment/billing bindings with API keys. |
| `hypernix.t1api.bootstrap` | Creates the restricted initial administrator/bootstrap credential. |
| `hypernix.t1api.chat_cli` | Command-line chat client for the T1 API. |
| `hypernix.t1api.claudecode` | Integrates Claude Code with the T1 API's supported model/tool path. |
| `hypernix.t1api.codebox` | Provides the server-side code workspace/sandbox boundary. |
| `hypernix.t1api.codegit` | Provides Git operations for the code workspace. |
| `hypernix.t1api.config` | Defines and validates T1 server configuration. |
| `hypernix.t1api.cost` | Estimates and reports model/request costs. |
| `hypernix.t1api.counter` | Tracks usage counters and metering data. |
| `hypernix.t1api.db` | Defines database/backend access primitives. |
| `hypernix.t1api.deploy` | Coordinates deployment configuration and lifecycle. |
| `hypernix.t1api.deps` | Checks optional server dependencies. |
| `hypernix.t1api.disconnect` | Handles disconnect/cleanup behavior for server connections. |
| `hypernix.t1api.envupgrade` | Migrates supported environment/configuration values between versions. |
| `hypernix.t1api.errors` | Defines T1 API error types and stable error codes. |
| `hypernix.t1api.events` | Publishes/polls server events and supports streaming event delivery. |
| `hypernix.t1api.hyperlink_web` | Serves the HyperLink web application from the T1 server. |
| `hypernix.t1api.jobs` | Queues and manages asynchronous jobs, status, and cancellation. |
| `hypernix.t1api.keys` | Implements API-key directory and key-related operations. |
| `hypernix.t1api.launchscript_cli` | CLI for creating/running managed launch scripts. |
| `hypernix.t1api.lmsoverride` | Applies configured LM Studio model/provider overrides. |
| `hypernix.t1api.localserver` | Manages the local server process/service integration. |
| `hypernix.t1api.mcp` | Implements the server-side MCP integration surface. |
| `hypernix.t1api.mcpclient` | Connects to configured external MCP servers. |
| `hypernix.t1api.mcptools` | Exposes approved T1 operations as MCP tools. |
| `hypernix.t1api.modelindex` | Indexes local GGUF models for server-side discovery. |
| `hypernix.t1api.modelindex_cli` | CLI for building/refreshing the model index. |
| `hypernix.t1api.modelsync` | Synchronizes model files/metadata across configured locations. |
| `hypernix.t1api.modelsync_cli` | CLI for model synchronization. |
| `hypernix.t1api.modules` | Stores/version-tracks registered modules and handles module transport metadata; stored modules are not automatically executed. |
| `hypernix.t1api.mtls` | Configures/verifies mutual TLS client certificates. |
| `hypernix.t1api.netpolicy` | Applies network allow/block and private-address policy. |
| `hypernix.t1api.picker` | Provides interactive model/backend selection. |
| `hypernix.t1api.pricing` | Defines pricing rules and cost tables. |
| `hypernix.t1api.privacy` | Enforces server privacy and retention behavior. |
| `hypernix.t1api.ratelimit` | Applies request rate limits and limit accounting. |
| `hypernix.t1api.registry` | Maintains the allowlisted model registry and model status. |
| `hypernix.t1api.routing` | Selects a model/server using configured routing and quota policy. |
| `hypernix.t1api.runner_cli` | CLI for running/inspecting the T1 inference runner. |
| `hypernix.t1api.schemas` | Defines request/response schemas for HTTP endpoints. |
| `hypernix.t1api.security` | Implements shared security guards, including URL/path validation. |
| `hypernix.t1api.servers` | Maintains remote server records, trust, and health/status. |
| `hypernix.t1api.setup_cli` | CLI for setting up/configuring the T1 server. |
| `hypernix.t1api.storage` | Provides persistent storage abstractions and backend selection. |
| `hypernix.t1api.training_cli` | CLI for inspecting/controlling server-managed training jobs. |
| `hypernix.t1api.transport` | Defines transport behavior for inter-server/API communication. |
| `hypernix.t1api.upgrade` | Implements supported T1 server upgrade/version-check operations. |
| `hypernix.t1api.usage` | Records and reports per-key/per-model usage. |
| `hypernix.t1api.version` | Defines the T1 API's independent six-part protocol version. |
| `hypernix.t1api.webauth` | Implements browser-based authentication flows. |
| `hypernix.t1api.webconfig_cli` | CLI for web-facing T1 configuration. |
| `hypernix.t1api.websearch` | Implements the legacy web-search endpoint/integration. |
| `hypernix.t1api.websearch_v2` | Implements the versioned web-search API and its policy-aware behavior. |

#### T1 API routers

Each router module owns one HTTP endpoint group; the core service logic remains in the modules above.

| Router module | Endpoint responsibility |
|---|---|
| `routers.accounts` | Account endpoints. |
| `routers.anthropic` | Anthropic-compatible messages endpoint. |
| `routers.audit` | Audit-log endpoints. |
| `routers.auth` | Authentication and sign-in endpoints. |
| `routers.authundo` | Authentication history/undo operations. |
| `routers.backup` | Backup/restore endpoints. |
| `routers.billing` | Billing and payment-binding endpoints. |
| `routers.bridge` | LM Studio/bridge endpoints. |
| `routers.code` | Code workspace and tool endpoints. |
| `routers.compact` | Context compaction endpoint. |
| `routers.config` | Server configuration endpoints. |
| `routers.counter` | Usage-counter endpoints. |
| `routers.events` | Event polling/stream endpoints. |
| `routers.health` | Health and status endpoints. |
| `routers.hypercompress` | HyperCompress endpoint. |
| `routers.hyperlink` | HyperLink pairing, session, file, and client endpoints. |
| `routers.inference` | Inference/chat endpoints. |
| `routers.jobs` | Async job endpoints. |
| `routers.keys` | API-key endpoints. |
| `routers.mcp` | MCP endpoints. |
| `routers.memory` | Memory endpoints. |
| `routers.models` | Model discovery and selection endpoints. |
| `routers.modules` | Module registry/transport endpoints. |
| `routers.net` | Network and connection endpoints. |
| `routers.noodle` | Noodle agent endpoints. |
| `routers.privacy` | Privacy and retention endpoints. |
| `routers.runner` | Runner lifecycle and model execution endpoints. |
| `routers.security` | Security policy/diagnostic endpoints. |
| `routers.servers` | Remote server registry endpoints. |
| `routers.t2c` | T2C key-related endpoints. |
| `routers.training` | Training job endpoints. |
| `routers.usage` | Usage and cost endpoints. |
| `routers.web` | Web application/static web endpoints. |

### T1 SDK

The SDK is a standard-library-only client; it does not require installing the optional T1 server dependencies.

| Module | What it does |
|---|---|
| `hypernix.t1sdk.client` | Implements the typed T1 API client methods. |
| `hypernix.t1sdk.errors` | Defines client-side typed API/transport errors. |
| `hypernix.t1sdk.models` | Defines typed response/request data models. |
| `hypernix.t1sdk.transport` | Implements HTTP transport, retries, and TLS configuration. |

### Waiter client

Waiter is the official T1 API TUI/CLI client and delegates its API contract to the T1 SDK.

| Module | What it does |
|---|---|
| `hypernix.waiter.cli` | Parses and dispatches Waiter commands. |
| `hypernix.waiter.client` | Compatibility client layer over the T1 SDK. |
| `hypernix.waiter.diagnose` | Diagnoses server connectivity and configuration. |
| `hypernix.waiter.discovery` | Discovers configured/available T1 servers. |
| `hypernix.waiter.kits` | Provides reusable command/workflow helpers. |
| `hypernix.waiter.local_config` | Loads and stores local Waiter configuration. |
| `hypernix.waiter.servargs` | Parses/normalizes server-operation arguments. |
| `hypernix.waiter.servops` | Performs server lifecycle/administration operations. |
| `hypernix.waiter.smoke` | Runs API smoke checks against a server. |
| `hypernix.waiter.tui` | Implements the curses terminal interface. |

## How modules fit together

A typical training path composes modules rather than relying on one monolith:

1. **Find/prepare data:** `data.scavenger` or `data.gather` discovers sources; `data.strainer`, `data.pans`, `data.qa`, and `data.cutting_board` prepare and split records.
2. **Build/load a model:** `models.arch`, `training.brewer`, `models.download`, and `models.old_oven` cover architecture selection, model creation, downloading, and loading.
3. **Train safely:** `training.train`, `optimizers.pressure_cooker_v*`, `training.compute_framework`, `system.freezer`, `system.vram`, and `timing.smoke_alarm` provide the training loop, optimizer, device, memory, and schedule controls.
4. **Evaluate:** `evaluation.espresso_maker`, `evaluation.*_range`, and `data.mediocre_fridge` provide prompt batteries, rubrics, and judge data.
5. **Export/serve:** `quant.convert`, `quant.quantize`, `quant.upload`, `models.multilama`, and `t1api` cover export, quantization, inference backends, and optional remote serving.

A chat path commonly uses `models.old_oven` + `chat.cookbook` + `chat.countertop`, with `chat.flour` for output controls and `timing.bell` for streaming. HyperLink adds pairing/session/attachment handling around the T1 API; the bridge package connects to external inference servers rather than managing them.

## Runnable examples and deeper references

- The repository's runnable scripts live in [`examples/`](../examples/); these are code examples, distinct from this module encyclopedia.
- The T1 deployment and request examples live in [`examples/t1api/`](../examples/t1api/).
- The [wiki home](Home.md) indexes focused guides for individual subsystems.
- The [CLI reference](CLI.md) documents commands and flags.
- The [Architecture map](Architecture.md) shows subsystem relationships.
- The [Roadmap](Roadmap.md) describes planned work; planned features are not claims about current behavior.

## Keeping this page accurate

- Treat `hypernix.MODULE_CATEGORIES` as the authoritative registry for the public categorized module surface.
- Treat the actual `src/hypernix/` tree as the inventory for additional packages and implementation modules.
- When a module moves or is renamed, update this page and the corresponding source/API references in the same change.
- For each new module, add its path, purpose, public/internal status, and a link to a detailed guide or source.
- Do not describe planned modules as implemented. Mark experimental/beta behavior and platform/provider limitations explicitly.
- Run the module-layout tests and docs/link checks after edits. Do not claim a module's behavior based solely on its filename.
