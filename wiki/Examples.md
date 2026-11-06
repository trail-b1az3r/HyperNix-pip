# Examples — HyperNix Module Encyclopedia

> A source-oriented guide to what the Python modules in `src/hypernix/` do. The current source tree contains **429 Python files: 399 implementation modules and 30 package initializers**. This page is an index and explanation of the package, not a replacement for API reference pages or runnable scripts in `examples/`. Public module names are identified separately from internal implementation modules.

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
- [Additional implementation modules](#additional-implementation-modules)
  - [Interfaces and compatibility](#interfaces-and-compatibility)
  - [Model and runtime implementation](#model-and-runtime-implementation)
  - [Monitoring implementation](#monitoring-implementation)
  - [Optimizer and quantization implementation](#optimizer-and-quantization-implementation)
  - [Security implementation](#security-implementation)
  - [System implementation](#system-implementation)
  - [T1 API router modules](#t1-api-router-modules)
- [Package initializers](#package-initializers)
- [How modules fit together](#how-modules-fit-together)
- [Runnable examples and deeper references](#runnable-examples-and-deeper-references)
- [Keeping this page accurate](#keeping-this-page-accurate)

## Using this page

The canonical public grouping is [`hypernix.MODULE_CATEGORIES`](../src/hypernix/MODULE_CATEGORIES.py) in [`src/hypernix/__init__.py`](../src/hypernix/__init__.py). Import paths shown below use the current category layout. The legacy flat import aliases are retained for compatibility where the package provides them; new code should prefer the category path.

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
| [`hypernix.chat.cookbook`](../src/hypernix/chat/cookbook.py) | Selects and applies model-appropriate chat templates, including ChatML, Llama, Alpaca, Vicuna, and HyperNix formats. | [Cookbook](Cookbook.md) |
| [`hypernix.chat.countertop`](../src/hypernix/chat/countertop.py) | Maintains multi-turn conversation history around an oven/model; supports sending turns, resetting, saving/loading, trimming history, and optional streaming/output processing. | [Countertop](Countertop.md) |
| [`hypernix.chat.flour`](../src/hypernix/chat/flour.py) | Applies generation-time output controls such as repetition/frequency/presence penalties, no-repeat n-grams, bad-word handling, and role-leak suppression. | [Flour](Flour.md) |
| [`hypernix.chat.injection`](../src/hypernix/chat/injection.py) | Provides token/prompt injection helpers for controlled thinking, testing, system overrides, and custom injected content. | [Camouflage](Camouflage.md) |
| [`hypernix.chat.menu`](../src/hypernix/chat/menu.py) | Stores named system-prompt/persona presets and resolves a requested persona for a chat session. | [Menu](Menu.md) |

### Data

Data modules discover, collect, transform, split, augment, package, and persist datasets.

| Module | What it does | Related guide |
|---|---|---|
| [`hypernix.data.blender`](../src/hypernix/data/blender.py) | Mixes/interleaves multiple data sources using selectable processing tiers. | [Blender](Blender.md) |
| [`hypernix.data.cardboard_box`](../src/hypernix/data/cardboard_box.py) | Stores tokenized datasets efficiently and supports appending tokens, reading chunks, and logically pruning old data. | [Tupperware](Tupperware.md) |
| [`hypernix.data.cutting_board`](../src/hypernix/data/cutting_board.py) | Splits records into train/validation/test sets, including deterministic and stratified splitting. | [CuttingBoard](CuttingBoard.md) |
| [`hypernix.data.food_processor`](../src/hypernix/data/food_processor.py) | Chunks or segments large text/data records using selectable chopping, slicing, shredding, and purée strategies. | [FoodProcessor](FoodProcessor.md) |
| [`hypernix.data.gather`](../src/hypernix/data/gather.py) | Collects and retrieves datasets/samples from configured sources; handles discovery and collection workflow rather than model training. | [Scavenger](Scavenger.md) |
| [`hypernix.data.gather_cli`](../src/hypernix/data/gather_cli.py) | Command-line entry point for the data-gathering workflow. | [CLI](CLI.md) |
| [`hypernix.data.lunchbox`](../src/hypernix/data/lunchbox.py) | Packages datasets into consistent schemas for evaluation or Hub upload, catching column mismatches before publishing. | [Lunchbox](Lunchbox.md) |
| [`hypernix.data.mediocre_fridge`](../src/hypernix/data/mediocre_fridge.py) | Builds judge-training corpora and collects model responses for later scoring/training. | [Fridges](Fridges.md) |
| [`hypernix.data.pans`](../src/hypernix/data/pans.py) | Applies staged text/data preprocessing through the FryingPan-to-Wok processing tiers. | [Kitchen](Kitchen.md) |
| [`hypernix.data.pepper_shaker`](../src/hypernix/data/pepper_shaker.py) | Applies stronger/specific text perturbations, including masking, typo, and negation-style variants. | [Shakers](Shakers.md) |
| [`hypernix.data.qa`](../src/hypernix/data/qa.py) | Converts structured question/answer or text records into causal-language-model training text. | [Kitchen](Kitchen.md) |
| [`hypernix.data.salt_shaker`](../src/hypernix/data/salt_shaker.py) | Applies gentler dataset augmentation/variation strategies. | [Shakers](Shakers.md) |
| [`hypernix.data.scavenger`](../src/hypernix/data/scavenger.py) | Searches Hugging Face datasets and filters candidates by configured relevance, quality, recency, and storage constraints. | [Scavenger](Scavenger.md) |
| [`hypernix.data.sink`](../src/hypernix/data/sink.py) | Writes output records through an append-oriented sink with optional rotation and deduplication. | [Kitchen](Kitchen.md) |
| [`hypernix.data.strainer`](../src/hypernix/data/strainer.py) | Filters dataset records, including near-duplicate detection, before training/evaluation. | [Kitchen](Kitchen.md) |
| [`hypernix.data.toaster`](../src/hypernix/data/toaster.py) | Formats or transforms records line by line using selectable output-processing tiers. | [Toaster](Toaster.md) |
| [`hypernix.data.tupperware`](../src/hypernix/data/tupperware.py) | Splits/organizes data into rounds or staged subsets, with optional training/evaluation support. | [Tupperware](Tupperware.md) |

### Evaluation

Evaluation modules score outputs, define rubrics, visualize results, and verify module quality.

| Module | What it does | Related guide |
|---|---|---|
| [`hypernix.evaluation.espresso_maker`](../src/hypernix/evaluation/espresso_maker.py) | Runs prompt batteries at selectable evaluation tiers and returns scored evaluation results. | [EspressoMaker](EspressoMaker.md) |
| [`hypernix.evaluation.industrial_range`](../src/hypernix/evaluation/industrial_range.py) | Uses a model-compatible judge to evaluate outputs, including pairwise preference comparisons. | [Ranges](Ranges.md) |
| [`hypernix.evaluation.new_fridge`](../src/hypernix/evaluation/new_fridge.py) | Parses training/evaluation logs and produces plots such as loss curves and score distributions. | [Fridges](Fridges.md) |
| [`hypernix.evaluation.new_range`](../src/hypernix/evaluation/new_range.py) | Provides a lightweight first-fail labeling rubric for evaluating responses. | [Ranges](Ranges.md) |
| [`hypernix.evaluation.old_range`](../src/hypernix/evaluation/old_range.py) | Provides weighted, explainable rule-based scoring for response evaluation. | [Ranges](Ranges.md) |
| [`hypernix.evaluation.vera`](../src/hypernix/evaluation/vera.py) | Checks modules for syntax, documentation/type coverage, and configured smoke-test expectations. | [Vera](Vera.md) |

### Interfaces

Interface modules expose HyperNix functionality through commands, terminal/desktop UIs, and application bridges.

| Module | What it does | Related guide |
|---|---|---|
| [`hypernix.interfaces.assistant`](../src/hypernix/interfaces/assistant.py) | Implements the assistant-facing interaction layer and its command integration. | [CLI](CLI.md) |
| [`hypernix.interfaces.cli`](../src/hypernix/interfaces/cli.py) | Implements the main HyperNix command-line interface and subcommand dispatch. | [CLI](CLI.md) |
| [`hypernix.interfaces.hyped`](../src/hypernix/interfaces/hyped.py) | Implements the basic local-AI terminal chat experience. | [CLI](CLI.md) |
| [`hypernix.interfaces.hyped_pro`](../src/hypernix/interfaces/hyped_pro.py) | Implements the expanded Hyped Pro terminal interface and its application workflow. | [CLI](CLI.md) |
| [`hypernix.interfaces.hyped_pro_bridge`](../src/hypernix/interfaces/hyped_pro_bridge.py) | Connects the Hyped Pro front end to its Python/backend operations. | [CLI](CLI.md) |
| [`hypernix.interfaces.hyped_pro_core`](../src/hypernix/interfaces/hyped_pro_core.py) | Contains the shared model/provider dispatch and core Hyped Pro behavior. | [CLI](CLI.md) |
| [`hypernix.interfaces.hyped_pro_git`](../src/hypernix/interfaces/hyped_pro_git.py) | Provides Git operations used by the Hyped Pro workspace. | [CLI](CLI.md) |
| [`hypernix.interfaces.hyped_pro_gui`](../src/hypernix/interfaces/hyped_pro_gui.py) | Launches/integrates the desktop GUI for Hyped Pro. | [CLI](CLI.md) |
| [`hypernix.interfaces.hyped_pro_otui`](../src/hypernix/interfaces/hyped_pro_otui.py) | Implements the OpenTUI-based terminal UI. | [CLI](CLI.md) |
| [`hypernix.interfaces.hyped_pro_tools`](../src/hypernix/interfaces/hyped_pro_tools.py) | Defines and dispatches tools exposed by Hyped Pro. | [CLI](CLI.md) |
| [`hypernix.interfaces.version_launcher`](../src/hypernix/interfaces/version_launcher.py) | Selects/launches the installed HyperNix version or entry point. | [CLI](CLI.md) |
| [`hypernix.interfaces.websearch`](../src/hypernix/interfaces/websearch.py) | Exposes the interactive web-search interface/command. | [CLI](CLI.md) |
| [`hypernix.interfaces.wiki_cli`](../src/hypernix/interfaces/wiki_cli.py) | Builds and browses module documentation from source documentation. | [Vera](Vera.md) |

### Models

Model modules define architectures, load snapshots, generate text, and connect supported model families.

| Module | What it does | Related guide |
|---|---|---|
| [`hypernix.models.arch`](../src/hypernix/models/arch.py) | Defines architecture presets and model-family configuration used when creating/loading models. | [Architectures](Architectures.md) |
| [`hypernix.models.brewer_adapter`](../src/hypernix/models/brewer_adapter.py) | Adapts the HyperNix Brewer model/configuration to supported external model/runtime interfaces. | [Architectures](Architectures.md) |
| [`hypernix.models.download`](../src/hypernix/models/download.py) | Downloads model snapshots and resolves short names, gated repositories, and cached/offline copies. | [Download](Download.md) |
| [`hypernix.models.generate`](../src/hypernix/models/generate.py) | Provides text-generation helpers over loaded models. | [Ovens](Ovens.md) |
| [`hypernix.models.microwave`](../src/hypernix/models/microwave.py) | Runs short-lived/one-shot inference using a selected model and generation tier. | [Microwave](Microwave.md) |
| [`hypernix.models.multilama`](../src/hypernix/models/multilama.py) | Dispatches GGUF inference across supported llama.cpp-family backends, with capability-aware selection/fallback. | [Runtime](Runtime.md) |
| [`hypernix.models.nano_nano`](../src/hypernix/models/nano_nano.py) | Implements the Nano-Nano model family and its model-specific configuration. | [Architectures](Architectures.md) |
| [`hypernix.models.neo_oven`](../src/hypernix/models/neo_oven.py) | Implements the Neo Oven model/training interface and its architecture-aware workflow. | [Ovens](Ovens.md) |
| [`hypernix.models.old_oven`](../src/hypernix/models/old_oven.py) | Provides the CodeOven snapshot wrapper for completion, chat, fill, training, and snapshot-related operations. | [Ovens](Ovens.md) |
| [`hypernix.models.stml`](../src/hypernix/models/stml.py) | Provides context-length/VRAM estimation and sequence-segmentation helpers for training. | [STML](STML.md) |
| [`hypernix.models.whisk`](../src/hypernix/models/whisk.py) | Averages model checkpoints using supported averaging strategies and writes a merged snapshot. | [Whisk](Whisk.md) |
| [`hypernix.models.workshop`](../src/hypernix/models/workshop.py) | Supplies reusable model-framework templates and task pipelines for LLM, vision, TTS, and ASR workloads. | [Workshop](Workshop.md) |

### Monitoring

Monitoring modules collect runtime/training telemetry, display dashboards, and inspect runs.

| Module | What it does | Related guide |
|---|---|---|
| [`hypernix.monitoring.cctvtop`](../src/hypernix/monitoring/cctvtop.py) | Runs the enhanced terminal dashboard combining training and host monitoring views. | [Dashboards](Dashboards.md) |
| [`hypernix.monitoring.hyper_log`](../src/hypernix/monitoring/hyper_log.py) | Provides structured training/runtime logging helpers. | [HyperLog](HyperLog.md) |
| [`hypernix.monitoring.map`](../src/hypernix/monitoring/map.py) | Maps/displays hardware and runtime information for monitoring components. | [Dashboards](Dashboards.md) |
| [`hypernix.monitoring.plasma`](../src/hypernix/monitoring/plasma.py) | Runs a short hardware calibration workload and returns a factor usable by training-time estimates. | [Alarms](Alarms.md) |
| [`hypernix.monitoring.table`](../src/hypernix/monitoring/table.py) | Displays tabular data, including parsed training logs and judge corpora. | [Dashboards](Dashboards.md) |
| [`hypernix.monitoring.thermometer`](../src/hypernix/monitoring/thermometer.py) | Samples supported CPU/GPU temperature sensors. | [Devices](Devices.md) |
| [`hypernix.monitoring.tv`](../src/hypernix/monitoring/tv.py) | Provides the original lightweight terminal training monitor. | [Dashboards](Dashboards.md) |
| [`hypernix.monitoring.tvtop`](../src/hypernix/monitoring/tvtop.py) | Preserves the legacy tvtop import/command surface and compatibility behavior. | [Dashboards](Dashboards.md) |
| [`hypernix.monitoring.tvtop_plus_plus`](../src/hypernix/monitoring/tvtop_plus_plus.py) | Implements the richer terminal training dashboard and its display components. | [Dashboards](Dashboards.md) |

### Optimizers

Optimizer modules implement the Pressure Cooker generations and their common dispatch/compatibility layer.

| Module | What it does | Related guide |
|---|---|---|
| [`hypernix.optimizers.deprecation`](../src/hypernix/optimizers/deprecation.py) | Centralizes optimizer deprecation warnings and compatibility notices. | [Optimizers](Optimizers.md) |
| [`hypernix.optimizers.optimizer_framework`](../src/hypernix/optimizers/optimizer_framework.py) | Provides shared optimizer selection/configuration and common optimizer plumbing. | [Optimizers](Optimizers.md) |
| [`hypernix.optimizers.pressure_cooker`](../src/hypernix/optimizers/pressure_cooker.py) | Implements the original tiered AdamW-style Pressure Cooker optimizer family. | [Optimizers](Optimizers.md) |
| [`hypernix.optimizers.pressure_cooker_v3`](../src/hypernix/optimizers/pressure_cooker_v3.py) | Implements the V3 optimizer with quantized-state and ZeRO-style sharding features. | [Pressure Cooker v3](Pressure-Cooker-V3.md) |
| [`hypernix.optimizers.pressure_cooker_v4`](../src/hypernix/optimizers/pressure_cooker_v4.py) | Implements the V4 optimizer generation and its quantization-aware training integration. | [Pressure Cooker v4](Pressure-Cooker-V4.md) |
| [`hypernix.optimizers.pressure_cooker_v5`](../src/hypernix/optimizers/pressure_cooker_v5.py) | Implements the V5/V5+ optimizer family, including compressed momentum/state, QAT, and EMA mechanisms. | [Pressure Cooker v5](Pressure-Cooker-V5.md) |
| [`hypernix.optimizers.pressure_cooker_v5s`](../src/hypernix/optimizers/pressure_cooker_v5s.py) | Implements the V5S optimizer variant and its oscillation/stability mechanisms. | [Pressure Cooker v5](Pressure-Cooker-V5.md) |
| [`hypernix.optimizers.pressure_cooker_v6`](../src/hypernix/optimizers/pressure_cooker_v6.py) | Implements the speed-oriented V6 optimizer with multi-tensor operations and reduced synchronization. | [Pressure Cooker v6](Pressure-Cooker-V6.md) |
| [`hypernix.optimizers.pressure_cooker_v6v`](../src/hypernix/optimizers/pressure_cooker_v6v.py) | Adds the V6V execution path, including optional compile/CUDA-graph acceleration where supported. | [Pressure Cooker v6](Pressure-Cooker-V6.md) |

### Quantization

Quant modules convert, inspect, quantize, fetch required tooling, and upload model artifacts.

| Module | What it does | Related guide |
|---|---|---|
| [`hypernix.quant.convert`](../src/hypernix/quant/convert.py) | Converts supported model checkpoints to GGUF. | [Convert](Convert.md) |
| [`hypernix.quant.fetcher`](../src/hypernix/quant/fetcher.py) | Finds/fetches the external quantization tooling required by selected workflows. | [Pipeline mechanics](PipelineMechanics.md) |
| [`hypernix.quant.quantize`](../src/hypernix/quant/quantize.py) | Drives llama.cpp's `llama-quantize` utility for supported GGUF quant types. | [Quantization](Quantization.md) |
| [`hypernix.quant.upload`](../src/hypernix/quant/upload.py) | Uploads generated model artifacts to a Hugging Face repository. | [Quantization](Quantization.md) |

### Security

Security modules manage API credentials, key lifecycle, and request authorization.

| Module | What it does | Related guide |
|---|---|---|
| [`hypernix.security.gatekeeper`](../src/hypernix/security/gatekeeper.py) | Validates credentials and enforces configured request/key access rules. | [T1 API](T1-API.md) |
| [`hypernix.security.gkey_cli`](../src/hypernix/security/gkey_cli.py) | Provides command-line operations for the key-management surface. | [T1 API](T1-API.md) |
| [`hypernix.security.keymaster`](../src/hypernix/security/keymaster.py) | Creates, stores, validates, rotates, and revokes HyperNix API keys and their metadata. | [T1 API](T1-API.md) |

### System

System modules detect/configure the environment, manage hardware/resources, and provide housekeeping and diagnostics.

| Module | What it does | Related guide |
|---|---|---|
| [`hypernix.system.compactor`](../src/hypernix/system/compactor.py) | Archives older checkpoints/artifacts according to retention settings. | [Training](Training.md) |
| [`hypernix.system.config`](../src/hypernix/system/config.py) | Loads and validates shared HyperNix configuration. | [CLI](CLI.md) |
| [`hypernix.system.deps`](../src/hypernix/system/deps.py) | Detects and manages optional runtime dependencies. | [Devices](Devices.md) |
| [`hypernix.system.dishwasher`](../src/hypernix/system/dishwasher.py) | Cleans selected stale logs, checkpoints, and build artifacts. | [Training](Training.md) |
| [`hypernix.system.doctor`](../src/hypernix/system/doctor.py) | Diagnoses installation, dependencies, hardware, and runtime readiness. | [CLI](CLI.md) |
| [`hypernix.system.ethanol`](../src/hypernix/system/ethanol.py) | Provides guarded GPU clock/overclock controls through available vendor utilities; requires explicit confirmation. | [Devices](Devices.md) |
| [`hypernix.system.freezer`](../src/hypernix/system/freezer.py) | Selects memory/VRAM management strategies and OOM-aware retry behavior. | [Freezer](Freezer.md) |
| [`hypernix.system.gpus`](../src/hypernix/system/gpus.py) | Detects and describes GPU devices and their available backend capabilities. | [Devices](Devices.md) |
| [`hypernix.system.fusebox`](../src/hypernix/system/fusebox.py) | Provides GPU thermal/power-aware training control and safety integration. | [Devices](Devices.md) |
| [`hypernix.system.errorcatalogue`](../src/hypernix/system/errorcatalogue.py) | Defines the structured error catalogue and human-readable explanations. | [Error codes](Error-Codes.md) |
| [`hypernix.system.errorcodes`](../src/hypernix/system/errorcodes.py) | Defines stable error-code identifiers and metadata. | [Error codes](Error-Codes.md) |
| [`hypernix.system.errors_cli`](../src/hypernix/system/errors_cli.py) | Exposes error lookup/explanation through the CLI. | [Error codes](Error-Codes.md) |
| [`hypernix.system.hubcompat`](../src/hypernix/system/hubcompat.py) | Handles compatibility boundaries for Hugging Face Hub APIs/versions. | [Download](Download.md) |
| [`hypernix.system.net`](../src/hypernix/system/net.py) | Provides network discovery and connectivity helpers. | [T1 API](T1-API.md) |
| [`hypernix.system.nettrust`](../src/hypernix/system/nettrust.py) | Evaluates configured network trust/allowlist policy. | [T1 API](T1-API.md) |
| [`hypernix.system.old_fridge`](../src/hypernix/system/old_fridge.py) | Provides memory housekeeping, parameter freezing/unfreezing, and CPU offload helpers. | [Fridges](Fridges.md) |
| [`hypernix.system.outage`](../src/hypernix/system/outage.py) | Temporarily blanks/suppresses supported displays during long-running work and restores them on exit. | [Training](Training.md) |
| [`hypernix.system.pathfix`](../src/hypernix/system/pathfix.py) | Applies or undoes the console-script PATH setup. | [CLI](CLI.md) |
| [`hypernix.system.protect`](../src/hypernix/system/protect.py) | Monitors display blanking, wake-word activity, and hardware health logging. | [CLI](CLI.md) |
| [`hypernix.system.release`](../src/hypernix/system/release.py) | Reads and reports HyperNix release/version information. | [CLI](CLI.md) |
| [`hypernix.system.torch_compat`](../src/hypernix/system/torch_compat.py) | Supplies PyTorch compatibility helpers for supported older runtimes/platforms. | [macOS legacy](macOS-legacy.md) |
| [`hypernix.system.ups`](../src/hypernix/system/ups.py) | Adds an uninterruptible-power/outage-aware training safety mode and checkpoint callback behavior. | [Training](Training.md) |
| [`hypernix.system.utils`](../src/hypernix/system/utils.py) | Collects shared utilities such as health checks, model listing, and environment diagnostics. | [CLI](CLI.md) |
| [`hypernix.system.vram`](../src/hypernix/system/vram.py) | Provides opt-in allocator, activation-checkpointing, optimizer-state, and peak-memory helpers. | [VRAM](VRAM.md) |

### Elements

The elements namespace is a small add-on framework: Hydrogen defines the extension model, Natural Gas attaches add-ons to an oven, and individual elements provide capabilities.

| Module | What it does |
|---|---|
| [`hypernix.elements.builtins`](../src/hypernix/elements/builtins.py) | Registers the built-in element implementations. |
| [`hypernix.elements.carbon`](../src/hypernix/elements/carbon.py) | Provides the Carbon element implementation. |
| [`hypernix.elements.elements_cli`](../src/hypernix/elements/elements_cli.py) | Exposes element discovery/management through the CLI. |
| [`hypernix.elements.hydrogen`](../src/hypernix/elements/hydrogen.py) | Defines the base element/add-on framework and lifecycle. |
| [`hypernix.elements.magnesium`](../src/hypernix/elements/magnesium.py) | Provides the Magnesium element implementation. |
| [`hypernix.elements.natural_gas`](../src/hypernix/elements/natural_gas.py) | Attaches and manages element add-ons on an oven. |

### Runtime

Runtime modules provide inference helpers and model/tool integration primitives.

| Module | What it does | Related guide |
|---|---|---|
| [`hypernix.runtime.hub_kernels`](../src/hypernix/runtime/hub_kernels.py) | Integrates supported Hugging Face Hub kernels for selected model layers/operations. | [Llama Easy](Llama-Easy.md) |
| [`hypernix.runtime.relay`](../src/hypernix/runtime/relay.py) | Splits long prompts into context-sized hops and coordinates sequential inference across smaller-context model instances. | [Context relay](Context-Relay.md) |
| [`hypernix.runtime.t1tools`](../src/hypernix/runtime/t1tools.py) | Exposes permitted T1 API operations as tools for a model/agent runtime. | [T1 API](T1-API.md) |
| [`hypernix.runtime.toolcalls`](../src/hypernix/runtime/toolcalls.py) | Parses and normalizes tool-call requests from model output before dispatch. | [T1 API](T1-API.md) |

### Timing

Timing modules handle timers, scheduled training/evaluation, progress indicators, and step planning.

| Module | What it does | Related guide |
|---|---|---|
| [`hypernix.timing.bell`](../src/hypernix/timing/bell.py) | Streams generated tokens and invokes completion notifications/callbacks. | [Bell](Bell.md) |
| [`hypernix.timing.coffee_maker`](../src/hypernix/timing/coffee_maker.py) | Schedules training/evaluation runs at configured times or intervals. | [CoffeeMaker](CoffeeMaker.md) |
| [`hypernix.timing.smoke_alarm`](../src/hypernix/timing/smoke_alarm.py) | Plans/recommends training steps and monitors expected duration using hardware presets or calibration. | [Alarms](Alarms.md) |
| [`hypernix.timing.spinner`](../src/hypernix/timing/spinner.py) | Provides terminal progress/spinner animation primitives. | [Dashboards](Dashboards.md) |
| [`hypernix.timing.timer`](../src/hypernix/timing/timer.py) | Provides reusable countdown, interval, and Pomodoro-style timers. | [CLI](CLI.md) |

### Training

Training modules construct models, coordinate training, configure schedules, and apply optional training-time transformations.

| Module | What it does | Related guide |
|---|---|---|
| [`hypernix.training.abbicus`](../src/hypernix/training/abbicus.py) | Adjusts training context/sequence length as a curriculum based on model size, step, and configured limits. | [Abbicus](Abbicus.md) |
| [`hypernix.training.monitor`](../src/hypernix/training/monitor.py) | Connects training runs to monitoring/telemetry and reports training progress. | [Dashboards](Dashboards.md) |
| [`hypernix.training.apron`](../src/hypernix/training/apron.py) | Snapshots and restores RNG states, optionally seeding Python, NumPy, and PyTorch for reproducible work. | [Apron](Apron.md) |
| [`hypernix.training.brewer`](../src/hypernix/training/brewer.py) | Defines the HyperNix0x-v2 Brewer architecture presets and model construction helpers. | [Architectures](Architectures.md) |
| [`hypernix.training.cake_pan`](../src/hypernix/training/cake_pan.py) | Guards training with non-finite checks, time/memory safeguards, and rollback support. | [CakePan](CakePan.md) |
| [`hypernix.training.camouflage`](../src/hypernix/training/camouflage.py) | Provides alignment/RLHF/RLAIF-style training scaffolding and optional evaluator integration. | [Camouflage](Camouflage.md) |
| [`hypernix.training.compute_framework`](../src/hypernix/training/compute_framework.py) | Abstracts supported compute devices and distributed training setup/gradient synchronization. | [Frameworks](Frameworks.md) |
| [`hypernix.training.deep_fryer`](../src/hypernix/training/deep_fryer.py) | Applies reversible model-weight perturbations for regularization or stress/negative examples. | [DeepFryer](DeepFryer.md) |
| [`hypernix.training.fizzle`](../src/hypernix/training/fizzle.py) | Fuses multiple Hugging Face models (for example, an ASR model and an LLM) into a unified architecture. | [Training](Training.md) |
| [`hypernix.training.instant_pot`](../src/hypernix/training/instant_pot.py) | Runs a configured end-to-end training recipe/pipeline. | [Kitchen](Kitchen.md) |
| [`hypernix.training.lazy_suzan`](../src/hypernix/training/lazy_suzan.py) | Coordinates supported multi-device/multi-GPU linking for distributed workloads. | [Frameworks](Frameworks.md) |
| [`hypernix.training.mtp`](../src/hypernix/training/mtp.py) | Implements Multi-Token Prediction heads/training support for predicting multiple future tokens. | [MTP](MTP.md) |
| [`hypernix.training.recipe_book`](../src/hypernix/training/recipe_book.py) | Stores named training/evaluation recipes and resolves them with per-run overrides. | [RecipeBook](RecipeBook.md) |
| [`hypernix.training.smoker`](../src/hypernix/training/smoker.py) | Provides tiered training-quality checks/assessment helpers. | [Smoker](Smoker.md) |
| [`hypernix.training.train`](../src/hypernix/training/train.py) | Exposes the core training entry points, including initialization, checkpoint expansion, and model training. | [Training](Training.md) |

## Additional package namespaces

These packages are part of the source tree but are not all entries in `MODULE_CATEGORIES`. Their modules are included here so the registry is not mistaken for the complete source inventory.

### Audio

The audio package reads files, processes signals, extracts features, and trains/runs wake-word detection. Its package documentation explicitly describes these boundaries.

| Module | What it does |
|---|---|
| [`hypernix.audio.audiofile`](../src/hypernix/audio/audiofile.py) | Reads audio input files and normalizes access to the incoming audio data. |
| [`hypernix.audio.features`](../src/hypernix/audio/features.py) | Converts audio into feature representations such as log-mel frames. |
| [`hypernix.audio.processor`](../src/hypernix/audio/processor.py) | Provides audio signal-processing operations such as resampling, filtering, level control, silence splitting, and pipeline composition. |
| [`hypernix.audio.wakeup`](../src/hypernix/audio/wakeup.py) | Implements wake-word training and streaming detection using the audio pipeline. |
| [`hypernix.audio.wakeup_cli`](../src/hypernix/audio/wakeup_cli.py) | Command-line interface for the wake-word workflow. |

### Bridge

A bridge connects to an inference server owned by another process. It does not manage that server's model lifecycle.

| Module | What it does |
|---|---|
| [`hypernix.bridge.lmstudio`](../src/hypernix/bridge/lmstudio.py) | Discovers and calls LM Studio's OpenAI-compatible local/LAN/Tailscale server. |

### C4 model analysis

C4 decomposes a model into analyzable parts, measures their contribution, and can assemble a derived model. It is a separate model-analysis workflow, not ordinary dataset preprocessing.

| Module | What it does |
|---|---|
| [`hypernix.c4.dial`](../src/hypernix/c4/dial.py) | Scores parameter groups/dials, identifies low-importance parts, and builds training-priority reports. |
| [`hypernix.c4.explode`](../src/hypernix/c4/explode.py) | Loads a model and decomposes its layers/weights into C4 analysis units. |
| [`hypernix.c4.model`](../src/hypernix/c4/model.py) | Defines/loads the model representation used by C4. |
| [`hypernix.c4.optim`](../src/hypernix/c4/optim.py) | Implements C4-specific optimization and comparison strategies. |
| [`hypernix.c4.probe`](../src/hypernix/c4/probe.py) | Runs calibration/evaluation inputs against a model to measure behavior. |
| [`hypernix.c4.reap`](../src/hypernix/c4/reap.py) | Selects and extracts chosen layers, weights, or channels from an exploded model. |
| [`hypernix.c4.restore`](../src/hypernix/c4/restore.py) | Restores snapshots and supports controlled temporary ablation/restore operations. |
| [`hypernix.c4.sow`](../src/hypernix/c4/sow.py) | Assembles selected components into a new model and validates/heals the result. |
| [`hypernix.c4.stem`](../src/hypernix/c4/stem.py) | Reports and tests layer/stem-level importance and behavior. |
| [`hypernix.c4.cli`](../src/hypernix/c4/cli.py) | Command-line entry point for C4 analysis operations. |

### Dilute

Dilute generates multiple candidate completions, scores them, retains stronger samples, and can stream accepted samples to a sink. The evaluators can be model-based or ordinary Python functions.

| Module | What it does |
|---|---|
| [`hypernix.dilute.backends`](../src/hypernix/dilute/backends.py) | Selects/adapts the model-generation backend used by the sampling workflow. |
| [`hypernix.dilute.cli`](../src/hypernix/dilute/cli.py) | Command-line entry point for Dilute. |
| [`hypernix.dilute.core`](../src/hypernix/dilute/core.py) | Implements candidate generation, temperature ladders, ranking/tie-breaking, and streaming distillation. |
| [`hypernix.dilute.evaluators`](../src/hypernix/dilute/evaluators.py) | Implements evaluator adapters, including model- and function-based scoring. |

### HyperLink

HyperLink is the phone/web client integration layer: pairing, sessions, attachments, model links, and communication with the T1 server. The HTTP routes themselves live under [`hypernix.t1api.routers`](../src/hypernix/t1api/routers.py).

| Module | What it does |
|---|---|
| [`hypernix.hyperlink.brewed`](../src/hypernix/hyperlink/brewed.py) | Finds native HyperNix model folders, fetches the configured/default native model when requested, and describes those models to the HyperLink catalogue. |
| [`hypernix.hyperlink.brewed_server`](../src/hypernix/hyperlink/brewed_server.py) | Serves a native HyperNix model through an OpenAI-compatible API, alongside the llama-server-compatible interface. |
| [`hypernix.hyperlink.capabilities`](../src/hypernix/hyperlink/capabilities.py) | Describes client/server capabilities used to negotiate supported behavior. |
| [`hypernix.hyperlink.catalogue`](../src/hypernix/hyperlink/catalogue.py) | Builds or serves the HyperLink model/catalogue view. |
| [`hypernix.hyperlink.codetools`](../src/hypernix/hyperlink/codetools.py) | Defines the sandbox-scoped code tools a model can use during a HyperLink code session. |
| [`hypernix.hyperlink.compaction`](../src/hypernix/hyperlink/compaction.py) | Compacts conversation/context content to fit model limits. |
| [`hypernix.hyperlink.datasets`](../src/hypernix/hyperlink/datasets.py) | Imports or exposes dataset context for chat/workflows. |
| [`hypernix.hyperlink.default_prompt`](../src/hypernix/hyperlink/default_prompt.py) | Defines the default HyperLink system prompt. |
| [`hypernix.hyperlink.defaultmodel`](../src/hypernix/hyperlink/defaultmodel.py) | Resolves the default model selection. |
| [`hypernix.hyperlink.discovery`](../src/hypernix/hyperlink/discovery.py) | Discovers and ranks local/network endpoints at which the server can be reached. |
| [`hypernix.hyperlink.files`](../src/hypernix/hyperlink/files.py) | Stores and retrieves content-addressed attachments such as images, documents, and code. |
| [`hypernix.hyperlink.generation`](../src/hypernix/hyperlink/generation.py) | Coordinates generation requests from HyperLink. |
| [`hypernix.hyperlink.handover`](../src/hypernix/hyperlink/handover.py) | Moves a model from LM Studio onto the HyperNix-managed runner, unloading before loading to avoid duplicate VRAM allocation and restoring LM Studio if the move fails. |
| [`hypernix.hyperlink.hfdownload`](../src/hypernix/hyperlink/hfdownload.py) | Downloads model files through the HyperLink Hugging Face workflow. |
| [`hypernix.hyperlink.hfmerge`](../src/hypernix/hyperlink/hfmerge.py) | Resolves Hugging Face model pages and direct links into a complete GGUF download plan. |
| [`hypernix.hyperlink.hyperchat`](../src/hypernix/hyperlink/hyperchat.py) | Implements the HyperLink chat orchestration layer. |
| [`hypernix.hyperlink.hypercompress`](../src/hypernix/hyperlink/hypercompress.py) | Provides the HyperLink compression/compaction integration. |
| [`hypernix.hyperlink.identity`](../src/hypernix/hyperlink/identity.py) | Represents and resolves the identity associated with a HyperLink client. |
| [`hypernix.hyperlink.imagecodec`](../src/hypernix/hyperlink/imagecodec.py) | Encodes/decodes image payloads for supported message transport. |
| [`hypernix.hyperlink.inference`](../src/hypernix/hyperlink/inference.py) | Connects HyperLink requests to the selected inference path. |
| [`hypernix.hyperlink.latest`](../src/hypernix/hyperlink/latest.py) | Resolves current/latest release information used by the client. |
| [`hypernix.hyperlink.managed`](../src/hypernix/hyperlink/managed.py) | Owns the llama.cpp server process so HyperLink can load, unload, switch, and place models without requiring LM Studio to manage them. |
| [`hypernix.hyperlink.memory`](../src/hypernix/hyperlink/memory.py) | Implements HyperLink memory storage/access behavior subject to configured privacy controls. |
| [`hypernix.hyperlink.modellinks`](../src/hypernix/hyperlink/modellinks.py) | Parses and resolves model links for the client. |
| [`hypernix.hyperlink.notify`](../src/hypernix/hyperlink/notify.py) | Handles notification delivery/events for HyperLink. |
| [`hypernix.hyperlink.offload`](../src/hypernix/hyperlink/offload.py) | Implements CPU offloading for llama.cpp and native HyperNix runners, including MoE expert/KV-cache placement and layer-fit calculations. |
| [`hypernix.hyperlink.ondevice`](../src/hypernix/hyperlink/ondevice.py) | Integrates on-device model execution. |
| [`hypernix.hyperlink.ownership`](../src/hypernix/hyperlink/ownership.py) | Enforces/represents ownership boundaries for user data and sessions. |
| [`hypernix.hyperlink.pairing`](../src/hypernix/hyperlink/pairing.py) | Turns short pairing codes into per-device credentials and device records. |
| [`hypernix.hyperlink.peers`](../src/hypernix/hyperlink/peers.py) | Tracks peer/server connections used by the client. |
| [`hypernix.hyperlink.preferences`](../src/hypernix/hyperlink/preferences.py) | Loads and persists HyperLink preferences. |
| [`hypernix.hyperlink.ring`](../src/hypernix/hyperlink/ring.py) | Reports context/usage ring state for the client UI. |
| [`hypernix.hyperlink.runner_history`](../src/hypernix/hyperlink/runner_history.py) | Stores or retrieves inference-run history. |
| [`hypernix.hyperlink.search`](../src/hypernix/hyperlink/search.py) | Implements the client-facing search integration. |
| [`hypernix.hyperlink.sessions`](../src/hypernix/hyperlink/sessions.py) | Stores server-side chat sessions and messages so conversations can continue across devices. |
| [`hypernix.hyperlink.shell`](../src/hypernix/hyperlink/shell.py) | Implements the HyperLink shell/tool interaction boundary. |
| [`hypernix.hyperlink.signing`](../src/hypernix/hyperlink/signing.py) | Signs and verifies requests/session-related payloads. |
| [`hypernix.hyperlink.sync`](../src/hypernix/hyperlink/sync.py) | Synchronizes supported client/server state. |
| [`hypernix.hyperlink.titles`](../src/hypernix/hyperlink/titles.py) | Creates or updates chat/session titles. |
| [`hypernix.hyperlink.toolloop`](../src/hypernix/hyperlink/toolloop.py) | Runs the HyperLink model/tool loop: dispatches requested tools, returns tool results to the model, and continues the conversation. |

### Noodle agents

Noodle is the multi-provider agent executor used by Hyped Pro. It normalizes provider calls, runs tools in a workspace, and verifies results. Tool execution and memory are subject to explicit configuration and approval.

| Module | What it does |
|---|---|
| [`hypernix.interfaces.noodle.agent`](../src/hypernix/interfaces/noodle/agent.py) | Runs an agent turn and records its events/results. |
| [`hypernix.interfaces.noodle.cli`](../src/hypernix/interfaces/noodle/cli.py) | Exposes Noodle operations through a command-line interface. |
| [`hypernix.interfaces.noodle.hpo`](../src/hypernix/interfaces/noodle/hpo.py) | Implements hyperparameter/search-space optimization helpers used by Noodle workflows. |
| [`hypernix.interfaces.noodle.hyped`](../src/hypernix/interfaces/noodle/hyped.py) | Connects Noodle sessions/events to Hyped Pro. |
| [`hypernix.interfaces.noodle.providers`](../src/hypernix/interfaces/noodle/providers.py) | Defines provider adapters and normalizes their chat/tool-call formats. |
| [`hypernix.interfaces.noodle.swarm`](../src/hypernix/interfaces/noodle/swarm.py) | Coordinates multiple agents/tasks into a swarm execution. |
| [`hypernix.interfaces.noodle.tools`](../src/hypernix/interfaces/noodle/tools.py) | Defines workspace-scoped tools and dispatches approved tool calls. |
| [`hypernix.interfaces.noodle.validate`](../src/hypernix/interfaces/noodle/validate.py) | Provides syntax/command/result verifiers for agent outputs. |

### Neuron

Neuron trains compact networks for action and perception tasks. It supports supervised learning, imitation learning, reinforcement learning, and execution-based evaluation.

| Module | What it does |
|---|---|
| [`hypernix.neuron.cli`](../src/hypernix/neuron/cli.py) | Command-line entry point for Neuron workflows. |
| [`hypernix.neuron.data`](../src/hypernix/neuron/data.py) | Prepares training/evaluation data for compact networks. |
| [`hypernix.neuron.envs`](../src/hypernix/neuron/envs.py) | Defines the environment protocol and deterministic test environments. |
| [`hypernix.neuron.evaluate`](../src/hypernix/neuron/evaluate.py) | Evaluates trained networks, including task execution/return metrics. |
| [`hypernix.neuron.imitation`](../src/hypernix/neuron/imitation.py) | Trains policies from recorded demonstrations. |
| [`hypernix.neuron.nets`](../src/hypernix/neuron/nets.py) | Defines the compact network architectures used by Neuron. |
| [`hypernix.neuron.rl`](../src/hypernix/neuron/rl.py) | Implements reinforcement-learning training. |
| [`hypernix.neuron.supervised`](../src/hypernix/neuron/supervised.py) | Implements supervised training for labeled examples. |

### Script generator

ScriptGen builds training scripts through a terminal wizard or GUI. Parameter definitions are shared between validation, widgets, and generated code.

| Module | What it does |
|---|---|
| [`hypernix.scriptgen.app`](../src/hypernix/scriptgen/app.py) | Holds form state, validation, preview, and GUI launch behavior. |
| [`hypernix.scriptgen.cli`](../src/hypernix/scriptgen/cli.py) | Implements the ScriptGen command-line entry point. |
| [`hypernix.scriptgen.highlight`](../src/hypernix/scriptgen/highlight.py) | Provides syntax highlighting for generated script previews. |
| [`hypernix.scriptgen.params`](../src/hypernix/scriptgen/params.py) | Defines parameter metadata, groups, defaults, validation, and search/suggestions. |
| [`hypernix.scriptgen.recipes`](../src/hypernix/scriptgen/recipes.py) | Provides reusable parameter recipes/presets. |
| [`hypernix.scriptgen.templates`](../src/hypernix/scriptgen/templates.py) | Generates complete scripts or injects configuration into existing scripts. |
| [`hypernix.scriptgen.theme`](../src/hypernix/scriptgen/theme.py) | Defines palettes/fonts and checks palette contrast. |
| [`hypernix.scriptgen.widgets`](../src/hypernix/scriptgen/widgets.py) | Implements custom form controls used by the GUI. |

### T1 API server

The T1 API is the optional HTTP server. Its core auth, registry, routing, storage, and policy modules are separated from FastAPI routers so core behavior can be tested without the HTTP extra. See [T1 API](T1-API.md) for endpoint contracts.

| Module | What it does |
|---|---|
| [`hypernix.t1api.accounts`](../src/hypernix/t1api/accounts.py) | Local account records and account lifecycle operations. |
| [`hypernix.t1api.accounts_cli`](../src/hypernix/t1api/accounts_cli.py) | Account-management CLI. |
| [`hypernix.t1api.anthropic`](../src/hypernix/t1api/anthropic.py) | Anthropic-compatible API/provider integration. |
| [`hypernix.t1api.app`](../src/hypernix/t1api/app.py) | Assembles the FastAPI application and middleware/routes. |
| [`hypernix.t1api.audit`](../src/hypernix/t1api/audit.py) | Durable audit events for security- and administration-sensitive actions. |
| [`hypernix.t1api.auth`](../src/hypernix/t1api/auth.py) | Authenticates requests and resolves caller credentials/scopes. |
| [`hypernix.t1api.authhistory`](../src/hypernix/t1api/authhistory.py) | Tracks authentication/session history. |
| [`hypernix.t1api.backup`](../src/hypernix/t1api/backup.py) | Backs up and restores supported server data. |
| [`hypernix.t1api.billing`](../src/hypernix/t1api/billing.py) | Tracks balances, transactions, and billing ledger state. |
| [`hypernix.t1api.billingkeys`](../src/hypernix/t1api/billingkeys.py) | Associates payment/billing bindings with API keys. |
| [`hypernix.t1api.bootstrap`](../src/hypernix/t1api/bootstrap.py) | Creates the restricted initial administrator/bootstrap credential. |
| [`hypernix.t1api.chat_cli`](../src/hypernix/t1api/chat_cli.py) | Command-line chat client for the T1 API. |
| [`hypernix.t1api.claudecode`](../src/hypernix/t1api/claudecode.py) | Integrates Claude Code with the T1 API's supported model/tool path. |
| [`hypernix.t1api.codebox`](../src/hypernix/t1api/codebox.py) | Provides the server-side code workspace/sandbox boundary. |
| [`hypernix.t1api.codegit`](../src/hypernix/t1api/codegit.py) | Provides Git operations for the code workspace. |
| [`hypernix.t1api.config`](../src/hypernix/t1api/config.py) | Defines and validates T1 server configuration. |
| [`hypernix.t1api.cost`](../src/hypernix/t1api/cost.py) | Estimates and reports model/request costs. |
| [`hypernix.t1api.counter`](../src/hypernix/t1api/counter.py) | Tracks usage counters and metering data. |
| [`hypernix.t1api.db`](../src/hypernix/t1api/db.py) | Defines database/backend access primitives. |
| [`hypernix.t1api.deploy`](../src/hypernix/t1api/deploy.py) | Coordinates deployment configuration and lifecycle. |
| [`hypernix.t1api.deps`](../src/hypernix/t1api/deps.py) | Checks optional server dependencies. |
| [`hypernix.t1api.disconnect`](../src/hypernix/t1api/disconnect.py) | Handles disconnect/cleanup behavior for server connections. |
| [`hypernix.t1api.envupgrade`](../src/hypernix/t1api/envupgrade.py) | Migrates supported environment/configuration values between versions. |
| [`hypernix.t1api.errors`](../src/hypernix/t1api/errors.py) | Defines T1 API error types and stable error codes. |
| [`hypernix.t1api.events`](../src/hypernix/t1api/events.py) | Publishes/polls server events and supports streaming event delivery. |
| [`hypernix.t1api.hyperlink_web`](../src/hypernix/t1api/hyperlink_web.py) | Serves the HyperLink web application from the T1 server. |
| [`hypernix.t1api.jobs`](../src/hypernix/t1api/jobs.py) | Queues and manages asynchronous jobs, status, and cancellation. |
| [`hypernix.t1api.keys`](../src/hypernix/t1api/keys.py) | Implements API-key directory and key-related operations. |
| [`hypernix.t1api.launchscript_cli`](../src/hypernix/t1api/launchscript_cli.py) | CLI for creating/running managed launch scripts. |
| [`hypernix.t1api.lmsoverride`](../src/hypernix/t1api/lmsoverride.py) | Applies configured LM Studio model/provider overrides. |
| [`hypernix.t1api.localserver`](../src/hypernix/t1api/localserver.py) | Manages the local server process/service integration. |
| [`hypernix.t1api.mcp`](../src/hypernix/t1api/mcp.py) | Implements the server-side MCP integration surface. |
| [`hypernix.t1api.mcpclient`](../src/hypernix/t1api/mcpclient.py) | Connects to configured external MCP servers. |
| [`hypernix.t1api.mcptools`](../src/hypernix/t1api/mcptools.py) | Exposes approved T1 operations as MCP tools. |
| [`hypernix.t1api.modelindex`](../src/hypernix/t1api/modelindex.py) | Indexes local GGUF models for server-side discovery. |
| [`hypernix.t1api.modelindex_cli`](../src/hypernix/t1api/modelindex_cli.py) | CLI for building/refreshing the model index. |
| [`hypernix.t1api.modelsync`](../src/hypernix/t1api/modelsync.py) | Synchronizes model files/metadata across configured locations. |
| [`hypernix.t1api.modelsync_cli`](../src/hypernix/t1api/modelsync_cli.py) | CLI for model synchronization. |
| [`hypernix.t1api.modules`](../src/hypernix/t1api/modules.py) | Stores/version-tracks registered modules and handles module transport metadata; stored modules are not automatically executed. |
| [`hypernix.t1api.mtls`](../src/hypernix/t1api/mtls.py) | Configures/verifies mutual TLS client certificates. |
| [`hypernix.t1api.netpolicy`](../src/hypernix/t1api/netpolicy.py) | Applies network allow/block and private-address policy. |
| [`hypernix.t1api.picker`](../src/hypernix/t1api/picker.py) | Provides interactive model/backend selection. |
| [`hypernix.t1api.pricing`](../src/hypernix/t1api/pricing.py) | Defines pricing rules and cost tables. |
| [`hypernix.t1api.privacy`](../src/hypernix/t1api/privacy.py) | Enforces server privacy and retention behavior. |
| [`hypernix.t1api.ratelimit`](../src/hypernix/t1api/ratelimit.py) | Applies request rate limits and limit accounting. |
| [`hypernix.t1api.registry`](../src/hypernix/t1api/registry.py) | Maintains the allowlisted model registry and model status. |
| [`hypernix.t1api.routing`](../src/hypernix/t1api/routing.py) | Selects a model/server using configured routing and quota policy. |
| [`hypernix.t1api.runner_cli`](../src/hypernix/t1api/runner_cli.py) | CLI for running/inspecting the T1 inference runner. |
| [`hypernix.t1api.schemas`](../src/hypernix/t1api/schemas.py) | Defines request/response schemas for HTTP endpoints. |
| [`hypernix.t1api.security`](../src/hypernix/t1api/security.py) | Implements shared security guards, including URL/path validation. |
| [`hypernix.t1api.servers`](../src/hypernix/t1api/servers.py) | Maintains remote server records, trust, and health/status. |
| [`hypernix.t1api.setup_cli`](../src/hypernix/t1api/setup_cli.py) | CLI for setting up/configuring the T1 server. |
| [`hypernix.t1api.storage`](../src/hypernix/t1api/storage.py) | Provides persistent storage abstractions and backend selection. |
| [`hypernix.t1api.training_cli`](../src/hypernix/t1api/training_cli.py) | CLI for inspecting/controlling server-managed training jobs. |
| [`hypernix.t1api.transport`](../src/hypernix/t1api/transport.py) | Defines transport behavior for inter-server/API communication. |
| [`hypernix.t1api.upgrade`](../src/hypernix/t1api/upgrade.py) | Implements supported T1 server upgrade/version-check operations. |
| [`hypernix.t1api.usage`](../src/hypernix/t1api/usage.py) | Records and reports per-key/per-model usage. |
| [`hypernix.t1api.version`](../src/hypernix/t1api/version.py) | Defines the T1 API's independent six-part protocol version. |
| [`hypernix.t1api.webauth`](../src/hypernix/t1api/webauth.py) | Implements browser-based authentication flows. |
| [`hypernix.t1api.webconfig_cli`](../src/hypernix/t1api/webconfig_cli.py) | CLI for web-facing T1 configuration. |
| [`hypernix.t1api.websearch`](../src/hypernix/t1api/websearch.py) | Implements the legacy web-search endpoint/integration. |
| [`hypernix.t1api.websearch_v2`](../src/hypernix/t1api/websearch_v2.py) | Implements the versioned web-search API and its policy-aware behavior. |

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
| [`hypernix.t1sdk.client`](../src/hypernix/t1sdk/client.py) | Implements the typed T1 API client methods. |
| [`hypernix.t1sdk.errors`](../src/hypernix/t1sdk/errors.py) | Defines client-side typed API/transport errors. |
| [`hypernix.t1sdk.models`](../src/hypernix/t1sdk/models.py) | Defines typed response/request data models. |
| [`hypernix.t1sdk.transport`](../src/hypernix/t1sdk/transport.py) | Implements HTTP transport, retries, and TLS configuration. |



### Additional implementation modules

The following source files are also part of the package. They are listed separately because they are not all exposed through `MODULE_CATEGORIES`. Internal implementation modules can change without constituting a public API change.

#### Interfaces and compatibility

| Module | What it does |
|---|---|
| [`hypernix.__main__`](../src/hypernix/__main__.py) | Makes `python -m hypernix` dispatch to the package CLI. |
| [`hypernix._compat._iter_legacy`](../src/hypernix/_compat/_iter_legacy.py) | Supplies the legacy-Python iterator compatibility implementation. |
| [`hypernix._compat._iter_py315`](../src/hypernix/_compat/_iter_py315.py) | Supplies the Python 3.15-specific iterator implementation behind the compatibility boundary. |
| [`hypernix.interfaces.dots`](../src/hypernix/interfaces/dots.py) | Implements the HyperNix dot/agent entry-point abstraction used by the basic assistant experience. |
| [`hypernix.interfaces.hyped_basic`](../src/hypernix/interfaces/hyped_basic.py) | Implements the basic Hyped terminal application. |
| [`hypernix.interfaces.list_all`](../src/hypernix/interfaces/list_all.py) | Lists discoverable modules/commands for the CLI and documentation surfaces. |

#### Model and runtime implementation

| Module | What it does |
|---|---|
| [`hypernix.models.brewer_gguf`](../src/hypernix/models/brewer_gguf.py) | Exports supported Brewer architecture checkpoints into GGUF form. |
| [`hypernix.models.ggufrun`](../src/hypernix/models/ggufrun.py) | Provides the GGUF model-runner integration used by HyperNix runtime commands. |
| [`hypernix.models.hflinks`](../src/hypernix/models/hflinks.py) | Parses/resolves Hugging Face model and file links. |
| [`hypernix.models.hnx0xv2image`](../src/hypernix/models/hnx0xv2image.py) | Implements the HyperNix0xV2 image/vision model integration. |
| [`hypernix.models.hnx0xv2image_cli`](../src/hypernix/models/hnx0xv2image_cli.py) | Command-line entry point for the HNX0xV2 image workflow. |
| [`hypernix.models.hnx0xv2image_gguf`](../src/hypernix/models/hnx0xv2image_gguf.py) | Handles GGUF-side packaging/export support for the HNX0xV2 image model. |
| [`hypernix.models.hnxdevice`](../src/hypernix/models/hnxdevice.py) | Describes/selects supported device execution paths for HyperNix models. |
| [`hypernix.models.hnxrun`](../src/hypernix/models/hnxrun.py) | Implements the HNX model execution path for supported model artifacts. |
| [`hypernix.models.hnxtokenizer`](../src/hypernix/models/hnxtokenizer.py) | Handles tokenizer metadata and tokenization for HyperNix-native model formats. |
| [`hypernix.models.hnxtorch`](../src/hypernix/models/hnxtorch.py) | Provides the PyTorch model/runtime integration for HyperNix-native models. |

#### Monitoring implementation

| Module | What it does |
|---|---|
| [`hypernix.monitoring.hostprobe`](../src/hypernix/monitoring/hostprobe.py) | Collects host CPU, memory, GPU, and system telemetry for dashboards. |
| [`hypernix.monitoring.livestream`](../src/hypernix/monitoring/livestream.py) | Streams live monitoring/training events to interactive clients. |
| [`hypernix.monitoring.remote_desktop`](../src/hypernix/monitoring/remote_desktop.py) | Implements the remote-desktop/RFB monitoring connection. |
| [`hypernix.monitoring.run_inspect`](../src/hypernix/monitoring/run_inspect.py) | Inspects a running training/inference process and its associated run metadata. |
| [`hypernix.monitoring.stale_log`](../src/hypernix/monitoring/stale_log.py) | Detects stale or inactive training logs and helps identify the process behind them. |
| [`hypernix.monitoring.tvtop_max`](../src/hypernix/monitoring/tvtop_max.py) | Implements the tvtop-max monitoring application and its run controls. |
| [`hypernix.monitoring.tvtop_max_bridge`](../src/hypernix/monitoring/tvtop_max_bridge.py) | Connects the tvtop-max UI to HyperNix Python/runtime services. |
| [`hypernix.monitoring.tvtoppro`](../src/hypernix/monitoring/tvtoppro.py) | Implements the tvtoppro terminal dashboard. |
| [`hypernix.monitoring.tvtoppro_modules`](../src/hypernix/monitoring/tvtoppro_modules.py) | Provides the modular stats/panel implementations consumed by tvtoppro. |

#### Optimizer and quantization implementation

| Module | What it does |
|---|---|
| [`hypernix.optimizers.sixbit`](../src/hypernix/optimizers/sixbit.py) | Implements six-bit optimizer-state/quantization support used by compatible optimizer paths. |
| [`hypernix.quant.auto`](../src/hypernix/quant/auto.py) | Coordinates automatic model conversion/quantization workflow selection. |
| [`hypernix.quant.convertq`](../src/hypernix/quant/convertq.py) | Connects conversion output to quantization stages. |
| [`hypernix.quant.dflash1`](../src/hypernix/quant/dflash1.py) | Builds DFlash1 draft-model artifacts. |
| [`hypernix.quant.dflash1_cli`](../src/hypernix/quant/dflash1_cli.py) | CLI for DFlash1 draft construction. |
| [`hypernix.quant.dflash2`](../src/hypernix/quant/dflash2.py) | Builds DFlash2 drafts embedded in a compatible base GGUF. |
| [`hypernix.quant.dflash2_cli`](../src/hypernix/quant/dflash2_cli.py) | CLI for DFlash2 draft construction. |
| [`hypernix.quant.forktypes`](../src/hypernix/quant/forktypes.py) | Defines/records quantization types supplied by supported llama.cpp forks. |
| [`hypernix.quant.formats`](../src/hypernix/quant/formats.py) | Defines the quantization-format metadata and format registry. |
| [`hypernix.quant.gguf`](../src/hypernix/quant/gguf.py) | Reads/writes or manipulates GGUF structures used by quantization tools. |
| [`hypernix.quant.ggufcheck`](../src/hypernix/quant/ggufcheck.py) | Validates GGUF metadata/tensor structure and reports malformed or unsupported artifacts. |
| [`hypernix.quant.hyprslug`](../src/hypernix/quant/hyprslug.py) | Implements the native HyprSlug quantization pipeline, including HyperNix-specific quant formats. |
| [`hypernix.quant.hyprslug_cli`](../src/hypernix/quant/hyprslug_cli.py) | CLI for HyprSlug conversion and quantization. |
| [`hypernix.quant.hyprslug_headers`](../src/hypernix/quant/hyprslug_headers.py) | Reads/writes self-describing HyperNix quantization headers and compatibility metadata. |
| [`hypernix.quant.hyprslug_headers_cli`](../src/hypernix/quant/hyprslug_headers_cli.py) | CLI for inspecting or producing HyprSlug headers. |
| [`hypernix.quant.hyprslug_server`](../src/hypernix/quant/hyprslug_server.py) | Serves the supported HyprSlug/runtime bridge over its configured server interface. |
| [`hypernix.quant.imatrix`](../src/hypernix/quant/imatrix.py) | Measures or reads importance matrices used by importance-aware quantization. |
| [`hypernix.quant.imatrix_cli`](../src/hypernix/quant/imatrix_cli.py) | CLI for importance-matrix generation and inspection. |
| [`hypernix.quant.llamaconvert`](../src/hypernix/quant/llamaconvert.py) | Implements llama-family checkpoint conversion helpers. |
| [`hypernix.quant.llamaquants`](../src/hypernix/quant/llamaquants.py) | Defines/normalizes llama.cpp quantization names and aliases. |
| [`hypernix.quant.lowbit`](../src/hypernix/quant/lowbit.py) | Defines low-bit and codebook quantization representations. |
| [`hypernix.quant.multiquant`](../src/hypernix/quant/multiquant.py) | Bundles multiple quantized variants into a shared model artifact where supported. |
| [`hypernix.quant.multiquant_cli`](../src/hypernix/quant/multiquant_cli.py) | CLI for multi-quant bundle creation and inspection. |
| [`hypernix.quant.qat`](../src/hypernix/quant/qat.py) | Implements quantization-aware training helpers. |
| [`hypernix.quant.runtime_bridge`](../src/hypernix/quant/runtime_bridge.py) | Connects HyperNix quantized artifacts to the compatible inference runtime. |
| [`hypernix.quant.runtime_bridge_cli`](../src/hypernix/quant/runtime_bridge_cli.py) | CLI for the quantized-runtime bridge. |
| [`hypernix.quant.steamroller`](../src/hypernix/quant/steamroller.py) | Implements the Steamroller llama.cpp quantizer integration. |
| [`hypernix.quant.steamroller_cli`](../src/hypernix/quant/steamroller_cli.py) | CLI for Steamroller quantization operations. |
| [`hypernix.quant.subbit`](../src/hypernix/quant/subbit.py) | Implements sub-bit quantization data structures and operations. |

#### Security implementation

| Module | What it does |
|---|---|
| [`hypernix.security.hfpin`](../src/hypernix/security/hfpin.py) | Handles Hugging Face credential/token input and safe local use. |
| [`hypernix.security.keyconfig`](../src/hypernix/security/keyconfig.py) | Loads and validates key-management configuration. |
| [`hypernix.security.keyversions`](../src/hypernix/security/keyversions.py) | Defines key-version formats and compatibility/conversion rules. |
| [`hypernix.security.rotorvault`](../src/hypernix/security/rotorvault.py) | Provides encrypted secret/key storage through Rotorvault. |
| [`hypernix.security.safeload`](../src/hypernix/security/safeload.py) | Provides guarded loading/validation for security-sensitive persisted data. |
| [`hypernix.security.safeurl`](../src/hypernix/security/safeurl.py) | Validates URLs before network access to reduce unsafe URL/SSRF behavior. |
| [`hypernix.security.t2c`](../src/hypernix/security/t2c.py) | Implements T2C credential/key handling. |
| [`hypernix.security.t2keys`](../src/hypernix/security/t2keys.py) | Implements T2 key formats, validation, and conversion support. |

#### System implementation

| Module | What it does |
|---|---|
| [`hypernix.system.archmap`](../src/hypernix/system/archmap.py) | Generates the package architecture map from source/module relationships. |
| [`hypernix.system.autoscan`](../src/hypernix/system/autoscan.py) | Implements scheduled repository/source scanning and its release checks. |
| [`hypernix.system.blanking`](../src/hypernix/system/blanking.py) | Provides platform-specific screen-off/screen-on controls across supported session types. |
| [`hypernix.system.deprecation`](../src/hypernix/system/deprecation.py) | Centralizes compatibility/deprecation helpers and warnings. |
| [`hypernix.system.fusebox_cli`](../src/hypernix/system/fusebox_cli.py) | CLI for the GPU thermal/resource controller. |
| [`hypernix.system.hardware`](../src/hypernix/system/hardware.py) | Detects and describes host CPU/GPU and hardware capabilities. |
| [`hypernix.system.launcher`](../src/hypernix/system/launcher.py) | Resolves and launches the selected HyperNix executable/runtime. |
| [`hypernix.system.linkwalk`](../src/hypernix/system/linkwalk.py) | Walks Markdown/documentation links for repository link validation. |
| [`hypernix.system.netops`](../src/hypernix/system/netops.py) | Provides network operations used by server/client connection workflows. |
| [`hypernix.system.pascal`](../src/hypernix/system/pascal.py) | Contains Pascal/sm_61 compatibility detection and helper behavior. |
| [`hypernix.system.pids`](../src/hypernix/system/pids.py) | Finds and inspects process IDs associated with HyperNix runs. |
| [`hypernix.system.torch_shim`](../src/hypernix/system/torch_shim.py) | Provides PyTorch compatibility shims for supported runtime combinations. |

#### T1 API router modules

These files are listed with their full import paths for completeness. Each owns the corresponding HTTP route group; it delegates policy/business logic to the T1 API core modules described above.

| Module | What it does |
|---|---|
| [`hypernix.t1api.routers.accounts`](../src/hypernix/t1api/routers/accounts.py) | Account endpoints. |
| [`hypernix.t1api.routers.anthropic`](../src/hypernix/t1api/routers/anthropic.py) | Anthropic-compatible messages endpoint. |
| [`hypernix.t1api.routers.audit`](../src/hypernix/t1api/routers/audit.py) | Audit-log endpoints. |
| [`hypernix.t1api.routers.auth`](../src/hypernix/t1api/routers/auth.py) | Authentication and sign-in endpoints. |
| [`hypernix.t1api.routers.authundo`](../src/hypernix/t1api/routers/authundo.py) | Authentication history/undo operations. |
| [`hypernix.t1api.routers.backup`](../src/hypernix/t1api/routers/backup.py) | Backup/restore endpoints. |
| [`hypernix.t1api.routers.billing`](../src/hypernix/t1api/routers/billing.py) | Billing and payment-binding endpoints. |
| [`hypernix.t1api.routers.bridge`](../src/hypernix/t1api/routers/bridge.py) | LM Studio/bridge endpoints. |
| [`hypernix.t1api.routers.code`](../src/hypernix/t1api/routers/code.py) | Code workspace and tool endpoints. |
| [`hypernix.t1api.routers.compact`](../src/hypernix/t1api/routers/compact.py) | Context compaction endpoint. |
| [`hypernix.t1api.routers.config`](../src/hypernix/t1api/routers/config.py) | Server configuration endpoints. |
| [`hypernix.t1api.routers.counter`](../src/hypernix/t1api/routers/counter.py) | Usage-counter endpoints. |
| [`hypernix.t1api.routers.events`](../src/hypernix/t1api/routers/events.py) | Event polling/stream endpoints. |
| [`hypernix.t1api.routers.health`](../src/hypernix/t1api/routers/health.py) | Health and status endpoints. |
| [`hypernix.t1api.routers.hypercompress`](../src/hypernix/t1api/routers/hypercompress.py) | HyperCompress endpoint. |
| [`hypernix.t1api.routers.hyperlink`](../src/hypernix/t1api/routers/hyperlink.py) | HyperLink pairing, session, file, and client endpoints. |
| [`hypernix.t1api.routers.inference`](../src/hypernix/t1api/routers/inference.py) | Inference/chat endpoints. |
| [`hypernix.t1api.routers.jobs`](../src/hypernix/t1api/routers/jobs.py) | Async job endpoints. |
| [`hypernix.t1api.routers.keys`](../src/hypernix/t1api/routers/keys.py) | API-key endpoints. |
| [`hypernix.t1api.routers.mcp`](../src/hypernix/t1api/routers/mcp.py) | MCP endpoints. |
| [`hypernix.t1api.routers.memory`](../src/hypernix/t1api/routers/memory.py) | Memory endpoints. |
| [`hypernix.t1api.routers.models`](../src/hypernix/t1api/routers/models.py) | Model discovery and selection endpoints. |
| [`hypernix.t1api.routers.modules`](../src/hypernix/t1api/routers/modules.py) | Module registry/transport endpoints. |
| [`hypernix.t1api.routers.net`](../src/hypernix/t1api/routers/net.py) | Network and connection endpoints. |
| [`hypernix.t1api.routers.noodle`](../src/hypernix/t1api/routers/noodle.py) | Noodle agent endpoints. |
| [`hypernix.t1api.routers.privacy`](../src/hypernix/t1api/routers/privacy.py) | Privacy and retention endpoints. |
| [`hypernix.t1api.routers.runner`](../src/hypernix/t1api/routers/runner.py) | Runner lifecycle and model execution endpoints. |
| [`hypernix.t1api.routers.security`](../src/hypernix/t1api/routers/security.py) | Security policy/diagnostic endpoints. |
| [`hypernix.t1api.routers.servers`](../src/hypernix/t1api/routers/servers.py) | Remote server registry endpoints. |
| [`hypernix.t1api.routers.t2c`](../src/hypernix/t1api/routers/t2c.py) | T2C key-related endpoints. |
| [`hypernix.t1api.routers.training`](../src/hypernix/t1api/routers/training.py) | Training job endpoints. |
| [`hypernix.t1api.routers.usage`](../src/hypernix/t1api/routers/usage.py) | Usage and cost endpoints. |
| [`hypernix.t1api.routers.web`](../src/hypernix/t1api/routers/web.py) | Web application/static web endpoints. |

### Waiter client

Waiter is the official T1 API TUI/CLI client and delegates its API contract to the T1 SDK.

| Module | What it does |
|---|---|
| [`hypernix.waiter.cli`](../src/hypernix/waiter/cli.py) | Parses and dispatches Waiter commands. |
| [`hypernix.waiter.client`](../src/hypernix/waiter/client.py) | Compatibility client layer over the T1 SDK. |
| [`hypernix.waiter.diagnose`](../src/hypernix/waiter/diagnose.py) | Diagnoses server connectivity and configuration. |
| [`hypernix.waiter.discovery`](../src/hypernix/waiter/discovery.py) | Discovers configured/available T1 servers. |
| [`hypernix.waiter.kits`](../src/hypernix/waiter/kits.py) | Provides reusable command/workflow helpers. |
| [`hypernix.waiter.local_config`](../src/hypernix/waiter/local_config.py) | Loads and stores local Waiter configuration. |
| [`hypernix.waiter.servargs`](../src/hypernix/waiter/servargs.py) | Parses/normalizes server-operation arguments. |
| [`hypernix.waiter.servops`](../src/hypernix/waiter/servops.py) | Performs server lifecycle/administration operations. |
| [`hypernix.waiter.smoke`](../src/hypernix/waiter/smoke.py) | Runs API smoke checks against a server. |
| [`hypernix.waiter.tui`](../src/hypernix/waiter/tui.py) | Implements the curses terminal interface. |


## Package initializers

The package initializers are also Python modules. They define package boundaries, lazy imports, public exports, and package-level documentation. Listed here separately from the 399 non-initializer implementation modules above.

| Initializer | Package responsibility |
|---|---|
| [`hypernix.__init__`](../src/hypernix/__init__.py) | Defines the top-level lazy public API, module/category registry, compatibility aliases, and exported names. |
| [`hypernix._compat.__init__`](../src/hypernix/_compat/__init__.py) | Holds private compatibility helpers for Python-version-specific behavior. |
| [`hypernix._native_flags.__init__`](../src/hypernix/_native_flags/__init__.py) | Defines native build/runtime feature flags used by compiled components. |
| [`hypernix._profiling.__init__`](../src/hypernix/_profiling/__init__.py) | Provides the internal profiling abstraction and version/backend selection. |
| [`hypernix.audio.__init__`](../src/hypernix/audio/__init__.py) | Lazy package boundary for audio file, feature, processing, and wake-word modules. |
| [`hypernix.bridge.__init__`](../src/hypernix/bridge/__init__.py) | Exports client bridges to inference servers owned by external processes. |
| [`hypernix.c4.__init__`](../src/hypernix/c4/__init__.py) | Exposes the C4 model decomposition, probing, scoring, selection, and rebuild workflow. |
| [`hypernix.chat.__init__`](../src/hypernix/chat/__init__.py) | Groups chat templates, session state, prompt injection, personas, and output processing. |
| [`hypernix.data.__init__`](../src/hypernix/data/__init__.py) | Groups dataset discovery, preparation, augmentation, splitting, and packaging. |
| [`hypernix.dilute.__init__`](../src/hypernix/dilute/__init__.py) | Exposes best-of-n sampling, evaluation, selection, and streaming self-distillation. |
| [`hypernix.elements.__init__`](../src/hypernix/elements/__init__.py) | Exposes the add-on/element framework and built-in elements. |
| [`hypernix.evaluation.__init__`](../src/hypernix/evaluation/__init__.py) | Groups evaluation batteries, rubrics, judge helpers, plots, and verification. |
| [`hypernix.hyperlink.__init__`](../src/hypernix/hyperlink/__init__.py) | Exposes HyperLink pairing, endpoint discovery, sessions, attachments, and Hugging Face link resolution. |
| [`hypernix.interfaces.__init__`](../src/hypernix/interfaces/__init__.py) | Defines the package boundary for command-line, terminal, desktop, and assistant interfaces. |
| [`hypernix.interfaces.noodle.__init__`](../src/hypernix/interfaces/noodle/__init__.py) | Exports Noodle agents, swarms, provider adapters, tools, and validators. |
| [`hypernix.models.__init__`](../src/hypernix/models/__init__.py) | Groups model architectures, loading, generation, and runtime helpers. |
| [`hypernix.monitoring.__init__`](../src/hypernix/monitoring/__init__.py) | Groups host probes, training logs, dashboards, and run inspection. |
| [`hypernix.neuron.__init__`](../src/hypernix/neuron/__init__.py) | Exposes compact-network training for supervised, imitation, and reinforcement-learning tasks. |
| [`hypernix.optimizers.__init__`](../src/hypernix/optimizers/__init__.py) | Groups optimizer generations and shared optimizer framework support. |
| [`hypernix.quant.__init__`](../src/hypernix/quant/__init__.py) | Groups conversion, GGUF formats, quantization, validation, and artifact upload. |
| [`hypernix.runtime.__init__`](../src/hypernix/runtime/__init__.py) | Groups inference kernels, context relay, T1 tools, and tool-call parsing. |
| [`hypernix.scriptgen.__init__`](../src/hypernix/scriptgen/__init__.py) | Exposes the training-script builder's form, parameter, recipe, template, and theme APIs. |
| [`hypernix.security.__init__`](../src/hypernix/security/__init__.py) | Groups credentials, key lifecycle, secret storage, and safe network/data loading. |
| [`hypernix.system.__init__`](../src/hypernix/system/__init__.py) | Groups environment, hardware, diagnostics, resource management, and housekeeping helpers. |
| [`hypernix.t1api.__init__`](../src/hypernix/t1api/__init__.py) | Exposes the T1 API's protocol version and lazy core/server interfaces. |
| [`hypernix.t1api.routers.__init__`](../src/hypernix/t1api/routers/__init__.py) | Defines the package boundary for the HTTP endpoint router modules. |
| [`hypernix.t1sdk.__init__`](../src/hypernix/t1sdk/__init__.py) | Exports the standard-library-only T1 API client, transport, errors, and typed models. |
| [`hypernix.timing.__init__`](../src/hypernix/timing/__init__.py) | Groups timers, alarms, scheduled runs, streaming callbacks, and progress animation. |
| [`hypernix.training.__init__`](../src/hypernix/training/__init__.py) | Groups training entry points, architecture presets, optimizers' integrations, and training-time controls. |
| [`hypernix.waiter.__init__`](../src/hypernix/waiter/__init__.py) | Defines Waiter's shared T1 API version and package boundary for the official TUI/CLI client. |

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

- Treat [`hypernix.MODULE_CATEGORIES`](../src/hypernix/MODULE_CATEGORIES.py) as the authoritative registry for the public categorized module surface.
- Treat the actual `src/hypernix/` tree as the inventory for additional packages and implementation modules.
- When a module moves or is renamed, update this page and the corresponding source/API references in the same change.
- For each new module, add its path, purpose, public/internal status, and a link to a detailed guide or source.
- Do not describe planned modules as implemented. Mark experimental/beta behavior and platform/provider limitations explicitly.
- Run the module-layout tests and docs/link checks after edits. Do not claim a module's behavior based solely on its filename.
