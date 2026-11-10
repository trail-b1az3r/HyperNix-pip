# HyperNix 0xv3 — Custom Architecture Design and Implementation Prompt

## Objective

Design and implement **HyperNix 0xv3**, a genuinely new, independently specified neural-network architecture that evolves beyond 0xv2. This is an architecture project, not a rename, preset refresh, or superficial wrapper around an existing Transformer. Preserve 0xv2 as a separately loadable legacy architecture; do not silently change its behavior or reinterpret old checkpoints as 0xv3.

Required architectural directions:
1. Custom attention heads with per-head configuration and heterogeneous behavior.
2. More layers, with depth selected as part of a flexible model configuration rather than a handful of hard-coded sizes.
3. A new, custom secondary KV-head system distinct from ordinary GQA/MQA.
4. Non-static model sizes: architecture dimensions and depth must be configurable across a continuous, validated design space, not limited to named presets.
5. A custom gated DeltaNet-inspired component that provides a recurrent/linear-attention path alongside the attention path.

Do not assume combining these ideas automatically improves a language model. Define the mathematics, implement a minimal correct version, and prove its value with controlled experiments and ablations before making performance or quality claims.

## 1. Architecture specification before implementation

Before coding, write an architecture proposal containing:
- A block-level diagram and exact tensor shapes.
- Forward equations for normalization, attention, secondary KV processing, gated delta updates, feed-forward layers, residual paths, and output projection.
- Parameter-count and activation-memory formulas.
- KV-cache and recurrent-state memory formulas.
- Causal masking and sequence/chunk-boundary semantics.
- Initialization, normalization, stability constraints, and numerical precision behavior.
- Training, incremental decoding, batching, padding, and cache-reset behavior.
- A compatibility table for CPU, CUDA, and other backends actually supported by the project.

Define every new operation precisely. Do not use names such as “custom attention” or “DeltaNet” as substitutes for a mathematical specification. If a proposed component is too unstable or expensive, revise it before integrating it into the full model.

## 2. Custom heterogeneous attention heads

Replace the assumption that every attention head must behave identically with a configurable head layout. Support per-layer and per-head metadata such as:
- Head dimension and head type/pattern.
- Local-window, global, dilated, or other explicitly implemented causal attention modes.
- Optional head-specific positional treatment where justified.
- Head grouping and routing metadata.
- Independent enablement of supported head types through model configuration.

Head layouts must be validated for compatible dimensions, divisibility, tensor layouts, and backend kernels. Do not create arbitrary per-head Python loops in the hot path if grouped/batched tensor operations can express the same computation.

The default configuration must remain practical for small models and older GPUs. Avoid multiplying projections or KV storage unnecessarily. Provide a simple baseline head layout and experimental layouts behind explicit configuration flags.

Tests must verify causality, masking, padding, variable sequence lengths, head isolation, gradients, and equivalence between optimized and reference implementations.

## 3. New secondary KV-head system: dual-tier KV memory

Design a HyperNix-specific **secondary KV-head (SKV) system** in addition to ordinary primary KV heads. It must not merely be a second name for GQA/MQA or duplicate the same K/V tensors.

Use a two-tier design as the initial research direction:

- **Primary KV heads:** retain high-fidelity keys and values needed for recent/local token interactions.
- **Secondary KV heads:** maintain a smaller, separately projected and learned representation of selected historical information. This may use learned compression, state/summary updates, or another mathematically specified mechanism.
- **Read path:** attention queries can retrieve from the primary tier and selectively consult the secondary tier. Define how retrieval is combined, normalized, and kept causal.
- **Write/update path:** define when secondary state is created or updated, how information is merged, and how state is bounded.
- **Cache policy:** define eviction, chunk transitions, sequence resets, batching, beam/parallel decoding behavior, and memory bounds.

The secondary tier must have its own configurable head count, head dimension, update/compression policy, and per-layer placement. It must be possible to disable it for ablations and small-device profiles.

Critical correctness constraints:
- No future-token leakage through compression, summaries, routing, or state updates.
- No cross-sample or cross-request state contamination.
- Padding tokens must not pollute persistent state.
- Incremental token decoding must match full-sequence causal execution within documented numerical tolerances.
- Cache growth must be bounded and predictable.
- Cache serialization/restoration must preserve generation behavior.
- Do not claim long-context retention merely because the secondary cache is large; evaluate retrieval accuracy at multiple distances.

Compare SKV against standard MHA, GQA, and MQA under equal parameter and memory budgets. Report quality, latency, throughput, and peak memory.

## 4. Custom gated DeltaNet-inspired path

Develop a **HyperNix Gated Delta Path (HGDP)**, inspired by gated delta-rule/recurrent linear-attention ideas, as a first-class optional sequence-mixing component.

The design must specify:
- The recurrent state representation and exact update equations.
- How key, value, query, and gate projections are produced.
- The delta-rule correction/update and its normalization or stability constraints.
- How gates control retention, update strength, and output contribution.
- How state is initialized, reset, detached or carried between chunks, and managed per batch item.
- How the recurrent path is combined with standard attention and the feed-forward path.
- Numerical behavior for long sequences, mixed precision, and gradient backpropagation.

The HGDP must support causal full-sequence training and efficient incremental inference. Where possible, implement a parallel/chunkwise training formulation mathematically equivalent to the recurrent reference formulation. Keep a simple, slow reference implementation for correctness tests.

Do not claim this is a novel DeltaNet algorithm merely because it is renamed. Clearly document which equations are established gated-delta-rule concepts and which modifications are HyperNix-specific. Respect upstream licenses and attribution if adapting code or equations from an implementation.

Compare at least these block variants:
1. Standard attention-only baseline.
2. Attention plus HGDP.
3. Attention plus SKV.
4. Attention plus both SKV and HGDP.
5. HGDP-heavy hybrid with attention at selected layers.

The model configuration must allow placement and ratio to vary by layer. Do not force every layer to use every mechanism.

## 5. Deeper architecture and non-static model sizing

Remove dependence on a few fixed model-size presets as the architecture's only valid configurations. Provide a validated configuration space where users can select, subject to hardware and implementation constraints:
- Number of layers/depth.
- Model/hidden width.
- Per-layer attention head counts and head dimensions.
- Primary and secondary KV-head counts and dimensions.
- Feed-forward expansion dimensions.
- HGDP state dimensions and layer placement.
- Vocabulary/output dimensions and supported positional/context settings.
- Optional per-layer width or block-type variation, if tensor and residual interfaces are explicitly defined.

Support deeper networks than 0xv2's shallow small-model configurations, but do not hard-code an arbitrary minimum or claim that more layers always improve quality. Provide sensible recommendations and resource estimates for small, medium, and large budgets while keeping the underlying configuration freely composable.

Use configuration validation to reject impossible or unsafe combinations early. Calculate parameter counts from the actual instantiated modules, and compare estimates with measured counts. Add a model-shape report that includes per-layer dimensions, parameter totals, attention/SKV/HGDP breakdowns, expected KV/recurrent-state memory, and estimated training memory.

Avoid shape assumptions hidden in model code. Save the complete architecture configuration and version it in checkpoints. Loading must fail with an actionable incompatibility message rather than silently reshaping or dropping tensors.

## 6. Block composition and residual design

Define a consistent block interface that allows different layers to use different combinations of:
- Heterogeneous attention.
- Primary and secondary KV processing.
- HGDP recurrent/linear sequence mixing.
- Gated feed-forward networks.
- Normalization and residual paths.

Evaluate pre-norm and other justified normalization arrangements; select one stable default based on experiments. Keep residual dimensions consistent or define explicit, tested projections for width changes. Ensure initialization and residual scaling remain stable as depth increases.

Do not add components to every layer by default solely to advertise features. Provide a compact default architecture and documented research configurations.

## 7. Training and optimization requirements

Integrate with existing HyperNix training interfaces without changing 0xv2 semantics. Ensure:
- Correct next-token prediction, shifting, labels, padding, and causal masks.
- Stable loss and gradients across variable depths and widths.
- Gradient checkpointing and accumulation where supported.
- Mixed precision with safe accumulation/state precision for HGDP and SKV.
- Optimizer compatibility based on actual tests, not registry labels.
- Save/resume including optimizer, scheduler, RNG, architecture config, and KV/recurrent state only where appropriate.
- Deterministic small-fixture tests and reproducible experiment configuration.

Do not introduce custom optimizer behavior as a hidden dependency of the architecture. Architecture correctness must be demonstrable with a standard supported optimizer first.

## 8. Inference, cache, and backend requirements

Implement a clear reference PyTorch path first. Then optimize only measured bottlenecks.

Support:
- Full-sequence causal forward.
- Incremental one-token decoding.
- Chunked prefill and chunked recurrent updates.
- Multiple independent sequences and batched requests.
- Cache reset, reorder, clone, serialize, and restore.
- Variable-length inputs and padding.
- CPU execution and CUDA execution where the installed stack supports the required operations.

Expose the selected attention, SKV, and HGDP execution paths in diagnostics. Unsupported kernels must use a correct portable fallback. Never silently move a CUDA workload to CPU or claim backend support based only on device-name detection.

Do not promise GGUF or llama.cpp compatibility unless the format and runtime can represent and execute every required operation. If those runtimes cannot support 0xv3, document that limitation and provide a supported native inference route rather than exporting a misleading or lossy model.

## 9. Checkpoint, conversion, and version compatibility

Give 0xv3 its own architecture identifier and config schema version. Preserve 0xv2 loading and generation behavior.
- Store all non-default head layouts, SKV settings, HGDP settings, layer compositions, and dimensions in the checkpoint config.
- Validate tensor names, shapes, dtypes, and required state.
- Make conversion explicit and loss-aware; never label a converted 0xv2 checkpoint as a native 0xv3 model.
- Add round-trip tests for save/load and any supported export format.
- Reject unsupported export formats with a clear explanation.

## 10. Research and evaluation plan

Before selecting defaults, train small controlled models on reproducible datasets and compare against a parameter-matched 0xv2 baseline. Keep tokenizer, data order, training tokens, optimizer, schedule, and evaluation protocol constant wherever possible.

Run ablations that independently remove or vary:
- Custom heterogeneous heads.
- Secondary KV heads.
- HGDP.
- Layer depth and width scaling.
- Layer placement and hybrid ratios.

Measure:
- Training loss and validation loss/perplexity.
- Held-out language modeling quality.
- Long-context retrieval at several distances and distractor densities.
- In-context copying and induction-style tasks.
- Token throughput, prefill latency, decode latency, and time to first token.
- Peak VRAM/RAM, cache size, and recurrent-state memory.
- Training stability, gradient norms, and numerical failures.

Use multiple seeds where feasible. Publish exact configs, dataset revisions, code commit, hardware, precision, and raw results. Compare both parameter-matched and memory/compute-matched baselines. Do not select a more expensive architecture based only on a single benchmark or a lower training loss.

## 11. Tests and definition of done

Required tests include:
- Mathematical/reference tests for attention, SKV updates/retrieval, HGDP recurrence, and gates.
- Causality and no-future-leakage tests for every path.
- Full-sequence versus incremental decoding equivalence.
- Full-sequence versus chunked HGDP equivalence.
- Batch isolation, padding, sequence reset, cache reorder, and cache restore.
- Gradient checks and finite-value tests across supported dtypes.
- Variable model dimensions and heterogeneous per-layer configurations.
- Parameter-count and memory-estimate validation.
- Save/load and configuration migration tests.
- CPU tests in ordinary CI; CUDA tests on compatible runners, with explicit skips when hardware is unavailable.
- Regression tests proving 0xv2 remains unchanged.

The architecture is not complete when classes merely instantiate or a training loop runs. It is complete only when the equations, reference implementation, tests, checkpoint format, supported inference path, and controlled ablation evidence agree.

## 12. Deliverables

1. docs/architecture/0xv3.md containing the mathematical specification and diagrams.
2. A versioned 0xv3 config schema and examples spanning small through large configurations.
3. The reference PyTorch implementation and correctness tests.
4. Optimized kernels only where benchmarked and numerically validated.
5. A model-shape/resource report.
6. Training and inference examples, including incremental decoding.
7. A comparison report with 0xv2 and component ablations.
8. Updated architecture registry and documentation with honest per-operation support statuses.

## Non-negotiable constraints

Do not edit, break, or silently migrate 0xv2. Do not invent benchmark results, claim novelty without a literature review, or claim an operation is supported when only its configuration can be parsed. Prefer a mathematically clear, testable design over a collection of fashionable components. If an idea fails correctness, stability, quality, or efficiency tests, document the failure and revise or remove it rather than shipping it as a feature.
