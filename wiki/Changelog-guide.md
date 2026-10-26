# hyperNix Changelog Guide

### Guide version: v2.2.1
> Whenever this guide changes, update the guide version.

This document defines the format and editorial rules for the canonical hyperNix release history. It records published releases, meaningful beta/dev work, features, fixes, API changes, site changes, removals, known issues, and temporary workarounds.

The changelog is a release history, not a Git log. It should help a future maintainer answer: **what changed, when, why, whether it shipped, what it affects, how to migrate, and what remains unresolved**.

---

## Changelog principles

- Give every PyPI-published version its own heading.
- Use the date the version was released to PyPI, formatted as `YYYY-MM-DD`.
- Group work between releases under the next release; do not list every commit.
- Record beta and dev versions when they contain meaningful experimental work.
- When a stable release follows beta/dev versions, include a **Beta / Dev → Release Summary**.
- Do not silently discard experimental work. Record work that was removed, reverted, or changed before release when that history is useful.
- Patch releases normally contain fixes, regressions, compatibility corrections, and UX papercuts. Minor releases normally contain features, APIs, integrations, or substantial improvements. Major releases may contain breaking changes or migrations.
- Keep unresolved problems documented until they are fixed, and preserve important historical context after they are fixed.
- Correct an inaccurate old entry explicitly; do not silently rewrite history.
- Do not claim benchmarks, security fixes, compatibility, or user impact unless they were verified.
- A known issue must be an actual unresolved bug or limitation in behavior. A prerequisite (for example, needing Bun) is not a known issue, and an incomplete feature should be documented as experimental or unfinished instead.
- Do not label a statement such as “only Chromium was tested” as a known issue. That is a caveat.
- For every major feature, include an expandable dropdown with a detailed list of its changes. This does not replace the per-release in-development bug-fix dropdown described below.

Example of a major-feature dropdown:

```markdown
<details>
<summary>HyperNix0xV4 architecture — expanded change list</summary>

- Added the 1B and 3B presets.
- Added rotary state mixing and architecture-specific checkpoint metadata.
- Added initialization, tensor-shape, and checkpoint-loading tests.
- Documented checkpoint compatibility and migration requirements.

</details>
```

---

## Entry formatting and item order

Use one change entry per logical change. Put the severity symbol first, followed by **exactly one space** and a clear sentence:

```markdown
✨ Added streaming generation to the inference API.
```

### Spacing and grouping rules

The symbols communicate hierarchy. Format related changes as follows:

1. A **major** feature, fix, removal, or API change gets its own line.
2. A **regular** feature or related change follows after **one blank line**.
3. A **minor** change must be placed immediately below, or directly beside, the most related major or regular item. Do not put minor items in an unrelated section or at the end of the release.
4. Use one blank line between separate change groups. Do not insert blank lines between a parent item and its directly related minor item.
5. Keep each bullet to one logical change. Use indented continuation lines for explanation, migration notes, or evidence.
6. Use nested bullets only for details of the item immediately above them.
7. A **patch** release must follow Python's `.postN` version format.

Example:

```markdown
### Added

๋࣭⭑ Added the HyperNix0xV4 architecture for 1B and 3B models.
  - Added rotary state mixing and architecture-specific checkpoint metadata.
𖥔 Added a configuration helper for selecting the V4 preset.

✨ Added streaming tokens to `generate()`.
  - The iterator yields decoded text without changing non-streaming behavior.
🛡️ Improved the error shown when a stream is consumed twice.

🛈 Tested on Linux only; Windows and macOS have not yet been tested.

```

Here, the major architecture item is separated from the next feature by one blank line. The minor configuration addition is kept directly with the architecture it belongs to, and the UX fix is kept directly with streaming.

Do not write this:

```markdown
๋࣭⭑ Added a new architecture.

𖥔 Fixed a streaming error.

𖥔 Added a configuration helper for the architecture.
```

The configuration helper is in the wrong group, and a bug is incorrectly marked as a feature.

### Recommended sentence style

Start with a past-tense action verb such as **Added**, **Changed**, **Fixed**, **Removed**, **Deprecated**, **Documented**, **Restored**, or **Rewrote**. Name the affected API, module, page, or behavior, then explain why the change matters when it is not self-evident.

Prefer:

```markdown
𖢥 Fixed Brewer attention leaking future tokens during training.
  - [Related bug or condition that had to be corrected]
  - [Another directly related correction, if applicable]
  - [Add a notice only when a caveat or scope clarification is useful]
```

over:

```markdown
🐛 Fixed attention bug.
```

---

## Legend

| Symbol | Use for |
| --- | --- |
| `๋࣭⭑` | Major new feature or architecture |
| `✨` | Regular user-facing or developer-facing feature |
| `𖥔` | Minor feature or enhancement; keep it beside its related feature |
| `𖢥` | Major bug affecting correctness, output, data, APIs, security, or stability |
| `🐛` | Minor bug, edge case, regression, or compatibility fix |
| `🛡️` | UX, error-message, warning, or safety polish |
| `🔁` | Refactor or integration improvement |
| `🔧` | Internal plumbing, build, dependency, or maintenance work |
| `⚡` | Performance or resource-use improvement |
| `🔒` | Security improvement or security-relevant correction |
| `⚠️` | Error-code creation, editing, or behavior change |
| `🧪` | Test or test-coverage change |
| `📚` | Documentation change |
| `🛜` | Website, wiki, or hosted-site change |
| `🔌` | Integration, plugin, provider, or platform support |
| `🔗` | Public API or compatibility change |
| `❌` | Deprecation or breaking change; explain migration |
| `✂️` | Removal or experimental work that did not ship |
| `꩜` | Restoration or rollback to an older implementation |
| `❗` | Known unresolved bug or limitation |
| `🩹` | Temporary workaround or mitigation; include removal criteria |
| `♻️` | Data, checkpoint, cache, or migration work |
| `📦` | Packaging, distribution, or release artifact change |
| `🛈` | Caveat, informational notice, or scope clarification; not a substitute for a known-issue entry |

Use the closest symbol consistently. A symbol does not replace an explanation.

### How to use `🛈` notices

Use `🛈` for useful context that helps a reader interpret a change but does **not** claim that the software has a bug. Notices should be specific, neutral, and short. State the scope or limitation directly; avoid vague phrases such as “may have issues.” Put the notice next to the change it qualifies, or under `### Caveats` / `### Notes` when it applies to the release as a whole.

Good examples:

```markdown
🛈 Tested on Linux only; Windows and macOS have not yet been tested.
🛈 This example requires Bun to be installed before it can run.
🛈 The benchmark covers the CPU backend only; CUDA performance was not measured.
🛈 The V4 checkpoint format is experimental and may change before the stable release.
🛈 The optional provider was not configured in this environment, so its integration path was not exercised.
```

Choose a different label when the entry describes a real problem or a corrective action:

```markdown
# Caveat: the feature has not been tested on these platforms.
🛈 Tested on Linux only; Windows and macOS have not yet been tested.

# Known issue: a reproducible bug is still present.
❗ Loading a V4 checkpoint fails when optimizer state is present.

# Temporary workaround: users can mitigate a known problem now.
🩹 Use the V3 loader for existing V3 checkpoints until conversion is fixed.

# Fixed bug: the release contains a verified correction.
𖢥 Fixed V4 checkpoint loading when optimizer state is present.
```

Do **not** use `🛈` to disguise a confirmed bug, imply a fix that has not been verified, or label an unmet prerequisite as a known issue. “Only Chromium was tested” is a caveat, not a known issue. “Bun must be installed” is a prerequisite, not a bug. A notice can explain the context of another entry, but it does not replace the required entry or its symbol.

---

## Version format

Published releases normally use:

```markdown
## 0.72.5 — 2026-09-25 — API improvements
### Bugs fixed: __; security issues fixed: __
```

Pre-releases may use:

```markdown
## 0.73.0-beta.1 — 2026-09-25
## 0.73.0-dev.20260925
```

Post releases and development revisions preserve their exact package version:

```markdown
## 0.72.4.post18
```

Patch releases published to PyPI must use this format:

```markdown
## 0.72.7.post14 — patch # — headline change or fix
```

For LTS releases, use this format:

```markdown
## 0.72.6.post7 — LTS-# — YYYY-MM-DD — number of security issues fixed
```

For the alternate patch-release format, use this form only for security patches in non-LTS releases:

```markdown
## 0.72.4.post18 — patch # — dependencies updated — [CRITICAL FIX, if applicable]
```

If an in-branch development point has no meaningful release date, omit the date rather than inventing one.

---

## Quick reminder

A release's ordinary bug-fix log must **not treat bugs introduced by work during the current update as pre-existing bugs fixed by that update**. Track update-created regressions and in-development bug-fix work in a separate dropdown for that release in the changelog. Create this dropdown for **every release**, including when there is nothing to report. Review dependencies across the **entire repository**, not only Python dependencies. Keep the canonical release log and its per-release dropdowns in the repository-root file `changeLog.root.md`.

## Per-release dropdown for in-development bug fixes

Use one expandable dropdown in each release entry to record bugs discovered, introduced, or being investigated while that release is under development. This is a development-tracking record, not proof that an issue is fixed or that a fix shipped. Do not count these entries as verified release fixes or include them in a “bugs fixed” total.

- Use the dropdown for a regression introduced by the current branch/update, an in-progress fix, a bug still being investigated, or a fix awaiting tests or verification.
- Use `### Fixed` for verified release fixes to pre-existing bugs and defects. Bugs introduced by work during the current update belong **only in that release's dropdown**, even if they are corrected before release. Mark them `FIXED DURING DEVELOPMENT` there and do not count them as pre-existing bugs fixed by the release.
- Review dependencies across the whole repository, including Python, JavaScript/Node, Rust, system/runtime packages, build tooling, CI workflows, and any other dependency manifests the repository actually uses. Do not assume a Python-only review is sufficient; record actual dependency changes in `### Dependencies and Packaging` as well.
- Use one dropdown per release. If there are no in-development bugs or update-created regressions to report, keep the dropdown and state that explicitly.

Example with in-development work still open:

```markdown
<details>
<summary>0.73.1 — In-development bug-fix log (not release-verified)</summary>

- **OPEN — introduced during update:** The current development branch can fail when the optional cache is empty; investigation is ongoing.
- **IN PROGRESS:** A fix for the cache regression is drafted but has not passed the regression test yet.
- **NEEDS VERIFICATION:** A dependency update may change Windows path handling; test results are still pending.

</details>
```

Example when the release has no in-development bug-fix items:

```markdown
<details>
<summary>0.73.1 — In-development bug-fix log</summary>

- No in-development bug fixes or update-created regressions are currently tracked for this release.

</details>
```

Example when a regression introduced during development is fixed before shipping (keep this entry in the dropdown only; do not count it as a pre-existing bug fixed by the release):

```markdown
<details>
<summary>0.73.1 — In-development bug-fix log</summary>

- **FIXED DURING DEVELOPMENT — introduced by this update:** The update temporarily broke empty-cache handling. The corrective change passed the targeted regression test before release.

</details>
```

The dropdown is the record for bugs introduced or worked on during the current update. Do not duplicate those regressions in the ordinary `### Fixed` list or include them in its bug-fix count. Use `### Fixed` for verified fixes to pre-existing bugs. Do not mark any item verified until the relevant test or reproduction check has passed.

## Release categories

Use only categories that contain entries. Keep categories in this order unless a release genuinely needs a different structure:

1. `Beta / Dev → Release Summary`
2. `Breaking Changes`
3. `Added`
4. `Changed`
5. `API Changes`
6. `Architecture`
7. `CLI and UX`
8. `Performance`
9. `Compatibility`
10. `Security`
11. `Fixed`
12. `Dependencies and Packaging`
13. `Data, Checkpoints, and Migrations`
14. `Deprecated`
15. `Removed`
16. `Documentation`
17. `Site Changes`
18. `Tests`
19. `Known Issues`
20. `Temporary Workarounds`
21. `Caveats`
22. `Notes`

`Site Changes` is the canonical category for wiki, documentation-site, landing-page, examples-site, navigation, and hosted-page changes. Use `Documentation` for repository documentation that is primarily consumed as package or developer documentation.

---

## Beta and dev releases

Beta and development versions are part of the historical record. Document experimental architecture, APIs, modules, training, tokenizers, inference, performance, compatibility, tests, regressions, temporary implementations, and features that may change.

```markdown
## 0.73.0-beta.2 — 2026-09-12

### Added

๋࣭⭑ Added the experimental HyperNix0xV4 architecture.
  - Supports the 1B and 3B presets.
𖥔 Added a configuration helper for selecting V4.

### API Changes

🔗 Added `ModelConfig.architecture` for explicit architecture selection.
  - Existing configurations continue to default to V3.

### Changed

🔁 Reworked training phases so each architecture can provide its own schedule.

### Tests

🧪 Added architecture initialization, tensor-shape, and checkpoint-loading tests.

### Known Issues

❗ V4 checkpoint conversion can sometimes corrupt checkpoints. Add the affected checkpoint type and reproduction steps when known.

### Temporary Workarounds

🩹 Use the V3 loader for existing V3 checkpoints until the converter is finalized.
  - Remove this workaround when V4 conversion passes the compatibility suite.

### Caveats

🛈 The V4 checkpoint format is experimental and may change before the stable release.
```

---

## Beta / Dev → Release Summary

Every stable release following one or more beta/dev versions must summarize the important work from that cycle. Focus on what actually shipped; do not copy every pre-release entry.

```markdown
## 0.73.0 — 2026-09-25

### Beta / Dev → Release Summary

This release includes the finalized work accumulated across
`0.73.0-dev` and `0.73.0-beta` releases.

#### Major Changes

๋࣭⭑ Shipped HyperNix0xV4 for the 1B and 3B presets.
✨ Added streaming generation to the inference API.
🔁 Reworked training phases for architecture-specific schedules.

#### API and Compatibility

🔗 Added `ModelConfig.architecture` while preserving the V3 default.
♻️ Added checkpoint metadata needed by the V4 loader.

#### Fixes Carried Into Release

𖢥 Fixed future-token leakage in Brewer attention.
🐛 Fixed an incorrect error when a checkpoint directory was empty.

#### Experimental Work That Did Not Ship

✂️ Removed the beta-only speculative decoder; it was not stable enough for release.

#### Testing

🧪 Added causality, checkpoint-conversion, and streaming-inference regression tests.

#### Documentation and Site

📚 Documented the architecture-selection and migration steps.
🛜 Updated the architecture comparison page and release navigation.
```

---

## API changes

Public API changes must be recorded separately from general changes. Include:

- the affected package, class, function, command, endpoint, or configuration key;
- whether the change is additive, behavioral, deprecated, or breaking;
- old and new signatures or values when relevant;
- compatibility behavior;
- migration instructions;
- the first version containing the change.

```markdown
### API Changes

🔗 Added optional `stream=True` to `hypernix.generate()`.
  - Default behavior is unchanged.
  - When enabled, the return value is an iterator of token chunks.

⚠️ Changed `temperature=None` to mean “use the model default” instead of `0.0`.
  - Pass `temperature=0.0` for deterministic sampling.

❌ Removed `Trainer.train_epoch()`.
  - Migration: call `Trainer.train()` with `epochs=1`.
  - The old method was deprecated in 0.72 and is no longer available in 0.74.
```

Record error-code creation, renaming, and semantic changes here as well as under `Fixed` when appropriate.

---

## Breaking changes and migrations

Never hide a breaking change in `Changed` or `Fixed`.

```markdown
### Breaking Changes

❌ Removed the legacy `BrewerConfig.window` option.
  - Use `BrewerConfig.sliding_window_size` instead.
  - Migration: rename the key; the value has the same unit and meaning.
  - The legacy key was deprecated in 0.72 and removed in 0.74.

### Data, Checkpoints, and Migrations

♻️ Added a converter for V3 checkpoints using the legacy key.
  - Conversion is one-way; retain the original checkpoint until loading succeeds.
```

Architecture changes should identify affected model sizes, layers, tensor shapes, attention behavior, training and inference implications, checkpoint compatibility, and migration requirements.

---

## Known issues and temporary workarounds

### Known Issues

Use `❗` for an unresolved problem. State who or what is affected, what is unaffected, how to reproduce it when useful, and whether a fix is planned.

```markdown
### Known Issues

❗ Loading V4 checkpoints produced by beta.1 can fail when optimizer state is present.
  - Inference-only loading is unaffected.
  - A conversion fix is planned for 0.73.1.
```

When the issue is fixed, add the fix under `Fixed` in the fixing release. Keep the historical known-issue entry in the old release.

### Temporary Workarounds

Use `🩹` for a mitigation that users can apply now. Every workaround must state its scope, risk, and removal condition.

```markdown
### Temporary Workarounds

🩹 Set `attention_backend="eager"` when the fused backend returns an invalid shape.
  - Affects CUDA 12.1 with the 0.73.0 wheel only.
  - This may reduce throughput; it does not change model results.
  - Remove the workaround when the backend fix in 0.73.1 is installed.
```

Do not present a workaround as a fix. If a workaround becomes permanent, rewrite it as a normal documented behavior in the release that adopts it.

---

## Site changes

Record changes to the wiki, hosted documentation, landing pages, examples site, release pages, navigation, search, styling, and public notices under `### Site Changes`.

```markdown
### Site Changes

🛜 Added the V4 architecture comparison page.
🛜 Updated the API reference navigation for the streaming methods.
🛜 Added a visible warning to the checkpoint-conversion guide.
🔗 Linked the migration guide from the release notes and package documentation.
```

Mention content or behavior changes, not every typo or generated-site build. Link the relevant page when the changelog supports links.

---

## Restorations and removals

When reverting to an older implementation because a newer one caused problems, use `꩜` and explain why:

```markdown
### Changed

꩜ Restored the 0.71.x tokenizer fallback because the new fallback corrupted byte-level tokens.
```

Use `✂️` for removed features, modules, pages, APIs, or experimental work that did not ship. Use `❌` for deprecations and explain the replacement, support period, and expected removal version.

---

## Testing and evidence

Document tests when they materially increase confidence:

```markdown
### Tests

🧪 Added a regression test that perturbs one input token and verifies that earlier
outputs do not change.
🧪 Added checkpoint compatibility tests for V3-to-V4 conversion.
```

For major correctness bugs, describe the property tested, not only the number of tests. Do not report unverified test counts or benchmark results.

---

## Example complete entry

```markdown
## 0.72.4.post18 — 2026-09-25

### Fixed

𖢥 Fixed Brewer attention leaking future tokens during training.
  - Corrected the sliding-window range to `0 <= dist <= window - 1`.
  - Combined additive causal and window masks with `torch.minimum`.

  🛡️ Clarified the configuration error shown for an invalid window size.

### Performance

⚡ Plain-causal layers without a padding mask now use `is_causal=True` with SDPA.
  - This avoids materializing a full `B × H × T × T` mask when the backend supports a fused path.

### API Changes

🔗 Preserved the explicit-mask path for callers that provide `attn_mask`.

### Tests

🧪 Added causality and sliding-window regression coverage, including a property-based
input-perturbation test.

### Caveats

🛈 Fused SDPA was not verified in this environment because the required PyTorch backend was unavailable. Treat this as an environment caveat unless testing confirms an actual SDPA bug.

<details>
<summary>0.72.4.post18 — In-development bug-fix log</summary>

- No in-development bug fixes or update-created regressions are documented in this example.

</details>

```

---

## Canonical release-entry template

Copy this template and remove empty sections. Keep minor items next to the change they refine.

```markdown
## X.Y.Z — YYYY-MM-DD

### Beta / Dev → Release Summary
This release includes finalized work from `X.Y.Z-dev` and `X.Y.Z-beta`.

#### Major Changes
๋࣭⭑ <major feature or architecture>
✨ <important regular feature>
𖥔 <minor addition immediately related to the item above>

### Breaking Changes
❌ <breaking change and migration>

### Added
✨ <feature>
𖥔 <minor feature directly related to its feature>

### Changed
🔁 <behavior or refactor>

### API Changes
🔗 <public API addition or change>
⚠️ <error-code or error-semantics change>

### Architecture
๋࣭⭑ <architecture, shape, checkpoint, and migration details>

### CLI and UX
🛡️ <CLI, message, warning, or setup improvement>

### Performance
⚡ <measured or carefully described improvement>

### Compatibility
🔌 <platform, provider, Python, PyTorch, or integration support>

### Security
🔒 <security improvement>

### Fixed
𖢥 <major correctness or stability fix>
🐛 <minor fix immediately related to the relevant feature when possible>

<details>
<summary>X.Y.Z — In-development bug-fix log</summary>

- <Update-created regression, in-progress fix, verification status, or explicit statement that there are no such items.>

</details>

### Dependencies and Packaging
📦 <dependency, wheel, build, or distribution change>

### Data, Checkpoints, and Migrations
♻️ <migration or checkpoint change>

### Deprecated
❌ <deprecated item, replacement, and removal timeline>

### Removed
✂️ <removed item>
꩜ <restored item and reason>

### Documentation
📚 <repository documentation>

### Site Changes
🛜 <wiki, hosted documentation, or website change>

### Tests
🧪 <regression, compatibility, or new test coverage>

### Known Issues
❗ <unresolved issue and affected scope>

### Temporary Workarounds
🩹 <temporary mitigation, risk, and removal condition>

### Caveats
🛈 <platform test coverage, environment prerequisite, benchmark scope, or other relevant caveat>

### Notes
<additional release context that does not fit another category>
```

---

## Release checklist

- [ ] Version matches package metadata and the PyPI artifact.
- [ ] Release date is `YYYY-MM-DD`.
- [ ] All changes since the previous release were reviewed and grouped.
- [ ] Beta/dev work is summarized, including work that did not ship.
- [ ] Major, regular, and minor items use the correct spacing and grouping.
- [ ] API changes, error-code changes, and breaking changes are explicit.
- [ ] Migration instructions exist for breaking changes and checkpoint changes.
- [ ] Known bugs and temporary workarounds state scope and expected resolution.
- [ ] Site changes are included when applicable.
- [ ] Major fixes explain impact and regression tests explain the protected property.
- [ ] Every release has its in-development bug-fix dropdown, including an explicit “none tracked” statement when empty.
- [ ] Bugs created or investigated during the update are not incorrectly counted as pre-existing bugs fixed by the release.
- [ ] Dependency review covers the entire repository, not only Python packages, and is recorded in `changeLog.root.md`.
- [ ] Notices (`🛈`) describe caveats or context rather than confirmed bugs.
- [ ] No benchmark or compatibility claim is unverified.
- [ ] `wiki/Home.md` Recent Highlights was updated for a stable release.
- [ ] Empty categories and commit-by-commit noise were removed.
- [ ] No unreleased changes are put in a numbered version

---

## Relationship with `wiki/Home.md`

`wiki/Home.md` is the quick overview; this file is the canonical history. Home should contain only concise recent highlights:

```markdown
## Recent Highlights

- **0.72.4** — Brewer attention causality fixes and SDPA improvements.
- **0.72.3** — Streaming generation API.
- **0.72.2** — Checkpoint compatibility improvements.
```

For each new stable release:

1. Add the complete entry here.
2. Add the Beta / Dev → Release Summary when applicable.
3. Update `wiki/Home.md`.
4. Move older highlights out of the quick list when appropriate.
5. Preserve the complete technical history here permanently.

---

## Historical accuracy

Describe hyperNix as it existed at the time of the release. If a technical description was wrong, make an explicit correction:

> Changelog correction: the original entry incorrectly described mask composition as additive. The implementation actually used an additive representation with maximum composition.

Do not remove historical context merely because a later release fixed, reverted, or replaced the behavior.
