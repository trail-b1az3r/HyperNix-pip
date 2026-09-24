hyperNix Changelog Guide

This document is the canonical release history for hyperNix.

It records every published release, important pre-release development, fixes, features, removals, and known issues. The top-level wiki/Home.md should maintain a short Recent Highlights list for quick navigation, while this file contains the complete historical record.

The format is inspired by Keep a Changelog, with additional conventions for hyperNix development and pre-release versions.

Changelog Principles

* Every PyPI-published release gets its own version header.
* Published releases use the date they were released to PyPI.
* Dates use YYYY-MM-DD.
* Commits made between releases are grouped beneath the next release header.
* Beta and dev versions may document experimental work before a stable release.
* When a stable release is published, its changelog entry must include a Beta / Dev → Release Summary covering the significant changes accumulated since the previous stable release.
* Do not silently discard beta/dev work just because it was experimental.
* If a beta/dev feature was removed before release, document its removal when useful.
* Patch releases should generally contain bug fixes, regressions, compatibility fixes, and UX papercuts.
* Minor releases should generally contain new features, integrations, APIs, architecture improvements, or meaningful user-facing changes.
* Major releases may contain breaking architecture/API changes, major migrations, or substantial redesigns.
* Known unresolved problems must remain documented until fixed.
* Corrections to previous changelog entries should be made explicitly rather than silently rewriting history.

⸻

Legend

Symbol	Meaning
🧪	New tests / test coverage
✨	Normal feature
🐛	Minor bug fix
🛡️	UX / error-message polish
📚	Documentation
🔧	Internal / plumbing
✂️	Cut / remove
🛜	Website update / pages update
🔁	Refactor / integration improvement
𖢥	Major bug fix
꩜	Restore to an older version of an item
❗	Unfixed known bug
❌	Deprecation
๋࣭⭑	Major new feature
𖥔	Minor new feature
⚠️      error code updating/creation/editing

⸻

Version Format

Published releases should normally use:

## 0.72.5 — 2026-09-XX

Pre-release versions may use:

## 0.73.0-beta.1 — 2026-09-XX
## 0.73.0-dev.20260914

Post releases and development revisions should preserve their exact package version:

## 0.72.4.post18

If a release has no meaningful date because it exists only as an in-branch development point, omit the date rather than inventing one.

⸻

Release Categories

Use the following categories where applicable.

### Added
### changes that effect developers 
### changes that effect users
### Changed
### Fixed
### Performance
### Security
### Deprecated
### Removed
### Documentation
### Tests
### Known Issues

Not every release needs every category.

Do not create empty categories.

⸻

Beta / Dev Release Policy

Beta and development releases are part of the historical record.

They should document:

* experimental architecture changes
* new APIs
* new modules
* training changes
* tokenizer changes
* inference changes
* performance work
* compatibility work
* tests
* known regressions
* temporary implementations
* features that may still change

For example:

## 0.73.0-beta.2 — 2026-09-12
### Added
๋࣭⭑ Added the experimental HyperNix0xV4 architecture.
𖥔 Added multi-threaded internal state processing.
### Changed
🔁 Reworked the training pipeline to support architecture-specific
learning phases.
### Tests
🧪 Added architecture initialization and shape-validation tests.
### Known Issues
❗ V4 checkpoint conversion is not yet guaranteed to be stable.

⸻

Beta / Dev → Release Summary

Every stable release that follows one or more beta/dev releases should contain a summary of the important work accumulated during that development cycle.

This prevents the stable changelog from appearing to contain only the final commit when a release actually represents weeks of development.

Use:

## 0.73.0 — 2026-09-XX
### Beta / Dev → Release Summary
This release includes the finalized changes developed across
0.73.0-dev and 0.73.0-beta releases.
#### Major Changes
๋࣭⭑ <major feature or architecture change>
✨ <important user-facing feature>
🔁 <important refactor or integration>
#### Fixes Carried Into Release
𖢥 <major bug fix>
🐛 <smaller fixes>
#### Experimental Work That Did Not Ship
✂️ <feature removed before release>
#### Testing
🧪 <important test coverage added>
#### Documentation
📚 <documentation completed or updated>

The summary should focus on what actually shipped.

Do not copy every beta/dev changelog entry into the stable release. Summarize the important changes and link/reference the earlier development versions when appropriate.

⸻

Development Changes Between Releases

Commits made after a published release but before the next release should be grouped beneath the next release header.

For example:

## 0.74.0 — 2026-10-01
### Beta / Dev → Release Summary
Development for 0.74.0 began immediately after 0.73.0.
### Added
๋࣭⭑ Added <feature>.
### Changed
🔁 Reworked <system>.
### Fixed
𖢥 Fixed <major regression>.
### Tests
🧪 Added <tests>.

Avoid creating dozens of entries such as:

commit abc123
commit def456
commit ghi789

The changelog is a release history, not a Git log.

⸻

Example Entry

0.72.4.post18 — Brewer attention was not causal

Fixed

𖢥 Brewer attention was not causal

Reported against BrewerAttention.forward, and confirmed as a genuine causality failure.

The causal and sliding-window masks were previously combined using torch.maximum.

Both masks are additive masks:

* 0 means attention is allowed.
* torch.finfo(...).min means attention is forbidden.

Using torch.maximum therefore kept the less masked value. In practice this implemented:

allow if either mask allows

when the required operation was:

mask if either mask masks.

A non-causal language model can train to an excellent loss because it can see tokens it is supposed to predict. The resulting loss curve therefore cannot, by itself, prove that the model is causal.

The bug was worse than simply leaking the sliding window.

Sliding-window failure

𖢥 _sliding_mask constructed its band from:

dist = i - j

and then masked:

dist >= 0

Because positive dist represents the past, this masked the past and the current token while leaving positions in the future visible.

The resulting mask was effectively anti-causal rather than causal.

There was also a second failure for sequences whose length was no greater than the configured window.

With:

sliding_window_size = 4096

the relevant band could forbid nothing at all.

The four presets using sliding-window attention configured windows of:

1024
4096
8192
16384

Therefore, training sequences at or below their configured window could cause the affected layers to attend bidirectionally across the entire sequence.

Both masking defects had to be fixed

𖢥 The corrected sliding-window mask now keeps only:

0 <= dist <= win - 1

𖢥 The causal and sliding-window masks are now combined with:

torch.minimum(...)

This produces the required intersection:

allowed = causal AND sliding_window

Using only torch.minimum with the old sliding mask would instead mask every position in a row, causing softmax to encounter an entirely masked row and produce NaN.

Using the corrected sliding mask with torch.maximum would technically remain causal, but the sliding window would become a silent no-op because the causal mask already forbids everything the window additionally forbids.

Both halves therefore required correction.

⸻

Causal SDPA path

🔧 Plain-causal even layers with no padding mask now pass:

is_causal=True

to scaled dot-product attention instead of materialising a full:

B × H × T × T

mask tensor.

This allows the backend to select a fused attention implementation when available.

Because is_causal and attn_mask are mutually exclusive, callers providing an explicit mask continue to use the explicit-mask path.

🧪 Added numerical tests confirming that the explicit causal-mask path and is_causal=True produce equivalent results.

⸻

Tests

🧪 Added tests/test_brewer_causality.py.

The test suite contains 25 causality and sliding-window tests.

The central regression test follows the reporter’s property-based approach:

1. Run the model normally.
2. Perturb one input token.
3. Verify that outputs belonging to earlier positions do not change.

This tests the actual causality property rather than relying on the implementation details of the mask.

The test suite was also checked against the known broken implementations:

Implementation	Result
Original implementation	❌ 13 tests failed
torch.minimum only	❌ 13 tests failed
Sliding-mask correction only	❌ 3 tests failed
Correct sliding mask + torch.minimum	✅ passes

The three remaining failures in the mask-fixed-only variant were window-behaviour tests: that implementation was causal, but the sliding-window restriction was ineffective.

⸻

Writing Good Release Notes

A changelog entry should explain what changed and why it matters.

Prefer:

𖢥 Fixed Brewer attention leaking future tokens during training.

over:

🐛 Fixed attention bug.

For larger changes, explain the mechanism when it is useful to developers maintaining the project.

For example:

🔁 Reworked mask composition to use intersection semantics for
multiple additive attention masks.

is more useful than:

🔁 Updated attention.

⸻

Bug-Fix Severity

Use the symbols consistently.

🐛 Minor Bug Fix

Use for:

* small regressions
* incorrect UI behaviour
* harmless edge cases
* minor compatibility problems
* error-message issues

𖢥 Major Bug Fix

Use when the bug can significantly affect:

* model correctness
* training correctness
* generated output
* data integrity
* API behaviour
* security
* checkpoint compatibility
* inference correctness
* system stability

A causality failure in an LLM architecture is a major bug, even if the implementation change itself is small.

🛡️ UX / Error Polish

Use for:

* clearer errors
* better warnings
* improved CLI messages
* friendlier setup failures
* improved progress indicators
* confusing configuration messages

⸻

Feature Severity

✨ Normal Feature

Use for normal user-facing or developer-facing functionality.

𖥔 Minor New Feature

Use for a smaller addition that is useful but does not significantly change hyperNix.

๋࣭⭑ Major New Feature

Use for:

* new model architectures
* major inference systems
* major training systems
* new package subsystems
* significant API systems
* major integrations
* substantial user-facing functionality

⸻

Architecture Changes

Architecture changes should explicitly identify:

* architecture name
* affected model sizes
* new layers
* removed layers
* tensor-shape changes
* attention changes
* training implications
* inference implications
* checkpoint compatibility
* migration requirements

Example:

### Architecture
๋࣭⭑ Introduced HyperNix0xV4.
The architecture adds:
- <layer>
- <layer>
- <module>
🔧 Updated the configuration system to expose the new architecture.
🧪 Added tensor-shape and checkpoint-loading tests.
❗ Existing 0xV3 checkpoints are not currently loadable without conversion.

⸻

Breaking Changes

Breaking changes must be clearly marked.

### Breaking Changes
❌ Removed the legacy `<API>` interface.
❌ `<configuration>` no longer accepts `<old value>`.
Migration:
```text
old configuration
        ↓
new configuration

See the migration documentation for the complete procedure.

Never hide breaking changes inside a generic "Changed" section.
---
# Deprecations
Use:
```markdown
❌ Deprecated `<feature>`.

and explain:

* why it is deprecated
* what replaces it
* when removal is expected
* whether existing users can continue using it temporarily

Example:

❌ Deprecated the legacy Brewer attention configuration.
Use the new Brewer configuration instead.
The legacy configuration remains available during the deprecation period
but will not receive new features.

⸻

Restorations

When reverting to an older implementation because a newer implementation caused problems, use:

꩜ Restored `<feature>` to the 0.71.x implementation.

Explain why the restoration happened.

Do not treat a restoration as a generic bug fix when the historical version is important to understanding the change.

⸻

Known Issues

Known bugs that remain unresolved should be explicitly listed.

### Known Issues
❗ <description of unresolved issue>
The issue does not affect <unaffected functionality>.
A fix is planned for <version>, if known.

When fixed, remove the ❗ entry and document the fix under Fixed.

Do not delete historical context if the issue was particularly significant.

⸻

Documentation Changes

Use:

### Documentation
📚 Updated <documentation>.
📚 Added <guide/reference>.
📚 Clarified <configuration/API>.

Website changes use:

### Website
🛜 Updated <page>.
🛜 Added <page>.
🛜 Reworked <section>.

⸻

Testing

Tests should be documented when they materially improve confidence in the release.

### Tests
🧪 Added tests for <feature>.
🧪 Added regression coverage for <bug>.
🧪 Added compatibility tests for <platform/version>.

For major correctness bugs, explain what property the tests verify rather than only reporting a test count.

⸻

Performance Changes

Use:

### Performance
🔧 Reduced memory usage during <operation>.
🔁 Reworked <operation> to avoid unnecessary tensor allocation.
✨ Added fused <operation> support when available.

When possible, include measurable information:

🔧 Reduced peak attention-memory usage by approximately X%
for sequences of length Y.

Do not claim benchmarks unless they were actually measured.

⸻

Release Checklist

Before publishing a release, verify:

* [ ]	Version number matches the package metadata.
* [ ]	PyPI release date is recorded as YYYY-MM-DD.
* [ ]	All commits since the previous release have been reviewed.
* [ ]	Important beta/dev changes are summarized.
* [ ]	Removed experimental features are documented.
* [ ]	Breaking changes are explicitly marked.
* [ ]	Known bugs are listed.
* [ ]	Major bug fixes explain their impact.
* [ ]	Important regression tests are documented.
* [ ]	Documentation changes are included.
* [ ]	Website changes are included when applicable.
* [ ]	Migration information exists for breaking changes.
* [ ]	wiki/Home.md Recent Highlights has been updated.
* [ ]	The changelog does not duplicate the entire Git history.

⸻

Relationship With wiki/Home.md

wiki/Home.md is the quick overview.

This file is the canonical history.

Home.md should contain only a concise list of recent highlights, for example:

## Recent Highlights
- **0.72.4** — Brewer attention causality fixes and SDPA improvements.
- **0.72.3** — <major feature>.
- **0.72.2** — <important fix>.

The complete technical details belong in this changelog.

When a new stable release is published:

1. Add the complete release entry to this file.
2. Add the Beta / Dev → Release Summary.
3. Update wiki/Home.md.
4. Move older highlights out of the Recent Highlights section when appropriate.
5. Preserve the complete history here permanently.

⸻

Canonical Release Entry Template

Copy this template for each published release:

## X.Y.Z — YYYY-MM-DD
### Beta / Dev → Release Summary
This release includes the finalized work accumulated across
X.Y.Z-dev and X.Y.Z-beta releases.
#### Major Changes
๋࣭⭑ <major change>
𖥔 <minor major-feature addition>
#### Important Fixes
𖢥 <major fix>
🐛 <minor fix>
#### Experimental Work
✂️ <experimental feature removed before release>
꩜ <feature restored to previous implementation>
### Added
✨ <feature>
### Changed
🔁 <change>
### Fixed
𖢥 <major bug fix>
🐛 <minor bug fix>
### Performance
🔧 <performance improvement>
### Security
🛡️ <security-related improvement>
### Deprecated
❌ <deprecated feature>
### Removed
✂️ <removed feature>
### Documentation
📚 <documentation change>
### Website
🛜 <website change>
### Tests
🧪 <new tests>
### Known Issues
❗ <known unresolved issue>

Omit sections that have no entries.

⸻

Historical Accuracy

The changelog should describe the state of hyperNix at the time of each release.

Do not rewrite old entries simply because the implementation has changed.

If an old entry contained an incorrect technical description, correct it only when necessary and make the correction clear.

Prefer:

> Changelog correction: the original entry incorrectly described
> the mask as additive. The implementation was actually using
> an additive representation with maximum composition.

over silently changing historical information.

The changelog should make it possible for a developer in the future to answer:

* What changed?
* When did it change?
* Why did it change?
* Was it experimental?
* Did it ship?
* Was it later reverted?
* Was it a breaking change?
* What tests protect it?
* Was there a known issue?
* What beta/dev work became part of the release?
* what api changes in the T series API occurred 

That historical trace is more important than keeping every entry short.
