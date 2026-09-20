# Generated API and code statistics

The docs site reads three source-derived files from `docs/public/v1/`:

- `api-deep.json` — public HyperNix API inventory, requirements, warnings/deprecations, errors, examples, and file history.
- `t1-api.json` — T1 server/SDK modules, HTTP routes, error types, requirements, examples, and file history.
- `code-stats.json` — non-blank code line count plus Git additions/deletions per contributor.
- `changelog.json` — versioned summaries parsed directly from `wiki/Changelog.md`, including stable, post, prerelease, and patch labels.

`.github/workflows/update-docs-data.yml` runs hourly with a full Git checkout. It commits only when one of the three generated files changes. The normal docs deployment then rebuilds GitHub Pages from that commit.
