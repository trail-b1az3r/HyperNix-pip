# HyperNix Documentation Site

This directory contains the React/Vite GitHub Pages site for HyperNix. The site is built as a static app and reads generated repository data from `public/v1/`.

## Site surfaces

- **Home** — product overview, quickstart, supported models, searchable subsystem reference, latest-release snapshot, and home-only credits.
- **Docs** — the wiki as a documentation site, laid out like PyTorch's and Hyprland's: grouped pages on the left (`src/lib/data/docs-nav.json` sets the groups; an unlisted page lands under More), the page in the middle, "On this page" on the right, breadcrumbs, previous/next, and Ctrl+K search over titles, headings and every page's full text. Pages render with HyperLink's Markdown renderer (`src/hypernix/hyperlink/web/markdown.js`), imported rather than copied. Every page and heading has an address: `#/docs/T1-API/<heading>`.
- **API** — public API reference plus the generated deep API view.
- **T1 API** (`#/t1-api`) — an overview and quickstart, every endpoint from `examples/t1api/openapi.json` (by group, searchable, with parameters, body fields and a curl line), every setting from `examples/t1api/.env.example`, the T1 error codes, and the source-generated reference.
- **Learn** — task-oriented guides and examples.
- **Stats** — PyPI/download, repository, code, and release-timeline data, with a multi-line graph of daily downloads (total, or by Python version or operating system) from the `series` block `scripts/update_json_stats.py` keeps in `public/v1/json`.
- **About** — project context and external links.
- **Security** (`#/security`, not in the menu: the footer's hidden button, a shield after the version line that shows on hover or Tab) — what the repository's own scanners found at the last build: the HyperNix T1 audit, Bandit, Semgrep, and OSV over every dependency floor and lockfile, by severity, filterable, each finding linked to its line at the scanned commit.

## Development

```bash
npm install
npm run dev
npm run build
npm run preview
```

## Generated data

At build time `vite.config.js` copies `../wiki/*.md` into the site as `wiki/<Page>.md` and writes `v1/docs-index.json` (groups, titles, headings, summaries), `v1/t1-openapi.json` and `v1/t1-settings.json`; `npm run dev` serves the same files from the checkout. None of them are committed. `public/v1/error-codes.json` is written by `../scripts/generate_error_codes.py`.

`public/v1/security.json` is written in the Pages build by `../scripts/security_report.py` (run it locally to see the page in `npm run dev`); it is not committed.

`../scripts/generate_docs_data.py` refreshes the API reference JSON, code statistics, and the parsed changelog at `public/v1/changelog.json`. The changelog summary fields are derived from the actual canonical change bullets in `wiki/Changelog.md`; release dates are kept separate from summaries.

The scheduled `.github/workflows/update-docs-data.yml` workflow regenerates these files and commits only when their contents change.

## Deployment

`.github/workflows/deploy-docs.yml` deploys the site to GitHub Pages after changes to `docs/`, `README.md`, `wiki/`, the code (`src/`, `scripts/`, `examples/`, `pyproject.toml`, so the security page follows it) or the security workflow, daily, and by hand. Each build runs the security scan first; a scanner that fails is shown as failed on the page and does not stop the deploy.

## Tech stack

- **React 18** — UI framework
- **Vite** — build/dev tooling
- **TypeScript/TSX** — site components
- **Custom CSS + inline component styling** — the visual system and responsive layout
- **Lucide React** — icons

The current site does not depend on Tailwind CSS or Framer Motion; older copies of this README described tooling that is no longer present.
