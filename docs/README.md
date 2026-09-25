# HyperNix Documentation Site

This directory contains the React/Vite GitHub Pages site for HyperNix. The site is built as a static app and reads generated repository data from `public/v1/`.

## Site surfaces

- **Home** — product overview, quickstart, supported models, searchable subsystem reference, latest-release snapshot, and home-only credits.
- **Docs** — package and subsystem documentation.
- **API** — public API reference plus the generated deep API/T1 API views.
- **Learn** — task-oriented guides and examples.
- **Stats** — PyPI/download, repository, code, and release-timeline data.
- **About** — project context and external links.

## Development

```bash
npm install
npm run dev
npm run build
npm run preview
```

## Generated data

`../scripts/generate_docs_data.py` refreshes the API reference JSON, code statistics, and the parsed changelog at `public/v1/changelog.json`. The changelog summary fields are derived from the actual canonical change bullets in `wiki/Changelog.md`; release dates are kept separate from summaries.

The scheduled `.github/workflows/update-docs-data.yml` workflow regenerates these files and commits only when their contents change.

## Deployment

`.github/workflows/deploy-docs.yml` deploys the site to GitHub Pages after changes to `docs/`, `README.md`, or `wiki/`, and can also be run manually.

## Tech stack

- **React 18** — UI framework
- **Vite** — build/dev tooling
- **TypeScript/TSX** — site components
- **Custom CSS + inline component styling** — the visual system and responsive layout
- **Lucide React** — icons

The current site does not depend on Tailwind CSS or Framer Motion; older copies of this README described tooling that is no longer present.
