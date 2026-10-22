import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'fs'
import path from 'path'

const WIKI = path.resolve(__dirname, '../wiki')
const NAV = path.resolve(__dirname, 'src/lib/data/docs-nav.json')

// The wiki, served by the site itself: every wiki/*.md is copied into the
// build as wiki/<Page>.md, and v1/docs-index.json holds the sidebar's
// groups and a search index (titles, headings, a summary, plain text).
// The docs page used to fetch each page from GitHub at view time, which
// the unauthenticated API allows sixty times an hour per visitor.

function plain(md) {
  return md
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`([^`]*)`/g, '$1')
    .replace(/!\[[^\]]*\]\([^)]*\)/g, ' ')
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/^\s{0,3}(#{1,6}|>|[-*+]|\d+\.)\s+/gm, '')
    .replace(/(^|\s)[*_~]+|[*_~]+(?=\s|$)/g, '$1')
    .replace(/\|/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function pageInfo(name, md) {
  const lines = md.split('\n')
  let title = name.replace(/-/g, ' ')
  const headings = []
  let inFence = false
  for (const line of lines) {
    if (/^\s*(```|~~~)/.test(line)) { inFence = !inFence; continue }
    if (inFence) continue
    const m = /^(#{1,4})\s+(.+?)\s*#*\s*$/.exec(line)
    if (!m) continue
    const text = m[2].replace(/\{[#.][^}]*\}\s*$/, '').replace(/[`*_]/g, '').trim()
    if (m[1].length === 1 && headings.length === 0 && title === name.replace(/-/g, ' ')) { title = text; continue }
    if (m[1].length <= 3) headings.push({ depth: m[1].length, text })
  }
  // The first real paragraph, for the landing cards and search results.
  const body = md.replace(/```[\s\S]*?```/g, '').split(/\n\s*\n/)
    .map((block) => block.trim())
    .find((block) => block && !/^(#|\||>|-{3,}|<|\[!\[|!\[|```)/.test(block)) || ''
  return {
    title,
    // The sidebar's label: "Convert — hypernix.convert" is "Convert".
    short: title.split(' — ')[0].trim(),
    headings: headings.slice(0, 160),
    summary: plain(body).slice(0, 220),
    // Full text is searched from the pages themselves (Docs.tsx loads
    // them on the first search); this is only for snippets.
    text: plain(md).slice(0, 1200),
    words: plain(md).split(' ').length,
  }
}

function buildIndex() {
  const nav = JSON.parse(fs.readFileSync(NAV, 'utf-8'))
  const files = fs.existsSync(WIKI) ? fs.readdirSync(WIKI).filter((f) => f.endsWith('.md')) : []
  const names = files.map((f) => f.slice(0, -3))
  const pages = {}
  for (const name of names) {
    pages[name] = pageInfo(name, fs.readFileSync(path.join(WIKI, `${name}.md`), 'utf-8'))
  }
  const listed = new Set()
  const groups = nav.groups.map((g) => {
    const present = g.pages.filter((p) => pages[p])
    present.forEach((p) => listed.add(p))
    return { id: g.id, title: g.title, blurb: g.blurb, pages: present }
  })
  const rest = names.filter((n) => !listed.has(n)).sort()
  if (rest.length) groups.push({ id: 'more', title: 'More', blurb: 'Everything else in the wiki.', pages: rest })
  for (const g of groups) for (const p of g.pages) pages[p].group = g.id
  // A short label two pages share says nothing: those keep their titles.
  const seen = {}
  for (const info of Object.values(pages)) seen[info.short] = (seen[info.short] || 0) + 1
  for (const info of Object.values(pages)) if (seen[info.short] > 1) info.short = info.title
  return { generated: new Date().toISOString(), groups, pages }
}

function wikiPlugin() {
  return {
    name: 'hypernix-wiki',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = decodeURIComponent((req.url || '').split('?')[0])
        if (url === '/v1/docs-index.json') {
          res.setHeader('Content-Type', 'application/json')
          res.end(JSON.stringify(buildIndex()))
          return
        }
        const m = /^\/wiki\/([A-Za-z0-9._-]+\.md)$/.exec(url)
        if (m && fs.existsSync(path.join(WIKI, m[1]))) {
          res.setHeader('Content-Type', 'text/markdown; charset=utf-8')
          res.end(fs.readFileSync(path.join(WIKI, m[1])))
          return
        }
        next()
      })
    },
    generateBundle() {
      if (!fs.existsSync(WIKI)) return
      for (const file of fs.readdirSync(WIKI).filter((f) => f.endsWith('.md'))) {
        this.emitFile({ type: 'asset', fileName: `wiki/${file}`, source: fs.readFileSync(path.join(WIKI, file)) })
      }
      this.emitFile({ type: 'asset', fileName: 'v1/docs-index.json', source: JSON.stringify(buildIndex()) })
    },
  }
}

export default defineConfig({
  plugins: [react(), wikiPlugin()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  // The Markdown renderer is HyperLink's (src/hypernix/hyperlink/web/
  // markdown.js), imported from outside docs/ rather than copied.
  server: { fs: { allow: [path.resolve(__dirname, '..')] } },
  base: './',
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
  },
})
