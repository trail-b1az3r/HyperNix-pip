import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'fs'
import path from 'path'

const WIKI = path.resolve(__dirname, '../wiki')
const NAV = path.resolve(__dirname, 'src/lib/data/docs-nav.json')
const OPENAPI = path.resolve(__dirname, '../examples/t1api/openapi.json')
const ENV_EXAMPLE = path.resolve(__dirname, '../examples/t1api/.env.example')

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

// The T1 page's endpoint explorer: examples/t1api/openapi.json (generated
// by driving a real server), cut down to what the page shows, with each
// body's fields resolved from its schema.
function buildT1Openapi() {
  if (!fs.existsSync(OPENAPI)) return { version: '', routes: [], tags: [] }
  const doc = JSON.parse(fs.readFileSync(OPENAPI, 'utf-8'))
  const schemas = (doc.components && doc.components.schemas) || {}
  const typeOf = (sch) => {
    if (!sch) return ''
    if (sch.$ref) return sch.$ref.split('/').pop()
    if (sch.anyOf) return sch.anyOf.map(typeOf).filter((t) => t && t !== 'null').join(' | ')
    if (sch.type === 'array') return `${typeOf(sch.items) || 'any'}[]`
    if (sch.enum) return sch.enum.map((e) => JSON.stringify(e)).join(' | ')
    return sch.type || (sch.const !== undefined ? JSON.stringify(sch.const) : 'any')
  }
  const fields = (ref) => {
    const sch = ref && ref.$ref ? schemas[ref.$ref.split('/').pop()] : ref
    if (!sch || !sch.properties) return []
    const required = new Set(sch.required || [])
    return Object.entries(sch.properties).map(([name, f]) => ({
      name, type: typeOf(f), required: required.has(name),
      default: f.default === undefined ? undefined : JSON.stringify(f.default),
      // A title is FastAPI's capitalised field name, not a description.
      description: (f.description || '').replace(/\s+/g, ' ').slice(0, 240),
    }))
  }
  const routes = []
  for (const [p, ops] of Object.entries(doc.paths || {})) {
    for (const [method, op] of Object.entries(ops)) {
      if (!op || typeof op !== 'object' || !op.responses) continue
      const bodyRef = op.requestBody?.content?.['application/json']?.schema
      routes.push({
        method: method.toUpperCase(), path: p, tag: (op.tags && op.tags[0]) || 'other',
        summary: op.summary || '', description: (op.description || '').trim().slice(0, 1600),
        // The credential headers are on every route; the page explains them once.
        params: (op.parameters || []).filter((x) => !(x.in === 'header' && /^(authorization|x-api-key)$/i.test(x.name)))
          .map((x) => ({ name: x.name, in: x.in, required: !!x.required,
          type: typeOf(x.schema), description: (x.description || '').slice(0, 200) })),
        body: bodyRef ? { type: typeOf(bodyRef), fields: fields(bodyRef) } : null,
        multipart: !!op.requestBody?.content?.['multipart/form-data'],
        responses: Object.keys(op.responses || {}),
        response: typeOf(op.responses?.['200']?.content?.['application/json']?.schema) || '',
      })
    }
  }
  const counts = {}
  for (const r of routes) counts[r.tag] = (counts[r.tag] || 0) + 1
  return { title: doc.info?.title, version: doc.info?.version, description: doc.info?.description,
           tags: Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([name, count]) => ({ name, count })),
           routes }
}

// Every T1_* setting, with its section, default and what it does, from
// examples/t1api/.env.example (which a test keeps complete).
function buildT1Settings() {
  if (!fs.existsSync(ENV_EXAMPLE)) return { settings: [] }
  const settings = []
  const seen = new Set()
  let section = 'General'
  let notes = []
  for (const raw of fs.readFileSync(ENV_EXAMPLE, 'utf-8').split('\n')) {
    const line = raw.trimEnd()
    const head = /^#\s*---\s*(.+?)\s*-{3,}\s*$/.exec(line)
    if (head) { section = head[1]; notes = []; continue }
    if (!line.trim()) { notes = []; continue }
    const set = /^(#\s*)?(T1_[A-Z0-9_]+)=(.*)$/.exec(line)
    if (set) {
      if (!seen.has(set[2])) {
        seen.add(set[2])
        settings.push({ name: set[2], value: set[3].replace(/^['"]|['"]$/g, ''), example: !!set[1], section,
                        help: notes.join(' ').replace(/\s+/g, ' ').trim() })
      }
      continue
    }
    if (/^#/.test(line) && !/^#\s*[-=]{5,}/.test(line)) notes.push(line.replace(/^#\s?/, ''))
  }
  return { settings }
}

const GENERATED = {
  '/v1/docs-index.json': () => buildIndex(),
  '/v1/t1-openapi.json': () => buildT1Openapi(),
  '/v1/t1-settings.json': () => buildT1Settings(),
}

function wikiPlugin() {
  return {
    name: 'hypernix-wiki',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = decodeURIComponent((req.url || '').split('?')[0])
        if (GENERATED[url]) {
          res.setHeader('Content-Type', 'application/json')
          res.end(JSON.stringify(GENERATED[url]()))
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
      for (const [url, build] of Object.entries(GENERATED)) {
        this.emitFile({ type: 'asset', fileName: url.slice(1), source: JSON.stringify(build()) })
      }
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
