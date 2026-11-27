import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { WIKI_PAGES } from '../lib/data/wiki'
import SUBSYSTEM_PAGES from '../lib/data/subsystem-pages.json'
import { anchorKey, highlight, renderMarkdown } from '../lib/markdown'
import { SUBSYSTEMS, GROUP_ORDER, subsystemGroup } from '../lib/data/subsystems'
import { MODULES } from '../lib/data/api-reference'

// The documentation, laid out the way PyTorch's and Hyprland's are: groups
// down the left, the page in the middle, "On this page" down the right,
// breadcrumbs and search on top, previous/next at the bottom. Pages are the
// repository's wiki/*.md, bundled into the site at build time (see
// vite.config.js) with a GitHub fallback, and every page and heading has
// its own address: #/docs/<Page>/<heading>.

const REPO = 'https://github.com/trail-b1az3r/HyperNix-pip'
const RAW = 'https://raw.githubusercontent.com/trail-b1az3r/HyperNix-pip/main'

type Heading = { depth: number, text: string }
type PageInfo = { title: string, short?: string, group?: string, summary: string, headings: Heading[], text: string, words: number, sourcePath?: string, path?: string }
const label = (index: Index, doc: string) => index.pages[doc]?.short || index.pages[doc]?.title || doc.replace(/-/g, ' ')
const SUBSYSTEM_PAGE_PATHS: Record<string, string> = Object.fromEntries(Object.entries(SUBSYSTEM_PAGES).flatMap(([slug, pages]) => pages.map((page) => [page, `subsystems/${slug}/pages/${page}.mdxl`])))
type Group = { id: string, title: string, blurb: string, pages: string[] }
type Index = { groups: Group[], pages: Record<string, PageInfo> }
type Route = { doc?: string, anchor?: string }

const FALLBACK: Index = {
  groups: [{ id: 'all', title: 'Wiki', blurb: '', pages: WIKI_PAGES }],
  pages: Object.fromEntries(WIKI_PAGES.map((p) => [p, { title: p.replace(/-/g, ' '), summary: '', headings: [], text: '', words: 0, group: 'all' }])),
}

export function docHref(doc?: string, anchor?: string) {
  if (!doc) return '#/docs'
  return `#/docs/${encodeURIComponent(doc)}${anchor ? `/${encodeURIComponent(anchor)}` : ''}`
}

function useIndex(): [Index, boolean] {
  const [index, setIndex] = useState<Index>(FALLBACK)
  const [ready, setReady] = useState(false)
  useEffect(() => {
    fetch('./v1/docs-index.json')
      .then((r) => { if (!r.ok) throw new Error(); return r.json() })
      .then((d) => { if (d && d.groups) setIndex(d) })
      .catch(() => {})
      .finally(() => setReady(true))
  }, [])
  return [index, ready]
}

async function loadPage(name: string, index?: Index): Promise<string> {
  const movedPath = index?.pages[name]?.path || index?.pages[name]?.sourcePath || SUBSYSTEM_PAGE_PATHS[name]
  const urls = movedPath ? [`./${movedPath}`, `${RAW}/${movedPath}`] : [`./wiki/${name}.md`, `${RAW}/wiki/${name}.md`]
  for (const url of urls) {
    try {
      const r = await fetch(url)
      if (r.ok) {
        const text = await r.text()
        // A dev server answers unknown paths with index.html.
        if (!/^\s*<!doctype html/i.test(text)) return text
      }
    } catch { /* try the next one */ }
  }
  throw new Error('not found')
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

type Hit = { doc: string, title: string, group: string, heading?: string, snippet: string, score: number }
type Section = { doc: string, heading?: string, text: string, lower: string }

const cleanHeading = (h: string) => h.replace(/\{[#.][^}]*\}\s*$/, '').replace(/[`*_]/g, '').trim()
const plainLine = (line: string) => line.replace(/`([^`]*)`/g, '$1').replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
  .replace(/^\s{0,3}(>|[-*+]|\d+\.)\s+/, '').replace(/\|/g, ' ').replace(/(^|\s)[*_~]+|[*_~]+(?=\s|$)/g, '$1')

// Every page, split at its headings, fetched once on the first search: the
// files the site already serves, so search covers all of every page and a
// hit opens at the section it is in.
let FULL_TEXT: Promise<Section[]> | null = null
function loadFullText(index: Index): Promise<Section[]> {
  if (FULL_TEXT) return FULL_TEXT
  FULL_TEXT = Promise.all(Object.keys(index.pages).map(async (doc) => {
    let md = ''
    try { md = await loadPage(doc, index) } catch { return [] as Section[] }
    const out: Section[] = []
    let heading: string | undefined
    let buf: string[] = []
    let fence = false
    const flush = () => {
      const text = buf.join(' ').replace(/\s+/g, ' ').trim()
      if (text) out.push({ doc, heading, text, lower: text.toLowerCase() })
      buf = []
    }
    for (const line of md.split('\n')) {
      if (/^\s*(```|~~~)/.test(line)) { fence = !fence; continue }
      const m = !fence && /^(#{1,4})\s+(.+?)\s*#*\s*$/.exec(line)
      if (m) { flush(); heading = m[1].length === 1 ? undefined : cleanHeading(m[2]); continue }
      buf.push(fence ? line : plainLine(line))
    }
    flush()
    return out
  })).then((all) => all.flat())
  return FULL_TEXT
}

function search(index: Index, query: string, sections: Section[] = []): Hit[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean)
  if (!words.length) return []
  const groupTitle = Object.fromEntries(index.groups.map((g) => [g.id, g.title]))
  const hits: Hit[] = []
  for (const [doc, info] of Object.entries(index.pages)) {
    const title = info.title.toLowerCase()
    const name = doc.toLowerCase()
    let score = 0
    let heading: string | undefined
    for (const w of words) {
      if (title === w || name === w) score += 60
      else if (title.startsWith(w) || name.startsWith(w)) score += 30
      else if (title.includes(w) || name.includes(w)) score += 18
    }
    const h = info.headings.find((x) => words.every((w) => x.text.toLowerCase().includes(w)))
    if (h) { score += 14; heading = h.text }
    const text = info.text.toLowerCase()
    const inText = words.every((w) => text.includes(w))
    if (inText) score += 6
    if (!score) continue
    let snippet = info.summary
    if (inText && !heading) {
      const at = text.indexOf(words[0])
      snippet = (at > 60 ? '…' : '') + info.text.slice(Math.max(0, at - 60), at + 120) + '…'
    }
    hits.push({ doc, title: info.title, group: groupTitle[info.group || ''] || '', heading, snippet, score })
  }
  // Sections of the full text: a page found only in its body, or the
  // section of a page found by its title.
  const best = new Map<string, Section>()
  for (const sec of sections) {
    if (!words.every((w) => sec.lower.includes(w))) continue
    const held = best.get(sec.doc)
    const here = sec.heading && words.some((w) => sec.heading!.toLowerCase().includes(w)) ? 2 : 1
    const there = held?.heading && words.some((w) => held.heading!.toLowerCase().includes(w)) ? 2 : held ? 1 : 0
    if (here > there) best.set(sec.doc, sec)
  }
  for (const [doc, sec] of best) {
    const info = index.pages[doc]
    if (!info) continue
    const at = sec.lower.indexOf(words[0])
    const snippet = (at > 70 ? '…' : '') + sec.text.slice(Math.max(0, at - 70), at + 130) + '…'
    const known = hits.find((h) => h.doc === doc)
    if (known) {
      if (!known.heading && sec.heading) { known.heading = sec.heading; known.snippet = snippet }
      known.score += 4
      continue
    }
    hits.push({ doc, title: info.title, group: groupTitle[info.group || ''] || '', heading: sec.heading, snippet,
                score: 5 + (sec.heading ? 2 : 0) })
  }
  return hits.sort((a, b) => b.score - a.score).slice(0, 14)
}

function Highlighted({ text, words }: { text: string, words: string[] }) {
  if (!words.length) return <>{text}</>
  const re = new RegExp(`(${words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'ig')
  return <>{text.split(re).map((part, i) => (i % 2 ? <mark key={i}>{part}</mark> : <span key={i}>{part}</span>))}</>
}

function SearchDialog({ index, onClose, onGo }: { index: Index, onClose: () => void, onGo: (doc: string, heading?: string) => void }) {
  const [query, setQuery] = useState('')
  const [active, setActive] = useState(0)
  const [sections, setSections] = useState<Section[]>([])
  const [reading, setReading] = useState(true)
  useEffect(() => {
    let live = true
    loadFullText(index).then((s) => { if (live) { setSections(s); setReading(false) } })
    return () => { live = false }
  }, [index])
  const hits = useMemo(() => search(index, query, sections), [index, query, sections])
  const words = query.toLowerCase().split(/\s+/).filter(Boolean)
  const input = useRef<HTMLInputElement>(null)
  useEffect(() => { input.current?.focus() }, [])
  useEffect(() => { setActive(0) }, [query])
  const go = (hit?: Hit) => { if (hit) { onGo(hit.doc, hit.heading); onClose() } }
  return (
    <div className="docs-search-backdrop" role="presentation" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div className="docs-search" role="dialog" aria-modal="true" aria-label="Search the documentation">
        <div className="docs-search-input">
          <span aria-hidden="true">⌕</span>
          <input ref={input} value={query} placeholder="Search the docs: a command, a setting, an error code…"
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Escape') onClose()
              else if (e.key === 'ArrowDown') { e.preventDefault(); setActive((a) => Math.min(a + 1, hits.length - 1)) }
              else if (e.key === 'ArrowUp') { e.preventDefault(); setActive((a) => Math.max(a - 1, 0)) }
              else if (e.key === 'Enter') go(hits[active])
            }} />
          <kbd>esc</kbd>
        </div>
        <ul className="docs-search-results" role="listbox">
          {hits.map((hit, i) => (
            <li key={hit.doc + (hit.heading || '')} role="option" aria-selected={i === active}
              className={i === active ? 'active' : ''} onMouseEnter={() => setActive(i)} onClick={() => go(hit)}>
              <div className="hit-title">
                <Highlighted text={hit.title} words={words} />
                {hit.heading && <span className="hit-heading"> › <Highlighted text={hit.heading} words={words} /></span>}
              </div>
              <div className="hit-group">{hit.group}</div>
              {hit.snippet && <div className="hit-snippet"><Highlighted text={hit.snippet} words={words} /></div>}
            </li>
          ))}
          {query && !hits.length && <li className="empty">Nothing matches “{query}”.</li>}
          {!query && <li className="empty">Titles, headings and the full text of every page. ↑ ↓ to choose, Enter to open.</li>}
          {query && reading && <li className="empty">Reading every page for the full-text results…</li>}
        </ul>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// The article: render, then make it behave like a docs page
// ---------------------------------------------------------------------------

const CALLOUTS: Record<string, string> = { note: 'Note', tip: 'Tip', important: 'Important', warning: 'Warning', caution: 'Caution' }

function enhance(root: HTMLElement, pages: Record<string, PageInfo>) {
  // Code: highlighted, with a copy button and the language shown.
  root.querySelectorAll('pre > code').forEach((code) => {
    const pre = code.parentElement as HTMLElement
    const lang = (/language-([\w+-]+)/.exec(code.className) || [])[1] || ''
    const text = code.textContent || ''
    if (lang) code.innerHTML = highlight(text, lang.toLowerCase())
    pre.classList.add('code-block')
    const bar = document.createElement('div')
    bar.className = 'code-bar'
    const label = document.createElement('span')
    label.textContent = lang || 'text'
    const copy = document.createElement('button')
    copy.type = 'button'
    copy.className = 'code-copy'
    copy.textContent = 'copy'
    copy.dataset.copy = text
    bar.append(label, copy)
    pre.prepend(bar)
  })
  // Callouts: GitHub's > [!NOTE], and quotes that open with **Note**.
  root.querySelectorAll('blockquote').forEach((quote) => {
    const first = quote.querySelector('p')
    if (!first) return
    const m = /^\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*/i.exec(first.innerHTML)
    const strong = first.firstElementChild
    let kind = ''
    if (m) {
      kind = m[1].toLowerCase()
      first.innerHTML = first.innerHTML.slice(m[0].length).replace(/^<br>\s*/, '')
    } else if (strong && strong.tagName === 'STRONG' && first.innerHTML.startsWith('<strong>')) {
      const word = (strong.textContent || '').replace(/[:.]$/, '').trim().toLowerCase()
      if (word in CALLOUTS) kind = word
    }
    if (!kind) return
    quote.classList.add('callout', `callout-${kind}`)
    const title = document.createElement('div')
    title.className = 'callout-title'
    title.textContent = CALLOUTS[kind]
    quote.prepend(title)
  })
  // Headings: a link to themselves.
  root.querySelectorAll('h2[id], h3[id], h4[id]').forEach((h) => {
    const a = document.createElement('a')
    a.className = 'heading-anchor'
    a.setAttribute('aria-label', 'Link to this section')
    a.dataset.anchor = h.id.replace(/^user-content-/, '')
    a.textContent = '#'
    h.append(a)
  })
  // Links: other wiki pages stay in the site; files elsewhere in the repo
  // open on GitHub; the rest open in a new tab.
  root.querySelectorAll('a[href]').forEach((a) => {
    const href = a.getAttribute('href') || ''
    if (a.classList.contains('heading-anchor')) return
    if (/^(https?:|mailto:)/i.test(href)) { a.setAttribute('target', '_blank'); a.setAttribute('rel', 'noreferrer'); return }
    if (href.startsWith('#')) { a.setAttribute('data-anchor', href.slice(1)); return }
    const [file, hash = ''] = href.split('#')
    const doc = /^(?:\.\/)?([A-Za-z0-9._-]+)\.md$/.exec(file)
    if (doc && pages[doc[1]]) { a.setAttribute('href', docHref(doc[1], hash)); return }
    const resolved = new URL(file, 'https://x/wiki/').pathname.replace(/^\//, '')
    a.setAttribute('href', `${REPO}/blob/main/${resolved}${hash ? `#${hash}` : ''}`)
    a.setAttribute('target', '_blank'); a.setAttribute('rel', 'noreferrer')
  })
  root.querySelectorAll('img[src]').forEach((img) => {
    const src = img.getAttribute('src') || ''
    if (!/^(https?:|data:)/i.test(src)) img.setAttribute('src', `${RAW}/${new URL(src, 'https://x/wiki/').pathname.replace(/^\//, '')}`)
  })
}

function findHeading(root: HTMLElement, anchor: string): HTMLElement | null {
  const want = anchorKey(decodeURIComponent(anchor))
  const all = Array.from(root.querySelectorAll<HTMLElement>('h1[id], h2[id], h3[id], h4[id], h5[id], h6[id]'))
  return all.find((h) => anchorKey(h.id) === want)
    || all.find((h) => anchorKey((h.textContent || '').replace(/#$/, '').trim().replace(/\s+/g, '-')) === want)
    || null
}

type TocItem = { id: string, text: string, depth: number }

function Article({ doc, anchor, index, navigate }: { doc: string, anchor?: string, index: Index, navigate: (doc?: string, anchor?: string) => void }) {
  const [md, setMd] = useState<string | null>(null)
  const [failed, setFailed] = useState(false)
  const [toc, setToc] = useState<TocItem[]>([])
  const [current, setCurrent] = useState('')
  const ref = useRef<HTMLDivElement>(null)
  const info = index.pages[doc]

  useEffect(() => {
    let live = true
    setMd(null); setFailed(false); setToc([])
    loadPage(doc, index).then((text) => { if (live) setMd(text) }).catch(() => { if (live) setFailed(true) })
    return () => { live = false }
  }, [doc])

  const html = useMemo(() => {
    if (md === null) return ''
    // The page's own title is shown in the header; drop a leading H1.
    return renderMarkdown(md.replace(/^\s*#\s+[^\n]*\n/, ''))
  }, [md])

  useLayoutEffect(() => {
    const root = ref.current
    if (!root || !html) return
    enhance(root, index.pages)
    setToc(Array.from(root.querySelectorAll<HTMLElement>('h2[id], h3[id]')).map((h) => ({
      id: h.id, depth: h.tagName === 'H2' ? 2 : 3, text: (h.textContent || '').replace(/#$/, '').trim(),
    })))
  }, [html, index.pages])

  // Scroll to the anchor, or the top, once the page is on screen.
  useEffect(() => {
    const root = ref.current
    if (!root || !html) return
    const target = anchor ? findHeading(root, anchor) : null
    if (target) target.scrollIntoView({ block: 'start' })
    else window.scrollTo({ top: 0 })
  }, [html, anchor])

  // Which section is being read.
  useEffect(() => {
    const root = ref.current
    if (!root || !toc.length) return
    const seen = new Map<string, boolean>()
    const observer = new IntersectionObserver((entries) => {
      for (const e of entries) seen.set(e.target.id, e.isIntersecting)
      const first = toc.find((t) => seen.get(t.id))
      if (first) setCurrent(first.id)
    }, { rootMargin: '-80px 0px -65% 0px' })
    toc.forEach((t) => { const el = document.getElementById(t.id); if (el) observer.observe(el) })
    return () => observer.disconnect()
  }, [toc])

  const onClick = useCallback((e: React.MouseEvent) => {
    const el = (e.target as HTMLElement).closest('[data-copy], [data-anchor]') as HTMLElement | null
    if (!el) return
    if (el.dataset.copy !== undefined) {
      navigator.clipboard?.writeText(el.dataset.copy)
      el.textContent = 'copied'
      setTimeout(() => { el.textContent = 'copy' }, 1400)
      return
    }
    if (el.dataset.anchor !== undefined) { e.preventDefault(); navigate(doc, el.dataset.anchor) }
  }, [doc, navigate])

  const flat = index.groups.flatMap((g) => g.pages)
  const at = flat.indexOf(doc)
  const prev = at > 0 ? flat[at - 1] : undefined
  const next = at >= 0 && at < flat.length - 1 ? flat[at + 1] : undefined
  const group = index.groups.find((g) => g.id === info?.group)

  return (
    <div className="docs-page">
      <div className="docs-article">
        <nav className="docs-crumbs" aria-label="Breadcrumb">
          <a href={docHref()}>Docs</a>
          {group && <><span>/</span><a href={docHref(group.pages[0])}>{group.title}</a></>}
          <span>/</span><span aria-current="page">{label(index, doc)}</span>
        </nav>
        <header className="docs-head">
          <h1>{info?.title || doc.replace(/-/g, ' ')}</h1>
          <div className="docs-meta">
            {info?.words ? <span>{Math.max(1, Math.round(info.words / 220))} min read</span> : null}
            <a href={`${REPO}/blob/main/${info?.sourcePath || info?.path || `wiki/${doc}.md`}`} target="_blank" rel="noreferrer">View source</a>
            <a href={`${REPO}/edit/main/${info?.sourcePath || `wiki/${doc}.md`}`} target="_blank" rel="noreferrer">Edit on GitHub</a>
          </div>
        </header>
        {failed && (
          <div className="callout callout-warning">
            <div className="callout-title">Could not load this page</div>
            <p>It may have been renamed. <a href={`${REPO}/tree/main/wiki`} target="_blank" rel="noreferrer">The wiki on GitHub</a> has every page.</p>
          </div>
        )}
        {md === null && !failed && <div className="docs-loading" aria-live="polite"><span className="spin-slow">◌</span> Loading…</div>}
        {/* HyperMarkdown escapes everything it did not make itself. */}
        <div ref={ref} className="md-body" onClick={onClick} dangerouslySetInnerHTML={{ __html: html }} />
        {md !== null && (
          <nav className="docs-pager" aria-label="Previous and next page">
            {prev ? <a className="pager-card" href={docHref(prev)}><span>Previous</span><strong>← {label(index, prev)}</strong></a> : <span />}
            {next ? <a className="pager-card next" href={docHref(next)}><span>Next</span><strong>{label(index, next)} →</strong></a> : <span />}
          </nav>
        )}
      </div>
      <aside className="docs-toc" aria-label="On this page">
        {toc.length > 0 && <>
          <div className="docs-toc-title">On this page</div>
          <ul>
            {toc.map((t) => (
              <li key={t.id} className={`depth-${t.depth}${current === t.id ? ' active' : ''}`}>
                <a href={docHref(doc, t.id.replace(/^user-content-/, ''))}>{t.text}</a>
              </li>
            ))}
          </ul>
          <button type="button" className="docs-top" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}>↑ Back to top</button>
        </>}
      </aside>
    </div>
  )
}

// ---------------------------------------------------------------------------
// The landing page and the sidebar
// ---------------------------------------------------------------------------

const QUICK = [
  ['Home', 'Start here'], ['CLI', 'Every command'], ['T1-API', 'The server'], ['Error-Codes', 'Error codes'],
  ['Quantization', 'Quantise a model'], ['Training', 'Train one'],
]

function systemHref(name: string) {
  const moduleSlug = name.replace(/^hypernix\\./, '').replace(/_/g, '-')
  // Keep the historical core URL working; it redirects to hyped-pro-core.
  const slug = moduleSlug === 'core' ? 'hypernix.core' : moduleSlug
  return `./subsystems/${slug}/system_index.html`
}

const GROUP_ICONS: Record<string, string> = {
  'Data': '◈', 'Training': '⌁', 'Inference & chat': '✳',
  'Evaluation': '◎', 'Runtime & memory': '▦', 'Ship & convert': '↗', 'Other': '◇',
}

function SubsystemIndex({ name, index }: { name: string, index: Index }) {
  const system = SUBSYSTEMS.find((item) => item.name === name)
  if (!system) return <div className="docs-article"><h1>System not found</h1><a href={docHref()}>Back to documentation</a></div>
  const group = subsystemGroup(name)
  const related = Object.entries(index.pages).filter(([key, info]) => {
    const haystack = `${key} ${info.title} ${info.summary} ${info.group || ''}`.toLowerCase()
    const terms: Record<string, string[]> = {
      'Data': ['data', 'dataset', 'preprocess', 'quantiz'],
      'Training': ['train', 'optimizer', 'pressure', 'brewer', 'checkpoint'],
      'Inference & chat': ['chat', 'inference', 'model', 'oven', 't1-api'],
      'Evaluation': ['evaluat', 'judge', 'score', 'range'],
      'Runtime & memory': ['memory', 'runtime', 'freezer', 'device', 'hardware'],
      'Ship & convert': ['convert', 'upload', 'download', 'gguf', 'quantiz'],
    }
    return (terms[group] || []).some((term) => haystack.includes(term))
  }).slice(0, 6)
  const apiModule = MODULES.find((item) => item.name === name)
  return (
    <div className="docs-article subsystem-article">
      <nav className="docs-crumbs" aria-label="Breadcrumb">
        <a href={docHref()}>Docs</a><span>/</span><a href={docHref()}>{group}</a><span>/</span>
        <span aria-current="page">{name}</span>
      </nav>
      <header className="docs-head">
        <div className="eyebrow" style={{ color: 'var(--accent-text)', marginBottom: 10 }}>{GROUP_ICONS[group] || '◇'} {group} system</div>
        <h1><code>{name}</code></h1>
        <p style={{ color: 'var(--text-dim)', fontSize: 16, lineHeight: 1.75, maxWidth: 760 }}>{system.desc}</p>
        <div className="docs-meta"><span>Python subsystem</span>
          <a href="https://github.com/trail-b1az3r/HyperNix-pip/tree/main/src/hypernix" target="_blank" rel="noreferrer">Browse source</a>
          <a href="#/api">API reference</a>
        </div>
      </header>
      <section className="docs-system-overview">
        <div className="eyebrow">At a glance</div><h2>What this system does</h2>
        <p>{system.desc}</p>
        <div className="docs-system-tags"><span>hypernix package</span><span>{group}</span><span>Module: {name}</span></div>
      </section>
      <section className="docs-system-explore">
        <div className="eyebrow">API surface</div><h2>Public functions and classes</h2>
        {apiModule?.fns?.length ? <div className="docs-system-api">
          {apiModule.fns.map((entry) => <div key={entry.fn} className="docs-system-api-row">
            <code>{entry.fn}</code><p>{entry.desc}</p>
          </div>)}
        </div> : <p>The generated API inventory does not currently list individual callables for this subsystem. Browse the full Python API reference or source tree for implementation details.</p>}
      </section>
      <section className="docs-system-explore">
        <div className="eyebrow">Explore</div><h2>Guides and references</h2>
        <p>Continue with documentation related to {name}. The API reference covers public Python entry points and CLI commands.</p>
        <div className="docs-system-grid">
          <a href="#/api" className="docs-system-link"><strong>Python API reference <span>↗</span></strong><small>Browse public modules, functions and source locations</small></a>
          {related.map(([key, info]) => <a key={key} href={docHref(key)} className="docs-system-link">
            <strong>{info.title} <span>→</span></strong><small>{info.summary || 'Open this guide in the documentation'}</small>
          </a>)}
          <a href="https://github.com/trail-b1az3r/HyperNix-pip/tree/main/src/hypernix" target="_blank" rel="noreferrer" className="docs-system-link">
            <strong>Source tree <span>↗</span></strong><small>Inspect the implementation in the HyperNix repository</small>
          </a>
        </div>
      </section>
      <section className="docs-system-related">
        <div className="eyebrow">More in {group}</div><h2>Related systems</h2>
        <div className="docs-system-grid">
          {SUBSYSTEMS.filter((item) => subsystemGroup(item.name) === group && item.name !== name).slice(0, 8).map((item) =>
            <a key={item.name} href={systemHref(item.name)} className="docs-system-link compact">
              <strong>{item.name} <span>→</span></strong><small>{item.desc}</small>
            </a>)}
        </div>
      </section>
      <nav className="docs-pager" aria-label="Documentation navigation">
        <a className="pager-card" href={docHref()}><span>Documentation</span><strong>← All systems and guides</strong></a>
        <a className="pager-card next" href="#/api"><span>Reference</span><strong>Python API →</strong></a>
      </nav>
    </div>
  )
}

function Landing({ index, openSearch }: { index: Index, openSearch: () => void }) {
  return (
    <div className="docs-landing anim-fade">
      <div className="docs-hero">
        <div className="eyebrow" style={{ color: 'var(--accent-text)' }}>Documentation</div>
        <h1>HyperNix docs</h1>
        <p>Guides and reference for every part of HyperNix — the CLI, the T1 server and HyperLink, the architectures, quantisation and training. {Object.keys(index.pages).length} pages, straight from the repository's wiki.</p>
        <button type="button" className="docs-hero-search" onClick={openSearch}>
          <span>⌕ Search the docs</span><kbd>Ctrl</kbd><kbd>K</kbd>
        </button>
        <div className="docs-quick">
          {QUICK.filter(([p]) => index.pages[p]).map(([p, label]) => <a key={p} href={docHref(p)}>{label}</a>)}
        </div>
      </div>
      <section className="docs-system-directory" aria-labelledby="docs-systems-title">
        <div className="docs-directory-heading"><div>
          <div className="eyebrow" style={{ color: 'var(--accent-text)' }}>HyperNix ecosystem</div>
          <h2 id="docs-systems-title">Explore the systems</h2>
          <p>Choose a subsystem to open its dedicated index, overview, related guides and source references.</p>
        </div><span className="docs-system-count">{SUBSYSTEMS.length} systems</span></div>
        {GROUP_ORDER.map((group) => {
          const systems = SUBSYSTEMS.filter((item) => subsystemGroup(item.name) === group)
          return <section key={group} className="docs-system-group">
            <div className="docs-system-group-title"><span>{GROUP_ICONS[group] || '◇'}</span><h3>{group}</h3><small>{systems.length} systems</small></div>
            <div className="docs-system-grid">{systems.map((item) => <a key={item.name} href={systemHref(item.name)} className="docs-system-card">
              <div className="docs-system-card-top"><code>{item.name}</code><span aria-hidden="true">↗</span></div>
              <p>{item.desc}</p><small>Open subsystem index →</small>
            </a>)}</div>
          </section>
        })}
      </section>
      <div className="docs-groups">
        {index.groups.map((g) => (
          <section key={g.id} className="docs-group-card lift-card">
            <h2><a href={docHref(g.pages[0])}>{g.title}</a></h2>
            {g.blurb && <p>{g.blurb}</p>}
            <ul>
              {g.pages.slice(0, 6).map((p) => <li key={p}><a href={docHref(p)}>{label(index, p)}</a></li>)}
              {g.pages.length > 6 && <li className="more"><a href={docHref(g.pages[6])}>+ {g.pages.length - 6} more</a></li>}
            </ul>
          </section>
        ))}
      </div>
    </div>
  )
}

function Sidebar({ index, doc, open, onPick }: { index: Index, doc?: string, open: boolean, onPick: () => void }) {
  const [filter, setFilter] = useState('')
  const [closed, setClosed] = useState<Record<string, boolean>>({})
  const f = filter.trim().toLowerCase()
  return (
    <aside className={`docs-side${open ? ' open' : ''}`} aria-label="Documentation">
      <input className="docs-filter" value={filter} onChange={(e) => setFilter(e.target.value)}
        placeholder="Filter pages" aria-label="Filter pages" />
      <a className={`docs-side-home${!doc ? ' active' : ''}`} href={docHref()} onClick={onPick}>Overview</a>
      {index.groups.map((g) => {
        const pages = g.pages.filter((p) => !f || p.toLowerCase().includes(f) || (index.pages[p]?.title || '').toLowerCase().includes(f))
        if (!pages.length) return null
        const shut = !f && closed[g.id] && !(doc && g.pages.includes(doc))
        return (
          <div key={g.id} className="docs-side-group">
            <button type="button" className="docs-side-title" aria-expanded={!shut}
              onClick={() => setClosed((c) => ({ ...c, [g.id]: !shut }))}>
              <span className={`caret${shut ? '' : ' open'}`}>›</span>{g.title}
            </button>
            {!shut && <ul>
              {pages.map((p) => (
                <li key={p}><a className={p === doc ? 'active' : ''} aria-current={p === doc ? 'page' : undefined}
                  href={docHref(p)} onClick={onPick}>{label(index, p)}</a></li>
              ))}
            </ul>}
          </div>
        )
      })}
      <div className="docs-side-group">
        <div className="docs-side-title">Systems</div>
        {GROUP_ORDER.map((group) => {
          const systems = SUBSYSTEMS.filter((item) => subsystemGroup(item.name) === group)
            .filter((item) => !f || item.name.toLowerCase().includes(f) || item.desc.toLowerCase().includes(f))
          if (!systems.length) return null
          return <div key={group}>
            <div className="docs-side-title">{group}</div>
            <ul>{systems.map((item) => <li key={item.name}>
              <a href={systemHref(item.name)} onClick={onPick}>{item.name.replace('hypernix.', '')}</a>
            </li>)}</ul>
          </div>
        })}
      </div>
    </aside>
  )
}

export function DocsPage({ route = {}, navigate }: { route?: Route, navigate?: (doc?: string, anchor?: string) => void }) {
  const [index] = useIndex()
  const [searching, setSearching] = useState(false)
  const [sideOpen, setSideOpen] = useState(false)
  const go = useCallback((doc?: string, anchor?: string) => {
    if (navigate) navigate(doc, anchor)
    else window.location.hash = docHref(doc, anchor)
  }, [navigate])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = /input|textarea|select/i.test((e.target as HTMLElement)?.tagName || '')
      if ((e.key === 'k' && (e.ctrlKey || e.metaKey)) || (e.key === '/' && !typing)) {
        e.preventDefault(); setSearching(true)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const doc = route.doc && index.pages[route.doc] ? route.doc : route.doc
  return (
    <div className="docs-shell">
      <style>{`
        .docs-system-directory{margin:34px 0 48px}
        .docs-directory-heading{display:flex;align-items:flex-end;justify-content:space-between;gap:18px;margin-bottom:22px}
        .docs-directory-heading h2{margin:7px 0;color:var(--text);font-size:clamp(23px,3vw,31px);letter-spacing:-.035em}
        .docs-directory-heading p{margin:0;color:var(--text-dim);font-size:13px;line-height:1.7;max-width:620px}
        .docs-system-count{white-space:nowrap;border:1px solid var(--border);border-radius:6px;padding:6px 10px;color:var(--text-muted);font:11px var(--font-mono)}
        .docs-system-group{margin:28px 0}
        .docs-system-group-title{display:flex;align-items:center;gap:10px;margin-bottom:12px}
        .docs-system-group-title>span{color:var(--accent);font-size:18px}
        .docs-system-group-title h3{margin:0;color:var(--text);font-size:16px}
        .docs-system-group-title small{color:var(--text-faint);font:10px var(--font-mono)}
        .docs-system-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(245px,100%),1fr));gap:10px}
        .docs-system-card,.docs-system-link{display:block;min-width:0;text-decoration:none;border:1px solid var(--border-strong);border-radius:9px;background:var(--surface-1);padding:14px 15px;transition:border-color .18s,transform .18s,background .18s}
        .docs-system-card:hover,.docs-system-link:hover{border-color:var(--accent);background:var(--surface-2);transform:translateY(-2px)}
        .docs-system-card-top{display:flex;justify-content:space-between;gap:8px;align-items:center}
        .docs-system-card-top code{color:var(--accent-text);font-size:12px;overflow-wrap:anywhere}
        .docs-system-card-top span,.docs-system-link strong span{color:var(--accent);flex-shrink:0}
        .docs-system-card p{margin:10px 0 12px;color:var(--text-dim);font-size:12px;line-height:1.65}
        .docs-system-card>small{color:var(--accent-text);font-size:10px}
        .docs-system-overview{margin:34px 0 30px;padding:22px 24px;border:1px solid var(--border-strong);border-radius:12px;background:var(--surface-1)}
        .docs-system-overview h2,.docs-system-explore h2,.docs-system-related h2{margin:8px 0;color:var(--text);font-size:21px}
        .docs-system-overview p,.docs-system-explore>p{color:var(--text-dim);line-height:1.8;font-size:14px}
        .docs-system-tags{display:flex;flex-wrap:wrap;gap:8px;margin-top:17px}
        .docs-system-tags span{border:1px solid var(--border);border-radius:6px;padding:5px 9px;color:var(--text-muted);font:10px var(--font-mono);overflow-wrap:anywhere}
.docs-system-api{border:1px solid var(--border);border-radius:9px;overflow:hidden;margin-top:14px}
        .docs-system-api-row{padding:13px 15px;border-bottom:1px solid var(--border)}
        .docs-system-api-row:last-child{border-bottom:0}
        .docs-system-api-row code{color:var(--accent-text);font-size:12px;overflow-wrap:anywhere}
        .docs-system-api-row p{margin:6px 0 0;color:var(--text-dim);font-size:12px;line-height:1.65}
        .docs-system-explore,.docs-system-related{margin:30px 0}
        .docs-system-link strong{display:flex;justify-content:space-between;gap:8px;color:var(--text);font-size:12px}
        .docs-system-link small{display:block;margin-top:7px;color:var(--text-dim);font-size:11px;line-height:1.55}
        .docs-system-link.compact{padding:12px}
        .docs-system-related{border-top:1px solid var(--border);padding-top:22px}
        .subsystem-article .docs-head h1 code{font-size:clamp(25px,4vw,38px);overflow-wrap:anywhere}
        @media(max-width:640px){.docs-directory-heading{align-items:flex-start;flex-direction:column}.docs-system-card{padding:12px}}
      `}</style>
      <div className="docs-topbar">
        <button type="button" className="docs-menu" aria-expanded={sideOpen} onClick={() => setSideOpen(!sideOpen)}>☰ Pages</button>
        <button type="button" className="docs-searchbar" onClick={() => setSearching(true)}>
          <span>⌕ Search docs</span><span className="keys"><kbd>Ctrl</kbd><kbd>K</kbd></span>
        </button>
      </div>
      <div className="docs-layout">
        <Sidebar index={index} doc={doc} open={sideOpen} onPick={() => setSideOpen(false)} />
        <main className="docs-content">
          {doc?.startsWith('@system:') ? <SubsystemIndex name={doc.slice(8)} index={index} />
            : doc ? <Article key={doc} doc={doc} anchor={route.anchor} index={index} navigate={go} />
            : <Landing index={index} openSearch={() => setSearching(true)} />}
        </main>
      </div>
      {searching && <SearchDialog index={index} onClose={() => setSearching(false)}
        onGo={(d, heading) => go(d, heading ? cleanHeading(heading).toLowerCase().replace(/[^\p{L}\p{N}\s-]/gu, '').trim().replace(/\s+/g, '-') : undefined)} />}
    </div>
  )
}
