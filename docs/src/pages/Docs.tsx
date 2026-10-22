import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { WIKI_PAGES } from '../lib/data/wiki'
import { anchorKey, highlight, renderMarkdown } from '../lib/markdown'

// The documentation, laid out the way PyTorch's and Hyprland's are: groups
// down the left, the page in the middle, "On this page" down the right,
// breadcrumbs and search on top, previous/next at the bottom. Pages are the
// repository's wiki/*.md, bundled into the site at build time (see
// vite.config.js) with a GitHub fallback, and every page and heading has
// its own address: #/docs/<Page>/<heading>.

const REPO = 'https://github.com/trail-b1az3r/HyperNix-pip'
const RAW = 'https://raw.githubusercontent.com/trail-b1az3r/HyperNix-pip/main'

type Heading = { depth: number, text: string }
type PageInfo = { title: string, short?: string, group?: string, summary: string, headings: Heading[], text: string, words: number }
const label = (index: Index, doc: string) => index.pages[doc]?.short || index.pages[doc]?.title || doc.replace(/-/g, ' ')
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

async function loadPage(name: string): Promise<string> {
  for (const url of [`./wiki/${name}.md`, `${RAW}/wiki/${name}.md`]) {
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
    try { md = await loadPage(doc) } catch { return [] as Section[] }
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
    loadPage(doc).then((text) => { if (live) setMd(text) }).catch(() => { if (live) setFailed(true) })
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
            <a href={`${REPO}/blob/main/wiki/${doc}.md`} target="_blank" rel="noreferrer">View source</a>
            <a href={`${REPO}/edit/main/wiki/${doc}.md`} target="_blank" rel="noreferrer">Edit on GitHub</a>
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
      <div className="docs-topbar">
        <button type="button" className="docs-menu" aria-expanded={sideOpen} onClick={() => setSideOpen(!sideOpen)}>☰ Pages</button>
        <button type="button" className="docs-searchbar" onClick={() => setSearching(true)}>
          <span>⌕ Search docs</span><span className="keys"><kbd>Ctrl</kbd><kbd>K</kbd></span>
        </button>
      </div>
      <div className="docs-layout">
        <Sidebar index={index} doc={doc} open={sideOpen} onPick={() => setSideOpen(false)} />
        <main className="docs-content">
          {doc ? <Article key={doc} doc={doc} anchor={route.anchor} index={index} navigate={go} />
            : <Landing index={index} openSearch={() => setSearching(true)} />}
        </main>
      </div>
      {searching && <SearchDialog index={index} onClose={() => setSearching(false)}
        onGo={(d, heading) => go(d, heading ? cleanHeading(heading).toLowerCase().replace(/[^\p{L}\p{N}\s-]/gu, '').trim().replace(/\s+/g, '-') : undefined)} />}
    </div>
  )
}
