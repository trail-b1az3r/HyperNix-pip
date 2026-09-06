import { useState, useEffect, useCallback } from 'react'
import { WIKI_PAGES } from '../lib/data/wiki'
import { PageHeading } from '../components/ui'

function getWikiFileAddedDate(name) {
  const path = `wiki/${name}.md`
  return fetch(`https://api.github.com/repos/trail-b1az3r/HyperNix-pip/commits?path=${encodeURIComponent(path)}&per_page=100`)
    .then(r => { if (!r.ok) throw new Error(); return r.json() })
    .then(commits => {
      if (!Array.isArray(commits) || !commits.length || commits.length >= 100) return null
      const oldest = commits[commits.length - 1]
      const dateStr = oldest && oldest.commit && (oldest.commit.committer?.date || oldest.commit.author?.date)
      return dateStr ? new Date(dateStr) : null
    })
    .catch(() => null)
}

function DocsPage() {
  const [activeWiki, setActiveWiki] = useState(null)
  const [wikiContent, setWikiContent] = useState('')
  const [loading, setLoading] = useState(false)
  const [wikiPages, setWikiPages] = useState(WIKI_PAGES)
  const [newPages, setNewPages] = useState(() => new Set())
  const [pagesSynced, setPagesSynced] = useState(false)

  // Stream the live /wiki folder from GitHub so newly-added doc pages show up
  // automatically without needing a code change here. Falls back to the
  // hardcoded WIKI_PAGES list (e.g. if rate-limited or offline).
  useEffect(() => {
    fetch('https://api.github.com/repos/trail-b1az3r/HyperNix-pip/contents/wiki')
      .then(r => { if (!r.ok) throw new Error(); return r.json() })
      .then(files => {
        const names = (Array.isArray(files) ? files : [])
          .filter(f => f.type === 'file' && f.name.endsWith('.md'))
          .map(f => f.name.replace(/\.md$/, ''))
        if (!names.length) return
        const rest = names.filter(n => n !== 'Home').sort((a, b) => a.localeCompare(b))
        const ordered = names.includes('Home') ? ['Home', ...rest] : rest
        setWikiPages(ordered)
        setPagesSynced(true)

        // Only pages missing from the static fallback list are even
        // candidates for "NEW" — check their actual first-commit date and
        // only badge the ones added within the last 2 days.
        const candidates = names.filter(n => !WIKI_PAGES.includes(n))
        if (!candidates.length) { setNewPages(new Set()); return }
        Promise.all(candidates.map(n => getWikiFileAddedDate(n).then(d => [n, d])))
          .then(results => {
            const TWO_DAYS_MS = 2 * 24 * 60 * 60 * 1000
            const now = Date.now()
            const stillNew = new Set()
            for (const [n, addedAt] of results) {
              if (addedAt && (now - addedAt.getTime()) < TWO_DAYS_MS) stillNew.add(n)
            }
            setNewPages(stillNew)
          })
          .catch(() => {})
      })
      .catch(() => {}) // keep static fallback list on failure
  }, [])

  const loadWiki = useCallback((name) => {
    setLoading(true); setActiveWiki(name)
    fetch(`https://raw.githubusercontent.com/trail-b1az3r/HyperNix-pip/main/wiki/${name}.md`)
      .then(r => { if (!r.ok) throw new Error(); return r.text() })
      .then(md => { setWikiContent(md); setLoading(false) })
      .catch(() => { setWikiContent(''); setLoading(false) })
  }, [])

  const renderMd = (md) => md.split('\n').map((line, i) => {
    if (line.startsWith('### ')) return <h3 key={i} style={{ color:'var(--text)', fontSize:15, margin:'18px 0 6px' }}>{line.slice(4)}</h3>
    if (line.startsWith('## '))  return <h2 key={i} style={{ color:'var(--text)', fontSize:19, margin:'26px 0 8px', borderBottom:'1px solid var(--border-strong)', paddingBottom:6 }}>{line.slice(3)}</h2>
    if (line.startsWith('# '))   return <h1 key={i} style={{ color:'var(--text)', fontSize:24, margin:'0 0 16px', fontWeight:800 }}>{line.slice(2)}</h1>
    if (line.startsWith('```'))  return null
    if (line.startsWith('- '))   return <li key={i} style={{ color:'var(--text-muted)', fontSize:13, lineHeight:1.7, marginLeft:16, marginBottom:2 }}>{line.slice(2)}</li>
    if (line.trim() === '')      return <div key={i} style={{ height:8 }} />
    return <p key={i} style={{ color:'var(--text-muted)', fontSize:13, lineHeight:1.7, margin:'4px 0' }}>{line}</p>
  })

  const [sidebarOpen, setSidebarOpen] = useState(false)
  return (
    <div style={{ maxWidth:1080, margin:'0 auto', padding:'90px 20px 60px' }}>
      <div style={{ display:'flex', gap:32, minHeight:'100vh', position:'relative' }}>
      <div style={{ width:180, flexShrink:0 }} className={!activeWiki ? 'docs-sidebar-browsing' : ''}>
        <div style={{ position:'sticky', top:72 }}>
          <button onClick={() => setSidebarOpen(!sidebarOpen)} className="press-btn mob-sidebar-btn"
            style={{ display:'none', background:'var(--surface-3)', border:'1px solid var(--border-strong)', borderRadius:6,
              color:'var(--accent)', padding:'6px 12px', fontSize:12, cursor:'pointer', marginBottom:10,
              width:'100%' }}>
            {sidebarOpen ? '▲ Hide pages' : '▼ Browse pages'}
          </button>
          <div className={sidebarOpen ? 'sidebar-open' : 'sidebar-closed'}>
          <div style={{ fontSize:10, color:'var(--text-faint)', textTransform:'uppercase',
            letterSpacing:'0.12em', marginBottom:14, fontWeight:700, display:'flex', alignItems:'center', gap:6 }}>
            Wiki Pages
            {pagesSynced && <span title="Synced live from GitHub /wiki" style={{ color:'#2ecc71', fontSize:9, textTransform:'none', letterSpacing:0, fontWeight:400 }}>● live</span>}
          </div>
          {wikiPages.map(p => (
            <button key={p} onClick={() => { loadWiki(p); setSidebarOpen(false) }} style={{
              display:'flex', alignItems:'center', gap:6, width:'100%', textAlign:'left', background:'none', border:'none',
              color: activeWiki === p ? 'var(--accent-text)' : 'var(--text-dim)', fontSize:12,
              padding:'5px 0 5px 10px', cursor:'pointer', fontFamily:'monospace',
              borderLeft: activeWiki === p ? '2px solid var(--accent)' : '2px solid transparent',
              transition:'all 0.15s', lineHeight:1.6, overflowWrap:'break-word', wordBreak:'break-all',
              transform: activeWiki === p ? 'translateX(2px)' : 'translateX(0)',
            }}>
              {p}
              {newPages.has(p) && <span style={{ fontSize:8, color:'#0a0a0a', background:'var(--accent)', borderRadius:3, padding:'1px 4px', fontFamily:'sans-serif', fontWeight:700, flexShrink:0 }}>NEW</span>}
            </button>
          ))}
          </div>
        </div>
      </div>
      <div style={{ flex:1, minWidth:0 }}>
        {!activeWiki ? (
          <div className="anim-fade">
            <PageHeading kicker="Wiki" title="Documentation"
              lede="Every page below is rendered straight from the repository's wiki/ directory — pick one from the sidebar or the grid." />
            <div className="anim-stagger" style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill,minmax(175px,1fr))', gap:10 }}>
              {wikiPages.map(p => (
                <button key={p} onClick={() => loadWiki(p)} className="lift-card" style={{
                  position:'relative', background:'var(--surface-3)', border:'1px solid var(--border-strong)', borderRadius:8,
                  padding:'14px 16px', textAlign:'left', cursor:'pointer', color:'var(--text)', fontSize:13, fontWeight:600,
                }}>
                  {newPages.has(p) && <span style={{ position:'absolute', top:8, right:8, fontSize:8, color:'#0a0a0a', background:'var(--accent)', borderRadius:3, padding:'1px 4px', fontFamily:'sans-serif', fontWeight:700 }}>NEW</span>}
                  {p.replace(/-/g, ' ')}
                  <div className="underline-grow" style={{ fontSize:11, color:'var(--text-faint)', marginTop:5, fontWeight:400, display:'block', width:'fit-content' }}>wiki ↗</div>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="anim-fade-up" key={activeWiki}>
            <button onClick={() => { setActiveWiki(null); setWikiContent('') }} className="press-btn" style={{
              background:'none', border:'none', color:'var(--accent)', cursor:'pointer', fontSize:13, marginBottom:20, padding:0
            }}>← Back</button>
            {loading
              ? <p style={{ color:'var(--text-faint)' }}>
                  <span className="spin-slow" style={{ display:'inline-block', marginRight:8 }}>◌</span>
                  Loading…
                </p>
              : wikiContent
                ? <div className="anim-fade">{renderMd(wikiContent)}</div>
                : <p style={{ color:'var(--text-dim)' }}>
                    Could not load.{' '}
                    <a href={`https://github.com/trail-b1az3r/HyperNix-pip/blob/main/wiki/${activeWiki}.md`}
                      target="_blank" rel="noreferrer" className="underline-grow" style={{ color:'var(--accent)' }}>View on GitHub ↗</a>
                  </p>
            }
          </div>
        )}
      </div>
      </div>
    </div>
  )
}

export { DocsPage }
