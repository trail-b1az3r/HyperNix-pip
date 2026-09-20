import { useEffect, useMemo, useState } from 'react'
import { CodeBlock, CountUp, PageHeading } from '../components/ui'

function Metric({ label, value }) {
  return (
    <div className="lift-card" style={{ background:'var(--surface-3)', border:'1px solid var(--border-strong)', borderRadius:10, padding:'14px 12px', textAlign:'center', cursor:'default' }}>
      <div style={{ fontSize:24, fontWeight:800, color:'var(--accent)' }}>
        {typeof value === 'number' ? <CountUp value={value} /> : value}
      </div>
      <div style={{ fontSize:11, color:'var(--text-faint)', marginTop:4 }}>{label}</div>
    </div>
  )
}

function Badge({ children, tone = 'normal' }) {
  const color = tone === 'warn' ? '#e8b04a' : tone === 'ok' ? '#5ad47a' : 'var(--accent)'
  return <span style={{ border:`1px solid ${color}44`, color, background:`${color}12`, borderRadius:4, padding:'2px 6px', fontSize:10.5, fontFamily:'var(--font-mono)' }}>{children}</span>
}

function SectionCard({ title, children, id }) {
  return (
    <section id={id} style={{ background:'var(--surface-3)', border:'1px solid var(--border-strong)', borderRadius:10, padding:20, marginBottom:16 }}>
      <h2 style={{ color:'var(--text)', fontSize:14, margin:'0 0 14px', letterSpacing:'-0.01em' }}>{title}</h2>
      {children}
    </section>
  )
}

function ApiMember({ item }) {
  const [open, setOpen] = useState(false)
  const hasMethods = item.kind === 'class' && item.methods && item.methods.length > 0
  return (
    <div style={{ border:'1px solid var(--border)', borderRadius:8, background:'var(--surface-1)', marginBottom:8 }}>
      <button onClick={() => setOpen(!open)} aria-expanded={open} style={{ width:'100%', background:'none', border:'none', color:'inherit', cursor: hasMethods || item.doc ? 'pointer' : 'default', padding:'11px 12px', textAlign:'left', fontFamily:'inherit' }}>
        <div style={{ display:'flex', gap:8, alignItems:'center', flexWrap:'wrap' }}>
          <Badge>{item.kind}</Badge>
          <code style={{ color:'var(--text)', fontSize:12.5 }}>{item.name}</code>
          <span style={{ color:'var(--text-faint)', fontSize:10 }}>L{item.line}</span>
          {hasMethods && <span style={{ marginLeft:'auto', color:'var(--accent)', fontSize:11 }}>{open ? 'collapse' : `${item.methods.length} methods`}</span>}
        </div>
        <code style={{ display:'block', color:'var(--accent)', fontSize:11.5, lineHeight:1.55, marginTop:6, whiteSpace:'pre-wrap', overflowWrap:'anywhere' }}>{item.signature}</code>
        {item.doc && <div style={{ color:'var(--text-muted)', fontSize:12, lineHeight:1.6, marginTop:6 }}>{item.doc}</div>}
      </button>
      {open && hasMethods && (
        <div style={{ borderTop:'1px solid var(--border)', padding:'10px 12px' }}>
          {item.methods.map(m => (
            <div key={`${m.name}-${m.line}`} style={{ padding:'9px 0', borderBottom:'1px solid var(--border)' }}>
              <div style={{ display:'flex', alignItems:'center', gap:8 }}>
                <code style={{ color:'var(--text)', fontSize:11.5 }}>{m.name}</code>
                <span style={{ color:'var(--text-faint)', fontSize:10 }}>L{m.line}</span>
              </div>
              <code style={{ display:'block', color:'var(--accent)', fontSize:11, lineHeight:1.55, marginTop:4, whiteSpace:'pre-wrap', overflowWrap:'anywhere' }}>{m.signature}</code>
              {m.doc && <div style={{ color:'var(--text-muted)', fontSize:11.5, lineHeight:1.55, marginTop:4 }}>{m.doc}</div>}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function ModuleView({ module }) {
  return (
    <div className="anim-fade-up">
      <div style={{ display:'flex', alignItems:'flex-start', justifyContent:'space-between', gap:12, flexWrap:'wrap', marginBottom:12 }}>
        <div>
          <div className="eyebrow" style={{ color:'var(--accent)', marginBottom:7 }}>{module.source}</div>
          <h2 style={{ color:'var(--text)', margin:0, fontSize:22, letterSpacing:'-0.02em' }}>{module.module}</h2>
          {module.summary && <p style={{ color:'var(--text-dim)', fontSize:13, lineHeight:1.65, maxWidth:'70ch', margin:'8px 0 0' }}>{module.summary}</p>}
        </div>
        <Badge>{module.line_count.toLocaleString()} non-blank lines</Badge>
      </div>

      {module.syntax_error && (
        <div style={{ background:'#160408', border:'1px solid #7a1a28', borderRadius:8, padding:12, color:'#d47070', fontSize:12 }}>
          Source could not be parsed: {module.syntax_error}
        </div>
      )}

      {module.required_modules?.length > 0 && (
        <SectionCard title="Required / detected modules">
          <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fit,minmax(220px,1fr))', gap:8 }}>
            {module.required_modules.map(d => (
              <div key={`${d.module}-${d.extra}`} style={{ background:'var(--surface-1)', border:'1px solid var(--border)', borderRadius:7, padding:'10px 11px' }}>
                <code style={{ color:'var(--text)', fontSize:11.5 }}>{d.module}</code>
                <div style={{ color:'var(--text-faint)', fontSize:10.5, marginTop:4 }}>extra: {d.extra} · {d.requirement}</div>
              </div>
            ))}
          </div>
        </SectionCard>
      )}

      {module.deprecations?.length > 0 && (
        <SectionCard title="Deprecation / warning behavior">
          {module.deprecations.map((d, i) => (
            <div key={`${d.name}-${i}`} style={{ background:'#1a1304', border:'1px solid #7a5a10', borderRadius:8, padding:'10px 12px', marginBottom:8 }}>
              <div style={{ display:'flex', flexWrap:'wrap', gap:7, alignItems:'center' }}>
                <Badge tone="warn">{d.name}</Badge>
                {d.since && <span style={{ color:'var(--text-faint)', fontSize:10.5 }}>since {d.since}</span>}
                {d.removed_in && <span style={{ color:'#e8b04a', fontSize:10.5 }}>removal target: {d.removed_in}</span>}
              </div>
              {d.instead && <div style={{ color:'var(--text)', fontSize:12, marginTop:6 }}>Replacement: <code style={{ color:'var(--accent)' }}>{d.instead}</code></div>}
              {d.extra && <div style={{ color:'var(--text-muted)', fontSize:11.5, lineHeight:1.55, marginTop:5 }}>{d.extra}</div>}
            </div>
          ))}
        </SectionCard>
      )}

      {module.errors?.length > 0 && (
        <SectionCard title="Errors surfaced by this module">
          <div style={{ display:'flex', flexWrap:'wrap', gap:7 }}>{module.errors.map(err => <Badge key={err} tone="warn">{err}</Badge>)}</div>
        </SectionCard>
      )}

      <SectionCard title={`Public API · ${module.api?.length || 0} entries`}>
        {module.api?.length ? module.api.map(item => <ApiMember key={`${item.kind}-${item.name}-${item.line}`} item={item} />) : <p style={{ color:'var(--text-faint)', fontSize:12 }}>No public top-level functions or classes were detected by the source parser.</p>}
      </SectionCard>

      {module.changes?.length > 0 ? (
        <SectionCard title="Recent repository changes affecting this file">
          {module.changes.map(c => (
            <div key={`${c.sha}-${c.date}`} style={{ display:'flex', gap:10, padding:'9px 0', borderBottom:'1px solid var(--border)', alignItems:'flex-start' }}>
              <code style={{ color:'var(--accent)', fontSize:10.5, flexShrink:0 }}>{c.sha}</code>
              <div style={{ flex:1 }}>
                <div style={{ color:'var(--text)', fontSize:11.5 }}>{c.subject}</div>
                <div style={{ color:'var(--text-faint)', fontSize:10.5, marginTop:3 }}>{c.date} · {c.author}</div>
                {c.api_changes?.length > 0 && (
                  <div style={{ marginTop:6, display:'flex', flexDirection:'column', gap:3 }}>
                    {c.api_changes.map((change, i) => (
                      <code key={`${change.kind}-${i}`} style={{ color:change.kind === 'added' ? '#5ad47a' : '#e8b04a', fontSize:10.5, whiteSpace:'pre-wrap', overflowWrap:'anywhere' }}>
                        {change.kind === 'added' ? '+' : '−'} {change.text}
                      </code>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
        </SectionCard>
      ) : (
        <SectionCard title="Change history">
          <p style={{ color:'var(--text-muted)', fontSize:12, lineHeight:1.6, margin:0 }}>
            This archive was generated without a local Git history. On GitHub, the hourly documentation workflow checks out full history and adds the recent commits affecting this module here.
          </p>
        </SectionCard>
      )}
    </div>
  )
}

function ExampleCard({ example }) {
  return (
    <div className="lift-card" style={{ background:'var(--surface-1)', border:'1px solid var(--border)', borderRadius:8, padding:14, marginBottom:10 }}>
      <div style={{ display:'flex', justifyContent:'space-between', gap:12, alignItems:'baseline', flexWrap:'wrap' }}>
        <h3 style={{ margin:0, color:'var(--text)', fontSize:13 }}>{example.title}</h3>
        <Badge>{example.module}</Badge>
      </div>
      <p style={{ color:'var(--text-muted)', fontSize:11.5, lineHeight:1.6, margin:'7px 0 8px' }}>{example.description}</p>
      <CodeBlock code={example.code} />
      <div style={{ color:'var(--text-faint)', fontSize:10.5, marginTop:6 }}>source: {example.source}</div>
    </div>
  )
}

function RouteList({ routes }) {
  return (
    <div style={{ overflowX:'auto' }}>
      <table style={{ width:'100%', borderCollapse:'collapse', fontSize:11.5 }}>
        <thead><tr>
          {['Method', 'Path', 'Handler', 'Response', 'Source'].map(h => <th key={h} style={{ textAlign:'left', padding:'8px 7px', borderBottom:'1px solid var(--border-strong)', color:'var(--text-faint)', fontSize:10 }}>{h}</th>)}
        </tr></thead>
        <tbody>{routes.map((r, i) => (
          <tr key={`${r.method}-${r.path}-${r.source}-${i}`}>
            <td style={{ padding:'8px 7px', borderBottom:'1px solid var(--border)', color:r.method === 'GET' ? '#5ad47a' : '#e8b04a', fontFamily:'var(--font-mono)', fontWeight:700 }}>{r.method}</td>
            <td style={{ padding:'8px 7px', borderBottom:'1px solid var(--border)' }}><code style={{ color:'var(--accent)' }}>{r.path}</code></td>
            <td style={{ padding:'8px 7px', borderBottom:'1px solid var(--border)', color:'var(--text)' }}>{r.handler}</td>
            <td style={{ padding:'8px 7px', borderBottom:'1px solid var(--border)', color:'var(--text-muted)' }}>{r.response_model || '—'}</td>
            <td style={{ padding:'8px 7px', borderBottom:'1px solid var(--border)', color:'var(--text-faint)' }}>{r.source}:{r.line}</td>
          </tr>
        ))}</tbody>
      </table>
    </div>
  )
}

export function ReferenceDocsPage({ dataUrl, kicker, title, lede, mode = 'deep' }) {
  const [data, setData] = useState(null)
  const [query, setQuery] = useState('')
  const [activeModule, setActiveModule] = useState(null)
  const [activeSection, setActiveSection] = useState('overview')

  useEffect(() => {
    fetch(dataUrl).then(r => { if (!r.ok) throw new Error(); return r.json() }).then(setData).catch(() => setData({ modules:[], routes:[], examples:[], errors:[], deprecations:[] }))
  }, [dataUrl])

  const modules = data?.modules || []
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return modules
    return modules.filter(m => {
      if (m.module.toLowerCase().includes(q) || m.source.toLowerCase().includes(q)) return true
      return (m.api || []).some(item => item.name.toLowerCase().includes(q) || item.signature.toLowerCase().includes(q))
    })
  }, [modules, query])

  const selected = activeModule ? modules.find(m => m.module === activeModule) : null
  const apiEntries = modules.reduce((n, m) => n + (m.api?.length || 0), 0)
  const deprecations = data?.deprecations || modules.flatMap(m => (m.deprecations || []).map(d => ({ ...d, module:m.module })))
  const errorCount = mode === 't1' ? (data?.errors || []).length : modules.reduce((n, m) => n + (m.errors?.length || 0), 0)

  return (
    <div className="anim-fade" style={{ maxWidth:1120, margin:'0 auto', padding:'90px 20px 60px' }}>
      <PageHeading kicker={kicker} title={title} lede={lede} />
      <div className="anim-stagger" style={{ display:'grid', gridTemplateColumns:'repeat(auto-fit,minmax(135px,1fr))', gap:10, marginBottom:20 }}>
        <Metric label="Modules" value={modules.length} />
        <Metric label="Public API entries" value={apiEntries} />
        {mode === 't1' && <Metric label="HTTP routes" value={(data?.routes || []).length} />}
        <Metric label="Errors" value={errorCount} />
        <Metric label="Deprecations / warnings" value={deprecations.length} />
      </div>

      <div className="reference-layout">
        <aside className="reference-sidebar" style={{ position:'sticky', top:72, maxHeight:'calc(100vh - 92px)', overflow:'auto' }}>
          <input value={query} onChange={e => setQuery(e.target.value)} placeholder="Search modules / symbols…" aria-label="Search API reference"
            style={{ width:'100%', boxSizing:'border-box', background:'var(--surface-3)', border:'1px solid var(--border-strong)', borderRadius:7, padding:'9px 10px', color:'var(--text)', fontFamily:'var(--font-mono)', fontSize:11.5, outline:'none', marginBottom:10 }} />
          <button onClick={() => { setActiveModule(null); setActiveSection('overview') }} style={{ width:'100%', textAlign:'left', background:activeSection === 'overview' && !activeModule ? 'var(--surface-2)' : 'transparent', border:'1px solid var(--border)', borderRadius:6, color:'var(--text)', padding:'7px 9px', cursor:'pointer', fontSize:11.5, marginBottom:7 }}>Overview</button>
          {mode === 't1' && <button onClick={() => { setActiveModule(null); setActiveSection('routes') }} style={{ width:'100%', textAlign:'left', background:activeSection === 'routes' ? 'var(--surface-2)' : 'transparent', border:'1px solid var(--border)', borderRadius:6, color:'var(--text)', padding:'7px 9px', cursor:'pointer', fontSize:11.5, marginBottom:7 }}>HTTP routes</button>}
          <div style={{ color:'var(--text-faint)', fontSize:10, letterSpacing:'.12em', textTransform:'uppercase', fontWeight:700, margin:'12px 0 7px' }}>Modules · {filtered.length}</div>
          {filtered.map(m => (
            <button key={m.module} onClick={() => { setActiveModule(m.module); setActiveSection('module') }} style={{ display:'block', width:'100%', textAlign:'left', background:activeModule === m.module ? 'var(--surface-2)' : 'transparent', border:'none', borderLeft:activeModule === m.module ? '2px solid var(--accent)' : '2px solid transparent', color:activeModule === m.module ? 'var(--accent-text)' : 'var(--text-dim)', padding:'5px 6px 5px 8px', cursor:'pointer', fontFamily:'var(--font-mono)', fontSize:10.5, lineHeight:1.45, overflowWrap:'anywhere' }}>
              {m.module}
            </button>
          ))}
        </aside>

        <div style={{ minWidth:0 }}>
          {!data ? (
            <SectionCard title="Loading"><p style={{ color:'var(--text-faint)', fontSize:12 }}>Loading generated API data…</p></SectionCard>
          ) : activeSection === 'routes' && mode === 't1' ? (
            <div className="anim-fade-up">
              <SectionCard title={`HTTP routes · ${data.routes.length}`}>
                <RouteList routes={data.routes} />
              </SectionCard>
              <SectionCard title="T1 server dependencies">
                <p style={{ color:'var(--text-muted)', fontSize:12, lineHeight:1.6, margin:'0 0 8px' }}>The server entry point and FastAPI route stack use the T1 server extra:</p>
                <CodeBlock code={data.server_extra} />
              </SectionCard>
            </div>
          ) : selected ? (
            <ModuleView module={selected} />
          ) : (
            <div className="anim-fade-up">
              <SectionCard title="What this page documents">
                <p style={{ color:'var(--text-muted)', fontSize:12.5, lineHeight:1.7, margin:0 }}>
                  This reference is generated from the checked-in source tree. It includes public top-level functions/classes, public class methods, detected required modules, surfaced errors, warnings/deprecations, and the recent Git commits that touched each file when repository history is available.
                </p>
              </SectionCard>

              {data.examples?.length > 0 && <SectionCard title="Real code uses"><div>{data.examples.map((e, i) => <ExampleCard key={`${e.title}-${i}`} example={e} />)}</div></SectionCard>}

              {deprecations.length > 0 && (
                <SectionCard title="Deprecation warnings and replacements">
                  {deprecations.map((d, i) => (
                    <div key={`${d.module}-${i}`} style={{ background:'#1a1304', border:'1px solid #7a5a10', borderRadius:8, padding:'10px 12px', marginBottom:8 }}>
                      <div style={{ display:'flex', gap:8, alignItems:'center', flexWrap:'wrap' }}><Badge tone="warn">{d.module}</Badge>{d.since && <span style={{ color:'var(--text-faint)', fontSize:10.5 }}>since {d.since}</span>}</div>
                      {d.instead && <div style={{ color:'var(--text)', fontSize:12, marginTop:6 }}>Use <code style={{ color:'var(--accent)' }}>{d.instead}</code> instead.</div>}
                      {d.extra && <div style={{ color:'var(--text-muted)', fontSize:11.5, lineHeight:1.55, marginTop:4 }}>{d.extra}</div>}
                    </div>
                  ))}
                </SectionCard>
              )}

              {mode === 't1' && <SectionCard title={`HTTP routes · ${data.routes.length}`}><RouteList routes={data.routes.slice(0, 35)} />{data.routes.length > 35 && <button onClick={() => setActiveSection('routes')} style={{ marginTop:10, background:'none', border:'none', color:'var(--accent)', cursor:'pointer', fontSize:11.5 }}>Show all routes →</button>}</SectionCard>}

              {mode === 't1' && data.errors?.length > 0 && (
                <SectionCard title="T1 error types">
                  <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fit,minmax(210px,1fr))', gap:8 }}>{data.errors.map((e, i) => <div key={`${e.name}-${i}`} style={{ background:'var(--surface-1)', border:'1px solid var(--border)', borderRadius:7, padding:'9px 10px' }}><code style={{ color:'var(--accent)', fontSize:11 }}>{e.name}</code><div style={{ color:'var(--text-faint)', fontSize:10, marginTop:4 }}>{e.module}{e.base ? ` · ${e.base}` : ''}</div></div>)}</div>
                </SectionCard>
              )}

              {data.history_available === false && <SectionCard title="Git-aware update history"><p style={{ color:'var(--text-muted)', fontSize:12, lineHeight:1.6, margin:0 }}>This uploaded archive does not include Git history. The hourly GitHub Action checks out full history and regenerates the same files, so future page entries can show which commit changed a module and what changed in that update.</p></SectionCard>}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
