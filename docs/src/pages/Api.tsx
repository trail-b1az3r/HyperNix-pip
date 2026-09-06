import { useState } from 'react'
import { CLI_COMMANDS, COMPANION_SCRIPTS, MODULES } from '../lib/data/api-reference'
import { PageHeading, CodeBlock, CopyButton } from '../components/ui'

function ApiPage() {
  const HYPERNIX_REPO_BRANCH = 'main'
  const HYPERNIX_REPO_URL = `https://github.com/trail-b1az3r/HyperNix-pip`
  const srcUrl = (path) => `${HYPERNIX_REPO_URL}/blob/${HYPERNIX_REPO_BRANCH}/${path}`

  const [tab, setTab] = useState('cli')
  const [openMod, setOpenMod] = useState(null)
  const [search, setSearch] = useState('')

  const filteredCli = CLI_COMMANDS.filter(c =>
    c.cmd.includes(search) || c.desc.toLowerCase().includes(search.toLowerCase()))
  const filteredMods = MODULES.map(m => ({
    ...m,
    fns: m.fns.filter(f => f.fn.toLowerCase().includes(search.toLowerCase()) || f.desc.toLowerCase().includes(search.toLowerCase()))
  })).filter(m => m.fns.length > 0 || m.name.toLowerCase().includes(search.toLowerCase()))

  return (
    <div className="anim-fade" style={{ maxWidth:860, margin:'0 auto', padding:'90px 20px 60px' }}>
      <PageHeading kicker="Reference" title="API Reference"
        lede="Every CLI subcommand and every public Python entry point, cross-checked against the installed source." />
      <p className="anim-fade-up" style={{ color:'var(--text-faint)', fontSize:12, marginBottom:24, animationDelay:'0.08s' }}>
        Sourced from{' '}
        <a href={HYPERNIX_REPO_URL} target="_blank" rel="noreferrer" className="underline-grow" style={{ color:'#4a9eff' }}>
          trail-b1az3r/HyperNix-pip
        </a>{' '}
        — every module links to its file under <code style={{ color:'var(--text-faint)' }}>src/hypernix/</code>
      </p>
      <input value={search} onChange={e => setSearch(e.target.value)}
        placeholder="Search commands or functions…"
        style={{ width:'100%', boxSizing:'border-box', background:'var(--surface-3)', border:'1px solid var(--border-strong)',
          borderRadius:8, padding:'10px 14px', color:'var(--text)', fontSize:14, marginBottom:24, outline:'none' }} />
      <div style={{ display:'flex', borderBottom:'1px solid var(--border-strong)', marginBottom:28 }}>
        {[['cli','CLI Commands'],['python','Python Modules']].map(([k,l]) => (
          <button key={k} onClick={() => setTab(k)} style={{
            background:'none', border:'none',
            borderBottom: tab === k ? '2px solid var(--accent)' : '2px solid transparent',
            color: tab === k ? 'var(--accent-text)' : 'var(--text-dim)', padding:'8px 20px', cursor:'pointer',
            fontSize:14, fontWeight: tab === k ? 700 : 400, marginBottom:-1,
            transition:'border-color 0.22s ease, color 0.18s ease',
          }}>{l}</button>
        ))}
      </div>
      {tab === 'cli' && (
        <div className="anim-stagger">
          {filteredCli.length === 0 && <p style={{ color:'var(--text-faint)' }}>No commands match.</p>}
          {filteredCli.map(c => (
            <div key={c.cmd} className="lift-card" style={{ background:'var(--surface-3)', border:'1px solid var(--border-strong)',
              borderRadius:10, padding:'18px 20px', marginBottom:12 }}>
              <div style={{ display:'flex', flexWrap:'wrap', gap:8, marginBottom:8, alignItems:'baseline' }}>
                <code style={{ color:'var(--accent)', fontWeight:700, fontSize:14, flexShrink:0 }}>{c.cmd}</code>
                <code style={{ color:'var(--text-faint)', fontSize:12, overflowWrap:'break-word', wordBreak:'break-all',
                  flex:1, minWidth:0 }}>{c.args}</code>
              </div>
              <p style={{ color:'var(--text-dim)', fontSize:13, margin:'0 0 10px' }}>{c.desc}</p>
              <CodeBlock code={c.cmd + ' ' + c.args.replace(/\[.*?\]/g,'').trim()} />
            </div>
          ))}
          {search === '' && (
            <>
              <h3 style={{ color:'var(--text)', fontSize:16, margin:'32px 0 6px' }}>Companion console scripts</h3>
              <p style={{ color:'var(--text-dim)', fontSize:13, marginBottom:16 }}>
                Installed alongside <code style={{ color:'var(--text-faint)' }}>hypernix</code> as their own executables (see{' '}
                <code style={{ color:'var(--text-faint)' }}>[project.scripts]</code> in pyproject.toml) — run directly, not via{' '}
                <code style={{ color:'var(--text-faint)' }}>hypernix &lt;subcommand&gt;</code>.
              </p>
              {COMPANION_SCRIPTS.map(s => (
                <div key={s.cmd} className="lift-card" style={{ background:'var(--surface-2)', border:'1px solid #212121',
                  borderRadius:10, padding:'14px 20px', marginBottom:10 }}>
                  <code style={{ color:'#4a9eff', fontWeight:700, fontSize:13 }}>{s.cmd}</code>
                  <p style={{ color:'var(--text-dim)', fontSize:13, margin:'6px 0 0' }}>{s.desc}</p>
                </div>
              ))}
            </>
          )}
        </div>
      )}
      {tab === 'python' && (
        <div className="anim-stagger">
          {filteredMods.length === 0 && <p style={{ color:'var(--text-faint)' }}>No modules match.</p>}
          {filteredMods.map(m => (
            <div key={m.name} style={{ marginBottom:12, border:'1px solid var(--border-strong)', borderRadius:10, overflow:'hidden',
              transition:'border-color 0.2s ease' }}>
              <button onClick={() => setOpenMod(openMod === m.name ? null : m.name)} style={{
                width:'100%', background:'var(--surface-3)', border:'none', padding:'14px 20px',
                display:'flex', justifyContent:'space-between', alignItems:'center',
                cursor:'pointer', textAlign:'left', gap:8
              }}>
                <code style={{ color:'#4a9eff', fontWeight:700, fontSize:14,
                  overflowWrap:'break-word', wordBreak:'break-all', flex:1, textAlign:'left' }}>{m.name}</code>
                <span style={{ color:'var(--text-faint)', fontSize:14, flexShrink:0, display:'inline-block',
                  transition:'transform 0.25s cubic-bezier(0.22,1,0.36,1)',
                  transform: openMod === m.name ? 'rotate(180deg)' : 'rotate(0deg)' }}>▼</span>
              </button>
              {openMod === m.name && (
                <div className="anim-fade-up" style={{ background:'var(--bg)', animationDuration:'0.3s' }}>
                  {m.src && (
                    <div style={{ padding:'10px 20px', borderTop:'1px solid var(--surface-3)',
                      display:'flex', alignItems:'center', justifyContent:'space-between', gap:8, flexWrap:'wrap' }}>
                      <code style={{ fontSize:11, color:'var(--text-faint)', overflowWrap:'break-word', wordBreak:'break-all' }}>
                        {m.src}
                      </code>
                      <a href={srcUrl(m.src)} target="_blank" rel="noreferrer" className="press-btn" style={{
                        fontSize:11, color:'#4a9eff', flexShrink:0, whiteSpace:'nowrap',
                        border:'1px solid #1a2a3a', borderRadius:5, padding:'3px 9px',
                        textDecoration:'none', display:'inline-block' }}>
                        View source ↗
                      </a>
                    </div>
                  )}
                  {m.fns.map(fn => (
                    <div key={fn.fn} style={{ borderTop:'1px solid var(--surface-3)', padding:'14px 20px' }}>
                      <div style={{ position:'relative', marginBottom:8 }}>
                        <pre style={{ margin:0, fontSize:12, color:'var(--accent)', fontFamily:'monospace',
                          whiteSpace:'pre-wrap', wordBreak:'break-all', paddingRight:52 }}>{fn.fn}</pre>
                        <CopyButton text={fn.fn} />
                      </div>
                      <p style={{ margin:0, fontSize:13, color:'var(--text-dim)' }}>{fn.desc}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export { ApiPage }
