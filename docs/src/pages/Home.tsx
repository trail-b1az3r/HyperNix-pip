import { useState } from 'react'
import { SUBSYSTEMS, FEATURES, MODELS, QUICKSTART, GROUP_ORDER, subsystemGroup, HERO_SESSION } from '../lib/data/subsystems'
import { FeatureIcon } from '../components/icons'
import { CountUp, SectionHeading, CodeBlock } from '../components/ui'
import { LogoLockup } from '../components/Logo'

function HeroTerminal({ version }) {
  return (
    <div className="term anim-fade-up" style={{ animationDelay:'0.3s' }}>
      <div className="term-bar">
        <span className="term-dot" style={{ background:'var(--accent)' }} />
        <span className="term-dot" style={{ background:'var(--border-hover)' }} />
        <span className="term-dot" style={{ background:'var(--border-hover)' }} />
        <span style={{ marginLeft:6, fontFamily:'var(--font-mono)', fontSize:11,
          color:'var(--text-faint)' }}>hypernix — v{version}</span>
      </div>
      <div className="term-body">
        {HERO_SESSION.map((l, i) => {
          if (l.kind === 'gap') return <div key={i} style={{ height:10 }} />
          if (l.kind === 'cmd') return (
            <div key={i} className="term-line">
              <span style={{ color:'var(--accent)' }}>$ </span>
              <span style={{ color:'var(--text)' }}>{l.text}</span>
            </div>
          )
          if (l.kind === 'ok') return (
            <div key={i} className="term-line" style={{ color:'#34c759' }}>  ✓ {l.text}</div>
          )
          if (l.kind === 'out') return (
            <div key={i} className="term-line" style={{ color:'var(--text-muted)' }}>{l.text}</div>
          )
          return <div key={i} className="term-line" style={{ color:'var(--text-faint)' }}>{l.text}</div>
        })}
        <div className="term-line">
          <span style={{ color:'var(--accent)' }}>$ </span>
          <span style={{ display:'inline-block', width:8, height:14, background:'var(--accent)',
            verticalAlign:'-2px', animation:'blink 1.1s step-end infinite' }} />
        </div>
      </div>
    </div>
  )
}

// Searchable, grouped replacement for the 40-card wall. Filtering happens on
// both the module name and its description so "quantize" finds `convert` too.
function SubsystemBrowser() {
  const [query, setQuery] = useState('')
  const [group, setGroup] = useState('All')

  const q = query.trim().toLowerCase()
  const matches = SUBSYSTEMS.filter(s => {
    if (group !== 'All' && subsystemGroup(s.name) !== group) return false
    if (!q) return true
    return s.name.toLowerCase().includes(q) || s.desc.toLowerCase().includes(q)
  })

  const groups = ['All', ...GROUP_ORDER]
  const countFor = g => g === 'All'
    ? SUBSYSTEMS.length
    : SUBSYSTEMS.filter(s => subsystemGroup(s.name) === g).length

  return (
    <div>
      <div style={{ display:'flex', flexWrap:'wrap', gap:8, alignItems:'center',
        justifyContent:'center', marginBottom:16 }}>
        {groups.map(g => (
          <button key={g} className="filter-chip" aria-pressed={group === g}
            onClick={() => setGroup(g)}>
            {g} <span style={{ opacity:0.6 }}>{countFor(g)}</span>
          </button>
        ))}
      </div>

      <div style={{ display:'flex', justifyContent:'center', marginBottom:8 }}>
        <input
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="Filter subsystems…"
          aria-label="Filter subsystems"
          style={{ width:'100%', maxWidth:380, background:'var(--surface-1)', border:'1px solid var(--border)',
            borderRadius:8, padding:'9px 13px', color:'var(--text)', fontSize:13,
            fontFamily:'var(--font-mono)', outline:'none' }}
        />
      </div>

      <div style={{ textAlign:'center', minHeight:22, marginBottom:10 }}>
        <span style={{ fontFamily:'var(--font-mono)', fontSize:11, color:'var(--text-faint)' }}>
          {matches.length} of {SUBSYSTEMS.length} modules
        </span>
      </div>

      <div style={{ borderBottom:'1px solid #171717' }}>
        {matches.map(s => (
          <div key={s.name} className="sub-row">
            <div style={{ minWidth:0 }}>
              <code style={{ fontSize:12.5, color:'var(--accent)', overflowWrap:'break-word',
                wordBreak:'break-word' }}>{s.name}</code>
              <div className="eyebrow" style={{ color:'var(--text-faint)', marginTop:5, fontSize:9.5 }}>
                {subsystemGroup(s.name)}
              </div>
            </div>
            <div style={{ fontSize:13, color:'var(--text-dim)', lineHeight:1.65, minWidth:0,
              overflowWrap:'break-word' }}>{s.desc}</div>
          </div>
        ))}
        {!matches.length && (
          <div style={{ padding:'36px 4px', textAlign:'center', color:'var(--text-faint)', fontSize:13,
            borderTop:'1px solid #171717' }}>
            Nothing matches “{query}”.
          </div>
        )}
      </div>
    </div>
  )
}

function HomePage({ setPage, downloads, olderDownloads, totalDownloads, ghStats, version }) {
  return (
    <div>
      {/* ── Hero ─────────────────────────────────────────────────────── */}
      <section style={{ position:'relative', padding:'clamp(72px,11vw,132px) var(--space-gutter) clamp(48px,7vw,76px)',
        overflow:'hidden' }}>
        <div className="hero-bg" aria-hidden="true" />
        <div className="shell" style={{ position:'relative', display:'grid', gap:'clamp(36px,5vw,56px)',
          gridTemplateColumns:'repeat(auto-fit,minmax(480px,1fr))', alignItems:'center' }}>

          <div>
            <div className="anim-fade-up" style={{ display:'flex', alignItems:'center', gap:11,
              marginBottom:24 }}>
              <LogoLockup theme="dark" height={26} />
              <span style={{ width:1, height:20, background:'#222' }} />
              <span className="eyebrow" style={{ color:'var(--text-faint)' }}>v{version}</span>
            </div>

            <h1 className="anim-fade-up" style={{ fontSize:'clamp(34px,5.4vw,60px)', fontWeight:900,
              lineHeight:1.02, color:'var(--text)', margin:'0 0 20px', letterSpacing:'-0.045em',
              animationDelay:'0.06s' }}>
              End-to-end toolkit<br/>
              <span style={{ color:'var(--accent)' }}>for PyTorch LLMs</span>
            </h1>

            <p className="anim-fade-up" style={{ fontSize:16.5, color:'var(--text-dim)', maxWidth:'46ch',
              margin:'0 0 28px', lineHeight:1.7, animationDelay:'0.12s' }}>
              Download, chat, fine-tune, evaluate, quantize, and ship — from a single
              package built for consumer hardware.
            </p>

            <div className="anim-fade-up" style={{ display:'flex', gap:10, flexWrap:'wrap',
              marginBottom:26, animationDelay:'0.18s' }}>
              <button onClick={() => setPage('docs')} className="press-btn glow-pulse" style={{
                background:'var(--accent)', border:'none', color:'#fff', borderRadius:8,
                padding:'12px 26px', fontSize:14.5, fontWeight:700, cursor:'pointer',
                fontFamily:'inherit' }}>
                Get Started →
              </button>
              <button onClick={() => setPage('api')} className="press-btn" style={{
                background:'var(--surface-3)', border:'1px solid var(--border-strong)', color:'var(--text)', borderRadius:8,
                padding:'12px 26px', fontSize:14.5, fontWeight:600, cursor:'pointer',
                fontFamily:'inherit' }}>
                API Reference
              </button>
              <a href="https://pypi.org/project/hypernix/" target="_blank" rel="noreferrer"
                className="press-btn" style={{
                  background:'none', border:'1px solid var(--border-strong)', color:'var(--text-muted)', borderRadius:8,
                  padding:'12px 26px', fontSize:14.5, textDecoration:'none',
                  display:'inline-block' }}>
                PyPI ↗
              </a>
            </div>

            <div className="anim-fade-up" style={{ maxWidth:360, animationDelay:'0.24s' }}>
              <CodeBlock code="pip install hypernix" />
            </div>
          </div>

          <HeroTerminal version={version} />
        </div>

        {/* Stat strip: one bordered rail with hairline dividers, not four boxes. */}
        <div className="shell anim-fade-up" style={{ position:'relative', marginTop:'clamp(38px,5vw,58px)',
          animationDelay:'0.34s' }}>
          <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fit,minmax(150px,1fr))',
            gap:1, background:'var(--surface-3)', border:'1px solid #1c1c1c', borderRadius:12,
            overflow:'hidden' }}>
            {[
              { label:'downloads / day', val: downloads.last_day },
              { label:'downloads / week', val: downloads.last_week },
              { label:'downloads / month', val: downloads.last_month },
              { label:'github stars', val: ghStats.stars },
            ].map(s => (
              <div key={s.label} style={{ padding:'20px 18px', textAlign:'center',
                background:'var(--surface-1)' }}>
                <div className="tabular" style={{ fontSize:27, fontWeight:800, color:'var(--accent)',
                  letterSpacing:'-0.03em' }}><CountUp value={s.val} /></div>
                <div className="eyebrow" style={{ fontSize:9.5, color:'var(--text-faint)', marginTop:7 }}>
                  {s.label}
                </div>
              </div>
            ))}
          </div>
          {olderDownloads && olderDownloads.days > 0 && (
            <p className="anim-fade" style={{ fontSize:12, color:'var(--text-faint)', marginTop:12,
              textAlign:'center' }}>
              +{olderDownloads.total_downloads.toLocaleString()} more downloads from before the last 30 days
              {totalDownloads > 0 ? ` · ${totalDownloads.toLocaleString()} all-time` : ''}
            </p>
          )}
        </div>
      </section>

      {/* ── Features ─────────────────────────────────────────────────── */}
      <section className="section" style={{ background:'var(--surface-1)', borderTop:'1px solid var(--surface-2)',
        borderBottom:'1px solid var(--surface-2)' }}>
        <div className="shell">
          <SectionHeading
            kicker="Capabilities"
            title="Everything in one kitchen"
            lede="A complete toolkit for every stage of the LLM lifecycle — no glue scripts between the stages."
          />
          <div className="anim-stagger" style={{ display:'grid',
            gridTemplateColumns:'repeat(auto-fit,minmax(230px,1fr))', gap:14 }}>
            {FEATURES.map((f, i) => (
              <div key={f.title} className="lift-card" style={{ background:'var(--surface-2)',
                border:'1px solid var(--border-strong)', borderRadius:12, padding:'22px 20px 24px',
                cursor:'default', position:'relative' }}>
                <span className="eyebrow" style={{ position:'absolute', top:18, right:18,
                  color:'var(--text-faint)', fontSize:9.5 }}>{String(i + 1).padStart(2,'0')}</span>
                <div className="feat-icon" style={{ marginBottom:14,
                  transition:'transform 0.25s ease' }}>
                  <FeatureIcon name={f.icon} />
                </div>
                <div className="accent-bar" style={{ background:f.color }} />
                <div style={{ fontWeight:700, fontSize:14.5, color:f.color, marginBottom:7,
                  letterSpacing:'-0.01em' }}>{f.title}</div>
                <p style={{ fontSize:13, color:'var(--text-dim)', lineHeight:1.65, margin:0 }}>{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Quickstart ───────────────────────────────────────────────── */}
      <section className="section" id="quickstart">
        <div className="shell-narrow">
          <SectionHeading
            kicker="Quickstart"
            title="Up in minutes"
            lede="Four commands from an empty environment to a quantized model on disk."
          />
          {/* The rail is drawn behind the step badges so the four steps read as
              one sequence instead of four unrelated blocks. */}
          <div style={{ position:'relative' }}>
            <div aria-hidden="true" style={{ position:'absolute', left:17, top:14, bottom:34,
              width:1, background:'linear-gradient(rgba(200,25,46,0.2), var(--surface-3))' }} />
            {QUICKSTART.map(q => (
              <div key={q.step} style={{ display:'flex', gap:18, marginBottom:30,
                alignItems:'flex-start', position:'relative' }}>
                <div style={{ minWidth:35, height:35, borderRadius:9, background:'#1a0305',
                  border:'1px solid rgba(200,25,46,0.27)', display:'flex', alignItems:'center',
                  justifyContent:'center', fontFamily:'var(--font-mono)', fontSize:11,
                  color:'var(--accent)', fontWeight:700, flexShrink:0 }}>{q.step}</div>
                <div style={{ flex:1, minWidth:0 }}>
                  <div style={{ fontWeight:700, color:'var(--text)', marginBottom:4,
                    letterSpacing:'-0.01em' }}>{q.title}</div>
                  <p style={{ color:'var(--text-dim)', fontSize:13, margin:'0 0 10px' }}>{q.desc}</p>
                  <CodeBlock code={q.code} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Models ───────────────────────────────────────────────────── */}
      <section className="section" style={{ background:'var(--surface-1)', borderTop:'1px solid var(--surface-2)',
        borderBottom:'1px solid var(--surface-2)' }}>
        <div className="shell">
          <SectionHeading
            kicker="Model support"
            title="Supported model families"
            lede="Short names resolve automatically in both the CLI and Python."
          >
            <p style={{ marginTop:16, marginBottom:0 }}>
              <a href="https://huggingface.co/collections/ray0rf1re/hyper-nix-v0x" target="_blank"
                rel="noreferrer" className="underline-grow"
                style={{ color:'var(--accent)', fontSize:13, textDecoration:'none', fontWeight:600 }}>
                Browse the HyperNix collection on Hugging Face ↗
              </a>
            </p>
          </SectionHeading>
          <div className="anim-stagger" style={{ display:'grid',
            gridTemplateColumns:'repeat(auto-fill,minmax(240px,1fr))', gap:12 }}>
            {MODELS.map(m => (
              <div key={m.family} className="lift-card" style={{ background:'var(--surface-2)',
                border:'1px solid var(--border-strong)', borderRadius:10, padding:'16px 16px 14px',
                cursor:'default' }}>
                <div style={{ display:'flex', alignItems:'baseline', justifyContent:'space-between',
                  gap:8, marginBottom:12 }}>
                  <span className="eyebrow" style={{ color:'var(--accent-text)', fontSize:10 }}>{m.family}</span>
                  <span className="tabular" style={{ fontSize:10, color:'var(--text-faint)',
                    fontFamily:'var(--font-mono)' }}>{m.models.length}</span>
                </div>
                <div style={{ display:'flex', flexWrap:'wrap', gap:5 }}>
                  {m.models.map(mod => (
                    <span key={mod} className="chip">{mod}</span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Subsystems ───────────────────────────────────────────────── */}
      <section className="section">
        <div style={{ maxWidth:960, margin:'0 auto' }}>
          <SectionHeading
            kicker="Reference"
            title="All subsystems"
            lede={`${SUBSYSTEMS.length} importable modules. Filter by stage, or search names and descriptions.`}
          />
          <SubsystemBrowser />
        </div>
      </section>
    </div>
  )
}


export { HomePage }
