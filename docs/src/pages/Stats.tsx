import { useState, useEffect } from 'react'
import { PageHeading, CountUp } from '../components/ui'

function StatsPage({ downloads, olderDownloads, threeMonthDownloads, totalDownloads, statsUpdatedAt, ghStats, pypiInfo, pythonVersionStats, systemStats, releaseTimeline, version, statsError }) {
  const windowBars = [
    { label: '24h', val: downloads.last_day },
    { label: '7d', val: downloads.last_week },
    { label: '30d', val: downloads.last_month },
    ...(threeMonthDownloads && threeMonthDownloads.total_downloads > 0
      ? [{ label: '3mo', val: threeMonthDownloads.total_downloads }] : []),
    ...(olderDownloads && olderDownloads.total_downloads > 0
      ? [{ label: 'Older', val: olderDownloads.total_downloads }] : []),
  ]
  const maxDl = Math.max(...windowBars.map(b => b.val), 1)
  const [mounted, setMounted] = useState(false)
  useEffect(() => { const t = setTimeout(() => setMounted(true), 50); return () => clearTimeout(t) }, [])
  return (
    <div className="anim-fade" style={{ maxWidth:860, margin:'0 auto', padding:'90px 20px 60px' }}>
      <PageHeading kicker="Telemetry" title="Package Stats"
        lede="Live download, release, and repository numbers pulled from PyPI and GitHub on page load." />
      {statsError && (
        <div role="status" style={{ background:'#1a1304', border:'1px solid #7a5a10', borderRadius:8,
          padding:'10px 14px', color:'#e8b04a', fontSize:12.5, marginBottom:20 }}>
          {statsError}
        </div>
      )}
      {/* 115px keeps all six tiles on one row at the page's 820px content width
          instead of orphaning "Issues" onto a second row. */}
      <div className="anim-stagger" style={{ display:'grid', gridTemplateColumns:'repeat(auto-fit,minmax(115px,1fr))', gap:12, marginBottom:28 }}>
        {[
          { label:'Downloads 24h', val: downloads.last_day },
          { label:'Downloads 7d', val: downloads.last_week },
          { label:'Downloads 30d', val: downloads.last_month },
          { label:'Stars', val: ghStats.stars },
          { label:'Forks', val: ghStats.forks },
          { label:'Issues', val: ghStats.issues },
        ].map(s => (
          <div key={s.label} className="lift-card" style={{ background:'var(--surface-3)', border:'1px solid var(--border-strong)',
            borderRadius:10, padding:'16px 12px', textAlign:'center', cursor:'default' }}>
            <div style={{ fontSize:26, fontWeight:800, color:'var(--accent)' }}><CountUp value={s.val} /></div>
            <div style={{ fontSize:11, color:'var(--text-faint)', marginTop:5 }}>{s.label}</div>
          </div>
        ))}
      </div>
      <div style={{ background:'var(--surface-3)', border:'1px solid var(--border-strong)', borderRadius:10, padding:'22px', marginBottom:18 }}>
        <h3 style={{ color:'var(--text)', margin:'0 0 18px', fontSize:14 }}>Download window</h3>
        <div style={{ display:'flex', gap:20, alignItems:'flex-end', height:100 }}>
          {windowBars.map(b => (
            <div key={b.label} style={{ flex:1, display:'flex', flexDirection:'column', alignItems:'center', gap:6 }}>
              <span style={{ fontSize:12, color:'var(--text-muted)' }}><CountUp value={b.val} /></span>
              <div style={{ width:'100%', background: b.label === 'Older' ? '#5a5a5a' : b.label === '3mo' ? '#8a2530' : 'var(--accent)',
                borderRadius:'4px 4px 0 0',
                height: mounted ? Math.max(8, (b.val / maxDl) * 64) : 0,
                transition:'height 0.7s cubic-bezier(0.22,1,0.36,1)' }} />
              <span style={{ fontSize:12, color:'var(--text-faint)' }}>{b.label}</span>
            </div>
          ))}
        </div>
      </div>
      {threeMonthDownloads && threeMonthDownloads.days > 0 && (
        <div className="anim-fade-up" style={{ background:'var(--surface-3)', border:'1px solid var(--border-strong)', borderRadius:10, padding:'22px', marginBottom:18 }}>
          <div style={{ display:'flex', justifyContent:'space-between', alignItems:'baseline',
            flexWrap:'wrap', gap:8, marginBottom:14 }}>
            <h3 style={{ color:'var(--text)', margin:0, fontSize:14 }}>3-month downloads</h3>
            <span style={{ fontSize:11, color:'var(--text-faint)' }}>
              31–90 days ago{threeMonthDownloads.from && threeMonthDownloads.to
                ? ` · ${threeMonthDownloads.from} – ${threeMonthDownloads.to}` : ''}
            </span>
          </div>
          <div className="anim-stagger" style={{ display:'grid', gridTemplateColumns:'repeat(auto-fit,minmax(130px,1fr))', gap:12 }}>
            <div className="lift-card" style={{ background:'var(--surface-1)', border:'1px solid var(--border)', borderRadius:8,
              padding:'14px 12px', textAlign:'center', cursor:'default' }}>
              <div style={{ fontSize:22, fontWeight:800, color:'var(--accent)' }}>
                <CountUp value={threeMonthDownloads.total_downloads} />
              </div>
              <div style={{ fontSize:11, color:'var(--text-faint)', marginTop:5 }}>Total downloads</div>
            </div>
            <div className="lift-card" style={{ background:'var(--surface-1)', border:'1px solid var(--border)', borderRadius:8,
              padding:'14px 12px', textAlign:'center', cursor:'default' }}>
              <div style={{ fontSize:22, fontWeight:800, color:'var(--accent)' }}>
                <CountUp value={threeMonthDownloads.days} />
              </div>
              <div style={{ fontSize:11, color:'var(--text-faint)', marginTop:5 }}>Days tracked</div>
            </div>
          </div>
        </div>
      )}
      {olderDownloads && olderDownloads.days > 0 && (
        <div className="anim-fade-up" style={{ background:'var(--surface-3)', border:'1px solid var(--border-strong)', borderRadius:10, padding:'22px', marginBottom:18 }}>
          <div style={{ display:'flex', justifyContent:'space-between', alignItems:'baseline',
            flexWrap:'wrap', gap:8, marginBottom:14 }}>
            <h3 style={{ color:'var(--text)', margin:0, fontSize:14 }}>Older downloads</h3>
            <span style={{ fontSize:11, color:'var(--text-faint)' }}>
              before the last 30 days{olderDownloads.from && olderDownloads.to
                ? ` · ${olderDownloads.from} – ${olderDownloads.to}` : ''}
            </span>
          </div>
          <div className="anim-stagger" style={{ display:'grid', gridTemplateColumns:'repeat(auto-fit,minmax(130px,1fr))', gap:12 }}>
            <div className="lift-card" style={{ background:'var(--surface-1)', border:'1px solid var(--border)', borderRadius:8,
              padding:'14px 12px', textAlign:'center', cursor:'default' }}>
              <div style={{ fontSize:22, fontWeight:800, color:'var(--accent)' }}>
                <CountUp value={olderDownloads.total_downloads} />
              </div>
              <div style={{ fontSize:11, color:'var(--text-faint)', marginTop:5 }}>Total downloads</div>
            </div>
            <div className="lift-card" style={{ background:'var(--surface-1)', border:'1px solid var(--border)', borderRadius:8,
              padding:'14px 12px', textAlign:'center', cursor:'default' }}>
              <div style={{ fontSize:22, fontWeight:800, color:'var(--accent)' }}>
                <CountUp value={olderDownloads.days} />
              </div>
              <div style={{ fontSize:11, color:'var(--text-faint)', marginTop:5 }}>Days tracked</div>
            </div>
            {totalDownloads > 0 && (
              <div className="lift-card" style={{ background:'var(--surface-1)', border:'1px solid var(--border)', borderRadius:8,
                padding:'14px 12px', textAlign:'center', cursor:'default' }}>
                <div style={{ fontSize:22, fontWeight:800, color:'var(--accent)' }}>
                  <CountUp value={totalDownloads} />
                </div>
                <div style={{ fontSize:11, color:'var(--text-faint)', marginTop:5 }}>All-time total</div>
              </div>
            )}
          </div>
          {statsUpdatedAt && (
            <p style={{ fontSize:11, color:'var(--text-faint)', margin:'14px 0 0' }}>
              Stats last updated {new Date(statsUpdatedAt).toLocaleString()}
            </p>
          )}
        </div>
      )}
      {pythonVersionStats.length > 0 && (
        <div style={{ background:'var(--surface-3)', border:'1px solid var(--border-strong)', borderRadius:10, padding:'22px', marginBottom:18 }}>
          <h3 style={{ color:'var(--text)', margin:'0 0 14px', fontSize:14 }}>Downloads by Python version</h3>
          {pythonVersionStats.slice(0,8).map((item, i) => {
            const pct = (item.downloads / pythonVersionStats[0].downloads) * 100
            return (
              <div key={item.version} style={{ display:'flex', alignItems:'center', gap:10, marginBottom:7 }}>
                <code style={{ width:40, fontSize:12, color:'var(--text-dim)' }}>{item.version}</code>
                <div style={{ flex:1, height:14, background:'var(--surface-1)', borderRadius:3, overflow:'hidden' }}>
                  <div style={{ width: mounted ? `${pct}%` : 0, height:'100%', background:'var(--accent)',
                    transition:`width 0.6s cubic-bezier(0.22,1,0.36,1) ${i * 0.05}s` }} />
                </div>
                <span style={{ width:52, textAlign:'right', fontSize:12, color:'var(--text-faint)' }}>{item.downloads.toLocaleString()}</span>
              </div>
            )
          })}
        </div>
      )}
      {systemStats.length > 0 && (
        <div style={{ background:'var(--surface-3)', border:'1px solid var(--border-strong)', borderRadius:10, padding:'22px', marginBottom:18 }}>
          <h3 style={{ color:'var(--text)', margin:'0 0 14px', fontSize:14 }}>Downloads by OS</h3>
          <div className="anim-stagger" style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill,minmax(120px,1fr))', gap:10 }}>
            {systemStats.map(s => (
              <div key={s.os} className="lift-card" style={{ background:'var(--surface-1)', border:'1px solid var(--border)', borderRadius:8, padding:'12px', textAlign:'center', cursor:'default' }}>
                <div style={{ fontSize:20, fontWeight:800, color:'var(--accent)' }}>{s.downloads.toLocaleString()}</div>
                <div style={{ fontSize:11, color:'var(--text-faint)', textTransform:'capitalize', marginTop:4 }}>{s.os}</div>
              </div>
            ))}
          </div>
        </div>
      )}
      {releaseTimeline.length > 0 && (
        <div style={{ background:'var(--surface-3)', border:'1px solid var(--border-strong)', borderRadius:10, padding:'22px', marginBottom:18 }}>
          <h3 style={{ color:'var(--text)', margin:'0 0 14px', fontSize:14 }}>Release history</h3>
          {releaseTimeline.map(r => (
            <div key={r.version} style={{ display:'flex', gap:12, padding:'9px 6px',
              margin:'0 -6px', borderRadius:6, borderBottom:'1px solid var(--surface-3)', alignItems:'flex-start',
              transition:'background-color 0.18s ease' }}
              onMouseEnter={e => e.currentTarget.style.backgroundColor = 'var(--border)'}
              onMouseLeave={e => e.currentTarget.style.backgroundColor = 'transparent'}>
              <div className={r.isPreRelease ? 'float-soft' : ''} style={{ width:7, height:7, borderRadius:'50%', marginTop:5, flexShrink:0,
                background: r.isPreRelease ? '#e8960a' : '#34c759', transition:'transform 0.2s ease' }} />
              <div style={{ flex:1 }}>
                <div style={{ display:'flex', gap:10, alignItems:'center', flexWrap:'wrap' }}>
                  <a href={r.url} target="_blank" rel="noreferrer" className="underline-grow"
                    style={{ color:'var(--accent)', fontFamily:'monospace', fontSize:13, textDecoration:'none' }}>{r.version}</a>
                  {r.isPreRelease && (
                    <span style={{ background:'#e8960a22', border:'1px solid #e8960a44', color:'#e8960a',
                      borderRadius:4, padding:'0 6px', fontSize:10 }}>pre</span>
                  )}
                  <span style={{ color:'var(--text-faint)', fontSize:12 }}>{r.date}</span>
                </div>
                <p style={{ margin:'3px 0 0', fontSize:12, color:'var(--text-faint)' }}>{r.description}</p>
              </div>
            </div>
          ))}
        </div>
      )}
      {pypiInfo && (
        <div className="lift-card" style={{ background:'var(--surface-3)', border:'1px solid var(--border-strong)', borderRadius:10, padding:'22px' }}>
          <h3 style={{ color:'var(--text)', margin:'0 0 14px', fontSize:14 }}>Package info</h3>
          <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill,minmax(170px,1fr))', gap:14 }}>
            {[
              { k:'Version', v: pypiInfo.version },
              { k:'License', v: pypiInfo.license || 'LLU-0.1 (dual-licensed w/ HOS-1.0)' },
              { k:'Author', v: pypiInfo.author || 'trail-b1az3r' },
              { k:'Python requires', v: pypiInfo.requires_python || '>=3.9' },
            ].map(item => (
              <div key={item.k}>
                <div style={{ fontSize:11, color:'var(--text-faint)', marginBottom:3 }}>{item.k}</div>
                <div style={{ color:'var(--text)', fontSize:13, fontFamily:'monospace' }}>{item.v}</div>
              </div>
            ))}
          </div>
          {pypiInfo.summary && <p style={{ marginTop:14, color:'var(--text-dim)', fontSize:13 }}>{pypiInfo.summary}</p>}
        </div>
      )}
    </div>
  )
}

export { StatsPage }
