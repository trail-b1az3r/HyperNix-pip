import { useEffect, useMemo, useState } from 'react'
import { PageHeading } from '../components/ui'

// What the repository's own scanners found when this site was built:
// v1/security.json, written by scripts/security_report.py in the Pages
// build (the T1 audit, Bandit, Semgrep, and OSV over every dependency
// floor and lockfile). Not in the menu; the footer has a way in.

const SEVERITIES = ['critical', 'high', 'medium', 'low']
const COLOURS = {
  critical: { fg: '#ff6b7a', bg: 'rgba(200, 25, 46, 0.16)', border: 'rgba(255, 107, 122, 0.45)' },
  high: { fg: '#f0954a', bg: 'rgba(240, 149, 74, 0.13)', border: 'rgba(240, 149, 74, 0.42)' },
  medium: { fg: '#e8c04a', bg: 'rgba(232, 192, 74, 0.11)', border: 'rgba(232, 192, 74, 0.38)' },
  low: { fg: '#a3a3a3', bg: 'rgba(163, 163, 163, 0.08)', border: 'rgba(163, 163, 163, 0.3)' },
}
const PAGE = 80

function Badge({ severity }) {
  const c = COLOURS[severity] || COLOURS.medium
  return (
    <span style={{ display: 'inline-block', minWidth: 62, textAlign: 'center', fontSize: 11, fontWeight: 700,
      letterSpacing: '0.04em', textTransform: 'uppercase', padding: '3px 7px', borderRadius: 5,
      color: c.fg, background: c.bg, border: `1px solid ${c.border}` }}>{severity}</span>
  )
}

function Status({ status }) {
  const fg = status === 'ok' ? 'var(--ok)' : status === 'failed' ? '#ff6b7a' : 'var(--text-faint)'
  return <span style={{ color: fg, fontWeight: 600, fontSize: 12 }}>{status === 'ok' ? 'ran' : status}</span>
}

function when(iso) {
  try { return new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }) } catch { return iso }
}

export function SecurityPage() {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [shown, setShown] = useState(new Set(['critical', 'high', 'medium']))
  const [tool, setTool] = useState('')
  const [query, setQuery] = useState('')
  const [tests, setTests] = useState(false)
  const [limit, setLimit] = useState(PAGE)

  useEffect(() => {
    fetch('./v1/security.json', { cache: 'no-store' })
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then(setData)
      .catch(e => setError(String(e.message || e)))
  }, [])

  const findings = useMemo(() => {
    if (!data) return []
    const words = query.toLowerCase().split(/\s+/).filter(Boolean)
    return data.findings.filter(f => shown.has(f.severity) && (tests || !f.test) && (!tool || f.tool === tool)
      && words.every(w => `${f.rule} ${f.path} ${f.message} ${f.title || ''} ${f.package || ''}`.toLowerCase().includes(w)))
  }, [data, shown, tool, query, tests])
  useEffect(() => { setLimit(PAGE) }, [shown, tool, query, tests])

  const toggle = (s) => setShown(prev => {
    const next = new Set(prev)
    next.has(s) ? next.delete(s) : next.add(s)
    return next
  })
  const blob = (f) => data && f.path
    ? `https://github.com/${data.repository}/blob/${data.commit || 'main'}/${f.path}${f.line ? `#L${f.line}` : ''}`
    : ''

  return (
    <div className="anim-fade" style={{ maxWidth: 1040, margin: '0 auto', padding: '90px var(--space-gutter) 60px' }}>
      <PageHeading kicker="Security" title="Security scan"
        lede="What this repository's own scanners found when this site was last built: the HyperNix T1 audit, Bandit, Semgrep, and OSV over every dependency floor and lockfile." />

      {error && (
        <div role="status" style={{ background: 'var(--surface-1)', border: '1px solid var(--border-strong)', borderRadius: 10,
          padding: '18px 20px', color: 'var(--text-dim)', fontSize: 13.5, lineHeight: 1.6 }}>
          No report in this build ({error}). It is written by the Pages build; to make one locally, run{' '}
          <code style={{ fontFamily: 'var(--font-mono)', color: 'var(--text)' }}>python scripts/security_report.py</code>.
        </div>
      )}
      {!data && !error && <p style={{ color: 'var(--text-faint)', fontSize: 13 }}>Loading the report…</p>}

      {data && (
        <>
          <p style={{ color: 'var(--text-faint)', fontSize: 12.5, margin: '0 0 18px' }}>
            Scanned {when(data.generated)}
            {data.commit && <> at <a href={`https://github.com/${data.repository}/commit/${data.commit}`} target="_blank"
              rel="noreferrer" style={{ color: 'var(--accent-text)', fontFamily: 'var(--font-mono)' }}>{data.commit.slice(0, 8)}</a></>}.
            {' '}Findings in tests, examples and benchmarks are counted apart.
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 12, marginBottom: 24 }}>
            {SEVERITIES.map(s => (
              <button key={s} onClick={() => toggle(s)} aria-pressed={shown.has(s)}
                style={{ background: 'var(--surface-3)', borderRadius: 10, padding: '16px 12px', textAlign: 'center',
                  cursor: 'pointer', fontFamily: 'inherit', opacity: shown.has(s) ? 1 : 0.5,
                  border: `1px solid ${shown.has(s) ? COLOURS[s].border : 'var(--border-strong)'}` }}>
                <div style={{ fontSize: 28, fontWeight: 800, color: COLOURS[s].fg }}>{data.totals[s] ?? 0}</div>
                <div style={{ fontSize: 11, color: 'var(--text-faint)', marginTop: 4, textTransform: 'capitalize' }}>{s}</div>
              </button>
            ))}
            <button onClick={() => setTests(t => !t)} aria-pressed={tests}
              style={{ background: 'var(--surface-3)', borderRadius: 10, padding: '16px 12px', textAlign: 'center',
                cursor: 'pointer', fontFamily: 'inherit', opacity: tests ? 1 : 0.5,
                border: `1px solid ${tests ? 'var(--border-hover)' : 'var(--border-strong)'}` }}>
              <div style={{ fontSize: 28, fontWeight: 800, color: 'var(--text-dim)' }}>{data.totals.tests ?? 0}</div>
              <div style={{ fontSize: 11, color: 'var(--text-faint)', marginTop: 4 }}>in tests</div>
            </button>
          </div>

          <section aria-labelledby="sec-tools" style={{ background: 'var(--surface-1)', border: '1px solid var(--border-strong)',
            borderRadius: 10, padding: '18px 20px', marginBottom: 20 }}>
            <h3 id="sec-tools" style={{ color: 'var(--text)', margin: '0 0 10px', fontSize: 14 }}>Scanners</h3>
            <div style={{ display: 'grid', fontSize: 12.5 }}>
              {data.tools.map(t => (
                <div key={t.name} style={{ borderTop: '1px solid var(--border)', padding: '8px 0',
                  display: 'flex', flexWrap: 'wrap', alignItems: 'baseline', columnGap: 14, rowGap: 2 }}>
                  <span style={{ color: 'var(--text)', minWidth: 150 }}>{t.label}</span>
                  <Status status={t.status} />
                  {t.status === 'ok' && <span style={{ color: 'var(--text-dim)' }}>
                    {t.count} finding{t.count === 1 ? '' : 's'}</span>}
                  <span style={{ color: 'var(--text-faint)', flex: '1 1 220px', overflowWrap: 'anywhere' }}>
                    {t.detail || t.version}</span>
                </div>
              ))}
            </div>
          </section>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', marginBottom: 14 }}>
            <input value={query} onChange={e => setQuery(e.target.value)} placeholder="Filter: rule, file, package, words…"
              aria-label="Filter findings"
              style={{ flex: '1 1 240px', minWidth: 0, background: 'var(--surface-2)', border: '1px solid var(--border-strong)',
                borderRadius: 8, padding: '9px 12px', color: 'var(--text)', fontFamily: 'inherit', fontSize: 13 }} />
            <select value={tool} onChange={e => setTool(e.target.value)} aria-label="Scanner"
              style={{ background: 'var(--surface-2)', border: '1px solid var(--border-strong)', borderRadius: 8,
                padding: '9px 10px', color: 'var(--text)', fontFamily: 'inherit', fontSize: 13 }}>
              <option value="">Every scanner</option>
              {data.tools.filter(t => t.status === 'ok').map(t => <option key={t.name} value={t.name}>{t.label}</option>)}
            </select>
            <span style={{ color: 'var(--text-faint)', fontSize: 12 }}>{findings.length} shown</span>
          </div>

          {findings.length === 0 && (
            <p style={{ color: 'var(--text-faint)', fontSize: 13, padding: '18px 0' }}>Nothing matches.</p>
          )}
          <ol style={{ listStyle: 'none', margin: 0, padding: 0, display: 'grid', gap: 8 }}>
            {findings.slice(0, limit).map((f, n) => (
              <li key={`${f.tool}-${f.rule}-${f.path}-${f.line}-${n}`}
                style={{ background: 'var(--surface-2)', border: '1px solid var(--border-strong)', borderRadius: 9,
                  padding: '11px 14px', display: 'grid', gap: 5 }}>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center' }}>
                  <Badge severity={f.severity} />
                  <span style={{ color: 'var(--text-faint)', fontSize: 11.5 }}>{f.tool}</span>
                  <code style={{ color: 'var(--text)', fontFamily: 'var(--font-mono)', fontSize: 12 }}>{f.rule}</code>
                  {f.test && <span style={{ color: 'var(--text-faint)', fontSize: 11, border: '1px solid var(--border-strong)',
                    borderRadius: 4, padding: '1px 6px' }}>test</span>}
                </div>
                <div style={{ color: 'var(--text-dim)', fontSize: 13, lineHeight: 1.5, overflowWrap: 'anywhere' }}>
                  {f.title && <strong style={{ color: 'var(--text)', fontWeight: 600 }}>{f.title}. </strong>}{f.message}
                  {f.fixed && <span style={{ color: 'var(--ok)' }}> Fixed in {f.fixed}.</span>}
                </div>
                {f.path && (
                  <a href={blob(f)} target="_blank" rel="noreferrer"
                    style={{ color: 'var(--accent-text)', fontFamily: 'var(--font-mono)', fontSize: 11.5,
                      overflowWrap: 'anywhere', width: 'fit-content' }}>{f.path}{f.line ? `:${f.line}` : ''}</a>
                )}
              </li>
            ))}
          </ol>
          {findings.length > limit && (
            <button onClick={() => setLimit(l => l + PAGE)}
              style={{ marginTop: 14, background: 'var(--surface-3)', border: '1px solid var(--border-strong)', borderRadius: 8,
                padding: '9px 16px', color: 'var(--text)', cursor: 'pointer', fontFamily: 'inherit', fontSize: 13 }}>
              Show {Math.min(PAGE, findings.length - limit)} more of {findings.length - limit}
            </button>
          )}
        </>
      )}
    </div>
  )
}
