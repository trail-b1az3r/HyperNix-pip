import { useMemo, useState } from 'react'

// Which hypernix version people download (0.72.8rc1), from the `versions`
// list scripts/update_json_stats.py writes into v1/json: every exact
// version with its downloads in the last 30 days and all time.
//
// One measure per bar, so one hue: the accent. The version the site
// documents is marked with a tag, not a second colour.

export type VersionRow = {
  version: string
  release: string
  downloads: number
  last_30_days: number
  first_seen: string
  current?: boolean
}

const SHOWN = 10
const WINDOWS = [['last_30_days', 'Last 30 days'], ['downloads', 'All time']] as const
const GROUPS = [['exact', 'Exact version'], ['release', 'By release']] as const

function Segmented({ options, value, onChange, label }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map(([v, text]) => (
        <button key={v} type="button" aria-pressed={value === v} className={value === v ? 'on' : ''} onClick={() => onChange(v)}>{text}</button>
      ))}
    </div>
  )
}

export function VersionBreakdown({ versions }: { versions: VersionRow[] | null }) {
  const [windowKey, setWindowKey] = useState<'last_30_days' | 'downloads'>('last_30_days')
  const [group, setGroup] = useState('exact')
  const [all, setAll] = useState(false)

  const rows = useMemo(() => {
    if (!versions?.length) return []
    if (group === 'exact') {
      return versions.map((v) => ({ name: v.version, count: v[windowKey] || 0, current: !!v.current, since: v.first_seen }))
    }
    const byRelease = new Map<string, { name: string, count: number, current: boolean, since: string }>()
    for (const v of versions) {
      const r = byRelease.get(v.release) || { name: v.release, count: 0, current: false, since: v.first_seen }
      r.count += v[windowKey] || 0
      r.current = r.current || !!v.current
      r.since = v.first_seen < r.since ? v.first_seen : r.since
      byRelease.set(v.release, r)
    }
    return [...byRelease.values()]
  }, [versions, windowKey, group])

  if (!rows.length) {
    return <p style={{ color: 'var(--text-faint)', fontSize: 13 }}>Downloads by version are not in this build's stats file yet.</p>
  }

  const ranked = rows.filter((r) => r.count > 0).sort((a, b) => b.count - a.count)
  const total = ranked.reduce((a, r) => a + r.count, 0)
  const max = Math.max(1, ...ranked.map((r) => r.count))
  const visible = all ? ranked : ranked.slice(0, SHOWN)
  const top = ranked[0]

  return (
    <div className="vb">
      <div className="dl-controls">
        <Segmented label="Window" options={WINDOWS} value={windowKey} onChange={setWindowKey} />
        <Segmented label="Grouping" options={GROUPS} value={group} onChange={setGroup} />
      </div>
      {top && (
        <p className="vb-lede">
          Most downloaded: <strong>v{top.name}</strong>, {Math.round((top.count / Math.max(1, total)) * 100)}% of{' '}
          {total.toLocaleString()} downloads{windowKey === 'last_30_days' ? ' in the last 30 days' : ''}.
        </p>
      )}
      <ol className="vb-list">
        {visible.map((r) => (
          <li key={r.name} className={r.current ? "vb-row is-current" : "vb-row"}>
            <span className="vb-name">
              v{r.name}
              {r.current && <span className="vb-tag">current</span>}
            </span>
            <span className="vb-track" aria-hidden="true">
              <span className="vb-bar" style={{ width: `${Math.max(1.5, (r.count / max) * 100)}%` }} />
            </span>
            <span className="vb-count">{r.count.toLocaleString()}</span>
            <span className="vb-share">{((r.count / Math.max(1, total)) * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ol>
      {ranked.length > SHOWN && (
        <button type="button" className="dl-table-toggle vb-more" onClick={() => setAll(!all)}>
          {all ? 'Show the top 10' : `Show all ${ranked.length}`}
        </button>
      )}
      <p className="dl-note">From ClickPy’s copy of the PyPI download log, mirrors included, so the totals run a little above pypistats’ “Downloads”.</p>
    </div>
  )
}
