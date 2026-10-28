import { useEffect, useMemo, useRef, useState } from 'react'

// Daily PyPI downloads as lines (0.72.8a4): the total, with and without
// mirrors, or one line per Python version or operating system, from the
// `series` block scripts/update_json_stats.py writes into v1/json.
//
// Colour follows the entity, never its rank: Python versions take slots
// in version order and systems in a fixed order, so a filter or a new day
// never repaints a line. The slots are the dark steps of the reference
// categorical palette, validated against this surface (#111111): every
// adjacent pair clears CVD ΔE 8 and normal-vision ΔE 15, all >= 3:1.
// "Unknown" (pypistats' null bucket) is a dashed neutral, not a hue.
//
// "Version" (0.72.8rc1) is which hypernix release was downloaded, from
// `series.by_version` (ClickPy's PyPI log; 0.72.7.post1 counts as 0.72.7).
// Fifty-odd releases are too many lines, so it shows the seven most
// downloaded in the range, newest first and coloured in that order, and
// the rest as one dashed "Older releases" line.

export type Series = {
  dates: string[]
  total: number[]
  with_mirrors: number[]
  by_python: Record<string, number[]>
  by_os: Record<string, number[]>
  by_version?: Record<string, number[]>
}

const SLOTS = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767']
const NEUTRAL = '#8f8f8f'
const OS_ORDER = ['Linux', 'Darwin', 'Windows']
const OS_LABEL: Record<string, string> = { Darwin: 'macOS' }
const RANGES = [['30d', 30], ['90d', 90], ['180d', 180], ['All', 0]] as const
const VIEWS = [['total', 'Total'], ['version', 'HyperNix version'], ['python', 'Python version'], ['os', 'Operating system']] as const
const VERSION_LINES = 7

type Line = { key: string, label: string, color: string, dashed: boolean, values: number[] }

function versionKey(v: string) {
  return v.split('.').map((n) => n.padStart(3, '0')).join('.')
}

// The releases worth a line: the most downloaded over the days from
// `from`, newest first; everything else summed into one neutral line.
function versionLines(group: Record<string, number[]>, from: number): Line[] {
  const sum = (v: number[]) => v.slice(from).reduce((a, b) => a + b, 0)
  const names = Object.keys(group).filter((n) => sum(group[n]) > 0)
  const top = names.slice().sort((a, b) => sum(group[b]) - sum(group[a])).slice(0, VERSION_LINES)
  top.sort((a, b) => versionKey(b).localeCompare(versionKey(a)))
  const lines: Line[] = top.map((name, i) => ({ key: name, label: `v${name}`, color: SLOTS[i], dashed: false, values: group[name] }))
  const rest = names.filter((n) => !top.includes(n))
  if (rest.length) {
    const length = Math.max(...rest.map((n) => group[n].length))
    const values = Array.from({ length }, (_, i) => rest.reduce((a, n) => a + (group[n][i] || 0), 0))
    lines.push({ key: 'older', label: `Older releases (${rest.length})`, color: NEUTRAL, dashed: true, values })
  }
  return lines
}

function linesFor(series: Series, view: string, from = 0): Line[] {
  if (view === 'version') return versionLines(series.by_version || {}, from)
  if (view === 'total') {
    return [
      { key: 'total', label: 'Downloads', color: SLOTS[0], dashed: false, values: series.total },
      { key: 'with_mirrors', label: 'Including mirrors', color: SLOTS[1], dashed: false, values: series.with_mirrors },
    ]
  }
  const group = view === 'python' ? series.by_python : series.by_os
  const names = Object.keys(group || {})
  const known = names.filter((n) => n !== 'Unknown')
  known.sort(view === 'python'
    ? (a, b) => versionKey(a).localeCompare(versionKey(b))
    : (a, b) => (OS_ORDER.indexOf(a) + 1 || 99) - (OS_ORDER.indexOf(b) + 1 || 99) || a.localeCompare(b))
  const lines: Line[] = known.slice(0, SLOTS.length).map((name, i) => ({
    key: name, label: view === 'python' ? `Python ${name}` : OS_LABEL[name] || name,
    color: SLOTS[i], dashed: false, values: group[name],
  }))
  if (group.Unknown) lines.push({ key: 'Unknown', label: 'Unknown', color: NEUTRAL, dashed: true, values: group.Unknown })
  return lines
}

function smooth(values: number[], window: number) {
  if (window <= 1) return values
  const out: number[] = []
  let sum = 0
  for (let i = 0; i < values.length; i++) {
    sum += values[i]
    if (i >= window) sum -= values[i - window]
    out.push(sum / Math.min(i + 1, window))
  }
  return out
}

// Four or five gridlines on round numbers: a step of 1, 2 or 5 x 10^k.
function niceScale(v: number) {
  const raw = Math.max(v, 1) / 4
  const p = Math.pow(10, Math.floor(Math.log10(raw)))
  const n = raw / p
  const step = (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * p
  const top = Math.max(step, Math.ceil(Math.max(v, 1) / step) * step)
  const ticks: number[] = []
  for (let t = 0; t <= top + step / 2; t += step) ticks.push(t)
  return { max: top, ticks }
}

const fmt = (n: number) => (n >= 10 ? Math.round(n) : Math.round(n * 10) / 10).toLocaleString()
const day = (iso: string) => new Date(`${iso}T00:00:00Z`).toLocaleDateString(undefined, { month: 'short', day: 'numeric', timeZone: 'UTC' })

function Segmented({ options, value, onChange, label }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map(([v, text]) => (
        <button key={v} type="button" aria-pressed={value === v} className={value === v ? 'on' : ''} onClick={() => onChange(v)}>{text}</button>
      ))}
    </div>
  )
}

export function DownloadsChart({ series }: { series: Series | null }) {
  const [view, setView] = useState('total')
  const [range, setRange] = useState<string>('90d')
  const [avg, setAvg] = useState('7')
  const [hidden, setHidden] = useState<Record<string, boolean>>({})
  const [hover, setHover] = useState<number | null>(null)
  const [table, setTable] = useState(false)
  const box = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(760)
  const ready = !!series?.dates?.length

  useEffect(() => {
    const el = box.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(300, Math.round(e.contentRect.width))))
    ro.observe(el)
    return () => ro.disconnect()
    // The plot is not rendered until the stats arrive, so the observer is
    // attached again then; with [table] alone it never was, and a phone
    // drew the 760px default, cut off at the edge.
  }, [table, ready])
  // pypistats' "Unknown" bucket is most of the downloads; shown, it flattens
  // every real line to zero, so it starts hidden (one click brings it back).
  useEffect(() => { setHidden(view === 'python' || view === 'os' ? { Unknown: true } : {}) }, [view])

  const hasVersions = !!series?.by_version && Object.keys(series.by_version).length > 0
  const views = VIEWS.filter(([v]) => v !== 'version' || hasVersions)
  const from = useMemo(() => {
    if (!series || !series.dates?.length) return 0
    const days = RANGES.find(([r]) => r === range)?.[1] || series.dates.length
    return Math.max(0, series.dates.length - days)
  }, [series, range])
  const data = useMemo(() => {
    if (!series || !series.dates?.length) return null
    // Smooth over the whole history, then cut, so the first days of a
    // range are averages of real days rather than of a partial window.
    const all = linesFor(series, view, from).map((l) => ({ ...l, values: smooth(l.values, Number(avg)).slice(from) }))
    return { dates: series.dates.slice(from), lines: all }
  }, [series, view, from, avg])

  if (!series || !data) {
    return <p style={{ color: 'var(--text-faint)', fontSize: 13 }}>The download history is not in this build's stats file yet.</p>
  }

  const shown = data.lines.filter((l) => !hidden[l.key])
  const labelled = shown.length <= 4 && width >= 520
  const H = 300, padL = 46, padR = labelled ? 120 : 18, padT = 14, padB = 30
  const W = width, innerW = W - padL - padR, innerH = H - padT - padB
  const n = data.dates.length
  const { max, ticks } = niceScale(Math.max(1, ...shown.flatMap((l) => l.values)))
  const x = (i: number) => padL + (n <= 1 ? innerW / 2 : (i / (n - 1)) * innerW)
  const y = (v: number) => padT + innerH - (v / max) * innerH
  const step = Math.max(1, Math.ceil(n / Math.max(2, Math.floor(innerW / 90))))
  const xTicks = data.dates.map((d, i) => [d, i] as const).filter(([, i]) => i % step === 0)
  const path = (vals: number[]) => vals.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('')

  // End labels, nudged apart so two lines ending close do not overprint.
  const ends = labelled
    ? shown.map((l) => ({ l, y: y(l.values[n - 1] || 0) })).sort((a, b) => a.y - b.y)
    : []
  // Down from the top, then up from the bottom, so none leaves the plot.
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 14) ends[i].y = ends[i - 1].y + 14
  const floor = padT + innerH
  if (ends.length && ends[ends.length - 1].y > floor) ends[ends.length - 1].y = floor
  for (let i = ends.length - 2; i >= 0; i--) if (ends[i + 1].y - ends[i].y < 14) ends[i].y = ends[i + 1].y - 14

  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const r = (e.currentTarget as SVGRectElement).getBoundingClientRect()
    const px = ((e.clientX - r.left) / r.width) * innerW
    setHover(Math.max(0, Math.min(n - 1, Math.round((px / innerW) * (n - 1)))))
  }

  const tipLeft = hover !== null ? Math.min(Math.max(x(hover) + 12, 8), W - 200) : 0
  const totalIn = (l: Line) => (avg === '1' ? l.values.reduce((a, b) => a + b, 0) : null)

  // Weekly rows for the table view: a year of days is too long to read.
  const weeks: Array<{ from: string, to: string, values: number[] }> = []
  if (table) {
    const raw = linesFor(series, view, from).filter((l) => !hidden[l.key])
    const off = series.dates.length - n
    for (let i = 0; i < n; i += 7) {
      const end = Math.min(n, i + 7)
      weeks.push({ from: data.dates[i], to: data.dates[end - 1],
        values: raw.map((l) => l.values.slice(off + i, off + end).reduce((a, b) => a + b, 0)) })
    }
  }

  return (
    <div className="dl-chart">
      <div className="dl-controls">
        <Segmented label="Lines" options={views} value={view} onChange={setView} />
        <Segmented label="Range" options={RANGES.map(([r]) => [r, r])} value={range} onChange={setRange} />
        <Segmented label="Smoothing" options={[['1', 'Daily'], ['7', '7-day avg']]} value={avg} onChange={setAvg} />
        <button type="button" className="dl-table-toggle" aria-pressed={table} onClick={() => setTable(!table)}>
          {table ? 'Chart' : 'Table'}
        </button>
      </div>
      <div className="dl-legend" role="list" aria-label="Lines">
        {data.lines.map((l) => (
          <button key={l.key} type="button" role="listitem" aria-pressed={!hidden[l.key]}
            className={hidden[l.key] ? 'off' : ''} onClick={() => setHidden((h) => ({ ...h, [l.key]: !h[l.key] }))}>
            <svg width="18" height="8" aria-hidden="true"><line x1="1" y1="4" x2="17" y2="4" stroke={l.color} strokeWidth="2.5"
              strokeDasharray={l.dashed ? '4 3' : undefined} strokeLinecap="round" /></svg>
            {l.label}
            {totalIn(l) !== null && <span className="dl-sum">{fmt(totalIn(l) as number)}</span>}
          </button>
        ))}
      </div>
      {table ? (
        <div className="dl-table-wrap">
          <table className="dl-table">
            <thead><tr><th>Week</th>{data.lines.filter((l) => !hidden[l.key]).map((l) => <th key={l.key}>{l.label}</th>)}</tr></thead>
            <tbody>{weeks.slice().reverse().map((w) => (
              <tr key={w.from}><td>{day(w.from)} – {day(w.to)}</td>{w.values.map((v, i) => <td key={i}>{v.toLocaleString()}</td>)}</tr>
            ))}</tbody>
          </table>
        </div>
      ) : (
        <div ref={box} className="dl-plot" style={{ position: 'relative' }}>
          <svg width={W} height={H} role="img"
            aria-label={`Daily downloads, ${shown.map((l) => l.label).join(', ')}, ${day(data.dates[0])} to ${day(data.dates[n - 1])}`}>
            {ticks.map((t) => (
              <g key={t}>
                <line x1={padL} x2={W - padR} y1={y(t)} y2={y(t)} stroke="#222" strokeWidth="1" />
                <text x={padL - 8} y={y(t) + 4} textAnchor="end" fontSize="11" fill="#7a7a7a">{fmt(t)}</text>
              </g>
            ))}
            {xTicks.map(([d, i]) => (
              <text key={d} x={x(i)} y={H - 8} textAnchor="middle" fontSize="11" fill="#7a7a7a">{day(d)}</text>
            ))}
            {shown.map((l) => (
              <path key={l.key} d={path(l.values)} fill="none" stroke={l.color} strokeWidth="2"
                strokeLinejoin="round" strokeLinecap="round" strokeDasharray={l.dashed ? '5 4' : undefined} />
            ))}
            {ends.map(({ l, y: ly }) => (
              <g key={l.key}>
                <line x1={W - padR + 4} x2={W - padR + 12} y1={ly} y2={ly} stroke={l.color} strokeWidth="2" />
                <text x={W - padR + 16} y={ly + 4} fontSize="11.5" fill="#d4d4d4">{l.label}</text>
              </g>
            ))}
            {hover !== null && (
              <g>
                <line x1={x(hover)} x2={x(hover)} y1={padT} y2={padT + innerH} stroke="#555" strokeWidth="1" />
                {shown.map((l) => (
                  <circle key={l.key} cx={x(hover)} cy={y(l.values[hover] || 0)} r="4.5" fill={l.color} stroke="#111111" strokeWidth="2" />
                ))}
              </g>
            )}
            <rect x={padL} y={padT} width={innerW} height={innerH} fill="transparent"
              onPointerMove={onMove} onPointerDown={onMove} onPointerLeave={() => setHover(null)} />
          </svg>
          {hover !== null && (
            <div className="dl-tip" style={{ left: tipLeft, top: padT + 6 }} role="status">
              <div className="dl-tip-date">{day(data.dates[hover])}{avg === '7' ? ' · 7-day average' : ''}</div>
              {shown.slice().sort((a, b) => (b.values[hover] || 0) - (a.values[hover] || 0)).map((l) => (
                <div key={l.key} className="dl-tip-row">
                  <svg width="12" height="8" aria-hidden="true"><line x1="1" y1="4" x2="11" y2="4" stroke={l.color} strokeWidth="2.5"
                    strokeDasharray={l.dashed ? '3 2' : undefined} /></svg>
                  <span>{l.label}</span><strong>{fmt(l.values[hover] || 0)}</strong>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      {hidden.Unknown && data.lines.some((l) => l.key === 'Unknown') && (
        <p className="dl-note">“Unknown” (downloads pypistats could not attribute) is hidden; click it in the legend to show it.</p>
      )}
      <p className="dl-note">{view === 'version'
        ? 'Downloads of each hypernix release from ClickPy’s copy of the PyPI download log, mirrors included; a .post or pre-release counts toward its release.'
        : 'PyPI downloads from pypistats.org. “Downloads” leaves out mirrors; pypistats keeps 180 days, and older days are kept from earlier runs.'}</p>
    </div>
  )
}
