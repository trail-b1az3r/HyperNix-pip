import { useEffect, useMemo, useState } from 'react'
import { ReferenceDocsPage } from './ReferenceDocs'
import { docHref } from './Docs'

// The T1 API, as its own reference (0.72.8a4): what it is and how to start
// one, every endpoint from the server's own OpenAPI document, every setting
// from examples/t1api/.env.example, its error codes from the registry, and
// the source-generated module reference that used to be the whole page.
// Each tab has an address: #/t1-api/endpoints.

type Field = { name: string, type: string, required: boolean, default?: string, description: string }
type Route = {
  method: string, path: string, tag: string, summary: string, description: string,
  params: Array<{ name: string, in: string, required: boolean, type: string, description: string }>,
  body: { type: string, fields: Field[] } | null, multipart: boolean, responses: string[], response: string
}
type Openapi = { title?: string, version?: string, description?: string, tags: Array<{ name: string, count: number }>, routes: Route[] }
type Setting = { name: string, value: string, example: boolean, section: string, help: string }
type Code = { code: string, domain: string, severity: number, severity_name: string, kind: string, kind_meaning: string,
              explanation: string, remedy: string, legacy: string[] }

const TABS = [
  ['overview', 'Overview'], ['endpoints', 'Endpoints'], ['settings', 'Settings'], ['errors', 'Error codes'], ['source', 'Source reference'],
] as const

const TAG_LABEL: Record<string, string> = {
  hyperlink: 'HyperLink', net: 'Net', code: 'Code sandboxes', auth: 'Authentication', accounts: 'Web accounts',
  modules: 'Modules', web: 'Web search', memory: 'Memory', security: 'Security', compaction: 'Compaction',
  mcp: 'MCP', runner: 'Runner', training: 'Training', usage: 'Usage', inference: 'Governed inference',
  billing: 'Billing', models: 'Models', backup: 'Backups', bridge: 'LM Studio bridge', privacy: 'Privacy',
  servers: 'Servers', health: 'Health', jobs: 'Jobs', keys: 'Keys', noodle: 'Noodle', anthropic: 'Anthropic Messages API',
  counter: 'Public counter', events: 'Events', audit: 'Audit', config: 'Config',
}
const tagLabel = (t: string) => TAG_LABEL[t] || t.replace(/(^|-)\w/g, (m) => m.toUpperCase())

function useJson<T>(url: string, empty: T): [T, boolean] {
  const [data, setData] = useState<T>(empty)
  const [ready, setReady] = useState(false)
  useEffect(() => {
    fetch(url).then((r) => { if (!r.ok) throw new Error(); return r.json() }).then(setData).catch(() => {}).finally(() => setReady(true))
  }, [url])
  return [data, ready]
}

function Method({ m }: { m: string }) {
  return <span className={`t1-method m-${m.toLowerCase()}`}>{m}</span>
}

function curl(r: Route) {
  const url = `$T1${r.path.replace(/\{([^}]+)\}/g, (_, n) => `<${n}>`)}`
  const parts = [`curl -s${r.method === 'GET' ? '' : ` -X ${r.method}`} "${url}"`, '-H "Authorization: Bearer $T1_API_KEY"']
  if (r.body) {
    const example = Object.fromEntries(r.body.fields.filter((f) => f.required).map((f) => [f.name,
      f.type.includes('[]') ? [] : f.type === 'integer' || f.type === 'number' ? 0 : f.type === 'boolean' ? false : '…']))
    parts.push("-H 'Content-Type: application/json'", `-d '${JSON.stringify(example)}'`)
  } else if (r.multipart) {
    parts.push('-F "file=@path/to/file"')
  }
  return parts.join(' \\\n  ')
}

function Copy({ text }: { text: string }) {
  const [done, setDone] = useState(false)
  return <button type="button" className="code-copy" onClick={() => { navigator.clipboard?.writeText(text); setDone(true); setTimeout(() => setDone(false), 1400) }}>{done ? 'copied' : 'copy'}</button>
}

function Shell({ code, lang = 'bash' }: { code: string, lang?: string }) {
  return (
    <pre className="t1-code"><div className="code-bar"><span>{lang}</span><Copy text={code} /></div><code>{code}</code></pre>
  )
}

function RouteCard({ r }: { r: Route }) {
  const [open, setOpen] = useState(false)
  return (
    <div className={`t1-route${open ? ' open' : ''}`}>
      <button type="button" className="t1-route-head" aria-expanded={open} onClick={() => setOpen(!open)}>
        <Method m={r.method} />
        <code className="t1-path">{r.path}</code>
        <span className="t1-summary">{r.summary}</span>
      </button>
      {open && (
        <div className="t1-route-body">
          {r.description && <p className="t1-desc">{r.description.replace(/``/g, '`')}</p>}
          {r.params.length > 0 && (
            <>
              <h4>Parameters</h4>
              <table className="t1-table"><thead><tr><th>Name</th><th>In</th><th>Type</th><th /></tr></thead>
                <tbody>{r.params.map((p) => (
                  <tr key={p.in + p.name}><td><code>{p.name}</code></td><td>{p.in}</td><td><code>{p.type}</code></td>
                    <td>{p.required ? <span className="t1-req">required</span> : ''} {p.description}</td></tr>
                ))}</tbody></table>
            </>
          )}
          {r.body && (
            <>
              <h4>Body <code>{r.body.type}</code></h4>
              {r.body.fields.length > 0 ? (
                <table className="t1-table"><thead><tr><th>Field</th><th>Type</th><th>Default</th><th /></tr></thead>
                  <tbody>{r.body.fields.map((f) => (
                    <tr key={f.name}><td><code>{f.name}</code></td><td><code>{f.type}</code></td><td>{f.default !== undefined ? <code>{f.default}</code> : '—'}</td>
                      <td>{f.required ? <span className="t1-req">required</span> : ''} {f.description}</td></tr>
                  ))}</tbody></table>
              ) : <p className="t1-desc">A JSON object; see the description above.</p>}
            </>
          )}
          {r.multipart && <p className="t1-desc">Body: <code>multipart/form-data</code> with a <code>file</code> part.</p>}
          <h4>Responses</h4>
          <p className="t1-desc">{r.responses.join(', ')}{r.response ? <> · 200 is <code>{r.response}</code></> : null}</p>
          <Shell code={curl(r)} />
        </div>
      )}
    </div>
  )
}

function Endpoints({ api, ready }: { api: Openapi, ready: boolean }) {
  const [query, setQuery] = useState('')
  const [tag, setTag] = useState('')
  const q = query.trim().toLowerCase()
  const routes = useMemo(() => api.routes.filter((r) => (!tag || r.tag === tag) && (!q
    || `${r.method} ${r.path} ${r.summary} ${r.description} ${tagLabel(r.tag)}`.toLowerCase().includes(q))), [api, tag, q])
  const groups = useMemo(() => {
    const out: Record<string, Route[]> = {}
    for (const r of routes) (out[r.tag] ||= []).push(r)
    return out
  }, [routes])
  if (ready && !api.routes.length) return <p className="t1-muted">The endpoint list is built from examples/t1api/openapi.json, which this build did not have.</p>
  return (
    <div className="t1-endpoints">
      <aside className="t1-tags" aria-label="Groups">
        <button type="button" className={!tag ? 'active' : ''} onClick={() => setTag('')}>All <span>{api.routes.length}</span></button>
        {api.tags.map((t) => (
          <button type="button" key={t.name} className={tag === t.name ? 'active' : ''} onClick={() => setTag(t.name)}>
            {tagLabel(t.name)} <span>{t.count}</span>
          </button>
        ))}
      </aside>
      <div className="t1-routes">
        <input className="docs-filter t1-search" value={query} onChange={(e) => setQuery(e.target.value)}
          placeholder="Filter: a path, a word, a method — /v1/messages, dataset, DELETE" aria-label="Filter endpoints" />
        <p className="t1-muted">
          Every request carries a credential — <code>Authorization: Bearer &lt;key or token&gt;</code>, or <code>x-api-key</code> on
          the Messages API — unless the server trusts the network it came from. <a href={docHref('T1-API', 'authentication')}>Authentication</a>.
        </p>
        {Object.entries(groups).map(([t, list]) => (
          <section key={t}>
            <h3 className="t1-group-title">{tagLabel(t)} <span>{list.length}</span></h3>
            {list.map((r) => <RouteCard key={r.method + r.path} r={r} />)}
          </section>
        ))}
        {ready && !routes.length && <p className="t1-muted">Nothing matches “{query}”.</p>}
      </div>
    </div>
  )
}

function Settings({ settings }: { settings: Setting[] }) {
  const [query, setQuery] = useState('')
  const q = query.trim().toLowerCase()
  const shown = settings.filter((s) => !q || `${s.name} ${s.help} ${s.section}`.toLowerCase().includes(q))
  const sections: Record<string, Setting[]> = {}
  for (const s of shown) (sections[s.section] ||= []).push(s)
  return (
    <div>
      <p className="t1-muted">Every setting is an environment variable, read from <code>~/.hypernix/t1api/.env</code> by
        <code> hnx-t1</code>. <code>hnx-t1 setup</code> asks about the ones that matter; <code>hnx-t1 upgrade -S env</code> adds
        the ones a newer version brings, commented out at their defaults.</p>
      <input className="docs-filter t1-search" value={query} onChange={(e) => setQuery(e.target.value)}
        placeholder="Filter settings: a name or a word — sandbox, MCP, TRUSTED" aria-label="Filter settings" />
      {Object.entries(sections).map(([section, list]) => (
        <section key={section} className="t1-settings">
          <h3 className="t1-group-title">{section}</h3>
          {list.map((s) => (
            <div key={s.name} className="t1-setting">
              <div className="t1-setting-head">
                <code className="t1-setting-name">{s.name}</code>
                {s.value !== '' && <code className="t1-setting-value">{s.example ? 'e.g. ' : ''}{s.value}</code>}
              </div>
              {s.help && <p>{s.help}</p>}
            </div>
          ))}
        </section>
      ))}
    </div>
  )
}

function Errors({ codes }: { codes: Code[] }) {
  const t1 = codes.filter((c) => c.domain === 'T')
  return (
    <div>
      <p className="t1-muted">Every error answers with <code>{'{"error": {"code": …, "message": …, "hx_code": …}}'}</code>: <code>code</code> is
        the T1 name, <code>hx_code</code> the registered HyperNix code you can search for. A refusal in a sandbox, MCP or dataset route
        also carries its own in <code>details.hx_code</code>. All {codes.length} codes are on <a href={docHref('Error-Codes')}>Error codes</a>.</p>
      <table className="t1-table t1-errors">
        <thead><tr><th>Code</th><th>Severity</th><th>Meaning</th></tr></thead>
        <tbody>{t1.map((c) => (
          <tr key={c.code}><td><code>{c.code}</code></td><td><span className={`sev sev-${c.severity}`}>{c.severity_name}</span></td>
            <td>{c.explanation}{c.remedy ? <span className="t1-remedy"> — {c.remedy}</span> : null}</td></tr>
        ))}</tbody>
      </table>
    </div>
  )
}

function Overview({ api, settings, codes }: { api: Openapi, settings: Setting[], codes: Code[] }) {
  const features = [
    ['Keys and tokens', 'T1 and T2 keys, scoped tokens, plans and quotas; a key per person or per device.', 'authentication'],
    ['HyperLink', 'The phone app and the site: pairing, chats with tools, files, datasets and code sandboxes.', 'hyperlink'],
    ['Claude Code', 'The Anthropic Messages API on any model the server runs: hnx claude.', 'the-anthropic-messages-api'],
    ['MCP', 'An MCP server at /mcp, and a client of others — Hugging Face\'s among them.', 'mcp'],
    ['Models', 'LM Studio, HyperNix\'s own llama.cpp runner, a registry, routing and governed inference.', 'model-registry'],
    ['Net', 'Tailscale peers, exposing a port, and running a file on another machine.', 'net'],
  ]
  return (
    <div className="t1-overview">
      <p className="t1-lede">{(api.description || 'A controlled gateway into HyperNix.').replace(/\s*See wiki\/[^.]*\.\s*$/, '')} One server per machine runs your models,
        keys and HyperLink; any number of clients — the app, the site, <code>waiter</code>, Claude Code, MCP clients — talk to it.</p>
      <div className="t1-stats">
        <div><strong>{api.version || '—'}</strong><span>T1 version</span></div>
        <div><strong>{api.routes.length || '—'}</strong><span>operations</span></div>
        <div><strong>{settings.length || '—'}</strong><span>settings</span></div>
        <div><strong>{codes.filter((c) => c.domain === 'T').length || '—'}</strong><span>T1 error codes</span></div>
      </div>
      <h3 className="t1-group-title">Run one</h3>
      <Shell code={`pip install 'hypernix[t1api]'
hnx-t1 setup              # the installer: asks, writes ~/.hypernix/t1api/.env, starts it
hnx-t1 status             # where it is, its version, and the first admin key
hnx claude                # Claude Code on the model it serves
waiter                    # the T1 client: a TUI and CLI for any T1 server`} />
      <h3 className="t1-group-title">Keep it current</h3>
      <Shell code={`hnx-t1 upgrade -S all     # HyperNix in the server's Python, then its .env, then restart`} />
      <div className="t1-features">
        {features.map(([title, text, anchor]) => (
          <a key={title} className="docs-group-card lift-card" href={docHref('T1-API', anchor)}>
            <h2>{title}</h2><p>{text}</p>
          </a>
        ))}
      </div>
      <p className="t1-muted">The full contract — every route's behaviour, every setting's reasoning — is <a href={docHref('T1-API')}>the T1 API guide</a>.
        Deploying it for others: <a href={docHref('T1-API-Security-Checklist')}>the security checklist</a>.</p>
    </div>
  )
}

export function T1ApiPage({ tab }: { tab?: string }) {
  const active = TABS.some(([t]) => t === tab) ? tab as string : 'overview'
  const [api, apiReady] = useJson<Openapi>('./v1/t1-openapi.json', { tags: [], routes: [] })
  const [env] = useJson<{ settings: Setting[] }>('./v1/t1-settings.json', { settings: [] })
  const [errors] = useJson<{ codes: Code[] }>('./v1/error-codes.json', { codes: [] })
  return (
    <div className="t1-page anim-fade">
      <header className="t1-head">
        <div className="eyebrow" style={{ color: 'var(--accent-text)' }}>Reference · T1</div>
        <h1>T1 API</h1>
        <nav className="t1-tabs" aria-label="T1 API sections">
          {TABS.map(([t, text]) => (
            <a key={t} href={`#/t1-api/${t}`} aria-current={active === t ? 'page' : undefined} className={active === t ? 'active' : ''}>{text}</a>
          ))}
        </nav>
      </header>
      {active === 'overview' && <Overview api={api} settings={env.settings} codes={errors.codes} />}
      {active === 'endpoints' && <Endpoints api={api} ready={apiReady} />}
      {active === 'settings' && <Settings settings={env.settings} />}
      {active === 'errors' && <Errors codes={errors.codes} />}
      {active === 'source' && (
        <ReferenceDocsPage dataUrl="./v1/t1-api.json" kicker="Reference · T1 · source" title="Source reference"
          lede="Generated from the source: the T1 server, SDK, HTTP routes, required server modules, typed errors, and raw endpoint escape hatches."
          mode="t1" embedded />
      )}
    </div>
  )
}
