// Every panel's body, as lines of styled segments for a given inner width.
// Pure functions of (info, frame, width): the tests call them directly.

import {
  brailleGraph,
  fmtBytes,
  fmtCount,
  fmtDuration,
  fmtMiB,
  fmtNumber,
  fmtValue,
  lineText,
  meter,
  pad,
  seg,
  sparkline,
  truncate,
  truncateStart,
  wrap,
  sliceLine,
  type Line,
} from "./format.ts"
import { theme } from "./theme.ts"
import type { Finding, Frame, ImportInfo, Info } from "./types.ts"

export type PanelId =
  | "cpu" | "mem" | "gpu" | "train" | "cooker" | "model" | "logs" | "warnings" | "modules" | "procs"
  | "temps" | "audio" | "disks" | "net" | "system"

export interface PanelSpec {
  id: PanelId
  key: string // the number that toggles it, btop-style ("⇧1" is Shift+1)
  title: string
  // Hidden until asked for with Shift+its number.
  hidden?: boolean
  // What `o` cycles through when the panel is selected; the first is the default.
  sorts?: string[]
  // Lines wider than the panel can be scrolled sideways (→ and Shift+→).
  scrollX?: boolean
}

export const PANELS: PanelSpec[] = [
  { id: "cpu", key: "1", title: "cpu", sorts: ["core", "busiest", "hottest"] },
  { id: "mem", key: "2", title: "mem", sorts: ["most used", "least used", "fastest growing"], scrollX: true },
  { id: "gpu", key: "3", title: "gpu" },
  { id: "train", key: "4", title: "training" },
  { id: "cooker", key: "5", title: "pressure cooker" },
  { id: "model", key: "6", title: "model", scrollX: true },
  { id: "logs", key: "7", title: "logs", scrollX: true },
  { id: "warnings", key: "8", title: "warnings" },
  { id: "modules", key: "9", title: "modules & libraries", scrollX: true },
  { id: "procs", key: "0", title: "processes", sorts: ["cpu", "memory", "age", "pid"], scrollX: true },
  // 0.72.8: five more, hidden until Shift+1 … Shift+5.
  { id: "temps", key: "⇧1", title: "temperatures", hidden: true, sorts: ["core", "hottest"] },
  { id: "audio", key: "⇧2", title: "sound", hidden: true },
  { id: "disks", key: "⇧3", title: "disks", hidden: true, sorts: ["most full", "largest", "busiest"], scrollX: true },
  { id: "net", key: "⇧4", title: "network", hidden: true, sorts: ["busiest", "name"] },
  { id: "system", key: "⇧5", title: "system", hidden: true },
]

// Shift+1 … Shift+5 as terminals send them: the shifted character on a US
// keyboard (and the UK/German ones that differ), or the digit with Shift
// held under the kitty keyboard protocol.
const SHIFTED: Record<string, string> = { "!": "1", "@": "2", "#": "3", $: "4", "%": "5", '"': "2", "£": "3", "§": "3" }

export function panelForKey(name: string, shift = false): PanelSpec | undefined {
  const digit = SHIFTED[name] ?? (shift && /^[1-5]$/.test(name) ? name : undefined)
  if (digit) return PANELS.find((p) => p.key === `⇧${digit}`)
  return PANELS.find((p) => p.key === name)
}

export interface PanelContext {
  info: Info | null
  frame: Frame | null
  width: number
  // Lines the panel may use. A panel with more says how many it hid.
  height: number
  compact: boolean
  // Log panel: lines scrolled back from the end (0 = following).
  logOffset?: number
  // The panel's sort (one of its spec's `sorts`), from `o`.
  sort?: string
}

function meterRow(name: string, fraction: number | null | undefined, value: string, width: number, nameWidth = 5, valueColor: string = theme.text): Line {
  const valueWidth = Math.max(value.length, 6)
  const barWidth = Math.max(1, width - nameWidth - valueWidth - 2)
  return [
    seg(pad(name, nameWidth), theme.textDim),
    seg(" "),
    ...meter(fraction, barWidth),
    seg(" "),
    seg(value.padStart(valueWidth), valueColor),
  ]
}

const tempColor = (celsius: number | null | undefined) =>
  celsius == null ? theme.text : celsius >= 90 ? theme.error : celsius >= 80 ? theme.warn : theme.text

const fmtTemp = (celsius: number | null | undefined) => (celsius == null ? "—" : `${celsius.toFixed(0)}°C`)

const fmtRate = (bps: number | null | undefined) => (bps == null ? "—" : `${fmtBytes(Math.abs(bps))}/s`)

const pct = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${v.toFixed(0)}%`)

function fit(lines: Line[], height: number, width: number): Line[] {
  if (height <= 0 || lines.length <= height) return lines
  const hidden = lines.length - height + 1
  return [...lines.slice(0, height - 1), [seg(truncate(`… ${hidden} more`, width), theme.hint)]]
}

// -- 1 cpu -----------------------------------------------------------------

export function cpuPanel(c: PanelContext): Line[] {
  const f = c.frame
  if (!f) return [[seg("waiting for the first sample…", theme.hint)]]
  const temps = f.cpu_temps?.per_core ?? []
  const pkg = f.cpu_temps?.package_c
  const lines: Line[] = [meterRow("total", (f.cpu_percent ?? NaN) / 100, `${pct(f.cpu_percent)}${pkg != null ? ` ${pkg.toFixed(0)}°` : ""}`, c.width, 5, tempColor(pkg))]
  const cores = f.cpu_per_core ?? []
  if (!c.compact && cores.length) {
    // Two columns of per-core meters when there is room for them, each
    // with its own temperature where the driver reports one.
    const order = cores.map((_, i) => i)
    if (c.sort === "busiest") order.sort((a, b) => cores[b] - cores[a])
    else if (c.sort === "hottest") order.sort((a, b) => (temps[b] ?? -1) - (temps[a] ?? -1))
    const columns = c.width >= 60 ? 2 : 1
    const colWidth = Math.floor((c.width - (columns - 1) * 2) / columns)
    for (let i = 0; i < order.length; i += columns) {
      const row: Line = []
      for (let j = 0; j < columns && i + j < order.length; j++) {
        const core = order[i + j]
        const t = temps[core]
        if (j) row.push(seg("  "))
        row.push(...meterRow(`c${core}`, cores[core] / 100, `${pct(cores[core])}${t != null ? ` ${t.toFixed(0)}°` : ""}`, colWidth, 4, tempColor(t)))
      }
      lines.push(row)
    }
    if (!temps.some((t) => t != null) && f.cpu_temps?.note && pkg == null) {
      lines.push([seg(truncate(`temps: ${f.cpu_temps.note}`, c.width), theme.hint)])
    }
  } else if (cores.length) {
    lines.push([seg(`${cores.length} cores  `, theme.textDim), seg(sparkline(cores, Math.max(1, c.width - 10)), theme.graph)])
  }
  const graphHeight = c.compact ? 2 : Math.max(2, Math.min(4, c.height - lines.length))
  for (const row of brailleGraph(f.cpu_history ?? [], c.width, graphHeight)) lines.push([seg(row, theme.graph)])
  return fit(lines, c.height, c.width)
}

// -- 2 mem -----------------------------------------------------------------

export function memPanel(c: PanelContext): Line[] {
  const f = c.frame
  if (!f) return [[seg("waiting…", theme.hint)]]
  const m = f.memory ?? {}
  const total = m.total_mib ?? 0
  // The bar is the numbers beside it: used over total, one measurement.
  const fraction = total && m.used_mib != null ? m.used_mib / total : (f.ram_percent ?? NaN) / 100
  const percent = Number.isFinite(fraction) ? ` ${(fraction * 100).toFixed(0)}%` : ""
  const lines: Line[] = [
    meterRow("ram", fraction, `${fmtMiB(m.used_mib)}/${fmtMiB(total)}${percent}`, c.width),
  ]
  if (m.swap_total_mib) {
    lines.push(meterRow("swap", (m.swap_used_mib ?? 0) / m.swap_total_mib, `${fmtMiB(m.swap_used_mib)}/${fmtMiB(m.swap_total_mib)}`, c.width))
  }
  if (!c.compact) {
    const available = m.available_mib ?? m.free_mib
    lines.push([
      seg("avail ", theme.textDim), seg(fmtMiB(available), theme.text),
      seg("  cache ", theme.textDim), seg(fmtMiB(m.cached_mib), theme.text),
      ...(m.available_mib != null ? [seg("  free ", theme.textDim), seg(fmtMiB(m.free_mib), theme.text)] : []),
    ].reduce<Line>((line, s) => (lineText([...line, s]).length <= c.width ? [...line, s] : line), []))
    if ((m as Record<string, unknown>).source === "cgroup") lines.push([seg(truncate("the container's memory limit, not the host's", c.width), theme.hint)])
    for (const row of brailleGraph(f.ram_history ?? [], c.width, 2)) lines.push([seg(row, theme.graph)])
  }
  const users = [...(f.memory_users ?? [])]
  if (users.length && c.height - lines.length > 2) {
    const sort = c.sort ?? "most used"
    if (sort === "least used") users.sort((a, b) => (a.rss_bytes ?? 0) - (b.rss_bytes ?? 0))
    else if (sort === "fastest growing") users.sort((a, b) => (b.growth_bps ?? -Infinity) - (a.growth_bps ?? -Infinity))
    else users.sort((a, b) => (b.rss_bytes ?? 0) - (a.rss_bytes ?? 0))
    lines.push([seg(truncate(`  rss  growth  process · ${sort}`, c.width), theme.textDim)])
    for (const u of users) {
      const growth = u.growth_bps == null ? "—" : `${u.growth_bps < 0 ? "-" : "+"}${fmtRate(u.growth_bps)}`
      const cols = `${fmtBytes(u.rss_bytes).padStart(6)} ${growth.padStart(8)} `
      const shown = truncate(cols, c.width)
      const room = c.width - shown.length
      const fast = (u.growth_bps ?? 0) > 50 * 1024 * 1024
      lines.push([
        seg(shown, fast ? theme.warn : theme.text),
        ...(room > 0 ? [seg(truncate(u.command || u.name || String(u.pid), room), theme.textMuted)] : []),
      ])
    }
  }
  return fit(lines, c.height, c.width)
}

// -- 3 gpu -----------------------------------------------------------------

export function gpuPanel(c: PanelContext): Line[] {
  const f = c.frame
  if (!f) return [[seg("waiting…", theme.hint)]]
  if (!f.gpu_name && f.gpu_util_percent == null) {
    return [[seg("no NVIDIA GPU visible (nvidia-smi did not answer)", theme.hint)]]
  }
  const lines: Line[] = [[seg(truncate(f.gpu_name ?? "GPU", c.width), theme.text, true)]]
  lines.push(meterRow("util", (f.gpu_util_percent ?? NaN) / 100, pct(f.gpu_util_percent), c.width))
  const used = f.gpu_mem_used_mib
  const total = f.gpu_mem_total_mib
  lines.push(meterRow("vram", used != null && total ? used / total : null, `${fmtMiB(used)}/${fmtMiB(total)}`, c.width))
  if (f.gpu_power_w != null) {
    const limit = f.gpu_power_limit_w
    lines.push(meterRow("power", limit ? f.gpu_power_w / limit : null, `${f.gpu_power_w.toFixed(0)}W`, c.width))
  }
  if (f.gpu_temp_c != null) {
    const hot = f.gpu_temp_c >= 83
    lines.push([seg(pad("temp", 6), theme.textDim), seg(`${f.gpu_temp_c.toFixed(0)}°C`, hot ? theme.error : theme.text), ...(hot ? [seg("  hot", theme.error)] : [])])
  }
  if (!c.compact) for (const row of brailleGraph(f.gpu_util_history ?? [], c.width, 2)) lines.push([seg(row, theme.graph)])
  return fit(lines, c.height, c.width)
}

// -- 4 training ------------------------------------------------------------

export function trainPanel(c: PanelContext): Line[] {
  const f = c.frame
  if (!f) return [[seg("waiting…", theme.hint)]]
  const lines: Line[] = []
  if (!f.has_training_data && f.discovering) {
    lines.push([seg("looking for the training run and its log…", theme.hint)])
  } else if (!f.has_training_data) {
    lines.push([seg(c.info?.log ? "no step lines in the log yet" : "no training log (pass --log)", theme.hint)])
  } else {
    const total = f.total_steps ?? 0
    const step = f.step ?? 0
    lines.push(meterRow("step", total ? step / total : null, total ? `${step}/${total}` : String(step), c.width))
    lines.push([
      seg("loss ", theme.textDim), seg(fmtNumber(f.loss), f.loss != null && !Number.isFinite(f.loss) ? theme.error : theme.text, true),
      seg("   lr ", theme.textDim), seg(fmtNumber(f.lr), theme.text),
    ])
    lines.push([
      seg("speed ", theme.textDim), seg(f.throughput != null ? `${fmtNumber(f.throughput, 2)} it/s` : "—", theme.text),
      seg("   eta ", theme.textDim), seg(fmtDuration(f.eta_seconds), theme.text),
    ])
    lines.push([seg("elapsed ", theme.textDim), seg(fmtDuration(f.elapsed_seconds), theme.text)])
    const losses = f.recent_losses ?? []
    if (losses.length > 1) {
      const top = Math.max(...losses.filter(Number.isFinite))
      const rows = brailleGraph(losses, c.width, c.compact ? 2 : Math.max(2, Math.min(5, c.height - lines.length - 1)), top)
      for (const row of rows) lines.push([seg(row, theme.ok)])
    }
  }
  if (f.log_age_seconds != null && f.log_age_seconds > 600) {
    lines.push([seg(truncate(`log untouched for ${fmtDuration(f.log_age_seconds)}: is the run still going?`, c.width), theme.warn)])
  }
  return fit(lines, c.height, c.width)
}

// -- 5 pressure cooker -----------------------------------------------------

export function cookerPanel(c: PanelContext): Line[] {
  const report = c.info?.report
  if (!c.info) return [[seg("reading the script…", theme.hint)]]
  if (!report) return [[seg("no script to read (pass --script)", theme.hint)]]
  const cookers = report.cookers ?? []
  if (!cookers.length) {
    return [[seg(truncate("no Pressure Cooker in the script (another optimizer, or chosen at run time)", c.width), theme.hint)]]
  }
  const lines: Line[] = []
  for (const cooker of cookers) {
    const gen = (cooker.generation ?? "").toUpperCase()
    lines.push([
      seg(truncate(cooker.name, c.width - 12), theme.text, true),
      seg(gen ? `  ${gen}` : "", theme.accentText),
      seg(cooker.line ? `  :${cooker.line}` : "", theme.hint),
    ])
    if (cooker.deprecated) {
      for (const text of wrap(`deprecated: use ${cooker.deprecated}`, c.width)) lines.push([seg(text, theme.warn)])
    } else if (cooker.summary && !c.compact) {
      for (const text of wrap(cooker.summary, c.width).slice(0, 2)) lines.push([seg(text, theme.textDim)])
    }
    for (const [key, value] of Object.entries(cooker.kwargs ?? {})) {
      lines.push([seg(pad(key, 14), theme.textDim), seg(truncate(fmtValue(value), c.width - 14), theme.text)])
    }
  }
  if (c.frame?.lr != null) lines.push([seg(pad("lr now (log)", 14), theme.textDim), seg(fmtNumber(c.frame.lr), theme.ok)])
  return fit(lines, c.height, c.width)
}

// -- 6 model ---------------------------------------------------------------

const ARCH_LABELS: Record<string, string> = {
  vocab_size: "vocab",
  hidden_size: "hidden",
  intermediate_size: "ffn",
  num_hidden_layers: "layers",
  num_attention_heads: "heads",
  num_key_value_heads: "kv heads",
  max_position_embeddings: "context",
  dtype: "dtype",
  rope_theta: "rope θ",
  rms_norm_eps: "rms eps",
  tie_word_embeddings: "tied emb",
  attention_bias: "attn bias",
  model_type: "type",
}

export function modelPanel(c: PanelContext): Line[] {
  if (!c.info) return [[seg("reading the script…", theme.hint)]]
  const arch = c.info.arch ?? c.info.report?.arch ?? {}
  if (!arch.source || arch.source === "none") {
    return [[seg(truncate("no architecture found: no new_oven()/preheat() in the script, and no config.json beside it", c.width), theme.hint)]]
  }
  const lines: Line[] = []
  const name = arch.arch || (arch.repo ? truncateStart(arch.repo, c.width - 10) : "?")
  lines.push([seg(name, theme.text, true), seg(`  from ${arch.source}${arch.line ? ` :${arch.line}` : ""}`, theme.hint)])
  if (arch.params) lines.push([seg(pad("params", 10), theme.textDim), seg(`~${fmtCount(arch.params)}`, theme.accentText, true)])
  const shown = new Set<string>()
  const add = (key: string, value: unknown, fromPreset: boolean) => {
    if (shown.has(key)) return
    shown.add(key)
    lines.push([
      seg(pad(ARCH_LABELS[key] ?? key, 10), theme.textDim),
      seg(truncate(fmtValue(value), c.width - 12), fromPreset ? theme.textMuted : theme.text),
    ])
  }
  for (const [key, value] of Object.entries(arch.fields ?? {})) add(key, value, false)
  if (!c.compact) for (const [key, value] of Object.entries(arch.preset ?? {})) add(key, value, true)
  if (arch.note) for (const text of wrap(arch.note, c.width)) lines.push([seg(text, theme.hint)])
  return fit(lines, c.height, c.width)
}

// -- 7 logs ----------------------------------------------------------------

export function logLineColor(line: string): string {
  if (/Traceback|Error|Exception|out of memory|\bKilled\b|loss\s*[=:]\s*(nan|inf)/i.test(line)) return theme.error
  if (/warn/i.test(line)) return theme.warn
  if (/step\s+\d+/i.test(line)) return theme.text
  return theme.textDim
}

export function logsPanel(c: PanelContext): Line[] {
  const lines = c.frame?.log_lines ?? []
  if (!lines.length && (!c.frame || c.frame.discovering)) return [[seg("looking for the training log…", theme.hint)]]
  if (!c.info?.log && !lines.length) return [[seg("no training log (pass --log)", theme.hint)]]
  if (!lines.length) return [[seg("the log is empty so far", theme.hint)]]
  const offset = Math.max(0, Math.min(c.logOffset ?? 0, Math.max(0, lines.length - c.height)))
  const end = lines.length - offset
  const visible = lines.slice(Math.max(0, end - c.height), end)
  const out = visible.map((text) => [seg(truncate(text.replace(/\t/g, "  "), c.width), logLineColor(text))])
  if (offset > 0 && out.length) out[out.length - 1] = [seg(truncate(`↓ ${offset} newer lines (end to follow)`, c.width), theme.accentText)]
  return out
}

// -- 8 warnings ------------------------------------------------------------

const ICON: Record<string, [string, string]> = {
  error: ["✖", theme.error],
  warn: ["▲", theme.warn],
  info: ["•", theme.textMuted],
}

export function allFindings(info: Info | null, frame: Frame | null): Finding[] {
  const order: Record<string, number> = { error: 0, warn: 1, info: 2 }
  const found = [...(info?.report?.findings ?? []), ...(frame?.log_findings ?? [])]
  const audio = frame?.audio
  if (audio?.warning && (audio.level === "loud" || audio.level === "unsafe")) {
    found.push({ level: audio.level === "unsafe" ? "error" : "warn", source: "audio", message: audio.warning })
  }
  if (info?.report?.error && !found.some((f) => f.level === "error" && f.source === "script")) {
    found.push({ level: "error", source: "script", message: info.report.error })
  }
  return found.sort((a, b) => (order[a.level] ?? 3) - (order[b.level] ?? 3))
}

export function warningsPanel(c: PanelContext): Line[] {
  const findings = allFindings(c.info, c.frame)
  if (!findings.length) {
    return [[seg(c.info?.report || c.frame?.log_lines?.length ? "✔ nothing to report" : "no script or log to check yet", c.info?.report ? theme.ok : theme.hint)]]
  }
  const lines: Line[] = []
  for (const f of findings) {
    const [icon, color] = ICON[f.level] ?? ICON.info
    const where = `${f.source}${f.line ? `:${f.line}` : ""}${f.count && f.count > 1 ? ` ×${f.count}` : ""}`
    const body = wrap(f.message, Math.max(8, c.width - 2))
    lines.push([seg(`${icon} `, color), seg(body[0] ?? "", theme.text)])
    for (const rest of body.slice(1, c.compact ? 2 : 3)) lines.push([seg("  "), seg(rest, theme.text)])
    lines.push([seg("  "), seg(truncate(where, c.width - 2), theme.hint)])
  }
  return fit(lines, c.height, c.width)
}

// -- 9 modules & libraries -------------------------------------------------

export function modulesPanel(c: PanelContext): Line[] {
  if (!c.info) return [[seg("reading the script…", theme.hint)]]
  const imports = c.info.report?.imports
  if (!imports) return [[seg("no script to read (pass --script)", theme.hint)]]
  const by = (kind: ImportInfo["kind"]) => imports.filter((i) => i.kind === kind)
  const lines: Line[] = []
  const hypernix = by("hypernix")
  if (hypernix.length) {
    lines.push([seg(`HyperNix ${c.info.hypernix_version ?? ""}`.trim(), theme.accentText, true)])
    for (const item of hypernix) {
      const name = item.module.replace(/^hypernix\./, "")
      const names = item.names?.length ? ` (${item.names.join(", ")})` : ""
      lines.push([seg("  "), seg(truncate(name + names, c.width - 2), item.deprecated ? theme.warn : theme.text)])
      if (!c.compact && item.summary) lines.push([seg("    "), seg(truncate(item.summary, c.width - 4), theme.hint)])
      if (item.deprecated) lines.push([seg("    "), seg(truncate(`deprecated → ${item.deprecated}`, c.width - 4), theme.warn)])
    }
  }
  const libraries = by("library")
  if (libraries.length) {
    lines.push([seg("libraries", theme.accentText, true)])
    const nameWidth = Math.min(22, Math.max(...libraries.map((l) => l.module.length)) + 1)
    for (const item of libraries) {
      lines.push([seg("  "), seg(pad(item.module, nameWidth), theme.text), seg(truncate(item.version || "?", c.width - nameWidth - 2), theme.textDim)])
    }
  }
  const missing = by("missing")
  if (missing.length) {
    lines.push([seg("not installed", theme.error, true)])
    for (const item of missing) lines.push([seg("  "), seg(truncate(item.module, c.width - 2), theme.error)])
  }
  const local = by("local")
  if (local.length) lines.push([seg("local  ", theme.textDim), seg(truncate(local.map((l) => l.module).join(", "), c.width - 7), theme.text)])
  const stdlib = by("stdlib")
  if (stdlib.length) lines.push([seg("stdlib ", theme.textDim), seg(truncate(stdlib.map((l) => l.module).join(", "), c.width - 7), theme.textMuted)])
  if (!lines.length) lines.push([seg("the script imports nothing", theme.hint)])
  return fit(lines, c.height, c.width)
}

// -- 0 processes -----------------------------------------------------------

export function procsPanel(c: PanelContext): Line[] {
  const rows = [...(c.frame?.processes ?? [])]
  if (!rows.length) return [[seg("no Python processes (or psutil is not installed)", theme.hint)]]
  const sort = c.sort ?? "cpu"
  if (sort === "memory") rows.sort((a, b) => (b.rss_bytes ?? 0) - (a.rss_bytes ?? 0))
  else if (sort === "age") rows.sort((a, b) => (b.age_seconds ?? 0) - (a.age_seconds ?? 0))
  else if (sort === "pid") rows.sort((a, b) => a.pid - b.pid)
  else rows.sort((a, b) => (b.cpu_percent ?? 0) - (a.cpu_percent ?? 0))
  const lines: Line[] = []
  const header = c.compact ? "pid     cpu%  command" : "pid     cpu%   mem   age      command"
  lines.push([seg(truncate(sort === "cpu" ? header : `${header} · by ${sort}`, c.width), theme.textDim)])
  for (const p of rows) {
    const watched = c.info?.pid === p.pid
    const cols = c.compact
      ? `${String(p.pid).padEnd(7)} ${(p.cpu_percent ?? 0).toFixed(0).padStart(4)}  `
      : `${String(p.pid).padEnd(7)} ${(p.cpu_percent ?? 0).toFixed(0).padStart(4)} ${fmtBytes(p.rss_bytes).padStart(5)}  ${fmtDuration(p.age_seconds).padEnd(8)} `
    // The columns can be wider than a phone's panel; they give way first.
    const shown = truncate(cols, c.width)
    const room = c.width - shown.length
    lines.push([
      seg(shown, watched ? theme.accentText : theme.text),
      ...(room > 0 ? [seg(truncate(p.command ?? p.name ?? "", room), watched ? theme.accentText : theme.textMuted)] : []),
    ])
  }
  return fit(lines, c.height, c.width)
}

// -- ⇧1 temperatures ----------------------------------------------------------

export function tempsPanel(c: PanelContext): Line[] {
  const t = c.frame?.cpu_temps
  if (!c.frame) return [[seg(truncate("waiting…", c.width), theme.hint)]]
  if (!t) return [[seg(truncate("no temperature readings from the bridge", c.width), theme.hint)]]
  const lines: Line[] = []
  lines.push([seg(pad("package", 9), theme.textDim), seg(fmtTemp(t.package_c), tempColor(t.package_c), true),
    seg(t.source ? `  ${t.source}` : "", theme.hint)])
  if (c.frame.gpu_temp_c != null) lines.push([seg(pad("gpu", 9), theme.textDim), seg(fmtTemp(c.frame.gpu_temp_c), tempColor(c.frame.gpu_temp_c))])
  const per = t.per_core ?? []
  const order = per.map((_, i) => i).filter((i) => per[i] != null)
  if (c.sort === "hottest") order.sort((a, b) => (per[b] ?? 0) - (per[a] ?? 0))
  for (const core of order) {
    const celsius = per[core] as number
    // 30-100 °C across the bar: below 30 is idle, 100 is where CPUs throttle.
    lines.push(meterRow(`c${core}`, (celsius - 30) / 70, `${celsius.toFixed(0)}°C`, c.width, 5, tempColor(celsius)))
  }
  for (const [name, celsius] of Object.entries(t.dies ?? {})) {
    lines.push(meterRow(truncate(name, 6), (celsius - 30) / 70, `${celsius.toFixed(0)}°C`, c.width, 6, tempColor(celsius)))
  }
  if (t.note) for (const text of wrap(t.note, c.width)) lines.push([seg(text, theme.hint)])
  return fit(lines, c.height, c.width)
}

// -- ⇧2 sound ------------------------------------------------------------------

export function audioPanel(c: PanelContext): Line[] {
  const a = c.frame?.audio
  if (!c.frame) return [[seg(truncate("waiting…", c.width), theme.hint)]]
  if (!a || !a.available) return [[seg(truncate(a?.note ?? "no sound information from the bridge", c.width), theme.hint)]]
  const lines: Line[] = []
  const levelColor = a.level === "unsafe" ? theme.error : a.level === "loud" ? theme.warn : theme.text
  lines.push([seg(truncate(`${a.device ?? "output"}${a.headphones ? " · headphones" : ""}`, c.width), theme.text, true)])
  const db = a.volume_db != null ? `${a.volume_db >= 0 ? "+" : ""}${a.volume_db.toFixed(1)} dB` : a.volume_db_silent ? "-∞ dB" : `${a.volume_percent ?? "—"}%`
  lines.push(meterRow("vol", a.volume_percent != null ? a.volume_percent / 100 : null, db, c.width, 5, levelColor))
  const state = a.muted ? "muted" : a.playing ? "▶ playing" : "nothing playing"
  lines.push([seg(pad("state", 7), theme.textDim), seg(state, a.playing && !a.muted ? theme.ok : theme.textDim),
    seg(a.volume_percent != null && a.volume_db != null ? `  ${a.volume_percent}%` : "", theme.textDim)])
  if (a.signal_peak_dbfs != null || a.signal_rms_dbfs != null) {
    const fmt = (v: number | null | undefined) => (v == null ? "—" : `${v.toFixed(1)}`)
    lines.push([seg(pad("signal", 7), theme.textDim), seg(truncate(`peak ${fmt(a.signal_peak_dbfs)} · rms ${fmt(a.signal_rms_dbfs)} dBFS`, c.width - 7), theme.text)])
  }
  if (a.warning) for (const text of wrap(a.warning, c.width)) lines.push([seg(text, levelColor, true)])
  if (!c.compact && a.note) for (const text of wrap(a.note, c.width)) lines.push([seg(text, theme.hint)])
  return fit(lines, c.height, c.width)
}

// -- ⇧3 disks ------------------------------------------------------------------

export function disksPanel(c: PanelContext): Line[] {
  const rows = [...(c.frame?.disks ?? [])]
  if (!c.frame) return [[seg(truncate("waiting…", c.width), theme.hint)]]
  if (!rows.length) return [[seg(truncate("no disks (or psutil is not installed)", c.width), theme.hint)]]
  const busy = (d: (typeof rows)[number]) => (d.read_bps ?? 0) + (d.write_bps ?? 0)
  if (c.sort === "largest") rows.sort((a, b) => (b.total_bytes ?? 0) - (a.total_bytes ?? 0))
  else if (c.sort === "busiest") rows.sort((a, b) => busy(b) - busy(a))
  else rows.sort((a, b) => (b.percent ?? 0) - (a.percent ?? 0))
  const lines: Line[] = []
  for (const d of rows) {
    lines.push(meterRow(truncateStart(d.mount, 8), (d.percent ?? NaN) / 100, `${fmtBytes(d.used_bytes)}/${fmtBytes(d.total_bytes)}`, c.width, 8))
    if (!c.compact) lines.push([seg(pad("", 9)), seg(truncate(`r ${fmtRate(d.read_bps)}  w ${fmtRate(d.write_bps)}  ${d.device ?? ""} ${d.fstype ?? ""}`, c.width - 9), theme.textDim)])
  }
  return fit(lines, c.height, c.width)
}

// -- ⇧4 network ----------------------------------------------------------------

export function netPanel(c: PanelContext): Line[] {
  const rows = [...(c.frame?.network ?? [])]
  if (!c.frame) return [[seg(truncate("waiting…", c.width), theme.hint)]]
  if (!rows.length) return [[seg(truncate("no network interfaces (or psutil is not installed)", c.width), theme.hint)]]
  if (c.sort === "name") rows.sort((a, b) => a.name.localeCompare(b.name))
  else rows.sort((a, b) => ((b.down_bps ?? 0) + (b.up_bps ?? 0)) - ((a.down_bps ?? 0) + (a.up_bps ?? 0)))
  const lines: Line[] = []
  for (const n of rows) {
    lines.push([seg(pad(n.name, 10), theme.text, true), seg(truncate(`↓ ${fmtRate(n.down_bps)}  ↑ ${fmtRate(n.up_bps)}`, c.width - 10), theme.text)])
    if (!c.compact) lines.push([seg(pad("", 10)), seg(truncate(`total ↓ ${fmtBytes(n.recv_bytes)} ↑ ${fmtBytes(n.sent_bytes)}${n.link_mbps ? `  ${n.link_mbps} Mb/s link` : ""}`, c.width - 10), theme.textDim)])
  }
  return fit(lines, c.height, c.width)
}

// -- ⇧5 system -----------------------------------------------------------------

export function systemPanel(c: PanelContext): Line[] {
  const s = c.frame?.system
  if (!c.frame) return [[seg(truncate("waiting…", c.width), theme.hint)]]
  if (!s) return [[seg(truncate("no system information from the bridge", c.width), theme.hint)]]
  const row = (name: string, value: string, color: string = theme.text): Line => [seg(pad(name, 8), theme.textDim), seg(truncate(value, c.width - 8), color)]
  const lines: Line[] = [row("host", s.hostname ?? "—"), row("os", s.os ?? "—"), row("uptime", fmtDuration(s.uptime_seconds))]
  if (s.load) {
    const busy = s.cores ? s.load[0] / s.cores : 0
    lines.push(row("load", `${s.load.map((v) => v.toFixed(2)).join(" ")}${s.cores ? ` / ${s.cores} cores` : ""}`, busy > 1 ? theme.warn : theme.text))
  }
  if (s.processes != null) lines.push(row("procs", String(s.processes)))
  if (s.battery) {
    const b = s.battery
    lines.push(row("battery", `${(b.percent ?? 0).toFixed(0)}% ${b.plugged ? "charging" : `on battery${b.seconds_left ? `, ${fmtDuration(b.seconds_left)} left` : ""}`}`,
      !b.plugged && (b.percent ?? 100) < 20 ? theme.warn : theme.text))
  }
  return fit(lines, c.height, c.width)
}

// -- sideways scrolling ----------------------------------------------------------

// The width a scrollable panel is drawn at to find how wide it really is.
export const VIRTUAL_WIDTH = 400

export interface HorizontalView {
  lines: Line[]
  offset: number // as clamped
  maxOffset: number // 0: everything fits
}

// A panel's lines seen from `offset` columns in. Panels whose lines are
// not cut to their width (meters stretch instead) never scroll.
export function horizontalView(id: PanelId, c: PanelContext, offset: number): HorizontalView {
  const spec = PANELS.find((p) => p.id === id)
  if (!spec?.scrollX) return { lines: BUILDERS[id](c), offset: 0, maxOffset: 0 }
  const wide = BUILDERS[id]({ ...c, width: VIRTUAL_WIDTH })
  const natural = Math.max(0, ...wide.map((line) => lineText(line).trimEnd().length))
  const maxOffset = Math.max(0, natural - c.width)
  const clamped = Math.max(0, Math.min(offset, maxOffset))
  if (!clamped) return { lines: BUILDERS[id](c), offset: 0, maxOffset }
  return { lines: wide.map((line) => sliceLine(line, clamped, c.width)), offset: clamped, maxOffset }
}

// -- the footer, while tvtop-max is still waiting ---------------------------

// After this long with no reply, the footer says how to see what is wrong.
export const BRIDGE_SLOW_SECONDS = 15

export interface BridgeState {
  startedAt: number // ms
  now: number // ms
  frame: Frame | null
  info: Info | null
  error: string
}

// What tvtop-max is waiting for, or null once it has everything. Shown in
// the footer so a slow start never looks like a blank, broken screen.
export function bridgeStatus(s: BridgeState): string | null {
  if (s.error) return null
  const seconds = Math.max(0, Math.floor((s.now - s.startedAt) / 1000))
  if (!s.frame) {
    const base = `starting the Python bridge… ${seconds}s`
    return seconds >= BRIDGE_SLOW_SECONDS ? `${base} (TVTOP_MAX_DEBUG=1 shows what it runs)` : base
  }
  if (s.frame.discovering) return `looking for the training run… ${seconds}s`
  if (!s.info) return "reading the script…"
  return null
}

export const BUILDERS: Record<PanelId, (c: PanelContext) => Line[]> = {
  cpu: cpuPanel,
  mem: memPanel,
  gpu: gpuPanel,
  train: trainPanel,
  cooker: cookerPanel,
  model: modelPanel,
  logs: logsPanel,
  warnings: warningsPanel,
  modules: modulesPanel,
  procs: procsPanel,
  temps: tempsPanel,
  audio: audioPanel,
  disks: disksPanel,
  net: netPanel,
  system: systemPanel,
}

export function panelTitle(spec: PanelSpec, c: Pick<PanelContext, "info" | "frame">): string {
  if (spec.id === "warnings") {
    const all = allFindings(c.info, c.frame)
    const errors = all.filter((f) => f.level === "error").length
    const warns = all.filter((f) => f.level === "warn").length
    if (errors || warns) return `${spec.title} ${errors}✖ ${warns}▲`
  }
  if (spec.id === "audio" && c.frame?.audio?.level && c.frame.audio.level !== "ok") return `${spec.title} ${c.frame.audio.level === "unsafe" ? "✖" : "▲"}`
  return spec.title
}

// For tests and the plain-text dump: a panel as the text it will show.
export function panelText(id: PanelId, c: PanelContext): string[] {
  return BUILDERS[id](c).map(lineText)
}
