// Pure formatting: text in, text out. Everything a panel draws is built
// from these, and none of them touch the terminal, so the tests can check
// exactly what a panel will show.

import { theme } from "./theme.ts"

// One run of text in one style. A panel body is a list of lines, each a
// list of segments; app.ts turns that into OpenTUI's StyledText.
export interface Seg {
  text: string
  fg?: string
  bold?: boolean
}
export type Line = Seg[]

export const seg = (text: string, fg?: string, bold = false): Seg => (bold ? { text, fg, bold } : { text, fg })

export function lineText(line: Line): string {
  return line.map((s) => s.text).join("")
}

export function truncate(text: string, width: number): string {
  if (width <= 0) return ""
  if (text.length <= width) return text
  if (width === 1) return "…"
  return text.slice(0, width - 1) + "…"
}

export function pad(text: string, width: number): string {
  const cut = truncate(text, width)
  return cut + " ".repeat(Math.max(0, width - cut.length))
}

// Keeps the end, which is the part that differs between two long paths.
export function truncateStart(text: string, width: number): string {
  if (text.length <= width) return text
  if (width <= 1) return "…"
  return "…" + text.slice(text.length - width + 1)
}

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "")
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)]
}

function rgbToHex([r, g, b]: [number, number, number]): string {
  return "#" + [r, g, b].map((v) => Math.round(v).toString(16).padStart(2, "0")).join("")
}

// Colour at position t (0..1) along low -> mid -> high.
export function rampColor(t: number, low = theme.meterLow, mid = theme.meterMid, high = theme.meterHigh): string {
  const x = Math.min(1, Math.max(0, t))
  const [a, b, u] = x < 0.5 ? [low, mid, x * 2] : [mid, high, (x - 0.5) * 2]
  const ca = hexToRgb(a)
  const cb = hexToRgb(b)
  return rgbToHex([0, 1, 2].map((i) => ca[i] + (cb[i] - ca[i]) * u) as [number, number, number])
}

// A btop-style meter: filled cells coloured by their own position.
export function meter(fraction: number | null | undefined, width: number): Line {
  if (width <= 0) return []
  if (fraction === null || fraction === undefined || Number.isNaN(fraction)) {
    return [seg("·".repeat(width), theme.hint)]
  }
  const f = Math.min(1, Math.max(0, fraction))
  const filled = Math.round(f * width)
  const out: Line = []
  for (let i = 0; i < filled; i++) out.push(seg("■", rampColor(width > 1 ? i / (width - 1) : 1)))
  if (width - filled > 0) out.push(seg("■".repeat(width - filled), theme.border))
  return out
}

// Braille graph: two samples per cell across, four levels per cell up.
// Returns `height` rows, top first. The newest sample is on the right.
export function brailleGraph(values: number[], width: number, height: number, max = 100): string[] {
  if (width <= 0 || height <= 0) return []
  const samples = values.slice(-width * 2)
  const padded = new Array(width * 2 - samples.length).fill(NaN).concat(samples)
  const levels = height * 4
  const top = max > 0 ? max : Math.max(1e-9, ...samples.filter((v) => Number.isFinite(v)))
  const heights = padded.map((v) => (Number.isFinite(v) ? Math.round((Math.min(Math.max(v, 0), top) / top) * levels) : 0))
  // Dot bits for (column, row-from-bottom-within-cell).
  const left = [0x40, 0x04, 0x02, 0x01]
  const right = [0x80, 0x20, 0x10, 0x08]
  const rows: string[] = []
  for (let row = height - 1; row >= 0; row--) {
    let line = ""
    for (let cell = 0; cell < width; cell++) {
      let bits = 0
      for (let dot = 0; dot < 4; dot++) {
        const level = row * 4 + dot + 1
        if (heights[cell * 2] >= level) bits |= left[dot]
        if (heights[cell * 2 + 1] >= level) bits |= right[dot]
      }
      line += String.fromCharCode(0x2800 + bits)
    }
    rows.push(line)
  }
  return rows
}

export function sparkline(values: number[], width: number): string {
  const blocks = "▁▂▃▄▅▆▇█"
  const samples = values.filter((v) => Number.isFinite(v)).slice(-width)
  if (!samples.length) return ""
  const lo = Math.min(...samples)
  const hi = Math.max(...samples)
  return samples.map((v) => blocks[hi > lo ? Math.round(((v - lo) / (hi - lo)) * 7) : 3]).join("")
}

export function fmtBytes(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—"
  const units = ["B", "K", "M", "G", "T"]
  let v = value
  let u = 0
  while (v >= 1024 && u < units.length - 1) {
    v /= 1024
    u++
  }
  return `${v >= 100 || u === 0 ? v.toFixed(0) : v.toFixed(1)}${units[u]}`
}

export function fmtMiB(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : fmtBytes(value * 1024 * 1024)
}

export function fmtDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "—"
  const s = Math.max(0, Math.round(seconds))
  const d = Math.floor(s / 86400)
  const h = Math.floor((s % 86400) / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  if (d) return `${d}d ${h}h`
  if (h) return `${h}h ${String(m).padStart(2, "0")}m`
  if (m) return `${m}m ${String(sec).padStart(2, "0")}s`
  return `${sec}s`
}

// 66331136 -> "66.3M"
export function fmtCount(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "—"
  const abs = Math.abs(n)
  if (abs >= 1e12) return `${(n / 1e12).toFixed(2)}T`
  if (abs >= 1e9) return `${(n / 1e9).toFixed(2)}B`
  if (abs >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (abs >= 1e3) return `${(n / 1e3).toFixed(1)}K`
  return String(n)
}

export function fmtNumber(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—"
  if (value !== 0 && (Math.abs(value) < 1e-3 || Math.abs(value) >= 1e5)) return value.toExponential(2)
  return Number(value.toFixed(digits)).toString()
}

export function fmtValue(value: unknown): string {
  if (value === null || value === undefined) return "—"
  if (typeof value === "number") return fmtNumber(value)
  if (typeof value === "string") return value
  return JSON.stringify(value)
}

// Wraps on spaces; a word longer than the width is cut.
export function wrap(text: string, width: number): string[] {
  if (width <= 0) return []
  const out: string[] = []
  for (const paragraph of text.split("\n")) {
    let line = ""
    for (const word of paragraph.split(/\s+/).filter(Boolean)) {
      if (!line) line = word
      else if (line.length + 1 + word.length <= width) line += " " + word
      else {
        out.push(line)
        line = word
      }
      while (line.length > width) {
        out.push(line.slice(0, width))
        line = line.slice(width)
      }
    }
    out.push(line)
  }
  return out
}
