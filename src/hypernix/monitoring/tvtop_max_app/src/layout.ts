// Where the panels go. Pure: which panels, in which rows, how tall.
//
// Wide (a desktop terminal): four rows, the machine on top, the run in
// the middle, its log and warnings below that, and what it is built from
// at the bottom -- read top to bottom, "is the box fine, is the run
// fine, what is it saying, what is it made of".
//
// Phone (-s): one column, sized for a phone's SSH app (Termius, Blink,
// a-Shell: 40-56 columns in portrait), scrolled with the thumb or j/k.
// Ordered by what someone checking a run from their phone wants first:
// is it still going, is anything wrong, then everything else.

import type { PanelId } from "./panels.ts"

export type Mode = "wide" | "phone"

export interface Row {
  panels: { id: PanelId; grow: number }[]
  grow: number
}

// The widest a phone layout gets, even on a wide terminal, so `-s` on a
// desktop previews what the phone will show.
export const PHONE_MAX_WIDTH = 56

const WIDE: Row[] = [
  { grow: 3, panels: [{ id: "cpu", grow: 3 }, { id: "mem", grow: 2 }, { id: "gpu", grow: 3 }] },
  { grow: 3, panels: [{ id: "train", grow: 3 }, { id: "cooker", grow: 2 }, { id: "model", grow: 2 }] },
  { grow: 4, panels: [{ id: "logs", grow: 5 }, { id: "warnings", grow: 3 }] },
  { grow: 3, panels: [{ id: "modules", grow: 3 }, { id: "procs", grow: 3 }] },
]

export const PHONE_ORDER: PanelId[] = ["train", "warnings", "gpu", "cpu", "mem", "cooker", "model", "logs", "modules", "procs"]

// Rows of lines each panel gets in the phone column, borders included.
export const PHONE_HEIGHT: Record<PanelId, number> = {
  train: 9,
  warnings: 12,
  gpu: 7,
  cpu: 6,
  mem: 4,
  cooker: 9,
  model: 12,
  logs: 16,
  modules: 16,
  procs: 9,
}

export function wideRows(visible: Set<PanelId>): Row[] {
  return WIDE.map((row) => ({ ...row, panels: row.panels.filter((p) => visible.has(p.id)) })).filter((row) => row.panels.length)
}

export function phoneColumn(visible: Set<PanelId>): { id: PanelId; height: number }[] {
  return PHONE_ORDER.filter((id) => visible.has(id)).map((id) => ({ id, height: PHONE_HEIGHT[id] }))
}

export function phoneWidth(terminalWidth: number): number {
  return Math.max(24, Math.min(terminalWidth, PHONE_MAX_WIDTH))
}

// Below this a wide layout's three-across rows cannot show a meter, so
// tvtop-max switches to the phone column on its own.
export const AUTO_PHONE_BELOW = 72

export function chooseMode(requested: Mode | "auto", terminalWidth: number): Mode {
  if (requested !== "auto") return requested
  return terminalWidth < AUTO_PHONE_BELOW ? "phone" : "wide"
}
