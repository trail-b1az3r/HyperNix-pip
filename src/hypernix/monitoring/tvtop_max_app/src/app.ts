// The tvtop-max screen. Panels are bordered boxes with their number and
// name in the border, btop-style; their bodies come from panels.ts and
// their places from layout.ts, so this file only builds boxes, feeds them,
// and handles keys.

import {
  BoxRenderable,
  ScrollBoxRenderable,
  StyledText,
  TextRenderable,
  bold,
  fg,
  type CliRenderer,
  type KeyEvent,
  type TextChunk,
} from "@opentui/core"

import { Bridge, BridgeError } from "./bridge.ts"
import { truncate, type Line } from "./format.ts"
import { chooseMode, drawOrder, phoneColumn, phoneWidth, wideRows, type Mode } from "./layout.ts"
import { bridgeStatus, horizontalView, PANELS, panelForKey, panelTitle, type PanelId, type PanelSpec } from "./panels.ts"
import { theme } from "./theme.ts"
import type { Frame, Info } from "./types.ts"

export const VERSION = "0.72.8-a0"

export interface AppOptions {
  mode: Mode | "auto"
  refreshMs: number
  hidden?: PanelId[]
  // Panels hidden by default (Shift+1 … Shift+5) to show from the start.
  shown?: PanelId[]
}

export const KEYS_WIDE = "q quit · 0-9 ⇧1-5 panels · ← select · → ⇧→ scroll · o sort · d deselect · s phone · p pause · ↑↓ log"
export const KEYS_PHONE = "q quit · ↑↓ scroll · ← select · o sort · d deselect · s wide · p pause"

// Columns → and Shift+→ move a panel's view by.
export const SCROLL_STEP = 8

// Which panels start visible: every one not hidden by default, less
// --hide, plus --show.
export function initialVisible(hidden: PanelId[] = [], shown: PanelId[] = []): Set<PanelId> {
  return new Set(PANELS.filter((p) => (!p.hidden || shown.includes(p.id)) && !hidden.includes(p.id)).map((p) => p.id))
}

// What the selection, sorting and sideways scrolling keys do, kept apart
// from the screen so it can be tested.
export class Selection {
  selected: PanelId | null = null
  private sortIndex = new Map<PanelId, number>()
  private offsets = new Map<PanelId, number>()
  private limits = new Map<PanelId, number>()

  // ← moves to the next visible panel in drawing order (Shift+← back).
  cycle(order: PanelId[], backwards = false): PanelId | null {
    if (!order.length) return (this.selected = null)
    const at = this.selected ? order.indexOf(this.selected) : -1
    const next = at < 0 ? (backwards ? order.length - 1 : 0) : (at + (backwards ? order.length - 1 : 1)) % order.length
    return (this.selected = order[next])
  }

  clear(): void {
    this.selected = null
  }

  sort(id: PanelId): string | undefined {
    const spec = PANELS.find((p) => p.id === id)
    return spec?.sorts?.[(this.sortIndex.get(id) ?? 0) % spec.sorts.length]
  }

  // `o`: the selected panel's next sort, or why there is none.
  nextSort(): { ok: boolean; message: string } {
    if (!this.selected) return { ok: false, message: "select a panel with ← first, then o sorts it" }
    const spec = PANELS.find((p) => p.id === this.selected) as PanelSpec
    if (!spec.sorts?.length) return { ok: false, message: `${spec.title} has nothing to sort` }
    this.sortIndex.set(spec.id, (this.sortIndex.get(spec.id) ?? 0) + 1)
    return { ok: true, message: `${spec.title}: sorted by ${this.sort(spec.id)}` }
  }

  offset(id: PanelId): number {
    return this.offsets.get(id) ?? 0
  }

  // What the last draw found: how far the panel can scroll.
  setLimit(id: PanelId, maxOffset: number, offset: number): void {
    this.limits.set(id, maxOffset)
    this.offsets.set(id, offset)
  }

  // → (or Shift+→, the other way) on the selected panel.
  scroll(back: boolean): { ok: boolean; message: string } {
    if (!this.selected) return { ok: false, message: "select a panel with ← first, then → scrolls it" }
    const spec = PANELS.find((p) => p.id === this.selected) as PanelSpec
    const max = this.limits.get(spec.id) ?? 0
    if (!spec.scrollX || max <= 0) return { ok: false, message: `${spec.title} fits: nothing to scroll` }
    const now = this.offset(spec.id)
    this.offsets.set(spec.id, Math.max(0, Math.min(max, now + (back ? -SCROLL_STEP : SCROLL_STEP))))
    return { ok: true, message: "" }
  }
}

export function styled(lines: Line[]): StyledText {
  const chunks: TextChunk[] = []
  lines.forEach((line, index) => {
    if (index) chunks.push(fg(theme.text)("\n"))
    for (const s of line) {
      const chunk = fg(s.fg ?? theme.text)(s.text)
      chunks.push(s.bold ? bold(chunk) : chunk)
    }
  })
  return new StyledText(chunks)
}

type PanelView = { box: BoxRenderable; text: TextRenderable }

export class App {
  private info: Info | null = null
  private frame: Frame | null = null
  private visible: Set<PanelId>
  private mode: Mode
  private paused = false
  private logOffset = 0
  private error = ""
  private selection = new Selection()
  private flash: { text: string; until: number } | null = null
  private quitting = false
  private timer: ReturnType<typeof setInterval> | null = null
  private inFlight = false
  private readonly startedAt = Date.now()

  private header!: TextRenderable
  private footer!: TextRenderable
  private body: BoxRenderable | ScrollBoxRenderable | null = null
  private root!: BoxRenderable
  private views = new Map<PanelId, PanelView>()

  constructor(
    private renderer: CliRenderer,
    private bridge: Bridge,
    private options: AppOptions,
  ) {
    this.visible = initialVisible(options.hidden, options.shown)
    this.mode = chooseMode(options.mode, renderer.width)
    this.bridge.onLog = () => {} // the bridge's stderr stays off the screen
  }

  async start(): Promise<void> {
    this.renderer.setBackgroundColor(theme.background)
    this.build()
    this.renderer.keyInput.on("keypress", (key: KeyEvent) => this.onKey(key))
    // Drawn now, before the bridge has said anything: the first frame
    // can take seconds, and a screen of empty boxes with no header
    // looked broken. The footer says what it is waiting for.
    this.draw()
    void this.loadInfo()
    void this.tick()
    this.timer = setInterval(() => void this.tick(), this.options.refreshMs)
  }

  // -- layout ----------------------------------------------------------------

  private build(): void {
    const r = this.renderer
    this.root = new BoxRenderable(r, {
      id: "root",
      flexDirection: "column",
      width: "100%",
      height: "100%",
      backgroundColor: theme.background,
    })
    this.header = new TextRenderable(r, { id: "header", content: "", height: 1, flexShrink: 0 })
    this.footer = new TextRenderable(r, { id: "footer", content: "", fg: theme.hint, height: 1, flexShrink: 0 })
    this.root.add(this.header)
    this.root.add(this.footer)
    r.root.add(this.root)
    this.buildBody()
  }

  private buildBody(): void {
    const r = this.renderer
    if (this.body) {
      this.root.remove(this.body)
      this.body.destroyRecursively()
    }
    this.views.clear()
    if (this.mode === "phone") {
      const width = phoneWidth(r.width)
      const scroll = new ScrollBoxRenderable(r, {
        id: "body",
        flexGrow: 1,
        // Without these the box takes its content's full height and the
        // footer (keys, and what tvtop-max is waiting for) is pushed off
        // the bottom of the screen.
        flexShrink: 1,
        minHeight: 0,
        width,
        contentOptions: { flexDirection: "column" },
      })
      for (const { id, height } of phoneColumn(this.visible)) scroll.add(this.panelBox(id, { height, flexShrink: 0 }))
      this.body = scroll
    } else {
      const column = new BoxRenderable(r, { id: "body", flexGrow: 1, flexDirection: "column" })
      wideRows(this.visible).forEach((row, index) => {
        const box = new BoxRenderable(r, { id: `row-${index}`, flexDirection: "row", flexGrow: row.grow, flexBasis: 0 })
        for (const panel of row.panels) box.add(this.panelBox(panel.id, { flexGrow: panel.grow, flexBasis: 0 }))
        column.add(box)
      })
      this.body = column
    }
    // Between the header and the footer.
    this.root.add(this.body, 1)
  }

  private panelBox(id: PanelId, size: Record<string, number>): BoxRenderable {
    const r = this.renderer
    const spec = PANELS.find((p) => p.id === id)!
    const box = new BoxRenderable(r, {
      id: `panel-${id}`,
      border: true,
      borderStyle: "rounded",
      borderColor: theme.border,
      title: ` ${spec.key} ${spec.title} `,
      titleColor: theme.title,
      backgroundColor: theme.background,
      flexDirection: "column",
      overflow: "hidden",
      ...size,
    })
    const text = new TextRenderable(r, { id: `panel-${id}-text`, content: "", flexGrow: 1 })
    box.add(text)
    this.views.set(id, { box, text })
    return box
  }

  // -- data ------------------------------------------------------------------

  private async loadInfo(): Promise<void> {
    try {
      this.info = await this.bridge.call<Info>("info")
      this.error = ""
    } catch (error) {
      this.error = error instanceof BridgeError ? error.message : String(error)
    }
    this.draw()
  }

  private async tick(): Promise<void> {
    if (this.paused || this.inFlight || this.quitting) {
      this.draw()
      return
    }
    this.inFlight = true
    try {
      this.frame = await this.bridge.call<Frame>("frame")
      this.error = ""
    } catch (error) {
      this.error = error instanceof BridgeError ? error.message : String(error)
    } finally {
      this.inFlight = false
    }
    this.draw()
  }

  private draw(): void {
    if (this.quitting) return
    const width = this.renderer.width
    const script = this.info?.script ? this.info.script.split("/").pop() : "no script"
    const log = this.info?.log ? this.info.log.split("/").pop() : "no log"
    const where = this.info
      ? `${script} · ${log}${this.info.pid ? ` · pid ${this.info.pid}` : ""}`
      : "looking for the run…"
    // A loud or unsafe volume is said at the top, whichever panels are shown.
    const audio = this.frame?.audio
    const loud = audio?.warning && audio.level && audio.level !== "ok" ? audio : null
    this.header.content = styled([[
      { text: " tvtop-max ", fg: theme.accentText, bold: true },
      { text: `v${VERSION} `, fg: theme.hint },
      loud
        ? { text: truncate(`🔊 ${loud.warning}`, Math.max(0, width - 24)), fg: loud.level === "unsafe" ? theme.error : theme.warn, bold: true }
        : { text: truncate(where, Math.max(0, width - 24)), fg: theme.textDim },
      ...(this.paused ? [{ text: "  paused", fg: theme.warn }] : []),
    ]])
    const keys = this.mode === "phone" ? KEYS_PHONE : KEYS_WIDE
    const waiting = bridgeStatus({
      startedAt: this.startedAt, now: Date.now(), frame: this.frame, info: this.info, error: this.error,
    })
    const flash = this.flash && this.flash.until > Date.now() ? this.flash.text : ""
    this.footer.content = styled([[
      this.error
        ? { text: truncate(` ${this.error}`, width), fg: theme.error }
        : flash
          ? { text: truncate(` ${flash}`, width), fg: theme.accentText }
          : waiting
            ? { text: truncate(` ${waiting}`, width), fg: theme.warn }
            : { text: truncate(` ${keys}`, width), fg: theme.hint },
    ]])
    for (const spec of PANELS) {
      const view = this.views.get(spec.id)
      if (!view) continue
      const inner = { width: Math.max(4, view.box.width - 2), height: Math.max(1, view.box.height - 2) }
      const sort = this.selection.sort(spec.id)
      const shown = horizontalView(spec.id, {
        info: this.info,
        frame: this.frame,
        width: inner.width,
        height: inner.height,
        compact: this.mode === "phone",
        logOffset: this.logOffset,
        sort,
      }, this.selection.offset(spec.id))
      this.selection.setLimit(spec.id, shown.maxOffset, shown.offset)
      view.text.content = styled(shown.lines)
      const selected = this.selection.selected === spec.id
      const sorted = sort && sort !== spec.sorts?.[0] ? ` · ${sort}` : ""
      const scrolled = shown.maxOffset ? (shown.offset ? ` ⇆ ${shown.offset}/${shown.maxOffset}` : " ⇆") : ""
      view.box.title = ` ${selected ? "▸ " : ""}${spec.key} ${panelTitle(spec, { info: this.info, frame: this.frame })}${sorted}${scrolled} `
      view.box.borderColor = selected ? theme.borderFocused : theme.border
    }
    this.renderer.requestRender()
  }

  // -- keys ------------------------------------------------------------------

  private onKey(key: KeyEvent): void {
    const name = key.name
    if ((key.ctrl && name === "c") || name === "q" || name === "escape") {
      this.quit()
      return
    }
    const spec = panelForKey(name, key.shift)
    if (spec) {
      if (this.visible.has(spec.id)) {
        this.visible.delete(spec.id)
        if (this.selection.selected === spec.id) this.selection.clear()
      } else {
        this.visible.add(spec.id)
      }
      this.buildBody()
    } else if (name === "left") {
      const id = this.selection.cycle(drawOrder(this.mode, this.visible), key.shift)
      const picked = PANELS.find((p) => p.id === id)
      this.say(picked ? `${picked.title} selected${picked.sorts ? " · o sorts" : ""}${picked.scrollX ? " · → scrolls" : ""}` : "no panels shown")
    } else if (name === "right") {
      const result = this.selection.scroll(key.shift)
      if (result.message) this.say(result.message)
    } else if (name === "o") {
      this.say(this.selection.nextSort().message)
    } else if (name === "d") {
      this.selection.clear()
      this.say("nothing selected")
    } else if (name === "s") {
      this.mode = this.mode === "phone" ? "wide" : "phone"
      this.buildBody()
    } else if (name === "p") {
      this.paused = !this.paused
    } else if (name === "r") {
      this.info = null
      void this.bridge.call("rescan").then(() => this.loadInfo(), () => this.loadInfo())
    } else if (this.mode === "phone" && this.body instanceof ScrollBoxRenderable) {
      const step = name === "pagedown" || name === "space" ? 10 : name === "pageup" ? -10 : name === "down" || name === "j" ? 2 : name === "up" || name === "k" ? -2 : 0
      if (step) this.body.scrollBy(step)
      if (name === "home" || name === "g") this.body.scrollTop = 0
    } else {
      const lines = this.frame?.log_lines?.length ?? 0
      if (name === "up" || name === "k") this.logOffset = Math.min(lines, this.logOffset + 1)
      else if (name === "down" || name === "j") this.logOffset = Math.max(0, this.logOffset - 1)
      else if (name === "pageup") this.logOffset = Math.min(lines, this.logOffset + 10)
      else if (name === "pagedown") this.logOffset = Math.max(0, this.logOffset - 10)
      else if (name === "end" || name === "f") this.logOffset = 0
    }
    this.draw()
  }

  private say(text: string): void {
    this.flash = { text, until: Date.now() + 2500 }
    setTimeout(() => this.draw(), 2600)
  }

  quit(): void {
    if (this.quitting) return
    this.quitting = true
    if (this.timer) clearInterval(this.timer)
    this.bridge.close()
    this.renderer.destroy()
    process.exit(0)
  }
}
