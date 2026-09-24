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
import { chooseMode, phoneColumn, phoneWidth, wideRows, type Mode } from "./layout.ts"
import { BUILDERS, PANELS, panelTitle, type PanelId } from "./panels.ts"
import { theme } from "./theme.ts"
import type { Frame, Info } from "./types.ts"

export const VERSION = "0.72.6-rc3"

export interface AppOptions {
  mode: Mode | "auto"
  refreshMs: number
  hidden?: PanelId[]
}

export const KEYS_WIDE = "q quit · 0-9 panels · s phone · p pause · r rescan · ↑↓ log · end follow"
export const KEYS_PHONE = "q quit · ↑↓ scroll · s wide · p pause · r rescan"

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
  private quitting = false
  private timer: ReturnType<typeof setInterval> | null = null
  private inFlight = false

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
    this.visible = new Set(PANELS.map((p) => p.id).filter((id) => !(options.hidden ?? []).includes(id)))
    this.mode = chooseMode(options.mode, renderer.width)
    this.bridge.onLog = () => {} // the bridge's stderr stays off the screen
  }

  async start(): Promise<void> {
    this.renderer.setBackgroundColor(theme.background)
    this.build()
    this.renderer.keyInput.on("keypress", (key: KeyEvent) => this.onKey(key))
    void this.loadInfo()
    await this.tick()
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
        width,
        contentOptions: { flexDirection: "column" },
      })
      for (const { id, height } of phoneColumn(this.visible)) scroll.add(this.panelBox(id, { height, width }))
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
    const where = `${script} · ${log}${this.info?.pid ? ` · pid ${this.info.pid}` : ""}`
    this.header.content = styled([[
      { text: " tvtop-max ", fg: theme.accentText, bold: true },
      { text: `v${VERSION} `, fg: theme.hint },
      { text: truncate(where, Math.max(0, width - 24)), fg: theme.textDim },
      ...(this.paused ? [{ text: "  paused", fg: theme.warn }] : []),
    ]])
    const keys = this.mode === "phone" ? KEYS_PHONE : KEYS_WIDE
    this.footer.content = styled([[
      this.error
        ? { text: truncate(` ${this.error}`, width), fg: theme.error }
        : { text: truncate(` ${keys}`, width), fg: theme.hint },
    ]])
    for (const spec of PANELS) {
      const view = this.views.get(spec.id)
      if (!view) continue
      const inner = { width: Math.max(4, view.box.width - 2), height: Math.max(1, view.box.height - 2) }
      const lines = BUILDERS[spec.id]({
        info: this.info,
        frame: this.frame,
        width: inner.width,
        height: inner.height,
        compact: this.mode === "phone",
        logOffset: this.logOffset,
      })
      view.text.content = styled(lines)
      view.box.title = ` ${spec.key} ${panelTitle(spec, { info: this.info, frame: this.frame })} `
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
    const spec = PANELS.find((p) => p.key === name)
    if (spec) {
      if (this.visible.has(spec.id)) this.visible.delete(spec.id)
      else this.visible.add(spec.id)
      this.buildBody()
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

  quit(): void {
    if (this.quitting) return
    this.quitting = true
    if (this.timer) clearInterval(this.timer)
    this.bridge.close()
    this.renderer.destroy()
    process.exit(0)
  }
}
