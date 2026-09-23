// The hyped-pro screen, laid out the way opencode's is: a header, the
// conversation filling the middle, a bordered prompt at the bottom and a
// line of key hints under it. Colours come from theme.ts, which is the
// site's palette.

import {
  BoxRenderable,
  InputRenderable,
  InputRenderableEvents,
  MarkdownRenderable,
  ScrollBoxRenderable,
  SelectRenderable,
  SelectRenderableEvents,
  SyntaxStyle,
  TextAttributes,
  TextRenderable,
  type CliRenderer,
  type KeyEvent,
  type Renderable,
} from "@opentui/core"

import { Bridge, BridgeError } from "./bridge.ts"
import { COMMANDS, completions, parseInput } from "./commands.ts"
import { loadPrefs, savePrefs, type Prefs } from "./prefs.ts"
import {
  describeNoodleEvent,
  findModel,
  modelLabel,
  needsDownload,
  Session,
  type Catalog,
  type Entry,
  type ModelInfo,
} from "./session.ts"
import { theme } from "./theme.ts"

export const VERSION = "0.72.5-post16"

const SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

const LOGO = [
  "█ █ █ █ █▀█ █▀▀ █▀▄   █▀█ █▀█ █▀█",
  "█▀█  █  █▀▀ ██▄ █▄▀   █▀▀ █▀▄ █▄█",
]

export interface AppOptions {
  model?: string
}

export class App {
  private session = new Session()
  private catalog: Catalog | null = null
  private model: ModelInfo | undefined
  private prefs: Prefs
  private busy: { label: string; requestId?: number; noodle?: string } | null = null
  private spinnerFrame = 0
  private spinnerTimer: ReturnType<typeof setInterval> | null = null
  private showThinking = false
  private enableTools = true
  private lastLog = ""
  private quitting = false

  private readonly markdownStyle: SyntaxStyle
  private header!: TextRenderable
  private headerModel!: TextRenderable
  private transcript!: ScrollBoxRenderable
  private promptBox!: BoxRenderable
  private input!: InputRenderable
  private hint!: TextRenderable
  private footerRight!: TextRenderable
  private pickerBox!: BoxRenderable
  private picker!: SelectRenderable

  constructor(
    private renderer: CliRenderer,
    private bridge: Bridge,
    private options: AppOptions = {},
  ) {
    this.prefs = loadPrefs()
    this.showThinking = this.prefs.showThinking ?? false
    this.enableTools = this.prefs.enableTools ?? true
    this.markdownStyle = SyntaxStyle.fromStyles({
      default: { fg: theme.text },
      "markup.heading": { fg: theme.accentText, bold: true },
      "markup.strong": { fg: theme.text, bold: true },
      "markup.italic": { fg: theme.text, italic: true },
      "markup.strikethrough": { fg: theme.textMuted },
      "markup.raw": { fg: theme.accentText },
      "markup.raw.block": { fg: theme.textDim },
      "markup.quote": { fg: theme.textMuted, italic: true },
      "markup.list": { fg: theme.accent },
      "markup.link": { fg: theme.accentText, underline: true },
      "markup.link.label": { fg: theme.accentText },
      "markup.link.url": { fg: theme.textMuted, underline: true },
      keyword: { fg: theme.accentText },
      string: { fg: theme.ok },
      comment: { fg: theme.hint, italic: true },
      number: { fg: theme.accentText },
      function: { fg: theme.text, bold: true },
      type: { fg: theme.textDim },
    })
    this.bridge.onLog = (line) => this.onBridgeLog(line)
  }

  async start(): Promise<void> {
    this.renderer.setBackgroundColor(theme.background)
    this.build()
    this.renderer.keyInput.on("keypress", (key: KeyEvent) => this.onKey(key))
    this.welcome()
    this.input.focus()
    await this.loadCatalog()
  }

  // -- layout -------------------------------------------------------------

  private build(): void {
    const r = this.renderer
    const root = new BoxRenderable(r, {
      id: "root",
      flexDirection: "column",
      width: "100%",
      height: "100%",
      backgroundColor: theme.background,
      paddingLeft: 2,
      paddingRight: 2,
      paddingTop: 1,
    })

    const header = new BoxRenderable(r, { id: "header", flexDirection: "row", height: 1, flexShrink: 0 })
    this.header = new TextRenderable(r, {
      id: "header-title",
      content: "hyped-pro",
      fg: theme.accentText,
      attributes: TextAttributes.BOLD,
    })
    const spacer = new BoxRenderable(r, { id: "header-spacer", flexGrow: 1 })
    this.headerModel = new TextRenderable(r, { id: "header-model", content: "", fg: theme.textDim })
    header.add(this.header)
    header.add(new TextRenderable(r, { id: "header-version", content: `  v${VERSION}`, fg: theme.hint }))
    header.add(spacer)
    header.add(this.headerModel)

    this.transcript = new ScrollBoxRenderable(r, {
      id: "transcript",
      flexGrow: 1,
      stickyScroll: true,
      stickyStart: "bottom",
      marginTop: 1,
      contentOptions: { flexDirection: "column", gap: 1, paddingRight: 1 },
      scrollbarOptions: { trackOptions: { foregroundColor: theme.borderFocused, backgroundColor: theme.panel } },
    })

    this.pickerBox = new BoxRenderable(r, {
      id: "picker",
      border: true,
      borderStyle: "rounded",
      borderColor: theme.accent,
      title: " models ",
      titleColor: theme.accentText,
      backgroundColor: theme.panel,
      height: 14,
      flexShrink: 0,
      visible: false,
    })
    this.picker = new SelectRenderable(r, {
      id: "picker-list",
      flexGrow: 1,
      backgroundColor: theme.panel,
      focusedBackgroundColor: theme.panel,
      textColor: theme.text,
      focusedTextColor: theme.text,
      selectedBackgroundColor: theme.selection,
      selectedTextColor: theme.accentText,
      descriptionColor: theme.hint,
      selectedDescriptionColor: theme.textDim,
      showDescription: true,
      showScrollIndicator: true,
      wrapSelection: true,
    })
    this.picker.on(SelectRenderableEvents.ITEM_SELECTED, (_index: number, option: { value?: any }) => {
      if (option?.value) this.selectModel(option.value as ModelInfo)
      this.closePicker()
    })
    this.pickerBox.add(this.picker)

    this.promptBox = new BoxRenderable(r, {
      id: "prompt",
      border: true,
      borderStyle: "rounded",
      borderColor: theme.borderFocused,
      backgroundColor: theme.panel,
      height: 3,
      flexShrink: 0,
      marginTop: 1,
      paddingLeft: 1,
      paddingRight: 1,
      title: "",
      titleColor: theme.textDim,
    })
    this.input = new InputRenderable(r, {
      id: "prompt-input",
      flexGrow: 1,
      placeholder: "Ask anything — / for commands",
      backgroundColor: theme.panel,
      focusedBackgroundColor: theme.panel,
      textColor: theme.text,
      focusedTextColor: theme.text,
      placeholderColor: theme.hint,
    })
    this.input.on(InputRenderableEvents.ENTER, () => this.submit())
    this.input.on(InputRenderableEvents.INPUT, () => this.refreshHint())
    this.promptBox.add(this.input)

    this.hint = new TextRenderable(r, { id: "hint", content: "", fg: theme.hint, height: 1, flexShrink: 0 })

    const footer = new BoxRenderable(r, { id: "footer", flexDirection: "row", height: 1, flexShrink: 0 })
    const keys = new TextRenderable(r, {
      id: "footer-keys",
      content: "enter send · esc stop · ctrl+p models · ctrl+n new · /help",
      fg: theme.hint,
    })
    this.footerRight = new TextRenderable(r, { id: "footer-right", content: "", fg: theme.hint, flexShrink: 0 })
    footer.add(keys)
    footer.add(new BoxRenderable(r, { id: "footer-spacer", flexGrow: 1 }))
    footer.add(this.footerRight)

    root.add(header)
    root.add(this.transcript)
    root.add(this.pickerBox)
    root.add(this.promptBox)
    root.add(this.hint)
    root.add(footer)
    r.root.add(root)
    this.refreshHeader()
  }

  // -- transcript -----------------------------------------------------------

  private welcome(): void {
    const r = this.renderer
    const box = new BoxRenderable(r, { id: `welcome-${Date.now()}`, flexDirection: "column", paddingTop: 1 })
    for (const [i, line] of LOGO.entries()) {
      box.add(new TextRenderable(r, { id: `logo-${i}-${Date.now()}`, content: line, fg: i === 0 ? theme.accentText : theme.accent }))
    }
    box.add(
      new TextRenderable(r, {
        id: `welcome-text-${Date.now()}`,
        content: "\nThe HyperNix terminal client. Type a message, or /help for commands.",
        fg: theme.textDim,
      }),
    )
    this.transcript.add(box)
  }

  private renderEntry(entry: Entry): Renderable {
    const r = this.renderer
    const id = `entry-${entry.id}`
    if (entry.role === "user") {
      const box = new BoxRenderable(r, {
        id,
        border: ["left"],
        borderColor: theme.accent,
        backgroundColor: theme.userMessage,
        paddingLeft: 1,
        paddingTop: 0,
        paddingBottom: 0,
      })
      box.add(new TextRenderable(r, { id: `${id}-text`, content: entry.text, fg: theme.text, wrapMode: "word" }))
      return box
    }
    if (entry.role === "note" || entry.role === "error") {
      return new TextRenderable(r, {
        id,
        content: entry.text,
        fg: entry.role === "error" ? theme.error : theme.textDim,
        wrapMode: "word",
        paddingLeft: 2,
      })
    }

    const box = new BoxRenderable(r, { id, flexDirection: "column", paddingLeft: 2 })
    if (entry.thinking && this.showThinking) {
      box.add(
        new TextRenderable(r, {
          id: `${id}-thinking`,
          content: entry.thinking.trim(),
          fg: theme.textMuted,
          attributes: TextAttributes.ITALIC,
          wrapMode: "word",
          marginBottom: 1,
        }),
      )
    }
    box.add(this.markdown(`${id}-body`, entry.text))
    const meta = [entry.model ?? "", entry.cancelled ? "stopped" : ""].filter(Boolean).join(" · ")
    if (meta) box.add(new TextRenderable(r, { id: `${id}-meta`, content: meta, fg: theme.hint, marginTop: 1 }))
    return box
  }

  private markdown(id: string, text: string): Renderable {
    try {
      return new MarkdownRenderable(this.renderer, {
        id,
        content: text,
        syntaxStyle: this.markdownStyle,
        fg: theme.text,
        conceal: true,
      })
    } catch {
      return new TextRenderable(this.renderer, { id, content: text, fg: theme.text, wrapMode: "word" })
    }
  }

  private show(entry: Entry): void {
    this.transcript.add(this.renderEntry(entry))
  }

  private note(text: string): void {
    this.show(this.session.add("note", text))
  }

  private error(text: string, code?: string): void {
    this.show(this.session.add("error", code ? `${code}  ${text}` : text))
  }

  private clearTranscript(): void {
    for (const child of [...this.transcript.getChildren()]) {
      this.transcript.remove(child)
      child.destroyRecursively()
    }
  }

  private redrawTranscript(): void {
    this.clearTranscript()
    if (this.session.entries.length === 0) this.welcome()
    for (const entry of this.session.entries) this.show(entry)
  }

  // -- status -----------------------------------------------------------------

  private refreshHeader(): void {
    const vendor = this.model ? this.catalog?.providers[this.model.vendor]?.label ?? this.model.vendor : ""
    this.headerModel.content = this.model ? `${modelLabel(this.model)}  ${vendor}` : "loading models…"
    this.promptBox.title = this.model ? ` ${this.model.short} ` : ""
    this.footerRight.content = [
      this.enableTools ? "tools on" : "tools off",
      this.showThinking ? "thinking shown" : "",
    ]
      .filter(Boolean)
      .join(" · ")
  }

  private refreshHint(): void {
    if (this.busy) {
      const frame = SPINNER[this.spinnerFrame % SPINNER.length]
      const log = this.lastLog ? `  ${this.lastLog}` : ""
      this.hint.content = `${frame} ${this.busy.label}${log}`
      this.hint.fg = theme.accentText
      return
    }
    const typed = this.input?.value ?? ""
    const matches = completions(typed)
    this.hint.fg = theme.hint
    if (matches.length > 0) {
      this.hint.content = matches.map((c) => `${c.usage}  ${c.summary}`).slice(0, 3).join("   ")
    } else {
      this.hint.content = ""
    }
  }

  private setBusy(label: string | null, requestId?: number): void {
    if (label === null) {
      this.busy = null
      if (this.spinnerTimer) clearInterval(this.spinnerTimer)
      this.spinnerTimer = null
      this.promptBox.borderColor = theme.borderFocused
      this.lastLog = ""
    } else {
      this.busy = { label, requestId }
      this.promptBox.borderColor = theme.border
      if (!this.spinnerTimer) {
        this.spinnerTimer = setInterval(() => {
          this.spinnerFrame++
          this.refreshHint()
        }, 90)
      }
    }
    this.refreshHint()
  }

  private onBridgeLog(line: string): void {
    // Keep the bridge's own prefix out of the status line.
    this.lastLog = line.replace(/^\[hyped-pro-bridge\]\s*/, "").slice(0, 100)
    if (this.busy) this.refreshHint()
  }

  // -- keys -------------------------------------------------------------------

  private onKey(key: KeyEvent): void {
    if (key.ctrl && key.name === "c") {
      this.quit()
      return
    }
    if (key.ctrl && key.name === "p") {
      this.pickerBox.visible ? this.closePicker() : this.openPicker()
      return
    }
    if (key.ctrl && key.name === "n") {
      this.newSession()
      return
    }
    if (key.name === "escape") {
      if (this.pickerBox.visible) this.closePicker()
      else if (this.busy) void this.stop()
      return
    }
    if (key.name === "pageup") this.transcript.scrollBy(-10)
    if (key.name === "pagedown") this.transcript.scrollBy(10)
  }

  private openPicker(): void {
    if (!this.catalog) return
    const models = this.catalog.models
    this.picker.options = models.map((m) => ({
      name: `${m.short}${m.badge ? `  ${m.badge}` : ""}`,
      description: `${this.catalog?.providers[m.vendor]?.label ?? m.vendor} · ${m.repo}`,
      value: m,
    }))
    const current = models.findIndex((m) => m.short === this.model?.short)
    if (current >= 0) this.picker.setSelectedIndex(current)
    this.pickerBox.visible = true
    this.picker.focus()
  }

  private closePicker(): void {
    this.pickerBox.visible = false
    this.input.focus()
  }

  // -- actions ----------------------------------------------------------------

  private async loadCatalog(): Promise<void> {
    try {
      this.catalog = await this.bridge.call<Catalog>("catalog")
    } catch (e) {
      const err = e as BridgeError
      this.error(`could not reach the Python side of hyped-pro: ${err.message}`, err.code)
      this.headerModel.content = "offline"
      return
    }
    const wanted = this.options.model ?? process.env.HYPED_PRO_MODEL ?? this.prefs.model
    const found = wanted ? findModel(this.catalog.models, wanted).model : undefined
    if (wanted && !found) this.note(`No model called "${wanted}" — using ${this.catalog.models[0]?.short}.`)
    this.model = found ?? this.catalog.models[0]
    this.refreshHeader()
  }

  private selectModel(model: ModelInfo): void {
    this.model = model
    this.prefs.model = model.short
    savePrefs(this.prefs)
    this.refreshHeader()
    this.note(`Model: ${modelLabel(model)}`)
  }

  private newSession(): void {
    if (this.busy) return
    this.session.clear()
    this.redrawTranscript()
  }

  private submit(): void {
    const raw = this.input.value
    if (this.busy) {
      // Enter while a reply is coming does nothing rather than queueing
      // a second one behind it; esc stops the first.
      return
    }
    this.input.value = ""
    this.refreshHint()
    const parsed = parseInput(raw)
    switch (parsed.kind) {
      case "empty":
        return
      case "message":
        void this.send(parsed.text)
        return
      case "ambiguous":
        this.error(`/${parsed.name} could be ${parsed.matches.map((m) => "/" + m).join(", ")}`)
        return
      case "unknown":
        this.error(
          `No command /${parsed.name}.` +
            (parsed.suggestions.length ? ` Did you mean ${parsed.suggestions.map((s) => "/" + s).join(", ")}?` : ""),
        )
        return
      case "command":
        void this.command(parsed.name, parsed.args, parsed.rest)
    }
  }

  private async command(name: string, args: string[], rest: string): Promise<void> {
    switch (name) {
      case "help": {
        const width = Math.max(...COMMANDS.map((c) => c.usage.length))
        this.note(
          COMMANDS.map((c) => `${c.usage.padEnd(width)}  ${c.summary}`).join("\n") +
            "\n\nesc stops a reply · pgup/pgdn scroll · // sends a message starting with /",
        )
        return
      }
      case "models":
        this.openPicker()
        return
      case "model": {
        if (!this.catalog) return
        if (!rest) {
          this.note(`Model: ${modelLabel(this.model)} — /models to pick another`)
          return
        }
        const { model, matches } = findModel(this.catalog.models, rest)
        if (model) this.selectModel(model)
        else if (matches.length) this.error(`"${rest}" matches ${matches.map((m) => m.short).join(", ")}`)
        else this.error(`No model called "${rest}". /models lists them.`)
        return
      }
      case "new":
        this.newSession()
        return
      case "system":
        this.session.system = rest || null
        this.note(rest ? "System prompt set for this conversation." : "System prompt cleared.")
        return
      case "thinking":
        this.showThinking = !this.showThinking
        this.prefs.showThinking = this.showThinking
        savePrefs(this.prefs)
        this.refreshHeader()
        this.redrawTranscript()
        return
      case "tools":
        this.enableTools = !this.enableTools
        this.prefs.enableTools = this.enableTools
        savePrefs(this.prefs)
        this.refreshHeader()
        this.note(this.enableTools ? "Tool calling on." : "Tool calling off.")
        return
      case "key":
        await this.keyCommand(args)
        return
      case "t1":
        await this.t1Command(rest)
        return
      case "noodle":
        await this.noodle(rest)
        return
      case "retry": {
        const text = this.session.rewindToLastUser()
        if (text === null) {
          this.note("Nothing to retry yet.")
          return
        }
        this.redrawTranscript()
        await this.reply()
        return
      }
      case "quit":
        this.quit()
    }
  }

  private async keyCommand(args: string[]): Promise<void> {
    const [vendor, value] = args
    if (!vendor) {
      this.note("Usage: /key <vendor> [key|clear] — vendors: " + Object.keys(this.catalog?.providers ?? {}).join(", "))
      return
    }
    try {
      if (!value) {
        const got = await this.bridge.call<{ set: boolean; masked: string | null }>("key_get", { vendor })
        this.note(got.set ? `${vendor}: ${got.masked}` : `${vendor}: no key set`)
      } else if (value === "clear") {
        await this.bridge.call("key_clear", { vendor })
        this.note(`${vendor}: key cleared`)
      } else {
        await this.bridge.call("key_set", { vendor, key: value })
        this.note(`${vendor}: key saved`)
      }
    } catch (e) {
      const err = e as BridgeError
      this.error(err.message, err.code)
    }
  }

  private async t1Command(url: string): Promise<void> {
    try {
      if (url) {
        await this.bridge.call("t1api_set_url", { url })
        if (this.catalog) this.catalog.t1_api_url = url
      }
      this.setBusy("checking the T1 API")
      const status = await this.bridge.call<Record<string, any>>("t1api_status")
      const lines = Object.entries(status).map(([k, v]) => `${k}: ${typeof v === "object" ? JSON.stringify(v) : v}`)
      this.note(lines.join("\n"))
    } catch (e) {
      const err = e as BridgeError
      this.error(err.message, err.code)
    } finally {
      this.setBusy(null)
    }
  }

  private async send(text: string): Promise<void> {
    if (!this.model) {
      this.error("No model yet — the catalog has not loaded.")
      return
    }
    this.show(this.session.add("user", text))
    await this.reply()
  }

  private async ensureDownloaded(model: ModelInfo): Promise<boolean> {
    if (!needsDownload(model)) return true
    const state = await this.bridge.call<{ downloaded: boolean }>("is_downloaded", { model: model.short })
    if (state.downloaded) return true
    const request = this.bridge.request("download", { model: model.short })
    this.setBusy(`downloading ${model.short}`, request.id)
    await request.done
    this.note(`${model.short} downloaded.`)
    return true
  }

  private async reply(): Promise<void> {
    const model = this.model
    if (!model) return
    try {
      await this.ensureDownloaded(model)
      const request = this.bridge.request<{ reply: string; thinking: string | null; cancelled: boolean }>("chat", {
        model: model.short,
        messages: this.session.history(),
        system: this.session.system,
        hide_thinking: !this.showThinking,
        enable_tools: this.enableTools,
      })
      this.setBusy(`${model.short} is thinking`, request.id)
      const result = await request.done
      this.show(
        this.session.add("assistant", result.reply ?? "", {
          thinking: result.thinking,
          model: model.short,
          cancelled: result.cancelled,
        }),
      )
    } catch (e) {
      const err = e as BridgeError
      this.error(err.message, err.code)
    } finally {
      this.setBusy(null)
    }
  }

  private async noodle(task: string): Promise<void> {
    if (!task) {
      try {
        const got = await this.bridge.call<Record<string, any>>("noodle_providers")
        this.note("Noodle providers:\n" + JSON.stringify(got, null, 2))
      } catch (e) {
        const err = e as BridgeError
        this.error(err.message, err.code)
      }
      return
    }
    this.show(this.session.add("user", `/noodle ${task}`))
    try {
      const started = await this.bridge.call<Record<string, any>>("noodle_start", {
        prompt: task,
        allow_execute: process.env.HYPED_NOODLE_EXEC === "1",
      })
      const session = String(started.session_id)
      this.setBusy("noodle running")
      if (this.busy) this.busy.noodle = session
      this.note(`Noodle ${session} · roster ${(started.roster ?? []).join(", ")} · root ${started.root}`)
      let running = true
      while (running) {
        const tick = await this.bridge.call<Record<string, any>>("noodle_poll", { session, timeout: 2.0 })
        const lines = (tick.events ?? []).map((ev: Record<string, any>) => {
          const { kind, detail } = describeNoodleEvent(ev)
          return `${kind.padEnd(18)} ${detail}`
        })
        if (lines.length) this.note(lines.join("\n"))
        running = Boolean(tick.running)
        if (!running) {
          if (tick.error) this.error(`Swarm failed: ${tick.error}`)
          else if (tick.report) {
            const failed = (tick.report.failed ?? []).length
            this.note(tick.report.ok ? `All tasks succeeded in ${tick.elapsed}s.` : `${failed} task(s) failed.`)
          }
        }
      }
    } catch (e) {
      const err = e as BridgeError
      this.error(err.message, err.code)
    } finally {
      this.setBusy(null)
    }
  }

  private async stop(): Promise<void> {
    const busy = this.busy
    if (!busy) return
    try {
      if (busy.noodle) await this.bridge.call("noodle_stop", { session: busy.noodle })
      else if (busy.requestId !== undefined) await this.bridge.cancel(busy.requestId)
      this.lastLog = "stopping…"
      this.refreshHint()
    } catch (e) {
      const err = e as BridgeError
      this.error(err.message, err.code)
    }
  }

  quit(): void {
    if (this.quitting) return
    this.quitting = true
    if (this.spinnerTimer) clearInterval(this.spinnerTimer)
    this.bridge.close()
    this.renderer.destroy()
    process.exit(0)
  }
}
