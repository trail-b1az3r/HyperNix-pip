// The hyped-pro screen, laid out the way opencode's is: a header, the
// conversation filling the middle, a bordered prompt at the bottom and a
// line of key hints under it. Colours come from theme.ts, which is the
// site's palette.

import {
  BoxRenderable,
  DiffRenderable,
  InputRenderable,
  InputRenderableEvents,
  MarkdownRenderable,
  ScrollBoxRenderable,
  SelectRenderable,
  SelectRenderableEvents,
  SyntaxStyle,
  TextAttributes,
  TextareaRenderable,
  TextRenderable,
  type CliRenderer,
  type KeyEvent,
  type Renderable,
} from "@opentui/core"

import { Bridge, BridgeError } from "./bridge.ts"
import { COMMANDS, completions, parseInput } from "./commands.ts"
import { EditorState, listingOptions, lineCount, type Listing } from "./files.ts"
import { branchBadge, diffStats, GIT_USAGE, logLines, parseGit, splitDiff, statusLines, type RepoStatus } from "./git.ts"
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

export const VERSION = "0.72.6"

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
  private git: RepoStatus | null = null
  private pickerMode: "models" | "files" = "models"
  private editor: { state: EditorState; box: BoxRenderable; area: TextareaRenderable; status: TextRenderable; armed: boolean } | null = null
  private questions: { text: string; detail: string; answer: (allow: boolean) => void }[] = []
  private consentBox!: BoxRenderable
  private consentText!: TextRenderable
  private headerGit!: TextRenderable
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
    this.bridge.onEvent = (event) => this.onBridgeEvent(event)
  }

  async start(): Promise<void> {
    this.renderer.setBackgroundColor(theme.background)
    this.build()
    this.renderer.keyInput.on("keypress", (key: KeyEvent) => this.onKey(key))
    this.welcome()
    this.input.focus()
    await this.loadCatalog()
    try {
      // Tell the bridge this client can answer consent questions and show
      // tool activity. An older bridge answers "unknown command"; then
      // gated tools are refused there, which is the safe way round.
      await this.bridge.call("hello", { features: ["consent", "events"] })
    } catch {
      // older bridge
    }
    await this.refreshGit()
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
    this.headerGit = new TextRenderable(r, { id: "header-git", content: "", fg: theme.accentText })
    header.add(this.headerGit)
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
      if (this.pickerMode === "files") {
        const value = option?.value as { path: string; dir: boolean } | undefined
        if (!value) return
        if (value.dir) void this.browse(value.path)
        else {
          this.closePicker()
          void this.openEditor(value.path)
        }
        return
      }
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
    this.consentBox = new BoxRenderable(r, {
      id: "consent",
      border: true,
      borderStyle: "rounded",
      borderColor: theme.accent,
      title: " allow? ",
      titleColor: theme.accentText,
      backgroundColor: theme.panel,
      flexShrink: 0,
      marginTop: 1,
      paddingLeft: 1,
      paddingRight: 1,
      visible: false,
    })
    this.consentText = new TextRenderable(r, { id: "consent-text", content: "", fg: theme.text, wrapMode: "word" })
    this.consentBox.add(this.consentText)

    root.add(this.transcript)
    root.add(this.pickerBox)
    root.add(this.consentBox)
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
    if (this.questions.length) {
      // Only y answers yes. Anything else that could be an answer is a no.
      if (key.name === "y") this.answer(true)
      else if (key.name === "n" || key.name === "escape" || key.name === "return") this.answer(false)
      return
    }
    if (this.editor) {
      if (key.ctrl && key.name === "s") {
        void this.saveEditor()
        return
      }
      if (key.name === "escape") {
        this.closeEditor()
        return
      }
      if (this.editor.armed && key.name !== "escape") {
        this.editor.armed = false
        this.refreshEditorStatus()
      }
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
    this.pickerMode = "models"
    this.pickerBox.title = " models "
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
      case "git":
        await this.gitCommand(rest)
        return
      case "diff":
        await this.gitCommand(`diff ${rest}`)
        return
      case "files":
        await this.browse(rest || ".")
        return
      case "edit":
        if (!rest) {
          this.error("Usage: /edit <path> — /files to browse")
          return
        }
        await this.openEditor(rest)
        return
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
      void this.refreshGit()
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

  // -- events from the bridge ---------------------------------------------------

  private onBridgeEvent(event: Record<string, any>): void {
    if (event.event === "tool") {
      const mark = event.ok ? "✎" : "✗"
      const detail = String(event.detail ?? "").split("\n")[0].slice(0, 100)
      this.note(`${mark} ${event.tool}  ${detail}${event.ok ? "" : `\n  ${event.error ?? ""}`}`)
      return
    }
    if (event.event === "consent") {
      const consentId = event.consent_id
      this.ask(`The model wants to run ${event.tool}`, String(event.detail ?? ""), (allow) => {
        void this.bridge.call("consent_reply", { consent_id: consentId, allow }).catch(() => {})
        this.note(`${allow ? "Allowed" : "Refused"}: ${event.tool}`)
      })
    }
  }

  // A yes/no question in the box above the prompt. Questions queue, so a
  // model asking twice in a row gets two answers, in order.
  private ask(text: string, detail: string, answer: (allow: boolean) => void): void {
    this.questions.push({ text, detail, answer })
    if (this.questions.length === 1) this.showQuestion()
  }

  private confirm(text: string, detail = ""): Promise<boolean> {
    return new Promise((resolve) => this.ask(text, detail, resolve))
  }

  private showQuestion(): void {
    const next = this.questions[0]
    if (!next) {
      this.consentBox.visible = false
      if (this.editor) this.editor.area.focus()
      else this.input.focus()
      return
    }
    const detail = next.detail.length > 1200 ? next.detail.slice(0, 1200) + "…" : next.detail
    this.consentText.content = `${next.text}\n\n${detail}\n\ny allow · n refuse`
    this.consentBox.visible = true
    // Nothing else takes keys while a question is open: a "y" typed into
    // the prompt must not be read as consent, or the other way round.
    this.input.blur()
    this.editor?.area.blur()
  }

  private answer(allow: boolean): void {
    const current = this.questions.shift()
    current?.answer(allow)
    this.showQuestion()
  }

  // -- git --------------------------------------------------------------------

  private async refreshGit(): Promise<void> {
    try {
      const got = await this.bridge.call<{ repository: boolean } & RepoStatus>("git", { op: "status" })
      this.git = got.repository ? got : null
    } catch {
      this.git = null
    }
    this.headerGit.content = this.git ? `${branchBadge(this.git)}   ` : ""
  }

  private async gitCommand(rest: string): Promise<void> {
    const command = parseGit(rest)
    try {
      switch (command.kind) {
        case "help":
          this.note(GIT_USAGE.join("\n"))
          return
        case "error":
          this.error(command.message)
          return
        case "status": {
          const got = await this.bridge.call<{ repository: boolean } & RepoStatus>("git", { op: "status" })
          if (!got.repository) this.note("Not inside a git repository.")
          else this.note(statusLines(got).join("\n"))
          break
        }
        case "diff": {
          const got = await this.bridge.call<{ diff: string }>("git", { op: "diff", path: command.path, staged: command.staged })
          this.showDiff(got.diff, command.staged)
          break
        }
        case "log": {
          const got = await this.bridge.call<{ commits: any[] }>("git", { op: "log", count: command.count, path: command.path })
          this.note(logLines(got.commits).join("\n"))
          break
        }
        case "branches": {
          const got = await this.bridge.call<{ branches: { name: string; current: boolean; upstream: string }[] }>("git", { op: "branches" })
          this.note(got.branches.map((b) => `${b.current ? "*" : " "} ${b.name}${b.upstream ? `  → ${b.upstream}` : ""}`).join("\n") || "No branches yet.")
          break
        }
        case "restore": {
          const sure = await this.confirm("Discard unstaged changes? This cannot be undone.", command.paths.join("\n"))
          if (!sure) {
            this.note("Kept the changes.")
            return
          }
          const got = await this.bridge.call<{ message: string }>("git", { op: "restore", paths: command.paths })
          this.note(got.message)
          break
        }
        case "push":
        case "pull": {
          const request = this.bridge.request<{ message: string }>("git", {
            op: command.kind,
            set_upstream: command.kind === "push" ? command.setUpstream : undefined,
            rebase: command.kind === "pull" ? command.rebase : undefined,
          })
          this.setBusy(`git ${command.kind}`, request.id)
          try {
            this.note((await request.done).message)
          } finally {
            this.setBusy(null)
          }
          break
        }
        default: {
          const fields: Record<string, unknown> = { op: command.kind }
          if (command.kind === "add" || command.kind === "unstage") fields.paths = command.paths
          if (command.kind === "commit") Object.assign(fields, { message: command.message, all: command.all })
          if (command.kind === "switch") Object.assign(fields, { branch: command.branch, create: command.create })
          const got = await this.bridge.call<{ message: string }>("git", fields)
          this.note(got.message)
        }
      }
    } catch (e) {
      const err = e as BridgeError
      this.error(err.message, err.code)
    }
    await this.refreshGit()
  }

  private showDiff(diff: string, staged: boolean): void {
    if (!diff.trim()) {
      this.note(staged ? "Nothing staged." : "No unstaged changes.")
      return
    }
    const stats = diffStats(diff)
    this.note(`${stats.files} file(s), +${stats.added} −${stats.removed}${staged ? " (staged)" : ""}`)
    const r = this.renderer
    for (const [i, file] of splitDiff(diff).entries()) {
      const id = `diff-${Date.now()}-${i}`
      const box = new BoxRenderable(r, {
        id,
        border: true,
        borderStyle: "rounded",
        borderColor: theme.border,
        title: ` ${file.path} `,
        titleColor: theme.textDim,
        flexDirection: "column",
      })
      try {
        box.add(new DiffRenderable(r, {
          id: `${id}-body`,
          diff: file.diff,
          view: "unified",
          showLineNumbers: true,
          fg: theme.text,
          lineNumberFg: theme.hint,
          addedBg: "#12261a",
          removedBg: "#2a1215",
          addedSignColor: theme.ok,
          removedSignColor: theme.accentText,
          wrapMode: "none",
        }))
      } catch {
        box.add(new TextRenderable(r, { id: `${id}-text`, content: file.diff, fg: theme.textDim }))
      }
      this.transcript.add(box)
    }
  }

  // -- files ------------------------------------------------------------------

  private async browse(path: string): Promise<void> {
    try {
      const listing = await this.bridge.call<Listing>("files_list", { path })
      this.pickerMode = "files"
      this.pickerBox.title = ` ${listing.path === "." ? "workspace" : listing.path} `
      this.picker.options = listingOptions(listing)
      this.picker.setSelectedIndex(0)
      this.pickerBox.visible = true
      this.picker.focus()
    } catch (e) {
      const err = e as BridgeError
      this.error(err.message, err.code)
    }
  }

  private async openEditor(path: string): Promise<void> {
    if (this.editor) this.closeEditor(true)
    let file: { path: string; content: string; hash: string; exists: boolean }
    try {
      file = await this.bridge.call("file_read", { path })
    } catch (e) {
      const err = e as BridgeError
      this.error(err.message, err.code)
      return
    }
    const r = this.renderer
    const state = new EditorState(file.path, file.content, file.hash, file.exists)
    state.current = file.content
    const box = new BoxRenderable(r, {
      id: `editor-${Date.now()}`,
      border: true,
      borderStyle: "rounded",
      borderColor: theme.accent,
      title: ` ${state.title} `,
      titleColor: theme.accentText,
      backgroundColor: theme.panel,
      flexGrow: 1,
      flexDirection: "column",
      marginTop: 1,
      paddingLeft: 1,
      paddingRight: 1,
    })
    const area = new TextareaRenderable(r, {
      id: `${box.id}-area`,
      flexGrow: 1,
      initialValue: file.content,
      backgroundColor: theme.panel,
      focusedBackgroundColor: theme.panel,
      textColor: theme.text,
      focusedTextColor: theme.text,
      wrapMode: "none",
      onContentChange: () => {
        if (this.editor) {
          this.editor.state.current = this.editor.area.plainText
          this.refreshEditorStatus()
        }
      },
    })
    const status = new TextRenderable(r, { id: `${box.id}-status`, content: "", fg: theme.hint, height: 1 })
    box.add(area)
    box.add(status)
    this.transcript.visible = false
    this.pickerBox.visible = false
    const root = this.transcript.parent
    root?.add(box, root.getChildren().indexOf(this.transcript) + 1)
    this.editor = { state, box, area, status, armed: false }
    this.refreshEditorStatus()
    area.focus()
  }

  private refreshEditorStatus(): void {
    const editor = this.editor
    if (!editor) return
    const text = editor.state.current
    const dirty = editor.state.isDirty(text)
    editor.box.title = ` ${editor.state.title}${dirty ? " ●" : ""} `
    editor.status.content = editor.armed
      ? "unsaved changes — esc again to discard, ctrl+s to save"
      : `${lineCount(text)} lines · ctrl+s save · esc close`
    editor.status.fg = editor.armed ? theme.accentText : theme.hint
  }

  private async saveEditor(): Promise<void> {
    const editor = this.editor
    if (!editor) return
    const text = editor.area.plainText
    try {
      const got = await this.bridge.call<{ message: string; hash: string }>("file_write", {
        path: editor.state.path,
        content: text,
        expected_hash: editor.state.hash,
      })
      editor.state.saved(text, got.hash)
      editor.armed = false
      this.refreshEditorStatus()
      editor.status.content = `saved · ${got.message}`
      void this.refreshGit()
    } catch (e) {
      const err = e as BridgeError
      editor.status.content = `not saved: ${err.message}`
      editor.status.fg = theme.error
    }
  }

  private closeEditor(force = false): void {
    const editor = this.editor
    if (!editor) return
    editor.state.current = editor.area.plainText
    if (!force && editor.state.isDirty() && !editor.armed) {
      // First esc on unsaved work only warns.
      editor.armed = true
      this.refreshEditorStatus()
      return
    }
    editor.box.parent?.remove(editor.box)
    editor.box.destroyRecursively()
    this.editor = null
    this.transcript.visible = true
    this.input.focus()
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
