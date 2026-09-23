// One conversation: what is on screen and what is sent to the model.

export type Role = "user" | "assistant" | "note" | "error"

export interface Entry {
  id: number
  role: Role
  text: string
  thinking?: string | null
  model?: string
  cancelled?: boolean
}

export interface ModelInfo {
  short: string
  repo: string
  vendor: string
  kind: string
  badge?: string
  context_window?: number | null
  notes?: string
  format?: string
}

export interface ProviderInfo {
  vendor: string
  kind: string
  label: string
  auth_env_var?: string | null
}

export interface Catalog {
  t1_api_url: string
  providers: Record<string, ProviderInfo>
  models: ModelInfo[]
}

export class Session {
  entries: Entry[] = []
  system: string | null = null
  private nextId = 1

  add(role: Role, text: string, extra: Partial<Entry> = {}): Entry {
    const entry: Entry = { id: this.nextId++, role, text, ...extra }
    this.entries.push(entry)
    return entry
  }

  clear(): void {
    this.entries = []
  }

  // What the model sees: the user and assistant turns, in order. Notes
  // and errors are for the person, not the model, and a reply that was
  // cut off by a cancel is still something the model said.
  history(): { role: "user" | "assistant"; content: string }[] {
    return this.entries
      .filter((e) => e.role === "user" || e.role === "assistant")
      .filter((e) => e.text.length > 0)
      .map((e) => ({ role: e.role as "user" | "assistant", content: e.text }))
  }

  // For /retry: drop everything after the last user turn and hand back
  // that turn's text, or null when there is nothing to retry.
  rewindToLastUser(): string | null {
    for (let i = this.entries.length - 1; i >= 0; i--) {
      if (this.entries[i].role === "user") {
        const text = this.entries[i].text
        this.entries = this.entries.slice(0, i + 1)
        return text
      }
    }
    return null
  }

  get turns(): number {
    return this.entries.filter((e) => e.role === "user").length
  }
}

// Find a model by what someone typed: exact short name, then exact repo,
// then a unique prefix or substring of either.
export function findModel(models: ModelInfo[], query: string): { model?: ModelInfo; matches: ModelInfo[] } {
  const q = query.trim().toLowerCase()
  if (!q) return { matches: [] }
  const exact = models.find((m) => m.short.toLowerCase() === q || m.repo.toLowerCase() === q)
  if (exact) return { model: exact, matches: [exact] }
  const prefix = models.filter((m) => m.short.toLowerCase().startsWith(q))
  if (prefix.length === 1) return { model: prefix[0], matches: prefix }
  const contains = models.filter((m) => m.short.toLowerCase().includes(q) || m.repo.toLowerCase().includes(q))
  if (contains.length === 1) return { model: contains[0], matches: contains }
  return { matches: prefix.length ? prefix : contains }
}

// Local models have to be on disk before the first message; cloud and
// T1 ones do not.
export function needsDownload(model: ModelInfo): boolean {
  return model.kind === "local" && model.vendor !== "t1"
}

export function modelLabel(model: ModelInfo | undefined): string {
  if (!model) return "no model"
  return model.badge ? `${model.short} ${model.badge}` : model.short
}

// One Noodle event as a single line, the way hyped-plus prints them.
export function describeNoodleEvent(event: Record<string, any>): { kind: string; detail: string } {
  const kind = String(event.kind ?? event.type ?? "event")
  const detail = String(event.message ?? event.text ?? event.tool ?? event.task_id ?? event.repr ?? "")
  return { kind, detail: detail.length > 120 ? detail.slice(0, 119) + "…" : detail }
}
