// Slash commands. Pure: parsing and lookup only, so it is testable
// without a terminal. The app decides what each one does.

export interface Command {
  name: string
  usage: string
  summary: string
}

export const COMMANDS: readonly Command[] = [
  { name: "help", usage: "/help", summary: "list commands and keys" },
  { name: "models", usage: "/models", summary: "pick a model (also ctrl+p)" },
  { name: "model", usage: "/model <name>", summary: "switch model by name" },
  { name: "new", usage: "/new", summary: "start a new conversation (also ctrl+n)" },
  { name: "system", usage: "/system [prompt]", summary: "set or clear the system prompt" },
  { name: "thinking", usage: "/thinking", summary: "show or hide model thinking" },
  { name: "tools", usage: "/tools", summary: "turn tool calling on or off" },
  { name: "key", usage: "/key <vendor> [key|clear]", summary: "show, set or clear a provider key" },
  { name: "t1", usage: "/t1 [url]", summary: "T1 API status, or point at another server" },
  { name: "noodle", usage: "/noodle <task>", summary: "run a Noodle swarm on a task" },
  { name: "retry", usage: "/retry", summary: "ask again for the last reply" },
  { name: "quit", usage: "/quit", summary: "leave hyped-pro (also ctrl+c)" },
]

const ALIASES: Record<string, string> = {
  exit: "quit",
  q: "quit",
  clear: "new",
  m: "model",
  "?": "help",
}

export type Parsed =
  | { kind: "empty" }
  | { kind: "message"; text: string }
  | { kind: "command"; name: string; args: string[]; rest: string }
  | { kind: "unknown"; name: string; suggestions: string[] }
  | { kind: "ambiguous"; name: string; matches: string[] }

export function resolveCommand(word: string): { name?: string; matches: string[] } {
  const lower = word.toLowerCase()
  if (ALIASES[lower]) return { name: ALIASES[lower], matches: [ALIASES[lower]] }
  const exact = COMMANDS.find((c) => c.name === lower)
  if (exact) return { name: exact.name, matches: [exact.name] }
  const matches = COMMANDS.filter((c) => c.name.startsWith(lower)).map((c) => c.name)
  return matches.length === 1 ? { name: matches[0], matches } : { matches }
}

export function parseInput(raw: string): Parsed {
  const text = raw.trim()
  if (!text) return { kind: "empty" }
  // "//" sends a message that starts with a slash.
  if (text.startsWith("//")) return { kind: "message", text: text.slice(1) }
  if (!text.startsWith("/")) return { kind: "message", text }

  const body = text.slice(1)
  const space = body.search(/\s/)
  const word = space < 0 ? body : body.slice(0, space)
  const rest = space < 0 ? "" : body.slice(space).trim()
  const args = rest ? rest.split(/\s+/) : []
  if (!word) return { kind: "unknown", name: "", suggestions: COMMANDS.map((c) => c.name) }

  const { name, matches } = resolveCommand(word)
  if (name) return { kind: "command", name, args, rest }
  if (matches.length > 1) return { kind: "ambiguous", name: word, matches }
  return { kind: "unknown", name: word, suggestions: suggest(word) }
}

// Closest command names by edit distance, for "did you mean".
export function suggest(word: string, limit = 3): string[] {
  const scored = COMMANDS.map((c) => ({ name: c.name, distance: editDistance(word.toLowerCase(), c.name) }))
  return scored
    .filter((s) => s.distance <= Math.max(2, Math.floor(s.name.length / 2)))
    .sort((a, b) => a.distance - b.distance || a.name.localeCompare(b.name))
    .slice(0, limit)
    .map((s) => s.name)
}

export function editDistance(a: string, b: string): number {
  const row = Array.from({ length: b.length + 1 }, (_, i) => i)
  for (let i = 1; i <= a.length; i++) {
    let previous = row[0]
    row[0] = i
    for (let j = 1; j <= b.length; j++) {
      const saved = row[j]
      row[j] = Math.min(row[j] + 1, row[j - 1] + 1, previous + (a[i - 1] === b[j - 1] ? 0 : 1))
      previous = saved
    }
  }
  return row[b.length]
}

// Commands that start with what has been typed so far, for the hint line
// under the prompt.
export function completions(partial: string): Command[] {
  if (!partial.startsWith("/") || /\s/.test(partial)) return []
  const word = partial.slice(1).toLowerCase()
  return COMMANDS.filter((c) => c.name.startsWith(word))
}
