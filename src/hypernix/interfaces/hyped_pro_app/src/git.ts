// Git, as hyped-pro shows it. Pure: formatting and parsing only; the
// bridge runs git (hypernix.interfaces.hyped_pro_git) and this turns its
// answers into lines for the screen.

export interface FileStatus {
  path: string
  index: string
  worktree: string
  original: string | null
  staged: boolean
  unstaged: boolean
  label: string
}

export interface RepoStatus {
  root: string
  branch: string | null
  upstream: string | null
  ahead: number
  behind: number
  detached: boolean
  clean: boolean
  files: FileStatus[]
}

export interface Commit {
  hash: string
  short: string
  author: string
  when: string
  subject: string
}

// The header's short form: "main ↑2 ↓1 ●3". Nothing at all outside a
// repository, so the header does not claim one.
export function branchBadge(status: RepoStatus | null): string {
  if (!status) return ""
  const name = status.detached ? "detached" : status.branch ?? "?"
  const parts = [name]
  if (status.ahead) parts.push(`↑${status.ahead}`)
  if (status.behind) parts.push(`↓${status.behind}`)
  if (!status.clean) parts.push(`●${status.files.length}`)
  return parts.join(" ")
}

export function statusLines(status: RepoStatus): string[] {
  const lines: string[] = []
  const where = status.detached ? "a detached HEAD" : `branch ${status.branch}`
  lines.push(`On ${where}` + (status.upstream ? `, tracking ${status.upstream}` : ""))
  if (status.upstream && (status.ahead || status.behind)) {
    lines.push(`${status.ahead} ahead, ${status.behind} behind`)
  }
  if (status.clean) {
    lines.push("Nothing to commit, working tree clean.")
    return lines
  }
  const staged = status.files.filter((f) => f.staged)
  const unstaged = status.files.filter((f) => f.unstaged && f.label !== "untracked")
  const untracked = status.files.filter((f) => f.label === "untracked")
  const section = (title: string, files: FileStatus[], label: (f: FileStatus) => string) => {
    if (!files.length) return
    lines.push("", title)
    for (const f of files) lines.push(`  ${label(f).padEnd(12)} ${f.path}${f.original ? `  (from ${f.original})` : ""}`)
  }
  section("Staged", staged, (f) => labelFor(f.index))
  section("Not staged", unstaged, (f) => labelFor(f.worktree))
  section("Untracked", untracked, () => "new")
  return lines
}

export function labelFor(code: string): string {
  switch (code) {
    case "M": return "modified"
    case "A": return "added"
    case "D": return "deleted"
    case "R": return "renamed"
    case "C": return "copied"
    case "T": return "type change"
    case "U": return "conflict"
    default: return "changed"
  }
}

export function logLines(commits: Commit[]): string[] {
  if (!commits.length) return ["No commits yet."]
  const width = Math.max(...commits.map((c) => c.when.length))
  return commits.map((c) => `${c.short}  ${c.when.padEnd(width)}  ${c.subject}  — ${c.author}`)
}

// Counts for a diff's summary line, from the unified diff itself.
export function diffStats(diff: string): { files: number; added: number; removed: number } {
  let files = 0
  let added = 0
  let removed = 0
  for (const line of diff.split("\n")) {
    if (line.startsWith("diff --git ")) files++
    else if (line.startsWith("+") && !line.startsWith("+++")) added++
    else if (line.startsWith("-") && !line.startsWith("---")) removed++
  }
  return { files, added, removed }
}

// Split a multi-file diff into one diff per file: DiffRenderable draws a
// single file's hunks.
export function splitDiff(diff: string): { path: string; diff: string }[] {
  const out: { path: string; diff: string }[] = []
  let current: string[] = []
  let path = ""
  const flush = () => {
    if (current.length) out.push({ path, diff: current.join("\n") + "\n" })
  }
  for (const line of diff.split("\n")) {
    if (line.startsWith("diff --git ")) {
      flush()
      current = [line]
      const match = /^diff --git a\/(.+?) b\/(.+)$/.exec(line)
      path = match ? match[2] : line.slice("diff --git ".length)
    } else if (current.length) {
      current.push(line)
    }
  }
  flush()
  return out.map((f) => ({ ...f, diff: f.diff.replace(/\n+$/, "\n") }))
}

// What the TUI's /git accepts. The person typing it is the person who
// would type it in a shell, so none of it asks for consent; the model's
// own git tools do.
export type GitCommand =
  | { kind: "status" }
  | { kind: "diff"; path?: string; staged: boolean }
  | { kind: "log"; count: number; path?: string }
  | { kind: "branches" }
  | { kind: "add"; paths: string[] }
  | { kind: "unstage"; paths: string[] }
  | { kind: "commit"; message: string; all: boolean }
  | { kind: "switch"; branch: string; create: boolean }
  | { kind: "restore"; paths: string[] }
  | { kind: "push"; setUpstream: boolean }
  | { kind: "pull"; rebase: boolean }
  | { kind: "help" }
  | { kind: "error"; message: string }

export const GIT_USAGE = [
  "/git                     status",
  "/git diff [--staged] [path]",
  "/git log [n] [path]",
  "/git branches",
  "/git add <path…>         stage (/git add . for everything)",
  "/git unstage <path…>",
  "/git commit [-a] <message>",
  "/git switch [-c] <branch>",
  "/git restore <path…>     discard unstaged changes (asks first)",
  "/git push [-u]",
  "/git pull [--rebase]",
]

export function parseGit(rest: string): GitCommand {
  const text = rest.trim()
  if (!text) return { kind: "status" }
  const [verb, ...args] = text.split(/\s+/)
  const flags = new Set(args.filter((a) => a.startsWith("-")))
  const plain = args.filter((a) => !a.startsWith("-"))
  switch (verb) {
    case "help":
      return { kind: "help" }
    case "status":
    case "st":
      return { kind: "status" }
    case "diff":
      return { kind: "diff", staged: flags.has("--staged") || flags.has("--cached"), path: plain[0] }
    case "log": {
      const n = plain.length && /^\d+$/.test(plain[0]) ? Number(plain.shift()) : 20
      return { kind: "log", count: Math.min(200, Math.max(1, n)), path: plain[0] }
    }
    case "branch":
    case "branches":
      return { kind: "branches" }
    case "add":
      return plain.length ? { kind: "add", paths: plain } : { kind: "error", message: "/git add <path…> — '.' for everything" }
    case "unstage":
    case "reset":
      return plain.length ? { kind: "unstage", paths: plain } : { kind: "error", message: "/git unstage <path…>" }
    case "commit": {
      // The message is everything after the leading flags, spaces and
      // all — only flags *before* it count, so "fix -a typo" keeps its -a.
      let body = text.slice(verb.length).trim()
      let all = false
      for (;;) {
        const flag = /^(-a|--all|-m)(\s+|$)/.exec(body)
        if (!flag) break
        if (flag[1] !== "-m") all = true
        body = body.slice(flag[0].length)
      }
      body = body.replace(/^(["'])([\s\S]*)\1$/, "$2").trim()
      if (!body) return { kind: "error", message: "/git commit [-a] <message>" }
      return { kind: "commit", message: body, all }
    }
    case "switch":
    case "checkout":
      return plain.length === 1
        ? { kind: "switch", branch: plain[0], create: flags.has("-c") || flags.has("-b") }
        : { kind: "error", message: "/git switch [-c] <branch>" }
    case "restore":
    case "discard":
      return plain.length ? { kind: "restore", paths: plain } : { kind: "error", message: "/git restore <path…>" }
    case "push":
      return { kind: "push", setUpstream: flags.has("-u") || flags.has("--set-upstream") }
    case "pull":
      return { kind: "pull", rebase: flags.has("--rebase") || flags.has("-r") }
    default:
      return { kind: "error", message: `No /git ${verb}. /git help lists what there is.` }
  }
}
