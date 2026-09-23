// Files, as hyped-pro browses and edits them. Pure: the bridge reads and
// writes (workspace-scoped, never under .git); this keeps the editor's
// state and formats listings.

export interface Entry {
  name: string
  dir: boolean
  size: number
}

export interface Listing {
  root: string
  path: string
  entries: Entry[]
}

export function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  const units = ["KB", "MB", "GB"]
  let value = bytes / 1024
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit++
  }
  return `${value < 10 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`
}

export function joinPath(dir: string, name: string): string {
  if (name === "..") return parentOf(dir)
  if (!dir || dir === ".") return name
  return `${dir.replace(/\/+$/, "")}/${name}`
}

export function parentOf(path: string): string {
  const parts = path.split("/").filter((p) => p && p !== ".")
  parts.pop()
  return parts.length ? parts.join("/") : "."
}

// Picker rows: directories first (already sorted by the bridge), with
// ".." to go up anywhere but the workspace root.
export function listingOptions(listing: Listing): { name: string; description: string; value: { path: string; dir: boolean } }[] {
  const rows = []
  if (listing.path !== ".") {
    rows.push({ name: "../", description: "up one level", value: { path: parentOf(listing.path), dir: true } })
  }
  for (const e of listing.entries) {
    rows.push({
      name: e.dir ? `${e.name}/` : e.name,
      description: e.dir ? "folder" : formatSize(e.size),
      value: { path: joinPath(listing.path, e.name), dir: e.dir },
    })
  }
  return rows
}

// One open file. `hash` is what the bridge said the file held when it was
// read; a save sends it back, so a file changed on disk meanwhile is not
// silently overwritten.
export class EditorState {
  constructor(
    readonly path: string,
    private original: string,
    public hash: string,
    readonly existed: boolean,
  ) {}

  current = ""

  isDirty(text: string = this.current): boolean {
    return text !== this.original
  }

  saved(text: string, hash: string): void {
    this.original = text
    this.current = text
    this.hash = hash
  }

  get title(): string {
    return this.existed ? this.path : `${this.path} (new)`
  }
}

// Line and column of a byte offset, for the editor's status line.
export function lineCount(text: string): number {
  return text.length === 0 ? 1 : text.split("\n").length
}
