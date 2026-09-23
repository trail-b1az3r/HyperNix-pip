// What hyped-pro remembers between runs: the last model and two toggles.
// Kept in its own file under ~/.hypernix so it never collides with the
// Python side's config.

import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs"
import { homedir } from "node:os"
import { dirname, join } from "node:path"

export interface Prefs {
  model?: string
  showThinking?: boolean
  enableTools?: boolean
}

export function prefsPath(env: Record<string, string | undefined> = process.env): string {
  const base = env.HYPERNIX_HOME || join(homedir(), ".hypernix")
  return join(base, "hyped-pro.json")
}

export function loadPrefs(path: string = prefsPath()): Prefs {
  try {
    if (!existsSync(path)) return {}
    const parsed = JSON.parse(readFileSync(path, "utf8"))
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? (parsed as Prefs) : {}
  } catch {
    // A damaged file is not worth refusing to start over.
    return {}
  }
}

export function savePrefs(prefs: Prefs, path: string = prefsPath()): void {
  try {
    mkdirSync(dirname(path), { recursive: true })
    const temporary = `${path}.tmp`
    writeFileSync(temporary, JSON.stringify(prefs, null, 2) + "\n")
    renameSync(temporary, path)
  } catch {
    // Not being able to remember a preference should never interrupt a chat.
  }
}
