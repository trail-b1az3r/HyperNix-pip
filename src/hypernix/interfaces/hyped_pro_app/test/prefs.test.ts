import { afterEach, describe, expect, test } from "bun:test"
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"

import { loadPrefs, prefsPath, savePrefs } from "../src/prefs.ts"

const made: string[] = []
function dir(): string {
  const d = mkdtempSync(join(tmpdir(), "hyped-pro-prefs-"))
  made.push(d)
  return d
}
afterEach(() => {
  for (const d of made.splice(0)) rmSync(d, { recursive: true, force: true })
})

describe("prefs", () => {
  test("lives under HYPERNIX_HOME when set", () => {
    expect(prefsPath({ HYPERNIX_HOME: "/x/y" })).toBe("/x/y/hyped-pro.json")
  })

  test("round trip, creating the directory", () => {
    const path = join(dir(), "nested", "hyped-pro.json")
    savePrefs({ model: "kimi-k3", showThinking: true }, path)
    expect(loadPrefs(path)).toEqual({ model: "kimi-k3", showThinking: true })
    expect(readFileSync(path, "utf8").endsWith("\n")).toBe(true)
  })

  test("missing, damaged or wrong-shaped files load as empty", () => {
    const d = dir()
    expect(loadPrefs(join(d, "none.json"))).toEqual({})
    writeFileSync(join(d, "bad.json"), "{not json")
    expect(loadPrefs(join(d, "bad.json"))).toEqual({})
    writeFileSync(join(d, "list.json"), "[1,2]")
    expect(loadPrefs(join(d, "list.json"))).toEqual({})
  })

  test("a save that cannot be written does not throw", () => {
    const d = dir()
    writeFileSync(join(d, "file"), "")
    expect(() => savePrefs({ model: "x" }, join(d, "file", "under-a-file.json"))).not.toThrow()
  })
})
