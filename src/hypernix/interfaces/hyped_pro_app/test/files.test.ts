import { describe, expect, test } from "bun:test"

import { EditorState, formatSize, joinPath, lineCount, listingOptions, parentOf } from "../src/files.ts"

describe("paths", () => {
  test("join", () => {
    expect(joinPath(".", "a")).toBe("a")
    expect(joinPath("src", "a.ts")).toBe("src/a.ts")
    expect(joinPath("src/", "a.ts")).toBe("src/a.ts")
    expect(joinPath("src/x", "..")).toBe("src")
  })
  test("parent", () => {
    expect(parentOf("src/x")).toBe("src")
    expect(parentOf("src")).toBe(".")
    expect(parentOf(".")).toBe(".")
  })
})

describe("sizes", () => {
  test("units", () => {
    expect(formatSize(12)).toBe("12 B")
    expect(formatSize(1536)).toBe("1.5 KB")
    expect(formatSize(20 * 1024 * 1024)).toBe("20 MB")
  })
})

describe("listing", () => {
  test("up only below the root, folders marked", () => {
    const root = listingOptions({ root: "/r", path: ".", entries: [{ name: "src", dir: true, size: 0 }, { name: "a.py", dir: false, size: 3 }] })
    expect(root.map((r) => r.name)).toEqual(["src/", "a.py"])
    expect(root[0].value).toEqual({ path: "src", dir: true })
    expect(root[1].value).toEqual({ path: "a.py", dir: false })
    const inner = listingOptions({ root: "/r", path: "src", entries: [] })
    expect(inner[0]).toMatchObject({ name: "../", value: { path: ".", dir: true } })
  })
})

describe("editor state", () => {
  test("dirty until saved, then clean with the new hash", () => {
    const state = new EditorState("a.py", "one\n", "h1", true)
    expect(state.isDirty("one\n")).toBe(false)
    expect(state.isDirty("two\n")).toBe(true)
    state.saved("two\n", "h2")
    expect(state.isDirty("two\n")).toBe(false)
    expect(state.hash).toBe("h2")
  })
  test("a new file says so", () => {
    expect(new EditorState("n.md", "", "h", false).title).toBe("n.md (new)")
  })
  test("lines", () => {
    expect(lineCount("")).toBe(1)
    expect(lineCount("a\nb")).toBe(2)
  })
})
