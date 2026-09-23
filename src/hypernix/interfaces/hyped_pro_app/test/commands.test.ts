import { describe, expect, test } from "bun:test"

import { COMMANDS, completions, editDistance, parseInput, resolveCommand, suggest } from "../src/commands.ts"

describe("parseInput", () => {
  test("blank input is empty", () => {
    expect(parseInput("   ")).toEqual({ kind: "empty" })
  })

  test("plain text is a message, trimmed", () => {
    expect(parseInput("  hello  ")).toEqual({ kind: "message", text: "hello" })
  })

  test("a double slash sends a message that starts with a slash", () => {
    expect(parseInput("//etc/hosts is odd")).toEqual({ kind: "message", text: "/etc/hosts is odd" })
  })

  test("a command keeps its arguments and the raw rest", () => {
    expect(parseInput("/system  be   brief")).toEqual({
      kind: "command",
      name: "system",
      args: ["be", "brief"],
      rest: "be   brief",
    })
  })

  test("a unique prefix resolves", () => {
    expect(parseInput("/noo fix it")).toMatchObject({ kind: "command", name: "noodle", rest: "fix it" })
  })

  test("an exact name wins over a longer one it prefixes", () => {
    // "model" is a prefix of "models"; typing it means /model.
    expect(parseInput("/model kimi")).toMatchObject({ kind: "command", name: "model" })
  })

  test("an ambiguous prefix lists the candidates", () => {
    expect(parseInput("/mo")).toEqual({ kind: "ambiguous", name: "mo", matches: ["models", "model"] })
  })

  test("aliases", () => {
    expect(parseInput("/exit")).toMatchObject({ kind: "command", name: "quit" })
    expect(parseInput("/clear")).toMatchObject({ kind: "command", name: "new" })
    expect(parseInput("/?")).toMatchObject({ kind: "command", name: "help" })
  })

  test("an unknown command suggests close ones", () => {
    const parsed = parseInput("/hlep")
    expect(parsed.kind).toBe("unknown")
    if (parsed.kind === "unknown") expect(parsed.suggestions).toContain("help")
  })

  test("a lone slash is unknown and suggests everything", () => {
    const parsed = parseInput("/")
    expect(parsed.kind).toBe("unknown")
    if (parsed.kind === "unknown") expect(parsed.suggestions.length).toBe(COMMANDS.length)
  })

  test("command names are case-insensitive", () => {
    expect(parseInput("/HELP")).toMatchObject({ kind: "command", name: "help" })
  })
})

describe("helpers", () => {
  test("every command name is unique", () => {
    const names = COMMANDS.map((c) => c.name)
    expect(new Set(names).size).toBe(names.length)
  })

  test("every usage starts with its own name", () => {
    for (const c of COMMANDS) expect(c.usage.startsWith(`/${c.name}`)).toBe(true)
  })

  test("resolveCommand on nothing matching", () => {
    expect(resolveCommand("zzz")).toEqual({ matches: [] })
  })

  test("editDistance", () => {
    expect(editDistance("", "abc")).toBe(3)
    expect(editDistance("kitten", "sitting")).toBe(3)
    expect(editDistance("same", "same")).toBe(0)
  })

  test("suggest does not offer far-off names", () => {
    expect(suggest("xyzzyplugh")).toEqual([])
  })

  test("completions only while the command word is being typed", () => {
    expect(completions("/t").map((c) => c.name)).toEqual(["thinking", "tools", "t1"])
    expect(completions("/tools on")).toEqual([])
    expect(completions("hello")).toEqual([])
  })
})
