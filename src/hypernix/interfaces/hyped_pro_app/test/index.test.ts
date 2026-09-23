import { describe, expect, test } from "bun:test"

import { parseArgs } from "../src/index.ts"
import { palette, theme } from "../src/theme.ts"

describe("parseArgs", () => {
  test("model forms", () => {
    expect(parseArgs(["--model", "kimi-k3"]).model).toBe("kimi-k3")
    expect(parseArgs(["-m", "kimi-k3"]).model).toBe("kimi-k3")
    expect(parseArgs(["--model=kimi-k3"]).model).toBe("kimi-k3")
  })

  test("help and version", () => {
    expect(parseArgs(["-h"]).help).toBe(true)
    expect(parseArgs(["--version"]).version).toBe(true)
    expect(parseArgs([])).toEqual({ help: false, version: false })
  })
})

describe("theme", () => {
  test("every theme colour is a palette colour", () => {
    const colours = new Set(Object.values(palette))
    for (const [part, colour] of Object.entries(theme)) expect(colours.has(colour as any), part).toBe(true)
  })

  test("the page is the site background and the accent is the site red", () => {
    expect(theme.background).toBe("#0d0d0d")
    expect(theme.accent).toBe("#c8192e")
  })
})
