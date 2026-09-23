#!/usr/bin/env bun
// hyped-pro: the HyperNix terminal client, on OpenTUI.
//
// The previous hyped-pro, a readline TUI, is still installed as
// `hyped-plus`. Both drive the same Python bridge.

import { createCliRenderer } from "@opentui/core"

import { App, VERSION } from "./app.ts"
import { spawnBridge } from "./bridge.ts"

export function parseArgs(argv: string[]): { model?: string; help: boolean; version: boolean } {
  const out: { model?: string; help: boolean; version: boolean } = { help: false, version: false }
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if (arg === "-h" || arg === "--help") out.help = true
    else if (arg === "-V" || arg === "--version") out.version = true
    else if (arg === "-m" || arg === "--model") out.model = argv[++i]
    else if (arg.startsWith("--model=")) out.model = arg.slice("--model=".length)
  }
  return out
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2))
  if (args.help) {
    console.log(`hyped-pro ${VERSION}\nUsage: hyped-pro [--model <name>]\n\nThe readline version is still available as hyped-plus.`)
    return
  }
  if (args.version) {
    console.log(`hyped-pro ${VERSION}`)
    return
  }
  const bridge = spawnBridge()
  // exitOnCtrlC only destroys the renderer: the bridge's pipes would keep
  // the process alive behind a blank terminal. App.quit closes both.
  const renderer = await createCliRenderer({ exitOnCtrlC: false, targetFps: 30 })
  const app = new App(renderer, bridge, { model: args.model })
  process.on("SIGTERM", () => app.quit())
  process.on("SIGINT", () => app.quit())
  await app.start()
}

if (import.meta.main) {
  main().catch((error) => {
    console.error(`hyped-pro: ${error instanceof Error ? error.message : String(error)}`)
    process.exit(1)
  })
}
