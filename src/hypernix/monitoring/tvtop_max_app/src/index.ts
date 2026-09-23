#!/usr/bin/env bun
// tvtop-max: tvtop-pro on OpenTUI, with the run's script, model, optimizer,
// logs and warnings. tvtop-pro (the Rich version) is still installed.

import { createCliRenderer } from "@opentui/core"

import { App, VERSION } from "./app.ts"
import { pythonCommand, spawnBridge } from "./bridge.ts"
import type { Mode } from "./layout.ts"
import { PANELS, type PanelId } from "./panels.ts"

export interface Args {
  help: boolean
  version: boolean
  mode: Mode | "auto"
  log?: string
  script?: string
  pid?: number
  refreshMs: number
  hidden: PanelId[]
  error?: string
}

export const USAGE = `tvtop-max ${VERSION}
Usage: tvtop-max [options]

  -s, --small           phone layout: one column, 40-56 columns wide
  -w, --wide            desktop layout even on a narrow terminal
  -l, --log FILE        the training log to follow (found automatically)
  -S, --script FILE     the training script to read (found from the run)
  -P, --pid PID         the process to watch (the busiest Python by default)
  -r, --refresh SECS    seconds between samples (default 1)
      --hide a,b        start with panels hidden: ${PANELS.map((p) => p.id).join(", ")}
  -V, --version
  -h, --help

Keys: 0-9 toggle panels, s switch layout, p pause, r rescan, ↑↓ scroll, q quit.
tvtop-pro, the Rich version, is still available.`

export function parseArgs(argv: string[]): Args {
  const out: Args = { help: false, version: false, mode: "auto", refreshMs: 1000, hidden: [] }
  const value = (i: number, flag: string): string | undefined => {
    const next = argv[i + 1]
    if (next === undefined || next.startsWith("-")) out.error = `${flag} needs a value`
    return next
  }
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    const [flag, inline] = arg.includes("=") && arg.startsWith("--") ? [arg.slice(0, arg.indexOf("=")), arg.slice(arg.indexOf("=") + 1)] : [arg, undefined]
    const take = () => inline ?? (value(i, flag) !== undefined ? argv[++i] : undefined)
    switch (flag) {
      case "-h": case "--help": out.help = true; break
      case "-V": case "--version": out.version = true; break
      case "-s": case "--small": case "--phone": out.mode = "phone"; break
      case "-w": case "--wide": out.mode = "wide"; break
      case "-l": case "--log": out.log = take(); break
      case "-S": case "--script": out.script = take(); break
      case "-P": case "--pid": {
        const pid = Number(take())
        if (Number.isInteger(pid) && pid > 0) out.pid = pid
        else out.error = "--pid needs a process id"
        break
      }
      case "-r": case "--refresh": {
        const secs = Number(take())
        if (Number.isFinite(secs) && secs >= 0.2) out.refreshMs = Math.round(secs * 1000)
        else out.error = "--refresh needs a number of seconds, 0.2 or more"
        break
      }
      case "--hide": {
        const ids = (take() ?? "").split(",").map((s) => s.trim()).filter(Boolean)
        const unknown = ids.filter((id) => !PANELS.some((p) => p.id === id))
        if (unknown.length) out.error = `no panel called ${unknown.join(", ")}`
        out.hidden.push(...(ids.filter((id) => PANELS.some((p) => p.id === id)) as PanelId[]))
        break
      }
      default:
        out.error = `unknown option ${arg}`
    }
  }
  return out
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2))
  if (args.help) {
    console.log(USAGE)
    return
  }
  if (args.version) {
    console.log(`tvtop-max ${VERSION}`)
    return
  }
  if (args.error) {
    console.error(`tvtop-max: ${args.error}\n\n${USAGE}`)
    process.exit(2)
  }
  const bridge = spawnBridge(pythonCommand({ log: args.log, script: args.script, pid: args.pid }))
  const renderer = await createCliRenderer({ exitOnCtrlC: false, targetFps: 15 })
  const app = new App(renderer, bridge, { mode: args.mode, refreshMs: args.refreshMs, hidden: args.hidden })
  process.on("SIGTERM", () => app.quit())
  process.on("SIGINT", () => app.quit())
  await app.start()
}

if (import.meta.main) {
  main().catch((error) => {
    console.error(`tvtop-max: ${error instanceof Error ? error.message : String(error)}`)
    process.exit(1)
  })
}
