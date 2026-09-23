import { describe, expect, test } from "bun:test"

import { Bridge, pythonCommand, type Transport } from "../src/bridge.ts"
import { parseArgs, USAGE } from "../src/index.ts"
import { AUTO_PHONE_BELOW, chooseMode, phoneColumn, phoneWidth, PHONE_MAX_WIDTH, PHONE_ORDER, wideRows } from "../src/layout.ts"
import { PANELS, type PanelId } from "../src/panels.ts"

describe("parseArgs", () => {
  test("-s is the phone layout", () => {
    expect(parseArgs(["-s"]).mode).toBe("phone")
    expect(parseArgs(["--small"]).mode).toBe("phone")
    expect(parseArgs([]).mode).toBe("auto")
    expect(parseArgs(["--wide"]).mode).toBe("wide")
  })

  test("the run can be named", () => {
    const args = parseArgs(["-l", "train.log", "--script=train.py", "-P", "42", "-r", "2"])
    expect(args).toMatchObject({ log: "train.log", script: "train.py", pid: 42, refreshMs: 2000 })
    expect(args.error).toBeUndefined()
  })

  test("mistakes are reported, not guessed", () => {
    expect(parseArgs(["--pid", "x"]).error).toContain("process id")
    expect(parseArgs(["--refresh", "0"]).error).toContain("seconds")
    expect(parseArgs(["--nope"]).error).toContain("unknown option")
    expect(parseArgs(["--log"]).error).toContain("needs a value")
    expect(parseArgs(["--hide", "logs,nope"]).error).toContain("nope")
  })

  test("--hide", () => {
    expect(parseArgs(["--hide", "logs,procs"]).hidden).toEqual(["logs", "procs"])
  })

  test("the usage lists every panel", () => {
    for (const panel of PANELS) expect(USAGE).toContain(panel.id)
  })
})

describe("layout", () => {
  const all = new Set(PANELS.map((p) => p.id))

  test("wide places every panel exactly once", () => {
    const ids = wideRows(all).flatMap((row) => row.panels.map((p) => p.id))
    expect(ids.sort()).toEqual([...all].sort())
  })

  test("a hidden panel leaves the layout, and an emptied row goes too", () => {
    const visible = new Set<PanelId>(["logs", "warnings"])
    const rows = wideRows(visible)
    expect(rows.length).toBe(1)
    expect(rows[0].panels.map((p) => p.id)).toEqual(["logs", "warnings"])
  })

  test("phone is one column with every panel, the run first", () => {
    const column = phoneColumn(all)
    expect(column.map((c) => c.id).sort()).toEqual([...all].sort())
    expect(column[0].id).toBe("train")
    expect(column[1].id).toBe("warnings")
    expect(PHONE_ORDER.length).toBe(PANELS.length)
  })

  test("phone width fits a phone and previews one on a desktop", () => {
    expect(phoneWidth(45)).toBe(45)
    expect(phoneWidth(200)).toBe(PHONE_MAX_WIDTH)
    expect(phoneWidth(10)).toBe(24)
  })

  test("auto picks the phone layout on a narrow terminal", () => {
    expect(chooseMode("auto", AUTO_PHONE_BELOW - 1)).toBe("phone")
    expect(chooseMode("auto", 160)).toBe("wide")
    expect(chooseMode("wide", 40)).toBe("wide")
    expect(chooseMode("phone", 200)).toBe("phone")
  })
})

describe("bridge", () => {
  test("the python command names the run", () => {
    expect(pythonCommand({ log: "a.log", script: "t.py", pid: 7 }, { TVTOP_MAX_PYTHON: "/v/bin/python" })).toEqual([
      "/v/bin/python", "-m", "hypernix.monitoring.tvtop_max_bridge", "serve",
      "--log", "a.log", "--script", "t.py", "--pid", "7",
    ])
    expect(pythonCommand({}, {})[0]).toBe("python3")
  })

  test("replies are matched by id, in any order", async () => {
    const sent: string[] = []
    const transport: Transport = { write: (line) => sent.push(line), close: () => {} }
    const bridge = new Bridge(transport)
    const info = bridge.call("info")
    const frame = bridge.call("frame")
    bridge.feed('{"id": 2, "ok": true, "data": {"step": 5}}\n{"id": 1, "ok": tr')
    bridge.feed('ue, "data": {"script": "t.py"}}\n')
    expect(await frame).toEqual({ step: 5 })
    expect(await info).toEqual({ script: "t.py" })
    expect(JSON.parse(sent[0])).toEqual({ id: 1, cmd: "info" })
  })

  test("an error reply rejects with its code", async () => {
    const bridge = new Bridge({ write: () => {}, close: () => {} })
    const pending = bridge.call("nope")
    bridge.feed('{"id": 1, "ok": false, "code": "TVM-BRIDGE-001", "error": "unknown command"}\n')
    await expect(pending).rejects.toMatchObject({ code: "TVM-BRIDGE-001" })
  })

  test("a bridge that dies fails what was waiting", async () => {
    const bridge = new Bridge({ write: () => {}, close: () => {} })
    const pending = bridge.call("frame")
    bridge.closedBy("the Python bridge exited (code 1)")
    await expect(pending).rejects.toThrow("exited")
  })
})
