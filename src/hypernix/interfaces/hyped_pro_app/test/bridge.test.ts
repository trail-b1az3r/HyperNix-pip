import { describe, expect, test } from "bun:test"

import { Bridge, BridgeError, pythonCommand, type Transport } from "../src/bridge.ts"

function fake(): { bridge: Bridge; sent: any[]; closed: { value: boolean } } {
  const sent: any[] = []
  const closed = { value: false }
  const transport: Transport = {
    write: (line) => {
      expect(line.endsWith("\n")).toBe(true)
      sent.push(JSON.parse(line))
    },
    close: () => {
      closed.value = true
    },
  }
  return { bridge: new Bridge(transport), sent, closed }
}

describe("Bridge", () => {
  test("a request is one JSON line with an id and the command", () => {
    const { bridge, sent } = fake()
    bridge.request("chat", { model: "m", messages: [] })
    expect(sent).toEqual([{ model: "m", messages: [], id: 1, cmd: "chat" }])
  })

  test("the caller cannot override id or cmd through fields", () => {
    const { bridge, sent } = fake()
    bridge.request("ping", { id: 99, cmd: "download" })
    expect(sent[0]).toMatchObject({ id: 1, cmd: "ping" })
  })

  test("replies are matched by id, in any order", async () => {
    const { bridge } = fake()
    const a = bridge.request("catalog")
    const b = bridge.request("ping")
    bridge.feed(JSON.stringify({ id: b.id, ok: true, data: "b" }) + "\n")
    bridge.feed(JSON.stringify({ id: a.id, ok: true, data: "a" }) + "\n")
    expect(await a.done).toBe("a")
    expect(await b.done).toBe("b")
    expect(bridge.inFlight).toBe(0)
  })

  test("a line split across chunks is put back together", async () => {
    const { bridge } = fake()
    const r = bridge.request("ping")
    const line = JSON.stringify({ id: r.id, ok: true, data: { pong: true } }) + "\n"
    bridge.feed(line.slice(0, 7))
    bridge.feed(line.slice(7, 20))
    bridge.feed(line.slice(20))
    expect(await r.done).toEqual({ pong: true })
  })

  test("two replies in one chunk both land", async () => {
    const { bridge } = fake()
    const a = bridge.request("ping")
    const b = bridge.request("ping")
    bridge.feed(
      JSON.stringify({ id: a.id, ok: true, data: 1 }) + "\n" + JSON.stringify({ id: b.id, ok: true, data: 2 }) + "\n",
    )
    expect([await a.done, await b.done]).toEqual([1, 2])
  })

  test("a failure rejects with the bridge's code and message", async () => {
    const { bridge } = fake()
    const r = bridge.request("chat")
    bridge.feed(JSON.stringify({ id: r.id, ok: false, code: "HPC-CLOUD-001", error: "no key" }) + "\n")
    const error = await r.done.catch((e) => e)
    expect(error).toBeInstanceOf(BridgeError)
    expect(error.code).toBe("HPC-CLOUD-001")
    expect(error.message).toBe("no key")
  })

  test("stray stdout goes to the log instead of breaking the stream", async () => {
    const { bridge } = fake()
    const seen: string[] = []
    bridge.onLog = (line) => seen.push(line)
    const r = bridge.request("ping")
    bridge.feed("Loading weights...\n" + JSON.stringify({ id: r.id, ok: true, data: 1 }) + "\n")
    expect(await r.done).toBe(1)
    expect(seen).toEqual(["Loading weights..."])
  })

  test("a reply for an unknown id is ignored", () => {
    const { bridge } = fake()
    bridge.feed(JSON.stringify({ id: 404, ok: true, data: 1 }) + "\n")
    expect(bridge.inFlight).toBe(0)
  })

  test("the log keeps only the most recent lines", () => {
    const { bridge } = fake()
    bridge.feedLog(Array.from({ length: 250 }, (_, i) => `line ${i}`).join("\n"))
    expect(bridge.log.length).toBe(200)
    expect(bridge.log.at(-1)).toBe("line 249")
  })

  test("cancel names the request it stops", () => {
    const { bridge, sent } = fake()
    const r = bridge.request("chat")
    void bridge.cancel(r.id)
    expect(sent[1]).toMatchObject({ cmd: "cancel", target: r.id })
  })

  test("when the process goes away everything waiting fails, and later calls fail at once", async () => {
    const { bridge, sent } = fake()
    const r = bridge.request("chat")
    bridge.closedBy("the Python bridge exited (code 1)")
    const error = await r.done.catch((e) => e)
    expect(error.code).toBe("HPO-BRIDGE-001")
    expect(error.message).toContain("exited")
    const later = await bridge.call("ping").catch((e) => e)
    expect(later.code).toBe("HPO-BRIDGE-001")
    expect(sent.length).toBe(1)
  })

  test("close ends the transport", () => {
    const { bridge, closed } = fake()
    bridge.close()
    expect(closed.value).toBe(true)
  })
})

describe("pythonCommand", () => {
  test("uses the interpreter the launcher chose", () => {
    expect(pythonCommand({ HYPED_PRO_PYTHON: "/opt/py/bin/python3.12" })).toEqual([
      "/opt/py/bin/python3.12",
      "-m",
      "hypernix.interfaces.hyped_pro_bridge",
      "serve",
    ])
  })

  test("falls back to python3", () => {
    expect(pythonCommand({})[0]).toBe("python3")
  })
})
