import { describe, expect, test } from "bun:test"

import { describeNoodleEvent, findModel, modelLabel, needsDownload, Session, type ModelInfo } from "../src/session.ts"

const models: ModelInfo[] = [
  { short: "kimi-k3", repo: "moonshot/kimi-k3", vendor: "moonshot", kind: "cloud", badge: "⚡" },
  { short: "qwen3.8", repo: "Qwen/Qwen3.8-9B", vendor: "huggingface", kind: "local" },
  { short: "qwen3.8-flash", repo: "Qwen/Qwen3.8-Flash", vendor: "huggingface", kind: "local" },
  { short: "hnx1-t1", repo: "hypernix/t1", vendor: "t1", kind: "local" },
]

describe("Session", () => {
  test("history is only user and assistant turns", () => {
    const s = new Session()
    s.add("user", "hi")
    s.add("note", "Model: x")
    s.add("assistant", "hello")
    s.add("error", "HPC-1 broke")
    s.add("user", "again")
    expect(s.history()).toEqual([
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
      { role: "user", content: "again" },
    ])
  })

  test("an empty reply is not sent back to the model", () => {
    const s = new Session()
    s.add("user", "hi")
    s.add("assistant", "")
    expect(s.history()).toEqual([{ role: "user", content: "hi" }])
  })

  test("a stopped reply is still part of the conversation", () => {
    const s = new Session()
    s.add("user", "hi")
    s.add("assistant", "partial", { cancelled: true })
    expect(s.history().at(-1)).toEqual({ role: "assistant", content: "partial" })
  })

  test("ids are unique and increasing", () => {
    const s = new Session()
    const a = s.add("user", "a")
    const b = s.add("assistant", "b")
    expect(b.id).toBeGreaterThan(a.id)
  })

  test("rewindToLastUser drops what came after the last question", () => {
    const s = new Session()
    s.add("user", "one")
    s.add("assistant", "1")
    s.add("user", "two")
    s.add("assistant", "2")
    s.add("error", "x")
    expect(s.rewindToLastUser()).toBe("two")
    expect(s.entries.map((e) => e.text)).toEqual(["one", "1", "two"])
  })

  test("rewindToLastUser with nothing asked", () => {
    const s = new Session()
    s.add("note", "hello")
    expect(s.rewindToLastUser()).toBeNull()
    expect(s.entries.length).toBe(1)
  })

  test("turns counts questions", () => {
    const s = new Session()
    s.add("user", "a")
    s.add("assistant", "b")
    s.add("user", "c")
    expect(s.turns).toBe(2)
    s.clear()
    expect(s.turns).toBe(0)
  })
})

describe("findModel", () => {
  test("exact short name, case-insensitive", () => {
    expect(findModel(models, "KIMI-K3").model?.short).toBe("kimi-k3")
  })

  test("exact name wins even when it prefixes another", () => {
    expect(findModel(models, "qwen3.8").model?.short).toBe("qwen3.8")
  })

  test("exact repo", () => {
    expect(findModel(models, "hypernix/t1").model?.short).toBe("hnx1-t1")
  })

  test("unique prefix", () => {
    expect(findModel(models, "hnx").model?.short).toBe("hnx1-t1")
  })

  test("ambiguous prefix returns the candidates and no model", () => {
    const found = findModel(models, "qwen")
    expect(found.model).toBeUndefined()
    expect(found.matches.map((m) => m.short)).toEqual(["qwen3.8", "qwen3.8-flash"])
  })

  test("unique substring", () => {
    expect(findModel(models, "flash").model?.short).toBe("qwen3.8-flash")
  })

  test("nothing", () => {
    expect(findModel(models, "gpt")).toEqual({ matches: [] })
    expect(findModel(models, "  ")).toEqual({ matches: [] })
  })
})

describe("model helpers", () => {
  test("only local non-T1 models need a download", () => {
    expect(models.map(needsDownload)).toEqual([false, true, true, false])
  })

  test("labels", () => {
    expect(modelLabel(models[0])).toBe("kimi-k3 ⚡")
    expect(modelLabel(models[1])).toBe("qwen3.8")
    expect(modelLabel(undefined)).toBe("no model")
  })

  test("noodle events", () => {
    expect(describeNoodleEvent({ kind: "task_started", task_id: "t1" })).toEqual({ kind: "task_started", detail: "t1" })
    expect(describeNoodleEvent({ type: "x", message: "m" })).toEqual({ kind: "x", detail: "m" })
    expect(describeNoodleEvent({})).toEqual({ kind: "event", detail: "" })
    const long = describeNoodleEvent({ kind: "k", text: "a".repeat(500) })
    expect(long.detail.length).toBe(120)
    expect(long.detail.endsWith("…")).toBe(true)
  })
})
