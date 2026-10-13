// 0.72.8: the five hidden panels, per-core temperatures, memory that adds
// up, the volume in dB with its warning, and the keys that select, sort
// and scroll a panel.

import { describe, expect, test } from "bun:test"

import { initialVisible, SCROLL_STEP, Selection } from "../src/app.ts"
import { lineText, seg, sliceLine } from "../src/format.ts"
import { parseArgs } from "../src/index.ts"
import { drawOrder } from "../src/layout.ts"
import { allFindings, BUILDERS, horizontalView, PANELS, panelText, panelTitle, type PanelContext, type PanelId } from "../src/panels.ts"
import { theme } from "../src/theme.ts"
import type { Frame } from "../src/types.ts"

const FRAME: Frame = {
  cpu_percent: 50,
  cpu_per_core: [10, 90, 30, 70],
  cpu_temps: { per_core: [50, 91, 55, 82], package_c: 92, source: "coretemp" },
  ram_percent: 99, // a stale psutil figure: the panel must not use it
  memory: { total_mib: 16000, used_mib: 4000, available_mib: 12000, free_mib: 2000, cached_mib: 6000, swap_total_mib: 0 },
  memory_users: [
    { pid: 1, name: "small", command: "small", rss_bytes: 100 * 2 ** 20, growth_bps: 200 * 2 ** 20 },
    { pid: 2, name: "big", command: "big", rss_bytes: 4 * 2 ** 30, growth_bps: 0 },
    { pid: 3, name: "mid", command: "mid", rss_bytes: 2 ** 30, growth_bps: -1024 },
  ],
  processes: [
    { pid: 30, command: "python a.py", cpu_percent: 5, rss_bytes: 9e9, age_seconds: 10 },
    { pid: 10, command: "python b.py", cpu_percent: 80, rss_bytes: 1e9, age_seconds: 99 },
  ],
  log_lines: ["x".repeat(200), "short"],
  audio: {
    available: true, backend: "pipewire", device: "USB Headset", headphones: true,
    volume_db: -2.3, volume_percent: 92, muted: false, playing: true,
    signal_peak_dbfs: -3, signal_rms_dbfs: -15, level: "loud",
    warning: "loud: playing at -2.3 dB on headphones — at or above -6 dB, long listening risks hearing damage",
    note: "dB is the output gain the sound server applies",
  },
  disks: [
    { mount: "/", device: "sda1", total_bytes: 1e12, used_bytes: 5e11, percent: 50, read_bps: 1e6, write_bps: 0 },
    { mount: "/data", device: "sdb1", total_bytes: 4e12, used_bytes: 3.6e12, percent: 90, read_bps: 0, write_bps: 0 },
  ],
  network: [{ name: "eth0", recv_bytes: 1e9, sent_bytes: 2e9, down_bps: 1e6, up_bps: 2e5, link_mbps: 1000 }],
  system: { hostname: "box", os: "Linux 6.8", load: [5, 4, 3], uptime_seconds: 90000, processes: 300, cores: 4, battery: null },
}

const ctx = (over: Partial<PanelContext> = {}): PanelContext => ({ info: null, frame: FRAME, width: 60, height: 30, compact: false, ...over })

describe("the hidden panels", () => {
  test("start hidden, and --show or Shift+number brings them", () => {
    const visible = initialVisible()
    for (const id of ["temps", "audio", "disks", "net", "system"] as PanelId[]) expect(visible.has(id)).toBe(false)
    expect(visible.has("cpu")).toBe(true)
    expect(initialVisible([], ["audio"]).has("audio")).toBe(true)
    expect(initialVisible(["cpu"]).has("cpu")).toBe(false)
    expect(parseArgs(["--show", "audio,net"]).shown).toEqual(["audio", "net"])
  })

  for (const id of ["temps", "audio", "disks", "net", "system"] as PanelId[]) {
    test(`${id} stays inside its box with real data`, () => {
      for (const width of [24, 40, 80]) {
        for (const height of [3, 8, 20]) {
          const lines = BUILDERS[id](ctx({ width, height }))
          expect(lines.length).toBeLessThanOrEqual(height)
          for (const line of lines) expect(lineText(line).length).toBeLessThanOrEqual(width)
        }
      }
    })
  }
})

describe("cpu temperatures", () => {
  test("each core shows its own, hot ones coloured", () => {
    const lines = BUILDERS.cpu(ctx({ width: 40 }))
    const text = lines.map(lineText).join("\n")
    expect(text).toContain("90% 91°")
    expect(text).toContain("10% 50°")
    expect(text).toContain("92°") // the package, on the total line
    const hot = lines.find((l) => lineText(l).includes("91°"))!
    expect(hot[hot.length - 1].fg).toBe(theme.error)
  })

  test("sorted busiest or hottest first", () => {
    const busiest = panelText("cpu", ctx({ width: 40, sort: "busiest" }))
    expect(busiest[1]).toContain("c1")
    expect(busiest[2]).toContain("c3")
    const hottest = panelText("cpu", ctx({ width: 40, sort: "hottest" }))
    expect(hottest[1]).toContain("c1")
    expect(hottest[2]).toContain("c3")
  })

  test("a driver without per-core readings says so", () => {
    const frame = { ...FRAME, cpu_temps: { per_core: [null, null, null, null], package_c: null, note: "no CPU temperature sensor is exposed here" } }
    expect(panelText("cpu", ctx({ frame, width: 80 })).join("\n")).toContain("no CPU temperature sensor")
  })

  test("the temperatures panel lists them", () => {
    const text = panelText("temps", ctx({ sort: "hottest" }))
    expect(text[0]).toContain("92°C")
    expect(text[1]).toContain("c1")
  })
})

describe("memory", () => {
  test("the bar is used over total, the same numbers it prints", () => {
    const text = panelText("mem", ctx())
    expect(text[0]).toContain("3.9G/15.6G 25%")
    expect(text[0]).not.toContain("99")
    expect(text.join("\n")).toContain("avail 11.7G")
  })

  test("its biggest users, sorted three ways", () => {
    const order = (sort?: string) =>
      panelText("mem", ctx({ sort })).filter((l) => /\b(small|big|mid)$/.test(l)).map((l) => l.trim().split(/\s+/).pop())
    expect(order()).toEqual(["big", "mid", "small"])
    expect(order("least used")).toEqual(["small", "mid", "big"])
    expect(order("fastest growing")).toEqual(["small", "big", "mid"])
  })

  test("a container's limit is said", () => {
    const frame = { ...FRAME, memory: { ...FRAME.memory, source: "cgroup" } as unknown as Record<string, number> }
    expect(panelText("mem", ctx({ frame })).join("\n")).toContain("container's memory limit")
  })
})

describe("sound", () => {
  test("the volume in dB, what is playing, and the warning", () => {
    const text = panelText("audio", ctx()).join("\n")
    expect(text).toContain("-2.3 dB")
    expect(text).toContain("▶ playing")
    expect(text).toContain("peak -3.0")
    expect(text).toContain("loud: playing at -2.3 dB")
  })

  test("the warning reaches the warnings panel and the title", () => {
    expect(allFindings(null, FRAME).some((f) => f.source === "audio" && f.level === "warn")).toBe(true)
    const unsafe = { ...FRAME, audio: { ...FRAME.audio, level: "unsafe" as const, warning: "UNSAFE: amplified" } }
    expect(allFindings(null, unsafe).find((f) => f.source === "audio")?.level).toBe("error")
    expect(panelTitle(PANELS.find((p) => p.id === "audio")!, { info: null, frame: unsafe })).toBe("sound ✖")
  })

  test("quiet is not a finding", () => {
    const quiet = { ...FRAME, audio: { ...FRAME.audio, level: "ok" as const, warning: "" } }
    expect(allFindings(null, quiet).length).toBe(0)
  })

  test("no sound server is said", () => {
    const frame = { ...FRAME, audio: { available: false, note: "no sound server answered" } }
    expect(panelText("audio", ctx({ frame }))[0]).toContain("no sound server")
  })
})

describe("processes, disks and network sort", () => {
  test("processes by cpu, memory, age or pid", () => {
    const first = (sort?: string) => panelText("procs", ctx({ sort }))[1].split(/\s+/)[0]
    expect(first()).toBe("10")
    expect(first("memory")).toBe("30")
    expect(first("age")).toBe("10")
    expect(first("pid")).toBe("10")
    expect(panelText("procs", ctx({ sort: "memory" }))[0]).toContain("by memory")
  })

  test("disks by fullness, size or activity", () => {
    expect(panelText("disks", ctx())[0]).toContain("/data")
    expect(panelText("disks", ctx({ sort: "busiest" }))[0]).toContain("/")
    expect(panelText("disks", ctx({ sort: "busiest" }))[0]).not.toContain("/data")
  })

  test("system: load against the cores", () => {
    const lines = BUILDERS.system(ctx())
    const load = lines.find((l) => lineText(l).startsWith("load"))!
    expect(lineText(load)).toContain("/ 4 cores")
    expect(load[1].fg).toBe(theme.warn)
  })
})

describe("selecting, sorting and scrolling", () => {
  test("← walks the visible panels in drawing order and wraps", () => {
    const order = drawOrder("wide", new Set<PanelId>(["cpu", "mem", "logs"]))
    expect(order).toEqual(["cpu", "mem", "logs"])
    const s = new Selection()
    expect(s.cycle(order)).toBe("cpu")
    expect(s.cycle(order)).toBe("mem")
    expect(s.cycle(order)).toBe("logs")
    expect(s.cycle(order)).toBe("cpu")
    expect(s.cycle(order, true)).toBe("logs")
    s.clear()
    expect(s.selected).toBeNull()
  })

  test("o cycles the selected panel's sorts, and says when it cannot", () => {
    const s = new Selection()
    expect(s.nextSort().ok).toBe(false)
    s.selected = "procs"
    expect(s.sort("procs")).toBe("cpu")
    expect(s.nextSort().message).toBe("processes: sorted by memory")
    expect(s.sort("procs")).toBe("memory")
    for (let i = 0; i < 3; i++) s.nextSort()
    expect(s.sort("procs")).toBe("cpu")
    s.selected = "gpu"
    expect(s.nextSort()).toEqual({ ok: false, message: "gpu has nothing to sort" })
  })

  test("→ scrolls a panel wider than its box, Shift+→ back, never past the ends", () => {
    const s = new Selection()
    s.selected = "logs"
    const c = ctx({ width: 40, height: 5 })
    const first = horizontalView("logs", c, 0)
    expect(first.maxOffset).toBe(160)
    s.setLimit("logs", first.maxOffset, first.offset)
    expect(s.scroll(false).ok).toBe(true)
    expect(s.offset("logs")).toBe(SCROLL_STEP)
    const moved = horizontalView("logs", c, s.offset("logs"))
    expect(moved.lines.every((l) => lineText(l).length <= 40)).toBe(true)
    s.scroll(true)
    s.scroll(true)
    expect(s.offset("logs")).toBe(0)
    for (let i = 0; i < 100; i++) s.scroll(false)
    expect(s.offset("logs")).toBe(160)
  })

  test("a panel that fits, or draws meters, does not scroll", () => {
    const s = new Selection()
    s.selected = "gpu"
    s.setLimit("gpu", 0, 0)
    expect(s.scroll(false).message).toContain("nothing to scroll")
    expect(horizontalView("cpu", ctx(), 50).maxOffset).toBe(0)
    expect(new Selection().scroll(false).message).toContain("select a panel")
  })

  test("sliceLine keeps each segment's colour", () => {
    const line = [seg("abc", "#111111"), seg("defgh", "#222222")]
    expect(sliceLine(line, 2, 3)).toEqual([seg("c", "#111111"), seg("de", "#222222")])
    expect(lineText(sliceLine(line, 6, 10))).toBe("gh")
  })
})
