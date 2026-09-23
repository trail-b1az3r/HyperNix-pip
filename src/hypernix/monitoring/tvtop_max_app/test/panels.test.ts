import { describe, expect, test } from "bun:test"

import { lineText } from "../src/format.ts"
import { allFindings, BUILDERS, logLineColor, PANELS, panelText, panelTitle, type PanelContext, type PanelId } from "../src/panels.ts"
import { theme } from "../src/theme.ts"
import type { Frame, Info } from "../src/types.ts"

const INFO: Info = {
  script: "/runs/a/train.py",
  log: "/runs/a/train.log",
  pid: 4242,
  hypernix_version: "0.72.6-rc2",
  arch: {
    source: "script",
    arch: "qwen3",
    fields: { hidden_size: 512, num_hidden_layers: 8 },
    preset: { rope_theta: 1000000, tie_word_embeddings: true },
    params: 66331136,
    line: 10,
  },
  report: {
    path: "/runs/a/train.py",
    imports: [
      { module: "hypernix.models.neo_oven", kind: "hypernix", names: ["new_oven"], summary: "the oven" },
      { module: "hypernix.models.old_oven", kind: "hypernix", deprecated: "hypernix.models.neo_oven" },
      { module: "torch", kind: "library", version: "2.14.0" },
      { module: "notreal", kind: "missing" },
      { module: "os", kind: "stdlib" },
    ],
    cookers: [
      { name: "PressureCookerV5", generation: "v5", kwargs: { lr: 3e-4 }, line: 11 },
      { name: "PressureCooker", generation: "v1", deprecated: "PressureCookerV4 (kept)", line: 13 },
    ],
    findings: [
      { level: "warn", source: "script", message: "torch.load without weights_only=True", line: 12 },
      { level: "info", source: "script", message: "no seed is set" },
    ],
  },
}

const FRAME: Frame = {
  has_training_data: true,
  step: 50,
  total_steps: 100,
  loss: 2.1,
  lr: 3e-4,
  throughput: 2.5,
  eta_seconds: 20,
  elapsed_seconds: 20,
  cpu_percent: 40,
  cpu_per_core: [10, 20, 30, 40],
  cpu_history: [10, 40, 80],
  ram_percent: 50,
  memory: { total_mib: 16000, used_mib: 8000, cached_mib: 1000, free_mib: 7000, swap_total_mib: 0 },
  gpu_name: "NVIDIA GeForce GTX 1080",
  gpu_util_percent: 90,
  gpu_mem_used_mib: 4000,
  gpu_mem_total_mib: 8000,
  gpu_temp_c: 70,
  gpu_power_w: 150,
  gpu_power_limit_w: 180,
  recent_losses: [3, 2.8, 2.5, 2.1],
  log_lines: ["step 49/100 loss=2.2", "UserWarning: something", "Traceback (most recent call last):", "step 50/100 loss=2.1"],
  log_findings: [{ level: "error", source: "log", message: "the run raised an exception", line: 3 }],
  processes: [{ pid: 4242, command: "python train.py", cpu_percent: 99, rss_bytes: 1e9, age_seconds: 60 }],
}

const ctx = (over: Partial<PanelContext> = {}): PanelContext => ({ info: INFO, frame: FRAME, width: 50, height: 20, compact: false, ...over })

describe("every panel", () => {
  for (const id of PANELS.map((p) => p.id) as PanelId[]) {
    test(`${id} stays inside its width and height`, () => {
      for (const width of [24, 40, 80]) {
        for (const height of [3, 8, 20]) {
          const lines = BUILDERS[id](ctx({ width, height }))
          expect(lines.length).toBeLessThanOrEqual(height)
          for (const line of lines) expect(lineText(line).length).toBeLessThanOrEqual(width)
        }
      }
    })

    test(`${id} copes with nothing yet`, () => {
      const lines = BUILDERS[id]({ info: null, frame: null, width: 40, height: 10, compact: false })
      expect(lines.length).toBeGreaterThan(0)
    })

    test(`${id} copes with the phone layout`, () => {
      expect(BUILDERS[id](ctx({ width: 40, compact: true })).length).toBeGreaterThan(0)
    })
  }

  test("the panel keys are the ten number keys, once each", () => {
    expect(PANELS.map((p) => p.key).sort()).toEqual(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
  })
})

describe("what the new panels say", () => {
  test("modules: HyperNix modules, libraries with versions, missing ones, deprecations", () => {
    const text = panelText("modules", ctx()).join("\n")
    expect(text).toContain("HyperNix 0.72.6-rc2")
    expect(text).toContain("models.neo_oven (new_oven)")
    expect(text).toContain("torch")
    expect(text).toContain("2.14.0")
    expect(text).toContain("not installed")
    expect(text).toContain("notreal")
    expect(text).toContain("deprecated → hypernix.models.neo_oven")
  })

  test("model: architecture, where it came from, parameters, fields and preset", () => {
    const text = panelText("model", ctx()).join("\n")
    expect(text).toContain("qwen3")
    expect(text).toContain("from script :10")
    expect(text).toContain("~66.3M")
    expect(text).toContain("hidden")
    expect(text).toContain("512")
    expect(text).toContain("rope θ")
  })

  test("pressure cooker: each one, its generation, arguments, deprecation and the live lr", () => {
    const text = panelText("cooker", ctx()).join("\n")
    expect(text).toContain("PressureCookerV5")
    expect(text).toContain("V5")
    expect(text).toContain("3.00e-4")
    expect(text).toContain("deprecated: use PressureCookerV4 (kept)")
    expect(text).toContain("lr now (log)")
  })

  test("warnings: script and log together, errors first, with where", () => {
    const text = panelText("warnings", ctx({ height: 40 }))
    expect(text[0]).toContain("✖ the run raised an exception")
    expect(text.join("\n")).toContain("log:3")
    expect(text.join("\n")).toContain("script:12")
    expect(allFindings(INFO, FRAME).map((f) => f.level)).toEqual(["error", "warn", "info"])
  })

  test("warnings: the title counts them", () => {
    expect(panelTitle(PANELS.find((p) => p.id === "warnings")!, { info: INFO, frame: FRAME })).toBe("warnings 1✖ 1▲")
  })

  test("warnings: a script that does not parse is an error even with no findings", () => {
    const info = { ...INFO, report: { error: "syntax error on line 3: invalid syntax" } }
    expect(panelText("warnings", ctx({ info, frame: null }))[0]).toContain("syntax error on line 3")
  })

  test("logs: the newest lines, coloured by what they are", () => {
    const text = panelText("logs", ctx({ height: 2 }))
    expect(text).toEqual(["Traceback (most recent call last):", "step 50/100 loss=2.1"])
    expect(logLineColor("Traceback (most recent call last):")).toBe(theme.error)
    expect(logLineColor("UserWarning: x")).toBe(theme.warn)
    expect(logLineColor("step 5 loss=1")).toBe(theme.text)
    expect(logLineColor("loss = nan")).toBe(theme.error)
  })

  test("logs: scrolled back, it says how far", () => {
    const text = panelText("logs", ctx({ height: 2, logOffset: 1 }))
    expect(text[0]).toBe("UserWarning: something")
    expect(text[1]).toContain("1 newer lines")
  })

  test("training: progress, loss, lr, eta", () => {
    const text = panelText("train", ctx()).join("\n")
    expect(text).toContain("50/100")
    expect(text).toContain("2.1")
    expect(text).toContain("eta")
  })

  test("training: a log nobody has written to for a while is called out", () => {
    const text = panelText("train", ctx({ frame: { ...FRAME, log_age_seconds: 7200 } })).join("\n")
    expect(text).toContain("log untouched for 2h 00m")
  })

  test("processes: the watched one is marked", () => {
    const lines = BUILDERS.procs(ctx())
    expect(lines[1][0].fg).toBe(theme.accentText)
  })

  test("gpu: no GPU is said, not drawn as zero", () => {
    const text = panelText("gpu", ctx({ frame: { ...FRAME, gpu_name: null, gpu_util_percent: null } }))
    expect(text[0]).toContain("no NVIDIA GPU")
  })
})
