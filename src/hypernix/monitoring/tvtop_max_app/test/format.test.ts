import { describe, expect, test } from "bun:test"

import {
  brailleGraph,
  fmtBytes,
  fmtCount,
  fmtDuration,
  fmtNumber,
  lineText,
  meter,
  rampColor,
  sparkline,
  truncate,
  truncateStart,
  wrap,
} from "../src/format.ts"
import { theme } from "../src/theme.ts"

describe("meter", () => {
  test("fills to the fraction and keeps its width", () => {
    const line = meter(0.5, 10)
    expect(lineText(line).length).toBe(10)
    expect(line.filter((s) => s.fg !== theme.border).length).toBe(5)
  })

  test("each filled cell takes its colour from its own position", () => {
    const full = meter(1, 10)
    expect(full[0].fg).toBe(theme.meterLow.toLowerCase())
    expect(full[9].fg).toBe(theme.meterHigh.toLowerCase())
    // A quarter bar shows only the cool end, not a bar coloured by value.
    expect(meter(0.25, 8)[0].fg).toBe(full[0].fg)
  })

  test("an unknown value is drawn as unknown, not as zero", () => {
    expect(lineText(meter(null, 6))).toBe("······")
    expect(lineText(meter(NaN, 6))).toBe("······")
  })

  test("clamps", () => {
    expect(lineText(meter(2, 4)).length).toBe(4)
    expect(meter(-1, 4).length).toBe(1)
  })
})

describe("rampColor", () => {
  test("ends and middle", () => {
    expect(rampColor(0)).toBe(theme.meterLow.toLowerCase())
    expect(rampColor(0.5)).toBe(theme.meterMid.toLowerCase())
    expect(rampColor(1)).toBe(theme.meterHigh.toLowerCase())
  })
})

describe("brailleGraph", () => {
  test("rows are the asked size, two samples per cell", () => {
    const rows = brailleGraph([0, 50, 100, 100], 2, 2)
    expect(rows.length).toBe(2)
    expect(rows.every((r) => r.length === 2)).toBe(true)
  })

  test("a full sample lights the whole column", () => {
    const [top, bottom] = brailleGraph([100, 100], 1, 2)
    expect(top).toBe("⣿")
    expect(bottom).toBe("⣿")
  })

  test("zero draws nothing", () => {
    expect(brailleGraph([0, 0], 1, 1)[0]).toBe("⠀")
  })

  test("newest on the right; missing history is blank on the left", () => {
    const [row] = brailleGraph([100], 2, 1)
    expect(row[0]).toBe("⠀")
    expect(row[1]).not.toBe("⠀")
  })
})

describe("text", () => {
  test("truncate keeps the start, truncateStart the end", () => {
    expect(truncate("abcdefgh", 5)).toBe("abcd…")
    expect(truncateStart("/very/long/path/train.log", 10)).toBe("…train.log")
    expect(truncate("abc", 5)).toBe("abc")
  })

  test("wrap never exceeds the width", () => {
    const lines = wrap("the quick brown fox jumps over a supercalifragilistic dog", 10)
    expect(lines.every((l) => l.length <= 10)).toBe(true)
    expect(lines.join(" ").replace(/\s+/g, "")).toBe("thequickbrownfoxjumpsoverasupercalifragilisticdog")
  })

  test("sparkline spans the range", () => {
    expect(sparkline([1, 2, 3], 3)).toBe("▁▅█")
  })
})

describe("numbers", () => {
  test("bytes, durations, counts", () => {
    expect(fmtBytes(1536)).toBe("1.5K")
    expect(fmtBytes(null)).toBe("—")
    expect(fmtDuration(59)).toBe("59s")
    expect(fmtDuration(3725)).toBe("1h 02m")
    expect(fmtDuration(90000)).toBe("1d 1h")
    expect(fmtCount(66_331_136)).toBe("66.3M")
    expect(fmtCount(7_000_000_000)).toBe("7.00B")
  })

  test("learning rates read as learning rates", () => {
    expect(fmtNumber(3e-4)).toBe("3.00e-4")
    expect(fmtNumber(2.3456789)).toBe("2.3457")
    expect(fmtNumber(Infinity)).toBe("—")
  })
})
