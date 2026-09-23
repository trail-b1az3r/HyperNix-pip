// tvtop-max's colours: the HyperNix site palette hyped-pro uses, plus the
// meter ramp tvtop-pro draws its bars with. tests/test_tvtop_max.py reads
// this file and hyped-pro's theme.ts and fails if the palettes drift.

export const palette = {
  bg: "#0d0d0d",
  surface1: "#111111",
  surface2: "#161616",
  surface3: "#1a1a1a",
  border: "#1f1f1f",
  borderStrong: "#252525",
  borderHover: "#2a2a2a",
  text: "#f5f5f4",
  textDim: "#a3a3a3",
  textMuted: "#8f8f8f",
  textFaint: "#7a7a7a",
  accent: "#c8192e",
  accentText: "#ff5b6c",
  ok: "#34c759",
} as const

export const theme = {
  background: palette.bg,
  panel: palette.surface1,
  border: palette.borderHover,
  borderFocused: palette.accent,
  title: palette.accentText,
  number: palette.textFaint,
  text: palette.text,
  textDim: palette.textDim,
  textMuted: palette.textMuted,
  hint: palette.textFaint,
  accent: palette.accent,
  accentText: palette.accentText,
  ok: palette.ok,
  warn: "#e5a50a",
  error: palette.accentText,
  // A meter's cells take their colour from their position along this
  // ramp, the way tvtop-pro and btop draw them: a full bar shows the
  // whole ramp, a quarter bar only the cool end.
  meterLow: "#34c759",
  meterMid: "#e5a50a",
  meterHigh: "#ff5b6c",
  graph: "#ff5b6c",
  graphDim: "#7a2830",
} as const
