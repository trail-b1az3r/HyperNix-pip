// The site's colours, so hyped-pro looks like the HyperNix site it is
// documented on. These are the custom properties at the top of
// docs/src/index.css; tests/test_hyped_pro_otui.py reads both files and
// fails if they drift apart, so change them together.

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

export type PaletteName = keyof typeof palette

// What each part of the screen is painted with. Kept separate from the
// palette so a part can change colour without the palette losing its
// one-to-one match with the site.
export const theme = {
  background: palette.bg,
  panel: palette.surface1,
  userMessage: palette.surface2,
  selection: palette.surface3,
  border: palette.border,
  borderFocused: palette.accent,
  text: palette.text,
  textDim: palette.textDim,
  textMuted: palette.textMuted,
  hint: palette.textFaint,
  accent: palette.accent,
  accentText: palette.accentText,
  ok: palette.ok,
  error: palette.accentText,
} as const
