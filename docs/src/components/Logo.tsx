// HyperNix mark: three equal bars staggered up and to the right, dark →
// grey → red — raw input climbing through a pipeline to a fast, optimized
// result. Kept as inline SVG (not an <img src="data:...">) so it costs
// ~1KB instead of the ~56KB base64 PNG this replaced, and so `theme` can
// swap fills without shipping two raster files.
//
// See /assets/logo-new/README.md for the full system (mono, white, dark,
// favicon) and usage rules.

const BAR_BOTTOM = 'M6,52 L16,40 L42,40 L32,52 Z'
const BAR_MIDDLE = 'M12,34 L22,22 L48,22 L38,34 Z'
const BAR_TOP = 'M18,16 L28,4 L54,4 L44,16 Z'

export function LogoMark({ theme = 'dark', size = 28, className }: {
  theme?: 'dark' | 'light' | 'mono'
  size?: number
  className?: string
}) {
  // theme names describe the BACKGROUND the mark sits on: 'dark' background
  // gets off-white/light-grey bars, 'light' background gets charcoal/grey.
  // 'mono' uses currentColor so it can be tinted from CSS.
  const bottom = theme === 'dark' ? '#f2f2f0' : theme === 'mono' ? 'currentColor' : '#141414'
  const middle = theme === 'dark' ? '#8f8f8f' : theme === 'mono' ? 'currentColor' : '#6b6b6b'
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      role="img"
      aria-label="HyperNix"
      className={className}
      style={{ flexShrink: 0 }}
    >
      <path d={BAR_BOTTOM} fill={bottom} />
      <path d={BAR_MIDDLE} fill={middle} opacity={theme === 'mono' ? 0.72 : 1} />
      <path d={BAR_TOP} fill={theme === 'mono' ? 'currentColor' : '#c8192e'} />
    </svg>
  )
}

// Icon + wordmark lockup. `theme` again refers to the background it sits on.
export function LogoLockup({ theme = 'dark', height = 22, className }: {
  theme?: 'dark' | 'light'
  height?: number
  className?: string
}) {
  const textColor = theme === 'dark' ? '#f2f2f0' : '#141414'
  const bottom = theme === 'dark' ? '#f2f2f0' : '#141414'
  const middle = theme === 'dark' ? '#8f8f8f' : '#6b6b6b'
  const width = height * (330 / 64)
  return (
    <svg
      width={width}
      height={height}
      viewBox="0 0 330 64"
      role="img"
      aria-label="HyperNix"
      className={className}
      style={{ flexShrink: 0 }}
    >
      <g transform="translate(0,4.48) scale(0.86)">
        <path d={BAR_BOTTOM} fill={bottom} />
        <path d={BAR_MIDDLE} fill={middle} />
        <path d={BAR_TOP} fill="#c8192e" />
      </g>
      <text
        x={79}
        y={43}
        fontFamily="'Inter Variable','Inter','Helvetica Neue',Arial,sans-serif"
        fontWeight={800}
        fontSize={34}
        letterSpacing="-0.5"
        fill={textColor}
      >
        HyperNix
      </text>
    </svg>
  )
}
