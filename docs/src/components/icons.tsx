// Hand-drawn line icons used across the site: homepage feature grid,
// About page connect row, and seasonal event banner glyphs. All colored
// via stroke/fill props or currentColor rather than an icon font, so the
// palette never drifts from the site's own colors.

export function PrideFlagIcon({ size = 18 }) {
  const stripes = ['#e40303', '#ff8c00', '#ffed00', '#008026', '#004dff', '#750787']
  // Nested hoist-side chevron, largest (black) to smallest (white), drawn
  // back-to-front so each color shows as a band around the next.
  const chevrons = [
    { tip: 18,   fill: '#000000' },
    { tip: 15.6, fill: '#7a4a20' },
    { tip: 13.2, fill: '#5bcefa' },
    { tip: 10.8, fill: '#f5a9b8' },
    { tip: 8.4,  fill: '#ffffff' },
  ]
  const h = size * 0.66
  return (
    <svg width={size} height={h} viewBox="0 0 30 20" style={{ borderRadius: 2, flexShrink: 0 }}>
      {stripes.map((c, i) => (
        <rect key={c} x="0" y={i * (20 / 6)} width="30" height={20 / 6 + 0.6} fill={c} />
      ))}
      {chevrons.map((ch) => (
        <polygon key={ch.fill} points={`0,0 ${ch.tip},10 0,20`} fill={ch.fill} />
      ))}
    </svg>
  )
}

export function AmericanFlagIcon({ size = 18 }) {
  const h = size * 0.66
  const stripeH = h / 7
  const stars = [[2,2],[5,2],[8,2],[2,5],[5,5],[8,5],[2,8],[5,8]]
  return (
    <svg width={size} height={h} viewBox="0 0 30 20" style={{ borderRadius: 2, flexShrink: 0 }}>
      {[...Array(7)].map((_, i) => (
        <rect key={i} x="0" y={i * stripeH} width="30" height={stripeH + 0.4} fill={i % 2 === 0 ? '#b31942' : '#ffffff'} />
      ))}
      <rect x="0" y="0" width="13" height={stripeH * 4} fill="#0a3161" />
      {stars.map(([x, y], idx) => (
        <circle key={idx} cx={x + 1} cy={y + 1} r="0.6" fill="#ffffff" />
      ))}
    </svg>
  )
}

export function BirthdayIcon({ size = 18 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="#ffffff" strokeWidth="1.6"
      strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
      <path d="M12 2c-1.1 1.6-1.1 2.6 0 4" />
      <ellipse cx="12" cy="6.6" rx="1.1" ry="1.4" fill="#ffffff" stroke="none" />
      <line x1="12" y1="7.8" x2="12" y2="12" />
      <path d="M4 20v-6a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v6" />
      <path d="M4 20h16" />
      <path d="M6 12c0-1 1-1 1-2s-1-1-1-2M12 12c0-1 1-1 1-2s-1-1-1-2M18 12c0-1 1-1 1-2s-1-1-1-2" />
    </svg>
  )
}

export function ChristmasTreeIcon({ size = 18 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" style={{ flexShrink: 0 }}>
      <circle cx="12" cy="2.3" r="1.3" fill="#ffd76a" />
      <polygon points="12,2 16,9 8,9" fill="#1a7a3c" />
      <polygon points="12,6 17.5,14 6.5,14" fill="#1a7a3c" />
      <polygon points="12,10.5 19,20 5,20" fill="#1a7a3c" />
      <rect x="10.5" y="20" width="3" height="3" fill="#7a4a20" />
      <circle cx="9.5" cy="17" r="0.8" fill="#c8192e" />
      <circle cx="14.5" cy="17.5" r="0.8" fill="#ffd76a" />
      <circle cx="12" cy="13" r="0.8" fill="#c8192e" />
    </svg>
  )
}

export function NewYearIcon({ size = 18 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="#ffd76a" strokeWidth="1.6"
      strokeLinecap="round" style={{ flexShrink: 0 }}>
      <line x1="12" y1="2" x2="12" y2="8" />
      <line x1="12" y1="16" x2="12" y2="22" />
      <line x1="2" y1="12" x2="8" y2="12" />
      <line x1="16" y1="12" x2="22" y2="12" />
      <line x1="4.9" y1="4.9" x2="9" y2="9" />
      <line x1="15" y1="15" x2="19.1" y2="19.1" />
      <line x1="19.1" y1="4.9" x2="15" y2="9" />
      <line x1="9" y1="15" x2="4.9" y2="19.1" />
      <circle cx="12" cy="12" r="2.2" fill="#ffd76a" stroke="none" />
    </svg>
  )
}

export function HalloweenIcon({ size = 18 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" style={{ flexShrink: 0 }}>
      <path d="M12 3c1.5 1 1.5 2.6 0 4" stroke="#3a7d3a" strokeWidth="1.6" fill="none" strokeLinecap="round" />
      <ellipse cx="12" cy="13" rx="9" ry="7.5" fill="#ff8c1a" />
      <polygon points="8,11 9.6,13.4 6.8,13.4" fill="#0d0d0d" />
      <polygon points="16,11 17.6,13.4 14.8,13.4" fill="#0d0d0d" />
      <path d="M7.5 16.5c1.5 1.6 3.2 1.6 4.5 1s3-0.6 4.5 1" stroke="#0d0d0d" strokeWidth="1.4" fill="none" strokeLinecap="round" />
    </svg>
  )
}

export function ThanksgivingIcon({ size = 18 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" style={{ flexShrink: 0 }}>
      <path d="M12 2c3 3 3 6 0 9-3-3-3-6 0-9z" fill="#d9722a" />
      <path d="M12 11c4 1 6 4 4 8-4-1-6-4-4-8z" fill="#c8192e" />
      <path d="M12 11c-4 1-6 4-4 8 4-1 6-4 4-8z" fill="#e8960a" />
      <line x1="12" y1="11" x2="12" y2="21" stroke="#7a4a20" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  )
}

export function TransFlagIcon({ size = 18 }) {
  const stripes = ['#5bcefa', '#f5a9b8', '#ffffff', '#f5a9b8', '#5bcefa']
  const h = size * 0.66
  return (
    <svg width={size} height={h} viewBox="0 0 30 20" style={{ borderRadius: 2, flexShrink: 0 }}>
      {stripes.map((c, i) => (
        <rect key={i} x="0" y={i * 4} width="30" height="4.4" fill={c} />
      ))}
    </svg>
  )
}

export function BiFlagIcon({ size = 18 }) {
  const h = size * 0.66
  return (
    <svg width={size} height={h} viewBox="0 0 30 20" style={{ borderRadius: 2, flexShrink: 0 }}>
      <rect x="0" y="0" width="30" height="8" fill="#d60270" />
      <rect x="0" y="8" width="30" height="4" fill="#9b4f96" />
      <rect x="0" y="12" width="30" height="8" fill="#0038a8" />
    </svg>
  )
}

export function HeartIcon({ size = 18 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" style={{ flexShrink: 0 }}>
      <path d="M12 20.5c-4.5-3-9-6.4-9-11A5 5 0 0 1 12 6a5 5 0 0 1 9 3.5c0 4.6-4.5 8-9 11z" fill="#e0245e" />
    </svg>
  )
}

export function ShamrockIcon({ size = 18 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" style={{ flexShrink: 0 }}>
      <circle cx="9" cy="9" r="4.4" fill="#0f8a3c" />
      <circle cx="15" cy="9" r="4.4" fill="#0f8a3c" />
      <circle cx="12" cy="14" r="4.4" fill="#0f8a3c" />
      <line x1="12" y1="16" x2="12" y2="22" stroke="#0f8a3c" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  )
}

export function EarthIcon({ size = 18 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" style={{ flexShrink: 0 }}>
      <circle cx="12" cy="12" r="10" fill="#1f6feb" />
      <path d="M4 10c2-1 3 1 5 0s2-3 4-2 1 3 3 2 2-2 4-1" fill="none" stroke="#2fa84f" strokeWidth="2.4" strokeLinecap="round" />
      <ellipse cx="8" cy="17" rx="3" ry="1.8" fill="#2fa84f" />
      <ellipse cx="16" cy="6.5" rx="2.4" ry="1.5" fill="#2fa84f" />
    </svg>
  )
}

export function MexicanFlagIcon({ size = 18 }) {
  const h = size * 0.66
  return (
    <svg width={size} height={h} viewBox="0 0 30 20" style={{ borderRadius: 2, flexShrink: 0 }}>
      <rect x="0" y="0" width="10" height="20" fill="#006847" />
      <rect x="10" y="0" width="10" height="20" fill="#ffffff" />
      <rect x="20" y="0" width="10" height="20" fill="#ce1126" />
      <circle cx="15" cy="10" r="1.6" fill="#8a6d1e" />
    </svg>
  )
}

export function CandleIcon({ size = 18 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" style={{ flexShrink: 0 }}>
      <path d="M12 3c1.4 1.4 1.4 3-0.2 4.4-1.2 1-1.2 2 0.2 2.6" fill="none" stroke="#ffb648" strokeWidth="1.6" strokeLinecap="round" />
      <rect x="9" y="10" width="6" height="11" rx="1" fill="#e8e2d0" />
      <rect x="9" y="13" width="6" height="1.4" fill="#c8192e" />
    </svg>
  )
}

export function WomensDayIcon({ size = 18 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="#8a2be2" strokeWidth="1.8"
      strokeLinecap="round" style={{ flexShrink: 0 }}>
      <circle cx="12" cy="9" r="6" />
      <line x1="12" y1="15" x2="12" y2="22" />
      <line x1="8.5" y1="18.5" x2="15.5" y2="18.5" />
    </svg>
  )
}

export function FeatureIcon({ name, size = 24 }) {
  const common = { width: size, height: size, viewBox: '0 0 24 24', fill: 'none',
    stroke: '#c8192e', strokeWidth: 1.6, strokeLinecap: 'round', strokeLinejoin: 'round' }
  switch (name) {
    case 'download':
      return (
        <svg {...common}>
          <line x1="12" y1="3" x2="12" y2="14" />
          <polyline points="7,9 12,14 17,9" />
          <path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" />
        </svg>
      )
    case 'train':
      return (
        <svg {...common}>
          <circle cx="12" cy="5" r="2" />
          <circle cx="5" cy="18" r="2" />
          <circle cx="19" cy="18" r="2" />
          <circle cx="12" cy="12" r="1.7" fill="#c8192e" stroke="none" />
          <line x1="12" y1="7" x2="12" y2="10.3" />
          <line x1="6.6" y1="16.6" x2="10.6" y2="13.2" />
          <line x1="17.4" y1="16.6" x2="13.4" y2="13.2" />
        </svg>
      )
    case 'chat':
      return (
        <svg {...common}>
          <rect x="3" y="5" width="18" height="11" rx="2" />
          <polyline points="8,16 8,20 12,16" />
          <line x1="7" y1="9" x2="17" y2="9" />
          <line x1="7" y1="12.4" x2="14" y2="12.4" />
        </svg>
      )
    case 'quantize':
      return (
        <svg {...common}>
          <line x1="4" y1="20" x2="20" y2="20" />
          <rect x="5" y="14.5" width="3" height="5.5" fill="#c8192e" stroke="none" />
          <rect x="10.5" y="10" width="3" height="10" fill="#c8192e" stroke="none" />
          <rect x="16" y="5" width="3" height="15" fill="#c8192e" stroke="none" />
        </svg>
      )
    case 'vram':
      return (
        <svg {...common}>
          <rect x="4" y="5" width="16" height="10" rx="1.5" />
          <line x1="9" y1="19" x2="15" y2="19" />
          <line x1="12" y1="15" x2="12" y2="19" />
          <line x1="8" y1="9" x2="16" y2="9" />
          <line x1="8" y1="12" x2="13" y2="12" />
        </svg>
      )
    case 'evaluate':
      return (
        <svg {...common}>
          <circle cx="12" cy="12" r="9" />
          <polyline points="8,12.4 11,15.4 16,8.8" />
        </svg>
      )
    case 'preprocess':
      return (
        <svg {...common}>
          <polygon points="4,5 20,5 14,13 14,19 10,19 10,13" />
        </svg>
      )
    case 'ship':
      return (
        <svg {...common}>
          <polygon points="3,12 21,4 13,21 11,13 3,12" />
          <line x1="11" y1="13" x2="21" y2="4" />
        </svg>
      )
    default:
      return null
  }
}

// Minimal line icons for the About page's Connect row — abstract shapes
// rather than brand marks, and colored via currentColor so the .connect-link
// hover rule can tint them without extra props.
export function ConnectIcon({ name, size = 15 }) {
  const common = { width: size, height: size, viewBox: '0 0 24 24', fill: 'none',
    stroke: 'currentColor', strokeWidth: 1.6, strokeLinecap: 'round', strokeLinejoin: 'round' }
  switch (name) {
    case 'github':
      return (
        <svg {...common}>
          <polyline points="9,6 3,12 9,18" />
          <polyline points="15,6 21,12 15,18" />
        </svg>
      )
    case 'huggingface':
      return (
        <svg {...common}>
          <circle cx="12" cy="12" r="8" />
          <circle cx="9" cy="10.5" r="0.9" fill="currentColor" stroke="none" />
          <circle cx="15" cy="10.5" r="0.9" fill="currentColor" stroke="none" />
          <path d="M8 15c1.4 1.3 2.8 1.3 4 1s2.6 0.3 4-1" />
        </svg>
      )
    case 'models':
      return (
        <svg {...common}>
          <polygon points="12,3 21,8 12,13 3,8" />
          <polyline points="3,13 12,18 21,13" />
        </svg>
      )
    case 'pypi':
      return (
        <svg {...common}>
          <path d="M21 8l-9-5-9 5 9 5 9-5z" />
          <path d="M3 8v8l9 5 9-5V8" />
          <line x1="12" y1="13" x2="12" y2="21" />
        </svg>
      )
    case 'steam':
      return (
        <svg {...common}>
          <rect x="3" y="8" width="18" height="9" rx="4" />
          <line x1="6.6" y1="12.5" x2="6.6" y2="10.3" />
          <line x1="5.5" y1="11.4" x2="7.7" y2="11.4" />
          <circle cx="16" cy="11" r="1" fill="currentColor" stroke="none" />
          <circle cx="18.3" cy="13.2" r="1" fill="currentColor" stroke="none" />
        </svg>
      )
    default:
      return null
  }
}
