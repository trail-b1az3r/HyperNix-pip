import { useState, useEffect, useRef } from 'react'

export function CopyButton({ text }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      onClick={() => { navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 1500) }}
      className="press-btn"
      aria-label={copied ? 'Copied to clipboard' : 'Copy to clipboard'}
      style={{ position:'absolute', top:8, right:8, background:'var(--surface-2)', border:'1px solid var(--border)',
        borderRadius:4, color: copied ? 'var(--ok)' : 'var(--text-dim)', padding:'3px 8px', fontSize:11,
        cursor:'pointer', fontFamily:'var(--font-mono)',
        transform: copied ? 'scale(1.08)' : 'scale(1)',
        borderColor: copied ? 'var(--ok-border)' : 'var(--border)' }}>
      {copied ? '✓' : 'copy'}
    </button>
  )
}

export function CountUp({ value, duration = 900 }) {
  const [display, setDisplay] = useState(0)
  const prevValue = useRef(0)

  useEffect(() => {
    const from = prevValue.current
    const to = value
    if (from === to) return
    const start = performance.now()
    let raf
    const tick = (now) => {
      const elapsed = now - start
      const t = Math.min(1, elapsed / duration)
      // ease-out cubic
      const eased = 1 - Math.pow(1 - t, 3)
      setDisplay(Math.round(from + (to - from) * eased))
      if (t < 1) raf = requestAnimationFrame(tick)
      else prevValue.current = to
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [value, duration])

  return <>{display.toLocaleString()}</>
}

export function CodeBlock({ code }) {
  return (
    <div style={{ position:'relative', background:'var(--surface-1)', border:'1px solid var(--border)',
      borderRadius:6, padding:'12px 52px 12px 14px', margin:'8px 0' }}>
      <pre style={{ margin:0, fontSize:13, color:'var(--accent)', fontFamily:'var(--font-mono)',
        overflowX:'auto', whiteSpace:'pre-wrap', wordBreak:'break-all' }}>
        <span style={{ color:'var(--text-faint)' }}>$ </span>{code}
      </pre>
      <CopyButton text={code} />
    </div>
  )
}

// Page-level header for the non-home routes.
export function PageHeading({ kicker, title, lede, delay = 0.04 }) {
  return (
    <div style={{ marginBottom:30 }}>
      <div className="eyebrow anim-fade-up" style={{ color:'var(--accent)', marginBottom:12 }}>{kicker}</div>
      <h1 className="anim-fade-up" style={{ fontSize:'clamp(28px,4vw,38px)', fontWeight:800,
        color:'var(--text)', margin:'0 0 10px', letterSpacing:'-0.035em',
        animationDelay:`${delay}s` }}>{title}</h1>
      {lede && (
        <p className="anim-fade-up" style={{ color:'var(--text-dim)', margin:0, fontSize:15, lineHeight:1.65,
          maxWidth:'58ch', animationDelay:`${delay + 0.04}s` }}>{lede}</p>
      )}
    </div>
  )
}

// Shared section header: monospace kicker, title, optional lede.
export function SectionHeading({ kicker, title, lede, align = 'center', children }) {
  const centered = align === 'center'
  return (
    <div style={{ textAlign: centered ? 'center' : 'left', marginBottom:36,
      maxWidth: centered ? 620 : undefined, marginLeft: centered ? 'auto' : undefined,
      marginRight: centered ? 'auto' : undefined }}>
      {kicker && (
        <div className="eyebrow" style={{ color:'var(--accent)', marginBottom:12 }}>{kicker}</div>
      )}
      <h2 style={{ fontSize:'clamp(24px,3.4vw,34px)', fontWeight:800, color:'var(--text)',
        margin:0, letterSpacing:'-0.03em' }}>{title}</h2>
      {lede && (
        <p style={{ color:'var(--text-dim)', margin:'12px 0 0', fontSize:15, lineHeight:1.65 }}>{lede}</p>
      )}
      {children}
    </div>
  )
}
