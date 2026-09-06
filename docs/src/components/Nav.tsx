import { useState, useEffect, useRef } from 'react'
import { LogoMark } from './Logo'

const NAV_LABELS = { home: 'Home', docs: 'Docs', api: 'API', learn: 'Learn', stats: 'Stats', about: 'About' }

// Scrolls to the quickstart section on the homepage — used by the "Get
// started" nav button so it works the same whether you're already home
// or need to navigate there first.
function goToQuickstart(setPage, page) {
  if (page !== 'home') {
    setPage('home')
    // Wait one paint for HomePage to mount before the target exists.
    requestAnimationFrame(() => requestAnimationFrame(() => {
      document.getElementById('quickstart')?.scrollIntoView({ behavior: 'smooth' })
    }))
  } else {
    document.getElementById('quickstart')?.scrollIntoView({ behavior: 'smooth' })
  }
}

export function Nav({ page, setPage, scrolled, topOffset = 0, pages }) {
  const [drawerOpen, setDrawerOpen] = useState(false)
  const drawerRef = useRef(null)
  const menuBtnRef = useRef(null)

  useEffect(() => {
    if (!drawerOpen) return
    const onKey = (e) => { if (e.key === 'Escape') setDrawerOpen(false) }
    document.addEventListener('keydown', onKey)
    // Move focus into the drawer so keyboard/screen-reader users land
    // somewhere sensible instead of on a now-hidden trigger.
    drawerRef.current?.querySelector('button, a')?.focus()
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = ''
    }
  }, [drawerOpen])

  const closeDrawer = () => {
    setDrawerOpen(false)
    menuBtnRef.current?.focus()
  }

  return (
    <nav aria-label="Primary" style={{
      position: 'fixed', top: topOffset, left: 0, right: 0, zIndex: 100,
      background: scrolled ? 'rgba(13,13,13,0.96)' : 'transparent',
      backdropFilter: scrolled ? 'blur(12px)' : 'none',
      borderBottom: scrolled ? '1px solid var(--border)' : '1px solid transparent',
      transition: 'background-color 0.2s ease, border-color 0.2s ease',
    }}>
      <div style={{ maxWidth: 1120, margin: '0 auto', padding: '0 var(--space-gutter)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between', height: 58 }}>
        <button onClick={() => setPage('home')} className="press-btn"
          style={{ display: 'flex', alignItems: 'center', gap: 9, cursor: 'pointer',
            background: 'none', border: 'none', padding: 0, font: 'inherit' }}>
          <LogoMark theme="dark" size={26} />
          <span style={{ color: 'var(--text)', fontWeight: 700, fontSize: 16, letterSpacing: '-0.4px' }}>hypernix</span>
        </button>

        <div className="desk-nav" style={{ display: 'flex', gap: 26, alignItems: 'center' }}>
          {pages.map(p => (
            <button key={p} onClick={() => setPage(p)}
              aria-current={page === p ? 'page' : undefined}
              className={`nav-link${page === p ? ' active' : ''}`} style={{
                background: 'none', border: 'none', cursor: 'pointer', fontSize: 13.5,
                color: page === p ? 'var(--accent-text)' : 'var(--text-muted)',
                fontWeight: page === p ? 700 : 500, padding: '4px 0',
              }}>{NAV_LABELS[p] || p}</button>
          ))}
          <a href="https://github.com/trail-b1az3r/hypernix-pip" target="_blank" rel="noreferrer"
            className="underline-grow" style={{ color: 'var(--text-faint)', fontSize: 13.5, textDecoration: 'none' }}>
            GitHub ↗
          </a>
          <button onClick={() => goToQuickstart(setPage, page)} className="press-btn" style={{
            background: 'var(--accent)', color: '#fff', border: 'none', borderRadius: 6,
            padding: '8px 16px', fontSize: 13, fontWeight: 700, cursor: 'pointer',
          }}>Get started</button>
        </div>

        <button
          ref={menuBtnRef}
          className="mob-btn press-btn"
          onClick={() => setDrawerOpen(true)}
          aria-haspopup="true"
          aria-expanded={drawerOpen}
          aria-controls="mobile-nav-drawer"
          aria-label="Open menu"
          style={{ display: 'none', background: 'none', border: 'none', color: 'var(--text-muted)',
            fontSize: 22, cursor: 'pointer', lineHeight: 1 }}
        >☰</button>
      </div>

      {drawerOpen && (
        <>
          <div className="nav-drawer-backdrop" onClick={closeDrawer} aria-hidden="true" />
          <div
            id="mobile-nav-drawer"
            ref={drawerRef}
            className="nav-drawer"
            role="dialog"
            aria-modal="true"
            aria-label="Site menu"
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '18px 20px', borderBottom: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <LogoMark theme="dark" size={22} />
                <span style={{ color: 'var(--text)', fontWeight: 700, fontSize: 15 }}>hypernix</span>
              </div>
              <button onClick={closeDrawer} aria-label="Close menu" className="press-btn"
                style={{ background: 'none', border: 'none', color: 'var(--text-muted)',
                  fontSize: 22, cursor: 'pointer', lineHeight: 1, padding: 4 }}>×</button>
            </div>
            <div style={{ flex: 1, overflowY: 'auto' }}>
              {pages.map(p => (
                <button key={p} onClick={() => { setPage(p); closeDrawer() }}
                  aria-current={page === p ? 'page' : undefined}
                  className={`nav-drawer-link${page === p ? ' active' : ''}`}>
                  {NAV_LABELS[p] || p}
                </button>
              ))}
              <a href="https://github.com/trail-b1az3r/hypernix-pip" target="_blank" rel="noreferrer"
                className="nav-drawer-link" style={{ textDecoration: 'none' }}>GitHub ↗</a>
            </div>
            <div style={{ padding: 18 }}>
              <button onClick={() => { goToQuickstart(setPage, page); closeDrawer() }} style={{
                width: '100%', background: 'var(--accent)', color: '#fff', border: 'none',
                borderRadius: 8, padding: '13px 16px', fontSize: 14.5, fontWeight: 700, cursor: 'pointer',
              }}>Get started</button>
            </div>
          </div>
        </>
      )}
    </nav>
  )
}
