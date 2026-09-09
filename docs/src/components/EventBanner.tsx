// ── Seasonal site events ──────────────────────────────────────────────
// HyperNix's "birthday" is April 20, 2026 — the counter turns 1 on April
// 20, 2027, 2 the year after, and so on. Everything else here is a plain
// date-match (or date-range, for Pride Month) against the visitor's local
// clock. April Fools' is handled separately (see AprilFoolsPage) since it
// replaces the whole site for a minute instead of just showing a banner.
import {
  PrideFlagIcon, AmericanFlagIcon, BirthdayIcon, ChristmasTreeIcon, NewYearIcon,
  HalloweenIcon, ThanksgivingIcon, TransFlagIcon, BiFlagIcon, HeartIcon, ShamrockIcon,
  EarthIcon, MexicanFlagIcon, CandleIcon, WomensDayIcon,
} from './icons'

export const HYPERNIX_BIRTH_YEAR = 2026
export const BANNER_HEIGHT = 40

function nthWeekdayOfMonth(year, monthIndex0, weekday, n) {
  const d = new Date(year, monthIndex0, 1)
  let count = 0
  while (d.getMonth() === monthIndex0) {
    if (d.getDay() === weekday) {
      count += 1
      if (count === n) return d.getDate()
    }
    d.setDate(d.getDate() + 1)
  }
  return null
}

// Last occurrence of a given weekday in a month, e.g. Memorial Day (last
// Monday of May). weekday: 0=Sun..6=Sat.
function lastWeekdayOfMonth(year, monthIndex0, weekday) {
  const d = new Date(year, monthIndex0 + 1, 0) // last calendar day of the month
  while (d.getDay() !== weekday) {
    d.setDate(d.getDate() - 1)
  }
  return d.getDate()
}

export function getActiveSiteEvent(now = new Date()) {
  const month = now.getMonth() + 1
  const day = now.getDate()
  const year = now.getFullYear()

  if (month === 4 && day === 20 && year > HYPERNIX_BIRTH_YEAR) {
    return { id: `birthday-${year}`, kind: 'birthday', age: year - HYPERNIX_BIRTH_YEAR }
  }
  if (month === 12 && day === 25) {
    return { id: `christmas-${year}`, kind: 'christmas' }
  }
  if (month === 12 && day === 24) {
    return { id: `christmas-eve-${year}`, kind: 'christmas-eve' }
  }
  if (month === 12 && day === 31) {
    return { id: `newyearseve-${year}`, kind: 'newyearseve' }
  }
  if (month === 1 && day === 1) {
    return { id: `newyear-${year}`, kind: 'newyear' }
  }
  if (month === 1 && day === nthWeekdayOfMonth(year, 0, 1, 3)) {
    return { id: `mlk-${year}`, kind: 'mlk' }
  }
  if (month === 2 && day === 14) {
    return { id: `valentines-${year}`, kind: 'valentines' }
  }
  if (month === 3 && day === 8) {
    return { id: `iwd-${year}`, kind: 'iwd' }
  }
  if (month === 3 && day === 17) {
    return { id: `stpatricks-${year}`, kind: 'stpatricks' }
  }
  if (month === 3 && day === 31) {
    return { id: `tdov-${year}`, kind: 'tdov' }
  }
  if (month === 4 && day === 22) {
    return { id: `earthday-${year}`, kind: 'earthday' }
  }
  if (month === 5 && day === 5) {
    return { id: `cincodemayo-${year}`, kind: 'cincodemayo' }
  }
  if (month === 5 && day === lastWeekdayOfMonth(year, 4, 1)) {
    return { id: `memorialday-${year}`, kind: 'memorialday' }
  }
  if (month === 6 && day === 19) {
    return { id: `juneteenth-${year}`, kind: 'juneteenth' }
  }
  if (month === 10 && day === 31) {
    return { id: `halloween-${year}`, kind: 'halloween' }
  }
  if (month === 11 && day === nthWeekdayOfMonth(year, 10, 4, 4)) {
    return { id: `thanksgiving-${year}`, kind: 'thanksgiving' }
  }
  if (month === 11 && day === 20) {
    return { id: `tdor-${year}`, kind: 'tdor' }
  }
  if (month === 7 && (day === 3 || day === 4)) {
    return { id: `july4-${year}`, kind: 'july4', eve: day === 3 }
  }
  if (month === 9 && day === nthWeekdayOfMonth(year, 8, 1, 1)) {
    return { id: `laborday-${year}`, kind: 'laborday' }
  }
  if (month === 9 && day === 23) {
    return { id: `bivisibility-${year}`, kind: 'bivisibility' }
  }
  if (month === 10 && day === 11) {
    return { id: `comingout-${year}`, kind: 'comingout' }
  }
  if (month === 6) {
    return { id: `pride-${year}`, kind: 'pride' }
  }
  return null
}

export function isPreV1(version) {
  if (!version) return true
  const major = parseInt(String(version).replace(/^v/i, '').split('.')[0], 10)
  return Number.isNaN(major) || major < 1
}

export function EventBanner({ event, dismissed, onDismiss, version }) {
  if (!event || dismissed) return null
  const preV1 = isPreV1(version)

  let bg, content
  if (event.kind === 'pride') {
    bg = 'linear-gradient(90deg,#e40303,#ff8c00,#ffed00,#008026,#004dff,#750787)'
    content = (
      <>
        <PrideFlagIcon />
        <span>Happy Pride Month from HyperNix — built for every kitchen.</span>
        <PrideFlagIcon />
      </>
    )
  } else if (event.kind === 'july4') {
    bg = '#0a3161'
    content = (
      <>
        <AmericanFlagIcon />
        <span>{event.eve
          ? 'The 4th of July is tomorrow — from the HyperNix team!'
          : 'Happy 4th of July from the HyperNix team!'}</span>
        <AmericanFlagIcon />
      </>
    )
  } else if (event.kind === 'christmas') {
    bg = 'linear-gradient(90deg,#7a1414,#0f5c33)'
    content = (
      <>
        <ChristmasTreeIcon />
        <span>Merry Christmas from the HyperNix team!</span>
        <ChristmasTreeIcon />
      </>
    )
  } else if (event.kind === 'christmas-eve') {
    bg = 'linear-gradient(90deg,#0d1b2a,#173355)'
    content = (
      <>
        <ChristmasTreeIcon />
        <span>It's Christmas Eve — HyperNix wishes you a cozy night in.</span>
        <ChristmasTreeIcon />
      </>
    )
  } else if (event.kind === 'newyear') {
    bg = '#0d0d0d'
    content = (
      <>
        <NewYearIcon />
        <span>Happy New Year from HyperNix — welcome to {event.id.split('-')[1]}!</span>
        <NewYearIcon />
      </>
    )
  } else if (event.kind === 'halloween') {
    bg = '#170f00'
    content = (
      <>
        <HalloweenIcon />
        <span>Happy Halloween from the HyperNix team!</span>
        <HalloweenIcon />
      </>
    )
  } else if (event.kind === 'thanksgiving') {
    bg = '#3a2410'
    content = (
      <>
        <ThanksgivingIcon />
        <span>Happy Thanksgiving from HyperNix — thanks for cooking with us.</span>
        <ThanksgivingIcon />
      </>
    )
  } else if (event.kind === 'tdov') {
    bg = 'linear-gradient(90deg,#5bcefa,#f5a9b8,#ffffff,#f5a9b8,#5bcefa)'
    content = (
      <>
        <TransFlagIcon />
        <span>Happy Trans Day of Visibility from the HyperNix team.</span>
        <TransFlagIcon />
      </>
    )
  } else if (event.kind === 'tdor') {
    bg = '#0d1520'
    content = (
      <>
        <TransFlagIcon />
        <span>Transgender Day of Remembrance — honoring those we've lost.</span>
        <TransFlagIcon />
      </>
    )
  } else if (event.kind === 'bivisibility') {
    bg = 'linear-gradient(90deg,#d60270,#9b4f96,#0038a8)'
    content = (
      <>
        <BiFlagIcon />
        <span>Happy Bisexual Visibility Day from the HyperNix team.</span>
        <BiFlagIcon />
      </>
    )
  } else if (event.kind === 'comingout') {
    bg = '#1a0f24'
    content = (
      <>
        <PrideFlagIcon />
        <span>Happy National Coming Out Day from the HyperNix team.</span>
        <PrideFlagIcon />
      </>
    )
  } else if (event.kind === 'mlk') {
    bg = '#241505'
    content = (
      <>
        <CandleIcon />
        <span>Honoring Dr. Martin Luther King Jr. Day.</span>
        <CandleIcon />
      </>
    )
  } else if (event.kind === 'valentines') {
    bg = 'linear-gradient(90deg,#7a0f2b,#c81f4d)'
    content = (
      <>
        <HeartIcon />
        <span>Happy Valentine's Day from the HyperNix team!</span>
        <HeartIcon />
      </>
    )
  } else if (event.kind === 'iwd') {
    bg = '#2c0f3d'
    content = (
      <>
        <WomensDayIcon />
        <span>Happy International Women's Day from HyperNix.</span>
        <WomensDayIcon />
      </>
    )
  } else if (event.kind === 'stpatricks') {
    bg = '#052e14'
    content = (
      <>
        <ShamrockIcon />
        <span>Happy St. Patrick's Day from the HyperNix team!</span>
        <ShamrockIcon />
      </>
    )
  } else if (event.kind === 'earthday') {
    bg = 'linear-gradient(90deg,#0b3d66,#155d2f)'
    content = (
      <>
        <EarthIcon />
        <span>Happy Earth Day from HyperNix.</span>
        <EarthIcon />
      </>
    )
  } else if (event.kind === 'cincodemayo') {
    bg = '#1c0f05'
    content = (
      <>
        <MexicanFlagIcon />
        <span>¡Feliz Cinco de Mayo! from the HyperNix team.</span>
        <MexicanFlagIcon />
      </>
    )
  } else if (event.kind === 'memorialday') {
    bg = '#10141f'
    content = (
      <>
        <AmericanFlagIcon />
        <span>Memorial Day — remembering those who gave everything.</span>
        <AmericanFlagIcon />
      </>
    )
  } else if (event.kind === 'juneteenth') {
    bg = 'linear-gradient(90deg,#0a1f4d,#c8192e)'
    content = (
      <>
        <NewYearIcon />
        <span>Happy Juneteenth from the HyperNix team.</span>
        <NewYearIcon />
      </>
    )
  } else if (event.kind === 'laborday') {
    bg = '#0d1a2e'
    content = (
      <>
        <AmericanFlagIcon />
        <span>Happy Labor Day from the HyperNix team!</span>
        <AmericanFlagIcon />
      </>
    )
  } else if (event.kind === 'newyearseve') {
    bg = '#0d0d0d'
    content = (
      <>
        <NewYearIcon />
        <span>See you next year — Happy New Year's Eve from HyperNix!</span>
        <NewYearIcon />
      </>
    )
  } else {
    bg = '#1a0305'
    content = (
      <>
        <BirthdayIcon />
        <span>HyperNix turns {event.age} today — thanks for {event.age} year{event.age === 1 ? '' : 's'} of downloads!</span>
        <BirthdayIcon />
      </>
    )
  }

  return (
    <div role="status" aria-live="polite" className="anim-fade" style={{
      position: 'fixed', top: 0, left: 0, right: 0, zIndex: 101, height: BANNER_HEIGHT,
      background: bg, display: 'flex', alignItems: 'center', justifyContent: 'center',
      gap: 10, padding: '0 44px', fontSize: 12.5, fontWeight: 600, color: '#ffffff',
      textShadow: '0 1px 2px rgba(0,0,0,0.5)', textAlign: 'center',
    }}>
      {content}
      {preV1 && <span style={{ opacity: 0.75, fontWeight: 500 }}> · ray0rf1re</span>}
      <button onClick={onDismiss} aria-label="Dismiss banner" className="press-btn" style={{
        position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)',
        background: 'rgba(0,0,0,0.28)', border: 'none', color: '#ffffff', width: 22, height: 22,
        borderRadius: '50%', cursor: 'pointer', fontSize: 13, lineHeight: 1,
      }}>×</button>
    </div>
  )
}

// April Fools' (April 1st): swap the whole site out for a fake 404 for the
// first minute of the visit, then reveal the real site automatically — see
// the aprilFoolsActive state + timer in App().
export function AprilFoolsPage() {
  return (
    <div style={{
      minHeight: '100vh', background: '#0d0d0d', color: '#f0f0f0', display: 'flex',
      flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      textAlign: 'center', padding: 20,
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    }}>
      <div style={{ fontSize: 'clamp(72px,18vw,120px)', fontWeight: 900, color: '#c8192e', lineHeight: 1, marginBottom: 10 }}>404</div>
      <div style={{ fontSize: 18, color: '#888', marginBottom: 22 }}>This page could not be found.</div>
      <code style={{ fontSize: 12, color: '#333' }}>Error: HYPERNIX_SITE_NOT_FOUND</code>
    </div>
  )
}
