import { useState, useEffect } from 'react'
import { Nav } from './components/Nav'
import { Footer } from './components/Footer'
import { EventBanner, AprilFoolsPage, getActiveSiteEvent, BANNER_HEIGHT } from './components/EventBanner'
import { HomePage } from './pages/Home'
import { DocsPage } from './pages/Docs'
import { ApiPage } from './pages/Api'
import { StatsPage } from './pages/Stats'
import { AboutPage } from './pages/About'
import { LearnPage } from './pages/Learn'

const PAGES = ['home', 'docs', 'api', 'learn', 'stats', 'about']

export default function App() {
  const [page, setPage] = useState('home')
  const [scrolled, setScrolled] = useState(false)
  const [activeEvent] = useState(() => getActiveSiteEvent())
  const [bannerDismissed, setBannerDismissed] = useState(false)
  const bannerVisible = !!activeEvent && !bannerDismissed
  const [aprilFoolsActive, setAprilFoolsActive] = useState(() => {
    const now = new Date()
    return now.getMonth() + 1 === 4 && now.getDate() === 1
  })
  useEffect(() => {
    if (!aprilFoolsActive) return
    const t = setTimeout(() => setAprilFoolsActive(false), 60000) // 1 minute, then reveal the real site
    return () => clearTimeout(t)
  }, [aprilFoolsActive])
  const [version, setVersion] = useState('0.70.3')
  const [downloads, setDownloads] = useState({ last_day: 4, last_week: 460, last_month: 1474 })
  const [olderDownloads, setOlderDownloads] = useState(null)
  const [threeMonthDownloads, setThreeMonthDownloads] = useState(null)
  const [totalDownloads, setTotalDownloads] = useState(0)
  const [statsUpdatedAt, setStatsUpdatedAt] = useState(null)
  const [ghStats, setGhStats] = useState({ stars: 0, forks: 0, issues: 0 })
  const [pypiInfo, setPypiInfo] = useState(null)
  const [pythonVersionStats, setPythonVersionStats] = useState([])
  const [systemStats, setSystemStats] = useState([])
  const [releaseTimeline, setReleaseTimeline] = useState([])
  // Surfaces best-effort data-fetch failures in the UI instead of silently
  // showing stale zeroes forever — StatsPage reads this to show a small
  // inline notice rather than pretending the numbers are current.
  const [statsError, setStatsError] = useState(null)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 30)
    window.addEventListener('scroll', onScroll)
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  useEffect(() => {
    // The local stats JSON (updated daily by the update-json-stats workflow) is
    // the source of truth for download counts, including the "older than 30
    // days" totals that pypistats.org's live endpoints don't provide at all.
    // Different deployments have put this file at different paths over time
    // (docs/v1/json, v1.json, stats.json, or just json next to index.html), so
    // try them in order and use whichever one actually resolves instead of
    // silently falling back and losing the "older downloads" numbers.
    const STATS_JSON_CANDIDATES = ['./v1/json', './v1.json', './stats.json', './json']

    const loadStatsJson = async () => {
      for (const path of STATS_JSON_CANDIDATES) {
        try {
          const res = await fetch(path)
          if (!res.ok) continue
          const d = await res.json()
          let loadedOwnDownloads = false
          let loadedPythonVersions = false
          let loadedSystemStats = false
          if (d.version) setVersion(d.version)
          if (d.last_day !== undefined) {
            setDownloads({ last_day: d.last_day, last_week: d.last_week, last_month: d.last_month })
            loadedOwnDownloads = true
          }
          if (d.total_downloads !== undefined) setTotalDownloads(d.total_downloads)
          if (d.downloads && d.downloads.older) setOlderDownloads(d.downloads.older)
          if (d.downloads && d.downloads.three_month) setThreeMonthDownloads(d.downloads.three_month)
          if (d.updated_at) setStatsUpdatedAt(d.updated_at)
          // Accept either a pre-aggregated array (python_versions / operating_systems,
          // matching the common "pypi-package-stats" action schema) or the raw
          // pypistats-style { data: [{category, downloads}, ...] } shape, since
          // different versions of the update-json-stats workflow may use either.
          const pyVersions = d.python_versions || (d.python_minor && d.python_minor.data)
          if (pyVersions && pyVersions.length) {
            const m = {}
            pyVersions.forEach(i => {
              const cat = i.version || i.category
              if (cat && cat !== 'null') m[cat] = (m[cat] || 0) + i.downloads
            })
            setPythonVersionStats(Object.entries(m).map(([version, downloads]) => ({ version, downloads })).sort((a, b) => b.downloads - a.downloads))
            loadedPythonVersions = true
          }
          const osStats = d.operating_systems || (d.system && d.system.data)
          if (osStats && osStats.length) {
            const m = {}
            osStats.forEach(i => {
              const cat = i.os || i.category
              if (cat && cat !== 'null') m[cat] = (m[cat] || 0) + i.downloads
            })
            setSystemStats(Object.entries(m).map(([os, downloads]) => ({ os, downloads })).sort((a, b) => b.downloads - a.downloads))
            loadedSystemStats = true
          }
          return { loadedOwnDownloads, loadedPythonVersions, loadedSystemStats }
        } catch {
          // try the next candidate path
        }
      }
      const msg = `Could not load local stats JSON from any of: ${STATS_JSON_CANDIDATES.join(', ')} — falling back to pypistats.org directly. Note: pypistats.org does not send CORS headers, so those direct browser calls will fail too (see console) and recent/system/python-version stats will be unavailable until the local JSON is reachable.`
      console.warn(msg)
      setStatsError('Live stats are temporarily unavailable — showing the last known numbers.')
      return { loadedOwnDownloads: false, loadedPythonVersions: false, loadedSystemStats: false }
    }

    loadStatsJson().then(({ loadedOwnDownloads, loadedPythonVersions, loadedSystemStats }) => {
      // These pypistats.org fallbacks are best-effort only: pypistats.org does not
      // return Access-Control-Allow-Origin, so browsers will block them (CORS error
      // in console, harmless — caught below). Kept here in case pypistats.org adds
      // CORS support later, or this ever runs through a proxy/server context.
      if (!loadedOwnDownloads) {
        fetch('https://pypistats.org/api/packages/hypernix/recent')
          .then(r => r.json())
          .then(d => { if (d.data) setDownloads(d.data) })
          .catch(() => {})
      }
      if (!loadedPythonVersions) {
        fetch('https://pypistats.org/api/packages/hypernix/python_minor').then(r => r.json()).then(d => {
          if (!d.data) return
          const m = {}
          d.data.forEach(i => { if (i.category && i.category !== 'null') m[i.category] = (m[i.category] || 0) + i.downloads })
          setPythonVersionStats(Object.entries(m).map(([version, downloads]) => ({ version, downloads })).sort((a, b) => b.downloads - a.downloads))
        }).catch(() => {})
      }
      if (!loadedSystemStats) {
        fetch('https://pypistats.org/api/packages/hypernix/system').then(r => r.json()).then(d => {
          if (!d.data) return
          const m = {}
          d.data.forEach(i => { if (i.category && i.category !== 'null') m[i.category] = (m[i.category] || 0) + i.downloads })
          setSystemStats(Object.entries(m).map(([os, downloads]) => ({ os, downloads })).sort((a, b) => b.downloads - a.downloads))
        }).catch(() => {})
      }
    })

    fetch('https://pypi.org/pypi/hypernix/json').then(r => r.json()).then(d => {
      if (d.info) { setVersion(d.info.version); setPypiInfo(d.info) }
    }).catch(() => {})

    fetch('https://api.github.com/repos/trail-b1az3r/HyperNix-pip').then(r => r.json()).then(d => {
      if (d.stargazers_count !== undefined)
        setGhStats({ stars: d.stargazers_count, forks: d.forks_count, issues: d.open_issues_count })
    }).catch(() => {})

    fetch('https://api.github.com/repos/trail-b1az3r/HyperNix-pip/releases').then(r => r.json()).then(d => {
      if (!Array.isArray(d)) return
      setReleaseTimeline(d.slice(0, 15).map(r => ({
        version: r.tag_name,
        date: new Date(r.published_at).toLocaleDateString(),
        description: r.body ? r.body.split('\n')[0] : 'Release',
        isPreRelease: r.prerelease,
        url: r.html_url,
      })))
    }).catch(() => {})
  }, [])

  const common = { downloads, olderDownloads, threeMonthDownloads, totalDownloads, statsUpdatedAt, ghStats, pypiInfo, pythonVersionStats, systemStats, releaseTimeline, version, statsError }

  if (aprilFoolsActive) {
    return <AprilFoolsPage />
  }

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)', color: 'var(--text)',
      fontFamily: 'var(--font-sans)' }}>
      <a href="#main-content" className="skip-link">Skip to content</a>
      <EventBanner event={activeEvent} dismissed={bannerDismissed} onDismiss={() => setBannerDismissed(true)} version={version} />
      <Nav page={page} setPage={setPage} scrolled={scrolled} topOffset={bannerVisible ? BANNER_HEIGHT : 0} pages={PAGES} />
      <main id="main-content" style={{ paddingTop: bannerVisible ? BANNER_HEIGHT : 0, transition: 'padding-top 0.2s ease' }}>
        {page === 'home' && <HomePage setPage={setPage} {...common} />}
        {page === 'docs' && <DocsPage />}
        {page === 'api' && <ApiPage />}
        {page === 'learn' && <LearnPage />}
        {page === 'stats' && <StatsPage {...common} />}
        {page === 'about' && <AboutPage />}
      </main>
      <Footer page={page} setPage={setPage} version={version} pages={PAGES} />
    </div>
  )
}
