import { LogoMark } from './Logo'

export function Footer({ page, setPage, version, pages }) {
  return (
    <footer style={{ borderTop: '1px solid var(--surface-2)', padding: '40px var(--space-gutter) 34px',
      marginTop: 40 }}>
      <div className="shell" style={{ display: 'flex', flexWrap: 'wrap', gap: 24,
        alignItems: 'flex-start', justifyContent: 'space-between' }}>
        <div style={{ minWidth: 220 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 9, marginBottom: 9 }}>
            <LogoMark theme="dark" size={20} />
            <span style={{ color: 'var(--text-faint)', fontSize: 13, fontWeight: 600 }}>hypernix</span>
          </div>
          <div style={{ color: 'var(--text-faint)', fontSize: 11.5, lineHeight: 1.7 }}>
            v{version} — LLU-0.1 / HOS-1.0 (dual)<br/>
            Built for Pascal-class and newer consumer GPUs.
          </div>
        </div>

        <div style={{ display: 'flex', gap: 44, flexWrap: 'wrap' }}>
          <div>
            <div className="eyebrow" style={{ color: 'var(--text-faint)', marginBottom: 11, fontSize: 9.5 }}>
              Project
            </div>
            {[['PyPI', 'https://pypi.org/project/hypernix/'],
              ['GitHub', 'https://github.com/trail-b1az3r/hypernix-pip'],
              ['HuggingFace', 'https://huggingface.co/ray0rf1re']].map(([l, h]) => (
              <a key={l} href={h} target="_blank" rel="noreferrer" className="underline-grow"
                style={{ color: 'var(--text-faint)', fontSize: 12.5, textDecoration: 'none',
                  display: 'block', marginBottom: 7, width: 'fit-content' }}>{l} ↗</a>
            ))}
          </div>
          <div>
            <div className="eyebrow" style={{ color: 'var(--text-faint)', marginBottom: 11, fontSize: 9.5 }}>
              Site
            </div>
            {pages.filter(pp => pp !== page).slice(0, 4).map(pp => (
              <button key={pp} onClick={() => setPage(pp)} className="underline-grow"
                style={{ background: 'none', border: 'none', padding: 0, cursor: 'pointer',
                  color: 'var(--text-faint)', fontSize: 12.5, display: 'block', marginBottom: 7,
                  fontFamily: 'inherit', textTransform: 'capitalize' }}>{pp}</button>
            ))}
          </div>
        </div>
      </div>
    </footer>
  )
}
