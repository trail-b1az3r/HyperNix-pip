import { LogoMark } from '../components/Logo'
import { PageHeading } from '../components/ui'
import { ConnectIcon } from '../components/icons'

function AboutPage() {
  return (
    <div style={{ maxWidth:640, margin:'0 auto', padding:'90px 20px 60px' }}>
      <LogoMark theme="dark" size={44} className="anim-fade-up" />
      <div style={{ height:22 }} />
      <PageHeading kicker="Background" title="About HyperNix" />
      <p style={{ color:'var(--text-muted)', lineHeight:1.75, marginBottom:14, fontSize:15 }}>
        I made this project for fun after getting a new PC — even though my GPU is now 10 years old.
        I wanted to train LLMs on it within a reasonable time, but it turns out that takes a while,
        and not having any tensor cores doesn't help.
      </p>
      <p style={{ color:'var(--text-muted)', lineHeight:1.75, marginBottom:14, fontSize:15 }}>
        Not long after, I got access to Claude Code with Opus 4.6 and later v4.7, and built this
        to help accomplish my task better. Someday I'll build a new PC and rewrite this without any
        AI assistance, but first I need to learn more Python.
      </p>
      <p style={{ color:'var(--accent)', lineHeight:1.75, marginBottom:36, fontSize:15, fontWeight:600 }}>
        Once I do, I really hope many people use the package and find it very useful. In version 2.00.X,
        I plan to do a full 100% rewrite — no AI slop, written by me and possibly a few friends.
      </p>
      <p style={{ color:'var(--text-dim)', lineHeight:1.75, marginBottom:40, fontSize:15 }}>Anyway, thanks for using HyperNix.</p>
      <div style={{ paddingTop:22, borderTop:'1px solid var(--surface-3)' }}>
        <div style={{ fontSize:10, color:'var(--text-faint)', textTransform:'uppercase', letterSpacing:'0.12em',
          fontWeight:700, marginBottom:12 }}>Connect</div>
        <div style={{ display:'flex', flexWrap:'wrap', columnGap:18, rowGap:8, marginBottom:22 }}>
          {[
            { label:'GitHub', href:'https://github.com/trail-b1az3r', icon:'github' },
            { label:'Hugging Face', href:'https://huggingface.co/ray0rf1re', icon:'huggingface' },
            { label:'Model Collection', href:'https://huggingface.co/collections/ray0rf1re/hyper-nix-v0x', icon:'models' },
            { label:'PyPI', href:'https://pypi.org/project/hypernix/', icon:'pypi' },
            { label:'Steam', href:'https://steamcommunity.com/id/transgenderfireball/', icon:'steam' },
          ].map(l => (
            <a key={l.label} href={l.href} target="_blank" rel="noreferrer" className="connect-link" style={{
              display:'flex', alignItems:'center', gap:6, textDecoration:'none', fontSize:12.5,
            }}>
              <ConnectIcon name={l.icon} />
              {l.label}
            </a>
          ))}
        </div>
        <p style={{ color:'var(--text-faint)', fontSize:12, lineHeight:1.7 }}>
          Hardware: GTX 1080 (8 GB, Pascal sm_61)<br/>Dual-licensed: LLU-0.1 (source-available, default) or HOS-1.0 (OSI-style open source) — recipient's choice, see LICENSE
        </p>
      </div>
    </div>
  )
}


export { AboutPage }
