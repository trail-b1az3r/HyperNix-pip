const MODEL_CREDITS = [
  { provider: 'Claude Opus', models: ['5.5', '5', '4.7', '4.6', '4.5'] },
  { provider: 'Claude Sonnet', models: ['4.6', '5'] },
  { provider: 'Google Gemini', models: ['3.7 Flash', '3.6 Flash', '3.1 Pro'] },
  { provider: 'Qwen', models: ['Qwen 3 Coder'] },
  { provider: 'OpenAI', models: ['GPT-5.6 Luna', 'GPT-5.5', 'GPT-5.4'] },
  { provider: 'OpenCode', models: ['Nemotron Ultra 3', 'Muse Spark 1.3'] },
]

export function CreditsPage() {
  return (
    <section
      aria-labelledby="credits-heading"
      style={{
        borderTop: '1px solid var(--surface-2)',
        background: 'var(--surface-1)',
        padding: '56px var(--space-gutter) 74px',
      }}
    >
      <div className="shell" style={{ maxWidth: 960 }}>
        <div style={{ maxWidth: 720, marginBottom: 26 }}>
          <div className="eyebrow" style={{ color: 'var(--accent)', marginBottom: 9 }}>
            Credits
          </div>
          <h2
            id="credits-heading"
            style={{
              margin: 0,
              color: 'var(--text)',
              fontSize: 'clamp(25px, 4vw, 36px)',
              lineHeight: 1.05,
              letterSpacing: '-0.035em',
            }}
          >
            Built with help from a lot of models.
          </h2>
          <p style={{ margin: '13px 0 0', color: 'var(--text-dim)', fontSize: 13.5, lineHeight: 1.7 }}>
            Model names are listed here exactly as supplied for the project credits. This credits section intentionally
            appears only at the bottom of the home page.
          </p>
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit,minmax(220px,1fr))',
            gap: 10,
          }}
        >
          {MODEL_CREDITS.map(group => (
            <div
              key={group.provider}
              style={{
                border: '1px solid var(--border-strong)',
                borderRadius: 12,
                background: 'var(--surface-2)',
                padding: '16px 16px 17px',
              }}
            >
              <div
                className="eyebrow"
                style={{ color: 'var(--accent-text)', marginBottom: 10, fontSize: 10.5 }}
              >
                {group.provider}
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
                {group.models.map(model => (
                  <span className="chip" key={model}>{model}</span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

export default CreditsPage
