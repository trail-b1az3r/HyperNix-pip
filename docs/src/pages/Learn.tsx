import { useState, useEffect } from 'react'
import { LEARN_LESSONS } from '../lib/data/learn-lessons'
import { CopyButton } from '../components/ui'

function LearnPage() {
  const stored = () => {
    try { return JSON.parse(sessionStorage.getItem('hnx_llm_cfg') || 'null') } catch { return null }
  }
  const [cfg, setCfg] = useState(stored)        // { url, model, backend, apiKey? }
  const [setupStep, setSetupStep] = useState(0) // 0=pick backend, 1=configure, 2=testing
  const [backend, setBackend] = useState('lmstudio')
  const [customUrl, setCustomUrl] = useState('')
  const [modelName, setModelName] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [testStatus, setTestStatus] = useState(null) // null | 'ok' | 'err' | 'testing'
  const [testErr, setTestErr] = useState('')

  const [lessonIdx, setLessonIdx] = useState(0)
  const [userCode, setUserCode] = useState(LEARN_LESSONS[0].starter)
  const [feedback, setFeedback] = useState(null)
  const [loading, setLoading] = useState(false)
  const [showAnswer, setShowAnswer] = useState(false)
  const [hintIdx, setHintIdx] = useState(0)
  const [chatInput, setChatInput] = useState('')
  const [chatHistory, setChatHistory] = useState([])
  const [chatLoading, setChatLoading] = useState(false)
  const [passed, setPassed] = useState(() => {
    try { return JSON.parse(localStorage.getItem('hnx_passed') || '{}') } catch { return {} }
  })
  const [wrongCounts, setWrongCounts] = useState(() => {
    try { return JSON.parse(localStorage.getItem('hnx_wrong') || '{}') } catch { return {} }
  })

  useEffect(() => {
    try { localStorage.setItem('hnx_passed', JSON.stringify(passed)) } catch {}
  }, [passed])
  useEffect(() => {
    try { localStorage.setItem('hnx_wrong', JSON.stringify(wrongCounts)) } catch {}
  }, [wrongCounts])

  const lesson = LEARN_LESSONS[lessonIdx]

  const BACKENDS = {
    lmstudio:  { label: 'LM Studio',  url: 'http://localhost:1234/v1',  note: 'Start LM Studio → Local Server tab → load any model → Start Server.', kind: 'openai' },
    ollama:    { label: 'Ollama',     url: 'http://localhost:11434/v1', note: 'Run: ollama serve   then: ollama pull mistral   (or any model).', kind: 'openai' },
    custom:    { label: 'Custom',     url: '',                          note: 'Any OpenAI-compatible endpoint, e.g. a remote vLLM or LiteLLM proxy.', kind: 'openai' },
    openai:    { label: 'OpenAI API', url: 'https://api.openai.com/v1', note: 'Paste an OpenAI API key. Default model: gpt-4o-mini.', kind: 'openai', needsKey: true },
    anthropic: { label: 'Anthropic API', url: 'https://api.anthropic.com/v1', note: 'Paste an Anthropic API key. Default model: claude-3-5-haiku-latest.', kind: 'anthropic', needsKey: true },
    none:      { label: 'No Model',   url: '',                          note: 'Self-graded mode: no hints, no AI feedback, no chat. You only see ✓/✗ against the exact solution text.' },
  }
  const isNoModel = backend === 'none'
  const meta = BACKENDS[backend] || {}
  const needsKey = !!meta.needsKey

  const DEFAULT_MODEL = { openai: 'gpt-4o-mini', anthropic: 'claude-3-5-haiku-latest', ollama: 'mistral' }

  const resolvedUrl  = backend === 'custom' ? customUrl.replace(/\/$/, '') : meta.url
  const resolvedModel = modelName.trim() || DEFAULT_MODEL[backend] || ''

  // Builds the request for a given provider "kind" and parses the response back
  // into a plain string. Keeps OpenAI-compatible servers (LM Studio, Ollama,
  // custom, OpenAI) on one code path and Anthropic's distinct message format
  // on another, so a missing/odd field on either side fails loudly instead of
  // silently returning an empty string the UI would render as a blank bubble.
  const callProvider = async ({ url, model, kind, key }, messages, system) => {
    if (kind === 'anthropic') {
      const res = await fetch(`${url}/messages`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': key,
          'anthropic-version': '2023-06-01',
          'anthropic-dangerous-direct-browser-access': 'true',
        },
        body: JSON.stringify({
          model: model || 'claude-3-5-haiku-latest',
          max_tokens: 512,
          system,
          messages: messages.map(m => ({ role: m.role, content: m.content })),
        }),
      })
      if (!res.ok) { const t = await res.text(); throw new Error(`HTTP ${res.status}: ${t.slice(0,160)}`) }
      const data = await res.json()
      const text = (data.content || []).filter(b => b.type === 'text').map(b => b.text).join('')
      if (!text) throw new Error('Model returned an empty response')
      return text
    }

    // OpenAI-compatible: LM Studio, Ollama, custom, OpenAI itself.
    const headers = { 'Content-Type': 'application/json' }
    if (key) headers['Authorization'] = `Bearer ${key}`
    const res = await fetch(`${url}/chat/completions`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        model: model || undefined,
        max_tokens: 512,
        temperature: 0.3,
        messages: [{ role: 'system', content: system }, ...messages],
      }),
    })
    if (!res.ok) { const t = await res.text(); throw new Error(`HTTP ${res.status}: ${t.slice(0,160)}`) }
    const data = await res.json()
    // Most OpenAI-compatible servers use choices[0].message.content, but a
    // few (older Ollama builds, some proxies) return message/content at the
    // top level instead — check both before giving up.
    const text = data.choices?.[0]?.message?.content ?? data.message?.content ?? ''
    if (!text) throw new Error('Model returned an empty response')
    return text
  }

  const testConnection = async () => {
    if (isNoModel) {
      const savedCfg = { url: '', model: '', backend: 'none', kind: '', key: '' }
      sessionStorage.setItem('hnx_llm_cfg', JSON.stringify(savedCfg))
      setCfg(savedCfg)
      setTestStatus('ok')
      return
    }
    if (needsKey && !apiKey.trim()) {
      setTestStatus('err'); setTestErr('An API key is required for this provider.')
      return
    }
    setTestStatus('testing'); setTestErr('')
    try {
      const provider = { url: resolvedUrl, model: resolvedModel, kind: meta.kind, key: apiKey.trim() }
      await callProvider(provider, [{ role: 'user', content: 'Reply with the word OK and nothing else.' }],
        'You are a connection test. Reply with exactly one word.')
      const savedCfg = { url: resolvedUrl, model: resolvedModel, backend, kind: meta.kind, key: needsKey ? apiKey.trim() : '' }
      sessionStorage.setItem('hnx_llm_cfg', JSON.stringify(savedCfg))
      setCfg(savedCfg)
      setTestStatus('ok')
    } catch (e) {
      setTestStatus('err'); setTestErr(e.message)
    }
  }

  const callModel = async (messages, system) => {
    if (cfg?.backend === 'none') throw new Error('No model connected — switch to a backend for AI feedback.')
    return callProvider({ url: cfg.url, model: cfg.model, kind: cfg.kind, key: cfg.key }, messages, system)
  }

  const goLesson = (idx) => {
    setLessonIdx(idx); setUserCode(LEARN_LESSONS[idx].starter)
    setFeedback(null); setShowAnswer(false); setHintIdx(0)
    setChatHistory([]); setChatInput('')
  }

  const wrongCount = wrongCounts[lesson.id] || 0
  const autoUnlocked = wrongCount >= 4

  const normalizeCode = (s) => s.replace(/\s+/g, ' ').trim().toLowerCase()

  const checkAnswer = async () => {
    if (isNoModel) {
      // Self-graded mode: no AI, no hints — exact normalized match against the solution only.
      setLoading(true); setFeedback(null)
      const ok = normalizeCode(userCode) === normalizeCode(lesson.solution)
      if (ok) {
        setFeedback({ ok: true, text: '✓ Matches the reference solution exactly.' })
        setPassed(p => ({ ...p, [lesson.id]: true }))
      } else {
        const newCount = (wrongCounts[lesson.id] || 0) + 1
        setWrongCounts(w => ({ ...w, [lesson.id]: newCount }))
        setFeedback({ ok: false,
          text: '✗ Does not match the reference solution. No-model mode gives no hints — connect LM Studio or Ollama for guided feedback.' })
      }
      setLoading(false)
      return
    }

    setLoading(true); setFeedback(null)
    const system = `You are a concise coding tutor for the HyperNix Python package (an AI/ML toolkit with a cooking-themed API).
Evaluate the student's code for the exercise below. Rules:
- CORRECT: start with "✓ Correct!" then explain in 1-2 sentences why it works.
- WRONG/INCOMPLETE: start with "✗" then explain the specific mistake in 1-2 sentences and give ONE concrete hint. Do NOT give the full answer.
- Max 4 sentences total. Reference actual HyperNix API names (function names, module paths, arg names).`
    const prompt = `Exercise: "${lesson.exercise}"
Expected pattern involves: ${lesson.solution.split('\n').slice(0,4).join(' | ')}
Student's code:\n\`\`\`python\n${userCode}\n\`\`\`\nIs it correct? Give feedback.`
    try {
      const text = await callModel([{ role: 'user', content: prompt }], system)
      const ok = text.trimStart().startsWith('✓')
      if (ok) {
        setFeedback({ ok: true, text })
        setPassed(p => ({ ...p, [lesson.id]: true }))
      } else {
        const newCount = (wrongCounts[lesson.id] || 0) + 1
        setWrongCounts(w => ({ ...w, [lesson.id]: newCount }))
        if (newCount >= 4) {
          setShowAnswer(true)
          setFeedback({ ok: false, autoUnlock: true,
            text: text + '\n\nYou\'ve made 4 attempts — the solution has been unlocked above. Study it and try again!' })
        } else {
          setFeedback({ ok: false, text,
            attemptsLeft: 4 - newCount })
        }
      }
    } catch (e) { setFeedback({ ok: false, text: `Connection error: ${e.message}` }) }
    setLoading(false)
  }

  const sendChat = async () => {
    if (!chatInput.trim()) return
    const userMsg = { role: 'user', content: chatInput.trim() }
    const newHistory = [...chatHistory, userMsg]
    setChatHistory(newHistory); setChatInput(''); setChatLoading(true)
    const system = `You are a helpful, concise tutor for the HyperNix Python package.
The student is on this exercise: "${lesson.exercise}"
Concept: "${lesson.concept}"
Answer directly and briefly (2-4 sentences). Reference specific HyperNix API names.
Never give the full solution — guide them toward it with hints.`
    try {
      const text = await callModel(newHistory, system)
      setChatHistory(h => [...h, { role: 'assistant', content: text }])
    } catch (e) { setChatHistory(h => [...h, { role: 'assistant', content: `Error: ${e.message}` }]) }
    setChatLoading(false)
  }

  const donePct = Math.round((Object.keys(passed).length / LEARN_LESSONS.length) * 100)

  // ── Setup screen ─────────────────────────────────────────────────────────────
  if (!cfg) return (
    <div style={{ minHeight:'100vh', display:'flex', alignItems:'center',
      justifyContent:'center', padding:'24px', background:'var(--bg)' }}>
      <div style={{ maxWidth:480, width:'100%' }}>
        <h1 style={{ fontSize:22, fontWeight:900, color:'var(--text)', textAlign:'center',
          marginBottom:6 }}>Interactive HyperNix Tutor</h1>
        <p style={{ color:'var(--text-dim)', fontSize:13, textAlign:'center', marginBottom:28, lineHeight:1.65 }}>
          Write real HyperNix code, get instant AI feedback — from a local model
          or a cloud API.
        </p>

        <div style={{ background:'var(--surface-1)', border:'1px solid var(--border)', borderRadius:12, padding:'24px' }}>

          {/* Step 0: pick backend */}
          <div style={{ marginBottom:20 }}>
            <div style={{ fontSize:10, color:'var(--text-dim)', textTransform:'uppercase',
              letterSpacing:'0.12em', fontWeight:700, marginBottom:10 }}>Model server</div>
            <div style={{ display:'flex', gap:8, flexWrap:'wrap' }}>
              {Object.entries(BACKENDS).map(([k, b]) => (
                <button key={k} onClick={() => { setBackend(k); setTestStatus(null); setTestErr('') }} style={{
                  flex:'1 1 calc(50% - 4px)', minWidth:100, padding:'9px 6px', borderRadius:7, cursor:'pointer',
                  fontSize:12, fontWeight: backend === k ? 700 : 400,
                  background: backend === k ? (k === 'none' ? '#1a1304' : '#1a0305') : 'var(--surface-3)',
                  border: `1px solid ${backend === k ? (k === 'none' ? '#e8960a' : 'var(--accent)') : 'var(--border-strong)'}`,
                  color: backend === k ? (k === 'none' ? '#e8960a' : 'var(--accent)') : 'var(--text-dim)',
                }}>{b.label}</button>
              ))}
            </div>
            <p style={{ color:'var(--text-faint)', fontSize:12, marginTop:10, lineHeight:1.6 }}>
              {BACKENDS[backend].note}
            </p>
          </div>

          {/* URL override for custom or info display */}
          {!isNoModel && backend === 'custom' && (
            <div style={{ marginBottom:16 }}>
              <label style={{ display:'block', color:'var(--text-dim)', fontSize:11,
                textTransform:'uppercase', letterSpacing:'0.08em', marginBottom:6 }}>
                Base URL
              </label>
              <input value={customUrl} onChange={e => setCustomUrl(e.target.value)}
                placeholder="http://localhost:1234/v1"
                style={{ width:'100%', boxSizing:'border-box', background:'var(--surface-3)',
                  border:'1px solid var(--border-strong)', borderRadius:7, padding:'9px 12px',
                  color:'var(--text)', fontSize:13, outline:'none', fontFamily:'monospace' }} />
            </div>
          )}

          {!isNoModel && backend !== 'custom' && (
            <div style={{ marginBottom:16, padding:'8px 12px', background:'var(--bg)',
              border:'1px solid var(--surface-3)', borderRadius:7 }}>
              <span style={{ color:'var(--text-faint)', fontSize:12, fontFamily:'monospace',
                overflowWrap:'break-word', wordBreak:'break-all' }}>
                {BACKENDS[backend].url}{meta.kind === 'anthropic' ? '/messages' : '/chat/completions'}
              </span>
            </div>
          )}

          {isNoModel && (
            <div style={{ marginBottom:16, padding:'10px 14px', background:'#1a1304',
              border:'1px solid #e8960a33', borderRadius:7 }}>
              <p style={{ color:'#c79238', fontSize:12, margin:0, lineHeight:1.65 }}>
                ⚠ No AI feedback, no hints, no chat help. Your code is checked against the
                exact solution text only — great for self-testing once you already know the API.
              </p>
            </div>
          )}

          {/* API key — only for cloud providers */}
          {needsKey && (
            <div style={{ marginBottom:16 }}>
              <label style={{ display:'block', color:'var(--text-dim)', fontSize:11,
                textTransform:'uppercase', letterSpacing:'0.08em', marginBottom:6 }}>
                API key
              </label>
              <input value={apiKey} onChange={e => setApiKey(e.target.value)}
                type="password" autoComplete="off"
                placeholder={backend === 'openai' ? 'sk-…' : 'sk-ant-…'}
                style={{ width:'100%', boxSizing:'border-box', background:'var(--surface-3)',
                  border:'1px solid var(--border-strong)', borderRadius:7, padding:'9px 12px',
                  color:'var(--text)', fontSize:13, outline:'none', fontFamily:'monospace' }} />
              <p style={{ color:'var(--text-faint)', fontSize:11, marginTop:6, lineHeight:1.5 }}>
                Stored only in this tab's session storage — cleared when you close the tab,
                never sent anywhere but {BACKENDS[backend].url}.
              </p>
            </div>
          )}

          {/* Model name (required for Ollama, optional for LM Studio) */}
          {!isNoModel && (
            <div style={{ marginBottom:20 }}>
              <label style={{ display:'block', color:'var(--text-dim)', fontSize:11,
                textTransform:'uppercase', letterSpacing:'0.08em', marginBottom:6 }}>
                Model name {(backend === 'lmstudio' || needsKey) ? `(optional — defaults to ${DEFAULT_MODEL[backend] || 'loaded model'})` : ''}
              </label>
              <input value={modelName} onChange={e => setModelName(e.target.value)}
                placeholder={backend === 'ollama' ? 'e.g. mistral, llama3.2, qwen2.5:3b' : DEFAULT_MODEL[backend] || 'leave blank to use loaded model'}
                style={{ width:'100%', boxSizing:'border-box', background:'var(--surface-3)',
                  border:'1px solid var(--border-strong)', borderRadius:7, padding:'9px 12px',
                  color:'var(--text)', fontSize:13, outline:'none', fontFamily:'monospace' }} />
            </div>
          )}

          {/* Test + Connect */}
          <button onClick={testConnection} disabled={testStatus === 'testing'
            || (backend === 'custom' && !isNoModel && !customUrl.trim())
            || (needsKey && !apiKey.trim())} style={{
            width:'100%', background: testStatus === 'ok' ? '#0a2a0f' : isNoModel ? '#e8960a' : 'var(--accent)',
            border: `1px solid ${testStatus === 'ok' ? '#34c75966' : 'transparent'}`,
            borderRadius:8, color: testStatus === 'ok' ? '#34c759' : isNoModel ? '#0a0a0a' : '#fff',
            padding:'11px', fontSize:14, fontWeight:700,
            cursor: testStatus === 'testing' ? 'wait' : 'pointer',
          }}>
            {testStatus === 'testing' ? 'Connecting…'
              : testStatus === 'ok' ? '✓ Connected — Start Learning →'
              : isNoModel ? 'Continue Without a Model →'
              : 'Test Connection & Start →'}
          </button>

          {testStatus === 'err' && (
            <div style={{ marginTop:12, padding:'10px 14px', background:'#1a0608',
              border:'1px solid rgba(200,25,46,0.27)', borderRadius:7 }}>
              <p style={{ color:'#e07070', fontSize:12, margin:0, lineHeight:1.6 }}>
                ✗ {testErr}
              </p>
              <p style={{ color:'var(--text-dim)', fontSize:11, margin:'8px 0 0', lineHeight:1.6 }}>
                Make sure your server is running and CORS is enabled.
                {backend === 'lmstudio' && ' In LM Studio: Settings → Server → Enable CORS.'}
                {backend === 'ollama' && ' For Ollama: set OLLAMA_ORIGINS=* before starting.'}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )

  // ── Main tutor UI ─────────────────────────────────────────────────────────────
  return (
    <div style={{ minHeight:'100vh', background:'var(--bg)' }}>

      {/* Progress bar */}
      <div style={{ position:'fixed', top:56, left:0, right:0, height:3,
        background:'var(--surface-3)', zIndex:99 }}>
        <div style={{ height:'100%', background:'var(--accent)',
          width:`${donePct}%`, transition:'width 0.5s cubic-bezier(0.22,1,0.36,1)' }} />
      </div>

      <div style={{ maxWidth:1120, margin:'0 auto', padding:'68px 20px 60px',
        display:'flex', gap:28, minHeight:'100vh' }}>

        {/* Sidebar */}
        <div style={{ width:208, flexShrink:0 }}>
          <div style={{ position:'sticky', top:76 }}>
            <div style={{ fontSize:10, color:'var(--text-faint)', textTransform:'uppercase',
              letterSpacing:'0.12em', marginBottom:10, fontWeight:700 }}>
              {Object.keys(passed).length}/{LEARN_LESSONS.length} done
            </div>
            {LEARN_LESSONS.map((l, i) => (
              <button key={l.id} onClick={() => goLesson(i)} style={{
                display:'flex', alignItems:'flex-start', gap:8, width:'100%', border:'none',
                background: lessonIdx === i ? 'var(--surface-3)' : 'none',
                borderLeft: `2px solid ${lessonIdx === i ? 'var(--accent)'
                  : passed[l.id] ? '#34c75966' : 'transparent'}`,
                padding:'7px 10px 7px 12px', cursor:'pointer', textAlign:'left',
                borderRadius:'0 6px 6px 0', marginBottom:2, transition:'all 0.12s',
              }}>
                <span style={{ fontSize:11, marginTop:1, flexShrink:0, width:14,
                  color: passed[l.id] ? '#34c759' : lessonIdx === i ? 'var(--accent)' : 'var(--text-faint)' }}>
                  {passed[l.id] ? '✓' : lessonIdx === i ? '▸' : '○'}
                </span>
                <div style={{ minWidth:0 }}>
                  <div style={{ fontSize:11, color: lessonIdx === i ? 'var(--text)' : 'var(--text-dim)',
                    overflowWrap:'break-word', wordBreak:'break-word', lineHeight:1.4 }}>
                    {l.title}
                  </div>
                  <div style={{ fontSize:10, color: l.trackColor + '99', marginTop:2 }}>
                    {l.track}
                  </div>
                </div>
              </button>
            ))}
            <button onClick={() => { sessionStorage.removeItem('hnx_llm_cfg'); setCfg(null) }}
              style={{ marginTop:18, background:'none', border:'1px solid var(--surface-3)',
                borderRadius:6, color:'var(--text-faint)', padding:'5px 10px', fontSize:11,
                cursor:'pointer', width:'100%', textAlign:'center' }}>
              ⚙ Change model server
            </button>
            <button onClick={() => {
                if (confirm('Reset all progress? This clears every completed exercise and attempt count.')) {
                  localStorage.removeItem('hnx_passed')
                  localStorage.removeItem('hnx_wrong')
                  setPassed({}); setWrongCounts({})
                }
              }}
              style={{ marginTop:6, background:'none', border:'1px solid var(--surface-3)',
                borderRadius:6, color:'var(--text-faint)', padding:'5px 10px', fontSize:11,
                cursor:'pointer', width:'100%', textAlign:'center' }}>
              ↺ Reset progress
            </button>
          </div>
        </div>

        {/* Main content */}
        <div style={{ flex:1, minWidth:0 }}>

          {/* Lesson header */}
          <div style={{ marginBottom:20 }}>
            <div style={{ display:'inline-flex', alignItems:'center', gap:6,
              background: lesson.trackColor + '14',
              border:`1px solid ${lesson.trackColor}33`,
              borderRadius:4, padding:'3px 10px', marginBottom:10 }}>
              <span style={{ fontSize:11, color: lesson.trackColor,
                fontWeight:700, textTransform:'uppercase', letterSpacing:'0.08em' }}>
                {lesson.track}
              </span>
            </div>
            <h1 style={{ fontSize:26, fontWeight:900, color:'var(--text)',
              letterSpacing:'-0.6px', margin:'0 0 4px' }}>{lesson.title}</h1>
            <div style={{ color:'var(--text-faint)', fontSize:12 }}>
              {lessonIdx + 1} / {LEARN_LESSONS.length}
              {cfg && <span> · {BACKENDS[cfg.backend]?.label || 'Custom'}{cfg.model ? ` · ${cfg.model}` : ''}</span>}
            </div>
          </div>

          {/* Two-column: concept/example | exercise */}
          <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr',
            gap:16, marginBottom:16, alignItems:'start' }}>

            {/* Left — concept + example */}
            <div style={{ display:'flex', flexDirection:'column', gap:12 }}>
              <div style={{ background:'var(--surface-1)', border:'1px solid var(--border)',
                borderRadius:10, padding:'16px 18px' }}>
                <div style={{ fontSize:10, color:'var(--accent)', textTransform:'uppercase',
                  letterSpacing:'0.1em', fontWeight:700, marginBottom:9 }}>Concept</div>
                <p style={{ color:'var(--text-dim)', fontSize:13, lineHeight:1.75, margin:0 }}>
                  {lesson.concept}
                </p>
              </div>

              <div style={{ background:'#0a0a0a', border:'1px solid var(--surface-3)',
                borderRadius:10, overflow:'hidden' }}>
                <div style={{ padding:'9px 14px', borderBottom:'1px solid var(--surface-3)',
                  display:'flex', justifyContent:'space-between', alignItems:'center' }}>
                  <span style={{ fontSize:10, color:'var(--text-faint)', textTransform:'uppercase',
                    letterSpacing:'0.1em', fontWeight:700 }}>Example</span>
                  <CopyButton text={lesson.example} />
                </div>
                <pre style={{ margin:0, padding:'14px 16px', fontSize:12, color:'#9a4a55',
                  fontFamily:'"SF Mono","Fira Code","Consolas",monospace',
                  overflowX:'auto', lineHeight:1.7, whiteSpace:'pre' }}>
                  {lesson.example}
                </pre>
              </div>
            </div>

            {/* Right — exercise + editor */}
            <div style={{ display:'flex', flexDirection:'column', gap:10 }}>
              <div style={{ background:'var(--surface-1)', border:'1px solid var(--border)',
                borderRadius:10, padding:'14px 16px' }}>
                <div style={{ fontSize:10, color:'#4a9eff', textTransform:'uppercase',
                  letterSpacing:'0.1em', fontWeight:700, marginBottom:8 }}>Exercise</div>
                <p style={{ color:'var(--text-muted)', fontSize:13, lineHeight:1.72, margin:0 }}>
                  {lesson.exercise}
                </p>
              </div>

              {/* Code editor */}
              <div style={{ background:'#0a0a0a', border:'1px solid var(--surface-3)',
                borderRadius:10, overflow:'hidden' }}>
                <div style={{ padding:'8px 14px', borderBottom:'1px solid var(--surface-3)',
                  display:'flex', justifyContent:'space-between', alignItems:'center' }}>
                  <span style={{ fontSize:10, color:'var(--text-faint)', textTransform:'uppercase',
                    letterSpacing:'0.1em', fontWeight:700 }}>Your code</span>
                  <button onClick={() => { setUserCode(lesson.starter); setFeedback(null) }}
                    style={{ background:'none', border:'none', color:'var(--text-faint)',
                      fontSize:11, cursor:'pointer' }}>reset</button>
                </div>
                <textarea
                  value={userCode}
                  onChange={e => { setUserCode(e.target.value); setFeedback(null) }}
                  spellCheck={false}
                  rows={Math.max(8, lesson.starter.split('\n').length + 2)}
                  style={{ width:'100%', boxSizing:'border-box', background:'transparent',
                    border:'none', outline:'none', resize:'vertical',
                    padding:'12px 14px', fontSize:12.5, color:'var(--accent)',
                    fontFamily:'"SF Mono","Fira Code","Consolas",monospace',
                    lineHeight:1.7, display:'block', minHeight:140 }}
                />
              </div>

              {/* Buttons */}
              <div style={{ display:'flex', gap:7, flexWrap:'wrap' }}>
                <button onClick={checkAnswer} disabled={loading} className="press-btn" style={{
                  flex:1, background: loading ? '#180204' : 'var(--accent)', border:'none',
                  borderRadius:7, color:'#fff', padding:'9px 16px', fontSize:13,
                  fontWeight:700, cursor: loading ? 'default' : 'pointer',
                  opacity: loading ? 0.65 : 1,
                }}>
                  {loading ? (<><span className="spin-slow" style={{ display:'inline-block', marginRight:6 }}>◌</span>Checking…</>) : '✓ Check My Code'}
                </button>
                {!isNoModel && hintIdx < lesson.hints.length && (
                  <button onClick={() => setHintIdx(h => h + 1)} className="press-btn" style={{
                    background:'var(--surface-3)', border:'1px solid var(--border-strong)', borderRadius:7,
                    color:'var(--text-muted)', padding:'9px 14px', fontSize:13, cursor:'pointer',
                  }}>Hint</button>
                )}
                <button onClick={() => setShowAnswer(s => !s)} className="press-btn" style={{
                  background:'var(--surface-3)', border:'1px solid var(--border-strong)', borderRadius:7,
                  color:'var(--text-faint)', padding:'9px 14px', fontSize:13, cursor:'pointer',
                }}>
                  {showAnswer ? 'Hide' : 'Answer'}
                </button>
              </div>

              {/* Hints (disabled entirely in no-model mode) */}
              {!isNoModel && hintIdx > 0 && lesson.hints.slice(0, hintIdx).map((h, i) => (
                <div key={i} className="anim-fade-up" style={{ background:'#0c1a10', border:'1px solid #1a3020',
                  borderRadius:7, padding:'9px 13px', fontSize:12.5,
                  color:'#4a7a4a', lineHeight:1.65, animationDuration:'0.3s' }}>
                  {h}
                </div>
              ))}

              {/* Answer */}
              {showAnswer && (
                <div className="anim-fade-up" style={{ background:'#0a0a0a', border:'1px solid #222',
                  borderRadius:9, overflow:'hidden', animationDuration:'0.3s' }}>
                  <div style={{ padding:'7px 14px', borderBottom:'1px solid var(--surface-3)',
                    display:'flex', justifyContent:'space-between', alignItems:'center' }}>
                    <span style={{ fontSize:10, color:'var(--text-faint)', textTransform:'uppercase',
                      letterSpacing:'0.1em', fontWeight:700 }}>Solution</span>
                    <button onClick={() => setUserCode(lesson.solution)} className="underline-grow" style={{
                      background:'none', border:'none', color:'var(--accent)',
                      fontSize:11, cursor:'pointer' }}>copy to editor</button>
                  </div>
                  <pre style={{ margin:0, padding:'12px 14px', fontSize:12, color:'var(--text-dim)',
                    fontFamily:'"SF Mono","Fira Code","Consolas",monospace',
                    overflowX:'auto', lineHeight:1.7, whiteSpace:'pre' }}>
                    {lesson.solution}
                  </pre>
                </div>
              )}

              {/* Wrong-attempt counter (only shows once you've gotten it wrong at least once) */}
              {wrongCount > 0 && !passed[lesson.id] && (
                <div className="anim-fade" style={{ display:'flex', gap:5, alignItems:'center', marginTop:-2 }}>
                  <span style={{ fontSize:11, color:'var(--text-dim)' }}>Attempts:</span>
                  {[0,1,2,3].map(i => (
                    <span key={i} style={{ width:7, height:7, borderRadius:'50%',
                      background: i < wrongCount ? (isNoModel ? '#e8960a' : 'var(--accent)') : 'var(--border)',
                      display:'inline-block', transition:'background-color 0.25s ease, transform 0.25s ease',
                      transform: i === wrongCount - 1 ? 'scale(1.3)' : 'scale(1)' }} />
                  ))}
                  <span style={{ fontSize:11, color:'var(--text-faint)', marginLeft:2 }}>
                    {isNoModel
                      ? `${wrongCount} wrong — no auto-unlock in this mode`
                      : wrongCount >= 4 ? '— solution unlocked' : `${4 - wrongCount} until auto-unlock`}
                  </span>
                </div>
              )}

              {/* Feedback */}
              {feedback && (
                <div className="anim-bounce-in" style={{
                  background: feedback.ok ? '#091508' : feedback.autoUnlock ? '#160c02' : '#160408',
                  border: `1px solid ${feedback.ok ? 'rgba(52,199,89,0.25)' : feedback.autoUnlock ? 'rgba(232,150,10,0.25)' : 'rgba(200,25,46,0.25)'}`,
                  borderRadius:9, padding:'13px 15px',
                }}>
                  <p style={{ margin:0, fontSize:13, lineHeight:1.72, whiteSpace:'pre-wrap',
                    color: feedback.ok ? '#5ad47a' : feedback.autoUnlock ? '#e8b04a' : '#d47070' }}>
                    {feedback.text}
                  </p>
                  {feedback.ok && lessonIdx < LEARN_LESSONS.length - 1 && (
                    <button onClick={() => goLesson(lessonIdx + 1)} className="press-btn glow-pulse" style={{
                      marginTop:10, background:'#34c759', border:'none', borderRadius:7,
                      color:'#050f07', padding:'7px 16px', fontSize:13,
                      fontWeight:700, cursor:'pointer',
                    }}>Next →</button>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Ask the model — hidden entirely in no-model mode */}
          {isNoModel ? (
            <div style={{ background:'#0e0e0e', border:'1px solid var(--surface-3)',
              borderRadius:12, padding:'16px 18px', display:'flex', alignItems:'center', gap:8 }}>
              <span style={{ fontSize:12, color:'var(--text-faint)' }}>
                Chat help is disabled in No Model mode.{' '}
                <button onClick={() => { sessionStorage.removeItem('hnx_llm_cfg'); setCfg(null) }}
                  style={{ background:'none', border:'none', color:'#4a9eff', cursor:'pointer',
                    fontSize:12, padding:0, textDecoration:'underline' }}>
                  Connect a model
                </button> for guided help.
              </span>
            </div>
          ) : (
          <div style={{ background:'#0e0e0e', border:'1px solid var(--surface-3)',
            borderRadius:12, overflow:'hidden' }}>
            <div style={{ padding:'11px 16px', borderBottom:'1px solid var(--surface-3)',
              display:'flex', alignItems:'center', gap:8 }}>
              <span style={{ fontSize:12, color:'var(--text-dim)', fontWeight:600 }}>
                Ask your model for help with this exercise
              </span>
            </div>

            {chatHistory.length > 0 && (
              <div style={{ maxHeight:220, overflowY:'auto', padding:'12px 16px',
                display:'flex', flexDirection:'column', gap:8 }}>
                {chatHistory.map((m, i) => (
                  <div key={i} style={{ alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
                    maxWidth:'84%' }}>
                    <div style={{
                      background: m.role === 'user' ? '#1a0305' : 'var(--surface-1)',
                      border: `1px solid ${m.role === 'user' ? 'rgba(200,25,46,0.16)' : 'var(--border)'}`,
                      borderRadius: m.role === 'user' ? '10px 10px 2px 10px' : '10px 10px 10px 2px',
                      padding:'8px 12px', fontSize:13,
                      color: m.role === 'user' ? '#d46070' : 'var(--text-muted)',
                      lineHeight:1.65, whiteSpace:'pre-wrap',
                    }}>{m.content}</div>
                  </div>
                ))}
                {chatLoading && (
                  <div style={{ color:'var(--text-faint)', fontSize:12, padding:'2px 0', alignSelf:'flex-start' }}>
                    Thinking…
                  </div>
                )}
              </div>
            )}

            <div style={{ padding:'10px 14px', display:'flex', gap:8 }}>
              <input value={chatInput} onChange={e => setChatInput(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat() } }}
                placeholder="Ask anything about this exercise…"
                style={{ flex:1, background:'var(--surface-3)', border:'1px solid #222',
                  borderRadius:7, padding:'8px 12px', color:'var(--text)', fontSize:13,
                  outline:'none', fontFamily:'inherit' }}
              />
              <button onClick={sendChat} disabled={chatLoading || !chatInput.trim()} style={{
                background:'var(--accent)', border:'none', borderRadius:7, color:'#fff',
                padding:'8px 14px', fontSize:14, cursor:'pointer',
                opacity: chatLoading || !chatInput.trim() ? 0.4 : 1,
              }}>↑</button>
            </div>
          </div>
          )}

          {/* Prev / Next */}
          <div style={{ display:'flex', justifyContent:'space-between', marginTop:20,
            paddingTop:16, borderTop:'1px solid var(--surface-3)', gap:10 }}>
            {lessonIdx > 0 ? (
              <button onClick={() => goLesson(lessonIdx - 1)} style={{
                background:'var(--surface-3)', border:'1px solid #222', borderRadius:7,
                color:'var(--text-muted)', padding:'8px 14px', fontSize:13, cursor:'pointer',
              }}>← {LEARN_LESSONS[lessonIdx - 1].title}</button>
            ) : <div />}
            {lessonIdx < LEARN_LESSONS.length - 1 && (
              <button onClick={() => goLesson(lessonIdx + 1)} style={{
                background:'var(--surface-3)', border:'1px solid #222', borderRadius:7,
                color:'var(--text-muted)', padding:'8px 14px', fontSize:13, cursor:'pointer',
              }}>{LEARN_LESSONS[lessonIdx + 1].title} →</button>
            )}
          </div>

        </div>
      </div>
    </div>
  )
}

export { LearnPage }
