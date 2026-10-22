// Markdown for the docs: HyperLink's renderer (CommonMark, GFM, Markdown
// Extra; escapes everything it did not make), imported rather than copied,
// plus a small highlighter for the languages the wiki's code blocks use.
import '../../../src/hypernix/hyperlink/web/markdown.js'

declare global {
  interface Window { HyperMarkdown: { render: (md: string) => string, escapeHtml: (s: string) => string } }
}

export function renderMarkdown(md: string): string {
  return window.HyperMarkdown.render(md || '')
}

const esc = (s: string) => s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c] as string)

const KEYWORDS: Record<string, string[]> = {
  python: ['def', 'class', 'return', 'import', 'from', 'as', 'if', 'elif', 'else', 'for', 'while', 'in', 'not', 'and',
    'or', 'with', 'try', 'except', 'finally', 'raise', 'yield', 'lambda', 'None', 'True', 'False', 'pass', 'async', 'await', 'is'],
  js: ['const', 'let', 'var', 'function', 'return', 'import', 'from', 'export', 'if', 'else', 'for', 'while', 'new',
    'class', 'extends', 'await', 'async', 'true', 'false', 'null', 'undefined', 'typeof', 'interface', 'type'],
  bash: ['if', 'then', 'else', 'fi', 'for', 'do', 'done', 'case', 'esac', 'in', 'export', 'function', 'while', 'sudo'],
  swift: ['let', 'var', 'func', 'struct', 'class', 'enum', 'return', 'if', 'else', 'guard', 'import', 'true', 'false', 'nil'],
}
const ALIASES: Record<string, string> = {
  py: 'python', python3: 'python', ts: 'js', tsx: 'js', javascript: 'js', typescript: 'js', jsx: 'js',
  sh: 'bash', shell: 'bash', zsh: 'bash', console: 'bash', fish: 'bash', json: 'json', jsonc: 'json',
  toml: 'ini', ini: 'ini', env: 'ini', dotenv: 'ini', yaml: 'yaml', yml: 'yaml',
}

// One pass with a combined regex: whichever token class matches first at
// a position wins, and everything between matches is escaped plain text.
export function highlight(code: string, langRaw: string): string {
  const lang = ALIASES[langRaw] || langRaw
  const words = KEYWORDS[lang] || []
  const parts: Array<[string, RegExp]> = []
  if (lang === 'python' || lang === 'bash' || lang === 'ini' || lang === 'yaml') parts.push(['comment', /#[^\n]*/])
  if (lang === 'js' || lang === 'swift' || lang === 'json') parts.push(['comment', /\/\/[^\n]*|\/\*[\s\S]*?\*\//])
  parts.push(['string', /"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\])*`/])
  if (lang === 'bash') parts.push(['flag', /(?<=\s)--?[A-Za-z][\w-]*/], ['var', /\$\{?[A-Za-z_][\w]*\}?/])
  if (lang === 'ini' || lang === 'yaml') parts.push(['key', /^[\t ]*[A-Za-z_][\w.-]*(?=\s*[=:])/m])
  if (lang === 'json') parts.push(['key', /"(?:\\.|[^"\\\n])*"(?=\s*:)/])
  parts.push(['number', /\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b/])
  if (words.length) parts.push(['keyword', new RegExp(`\\b(?:${words.join('|')})\\b`)])
  if (lang === 'python') parts.push(['decorator', /@[\w.]+/])
  if (!parts.length) return esc(code)

  // JSON keys must beat strings, so they go first for json.
  if (lang === 'json') parts.sort((a) => (a[0] === 'key' ? -1 : 0))
  const big = new RegExp(parts.map(([, r]) => `(${r.source})`).join('|'), 'g' + (lang === 'ini' || lang === 'yaml' ? 'm' : ''))
  let out = ''
  let last = 0
  for (const m of code.matchAll(big)) {
    const at = m.index ?? 0
    out += esc(code.slice(last, at))
    const which = m.slice(1).findIndex((g) => g !== undefined)
    out += `<span class="tok-${parts[which][0]}">${esc(m[0])}</span>`
    last = at + m[0].length
  }
  return out + esc(code.slice(last))
}

// GitHub-style anchors keep runs of hyphens ("m--models"); the renderer
// collapses them ("m-models"). Compare the collapsed forms.
export const anchorKey = (s: string) => s.toLowerCase().replace(/^user-content-/, '').replace(/-+/g, '-').replace(/^-|-$/g, '')
