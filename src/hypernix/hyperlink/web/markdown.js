// HyperLink on the web: Markdown, as models write it (0.72.8a3).
//
// CommonMark, with GitHub Flavored Markdown (tables, task lists,
// strikethrough, autolinked URLs, footnotes) and PHP Markdown Extra
// (definition lists, abbreviations, footnotes, {#id .class} attributes
// on headings and fenced code). One file, no dependencies: the page's
// content-security policy allows scripts from this origin only, and a
// parser that is ours is a parser whose output we know.
//
// Safe by construction: every character of the source reaches the page
// escaped unless a rule below produced it. Raw HTML is shown as text,
// except a short list of harmless inline tags written with no attributes
// (<kbd>, <sup>, <sub>, <mark>, <ins>, <del>, <s>, <u>, <small>, <br>).
// Links and images keep only http(s), mailto, relative and #anchor
// addresses; an image may also be a data: PNG/JPEG/GIF/WebP.
//
// Half-written Markdown is the normal case while a reply streams in, so
// nothing here throws: whatever cannot be parsed yet is text.

(function (root) {
  "use strict";

  const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ESC[c]);
  const escapeAttr = escapeHtml;
  const PUNCT = /[!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]/;
  const INLINE_TAGS = new Set(["kbd", "sup", "sub", "mark", "ins", "del", "s", "u", "small", "br", "abbr", "q", "cite", "var", "samp"]);

  function normalizeLabel(label) {
    return label.trim().replace(/\s+/g, " ").toLowerCase();
  }

  function safeUrl(url, image) {
    const u = String(url || "").trim();
    if (!u) return "";
    const scheme = /^([a-zA-Z][a-zA-Z0-9+.-]*):/.exec(u);
    if (!scheme) return u;                                     // relative, #anchor
    const s = scheme[1].toLowerCase();
    if (s === "http" || s === "https" || s === "mailto") return u;
    if (image && /^data:image\/(png|jpe?g|gif|webp);base64,[a-z0-9+/=\s]+$/i.test(u)) return u;
    return "";
  }

  function slug(text) {
    return text.toLowerCase().replace(/<[^>]+>/g, "").replace(/&[a-z0-9#]+;/g, "")
      .replace(/[^\p{L}\p{N}\s-]/gu, "").trim().replace(/\s+/g, "-");
  }

  // `{#id .class key=value}` at the end of a line (Markdown Extra).
  function takeAttributes(text) {
    const m = /\s*\{\s*((?:[#.][\w-]+\s*)+)\}\s*$/.exec(text);
    if (!m) return { text, id: "", classes: [] };
    const id = (/#([\w-]+)/.exec(m[1]) || [, ""])[1];
    const classes = [...m[1].matchAll(/\.([\w-]+)/g)].map((x) => x[1]);
    return { text: text.slice(0, m.index), id, classes };
  }

  // -------------------------------------------------------------------
  // Inline
  // -------------------------------------------------------------------

  class Inline {
    constructor(doc) { this.doc = doc; }

    render(text) {
      const nodes = this.parse(text);
      this.emphasis(nodes);
      return nodes.map((n) => (n.t === "delim" ? this.text(n.ch.repeat(n.n)) : n.t === "text" ? this.text(n.v) : n.v)).join("");
    }

    text(s) {
      let out = escapeHtml(s);
      for (const [term, title] of this.doc.abbrs) {
        const re = new RegExp(`(^|[^\\p{L}\\p{N}_])(${term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})(?=$|[^\\p{L}\\p{N}_])`, "gu");
        out = out.replace(re, (_, pre, word) => `${pre}<abbr title="${escapeAttr(title)}">${word}</abbr>`);
      }
      return out;
    }

    parse(src) {
      const nodes = [];
      let buf = "";
      const flush = () => { if (buf) { nodes.push({ t: "text", v: buf }); buf = ""; } };
      const html = (v) => { flush(); nodes.push({ t: "html", v }); };
      let i = 0;
      while (i < src.length) {
        const c = src[i];
        if (c === "\\" && i + 1 < src.length) {
          const n = src[i + 1];
          if (n === "\n") { html("<br>\n"); i += 2; continue; }
          if (PUNCT.test(n)) { buf += n; i += 2; continue; }
        }
        if (c === "`") {
          let run = 0;
          while (src[i + run] === "`") run++;
          const close = this.findBackticks(src, i + run, run);
          if (close >= 0) {
            let code = src.slice(i + run, close).replace(/\n/g, " ");
            if (/^ .*[^ ].* $/.test(code) || /^ [^ ] $/.test(code)) code = code.slice(1, -1);
            html(`<code>${escapeHtml(code)}</code>`);
            i = close + run;
            continue;
          }
          buf += "`".repeat(run); i += run; continue;
        }
        if (c === "<") {
          const auto = /^<([a-zA-Z][a-zA-Z0-9+.-]{1,31}:[^\s<>]*)>/.exec(src.slice(i));
          if (auto) {
            const url = safeUrl(auto[1]);
            if (url) { html(`<a href="${escapeAttr(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(auto[1])}</a>`); i += auto[0].length; continue; }
          }
          const mail = /^<([a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*)>/.exec(src.slice(i));
          if (mail) { html(`<a href="mailto:${escapeAttr(mail[1])}">${escapeHtml(mail[1])}</a>`); i += mail[0].length; continue; }
          const tag = /^<(\/?)([a-zA-Z]+)\s*(\/?)>/.exec(src.slice(i));
          if (tag && INLINE_TAGS.has(tag[2].toLowerCase())) {
            const name = tag[2].toLowerCase();
            html(name === "br" ? "<br>" : `<${tag[1]}${name}>`);
            i += tag[0].length; continue;
          }
          buf += "<"; i += 1; continue;
        }
        if (c === "!" && src[i + 1] === "[") {
          const link = this.link(src, i + 1, true);
          if (link) { html(link.html); i = link.end; continue; }
        }
        if (c === "[") {
          const note = /^\[\^([^\]\s]+)\]/.exec(src.slice(i));
          if (note && this.doc.footnotes.has(normalizeLabel(note[1]))) {
            html(this.doc.footnoteRef(normalizeLabel(note[1])));
            i += note[0].length; continue;
          }
          const link = this.link(src, i, false);
          if (link) { html(link.html); i = link.end; continue; }
        }
        if (c === "*" || c === "_" || c === "~") {
          let run = 0;
          while (src[i + run] === c) run++;
          const before = i === 0 ? " " : src[i - 1];
          const after = i + run >= src.length ? " " : src[i + run];
          const ws = (ch) => /\s/u.test(ch);
          const pu = (ch) => /[\p{P}\p{S}]/u.test(ch);
          const left = !ws(after) && (!pu(after) || ws(before) || pu(before));
          const right = !ws(before) && (!pu(before) || ws(after) || pu(after));
          let open = left, close = right;
          if (c === "_") { open = left && (!right || pu(before)); close = right && (!left || pu(after)); }
          if (c === "~" && run > 2) { buf += c.repeat(run); i += run; continue; }
          flush();
          nodes.push({ t: "delim", ch: c, n: run, orig: run, open, close });
          i += run; continue;
        }
        if (c === "&") {
          const ent = /^&(?:#[xX][0-9a-fA-F]{1,6}|#[0-9]{1,7}|[a-zA-Z][a-zA-Z0-9]{1,31});/.exec(src.slice(i));
          if (ent) { html(ent[0]); i += ent[0].length; continue; }
        }
        if (c === "\n") {
          const hard = / {2,}$/.test(buf);
          buf = buf.replace(/ +$/, "");
          // Soft breaks show as breaks, as in a chat (and GitHub comments).
          html(hard ? "<br>\n" : "<br>\n");
          i += 1;
          while (src[i] === " ") i++;
          continue;
        }
        if ((c === "h" || c === "w") && /(^|[\s*_~(])$/.test(buf.slice(-1) || " ")) {
          const m = /^(?:https?:\/\/|www\.)[^\s<"']*[^\s<"'?!.,:*_~)]/i.exec(src.slice(i));
          if (m) {
            let url = m[0];
            while (url.endsWith(")") && (url.match(/\)/g) || []).length > (url.match(/\(/g) || []).length) url = url.slice(0, -1);
            const ent = /&[a-zA-Z0-9]+;$/.exec(url);
            if (ent) url = url.slice(0, ent.index);
            const href = url.startsWith("www.") ? `http://${url}` : url;
            html(`<a href="${escapeAttr(href)}" target="_blank" rel="noopener noreferrer">${escapeHtml(url)}</a>`);
            i += url.length; continue;
          }
        }
        if (/[a-zA-Z0-9]/.test(c) && /(^|[\s(])$/.test(buf.slice(-1) || " ")) {
          const m = /^[a-zA-Z0-9.+_-]+@[a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_-]+)*\.[a-zA-Z]{2,}(?![\w@])/.exec(src.slice(i));
          if (m && !m[0].endsWith(".")) { html(`<a href="mailto:${escapeAttr(m[0])}">${escapeHtml(m[0])}</a>`); i += m[0].length; continue; }
        }
        buf += c; i += 1;
      }
      flush();
      return nodes;
    }

    findBackticks(src, from, run) {
      let j = from;
      while (j < src.length) {
        if (src[j] === "`") {
          let k = 0;
          while (src[j + k] === "`") k++;
          if (k === run) return j;
          j += k;
        } else j++;
      }
      return -1;
    }

    // [text](url "title"), [text][ref], [text][], [text]; with image, ![...].
    link(src, at, image) {
      let depth = 0, j = at;
      for (; j < src.length; j++) {
        const ch = src[j];
        if (ch === "\\") { j++; continue; }
        if (ch === "`") { const r = /^`+/.exec(src.slice(j))[0].length; const e = this.findBackticks(src, j + r, r); if (e >= 0) { j = e + r - 1; continue; } }
        if (ch === "[") depth++;
        else if (ch === "]") { depth--; if (depth === 0) break; }
      }
      if (depth !== 0 || j >= src.length) return null;
      const inner = src.slice(at + 1, j);
      let end = j + 1, url = null, title = "";
      const rest = src.slice(end);
      const inline = /^\(\s*(<[^<>\n]*>|[^\s()<>]*(?:\([^\s()]*\)[^\s()<>]*)*)(?:\s+("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\((?:[^()\\]|\\.)*\)))?\s*\)/.exec(rest);
      if (inline) {
        url = inline[1].replace(/^<|>$/g, "");
        title = inline[2] ? inline[2].slice(1, -1) : "";
        end += inline[0].length;
      } else {
        const ref = /^\[([^\]]*)\]/.exec(rest);
        const label = normalizeLabel(ref && ref[1] ? ref[1] : inner);
        const def = this.doc.refs.get(label);
        if (!def) return null;
        url = def.url; title = def.title;
        if (ref) end += ref[0].length;
      }
      url = url.replace(/\\([!-/:-@[-`{-~])/g, "$1");
      const safe = safeUrl(url, image);
      const t = title ? ` title="${escapeAttr(title)}"` : "";
      if (image) {
        const alt = inner.replace(/[*_`~[\]]/g, "");
        if (!safe) return { html: escapeHtml(alt), end };
        return { html: `<img src="${escapeAttr(safe)}" alt="${escapeAttr(alt)}"${t} loading="lazy">`, end };
      }
      const label = new Inline(this.doc).render(inner);
      if (!safe) return { html: label, end };
      const external = /^(https?:|mailto:)/i.test(safe);
      return { html: `<a href="${escapeAttr(safe)}"${t}${external ? ' target="_blank" rel="noopener noreferrer"' : ""}>${label}</a>`, end };
    }

    // CommonMark's delimiter algorithm, with ~ and ~~ for strikethrough.
    emphasis(nodes) {
      for (let closer = 0; closer < nodes.length; closer++) {
        const c = nodes[closer];
        if (c.t !== "delim" || !c.close || c.n === 0) continue;
        for (let opener = closer - 1; opener >= 0; opener--) {
          const o = nodes[opener];
          if (o.t !== "delim" || o.ch !== c.ch || !o.open || o.n === 0) continue;
          if (c.ch === "~") {
            if (o.n !== c.n) continue;
            const inner = this.flatten(nodes.slice(opener + 1, closer));
            nodes.splice(opener, closer - opener + 1, { t: "html", v: `<del>${inner}</del>` });
            closer = opener;
            break;
          }
          if ((o.close || c.open) && (o.orig + c.orig) % 3 === 0 && !(o.orig % 3 === 0 && c.orig % 3 === 0)) continue;
          const use = o.n >= 2 && c.n >= 2 ? 2 : 1;
          const tag = use === 2 ? "strong" : "em";
          const inner = this.flatten(nodes.slice(opener + 1, closer));
          o.n -= use; c.n -= use;
          const middle = { t: "html", v: `<${tag}>${inner}</${tag}>` };
          nodes.splice(opener + 1, closer - opener - 1, middle);
          closer = opener + 2;
          if (o.n === 0) { nodes.splice(opener, 1); closer--; }
          if (c.n === 0) { nodes.splice(closer, 1); }
          closer -= 1;
          break;
        }
      }
    }

    flatten(nodes) {
      return nodes.map((n) => (n.t === "delim" ? this.text(n.ch.repeat(n.n)) : n.t === "text" ? this.text(n.v) : n.v)).join("");
    }
  }

  // -------------------------------------------------------------------
  // Blocks
  // -------------------------------------------------------------------

  const RE = {
    blank: /^[ \t]*$/,
    hr: /^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$/,
    atx: /^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$/,
    fence: /^( {0,3})(`{3,}|~{3,})[ \t]*([^`]*?)[ \t]*$/,
    quote: /^ {0,3}> ?/,
    bullet: /^( {0,3})([-+*])( {1,4}|\t|$)/,
    ordered: /^( {0,3})(\d{1,9})([.)])( {1,4}|\t|$)/,
    setext: /^ {0,3}(=+|-+)[ \t]*$/,
    tableDelim: /^ {0,3}\|?(?:[ \t]*:?-+:?[ \t]*\|)+(?:[ \t]*:?-+:?[ \t]*)?\|?[ \t]*$|^ {0,3}\|?[ \t]*:?-+:?[ \t]*\|?[ \t]*$/,
    def: /^ {0,3}:[ \t]+/,
    html: /^ {0,3}<\/?(?:div|p|table|pre|section|details|summary|script|style|iframe|img|br|hr|h[1-6]|ul|ol|li|blockquote)\b/i,
  };

  function expandTabs(line) {
    let out = "";
    for (const ch of line) out += ch === "\t" ? " ".repeat(4 - (out.length % 4)) : ch;
    return out;
  }

  function indentOf(line) { return /^ */.exec(line)[0].length; }

  function startsBlock(line) {
    return RE.hr.test(line) || RE.atx.test(line) || RE.fence.test(line) || RE.quote.test(line)
      || (RE.bullet.test(line) && !RE.blank.test(line.replace(RE.bullet, "")))
      || (RE.ordered.test(line) && /^ {0,3}1[.)]/.test(line) && !RE.blank.test(line.replace(RE.ordered, "")));
  }

  function splitRow(line) {
    let s = line.trim();
    if (s.startsWith("|")) s = s.slice(1);
    if (s.endsWith("|") && !s.endsWith("\\|")) s = s.slice(0, -1);
    const cells = [];
    let cur = "", code = false;
    for (let i = 0; i < s.length; i++) {
      const ch = s[i];
      if (ch === "\\" && s[i + 1] === "|") { cur += "|"; i++; continue; }
      if (ch === "`") code = !code;
      if (ch === "|" && !code) { cells.push(cur.trim()); cur = ""; continue; }
      cur += ch;
    }
    cells.push(cur.trim());
    return cells;
  }

  class Document {
    constructor() {
      this.refs = new Map();
      this.footnotes = new Map();
      this.abbrs = [];
      this.noteOrder = [];
      this.inline = new Inline(this);
    }

    footnoteRef(label) {
      let n = this.noteOrder.indexOf(label) + 1;
      if (!n) { this.noteOrder.push(label); n = this.noteOrder.length; }
      return `<sup class="fnref"><a href="#fn-${escapeAttr(label)}" id="fnref-${escapeAttr(label)}">${n}</a></sup>`;
    }

    // Link references, footnotes and abbreviations, wherever they are
    // (outside code), taken out before the blocks are read.
    collect(lines) {
      const out = [];
      let fence = null;
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const f = RE.fence.exec(line);
        if (fence) { if (f && f[2][0] === fence[0] && f[2].length >= fence.length && !f[3]) fence = null; out.push(line); continue; }
        if (f) { fence = f[2]; out.push(line); continue; }
        const ref = /^ {0,3}\[([^\]^][^\]]*)\]:[ \t]*<?([^\s>]+)>?(?:[ \t]+("[^"]*"|'[^']*'|\([^)]*\)))?[ \t]*$/.exec(line);
        if (ref) {
          const label = normalizeLabel(ref[1]);
          if (!this.refs.has(label)) this.refs.set(label, { url: ref[2], title: ref[3] ? ref[3].slice(1, -1) : "" });
          continue;
        }
        const note = /^ {0,3}\[\^([^\]\s]+)\]:[ \t]?(.*)$/.exec(line);
        if (note) {
          const body = [note[2]];
          while (i + 1 < lines.length && (RE.blank.test(lines[i + 1]) ? (i + 2 < lines.length && indentOf(expandTabs(lines[i + 2])) >= 4) : indentOf(expandTabs(lines[i + 1])) >= 4)) {
            i++;
            body.push(expandTabs(lines[i]).replace(/^ {1,4}/, ""));
          }
          this.footnotes.set(normalizeLabel(note[1]), body.join("\n"));
          continue;
        }
        const abbr = /^ {0,3}\*\[([^\]]+)\]:[ \t]*(.*)$/.exec(line);
        if (abbr) { this.abbrs.push([abbr[1].trim(), abbr[2].trim()]); continue; }
        out.push(line);
      }
      this.abbrs.sort((a, b) => b[0].length - a[0].length);
      return out;
    }

    blocks(lines) {
      let html = "";
      let i = 0;
      while (i < lines.length) {
        const line = expandTabs(lines[i]);
        if (RE.blank.test(line)) { i++; continue; }

        let m = RE.fence.exec(line);
        if (m && !(m[2][0] === "`" && m[3].includes("`"))) {
          const indent = m[1].length, mark = m[2];
          const body = [];
          i++;
          while (i < lines.length) {
            const l = expandTabs(lines[i]);
            const close = /^( {0,3})(`{3,}|~{3,})[ \t]*$/.exec(l);
            if (close && close[2][0] === mark[0] && close[2].length >= mark.length) { i++; break; }
            body.push(l.replace(new RegExp(`^ {0,${indent}}`), ""));
            i++;
          }
          const info = takeAttributes(m[3].replace(/^\{\s*\.?/, (s) => (s.includes("{") ? "{." : s)));
          let lang = info.text.trim().split(/\s+/)[0] || info.classes[0] || "";
          lang = lang.replace(/^\./, "").replace(/[^\w+#.-]/g, "");
          const cls = lang ? ` class="language-${escapeAttr(lang)}"` : "";
          html += `<pre><code${cls}>${escapeHtml(body.join("\n"))}${body.length ? "\n" : ""}</code></pre>`;
          continue;
        }

        if (indentOf(line) >= 4) {
          const body = [];
          while (i < lines.length && (indentOf(expandTabs(lines[i])) >= 4 || RE.blank.test(lines[i]))) {
            body.push(expandTabs(lines[i]).slice(4));
            i++;
          }
          while (body.length && RE.blank.test(body[body.length - 1])) body.pop();
          html += `<pre><code>${escapeHtml(body.join("\n"))}\n</code></pre>`;
          continue;
        }

        m = RE.atx.exec(line);
        if (m) {
          const attrs = takeAttributes(m[2] || "");
          html += this.heading(m[1].length, attrs);
          i++; continue;
        }

        if (RE.hr.test(line)) { html += "<hr>"; i++; continue; }

        if (RE.quote.test(line)) {
          const body = [];
          while (i < lines.length) {
            const l = expandTabs(lines[i]);
            if (RE.quote.test(l)) body.push(l.replace(RE.quote, ""));
            else if (!RE.blank.test(l) && body.length && !RE.blank.test(body[body.length - 1]) && !startsBlock(l)) body.push(l);
            else break;
            i++;
          }
          html += `<blockquote>${this.blocks(body)}</blockquote>`;
          continue;
        }

        if (RE.bullet.test(line) || RE.ordered.test(line)) {
          const out = this.list(lines, i);
          html += out.html; i = out.end; continue;
        }

        if (line.includes("|") && i + 1 < lines.length && RE.tableDelim.test(expandTabs(lines[i + 1]))) {
          const header = splitRow(line);
          const align = splitRow(expandTabs(lines[i + 1])).map((c) => (c.startsWith(":") && c.endsWith(":") ? "center" : c.endsWith(":") ? "right" : c.startsWith(":") ? "left" : ""));
          if (header.length === align.length) {
            const cell = (tag, text, k) => `<${tag}${align[k] ? ` style="text-align:${align[k]}"` : ""}>${this.inline.render(text || "")}</${tag}>`;
            let t = `<div class="table-wrap"><table><thead><tr>${header.map((h, k) => cell("th", h, k)).join("")}</tr></thead>`;
            i += 2;
            const rows = [];
            while (i < lines.length && !RE.blank.test(lines[i]) && !startsBlock(expandTabs(lines[i]))) {
              const cells = splitRow(expandTabs(lines[i]));
              rows.push(`<tr>${header.map((_, k) => cell("td", cells[k], k)).join("")}</tr>`);
              i++;
            }
            t += rows.length ? `<tbody>${rows.join("")}</tbody>` : "";
            html += `${t}</table></div>`;
            continue;
          }
        }

        if (RE.html.test(line)) {
          // Raw HTML blocks are shown, not run.
          const body = [];
          while (i < lines.length && !RE.blank.test(lines[i])) { body.push(lines[i]); i++; }
          html += `<p>${escapeHtml(body.join("\n")).replace(/\n/g, "<br>\n")}</p>`;
          continue;
        }

        // A paragraph: until a blank line or the start of another block.
        const para = [line];
        i++;
        let setext = 0;
        while (i < lines.length) {
          const l = expandTabs(lines[i]);
          if (RE.blank.test(l)) break;
          const s = RE.setext.exec(l);
          if (s && !RE.bullet.test(l.trim() + " x")) { setext = s[1][0] === "=" ? 1 : 2; i++; break; }
          if (s && s[1][0] === "-" && /^ {0,3}-+[ \t]*$/.test(l)) { setext = 2; i++; break; }
          if (startsBlock(l) || RE.def.test(l) || (l.includes("|") && i + 1 < lines.length && RE.tableDelim.test(expandTabs(lines[i + 1])))) break;
          para.push(l);
          i++;
        }
        if (setext) { html += this.heading(setext, takeAttributes(para.map((p) => p.trim()).join("\n"))); continue; }
        if (i < lines.length && RE.def.test(expandTabs(lines[i]))) {
          const out = this.definitions(para, lines, i);
          html += out.html; i = out.end; continue;
        }
        html += `<p>${this.inline.render(para.map((p) => p.replace(/^ {0,3}/, "")).join("\n").trim())}</p>`;
      }
      return html;
    }

    heading(level, attrs) {
      const inner = this.inline.render(attrs.text.trim());
      const id = attrs.id || slug(inner);
      const cls = attrs.classes.length ? ` class="${escapeAttr(attrs.classes.join(" "))}"` : "";
      return `<h${level}${id ? ` id="user-content-${escapeAttr(id)}"` : ""}${cls}>${inner}</h${level}>`;
    }

    definitions(terms, lines, i) {
      let html = `<dl>${terms.map((t) => `<dt>${this.inline.render(t.trim())}</dt>`).join("")}`;
      while (i < lines.length) {
        const l = expandTabs(lines[i]);
        if (RE.def.test(l)) {
          const body = [l.replace(RE.def, "")];
          i++;
          while (i < lines.length && !RE.blank.test(lines[i]) && !RE.def.test(expandTabs(lines[i]))) {
            body.push(expandTabs(lines[i]).replace(/^ {1,4}/, "")); i++;
          }
          html += `<dd>${this.inline.render(body.join("\n").trim())}</dd>`;
          continue;
        }
        if (RE.blank.test(l) && i + 1 < lines.length && RE.def.test(expandTabs(lines[i + 1]))) { i++; continue; }
        // A blank line, then another term with its definitions: the same list.
        if (RE.blank.test(l) && i + 2 < lines.length && !RE.blank.test(lines[i + 1])
            && !startsBlock(expandTabs(lines[i + 1])) && RE.def.test(expandTabs(lines[i + 2]))) { i++; continue; }
        // Another term, followed by its definitions.
        if (!RE.blank.test(l) && i + 1 < lines.length && RE.def.test(expandTabs(lines[i + 1])) && !startsBlock(l)) {
          html += `<dt>${this.inline.render(l.trim())}</dt>`; i++; continue;
        }
        break;
      }
      return { html: `${html}</dl>`, end: i };
    }

    list(lines, start) {
      const first = expandTabs(lines[start]);
      const ordered = !RE.bullet.test(first);
      const mark = ordered ? RE.ordered.exec(first)[3] : RE.bullet.exec(first)[2];
      const startNum = ordered ? parseInt(RE.ordered.exec(first)[2], 10) : 1;
      const items = [];
      let i = start, loose = false, sawBlank = false;
      while (i < lines.length) {
        const line = expandTabs(lines[i]);
        const m = ordered ? RE.ordered.exec(line) : RE.bullet.exec(line);
        if (!m || (ordered ? m[3] !== mark : m[2] !== mark) || RE.hr.test(line)) break;
        if (sawBlank) loose = true;
        // Where the item's content starts: after the marker and its
        // spaces (one, if there are more than four: the rest is code).
        const markerEnd = m[1].length + (ordered ? m[2].length + 1 : 1);
        const spaces = m[m.length - 1];
        const pad = spaces === "" || spaces.length > 4 ? 1 : spaces.length;
        const contentIndent = markerEnd + pad;
        const body = [line.length > markerEnd ? line.slice(markerEnd + Math.min(pad, spaces.length)) : ""];
        i++;
        sawBlank = false;
        let blankInside = false;
        while (i < lines.length) {
          const l = expandTabs(lines[i]);
          if (RE.blank.test(l)) { body.push(""); sawBlank = true; i++; continue; }
          if (indentOf(l) >= contentIndent) {
            if (sawBlank) blankInside = true;
            body.push(l.slice(contentIndent)); sawBlank = false; i++; continue;
          }
          if (!sawBlank && !startsBlock(l) && !RE.bullet.test(l) && !RE.ordered.test(l)) { body.push(l.trim()); i++; continue; }
          break;
        }
        while (body.length && body[body.length - 1] === "") body.pop();
        if (blankInside && body.some((b, k) => b === "" && k > 0 && k < body.length - 1)) loose = true;
        items.push(body);
      }
      const rendered = items.map((body) => {
        let text = body.join("\n");
        let check = "";
        const task = /^\[([ xX])\][ \t]+/.exec(text);
        if (task) {
          check = `<input type="checkbox" disabled${task[1] !== " " ? " checked" : ""}> `;
          text = text.slice(task[0].length);
        }
        let inner = this.blocks(text.split("\n"));
        if (!loose) inner = inner.replace(/^<p>([\s\S]*?)<\/p>/, "$1").replace(/<p>([\s\S]*?)<\/p>/g, "$1");
        return `<li${check ? ' class="task"' : ""}>${check}${inner}</li>`;
      }).join("");
      const tag = ordered ? "ol" : "ul";
      const startAttr = ordered && startNum !== 1 ? ` start="${startNum}"` : "";
      const cls = items.some((b) => /^\[[ xX]\][ \t]/.test(b[0])) ? ' class="contains-task-list"' : "";
      return { html: `<${tag}${startAttr}${cls}>${rendered}</${tag}>`, end: i };
    }

    footnotesHtml() {
      if (!this.noteOrder.length) return "";
      const items = [];
      for (let k = 0; k < this.noteOrder.length; k++) {
        const label = this.noteOrder[k];
        const body = this.blocks((this.footnotes.get(label) || "").split("\n"));
        const back = ` <a href="#fnref-${escapeAttr(label)}" class="fnback">&#8617;</a>`;
        items.push(`<li id="fn-${escapeAttr(label)}">${body.replace(/<\/p>$/, `${back}</p>`) || back}</li>`);
      }
      return `<section class="footnotes"><hr><ol>${items.join("")}</ol></section>`;
    }
  }

  /** Markdown to safe HTML. */
  function render(source) {
    const doc = new Document();
    const lines = doc.collect(String(source || "").replace(/\r\n?/g, "\n").replace(/\u0000/g, "�").split("\n"));
    return doc.blocks(lines) + doc.footnotesHtml();
  }

  const api = { render, escapeHtml, safeUrl };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.HyperMarkdown = api;
})(typeof window !== "undefined" ? window : globalThis);
