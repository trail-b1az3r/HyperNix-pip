// HyperLink on the web. The same server routes the app uses, from the
// same origin, with no build step and no dependencies.
//
// Everything a reply or a memory says is text from a model, so it goes
// through textContent or through render(), which escapes first and adds
// only a handful of known-safe tags. Nothing is ever assigned to
// innerHTML unescaped.
"use strict";

const TOKEN_KEY = "hyperlink.web.token";
const state = {
  token: "",
  sessions: [],
  loaded: [],
  signer: "",
  serverSessionId: "",
  sessionId: "",
  messages: [],
  models: [],
  modelId: "",
  generationId: "",
  streaming: null,
  view: "chats",
  // Files attached to the next message, each {file, pick}: pick is the
  // part of a dataset chosen for the model, or null for the default (0.72.8a3).
  pending: [],
  // The /code sandbox the model works in for this chat, or "".
  codeSandboxId: "",
};

const $ = (id) => document.getElementById(id);

function storedToken() {
  try { return localStorage.getItem(TOKEN_KEY) || ""; } catch { return ""; }
}
function storeToken(token) {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch { /* private window: signed in for this tab only */ }
}

class ApiError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}

async function api(path, { method = "GET", body, signal } = {}) {
  const headers = { Accept: "application/json" };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  const response = await fetch(path, {
    method, headers, signal, body: body === undefined ? undefined : JSON.stringify(body),
  });
  let data = null;
  try { data = await response.json(); } catch { data = null; }
  if (!response.ok) {
    const message = (data && data.error && data.error.message) || `HTTP ${response.status}`;
    throw new ApiError(response.status, message);
  }
  return data;
}

// ---------------------------------------------------------------------------
// The ping before a prompt, and signed prompts (0.72.8)
// ---------------------------------------------------------------------------
//
// When Send is pressed: a ping, answered without a model, that must come
// back within four seconds -- otherwise the server is not answering and
// we say so, rather than waiting on a prompt that may never return. Then
// the prompt, signed (hypernix/hyperlink/signing.py) with who sent it,
// when Send was pressed, a fresh 23-26 character nonce, the chat and the
// server session. Signing needs the browser's crypto, which only a
// secure page (https, or localhost) has; elsewhere prompts go unsigned,
// which a server accepts unless it requires signatures.

const SIGN_VERSION = "hyperlink-sign/v1";
const PING_TIMEOUT_MS = 4000;
const NONCE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

function canSign() {
  return Boolean(state.token && window.crypto && window.crypto.subtle);
}

function newNonce(length = 24) {
  const bytes = new Uint8Array(length);
  window.crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => NONCE_CHARS[b % NONCE_CHARS.length]).join("");
}

const hex = (buffer) => Array.from(new Uint8Array(buffer), (b) => b.toString(16).padStart(2, "0")).join("");

async function sha256Hex(text) {
  return hex(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)));
}

async function signature({ action, chatSessionId = "", content = "", serverSessionId = "", signedAt, signer }) {
  const fields = {
    signer, signed_at: signedAt, nonce: newNonce(24 + Math.floor(Math.random() * 3)),
    chat_session_id: chatSessionId, server_session_id: serverSessionId, action,
    content_sha256: content ? await sha256Hex(content) : "",
  };
  const canonical = [SIGN_VERSION, fields.signer, String(fields.signed_at), fields.nonce,
    fields.chat_session_id, fields.server_session_id, fields.action, fields.content_sha256].join("\n");
  const keyBytes = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(`${SIGN_VERSION}:${state.token}`));
  const key = await crypto.subtle.importKey("raw", keyBytes, { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  fields.signature = hex(await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(canonical)));
  return fields;
}

// The ping, stamped with when Send was pressed. Resolves to the server
// session and the signer, or throws "not answering" after four seconds.
async function pingBeforeSend(chatSessionId, sentAt) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), PING_TIMEOUT_MS);
  try {
    if (!state.signer) state.signer = await learnSigner(controller.signal);
    const sig = await signature({ action: "ping", chatSessionId, signedAt: Math.floor(sentAt),
      signer: state.signer, serverSessionId: "" });
    const answer = await api("/hyperlink/ping", { method: "POST", signal: controller.signal,
      body: { sent_at: sentAt, chat_session_id: chatSessionId, signature: sig } });
    state.signer = answer.signer;
    state.serverSessionId = answer.server_session_id;
    return answer;
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error("The server did not answer within 4 seconds. Check the connection and send again.");
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

// Who this credential signs as: key:<key id> or account:<id>.
async function learnSigner(signal) {
  return (await api("/hyperlink/whoami", { signal })).signer || "";
}

// ---------------------------------------------------------------------------
// Rendering text safely
// ---------------------------------------------------------------------------

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

/**
 * Model text as HTML: CommonMark, GFM and Markdown Extra, escaped first
 * (web/markdown.js, loaded before this file). Every model string reaches
 * innerHTML through here.
 */
function render(text) {
  return window.HyperMarkdown.render(text || "");
}

function shortModel(id) {
  return (id || "").split("/").pop();
}

// ---------------------------------------------------------------------------
// Signing in
// ---------------------------------------------------------------------------

async function signedIn() {
  try {
    await api("/hyperlink/sessions?limit=1");
    return true;
  } catch (error) {
    if (error.status === 401 || error.status === 403) return false;
    throw error;
  }
}

async function signIn(credential) {
  const value = credential.trim();
  if (!value) return;
  // A pairing code is short and made of letters and digits; anything
  // else is taken to be a key or a device token.
  if (/^[A-Za-z0-9-]{4,16}$/.test(value)) {
    const reply = await api("/hyperlink/pair/redeem", {
      method: "POST",
      body: { code: value, device_name: `Browser (${location.hostname})`,
              platform: "web", app_version: "web" },
    });
    state.token = reply.device_token;
  } else {
    state.token = value;
  }
  if (!(await signedIn())) {
    state.token = "";
    throw new Error("That was not accepted by this server.");
  }
  storeToken(state.token);
}

function showSignIn(show) {
  $("sign-in").hidden = !show;
  $("app").hidden = show;
}

// ---------------------------------------------------------------------------
// Chats
// ---------------------------------------------------------------------------

async function loadSessions() {
  const data = await api("/hyperlink/sessions?limit=200");
  state.sessions = data.sessions || [];
  drawSessions();
}

function drawSessions() {
  const list = $("sessions");
  const needle = $("search").value.trim().toLowerCase();
  list.replaceChildren();
  for (const session of state.sessions) {
    if (needle && !(session.title || "").toLowerCase().includes(needle)) continue;
    const item = document.createElement("li");
    item.className = session.session_id === state.sessionId ? "active" : "";
    item.textContent = session.title || "New chat";
    const small = document.createElement("small");
    small.textContent = shortModel(session.model_id) || new Date(session.updated_at * 1000).toLocaleString();
    item.append(small);
    item.addEventListener("click", () => openSession(session.session_id));
    list.append(item);
  }
}

// ---------------------------------------------------------------------------
// The two-layer ring (0.72.8)
// ---------------------------------------------------------------------------
//
// Inner: how much of the model's context the next turn uses. Outer: the
// person's usage allowance when their key and the server have one, else
// GPU load, else CPU load. The server does the arithmetic; this draws.

function drawRing(data) {
  const ring = $("ring");
  if (!ring) return;
  const put = (id, layer) => {
    const circle = $(id);
    const percent = Math.round(Math.min(1, Math.max(0, (layer && layer.fraction) || 0)) * 100);
    circle.style.strokeDasharray = `${percent} 100`;
    circle.classList.toggle("warn", percent >= 90);
    if (id === "ring-outer") {
      circle.classList.toggle("gpu", !!layer && layer.kind === "gpu");
      circle.classList.toggle("cpu", !!layer && layer.kind === "cpu");
    }
  };
  put("ring-inner", data && data.inner);
  put("ring-outer", data && data.outer);
  const label = data ? `Context: ${data.inner.label}\n${data.outer.kind === "usage" ? "Usage" : "Load"}: ${data.outer.label}` : "";
  ring.title = label;
  ring.setAttribute("aria-label", label.replace("\n", "; "));
}

async function refreshRing() {
  if (!state.sessionId) { drawRing(null); return; }
  try {
    drawRing(await api(`/hyperlink/sessions/${encodeURIComponent(state.sessionId)}/ring`));
  } catch { /* an older server has no ring; leave it empty */ }
}

setInterval(() => { if (!document.hidden && !state.streaming) refreshRing(); }, 5000);

async function openSession(sessionId) {
  state.sessionId = sessionId;
  const session = state.sessions.find((s) => s.session_id === sessionId);
  $("chat-title").textContent = (session && session.title) || "New chat";
  if (session && session.model_id) {
    state.modelId = session.model_id;
    $("model-picker").value = session.model_id;
  }
  const data = await api(`/hyperlink/sessions/${encodeURIComponent(sessionId)}/messages`);
  state.messages = data.messages || [];
  const lastMine = [...state.messages].reverse().find((m) => m.role === "user");
  state.codeSandboxId = (lastMine && lastMine.metadata && lastMine.metadata.code_sandbox_id) || "";
  drawThread();
  drawTray();
  drawSessions();
  refreshRing();
  $("sidebar").classList.remove("open");
  switchView("chats");
}

function newChat({ keepSandbox = false } = {}) {
  state.sessionId = "";
  state.messages = [];
  if (!keepSandbox) state.codeSandboxId = "";
  drawTray();
  drawRing(null);
  $("chat-title").textContent = "New chat";
  drawThread();
  drawSessions();
  $("message").focus();
}

function drawThread() {
  const thread = $("thread");
  thread.replaceChildren();
  // A reply that failed before a word arrived is saved empty; it is shown
  // as its error, never as a blank bubble.
  const shown = state.messages.filter((m) => m.role === "user" || (m.role === "assistant"
    && (m.content || failure(m) || ((m.metadata && m.metadata.tool_rounds) || []).length)));
  shown.forEach((message, index) => {
    const next = shown[index + 1];
    const tail = !next || next.role !== message.role;
    const row = message.content || message.role === "user"
      ? bubble(message.role === "user" ? "me" : "them", message.content || attachedNote(message), {
        tail,
        meta: message.role === "assistant" && tail ? shortModel(message.model_id) : "",
        tools: (message.metadata && message.metadata.tool_rounds) || [],
      })
      : errorRow("");
    if (message.role === "assistant" && failure(message)) row.append(errorNote(failure(message)));
    for (const doc of (message.metadata && message.metadata.documents) || []) row.append(documentChip(doc));
    if (message.role === "user") {
      const picks = (message.metadata && message.metadata.datasets) || [];
      if (picks.length) row.append(infoChip(`▤ ${picks.length === 1 ? "part of a dataset" : `parts of ${picks.length} datasets`}`));
      if (message.metadata && message.metadata.code_sandbox_id) row.append(infoChip("⌨ in a code sandbox"));
    }
    thread.append(row);
  });
  if (!shown.length) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.style.textAlign = "center";
    empty.style.marginTop = "30vh";
    empty.textContent = "Ask anything. Replies come from the model this server is serving.";
    thread.append(empty);
  }
  thread.scrollTop = thread.scrollHeight;
}

// A message sent with only files has no text of its own.
function attachedNote(message) {
  const count = (message.attachment_ids || []).length + ((message.metadata && message.metadata.datasets) || []).length;
  return count ? `📎 ${count === 1 ? "an attachment" : `${count} attachments`}` : "";
}

function infoChip(text) {
  const chip = document.createElement("span");
  chip.className = "chip";
  chip.textContent = text;
  return chip;
}

function failure(message) {
  const error = message.metadata && message.metadata.error;
  return (error && explain(error.message || error.code, error.code)) || "";
}

// Nothing answering is the usual failure on a fresh server, and the
// backend's own message only knows about LM Studio.
function explain(text, code) {
  if (code === "unreachable" || /No LM Studio server answering/.test(text || "")) {
    return `${text} Or load a model on this server itself: Runner tab.`;
  }
  return text;
}

function errorNote(text) {
  const note = document.createElement("div");
  note.className = "meta error";
  note.textContent = text;
  return note;
}

function errorRow(text) {
  const row = document.createElement("div");
  row.className = "row them";
  if (text) row.append(errorNote(text));
  return row;
}

// A document the model wrote with create_document (0.72.8): fetched with
// the person's own credential, then saved, because a plain link would
// carry no Authorization header.
function documentChip(doc) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "chip doc";
  button.textContent = `⬇ ${doc.filename}`;
  button.title = `Download ${doc.filename}`;
  button.addEventListener("click", async () => {
    try {
      const headers = state.token ? { Authorization: `Bearer ${state.token}` } : {};
      const response = await fetch(doc.download_path || `/hyperlink/files/${encodeURIComponent(doc.file_id)}`, { headers });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement("a");
      link.href = url;
      link.download = doc.filename || "document.md";
      document.body.append(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 10_000);
    } catch (error) {
      button.textContent = `✗ ${doc.filename}: ${error.message}`;
    }
  });
  return button;
}

function toolChip(tool, ok) {
  const chip = document.createElement("span");
  chip.className = ok ? "chip" : "chip bad";
  chip.textContent = `${ok ? "✓" : "✗"} ${tool}`;
  return chip;
}

function bubble(who, text, { tail = true, meta = "", tools = [] } = {}) {
  const row = document.createElement("div");
  row.className = `row ${who}${tail ? " tail" : ""}`;
  for (const round of tools) row.append(toolChip(round.tool, round.ok));
  const body = document.createElement("div");
  body.className = "bubble";
  body.innerHTML = render(text);
  row.append(body);
  if (meta) {
    const label = document.createElement("div");
    label.className = "meta";
    label.textContent = meta;
    row.append(label);
  }
  return row;
}

async function ensureSession(firstMessage) {
  if (state.sessionId) return state.sessionId;
  const data = await api("/hyperlink/sessions", {
    method: "POST",
    body: { title: firstMessage.slice(0, 60), model_id: state.modelId || "" },
  });
  state.sessionId = data.session.session_id;
  state.sessions.unshift(data.session);
  drawSessions();
  return state.sessionId;
}

function setStreaming(on) {
  $("send").hidden = on;
  $("stop").hidden = !on;
  $("message").disabled = on;
}

async function send(text) {
  const attached = state.pending.splice(0);
  drawTray();
  const sessionId = await ensureSession(text || attached.map((a) => a.file.filename).join(", "));
  state.messages.push({
    role: "user", content: text, attachment_ids: attached.map((a) => a.file.file_id),
    metadata: { datasets: attached.filter((a) => a.pick), code_sandbox_id: state.codeSandboxId },
  });
  drawThread();

  const thread = $("thread");
  const reply = bubble("them", "", { tail: true });
  const body = reply.querySelector(".bubble");
  body.classList.add("typing");
  thread.append(reply);
  let collected = "";
  let failed = "";
  const controller = new AbortController();
  state.streaming = controller;
  setStreaming(true);

  const headers = { "Content-Type": "application/json", Accept: "text/event-stream" };
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  const payload = { content: text };
  if (state.modelId) payload.model_id = state.modelId;
  if (attached.length) payload.attachment_ids = attached.map((a) => a.file.file_id);
  const picks = attached.filter((a) => a.pick).map((a) => ({ file_id: a.file.file_id, ...a.pick }));
  if (picks.length) payload.datasets = picks;
  if (state.codeSandboxId) payload.code_sandbox_id = state.codeSandboxId;
  // When Send was pressed: what the ping and the signature both carry.
  const pressedAt = Date.now() / 1000;
  try {
    if (canSign()) {
      await pingBeforeSend(sessionId, pressedAt);
      payload.signature = await signature({
        action: "send", chatSessionId: sessionId, content: text, signedAt: Math.floor(pressedAt),
        signer: state.signer, serverSessionId: state.serverSessionId,
      });
    }
    const response = await fetch(
      `/hyperlink/sessions/${encodeURIComponent(sessionId)}/chat/stream`,
      { method: "POST", headers, body: JSON.stringify(payload), signal: controller.signal });
    if (!response.ok || !response.body) {
      let message = `HTTP ${response.status}`;
      try { message = (await response.json()).error.message || message; } catch { /* keep */ }
      throw new Error(message);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const block = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        for (const line of block.split("\n")) {
          if (!line.startsWith("data: ") || line === "data: [DONE]") continue;
          let frame;
          try { frame = JSON.parse(line.slice(6)); } catch { continue; }
          if (frame.type === "start") state.generationId = frame.generation_id || "";
          else if (frame.type === "delta") {
            collected += frame.text || "";
            body.innerHTML = render(collected);
            thread.scrollTop = thread.scrollHeight;
          } else if (frame.type === "tool") {
            reply.insertBefore(toolChip(frame.tool, frame.ok), body);
          } else if (frame.type === "error") {
            const error = frame.error || {};
            throw new Error(explain(error.message || "The model failed.", error.code));
          } else if (frame.type === "title") {
            $("chat-title").textContent = frame.title;
          }
        }
      }
    }
  } catch (error) {
    if (error.name !== "AbortError") {
      failed = error.message;
      reply.append(errorNote(failed));
    }
  } finally {
    body.classList.remove("typing");
    state.streaming = null;
    state.generationId = "";
    setStreaming(false);
    $("message").focus();
  }
  // The server's copy is the true one: ids, tool rounds, model names.
  // Redrawing from it used to wipe the error off the screen, so a reply
  // that failed (no model loaded, LM Studio not running) showed nothing
  // at all. A server that saved the error shows it from its copy; for
  // one that did not, or a request that never reached it, it is kept.
  await openSession(sessionId).catch(() => {});
  if (failed) {
    const last = state.messages[state.messages.length - 1];
    if (!(last && last.role === "assistant" && failure(last))) $("thread").append(errorRow(failed));
    $("thread").scrollTop = $("thread").scrollHeight;
  }
  loadSessions().catch(() => {});
}


// ---------------------------------------------------------------------------
// Attachments, and parts of datasets (0.72.8a3)
// ---------------------------------------------------------------------------
//
// Files go to /hyperlink/files first and the message carries their ids.
// A dataset (JSONL, JSON, Parquet, CSV/TSV, Arrow, text, Markdown) is
// given to the model as an excerpt: the first rows by default, or the
// part chosen here -- rows, columns, a search, a sample -- previewed
// with /hyperlink/datasets/preview exactly as the model will see it.

const DATASET_SUFFIXES = [".jsonl", ".ndjson", ".json", ".parquet", ".pq", ".csv", ".tsv", ".arrow",
  ".feather", ".txt", ".md", ".markdown", ".rst", ".log", ".jsonl.gz", ".csv.gz"];

function isDatasetName(name) {
  const lowered = (name || "").toLowerCase();
  return DATASET_SUFFIXES.some((suffix) => lowered.endsWith(suffix));
}

async function upload(file) {
  const form = new FormData();
  form.append("file", file, file.name);
  if (state.sessionId) form.append("session_id", state.sessionId);
  const headers = { Accept: "application/json" };
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  const response = await fetch("/hyperlink/files", { method: "POST", headers, body: form });
  let data = null;
  try { data = await response.json(); } catch { data = null; }
  if (!response.ok) throw new ApiError(response.status, (data && data.error && data.error.message) || `HTTP ${response.status}`);
  return data.file;
}

async function attachFiles(files) {
  for (const file of files) {
    const placeholder = { file: { file_id: "", filename: file.name }, pick: null, uploading: true };
    state.pending.push(placeholder);
    drawTray();
    try {
      placeholder.file = await upload(file);
      placeholder.uploading = false;
    } catch (error) {
      state.pending.splice(state.pending.indexOf(placeholder), 1);
      $("thread").append(errorRow(`${file.name}: ${error.message}`));
    }
    drawTray();
  }
}

function drawTray() {
  const tray = $("tray");
  if (!tray) return;
  tray.replaceChildren();
  for (const item of state.pending) {
    const chip = document.createElement("span");
    chip.className = item.pick ? "chip attached picked" : "chip attached";
    const label = document.createElement("span");
    label.textContent = `${item.uploading ? "⏳" : "📎"} ${item.file.filename}${item.pick ? " · " + describePick(item.pick) : ""}`;
    chip.append(label);
    if (!item.uploading && isDatasetName(item.file.filename)) {
      const choose = document.createElement("button");
      choose.type = "button";
      choose.className = "chip-button";
      choose.textContent = "▤";
      choose.title = "Choose which part the model gets";
      choose.addEventListener("click", () => openDatasetPanel(item));
      chip.append(choose);
    }
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "chip-button";
    remove.textContent = "✕";
    remove.title = "Remove";
    remove.addEventListener("click", () => {
      state.pending.splice(state.pending.indexOf(item), 1);
      if (dataset.item === item) $("dataset-panel").hidden = true;
      drawTray();
    });
    chip.append(remove);
    tray.append(chip);
  }
  if (state.codeSandboxId) {
    const chip = document.createElement("span");
    chip.className = "chip attached code";
    const label = document.createElement("span");
    const box = code.boxes.find((b) => b.id === state.codeSandboxId);
    label.textContent = `⌨ ${(box && box.name) || state.codeSandboxId}`;
    label.title = "The model can read, write, search, run and use git in this sandbox";
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "chip-button";
    remove.textContent = "✕";
    remove.title = "Stop working in this sandbox";
    remove.addEventListener("click", () => { state.codeSandboxId = ""; drawTray(); syncBind(); });
    chip.append(label, remove);
    tray.append(chip);
  }
  tray.hidden = !tray.children.length;
}

function describePick(pick) {
  const parts = [];
  if (pick.sample) parts.push(`${pick.sample} random rows`);
  else if (pick.rows) parts.push(`rows ${pick.rows}`);
  if (pick.columns && pick.columns.length) parts.push(pick.columns.join(", "));
  if (pick.query) parts.push(`“${pick.query}”`);
  return parts.join(" · ") || "chosen part";
}

const dataset = { item: null };

function readPick() {
  const pick = {};
  const rows = $("ds-rows").value.trim();
  const columns = $("ds-columns").value.split(",").map((c) => c.trim()).filter(Boolean);
  const query = $("ds-query").value.trim();
  const sample = parseInt($("ds-sample").value, 10);
  const most = parseInt($("ds-max").value, 10);
  if (rows) pick.rows = rows;
  if (columns.length) pick.columns = columns;
  if (query) pick.query = query;
  if (sample > 0) pick.sample = Math.min(sample, 2000);
  if ($("ds-style").value !== "auto") pick.style = $("ds-style").value;
  if (most >= 500) pick.max_chars = Math.min(most, 100000);
  return pick;
}

function openDatasetPanel(item) {
  dataset.item = item;
  const pick = item.pick || {};
  $("ds-title").textContent = item.file.filename;
  $("ds-rows").value = pick.rows || "";
  $("ds-columns").value = (pick.columns || []).join(", ");
  $("ds-query").value = pick.query || "";
  $("ds-sample").value = pick.sample || "";
  $("ds-style").value = pick.style || "auto";
  $("ds-max").value = pick.max_chars || "";
  $("ds-info").textContent = "";
  $("ds-text").hidden = true;
  $("dataset-panel").hidden = false;
  previewDataset();
}

async function previewDataset() {
  if (!dataset.item) return;
  $("ds-info").textContent = "Reading…";
  try {
    const part = await api("/hyperlink/datasets/preview", {
      method: "POST", body: { file_id: dataset.item.file.file_id, ...readPick() } });
    const known = part.rows_total !== null && part.rows_total !== undefined;
    $("ds-info").textContent = [`${part.format}: ${part.rows_shown}${known ? ` of ${part.rows_total}` : ""} rows`,
      part.columns && part.columns.length ? `columns: ${part.columns.join(", ")}` : "",
      part.truncated ? "cut to fit" : "", `${part.text.length} characters`].filter(Boolean).join(" · ");
    $("ds-text").textContent = part.text;
    $("ds-text").hidden = false;
  } catch (error) {
    $("ds-info").textContent = error.message;
    $("ds-text").hidden = true;
  }
}

async function stop() {
  if (!state.sessionId) return;
  const query = state.generationId ? `?generation_id=${encodeURIComponent(state.generationId)}` : "";
  try {
    await api(`/hyperlink/sessions/${encodeURIComponent(state.sessionId)}/chat/stop${query}`,
      { method: "POST", body: {} });
  } catch { /* fall through to aborting locally */ }
  if (state.streaming) state.streaming.abort();
}

async function loadModels() {
  try {
    const data = await api("/hyperlink/models");
    // Everything, for the runner page, which says why a model cannot
    // run (a link whose target has gone, say); the picker only offers
    // the ones that can.
    state.allModels = data.models || [];
    state.models = state.allModels.filter((m) => m.runnable !== false);
  } catch { state.models = []; state.allModels = []; }
  // Every model loaded now, on the runner and in LM Studio (0.72.8). A
  // server from before has no such endpoint, and the list stays empty.
  try {
    state.loaded = (await api("/hyperlink/models/loaded")).models || [];
  } catch { state.loaded = []; }
  const picker = $("model-picker");
  picker.replaceChildren();
  const auto = document.createElement("option");
  auto.value = "";
  const fallback = state.loaded.find((m) => m.default);
  auto.textContent = fallback ? `Default (${shortModel(fallback.model_id)})` : "Whatever is loaded";
  picker.append(auto);
  const option = (model, label) => {
    const o = document.createElement("option");
    o.value = model.model_id;
    o.textContent = label;
    return o;
  };
  let all = picker;
  if (state.loaded.length > 1) {
    // Several loaded: those first, to pick which answers.
    const group = document.createElement("optgroup");
    group.label = "Loaded now";
    for (const m of state.loaded) group.append(option(m, `● ${shortModel(m.model_id)} — ${m.label}`));
    picker.append(group);
    all = document.createElement("optgroup");
    all.label = "All models";
    picker.append(all);
  }
  const loadedIds = new Set(state.loaded.length > 1 ? state.loaded.map((m) => m.model_id) : []);
  for (const model of state.models) {
    if (loadedIds.has(model.model_id)) continue;
    all.append(option(model, `${model.loaded ? "● " : ""}${model.name || model.model_id}`));
  }
  picker.value = state.modelId;
}


// ---------------------------------------------------------------------------
// Code (0.72.8a3): /code sandboxes, their files, runs and git
// ---------------------------------------------------------------------------
//
// Everything here is the server's /code routes; nothing runs in the
// browser. Ticking "the model works in this sandbox" sends the sandbox
// with each message, and the model gets tools for exactly that sandbox,
// held to the same permissions (s1 run, s2 network, budgets) as a person.

const code = { boxes: [], boxId: "", path: "", mayRun: false };

function codeError(message) {
  $("code-error").hidden = !message;
  $("code-error").textContent = message || "";
}

const filePath = (path) => path.split("/").map(encodeURIComponent).join("/");
const boxPath = (suffix = "") => `/code/sandbox/${encodeURIComponent(code.boxId)}${suffix}`;

function showOutput(id, text) {
  $(id).textContent = text;
  $(id).hidden = !text;
}

async function loadCode() {
  codeError("");
  try {
    const index = await api("/code");
    code.mayRun = Boolean(index.you_may_run);
    const machine = index.enabled
      ? `Runs ${(index.installed_languages || []).join(", ") || "nothing installed"}` +
        `${index.network_isolation ? ", with no network unless an admin allows it" : ""}.`
      : `This server does not run code. ${index.how_to_enable || ""}`;
    $("code-machine").textContent = code.mayRun ? machine : `${machine} This credential can read sandboxes but not change them.`;
    code.boxes = (await api("/code/create/sandbox")).sandboxes || [];
  } catch (error) {
    codeError(error.status === 404 ? "This server has no /code." : error.message);
    code.boxes = [];
  }
  const picker = $("code-boxes");
  picker.replaceChildren();
  for (const box of code.boxes) {
    const option = document.createElement("option");
    option.value = box.id;
    option.textContent = box.name ? `${box.name} (${box.id.slice(0, 8)})` : box.id;
    picker.append(option);
  }
  if (!code.boxes.some((b) => b.id === code.boxId)) code.boxId = state.codeSandboxId || (code.boxes[0] && code.boxes[0].id) || "";
  if (!code.boxes.some((b) => b.id === code.boxId)) code.boxId = (code.boxes[0] && code.boxes[0].id) || "";
  picker.value = code.boxId;
  for (const id of ["code-new", "code-save", "code-run", "code-remove", "git-commit", "git-checkout",
    "git-branch", "git-clone", "code-import"]) $(id).disabled = !code.mayRun;
  syncBind();
  await Promise.all([drawBox(), loadAttachmentChoices()]);
}

function syncBind() {
  $("code-bind").checked = Boolean(code.boxId) && state.codeSandboxId === code.boxId;
  $("code-bind").disabled = !code.boxId;
}

async function drawBox() {
  const list = $("code-files");
  list.replaceChildren();
  $("code-usage").textContent = "";
  if (!code.boxId) {
    const none = document.createElement("li");
    none.className = "muted";
    none.textContent = "No sandbox yet. Make one above.";
    list.append(none);
    return;
  }
  let box;
  try { box = await api(boxPath()); } catch (error) { codeError(error.message); return; }
  const kib = (n) => `${Math.round(n / 1024)} KiB`;
  $("code-usage").textContent = `${kib(box.bytes_used)} of ${kib(box.disk_budget_bytes)}`;
  for (const file of box.files || []) {
    const item = document.createElement("li");
    item.className = file.path === code.path ? "active" : "";
    item.textContent = file.path;
    const size = document.createElement("small");
    size.textContent = ` ${file.bytes} B`;
    item.append(size);
    item.addEventListener("click", () => openFile(file.path));
    list.append(item);
  }
  if (!list.children.length) {
    const none = document.createElement("li");
    none.className = "muted";
    none.textContent = "Empty. Write a file, clone a repository, or copy in an attachment.";
    list.append(none);
  }
}

async function openFile(path) {
  codeError("");
  try {
    const file = await api(boxPath(`/files/${filePath(path)}`));
    code.path = path;
    $("code-path").value = path;
    $("code-editor").value = file.content;
    if (file.truncated) codeError("Only the start of this file is shown; saving would cut it.");
    drawBox();
  } catch (error) { codeError(error.message); }
}

async function saveFile() {
  const path = $("code-path").value.trim();
  if (!code.boxId || !path) { codeError("Name the file first."); return; }
  codeError("");
  try {
    await api(boxPath(`/files/${filePath(path)}`), { method: "PUT", body: { content: $("code-editor").value } });
    code.path = path;
    await drawBox();
  } catch (error) { codeError(error.message); }
}

async function runFile() {
  const path = $("code-path").value.trim();
  if (!code.boxId || !path) { codeError("Open or name a file to run."); return; }
  codeError("");
  showOutput("code-output", "Running…");
  try {
    await saveFile();
    const result = await api(boxPath("/run"), { method: "POST", body: { path } });
    showOutput("code-output", [
      `exit ${result.exit_code} · ${result.elapsed_seconds}s${result.timed_out ? " · timed out" : ""}`,
      result.stdout, result.stderr ? `--- stderr ---\n${result.stderr}` : ""].filter(Boolean).join("\n"));
    drawBox();
  } catch (error) { showOutput("code-output", error.message); }
}

async function removeFile() {
  const path = $("code-path").value.trim();
  if (!code.boxId || !path || !confirm(`Delete ${path}?`)) return;
  try {
    await api(boxPath(`/files/${filePath(path)}`), { method: "DELETE" });
    code.path = "";
    $("code-path").value = "";
    $("code-editor").value = "";
    drawBox();
  } catch (error) { codeError(error.message); }
}

async function git(op, args = {}) {
  if (!code.boxId) return;
  showOutput("git-output", `git ${op}…`);
  try {
    const result = await api(boxPath("/git"), { method: "POST", body: { op, ...args } });
    const text = (result.output || "").trim() || (result.error || "").trim() || (result.ok ? "done" : "failed");
    showOutput("git-output", result.ok ? text : `git ${op} failed:\n${text}`);
    if (!["status", "diff", "log", "show"].includes(op)) drawBox();
    return result;
  } catch (error) {
    showOutput("git-output", error.message);
    return null;
  }
}

async function commitAll() {
  const message = $("git-message").value.trim();
  if (!message) { showOutput("git-output", "Write a commit message first."); return; }
  const added = await git("add", { paths: ["."] });
  if (added && added.ok && (await git("commit", { message }))?.ok) $("git-message").value = "";
}

async function searchBox(event) {
  event.preventDefault();
  const list = $("code-matches");
  list.replaceChildren();
  const q = $("code-search").value.trim();
  if (!code.boxId || !q) return;
  try {
    const found = await api(boxPath(`/search?q=${encodeURIComponent(q)}`));
    for (const match of found.matches || []) {
      const item = document.createElement("li");
      item.textContent = `${match.path}:${match.line}  ${match.text}`;
      item.addEventListener("click", () => openFile(match.path));
      list.append(item);
    }
    if (!list.children.length) {
      const none = document.createElement("li");
      none.className = "muted";
      none.textContent = "No matches.";
      list.append(none);
    }
  } catch (error) { codeError(error.message); }
}

async function loadAttachmentChoices() {
  const picker = $("code-attachments");
  picker.replaceChildren();
  try {
    for (const file of (await api("/hyperlink/files?limit=100")).files || []) {
      const option = document.createElement("option");
      option.value = file.file_id;
      option.textContent = file.filename;
      picker.append(option);
    }
  } catch { /* no attachments to offer */ }
  $("code-import").disabled = !code.mayRun || !picker.options.length;
}

async function importAttachment() {
  const fileId = $("code-attachments").value;
  if (!code.boxId || !fileId) return;
  try {
    const done = await api(boxPath("/import"), { method: "POST", body: { file_id: fileId } });
    await drawBox();
    openFile(done.path);
  } catch (error) { codeError(error.message); }
}

async function newBox() {
  codeError("");
  try {
    const box = await api("/code/create/sandbox", { method: "POST", body: { name: $("code-new-name").value.trim() } });
    $("code-new-name").value = "";
    code.boxId = box.id;
    code.path = "";
    await loadCode();
  } catch (error) { codeError(error.message); }
}

async function deleteBox() {
  if (!code.boxId || !confirm("Delete this sandbox and everything in it?")) return;
  try {
    await api(boxPath(), { method: "DELETE" });
    if (state.codeSandboxId === code.boxId) state.codeSandboxId = "";
    code.boxId = "";
    drawTray();
    await loadCode();
  } catch (error) { codeError(error.message); }
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

function runnerError(message) {
  $("runner-error").hidden = !message;
  $("runner-error").textContent = message || "";
}

async function loadRunner() {
  runnerError("");
  const now = $("runner-now");
  now.replaceChildren();
  try {
    const status = await api("/runner/status");
    const title = document.createElement("h2");
    title.textContent = status.loaded ? shortModel(status.model.model_id) : "Nothing loaded";
    now.append(title);
    if (status.loaded) {
      const detail = document.createElement("p");
      detail.className = "muted";
      detail.textContent = (status.model.placement && status.model.placement.reason) || "";
      const unload = document.createElement("button");
      unload.textContent = "Unload";
      unload.addEventListener("click", async () => {
        if (!confirm("Stop serving? Anyone talking to this server stops getting answers.")) return;
        try { await api("/runner/unload", { method: "POST", body: {} }); } catch (e) { runnerError(e.message); }
        loadRunner();
      });
      now.append(detail, unload);
    }
  } catch (error) {
    now.textContent = error.status === 404 ? "This server has no runner." : error.message;
  }
  try {
    const loaded = (await api("/hyperlink/models/loaded")).models || [];
    if (loaded.length > 1) {
      const heading = document.createElement("h3");
      heading.textContent = "Loaded now";
      const list = document.createElement("ul");
      for (const m of loaded) {
        const item = document.createElement("li");
        item.textContent = `${shortModel(m.model_id)} — ${m.label}${m.default ? " (default)" : ""}`;
        list.append(item);
      }
      const hint = document.createElement("p");
      hint.className = "muted";
      hint.textContent = "Pick which one answers from the model menu in a chat.";
      now.append(heading, list, hint);
    }
  } catch { /* an older server: nothing more to show */ }
  await loadAdopt();
  const list = $("runner-models");
  list.replaceChildren();
  for (const model of (state.allModels || state.models).filter((m) => m.path)) {
    const item = document.createElement("li");
    const grow = document.createElement("div");
    grow.className = "grow";
    const name = document.createElement("strong");
    name.textContent = model.name || model.model_id;
    const detail = document.createElement("div");
    detail.className = "muted";
    detail.textContent = model.detail || model.architecture || "";
    grow.append(name, detail);
    if (model.linked_to) {
      const from = document.createElement("div");
      from.className = "muted";
      from.textContent = `linked from ${model.linked_to}`;
      grow.append(from);
    }
    item.append(grow);
    if (model.runnable !== false) {
      const load = document.createElement("button");
      load.textContent = model.loaded ? "Loaded" : "Load";
      load.disabled = model.loaded;
      load.addEventListener("click", async () => {
        load.disabled = true;
        load.textContent = "Loading…";
        try { await api("/runner/load", { method: "POST", body: { model_id: model.model_id } }); }
        catch (e) { runnerError(e.message); }
        await loadModels();
        loadRunner();
      });
      item.append(load);
    }
    if (model.link_name) {
      const unlink = document.createElement("button");
      unlink.textContent = "Remove link";
      unlink.title = "Removes the link from the models folder. The model itself is not touched.";
      unlink.addEventListener("click", async () => {
        if (!confirm(`Remove ${model.link_name} from the list? The file it points at stays.`)) return;
        try {
          await api(`/hyperlink/models/link/${encodeURIComponent(model.link_name)}`, { method: "DELETE" });
        } catch (e) { runnerError(e.message); }
        await loadModels();
        loadRunner();
      });
      item.append(unlink);
    }
    list.append(item);
  }
  if (!list.children.length) {
    const none = document.createElement("li");
    none.className = "muted";
    none.textContent = "No model with a file on this machine yet. Put a .gguf under " +
      "~/.hypernix/models, link one below, or run `hypernix-t1 runner start` for the default model.";
    list.append(none);
  }
}

async function linkModel(event) {
  event.preventDefault();
  runnerError("");
  const path = $("link-path").value.trim();
  if (!path) return;
  const button = $("link-submit");
  button.disabled = true;
  try {
    const made = await api("/hyperlink/models/link", {
      method: "POST", body: { path, name: $("link-name").value.trim() },
    });
    $("link-path").value = "";
    $("link-name").value = "";
    $("link-note").textContent = `Linked ${made.name} → ${made.linked_to}`;
  } catch (e) {
    runnerError(e.status === 403 ? "Linking a model needs an admin key." : e.message);
  }
  button.disabled = false;
  await loadModels();
  loadRunner();
}

async function loadAdopt() {
  const card = $("runner-adopt");
  card.replaceChildren();
  let preview;
  try { preview = await api("/runner/adopt"); } catch { preview = null; }
  if (!preview || !preview.available) { card.hidden = true; return; }
  card.hidden = false;
  const title = document.createElement("h2");
  title.textContent = "In LM Studio";
  const note = document.createElement("p");
  note.className = "muted";
  note.textContent = "Unloads it from LM Studio and loads the same file on the HyperNix runner. " +
    "If the runner cannot load it, it goes back into LM Studio. Admins and tailnet devices only.";
  card.append(title, note);
  for (const modelId of preview.lmstudio_loaded) {
    const button = document.createElement("button");
    button.textContent = `Move ${shortModel(modelId)} to the HyperNix runner`;
    button.addEventListener("click", async () => {
      if (!confirm("Replies pause while it reloads. Move it?")) return;
      button.disabled = true;
      button.textContent = "Moving…";
      try { await api("/runner/adopt", { method: "POST", body: { model_id: modelId } }); }
      catch (e) { runnerError(e.message); }
      await loadModels();
      loadRunner();
    });
    card.append(button);
  }
}

// ---------------------------------------------------------------------------
// Memory, kept current with /memory/sync as the app does
// ---------------------------------------------------------------------------

const memory = { cursor: 0, items: new Map() };

async function syncMemory() {
  for (let page = 0; page < 50; page += 1) {
    const data = await api(`/memory/sync?cursor=${memory.cursor}`);
    if (data.full) memory.items.clear();
    for (const item of data.memories || []) memory.items.set(item.memory_id, item);
    for (const id of data.deleted || []) memory.items.delete(id);
    memory.cursor = data.cursor;
    if (!data.more) break;
  }
  drawMemory();
}

function drawMemory() {
  const list = $("memories");
  list.replaceChildren();
  const items = [...memory.items.values()].sort(
    (a, b) => (b.pinned - a.pinned) || (b.updated_at - a.updated_at));
  for (const item of items) {
    const row = document.createElement("li");
    const grow = document.createElement("div");
    grow.className = "grow";
    grow.textContent = item.content;
    const detail = document.createElement("div");
    detail.className = "muted";
    detail.textContent = [item.category, item.source === "auto" ? "noticed by the model" : ""]
      .filter(Boolean).join(" · ");
    grow.append(detail);
    const pin = document.createElement("button");
    pin.className = "secondary";
    pin.textContent = item.pinned ? "Unpin" : "Pin";
    pin.addEventListener("click", async () => {
      await api("/memory/edit", { method: "POST", body: { memory_id: item.memory_id, pinned: !item.pinned } });
      syncMemory();
    });
    const forget = document.createElement("button");
    forget.textContent = "Forget";
    forget.addEventListener("click", async () => {
      await api(`/memory/delete?memory_id=${encodeURIComponent(item.memory_id)}`, { method: "POST", body: {} });
      syncMemory();
    });
    row.append(grow, pin, forget);
    list.append(row);
  }
  $("memory-note").textContent = items.length ? "" : "Nothing remembered yet.";
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

async function loadSettings() {
  const data = await api("/hyperlink/preferences");
  const form = $("settings");
  const prefs = data.preferences || {};
  for (const element of form.elements) {
    if (!element.name) continue;
    if (element.type === "checkbox") element.checked = Boolean(prefs[element.name]);
    else element.value = prefs[element.name] || "";
  }
  $("sign-in-state").textContent = state.token
    ? "Signed in with a pairing or key stored in this browser."
    : "Signed in without a key: this server trusts this network.";
}

async function saveSettings(event) {
  event.preventDefault();
  const body = {};
  for (const element of $("settings").elements) {
    if (!element.name) continue;
    body[element.name] = element.type === "checkbox" ? element.checked : element.value;
  }
  const data = await api("/hyperlink/preferences", { method: "PATCH", body });
  $("settings-note").textContent = (data.notes || []).join(" ") || "Saved.";
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

function switchView(view) {
  state.view = view;
  for (const tab of document.querySelectorAll(".tab")) {
    tab.classList.toggle("active", tab.dataset.view === view);
  }
  for (const name of ["chats", "code", "runner", "memory", "settings"]) {
    $(`view-${name}`).hidden = name !== view;
  }
  if (view === "runner") loadRunner();
  if (view === "code") loadCode();
  if (view === "memory") syncMemory().catch((e) => { $("memory-note").textContent = e.message; });
  if (view === "settings") loadSettings().catch((e) => { $("settings-note").textContent = e.message; });
}

async function start() {
  const server = await fetch("/server/info").then((r) => (r.ok ? r.json() : {})).catch(() => ({}));
  if (server && server.name) $("server-name").textContent = server.name;
  $("who").textContent = `${location.host}`;
  await loadModels();
  await loadSessions();
  if (state.sessions.length) await openSession(state.sessions[0].session_id);
  else newChat();
}

function wire() {
  $("sign-in-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    $("sign-in-error").hidden = true;
    try {
      await signIn($("credential").value);
      showSignIn(false);
      await start();
    } catch (error) {
      $("sign-in-error").textContent = error.message;
      $("sign-in-error").hidden = false;
    }
  });
  $("composer").addEventListener("submit", (event) => {
    event.preventDefault();
    const text = $("message").value.trim();
    if ((!text && !state.pending.length) || state.streaming) return;
    if (state.pending.some((a) => a.uploading)) return;
    $("message").value = "";
    send(text);
  });
  $("message").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("composer").requestSubmit();
    }
  });
  $("message").addEventListener("input", (event) => {
    event.target.style.height = "auto";
    event.target.style.height = `${Math.min(event.target.scrollHeight, 200)}px`;
  });
  $("stop").addEventListener("click", stop);
  $("new-chat").addEventListener("click", () => { newChat(); switchView("chats"); });
  $("search").addEventListener("input", drawSessions);
  $("model-picker").addEventListener("change", async (event) => {
    state.modelId = event.target.value;
    if (state.sessionId) {
      await api(`/hyperlink/sessions/${encodeURIComponent(state.sessionId)}`,
        { method: "PATCH", body: { model_id: state.modelId } }).catch(() => {});
    }
  });
  $("rename-chat").addEventListener("click", async () => {
    if (!state.sessionId) return;
    const title = prompt("Name this chat", $("chat-title").textContent);
    if (!title) return;
    await api(`/hyperlink/sessions/${encodeURIComponent(state.sessionId)}`,
      { method: "PATCH", body: { title } });
    $("chat-title").textContent = title;
    loadSessions();
  });
  $("delete-chat").addEventListener("click", async () => {
    if (!state.sessionId || !confirm("Delete this chat?")) return;
    await api(`/hyperlink/sessions/${encodeURIComponent(state.sessionId)}`, { method: "DELETE" });
    newChat();
    loadSessions();
  });
  $("show-sidebar").addEventListener("click", () => $("sidebar").classList.toggle("open"));
  for (const tab of document.querySelectorAll(".tab")) {
    tab.addEventListener("click", () => {
      switchView(tab.dataset.view);
      $("sidebar").classList.remove("open");
    });
  }
  $("refresh-runner").addEventListener("click", loadRunner);
  $("attach").addEventListener("click", () => $("attach-input").click());
  $("attach-input").addEventListener("change", (event) => {
    attachFiles([...event.target.files]);
    event.target.value = "";
  });
  $("thread").addEventListener("dragover", (event) => event.preventDefault());
  $("thread").addEventListener("drop", (event) => {
    event.preventDefault();
    if (event.dataTransfer && event.dataTransfer.files.length) attachFiles([...event.dataTransfer.files]);
  });
  $("ds-close").addEventListener("click", () => { $("dataset-panel").hidden = true; });
  $("ds-preview").addEventListener("click", previewDataset);
  $("ds-use").addEventListener("click", () => {
    const pick = readPick();
    if (dataset.item) dataset.item.pick = Object.keys(pick).length ? pick : null;
    $("dataset-panel").hidden = true;
    drawTray();
  });
  $("ds-clear").addEventListener("click", () => {
    for (const id of ["ds-rows", "ds-columns", "ds-query", "ds-sample", "ds-max"]) $(id).value = "";
    $("ds-style").value = "auto";
    previewDataset();
  });
  $("refresh-code").addEventListener("click", loadCode);
  $("code-boxes").addEventListener("change", (event) => {
    code.boxId = event.target.value;
    code.path = "";
    syncBind();
    drawBox();
  });
  $("code-new").addEventListener("click", newBox);
  $("code-delete-box").addEventListener("click", deleteBox);
  $("code-bind").addEventListener("change", (event) => {
    state.codeSandboxId = event.target.checked ? code.boxId : "";
    drawTray();
  });
  $("code-chat").addEventListener("click", () => {
    if (!code.boxId) return;
    state.codeSandboxId = code.boxId;
    newChat({ keepSandbox: true });
    switchView("chats");
  });
  $("code-save").addEventListener("click", saveFile);
  $("code-run").addEventListener("click", runFile);
  $("code-remove").addEventListener("click", removeFile);
  $("code-search-form").addEventListener("submit", searchBox);
  $("code-import").addEventListener("click", importAttachment);
  for (const button of document.querySelectorAll("[data-git]")) {
    button.addEventListener("click", () => git(button.dataset.git));
  }
  $("git-commit").addEventListener("click", commitAll);
  $("git-checkout").addEventListener("click", () => git("checkout", { ref: $("git-ref").value.trim() }));
  $("git-branch").addEventListener("click", () => git("checkout", { ref: $("git-ref").value.trim(), create: true }));
  $("git-clone").addEventListener("click", () => git("clone", { url: $("git-url").value.trim() }));
  $("code-editor").addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "s") { event.preventDefault(); saveFile(); }
    if (event.key === "Tab" && !event.shiftKey) {
      event.preventDefault();
      event.target.setRangeText("    ", event.target.selectionStart, event.target.selectionEnd, "end");
    }
  });
  $("link-form").addEventListener("submit", linkModel);
  $("remember").addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = $("remember-text").value.trim();
    if (!text) return;
    await api("/memory/create", { method: "POST", body: { content: text } });
    $("remember-text").value = "";
    syncMemory();
  });
  $("settings").addEventListener("submit", (event) => {
    saveSettings(event).catch((e) => { $("settings-note").textContent = e.message; });
  });
  $("sign-out").addEventListener("click", () => {
    storeToken("");
    state.token = "";
    location.reload();
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  wire();
  state.token = storedToken();
  let ok = false;
  try { ok = await signedIn(); } catch { ok = false; }
  if (!ok && state.token) {
    state.token = "";
    storeToken("");
  }
  showSignIn(!ok);
  if (ok) await start();
});
