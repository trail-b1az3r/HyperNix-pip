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
  sessionId: "",
  messages: [],
  models: [],
  modelId: "",
  generationId: "",
  streaming: null,
  view: "chats",
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
// Rendering text safely
// ---------------------------------------------------------------------------

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

function inline(text) {
  return text
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    // A URL ends at whitespace or an escaped quote or bracket (the text
    // is already escaped), and trailing punctuation belongs to the prose.
    .replace(/\bhttps?:\/\/(?:(?!&quot;|&#39;|&lt;|&gt;)[^\s<])+/g, (match) => {
      const url = match.replace(/[.,;:!?)]+$/, "");
      const rest = match.slice(url.length);
      return `<a href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>${rest}`;
    });
}

/** A small, safe Markdown: escaped text, code blocks, inline code, bold, links. */
function render(text) {
  const parts = escapeHtml(text || "").split(/```[a-zA-Z0-9_-]*\n?/);
  return parts.map((part, index) => {
    if (index % 2 === 1) return `<pre><code>${part.replace(/\n$/, "")}</code></pre>`;
    return part.split(/\n{2,}/).filter((p) => p.trim()).map(
      (p) => `<p>${inline(p).replace(/\n/g, "<br>")}</p>`).join("");
  }).join("");
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
  drawThread();
  drawSessions();
  $("sidebar").classList.remove("open");
  switchView("chats");
}

function newChat() {
  state.sessionId = "";
  state.messages = [];
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
      ? bubble(message.role === "user" ? "me" : "them", message.content, {
        tail,
        meta: message.role === "assistant" && tail ? shortModel(message.model_id) : "",
        tools: (message.metadata && message.metadata.tool_rounds) || [],
      })
      : errorRow("");
    if (message.role === "assistant" && failure(message)) row.append(errorNote(failure(message)));
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
  const sessionId = await ensureSession(text);
  state.messages.push({ role: "user", content: text });
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
  try {
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
  const picker = $("model-picker");
  picker.replaceChildren();
  const auto = document.createElement("option");
  auto.value = "";
  auto.textContent = "Whatever is loaded";
  picker.append(auto);
  for (const model of state.models) {
    const option = document.createElement("option");
    option.value = model.model_id;
    option.textContent = `${model.loaded ? "● " : ""}${model.name || model.model_id}`;
    picker.append(option);
  }
  picker.value = state.modelId;
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
  for (const name of ["chats", "runner", "memory", "settings"]) {
    $(`view-${name}`).hidden = name !== view;
  }
  if (view === "runner") loadRunner();
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
    if (!text || state.streaming) return;
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
